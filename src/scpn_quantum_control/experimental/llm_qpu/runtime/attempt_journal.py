# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — durable experimental QPU attempt journal
"""Write-ahead job states; no provider connection or submission lives here."""

from __future__ import annotations

import os
import re
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA = "scpn.experimental.llm_qpu.attempt_journal.v1"


class JournalStateError(RuntimeError):
    """The durable state forbids the requested attempt transition."""


@dataclass(frozen=True, slots=True)
class JobPlan:
    """One ordered, immutable provider request identity.

    Parameters
    ----------
    name
        Distinct phase name within one attempt.
    payload_sha256
        Digest of the exact serialized provider request.
    circuits
        Number of circuits in the request.
    shots_per_circuit
        Uniform shots value required by IQM's request envelope.

    """

    name: str
    payload_sha256: str
    circuits: int
    shots_per_circuit: int

    def __post_init__(self) -> None:
        """Refuse malformed phase names, digests and request dimensions."""
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", self.name):
            raise ValueError("invalid provider job phase name")
        if (
            not isinstance(self.payload_sha256, str)
            or _DIGEST.fullmatch(self.payload_sha256) is None
        ):
            raise ValueError("provider request digest must be lowercase SHA-256")
        if type(self.circuits) is not int or self.circuits < 1:
            raise ValueError("provider job circuit count must be positive")
        if type(self.shots_per_circuit) is not int or self.shots_per_circuit < 1:
            raise ValueError("provider job shots must be positive")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _uuid(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("attempt or job ID must be UUID text")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ValueError("attempt or job ID must be UUID text") from exc
    if str(parsed) != value:
        raise ValueError("attempt or job ID must be canonical UUID text")
    return value


def _digest(value: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError("digest must be lowercase SHA-256")
    return value


def _private_directory(path: Path) -> None:
    for component in (path, *path.parents):
        try:
            if stat.S_ISLNK(component.lstat().st_mode):
                raise ValueError("journal path contains a symlink")
        except FileNotFoundError:
            continue
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("journal directory must be owned and private (0700)")


class AttemptJournal:
    """SQLite write-ahead record for a fixed ordered QPU attempt.

    The caller must commit ``begin_submit`` before crossing the network
    boundary. Reopening a ``submitting`` job never triggers a retry.
    """

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise TypeError("journal root must be a Path")
        _private_directory(root)
        self.path = root / "attempts.sqlite3"
        if self.path.is_symlink():
            raise ValueError("journal database must not be a symlink")
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
            ):
                raise ValueError("journal database must be owned and private (0600)")
        finally:
            os.close(descriptor)
        with self._connect() as database:
            database.execute("PRAGMA journal_mode=WAL")
            database.executescript("""
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    bundle_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    schema TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
                    ordinal INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    circuits INTEGER NOT NULL CHECK(circuits > 0),
                    shots_per_circuit INTEGER NOT NULL CHECK(shots_per_circuit > 0),
                    state TEXT NOT NULL CHECK(state IN ('prepared','submitting','recovery_required','submitted','verified')),
                    provider_job_id TEXT UNIQUE,
                    result_sha256 TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (attempt_id, ordinal),
                    UNIQUE (attempt_id, name)
                );
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        database = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA foreign_keys=ON")
        database.execute("PRAGMA synchronous=FULL")
        try:
            yield database
        finally:
            database.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            try:
                yield database
                database.execute("COMMIT")
            except BaseException:
                database.execute("ROLLBACK")
                raise

    def prepare(self, attempt_id: str, bundle_sha256: str, jobs: tuple[JobPlan, ...]) -> None:
        """Commit an entire ordered request inventory before any network call."""
        _uuid(attempt_id)
        _digest(bundle_sha256)
        if (
            not isinstance(jobs, tuple)
            or not jobs
            or any(not isinstance(job, JobPlan) for job in jobs)
        ):
            raise ValueError("attempt needs a nonempty immutable job tuple")
        if len({job.name for job in jobs}) != len(jobs):
            raise ValueError("provider job phase names must be distinct")
        now = _utc_now()
        with self._transaction() as database:
            if database.execute(
                "SELECT 1 FROM attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone():
                raise JournalStateError("attempt ID already exists; inspect durable state")
            database.execute(
                "INSERT INTO attempts VALUES (?,?,?,?)", (attempt_id, bundle_sha256, now, _SCHEMA)
            )
            database.executemany(
                "INSERT INTO jobs(attempt_id,ordinal,name,payload_sha256,circuits,shots_per_circuit,state,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    (
                        attempt_id,
                        ordinal,
                        job.name,
                        job.payload_sha256,
                        job.circuits,
                        job.shots_per_circuit,
                        "prepared",
                        now,
                    )
                    for ordinal, job in enumerate(jobs)
                ),
            )

    def snapshot(self, attempt_id: str) -> dict[str, object]:
        """Read persisted jobs without submitting or advancing state."""
        _uuid(attempt_id)
        with self._connect() as database:
            attempt = database.execute(
                "SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if attempt is None:
                raise JournalStateError("unknown attempt ID")
            rows = database.execute(
                "SELECT * FROM jobs WHERE attempt_id=? ORDER BY ordinal", (attempt_id,)
            ).fetchall()
            return {
                "schema": attempt["schema"],
                "attempt_id": attempt_id,
                "bundle_sha256": attempt["bundle_sha256"],
                "created_at": attempt["created_at"],
                "jobs": [dict(row) for row in rows],
            }

    def _job(self, database: sqlite3.Connection, attempt_id: str, ordinal: int) -> sqlite3.Row:
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("job ordinal must be a nonnegative integer")
        row = database.execute(
            "SELECT * FROM jobs WHERE attempt_id=? AND ordinal=?", (attempt_id, ordinal)
        ).fetchone()
        if not isinstance(row, sqlite3.Row):
            raise JournalStateError("unknown attempt job")
        return row

    def begin_submit(self, attempt_id: str, ordinal: int, payload_sha256: str) -> None:
        """Durably mark the network boundary; refuse replay or unverified predecessors."""
        _uuid(attempt_id)
        _digest(payload_sha256)
        with self._transaction() as database:
            row = self._job(database, attempt_id, ordinal)
            if row["payload_sha256"] != payload_sha256:
                raise JournalStateError("request payload differs from prepared digest")
            if row["state"] != "prepared":
                raise JournalStateError("job is not prepared; never retry an ambiguous submission")
            prior = database.execute(
                "SELECT COUNT(*) FROM jobs WHERE attempt_id=? AND ordinal<? AND state!='verified'",
                (attempt_id, ordinal),
            ).fetchone()[0]
            if prior:
                raise JournalStateError("earlier job has not been result-verified")
            database.execute(
                "UPDATE jobs SET state='submitting',updated_at=? WHERE attempt_id=? AND ordinal=?",
                (_utc_now(), attempt_id, ordinal),
            )

    def mark_recovery_required(self, attempt_id: str, ordinal: int) -> None:
        """Retain the reservation after an uncertain provider boundary."""
        _uuid(attempt_id)
        with self._transaction() as database:
            row = self._job(database, attempt_id, ordinal)
            if row["state"] not in {"submitting", "recovery_required"}:
                raise JournalStateError("only an uncertain submission can require recovery")
            database.execute(
                "UPDATE jobs SET state='recovery_required',updated_at=? WHERE attempt_id=? AND ordinal=?",
                (_utc_now(), attempt_id, ordinal),
            )

    def record_submitted(self, attempt_id: str, ordinal: int, provider_job_id: str) -> None:
        """Bind the real provider ID after a successful submit response."""
        _uuid(attempt_id)
        _uuid(provider_job_id)
        with self._transaction() as database:
            row = self._job(database, attempt_id, ordinal)
            if row["state"] != "submitting":
                raise JournalStateError("provider job ID needs an active submitting boundary")
            database.execute(
                "UPDATE jobs SET state='submitted',provider_job_id=?,updated_at=? WHERE attempt_id=? AND ordinal=?",
                (provider_job_id, _utc_now(), attempt_id, ordinal),
            )

    def record_recovered(
        self,
        attempt_id: str,
        ordinal: int,
        provider_job_id: str,
        observed_payload_sha256: str,
    ) -> None:
        """Bind an ambiguous call only after read-only provider payload proof."""
        _uuid(attempt_id)
        _uuid(provider_job_id)
        _digest(observed_payload_sha256)
        with self._transaction() as database:
            row = self._job(database, attempt_id, ordinal)
            if row["state"] not in {"submitting", "recovery_required"}:
                raise JournalStateError("recovery needs an uncertain submit boundary")
            if row["payload_sha256"] != observed_payload_sha256:
                raise JournalStateError("provider job payload differs from prepared request")
            database.execute(
                "UPDATE jobs SET state='submitted',provider_job_id=?,updated_at=? WHERE attempt_id=? AND ordinal=?",
                (provider_job_id, _utc_now(), attempt_id, ordinal),
            )

    def record_verified(self, attempt_id: str, ordinal: int, result_sha256: str) -> None:
        """Mark an already submitted job verified by its separate raw-result validator."""
        _uuid(attempt_id)
        _digest(result_sha256)
        with self._transaction() as database:
            row = self._job(database, attempt_id, ordinal)
            if row["state"] != "submitted":
                raise JournalStateError("raw result cannot verify an unsubmitted job")
            database.execute(
                "UPDATE jobs SET state='verified',result_sha256=?,updated_at=? WHERE attempt_id=? AND ordinal=?",
                (result_sha256, _utc_now(), attempt_id, ordinal),
            )
