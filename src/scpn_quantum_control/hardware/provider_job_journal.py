# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — durable provider job custody
"""Persist immutable native requests before effects and retain observation history."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from uuid import UUID

JOURNAL_SCHEMA = "provider_job_journal.v1"
"""Version of the durable companion; original provider payloads are unchanged."""
TERMINAL_STATES = frozenset({"completed", "cancelled", "failed"})


class SubmissionUnknownError(RuntimeError):
    """An earlier dispatch may have taken effect and cannot be retried blindly."""


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ValueError(f"{name} must be nonempty bounded text")


class ProviderJobJournal:
    """Own a SQLite write-ahead journal for immutable single native batch attempts.

    Parameters
    ----------
    root
        Caller-owned local directory. No provider credentials are stored here.
        Each committed request has a canonical UUID and exact native payload.

    """

    def __init__(self, root: Path) -> None:
        """Open an owned journal, refusing an unknown database version."""
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "provider_jobs.sqlite3"
        with self._connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("unsupported provider job journal version")
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY, target TEXT NOT NULL,
                    payload BLOB NOT NULL, payload_sha256 TEXT NOT NULL,
                    shots INTEGER NOT NULL, circuits INTEGER NOT NULL,
                    experiment TEXT NOT NULL, state TEXT NOT NULL,
                    provider_job_id TEXT UNIQUE, billing TEXT NOT NULL DEFAULT 'unknown'
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
                    state TEXT NOT NULL, observation TEXT NOT NULL
                );
                PRAGMA user_version=1;
            """)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def prepare(
        self,
        attempt_id: str,
        *,
        target: str,
        payload: bytes,
        shots: int,
        experiment: str,
        circuits: int,
    ) -> None:
        """Persist the exact request or confirm an identical existing preparation.

        Parameters
        ----------
        attempt_id
            Canonical UUID identifying one attempt, never a retry counter.
        target
            Exact selected backend name.
        payload
            Original native encoded batch, at most sixteen mebibytes.
        shots
            Positive integral requested shots per circuit.
        experiment
            Retained caller label.
        circuits
            Number of circuits, between one and256.

        Raises
        ------
        ValueError
            If identity, dimensions or immutable contents disagree.

        """
        if str(UUID(attempt_id)) != attempt_id:
            raise ValueError("attempt ID must be a canonical UUID")
        _text(target, "target")
        _text(experiment, "experiment")
        if not isinstance(payload, bytes) or not 0 < len(payload) <= 16 * 1024 * 1024:
            raise ValueError("native payload must contain at most sixteen mebibytes")
        if type(shots) is not int or shots < 1 or shots > 2**63 - 1:
            raise ValueError("shots must be a positive SQLite integer")
        if type(circuits) is not int or not 1 <= circuits <= 256:
            raise ValueError("circuits must be between one and256")
        identity = (
            target,
            payload,
            hashlib.sha256(payload).hexdigest(),
            shots,
            circuits,
            experiment,
        )
        with self._transaction() as db:
            row = db.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is not None:
                if (
                    tuple(
                        row[k]
                        for k in (
                            "target",
                            "payload",
                            "payload_sha256",
                            "shots",
                            "circuits",
                            "experiment",
                        )
                    )
                    != identity
                ):
                    raise ValueError("attempt identity differs from the original request")
                return
            db.execute(
                "INSERT INTO attempts(attempt_id,target,payload,payload_sha256,shots,circuits,experiment,state) VALUES (?,?,?,?,?,?,?,?)",
                (attempt_id, *identity, "prepared"),
            )
            self._event(db, attempt_id, "prepared", {})

    def snapshot(self, attempt_id: str) -> dict[str, object]:
        """Read a detached request and its ordered history without changing state.

        Parameters
        ----------
        attempt_id
            Existing attempt identity.

        Returns
        -------
        dict[str, object]
            Immutable request values, current state and detached event records.

        Raises
        ------
        KeyError
            If the attempt is unknown.
        ValueError
            If stored native payload integrity is invalid.

        """
        with self._connection() as db:
            db.execute("BEGIN")
            row = self._row(db, attempt_id)
            result: dict[str, object] = dict(row)
            result["schema"] = JOURNAL_SCHEMA
            result["events"] = [
                {
                    "sequence": r["sequence"],
                    "state": r["state"],
                    "observation": json.loads(r["observation"]),
                }
                for r in db.execute(
                    "SELECT * FROM events WHERE attempt_id=? ORDER BY sequence", (attempt_id,)
                )
            ]
            return result

    def begin_submit(self, attempt_id: str) -> None:
        """Commit uncertainty before dispatch; exactly one caller may cross it.

        Parameters
        ----------
        attempt_id
            Existing prepared attempt.

        Raises
        ------
        SubmissionUnknownError
            If any previous caller has already reserved the dispatch.

        """
        with self._transaction() as db:
            row = self._row(db, attempt_id)
            if row["state"] != "prepared":
                raise SubmissionUnknownError(
                    "existing attempt requires observation or reconciliation"
                )
            db.execute(
                "UPDATE attempts SET state='submission_unknown' WHERE attempt_id=?", (attempt_id,)
            )
            self._event(db, attempt_id, "submission_unknown", {})

    def bind_job(self, attempt_id: str, job_id: str) -> None:
        """Record a returned or exactly reconciled original provider handle.

        Parameters
        ----------
        attempt_id
            Attempt at its uncertain dispatch boundary.
        job_id
            Actual provider handle; uniqueness prevents cross-attempt adoption.

        Raises
        ------
        ValueError
            If another handle was already bound or the state is incompatible.

        """
        _text(job_id, "provider job ID")
        with self._transaction() as db:
            row = self._row(db, attempt_id)
            if row["provider_job_id"] == job_id:
                return
            if row["state"] != "submission_unknown" or row["provider_job_id"] is not None:
                raise ValueError("attempt cannot adopt a different provider handle")
            db.execute(
                "UPDATE attempts SET provider_job_id=?,state='submitted' WHERE attempt_id=?",
                (job_id, attempt_id),
            )
            self._event(db, attempt_id, "submitted", {"provider_job_id": job_id})

    def record(self, attempt_id: str, state: str, observation: Mapping[str, object]) -> None:
        """Retain observations and allow complete samples to resolve a terminal race.

        Parameters
        ----------
        attempt_id
            Existing bound attempt.
        state
            Submitted, cancellation requested, or observed terminal state.
        observation
            JSON-compatible detached native result/status evidence. Billing
            remains unknown; lifecycle annotations do not prove a debit.
            Late raw/partial evidence remains in history. Validated completed
            samples can resolve an earlier cancellation or failure annotation;
            an existing completed result is never replaced.

        Raises
        ------
        ValueError
            If an observation is malformed or has no original provider handle.

        """
        if state not in {"submitted", "cancellation_requested", *TERMINAL_STATES}:
            raise ValueError("unsupported observation state")
        encoded = json.dumps(dict(observation), allow_nan=False, sort_keys=True)
        with self._transaction() as db:
            row = self._row(db, attempt_id)
            if row["provider_job_id"] is None:
                raise ValueError("observation requires a bound provider handle")
            effective = state
            if row["state"] in TERMINAL_STATES:
                if state == "completed" and row["state"] != "completed":
                    effective = "completed"
                elif "native_runtime_json" in observation or "partial_results" in observation:
                    effective = row["state"]
                else:
                    return
            if row["state"] == "cancellation_requested" and state == "submitted":
                effective = "cancellation_requested"
            db.execute("UPDATE attempts SET state=? WHERE attempt_id=?", (effective, attempt_id))
            db.execute(
                "INSERT INTO events(attempt_id,state,observation) VALUES (?,?,?)",
                (attempt_id, effective, encoded),
            )

    @staticmethod
    def _event(
        db: sqlite3.Connection, attempt_id: str, state: str, observation: Mapping[str, object]
    ) -> None:
        db.execute(
            "INSERT INTO events(attempt_id,state,observation) VALUES (?,?,?)",
            (attempt_id, state, json.dumps(dict(observation))),
        )

    @staticmethod
    def _row(db: sqlite3.Connection, attempt_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        if row is None:
            raise KeyError("unknown provider attempt")
        if hashlib.sha256(bytes(row["payload"])).hexdigest() != row["payload_sha256"]:
            raise ValueError("stored native payload digest differs")
        return cast(sqlite3.Row, row)
