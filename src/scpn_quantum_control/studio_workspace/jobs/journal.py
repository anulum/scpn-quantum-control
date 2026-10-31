# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — transactional attempt and event journal of the local job service
"""Record local job attempts and their ordered events in one SQLite file.

The journal is the only source of truth about an attempt. Its guarantees are
enforced by SQLite itself, not by the callers:

* one attempt per idempotency key (a ``UNIQUE`` column);
* events numbered ``1, 2, 3, ...`` per attempt, appended in the same
  transaction that changes the attempt's state;
* terminal attempts and every event are immutable (triggers abort an update
  or a delete).

Every write runs in an ``IMMEDIATE`` transaction, so two service threads never
interleave a state change with its event.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

AttemptState = Literal[
    "queued", "running", "succeeded", "failed", "cancelled", "expired", "interrupted"
]
TERMINAL_STATES: Final[frozenset[str]] = frozenset(
    {"succeeded", "failed", "cancelled", "expired", "interrupted"}
)
"""States after which an attempt never changes again."""

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    request_digest TEXT NOT NULL,
    request_json TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'queued', 'running', 'succeeded', 'failed', 'cancelled', 'expired', 'interrupted'
    )),
    owner TEXT,
    pid INTEGER,
    pid_start TEXT,
    deadline REAL,
    result_digest TEXT,
    reason TEXT,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
    seq INTEGER NOT NULL CHECK (seq >= 1),
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    at REAL NOT NULL,
    PRIMARY KEY (attempt_id, seq)
);
CREATE TRIGGER IF NOT EXISTS terminal_attempts_are_immutable
BEFORE UPDATE ON attempts
WHEN OLD.state IN ('succeeded', 'failed', 'cancelled', 'expired', 'interrupted')
BEGIN SELECT RAISE(ABORT, 'a terminal attempt is immutable'); END;
CREATE TRIGGER IF NOT EXISTS attempts_are_never_deleted
BEFORE DELETE ON attempts
BEGIN SELECT RAISE(ABORT, 'attempts are never deleted'); END;
CREATE TRIGGER IF NOT EXISTS events_are_immutable
BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
CREATE TRIGGER IF NOT EXISTS events_are_never_deleted
BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events are never deleted'); END;
"""


@dataclass(frozen=True)
class Attempt:
    """One journalled attempt.

    Parameters
    ----------
    attempt_id
        Stable identity derived from the request and its idempotency key.
    idempotency_key
        Caller key; at most one attempt exists per key.
    request_digest
        Digest of the admitted request, compared on a repeated key.
    request
        The admitted request.
    plan
        The handler's plan, resolved before admission.
    state
        Current state.
    owner
        Identity of the service instance running the attempt, if any.
    pid
        Process id of the attempt's worker, once started.
    pid_start
        Kernel start time of that process, which tells a reused pid apart.
    deadline
        Wall-clock time after which a running attempt expires.
    result_digest
        Digest of the sealed record, once succeeded.
    reason
        Why the attempt ended, for a non-succeeded terminal state.
    created
        Admission time.

    """

    attempt_id: str
    idempotency_key: str
    request_digest: str
    request: Mapping[str, Any]
    plan: Mapping[str, Any]
    state: AttemptState
    owner: str | None
    pid: int | None
    pid_start: str | None
    deadline: float | None
    result_digest: str | None
    reason: str | None
    created: float

    @property
    def terminal(self) -> bool:
        """Whether the attempt has ended and can no longer change."""
        return self.state in TERMINAL_STATES


@dataclass(frozen=True)
class Event:
    """One ordered attempt event.

    Parameters
    ----------
    attempt_id
        Attempt the event belongs to.
    seq
        Position in the attempt's history, starting at one.
    kind
        Event kind, such as ``admitted``, ``started`` or a terminal state.
    payload
        JSON-ready details.
    at
        Wall-clock time of the event.

    """

    attempt_id: str
    seq: int
    kind: str
    payload: Mapping[str, Any]
    at: float

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready event."""
        return {
            "attempt_id": self.attempt_id,
            "seq": self.seq,
            "kind": self.kind,
            "payload": dict(self.payload),
            "at": self.at,
        }


def _attempt(row: sqlite3.Row) -> Attempt:
    """Build an attempt from its row."""
    return Attempt(
        attempt_id=row["attempt_id"],
        idempotency_key=row["idempotency_key"],
        request_digest=row["request_digest"],
        request=json.loads(row["request_json"]),
        plan=json.loads(row["plan_json"]),
        state=row["state"],
        owner=row["owner"],
        pid=row["pid"],
        pid_start=row["pid_start"],
        deadline=row["deadline"],
        result_digest=row["result_digest"],
        reason=row["reason"],
        created=row["created"],
    )


class JobJournal:
    """SQLite attempt and event journal.

    Parameters
    ----------
    path
        Journal file; created with its schema when absent.

    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the connection."""
        self._connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run the body in one immediate transaction, rolled back on any error."""
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        self._connection.execute("COMMIT")

    def _append(
        self,
        connection: sqlite3.Connection,
        attempt_id: str,
        kind: str,
        payload: Mapping[str, Any],
        at: float,
    ) -> Event:
        """Append the next event of an attempt inside an open transaction."""
        (last,) = connection.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM events WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        event = Event(attempt_id, int(last) + 1, kind, dict(payload), at)
        connection.execute(
            "INSERT INTO events (attempt_id, seq, kind, payload_json, at) VALUES (?, ?, ?, ?, ?)",
            (attempt_id, event.seq, kind, json.dumps(event.payload, sort_keys=True), at),
        )
        return event

    def admit(
        self,
        *,
        attempt_id: str,
        idempotency_key: str,
        request_digest: str,
        request: Mapping[str, Any],
        plan: Mapping[str, Any],
        at: float,
        max_queued: int,
    ) -> tuple[Attempt, bool]:
        """Admit a request once per idempotency key.

        Parameters
        ----------
        attempt_id
            Identity of a new attempt.
        idempotency_key
            Caller key.
        request_digest
            Digest of the request.
        request
            The request.
        plan
            Its resolved plan.
        at
            Admission time.
        max_queued
            Most attempts that may wait for a worker at once.

        Returns
        -------
        tuple
            The attempt and whether it was created now (``False`` for an
            identical repeat of an earlier request).

        Raises
        ------
        ValueError
            The key was used for a different request, or the queue is full.
            Nothing is written in either case.

        """
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM attempts WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if row is not None:
                existing = _attempt(row)
                if existing.request_digest != request_digest:
                    raise ValueError("idempotency key was already used for a different request")
                return existing, False
            (queued,) = connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE state = 'queued'"
            ).fetchone()
            if int(queued) >= max_queued:
                raise ValueError("the local job queue is full")
            connection.execute(
                "INSERT INTO attempts (attempt_id, idempotency_key, request_digest, request_json,"
                " plan_json, state, created) VALUES (?, ?, ?, ?, ?, 'queued', ?)",
                (
                    attempt_id,
                    idempotency_key,
                    request_digest,
                    json.dumps(dict(request), sort_keys=True),
                    json.dumps(dict(plan), sort_keys=True),
                    at,
                ),
            )
            self._append(connection, attempt_id, "admitted", {"plan": dict(plan)}, at)
        return self.attempt(attempt_id), True

    def attempt(self, attempt_id: str) -> Attempt:
        """Return one attempt.

        Parameters
        ----------
        attempt_id
            Attempt identity.

        Returns
        -------
        Attempt
            The journalled attempt.

        Raises
        ------
        KeyError
            No such attempt.

        """
        row = self._connection.execute(
            "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        return _attempt(row)

    def attempts(self, state: AttemptState) -> list[Attempt]:
        """Return the attempts in one state, oldest first.

        Parameters
        ----------
        state
            State to select.

        Returns
        -------
        list[Attempt]
            Matching attempts in admission order.

        """
        rows = self._connection.execute(
            "SELECT * FROM attempts WHERE state = ? ORDER BY created, attempt_id", (state,)
        ).fetchall()
        return [_attempt(row) for row in rows]

    def start(
        self,
        attempt_id: str,
        *,
        owner: str,
        pid: int,
        pid_start: str,
        deadline: float,
        at: float,
    ) -> Event:
        """Mark a queued attempt as running under a recorded worker process.

        Parameters
        ----------
        attempt_id
            Attempt identity.
        owner
            Service instance that owns the worker.
        pid
            Worker process id.
        pid_start
            Kernel start time of the worker.
        deadline
            Wall-clock expiry of the attempt.
        at
            Start time.

        Returns
        -------
        Event
            The ``started`` event.

        Raises
        ------
        ValueError
            The attempt is not queued.

        """
        with self._transaction() as connection:
            changed = connection.execute(
                "UPDATE attempts SET state = 'running', owner = ?, pid = ?, pid_start = ?,"
                " deadline = ? WHERE attempt_id = ? AND state = 'queued'",
                (owner, pid, pid_start, deadline, attempt_id),
            ).rowcount
            if changed != 1:
                raise ValueError("only a queued attempt can start")
            return self._append(
                connection, attempt_id, "started", {"pid": pid, "deadline": deadline}, at
            )

    def finish(
        self,
        attempt_id: str,
        state: AttemptState,
        *,
        at: float,
        reason: str | None = None,
        result_digest: str | None = None,
    ) -> Event:
        """Move an attempt to a terminal state with its terminal event.

        Parameters
        ----------
        attempt_id
            Attempt identity.
        state
            Terminal state.
        at
            Time of the transition.
        reason
            Why it ended, for a non-succeeded state.
        result_digest
            Digest of the sealed record, for a succeeded state.

        Returns
        -------
        Event
            The terminal event.

        Raises
        ------
        ValueError
            ``state`` is not terminal, or the attempt has already ended.

        """
        if state not in TERMINAL_STATES:
            raise ValueError(f"{state!r} is not a terminal state")
        with self._transaction() as connection:
            changed = connection.execute(
                "UPDATE attempts SET state = ?, reason = ?, result_digest = ?"
                " WHERE attempt_id = ? AND state IN ('queued', 'running')",
                (state, reason, result_digest, attempt_id),
            ).rowcount
            if changed != 1:
                raise ValueError("the attempt has already ended")
            payload: dict[str, Any] = {}
            if reason is not None:
                payload["reason"] = reason
            if result_digest is not None:
                payload["result_digest"] = result_digest
            return self._append(connection, attempt_id, state, payload, at)

    def events(self, attempt_id: str, after: int) -> list[Event]:
        """Return the events of an attempt after a cursor, in order.

        Parameters
        ----------
        attempt_id
            Attempt identity.
        after
            Last sequence number the caller has seen; ``0`` for all.

        Returns
        -------
        list[Event]
            Events with ``seq > after``.

        """
        rows = self._connection.execute(
            "SELECT * FROM events WHERE attempt_id = ? AND seq > ? ORDER BY seq",
            (attempt_id, after),
        ).fetchall()
        return [
            Event(
                row["attempt_id"],
                row["seq"],
                row["kind"],
                json.loads(row["payload_json"]),
                row["at"],
            )
            for row in rows
        ]


__all__ = ["TERMINAL_STATES", "Attempt", "AttemptState", "Event", "JobJournal"]
