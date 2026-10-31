# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — optional local job service around the executive handlers
"""Run admitted Studio actions as durable local jobs that outlive their client.

A Studio user can start an existing local handler, such as ``simulate``, close
or lose the client, and later reconnect to the same logical attempt, its
ordered events and its sealed result, without running it again.

* **Local only.** The HTTP transport binds a loopback address, requires a
  per-run session token and an allowed ``Host`` and ``Origin``, and exposes no
  provider route: the registry holds only verbs whose federation contract has
  no approval requirement and no live-hardware effect.
* **Typed operations.** ``plan``, ``submit``, ``events(after)``, ``retrieve``
  and ``cancel``. A request names a project, a revision, an idempotency key,
  a verb, an action id, parameters and optionally a backend, and nothing else:
  a caller cannot choose an executable, an import path, a file, a remote
  target or credentials.
* **One attempt per key.** A repeated identical request returns the same
  attempt; a different request under a used key is refused without any write.
* **Owned processes.** Each attempt runs in its own process group with memory,
  CPU-time and file-size limits. Cancellation, expiry and shutdown record their
  terminal event only after the group is terminated and reaped.
* **Restart reconciliation.** A new service instance first settles every
  attempt the previous one left running: it stops a surviving worker, accepts
  a complete sealed record as proof of completion, and otherwise records the
  attempt as interrupted. It never runs an indeterminate attempt again.

Every bound is an explicit :class:`LocalJobPolicy` value; nothing is inferred
from the host.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import signal
import subprocess  # nosec B404 - fixed worker module, no shell, no caller argument
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final
from urllib.parse import parse_qs, urlsplit

from ...studio.executive import ExecutiveRequest, preview_action
from .journal import Attempt, JobJournal
from .registry import build_local_registry, local_verbs
from .worker import RECORD_FILE, REQUEST_FILE

REQUEST_FIELDS: Final[frozenset[str]] = frozenset(
    {"project", "revision", "idempotency_key", "verb", "action_id", "parameters", "backend"}
)
"""The only fields a job request may carry."""
TOKEN_HEADER: Final[str] = "X-Studio-Job-Token"
"""Header carrying the per-run session token."""
LOOPBACK_HOST: Final[str] = "127.0.0.1"
"""The only address the transport binds."""
UNEXPECTED_ERROR: Final[str] = "the local job service could not complete the request"
"""Fixed text returned for any unexpected server error."""

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_WORKER_MODULE: Final[str] = "scpn_quantum_control.studio_workspace.jobs.worker"


class JobRefused(ValueError):
    """A request the service refuses, with a stable code and caller-safe text.

    Parameters
    ----------
    code
        Stable refusal category.
    message
        Explanation that is safe to return to the caller.

    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class LocalJobPolicy:
    """Explicit bounds of one service instance.

    Parameters
    ----------
    max_workers
        Attempts that may run at once.
    max_queued
        Attempts that may wait for a worker.
    worker_seconds
        Wall-clock limit of one attempt; it expires after that.
    worker_memory_bytes
        Address-space limit of one worker process.
    max_request_bytes
        Largest encoded request.
    max_record_bytes
        Largest sealed record a worker may write.
    max_journal_bytes
        Journal size above which new requests are refused.
    cancel_grace_seconds
        Time a worker group gets after ``SIGTERM`` before ``SIGKILL``.

    """

    max_workers: int = 2
    max_queued: int = 16
    worker_seconds: float = 600.0
    worker_memory_bytes: int = 4 * 1024**3
    max_request_bytes: int = 1024**2
    max_record_bytes: int = 16 * 1024**2
    max_journal_bytes: int = 256 * 1024**2
    cancel_grace_seconds: float = 2.0

    def __post_init__(self) -> None:
        """Refuse bounds that are not positive and finite."""
        for name in (
            "max_workers",
            "max_queued",
            "worker_memory_bytes",
            "max_request_bytes",
            "max_record_bytes",
            "max_journal_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("worker_seconds", "cancel_grace_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be a number")
            if not 0.0 < float(value) <= 86400.0:
                raise ValueError(f"{name} must lie in (0, 86400] seconds")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready policy."""
        return {
            "max_workers": self.max_workers,
            "max_queued": self.max_queued,
            "worker_seconds": self.worker_seconds,
            "worker_memory_bytes": self.worker_memory_bytes,
            "max_request_bytes": self.max_request_bytes,
            "max_record_bytes": self.max_record_bytes,
            "max_journal_bytes": self.max_journal_bytes,
            "cancel_grace_seconds": self.cancel_grace_seconds,
        }


def _digest(payload: object) -> str:
    """Return the SHA-256 of a payload's canonical JSON."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _identifier(value: object, field: str) -> str:
    """Return a bounded identifier, refusing anything path- or code-like."""
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise JobRefused(
            "invalid_request",
            f"{field} must be 1 to 128 letters, digits, '.', '_', ':' or '-', "
            "starting with a letter or digit",
        )
    return value


def admit_request(payload: object, *, max_request_bytes: int) -> dict[str, Any]:
    """Validate a job request and return its normalised form.

    Parameters
    ----------
    payload
        Decoded request.
    max_request_bytes
        Largest admitted encoded request.

    Returns
    -------
    dict
        The request with every field present (``backend`` may be ``None``).

    Raises
    ------
    JobRefused
        The request has unknown or missing fields, malformed identifiers, a
        verb outside the local registry, non-object parameters, or is too
        large. Nothing is written.

    """
    if not isinstance(payload, Mapping):
        raise JobRefused("invalid_request", "a job request must be a JSON object")
    unknown = sorted(set(payload) - REQUEST_FIELDS)
    if unknown:
        raise JobRefused("invalid_request", f"unknown request fields: {', '.join(unknown)}")
    missing = sorted(REQUEST_FIELDS - {"backend"} - set(payload))
    if missing:
        raise JobRefused("invalid_request", f"missing request fields: {', '.join(missing)}")
    verb = payload["verb"]
    if verb not in local_verbs():
        raise JobRefused("verb_not_local", f"verb {verb!r} is not available to local jobs")
    parameters = payload["parameters"]
    if not isinstance(parameters, Mapping):
        raise JobRefused("invalid_request", "parameters must be a JSON object")
    backend = payload.get("backend")
    request = {
        "project": _identifier(payload["project"], "project"),
        "revision": _identifier(payload["revision"], "revision"),
        "idempotency_key": _identifier(payload["idempotency_key"], "idempotency_key"),
        "verb": verb,
        "action_id": _identifier(payload["action_id"], "action_id"),
        "parameters": dict(parameters),
        "backend": None if backend is None else _identifier(backend, "backend"),
    }
    try:
        encoded = json.dumps(request, sort_keys=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise JobRefused("invalid_request", "parameters must be finite JSON values") from exc
    if len(encoded) > max_request_bytes:
        raise JobRefused(
            "request_too_large", f"a request may have at most {max_request_bytes} bytes"
        )
    return request


def _executive_request(request: Mapping[str, Any]) -> ExecutiveRequest:
    """Return the executive request of an admitted job request; approval is never set."""
    return ExecutiveRequest(
        verb=request["verb"],
        action_id=request["action_id"],
        parameters=request["parameters"],
        backend=request["backend"],
    )


def _process_start(pid: int) -> str:
    """Return the kernel start time of a running process, or ``""`` when it has ended.

    Together with the pid it identifies one process, since a pid is reused. A
    zombie has ended; only its parent's wait remains.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError:
        return ""
    # The fields follow the parenthesised command name, which may contain spaces:
    # field 3 is the state and field 22 the start time.
    fields = stat.rsplit(")", 1)[1].split()
    return "" if fields[0] == "Z" else fields[19]


def _wait_gone(pid: int, start: str, timeout: float) -> bool:
    """Wait until a process with this pid and start time no longer exists."""
    stop = time.monotonic() + timeout
    while _process_start(pid) == start:
        if time.monotonic() >= stop:
            return False
        time.sleep(0.02)
    return True


def _signal_group(pid: int, signum: int) -> None:
    """Signal a worker's process group, which is led by the worker itself."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signum)


def attempt_view(attempt: Attempt) -> dict[str, Any]:
    """Return the caller-facing state of an attempt.

    Parameters
    ----------
    attempt
        Journalled attempt.

    Returns
    -------
    dict
        Identity, request identity, plan, state and, once ended, the reason or
        the record digest. Process ids and owners are not exposed.

    """
    request = attempt.request
    return {
        "attempt_id": attempt.attempt_id,
        "project": request["project"],
        "revision": request["revision"],
        "idempotency_key": attempt.idempotency_key,
        "verb": request["verb"],
        "action_id": request["action_id"],
        "plan": dict(attempt.plan),
        "state": attempt.state,
        "terminal": attempt.terminal,
        "reason": attempt.reason,
        "result_digest": attempt.result_digest,
    }


class LocalJobService:
    """Durable local jobs over the local-only executive registry.

    Parameters
    ----------
    root
        Service-owned directory for the journal and attempt directories.
    policy
        Explicit bounds.
    clock
        Wall-clock source, used for events and deadlines.

    Notes
    -----
    Construction reconciles every attempt a previous instance left running.

    """

    def __init__(
        self,
        root: Path,
        policy: LocalJobPolicy | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = root
        self.policy = LocalJobPolicy() if policy is None else policy
        self._clock = clock
        self._attempts_dir = root / "attempts"
        self._attempts_dir.mkdir(parents=True, exist_ok=True)
        self.journal = JobJournal(root / "journal.sqlite3")
        self.owner = f"{os.getpid()}:{uuid.uuid4().hex}"
        self._lock = threading.RLock()
        self._workers: dict[str, subprocess.Popen[bytes]] = {}
        self._registry = build_local_registry()
        self.reconcile()

    # -- requests ---------------------------------------------------------

    def _planned(self, payload: object) -> tuple[dict[str, Any], dict[str, object]]:
        """Admit a request and resolve its plan through the handler."""
        request = admit_request(payload, max_request_bytes=self.policy.max_request_bytes)
        try:
            plan = preview_action(_executive_request(request), registry=self._registry)
        except (KeyError, ValueError) as exc:
            raise JobRefused("invalid_parameters", str(exc)) from exc
        return request, plan.to_dict()

    def plan(self, payload: object) -> dict[str, Any]:
        """Return the plan of a request without admitting it.

        Parameters
        ----------
        payload
            Decoded job request.

        Returns
        -------
        dict
            The handler's inspectable plan.

        Raises
        ------
        JobRefused
            The request or its parameters are refused.

        """
        return self._planned(payload)[1]

    def submit(self, payload: object) -> dict[str, Any]:
        """Admit a request once per idempotency key and start it when a worker is free.

        Parameters
        ----------
        payload
            Decoded job request.

        Returns
        -------
        dict
            The attempt view with ``created``: ``False`` for an identical repeat.

        Raises
        ------
        JobRefused
            The request is refused, its key was used for a different request,
            the queue is full or the journal has reached its bound. Nothing is
            written.

        """
        request, plan = self._planned(payload)
        if self._journal_bytes() > self.policy.max_journal_bytes:
            raise JobRefused("journal_full", "the local job journal has reached its size bound")
        request_digest = _digest(request)
        attempt_id = (
            "job-" + _digest({"request": request_digest, "key": request["idempotency_key"]})[:32]
        )
        with self._lock:
            try:
                attempt, created = self.journal.admit(
                    attempt_id=attempt_id,
                    idempotency_key=request["idempotency_key"],
                    request_digest=request_digest,
                    request=request,
                    plan=plan,
                    at=self._clock(),
                    max_queued=self.policy.max_queued,
                )
            except ValueError as exc:
                code = "queue_full" if "queue" in str(exc) else "idempotency_conflict"
                raise JobRefused(code, str(exc)) from exc
            self.pump()
            view = attempt_view(self.journal.attempt(attempt.attempt_id))
        return {**view, "created": created}

    def _journal_bytes(self) -> int:
        """Return the journal's size on disk, including its write-ahead log."""
        return sum(
            path.stat().st_size
            for path in (self.journal.path, self.journal.path.with_name("journal.sqlite3-wal"))
            if path.exists()
        )

    def _attempt(self, attempt_id: object) -> Attempt:
        """Return a journalled attempt or refuse an unknown identity."""
        if not isinstance(attempt_id, str):
            raise JobRefused("unknown_attempt", "no such attempt")
        try:
            return self.journal.attempt(attempt_id)
        except KeyError as exc:
            raise JobRefused("unknown_attempt", "no such attempt") from exc

    def events(self, attempt_id: str, after: int = 0) -> list[dict[str, Any]]:
        """Return an attempt's events after a cursor.

        Parameters
        ----------
        attempt_id
            Attempt identity.
        after
            Last sequence number already seen; ``0`` replays the whole history.

        Returns
        -------
        list of dict
            Events in order, each with its ``seq``.

        Raises
        ------
        JobRefused
            Unknown attempt or a negative cursor.

        """
        self._attempt(attempt_id)
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise JobRefused("invalid_request", "after must be a non-negative integer")
        self.pump()
        return [event.to_dict() for event in self.journal.events(attempt_id, after)]

    def retrieve(self, attempt_id: str) -> dict[str, Any]:
        """Return an attempt and, once it has one, its sealed record.

        Parameters
        ----------
        attempt_id
            Attempt identity.

        Returns
        -------
        dict
            The attempt view and ``record`` (``None`` until a record exists).

        Raises
        ------
        JobRefused
            Unknown attempt.

        """
        self.pump()
        attempt = self._attempt(attempt_id)
        record = None
        if attempt.result_digest is not None:
            record = self._read_record(attempt.attempt_id)
        return {**attempt_view(attempt), "record": record}

    def cancel(self, attempt_id: str) -> dict[str, Any]:
        """Cancel a queued or running attempt; an ended attempt is returned as it is.

        Parameters
        ----------
        attempt_id
            Attempt identity.

        Returns
        -------
        dict
            The attempt view after cancellation.

        Raises
        ------
        JobRefused
            Unknown attempt.

        """
        with self._lock:
            attempt = self._attempt(attempt_id)
            if not attempt.terminal:
                worker = self._workers.pop(attempt.attempt_id, None)
                if worker is not None:
                    self._terminate(worker)
                self.journal.finish(
                    attempt.attempt_id,
                    "cancelled",
                    at=self._clock(),
                    reason="cancelled by request",
                )
            return attempt_view(self.journal.attempt(attempt.attempt_id))

    # -- workers ----------------------------------------------------------

    def _terminate(self, worker: subprocess.Popen[bytes]) -> None:
        """Stop a worker's process group and reap the worker."""
        _signal_group(worker.pid, signal.SIGTERM)
        try:
            worker.wait(timeout=self.policy.cancel_grace_seconds)
        except subprocess.TimeoutExpired:
            _signal_group(worker.pid, signal.SIGKILL)
            worker.wait()
        # Any other member of the group is stopped too.
        _signal_group(worker.pid, signal.SIGKILL)

    def _cpu_seconds(self) -> int:
        """CPU-time limit of a worker: every core busy for the whole wall-clock limit.

        The wall-clock deadline is the attempt's limit; the CPU limit only
        backstops a worker the service can no longer watch, so it never ends
        an attempt before its deadline.
        """
        return int(self.policy.worker_seconds * (os.cpu_count() or 1)) + 1

    def _start(self, attempt: Attempt) -> None:
        """Start the worker of a queued attempt and record its process."""
        directory = self._attempts_dir / attempt.attempt_id
        directory.mkdir(exist_ok=True)
        (directory / REQUEST_FILE).write_text(json.dumps(dict(attempt.request)), encoding="utf-8")
        log = (directory / "worker.log").open("wb")
        try:
            worker = subprocess.Popen(  # nosec B603 - fixed interpreter and module
                [
                    sys.executable,
                    "-m",
                    _WORKER_MODULE,
                    str(directory),
                    f"--memory-bytes={self.policy.worker_memory_bytes}",
                    f"--cpu-seconds={self._cpu_seconds()}",
                    f"--file-bytes={self.policy.max_record_bytes}",
                ],
                cwd=directory,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log.close()
        now = self._clock()
        self._workers[attempt.attempt_id] = worker
        self.journal.start(
            attempt.attempt_id,
            owner=self.owner,
            pid=worker.pid,
            pid_start=_process_start(worker.pid),
            deadline=now + self.policy.worker_seconds,
            at=now,
        )

    def _read_record(self, attempt_id: str) -> dict[str, Any] | None:
        """Return a complete sealed record of an attempt, or ``None``."""
        path = self._attempts_dir / attempt_id / RECORD_FILE
        try:
            if path.stat().st_size > self.policy.max_record_bytes:
                return None
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(record, dict) or not isinstance(record.get("digest"), str):
            return None
        return record

    def _settle(self, attempt_id: str, *, recovered: bool, status: int | None = None) -> None:
        """Record the terminal state of an attempt whose worker has stopped."""
        record = self._read_record(attempt_id)
        now = self._clock()
        if record is None:
            if recovered:
                reason = "the service stopped while the attempt was running"
            elif status is not None and status < 0:
                reason = (
                    f"the worker stopped without a sealed record ({signal.Signals(-status).name})"
                )
            else:
                reason = f"the worker stopped without a sealed record (exit status {status})"
            self.journal.finish(
                attempt_id, "interrupted" if recovered else "failed", at=now, reason=reason
            )
            return
        result = record.get("result", {})
        if isinstance(result, dict) and result.get("status") == "succeeded":
            self.journal.finish(attempt_id, "succeeded", at=now, result_digest=record["digest"])
        else:
            error = result.get("error") if isinstance(result, dict) else None
            self.journal.finish(
                attempt_id,
                "failed",
                at=now,
                reason=error if isinstance(error, str) else "the action did not succeed",
                result_digest=record["digest"],
            )

    def pump(self) -> None:
        """Settle finished and expired workers, then start queued attempts."""
        with self._lock:
            now = self._clock()
            for attempt_id, worker in list(self._workers.items()):
                attempt = self.journal.attempt(attempt_id)
                status = worker.poll()
                if status is not None:
                    _signal_group(worker.pid, signal.SIGKILL)
                    del self._workers[attempt_id]
                    self._settle(attempt_id, recovered=False, status=status)
                elif attempt.deadline is not None and now > attempt.deadline:
                    del self._workers[attempt_id]
                    self._terminate(worker)
                    self.journal.finish(
                        attempt_id,
                        "expired",
                        at=self._clock(),
                        reason=f"the attempt exceeded {self.policy.worker_seconds} s",
                    )
            for attempt in self.journal.attempts("queued"):
                if len(self._workers) >= self.policy.max_workers:
                    break
                self._start(attempt)

    def reconcile(self) -> None:
        """Settle every attempt a previous instance left running, without rerunning it."""
        with self._lock:
            # Called once, before this instance has started any worker, so every
            # running attempt belongs to an earlier instance.
            for attempt in self.journal.attempts("running"):
                if (
                    attempt.pid is not None
                    and attempt.pid_start
                    and _process_start(attempt.pid) == attempt.pid_start
                ):
                    _signal_group(attempt.pid, signal.SIGKILL)
                    _wait_gone(attempt.pid, attempt.pid_start, 5.0)
                self._settle(attempt.attempt_id, recovered=True)

    def close(self) -> None:
        """Stop every running worker, record it as interrupted and close the journal."""
        with self._lock:
            for attempt_id, worker in list(self._workers.items()):
                self._terminate(worker)
                del self._workers[attempt_id]
                self.journal.finish(
                    attempt_id, "interrupted", at=self._clock(), reason="the service was stopped"
                )
            self.journal.close()


# -- loopback HTTP transport ------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    """Route the five operations; every refusal is JSON with a stable code."""

    server: LocalJobServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep request lines out of the service's output."""

    def _send(self, status: int, body: object) -> None:
        payload = json.dumps(body, sort_keys=True).encode("utf-8")
        self.send_response(status)
        origin = self.headers.get("Origin")
        if origin is not None and origin in self.server.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _refuse(self, status: int, code: str, message: str) -> None:
        # A refused request may leave its body unread, so the connection ends here.
        self.close_connection = True
        self._send(status, {"error": code, "message": message})

    def _admitted(self) -> bool:
        """Check host, origin and token before anything else is read."""
        host = self.headers.get("Host", "")
        if host not in self.server.allowed_hosts:
            self._refuse(421, "host_refused", "requests must address the loopback service")
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.server.allowed_origins:
            self._refuse(403, "origin_refused", "this origin may not use the local job service")
            return False
        token = self.headers.get(TOKEN_HEADER, "")
        if not hmac.compare_digest(token.encode("utf-8"), self.server.token.encode("utf-8")):
            self._refuse(401, "token_refused", "a valid session token is required")
            return False
        return True

    def _body(self) -> object:
        length = self.headers.get("Content-Length")
        if length is None or not length.isdigit():
            raise JobRefused("invalid_request", "a JSON body with Content-Length is required")
        if int(length) > self.server.service.policy.max_request_bytes:
            raise JobRefused("request_too_large", "the request body is too large")
        if self.headers.get_content_type() != "application/json":
            raise JobRefused("invalid_request", "the body must be application/json")
        try:
            return json.loads(self.rfile.read(int(length)))
        except ValueError as exc:
            raise JobRefused("invalid_request", "the body is not valid JSON") from exc

    def _dispatch(self, method: str) -> None:
        if not self._admitted():
            return
        service = self.server.service
        target = urlsplit(self.path)
        parts = [part for part in target.path.split("/") if part]
        try:
            if method == "POST" and parts == ["v1", "plan"]:
                self._send(200, {"plan": service.plan(self._body())})
            elif method == "POST" and parts == ["v1", "jobs"]:
                view = service.submit(self._body())
                self._send(201 if view["created"] else 200, view)
            elif method == "GET" and len(parts) == 3 and parts[:2] == ["v1", "jobs"]:
                self._send(200, service.retrieve(parts[2]))
            elif method == "GET" and len(parts) == 4 and parts[3] == "events":
                after = parse_qs(target.query).get("after", ["0"])[-1]
                if not after.isdigit():
                    raise JobRefused("invalid_request", "after must be a non-negative integer")
                self._send(200, {"events": service.events(parts[2], int(after))})
            elif method == "POST" and len(parts) == 4 and parts[3] == "cancel":
                self._send(200, service.cancel(parts[2]))
            else:
                self._refuse(404, "unknown_route", "no such operation")
        except JobRefused as refusal:
            status = (
                404
                if refusal.code == "unknown_attempt"
                else 409
                if refusal.code
                in {
                    "idempotency_conflict",
                    "queue_full",
                    "journal_full",
                }
                else 400
            )
            self._refuse(status, refusal.code, str(refusal))
        except Exception:  # noqa: BLE001 - caller receives fixed text only
            self._refuse(500, "unexpected_error", UNEXPECTED_ERROR)

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        """Serve a read operation."""
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        """Serve a write operation."""
        self._dispatch("POST")

    def do_OPTIONS(self) -> None:  # noqa: N802 - http.server naming
        """Answer a browser preflight for an allowed origin only."""
        origin = self.headers.get("Origin")
        if self.headers.get("Host", "") not in self.server.allowed_hosts or (
            origin not in self.server.allowed_origins
        ):
            self._refuse(403, "origin_refused", "this origin may not use the local job service")
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", str(origin))
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", f"Content-Type, {TOKEN_HEADER}")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Vary", "Origin")
        self.send_header("Content-Length", "0")
        self.end_headers()


class LocalJobServer(ThreadingHTTPServer):
    """Loopback HTTP server of one :class:`LocalJobService`.

    Parameters
    ----------
    service
        The service it exposes.
    host
        Address to bind; only ``127.0.0.1`` is admitted.
    port
        Port, or ``0`` for any free port.
    token
        Session token every request must carry.
    allowed_origins
        Browser origins that may call the service.

    Raises
    ------
    ValueError
        ``host`` is not ``127.0.0.1`` or the token is shorter than 32
        characters.

    """

    daemon_threads = True

    def __init__(
        self,
        service: LocalJobService,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        token: str,
        allowed_origins: Sequence[str] = (),
    ) -> None:
        if host != LOOPBACK_HOST:
            raise ValueError("the local job service binds 127.0.0.1 only")
        if len(token) < 32:
            raise ValueError("the session token must have at least 32 characters")
        super().__init__((host, port), _Handler)
        self.service = service
        self.token = token
        self.allowed_origins = frozenset(allowed_origins)
        bound = self.server_address[1]
        self.allowed_hosts = frozenset({f"127.0.0.1:{bound}", f"localhost:{bound}"})
        self._stop = threading.Event()
        self._pump = threading.Thread(target=self._pump_loop, daemon=True)
        self._pump.start()

    @property
    def url(self) -> str:
        """Base URL of the service."""
        return f"http://{LOOPBACK_HOST}:{self.server_address[1]}"

    def _pump_loop(self) -> None:
        """Settle and start workers while no client is asking."""
        while not self._stop.wait(0.2):
            self.service.pump()

    def server_close(self) -> None:
        """Stop the pump thread and close the socket."""
        self._stop.set()
        self._pump.join()
        super().server_close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the service until interrupted.

    Parameters
    ----------
    argv
        Optional argument vector, defaulting to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` after a clean stop.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="service-owned storage directory")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-origin", action="append", default=[], help="browser origin allowed to call"
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        help="write the session token here (mode 0600) instead of stdout",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    token = secrets.token_urlsafe(32)
    service = LocalJobService(args.root)
    server = LocalJobServer(
        service, port=args.port, token=token, allowed_origins=args.allow_origin
    )
    if args.token_file is not None:
        descriptor = os.open(args.token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(token)
        print(json.dumps({"url": server.url}), flush=True)
    else:
        print(json.dumps({"url": server.url, "token": token}), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()
    return 0


__all__ = [
    "LOOPBACK_HOST",
    "REQUEST_FIELDS",
    "TOKEN_HEADER",
    "UNEXPECTED_ERROR",
    "JobRefused",
    "LocalJobPolicy",
    "LocalJobServer",
    "LocalJobService",
    "admit_request",
    "attempt_view",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
