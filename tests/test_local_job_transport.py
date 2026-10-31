# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — the local job service through its real transport and processes
"""Durable local jobs: real worker processes, real HTTP and a real SQLite journal.

Every attempt here runs the shipped ``simulate`` handler in its own process.
A three-node network finishes in about a second; an eight-node network on
256 steps runs long enough to be cancelled, expired or orphaned by a crash.
"""

from __future__ import annotations

import http.client
import json
import os
import resource
import runpy
import signal
import socket
import socketserver
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from scpn_quantum_control.studio_workspace.jobs import (
    TERMINAL_STATES,
    JobJournal,
    JobRefused,
    LocalJobPolicy,
    LocalJobServer,
    LocalJobService,
    admit_request,
    local_verbs,
)
from scpn_quantum_control.studio_workspace.jobs import service as service_module
from scpn_quantum_control.studio_workspace.jobs import worker as worker_module

TOKEN = "t" * 40
ORIGIN = "http://127.0.0.1:4173"
QUICK = {
    "K_nm": [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
    "omega": [0.9, 1.0, 1.1],
    "t_max": 1.0,
    "dt": 0.1,
    "trotter_per_step": 1,
    "trotter_order": 1,
}
SLOW = {
    "K_nm": [[0.0 if i == j else 0.5 for j in range(8)] for i in range(8)],
    "omega": [0.8 + 0.05 * i for i in range(8)],
    "t_max": 12.8,
    "dt": 0.05,
    "trotter_per_step": 1,
    "trotter_order": 1,
}


def _request(key: str, parameters: dict[str, Any] = QUICK, **changes: object) -> dict[str, Any]:
    """Return a job request for the shipped ``simulate`` handler."""
    request: dict[str, Any] = {
        "project": "project-1",
        "revision": "rev-1",
        "idempotency_key": key,
        "verb": "simulate",
        "action_id": f"sim-{key}",
        "parameters": parameters,
    }
    request.update(changes)
    return request


def _alive(pid: int) -> bool:
    """Whether a process exists and is not a zombie."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def _until(predicate: Any, timeout: float = 60.0) -> None:
    """Wait for a condition, failing the test at the timeout."""
    stop = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < stop, "condition not reached"
        time.sleep(0.05)


def _pid(root: Path, attempt_id: str) -> int:
    """Read a running attempt's worker pid straight from the journal file."""
    with sqlite3.connect(root / "journal.sqlite3") as connection:
        (pid,) = connection.execute(
            "SELECT pid FROM attempts WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
    return int(pid)


def _counts(root: Path) -> tuple[int, int, int]:
    """Count attempts, events and attempt directories, to prove a refusal wrote nothing."""
    with sqlite3.connect(root / "journal.sqlite3") as connection:
        (attempts,) = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()
        (events,) = connection.execute("SELECT COUNT(*) FROM events").fetchone()
    return int(attempts), int(events), len(list((root / "attempts").iterdir()))


@pytest.fixture
def service(tmp_path: Path) -> Iterator[LocalJobService]:
    """Yield a service on an isolated root and close it after the test."""
    instance = LocalJobService(tmp_path / "jobs")
    yield instance
    instance.close()


@pytest.fixture
def server(service: LocalJobService) -> Iterator[LocalJobServer]:
    """Serve the loopback transport of ``service`` in a thread."""
    instance = LocalJobServer(service, port=0, token=TOKEN, allowed_origins=[ORIGIN])
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()
    thread.join()


def _call(
    server: LocalJobServer,
    method: str,
    path: str,
    body: object = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Make one request on a fresh connection and close it."""
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
    sent = (
        {TOKEN_HEADER: TOKEN, "Content-Type": "application/json"} if headers is None else headers
    )
    payload = None if body is None else json.dumps(body).encode("utf-8")
    try:
        connection.request(method, path, body=payload, headers=sent)
        response = connection.getresponse()
        data = response.read()
        return response.status, json.loads(data) if data else {}
    finally:
        connection.close()


TOKEN_HEADER = service_module.TOKEN_HEADER


def test_local_job_transport_01_handler_survives_a_disconnected_client(
    server: LocalJobServer,
) -> None:
    """A client that leaves mid-request or after submitting does not own the job."""
    # A client that sends half a request and leaves changes nothing.
    port = server.server_address[1]
    with socket.create_connection(("127.0.0.1", port)) as half:
        half.sendall(b"POST /v1/jobs HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n" % port)
    status, submitted = _call(server, "POST", "/v1/jobs", _request("survive"))
    assert status == 201 and submitted["created"] is True
    # No client is connected while the real handler runs to completion.
    attempt_id = submitted["attempt_id"]
    _until(lambda: _call(server, "GET", f"/v1/jobs/{attempt_id}")[1]["terminal"])
    status, retrieved = _call(server, "GET", f"/v1/jobs/{attempt_id}")
    assert status == 200 and retrieved["state"] == "succeeded"
    record = retrieved["record"]
    assert record["digest"] == retrieved["result_digest"]
    assert record["result"]["status"] == "succeeded"
    assert record["request"]["action_id"] == "sim-survive"
    assert record["request"]["approved"] is False
    assert retrieved["plan"]["verb"] == "simulate"


def test_local_job_transport_02_cursor_reconstructs_monotonic_events(
    server: LocalJobServer,
) -> None:
    """A reconnecting client replays exactly the events after its cursor."""
    _, submitted = _call(server, "POST", "/v1/jobs", _request("cursor"))
    attempt_id = submitted["attempt_id"]
    seen: list[dict[str, Any]] = []
    while not seen or seen[-1]["kind"] not in TERMINAL_STATES:
        after = seen[-1]["seq"] if seen else 0
        status, page = _call(server, "GET", f"/v1/jobs/{attempt_id}/events?after={after}")
        assert status == 200
        seen.extend(page["events"])
        time.sleep(0.05)
    assert [event["seq"] for event in seen] == list(range(1, len(seen) + 1))
    assert [event["kind"] for event in seen] == ["admitted", "started", "succeeded"]
    _, full = _call(server, "GET", f"/v1/jobs/{attempt_id}/events")
    assert full["events"] == seen
    _, tail = _call(server, "GET", f"/v1/jobs/{attempt_id}/events?after=1")
    assert tail["events"] == seen[1:]
    status, refused = _call(server, "GET", f"/v1/jobs/{attempt_id}/events?after=-1")
    assert (status, refused["error"]) == (400, "invalid_request")


def test_local_job_transport_03_duplicate_key_returns_the_same_attempt(
    server: LocalJobServer, service: LocalJobService
) -> None:
    """An identical repeat is the same attempt; a conflicting reuse writes nothing."""
    status, first = _call(server, "POST", "/v1/jobs", _request("same"))
    assert status == 201
    status, again = _call(server, "POST", "/v1/jobs", _request("same"))
    assert status == 200 and again["created"] is False
    assert again["attempt_id"] == first["attempt_id"]
    before = _counts(service.root)
    status, refused = _call(server, "POST", "/v1/jobs", _request("same", {**QUICK, "t_max": 2.0}))
    assert (status, refused["error"]) == (409, "idempotency_conflict")
    assert _counts(service.root) == before
    _until(lambda: service.retrieve(first["attempt_id"])["terminal"])
    events = service.events(first["attempt_id"])
    assert [event["kind"] for event in events].count("started") == 1


def test_local_job_transport_04_crash_and_restart_reconcile(tmp_path: Path) -> None:
    """A killed service leaves a worker behind; the next instance stops it and records why."""
    root = tmp_path / "jobs"
    token_file = tmp_path / "token"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scpn_quantum_control.studio_workspace.jobs.service",
            "--root",
            str(root),
            "--port",
            "0",
            "--token-file",
            str(token_file),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        url = json.loads(process.stdout.readline())["url"]
        port = int(url.rsplit(":", 1)[1])
        assert token_file.stat().st_mode & 0o777 == 0o600
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        connection.request(
            "POST",
            "/v1/jobs",
            body=json.dumps(_request("crash", SLOW)),
            headers={
                TOKEN_HEADER: token_file.read_text(encoding="utf-8"),
                "Content-Type": "application/json",
            },
        )
        attempt_id = json.loads(connection.getresponse().read())["attempt_id"]
        connection.close()
        _until(lambda: _pid(root, attempt_id) is not None)
        worker = _pid(root, attempt_id)
        assert _alive(worker)
    finally:
        process.kill()
        process.wait()
    # The service is gone; its worker is not.
    assert _alive(worker)
    restarted = LocalJobService(root)
    try:
        assert not _alive(worker)
        view = restarted.retrieve(attempt_id)
        assert view["state"] == "interrupted"
        assert view["reason"] == "the service stopped while the attempt was running"
        assert [event["kind"] for event in restarted.events(attempt_id)] == [
            "admitted",
            "started",
            "interrupted",
        ]
        # An indeterminate attempt is never run again.
        restarted.pump()
        assert restarted.retrieve(attempt_id)["state"] == "interrupted"
    finally:
        restarted.close()


def test_restart_accepts_a_complete_record_as_proof(tmp_path: Path) -> None:
    """A worker that finished while no service watched it is settled from its record."""
    root = tmp_path / "jobs"
    first = LocalJobService(root)
    attempt_id = first.submit(_request("proof"))["attempt_id"]
    worker = _pid(root, attempt_id)
    _until(lambda: (root / "attempts" / attempt_id / "record.json").exists())
    _until(lambda: not _alive(worker))
    # The first instance never settles it; a second one does.
    second = LocalJobService(root)
    view = second.retrieve(attempt_id)
    assert view["state"] == "succeeded"
    assert view["record"]["digest"] == view["result_digest"]
    second.close()
    first.journal.close()


def test_restart_settles_failed_and_unreadable_records(tmp_path: Path) -> None:
    """A recovered failed record is a failure; a damaged one is an interruption."""
    root = tmp_path / "jobs"
    first = LocalJobService(root)
    failed = first.submit(_request("failed-record", SLOW))["attempt_id"]
    damaged = first.submit(_request("damaged-record", SLOW))["attempt_id"]
    first.policy = LocalJobPolicy(max_workers=5)
    odd = first.submit(_request("odd-record", SLOW))["attempt_id"]
    listed = first.submit(_request("listed-record", SLOW))["attempt_id"]
    oversized = first.submit(_request("oversized-record", SLOW))["attempt_id"]
    for attempt_id in (failed, damaged, odd, listed, oversized):
        os.killpg(_pid(root, attempt_id), signal.SIGKILL)
        _until(lambda attempt_id=attempt_id: not _alive(_pid(root, attempt_id)))
    (root / "attempts" / failed / "record.json").write_text(
        json.dumps({"digest": "sha256:x", "result": {"status": "failed", "error": "E: boom"}}),
        encoding="utf-8",
    )
    (root / "attempts" / damaged / "record.json").write_text("{", encoding="utf-8")
    (root / "attempts" / odd / "record.json").write_text(
        json.dumps({"digest": "sha256:y", "result": "not an object"}), encoding="utf-8"
    )
    (root / "attempts" / listed / "record.json").write_text("[]", encoding="utf-8")
    (root / "attempts" / oversized / "record.json").write_text(
        json.dumps({"digest": "sha256:z", "padding": "x" * 8192}), encoding="utf-8"
    )
    second = LocalJobService(root, LocalJobPolicy(max_record_bytes=4096))
    assert second.retrieve(failed)["state"] == "failed"
    assert second.retrieve(failed)["reason"] == "E: boom"
    assert second.retrieve(damaged)["state"] == "interrupted"
    assert second.retrieve(odd)["reason"] == "the action did not succeed"
    for attempt_id in (listed, oversized):
        assert second.retrieve(attempt_id)["state"] == "interrupted"
    second.close()
    first.journal.close()


def test_local_job_transport_05_cancellation_reaps_the_process(
    server: LocalJobServer, service: LocalJobService
) -> None:
    """Cancelling a running attempt ends only after its worker is gone."""
    _, submitted = _call(server, "POST", "/v1/jobs", _request("cancel", SLOW))
    attempt_id = submitted["attempt_id"]
    worker = _pid(service.root, attempt_id)
    assert _alive(worker)
    status, cancelled = _call(server, "POST", f"/v1/jobs/{attempt_id}/cancel", {})
    assert status == 200 and cancelled["state"] == "cancelled"
    assert not _alive(worker)
    assert not Path(f"/proc/{worker}").exists()
    status, again = _call(server, "POST", f"/v1/jobs/{attempt_id}/cancel", {})
    assert status == 200 and again["state"] == "cancelled"
    kinds = [event["kind"] for event in service.events(attempt_id)]
    assert kinds == ["admitted", "started", "cancelled"]


def test_queued_cancel_expiry_and_escalation(tmp_path: Path) -> None:
    """A queued attempt cancels without starting; a slow one expires and is killed."""
    policy = LocalJobPolicy(max_workers=1, worker_seconds=1.5, cancel_grace_seconds=0.001)
    service = LocalJobService(tmp_path / "jobs", policy)
    try:
        running = service.submit(_request("slow", SLOW))
        waiting = service.submit(_request("waiting", SLOW))
        assert (running["state"], waiting["state"]) == ("running", "queued")
        assert service.cancel(waiting["attempt_id"])["state"] == "cancelled"
        assert [event["kind"] for event in service.events(waiting["attempt_id"])] == [
            "admitted",
            "cancelled",
        ]
        worker = _pid(service.root, running["attempt_id"])
        _until(lambda: service.retrieve(running["attempt_id"])["terminal"])
        expired = service.retrieve(running["attempt_id"])
        assert expired["state"] == "expired" and "1.5 s" in expired["reason"]
        assert not _alive(worker)
    finally:
        service.close()


def test_queue_and_journal_bounds_refuse_without_writing(tmp_path: Path) -> None:
    """A full queue or journal refuses new work; nothing is written."""
    policy = LocalJobPolicy(max_workers=1, max_queued=1)
    service = LocalJobService(tmp_path / "jobs", policy)
    try:
        service.submit(_request("one", SLOW))
        service.submit(_request("two", SLOW))
        before = _counts(service.root)
        with pytest.raises(JobRefused, match="queue is full") as full:
            service.submit(_request("three", SLOW))
        assert full.value.code == "queue_full"
        assert _counts(service.root) == before
    finally:
        service.close()
    tiny = LocalJobService(tmp_path / "tiny", LocalJobPolicy(max_journal_bytes=1))
    with pytest.raises(JobRefused) as journal:
        tiny.submit(_request("four"))
    assert journal.value.code == "journal_full"
    tiny.close()


def test_a_record_over_its_bound_is_never_accepted(tmp_path: Path) -> None:
    """A worker may not write past the record bound; the attempt fails without a record."""
    service = LocalJobService(tmp_path / "jobs", LocalJobPolicy(max_record_bytes=1024))
    try:
        attempt_id = service.submit(_request("big"))["attempt_id"]
        _until(lambda: service.retrieve(attempt_id)["terminal"])
        view = service.retrieve(attempt_id)
        assert view["state"] == "failed"
        # Python ignores SIGXFSZ, so the oversized write fails inside the worker.
        assert view["reason"].startswith("the worker stopped without a sealed record")
        assert view["record"] is None
    finally:
        service.close()


def test_a_worker_killed_from_outside_is_reported_with_its_signal(
    service: LocalJobService,
) -> None:
    """A worker ended by a signal, for example the kernel's out-of-memory killer, is named."""
    attempt_id = service.submit(_request("killed", SLOW))["attempt_id"]
    os.killpg(_pid(service.root, attempt_id), signal.SIGKILL)
    _until(lambda: service.retrieve(attempt_id)["terminal"])
    view = service.retrieve(attempt_id)
    assert (view["state"], view["reason"]) == (
        "failed",
        "the worker stopped without a sealed record (SIGKILL)",
    )


def test_the_memory_bound_stops_a_worker(tmp_path: Path) -> None:
    """A worker cannot load the numerical stack within a 64 MiB address space."""
    service = LocalJobService(tmp_path / "jobs", LocalJobPolicy(worker_memory_bytes=64 * 1024**2))
    try:
        attempt_id = service.submit(_request("small-memory"))["attempt_id"]
        _until(lambda: service.retrieve(attempt_id)["terminal"])
        view = service.retrieve(attempt_id)
        assert view["state"] == "failed"
        # The allocation that fails first, and how it reports, depends on the libraries.
        assert view["reason"].startswith("the worker stopped without a sealed record")
        assert view["record"] is None
        log = service.root / "attempts" / attempt_id / "worker.log"
        assert log.read_text(encoding="utf-8")
    finally:
        service.close()


def test_closing_the_service_interrupts_running_work(tmp_path: Path) -> None:
    """A stopped service leaves no worker and says why each attempt ended."""
    service = LocalJobService(tmp_path / "jobs")
    attempt_id = service.submit(_request("closing", SLOW))["attempt_id"]
    worker = _pid(service.root, attempt_id)
    service.close()
    assert not _alive(worker)
    reopened = LocalJobService(tmp_path / "jobs")
    view = reopened.retrieve(attempt_id)
    assert (view["state"], view["reason"]) == ("interrupted", "the service was stopped")
    reopened.close()


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"executable": "/bin/sh"}, "invalid_request"),
        ({"script": "import os"}, "invalid_request"),
        ({"cwd": "/tmp"}, "invalid_request"),
        ({"approved": True}, "invalid_request"),
        ({"project": "../escape"}, "invalid_request"),
        ({"action_id": "a/b"}, "invalid_request"),
        ({"revision": 7}, "invalid_request"),
        ({"backend": "/usr/bin/python"}, "invalid_request"),
        ({"verb": "execute"}, "verb_not_local"),
        ({"verb": "shell"}, "verb_not_local"),
        ({"parameters": ["not", "an", "object"]}, "invalid_request"),
        ({"parameters": {"t_max": float("nan")}}, "invalid_request"),
        ({"parameters": {**QUICK, "K_nm": [[0.0]]}}, "invalid_parameters"),
    ],
    ids=[
        "executable",
        "script",
        "cwd",
        "approval",
        "path-project",
        "path-action",
        "number-revision",
        "path-backend",
        "provider-verb",
        "unknown-verb",
        "list-parameters",
        "nan",
        "one-node",
    ],
)
def test_local_job_transport_06_paths_and_code_are_refused(
    service: LocalJobService, change: dict[str, object], code: str
) -> None:
    """No executable, path, script, approval or provider route reaches a worker.

    Parameters
    ----------
    service
        Isolated service.
    change
        Field added to or replaced in a valid request.
    code
        Expected refusal category.

    """
    before = _counts(service.root)
    with pytest.raises(JobRefused) as refused:
        service.submit({**_request("refused"), **change})
    assert refused.value.code == code
    assert _counts(service.root) == before


def test_requests_are_shaped_and_bounded() -> None:
    """Missing fields, non-objects and oversized requests are refused."""
    with pytest.raises(JobRefused, match="JSON object"):
        admit_request([], max_request_bytes=1024)
    missing = _request("missing")
    del missing["action_id"]
    with pytest.raises(JobRefused, match="missing request fields: action_id"):
        admit_request(missing, max_request_bytes=1024)
    with pytest.raises(JobRefused) as large:
        admit_request(_request("large"), max_request_bytes=16)
    assert large.value.code == "request_too_large"
    admitted = admit_request(_request("ok", backend="python"), max_request_bytes=4096)
    assert admitted["backend"] == "python"
    assert "execute" not in local_verbs() and "simulate" in local_verbs()


def test_plan_previews_without_admitting(service: LocalJobService) -> None:
    """A plan is the handler's own and leaves the journal untouched."""
    before = _counts(service.root)
    plan = service.plan(_request("plan"))
    assert plan["verb"] == "simulate" and plan["contract"]["requires_approval"] is False
    assert _counts(service.root) == before


def test_unknown_attempts_and_cursors_are_refused(service: LocalJobService) -> None:
    """Unknown identities and malformed cursors are refused with stable codes."""
    for call in (service.retrieve, service.cancel, service.events):
        with pytest.raises(JobRefused) as unknown:
            call("job-missing")
        assert unknown.value.code == "unknown_attempt"
    with pytest.raises(JobRefused):
        service.retrieve(cast(str, 7))
    attempt_id = service.submit(_request("cursor"))["attempt_id"]
    with pytest.raises(JobRefused, match="non-negative"):
        service.events(attempt_id, -1)


def test_transport_admits_only_its_host_origin_and_token(server: LocalJobServer) -> None:
    """Host, origin and token are checked before any operation runs."""
    port = server.server_address[1]
    base = {TOKEN_HEADER: TOKEN, "Content-Type": "application/json"}
    status, body = _call(server, "GET", "/v1/jobs/x", headers={**base, "Host": "evil.example"})
    assert (status, body["error"]) == (421, "host_refused")
    status, body = _call(server, "GET", "/v1/jobs/x", headers={**base, "Origin": "http://evil"})
    assert (status, body["error"]) == (403, "origin_refused")
    status, body = _call(server, "GET", "/v1/jobs/x", headers={**base, TOKEN_HEADER: "wrong"})
    assert (status, body["error"]) == (401, "token_refused")
    status, body = _call(server, "GET", "/v1/jobs/x", headers={**base, "Origin": ORIGIN})
    assert (status, body["error"]) == (404, "unknown_attempt")
    for method, headers, expected in (
        ("GET", {**base, "Origin": ORIGIN}, 404),
        ("OPTIONS", {"Origin": ORIGIN}, 204),
        ("OPTIONS", {"Origin": "http://evil"}, 403),
    ):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        connection.request(method, "/v1/jobs/x", headers=headers)
        response = connection.getresponse()
        response.read()
        assert response.status == expected
        if expected != 403:
            assert response.getheader("Access-Control-Allow-Origin") == ORIGIN
        connection.close()
    assert server.url == f"http://127.0.0.1:{port}"


def test_transport_refuses_malformed_bodies_and_routes(server: LocalJobServer) -> None:
    """Bodies must be bounded JSON; unknown routes and server faults use fixed text."""
    port = server.server_address[1]
    status, body = _call(server, "POST", "/v1/nowhere", {})
    assert (status, body["error"]) == (404, "unknown_route")
    status, body = _call(server, "POST", "/v1/plan", _request("plan"))
    assert status == 200 and body["plan"]["verb"] == "simulate"
    cases: list[tuple[bytes, dict[str, str], str]] = [
        (b"{", {TOKEN_HEADER: TOKEN, "Content-Type": "application/json"}, "not valid JSON"),
        (b"{}", {TOKEN_HEADER: TOKEN, "Content-Type": "text/plain"}, "application/json"),
        (
            b"x" * (server.service.policy.max_request_bytes + 1),
            {TOKEN_HEADER: TOKEN, "Content-Type": "application/json"},
            "too large",
        ),
    ]
    for payload, headers, message in cases:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        connection.request("POST", "/v1/jobs", body=payload, headers=headers)
        response = connection.getresponse()
        assert message in json.loads(response.read())["message"]
        connection.close()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.putrequest("POST", "/v1/jobs")
    connection.putheader(TOKEN_HEADER, TOKEN)
    connection.endheaders()
    response = connection.getresponse()
    assert "Content-Length" in json.loads(response.read())["message"]
    connection.close()


def test_transport_hides_unexpected_errors(
    server: LocalJobServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected fault returns fixed text, never the interpreter's message."""

    def broken(_payload: object) -> dict[str, object]:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(server.service, "plan", broken)
    status, body = _call(server, "POST", "/v1/plan", _request("broken"))
    assert status == 500
    assert body == {"error": "unexpected_error", "message": service_module.UNEXPECTED_ERROR}


def test_server_binds_loopback_with_a_long_token(service: LocalJobService) -> None:
    """The transport refuses any other address and a short token."""
    with pytest.raises(ValueError, match="127.0.0.1 only"):
        LocalJobServer(service, host="0.0.0.0", token=TOKEN)  # nosec B104 - refusal under test
    with pytest.raises(ValueError, match="at least 32"):
        LocalJobServer(service, token="short")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_workers", 0),
        ("max_queued", True),
        ("worker_memory_bytes", 1.5),
        ("worker_seconds", "1"),
        ("worker_seconds", 0.0),
        ("cancel_grace_seconds", 1e6),
    ],
)
def test_policy_refuses_unbounded_values(field: str, value: object) -> None:
    """Every bound must be a positive, finite, explicit value.

    Parameters
    ----------
    field
        Policy field.
    value
        Refused value.

    """
    with pytest.raises(ValueError, match=field):
        LocalJobPolicy(**cast(dict[str, Any], {field: value}))


def test_policy_is_reported_as_given() -> None:
    """The JSON form names every bound."""
    policy = LocalJobPolicy(max_workers=3)
    assert policy.to_dict()["max_workers"] == 3
    assert set(policy.to_dict()) == {
        "max_workers",
        "max_queued",
        "worker_seconds",
        "worker_memory_bytes",
        "max_request_bytes",
        "max_record_bytes",
        "max_journal_bytes",
        "cancel_grace_seconds",
    }


def test_journal_enforces_its_own_invariants(tmp_path: Path) -> None:
    """Terminal attempts and events cannot change; transitions are checked."""
    journal = JobJournal(tmp_path / "journal.sqlite3")
    attempt, created = journal.admit(
        attempt_id="a",
        idempotency_key="k",
        request_digest="d",
        request={},
        plan={},
        at=1.0,
        max_queued=4,
    )
    assert created and not attempt.terminal
    with pytest.raises(ValueError, match="not a terminal state"):
        journal.finish("a", "running", at=2.0)
    journal.finish("a", "cancelled", at=2.0)
    with pytest.raises(ValueError, match="already ended"):
        journal.finish("a", "failed", at=3.0)
    with pytest.raises(ValueError, match="only a queued attempt"):
        journal.start("a", owner="o", pid=1, pid_start="1", deadline=9.0, at=3.0)
    with pytest.raises(KeyError):
        journal.attempt("missing")
    journal.close()
    with sqlite3.connect(tmp_path / "journal.sqlite3") as connection:
        for statement in (
            "UPDATE attempts SET reason = 'x'",
            "DELETE FROM attempts",
            "UPDATE events SET kind = 'x'",
            "DELETE FROM events",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(statement)


def test_package_exports_resolve_lazily() -> None:
    """Exports load from their owners; anything else is an attribute error."""
    import scpn_quantum_control.studio_workspace.jobs as jobs

    assert jobs.LocalJobService is LocalJobService
    with pytest.raises(AttributeError, match="no attribute 'missing'"):
        _ = jobs.missing


def test_process_identity_helpers() -> None:
    """A vanished pid has no start time; a living one is not reported gone."""
    assert service_module._process_start(2**22 + 12345) == ""
    start = service_module._process_start(os.getpid())
    assert start and not service_module._wait_gone(os.getpid(), start, 0.0)


def test_worker_runs_an_attempt_in_process(tmp_path: Path) -> None:
    """The worker entry applies its limits and seals the record atomically."""
    (tmp_path / worker_module.REQUEST_FILE).write_text(
        json.dumps({**_request("inline"), "backend": None}), encoding="utf-8"
    )
    unlimited = str(resource.RLIM_INFINITY)
    assert (
        worker_module.main(
            [
                str(tmp_path),
                f"--memory-bytes={unlimited}",
                f"--cpu-seconds={unlimited}",
                f"--file-bytes={unlimited}",
            ]
        )
        == 0
    )
    record = json.loads((tmp_path / worker_module.RECORD_FILE).read_text(encoding="utf-8"))
    assert record["result"]["status"] == "succeeded"
    assert not (tmp_path / (worker_module.RECORD_FILE + ".partial")).exists()


def test_service_main_prints_the_token_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a token file the session token is printed once, with the URL."""

    def stop(_self: LocalJobServer) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(LocalJobServer, "serve_forever", stop)
    monkeypatch.setattr(signal, "signal", lambda *_args: None)
    assert service_module.main(["--root", str(tmp_path / "jobs"), "--port", "0"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["url"].startswith("http://127.0.0.1:") and len(printed["token"]) >= 32
    token_file = tmp_path / "token"
    assert (
        service_module.main(
            ["--root", str(tmp_path / "jobs"), "--port", "0", "--token-file", str(token_file)]
        )
        == 0
    )
    assert "token" not in json.loads(capsys.readouterr().out)
    assert len(token_file.read_text(encoding="utf-8")) >= 32


def test_modules_run_as_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``python -m`` runs the service and the worker through their entry points."""

    def stop(_self: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(socketserver.BaseServer, "serve_forever", stop)
    monkeypatch.setattr(signal, "signal", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", ["service", "--root", str(tmp_path / "jobs"), "--port", "0"])
    with pytest.raises(SystemExit) as served:
        runpy.run_module("scpn_quantum_control.studio_workspace.jobs.service", run_name="__main__")
    assert served.value.code == 0
    capsys.readouterr()
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / worker_module.REQUEST_FILE).write_text(
        json.dumps({**_request("script"), "backend": None}), encoding="utf-8"
    )
    unlimited = str(resource.RLIM_INFINITY)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker",
            str(attempt),
            f"--memory-bytes={unlimited}",
            f"--cpu-seconds={unlimited}",
            f"--file-bytes={unlimited}",
        ],
    )
    with pytest.raises(SystemExit) as worked:
        runpy.run_module("scpn_quantum_control.studio_workspace.jobs.worker", run_name="__main__")
    assert worked.value.code == 0
    assert (attempt / worker_module.RECORD_FILE).exists()


FIXTURES = json.loads(
    (
        Path(__file__).resolve().parents[1] / "data/studio/local_job_transport_fixtures.json"
    ).read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", FIXTURES["requests"], ids=lambda case: case["name"])
def test_shared_fixtures_match_the_service(service: LocalJobService, case: dict[str, Any]) -> None:
    """The browser client and the service decide every shared request the same way.

    Parameters
    ----------
    service
        Isolated service.
    case
        Shared request, with its refusal code or ``None`` when accepted.

    """
    if case["code"] is None:
        assert service.plan(case["request"])["verb"] == case["request"]["verb"]
    else:
        with pytest.raises(JobRefused) as refused:
            service.plan(case["request"])
        assert refused.value.code == case["code"]


def test_shared_response_shapes_match_the_service(service: LocalJobService) -> None:
    """The fixture attempt and events have exactly the service's fields."""
    assert sorted(FIXTURES["request_fields"]) == sorted(service_module.REQUEST_FIELDS)
    assert FIXTURES["token_header"] == TOKEN_HEADER
    assert sorted(FIXTURES["terminal_states"]) == sorted(TERMINAL_STATES)
    view = service.submit(_request("shape"))
    _until(lambda: service.retrieve(view["attempt_id"])["terminal"])
    retrieved = service.retrieve(view["attempt_id"])
    assert set(FIXTURES["attempt"]) == set(retrieved) - {"record"}
    events = service.events(view["attempt_id"])
    assert [event["kind"] for event in events] == [event["kind"] for event in FIXTURES["events"]]
    for live, shared in zip(events, FIXTURES["events"], strict=True):
        assert set(live) == set(shared) and set(live["payload"]) == set(shared["payload"])
