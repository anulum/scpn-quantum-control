# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — provider job journal tests
"""Exercise actual SQLite identity, transactional dispatch and retained history."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from scpn_quantum_control.hardware.provider_job_journal import (
    ProviderJobJournal,
    SubmissionUnknownError,
)


def prepared(root: Path) -> tuple[ProviderJobJournal, str]:
    """Create one actual persisted request for state-transition fixtures."""
    journal = ProviderJobJournal(root)
    identity = str(uuid4())
    journal.prepare(
        identity,
        target="ibm_exact",
        payload=b"original-native-request",
        shots=16,
        experiment="custody",
        circuits=1,
    )
    return journal, identity


def test_immutable_prepare_and_detached_history(tmp_path: Path) -> None:
    """Idempotent preparation never permits a changed immutable native request."""
    journal, identity = prepared(tmp_path)
    before = journal.snapshot(identity)
    journal.prepare(
        identity,
        target="ibm_exact",
        payload=b"original-native-request",
        shots=16,
        experiment="custody",
        circuits=1,
    )
    assert journal.snapshot(identity) == before
    with pytest.raises(ValueError, match="identity differs"):
        journal.prepare(
            identity,
            target="ibm_other",
            payload=b"original-native-request",
            shots=16,
            experiment="custody",
            circuits=1,
        )
    assert journal.snapshot(identity) == before
    before["target"] = "edited"
    assert journal.snapshot(identity)["target"] == "ibm_exact"


@pytest.mark.parametrize(
    "override",
    [
        {"target": ""},
        {"experiment": ""},
        {"payload": b""},
        {"shots": True},
        {"shots": 0},
        {"shots": 2**63},
        {"circuits": True},
        {"circuits": 0},
        {"circuits": 257},
    ],
)
def test_invalid_request_never_creates_an_attempt(
    tmp_path: Path, override: dict[str, Any]
) -> None:
    """Malformed settings refuse before creating a durable dispatch identity."""
    journal = ProviderJobJournal(tmp_path)
    identity = str(uuid4())
    values: dict[str, Any] = {
        "target": "ibm_exact",
        "payload": b"request",
        "shots": 16,
        "experiment": "invalid",
        "circuits": 1,
    }
    values.update(override)
    with pytest.raises(ValueError):
        journal.prepare(identity, **values)
    with pytest.raises(KeyError):
        journal.snapshot(identity)


def test_concurrent_process_connections_reserve_only_one_dispatch(tmp_path: Path) -> None:
    """Distinct SQLite connections race through one atomic write-ahead boundary."""
    journal, identity = prepared(tmp_path)

    def enter() -> bool:
        try:
            ProviderJobJournal(tmp_path).begin_submit(identity)
        except SubmissionUnknownError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: enter(), range(4)))
    assert sum(results) == 1
    assert journal.snapshot(identity)["state"] == "submission_unknown"


def test_handle_binding_and_completion_preserve_cancellation_history(tmp_path: Path) -> None:
    """Completion wins later cancellation without overwriting partial evidence."""
    journal, identity = prepared(tmp_path)
    journal.begin_submit(identity)
    journal.bind_job(identity, "provider_original")
    journal.bind_job(identity, "provider_original")
    journal.record(identity, "submitted", {"partial_counts": {"1": 7}})
    journal.record(identity, "cancellation_requested", {})
    journal.record(identity, "submitted", {"provider_status": "RUNNING"})
    assert journal.snapshot(identity)["state"] == "cancellation_requested"
    journal.record(identity, "completed", {"counts": {"1": 16}})
    terminal = journal.snapshot(identity)
    journal.record(identity, "cancelled", {"provider_status": "CANCELLED"})
    assert journal.snapshot(identity) == terminal
    assert terminal["billing"] == "unknown"
    events = terminal["events"]
    assert isinstance(events, list)
    assert events[3]["observation"] == {"partial_counts": {"1": 7}}
    with pytest.raises(ValueError, match="different provider handle"):
        journal.bind_job(identity, "other_job")


def test_observation_requires_original_bound_handle(tmp_path: Path) -> None:
    """Unsubmitted attempts cannot acquire result or cancellation evidence."""
    journal, identity = prepared(tmp_path)
    with pytest.raises(ValueError, match="bound provider handle"):
        journal.record(identity, "completed", {})
    with pytest.raises(ValueError, match="unsupported"):
        journal.record(identity, "fictional", {})
    with pytest.raises(ValueError):
        journal.bind_job(identity, "")


def test_unknown_version_and_payload_corruption_refuse(tmp_path: Path) -> None:
    """Actual stored bytes and SQLite version remain explicit admission boundaries."""
    journal, identity = prepared(tmp_path)
    with sqlite3.connect(journal.path) as db:
        db.execute("UPDATE attempts SET payload=? WHERE attempt_id=?", (b"changed", identity))
    with pytest.raises(ValueError, match="digest differs"):
        journal.snapshot(identity)
    with sqlite3.connect(journal.path) as db:
        db.execute("PRAGMA user_version=2")
    with pytest.raises(ValueError, match="unsupported"):
        ProviderJobJournal(tmp_path)


def test_noncanonical_attempt_id_refuses_before_preparation(tmp_path: Path) -> None:
    """An uppercase UUID alias cannot introduce a second textual attempt key."""
    journal = ProviderJobJournal(tmp_path)
    with pytest.raises(ValueError, match="canonical"):
        journal.prepare(
            str(uuid4()).upper(),
            target="ibm_exact",
            payload=b"request",
            shots=16,
            experiment="identity",
            circuits=1,
        )


def test_terminal_failure_preserves_late_raw_and_partial_receipts(tmp_path: Path) -> None:
    """A late native receipt survives an earlier failed status without losing history."""
    import json

    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler
    from qiskit_ibm_runtime import RuntimeEncoder

    journal, identity = prepared(tmp_path)
    journal.begin_submit(identity)
    journal.bind_job(identity, "original_job")
    journal.record(identity, "failed", {"provider_status": "ERROR"})
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.measure_all()
    result = StatevectorSampler(seed=47).run([circuit], shots=16).result()
    encoded = json.dumps(result, cls=RuntimeEncoder)
    journal.record(identity, "submitted", {"native_runtime_json": encoded})
    journal.record(identity, "submitted", {"partial_results": []})
    assert journal.snapshot(identity)["state"] == "failed"
    journal.record(identity, "completed", {"counts": result[0].data.meas.get_counts()})
    snapshot = journal.snapshot(identity)
    assert snapshot["state"] == "completed"
    assert snapshot["billing"] == "unknown"
    events = snapshot["events"]
    assert isinstance(events, list)
    assert any(event["observation"].get("native_runtime_json") == encoded for event in events)
