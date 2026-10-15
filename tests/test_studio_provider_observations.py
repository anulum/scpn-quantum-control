# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — read-only provider observation tests
"""Exercise Studio exports through the original native sampler and durable journal."""

import asyncio
import base64
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.hardware.async_runner import AsyncHardwareRunner
from scpn_quantum_control.hardware.provider_job_journal import ProviderJobJournal
from scpn_quantum_control.hardware.provider_job_lifecycle import (
    native_batch_payload,
    store_job_results,
)
from scpn_quantum_control.hardware.runner import HardwareRunner
from scpn_quantum_control.studio.provider_observations import (
    MAX_OBSERVATION_BYTES,
    export_provider_observation,
)


def test_native_completed_observation_retains_original_job_and_late_partial(
    tmp_path: Path,
) -> None:
    """Repeated exports and cached reads retain actual SDK counts and all original evidence.

    Parameters
    ----------
    tmp_path
        Owned native Aer results and SQLite journal.

    """
    runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "results"))
    runner.connect()
    facade = AsyncHardwareRunner(runner)
    journal = ProviderJobJournal(tmp_path / "journal")
    identity = str(uuid4())
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.measure_all()

    async def execute() -> None:
        """Sample once with the installed SDK, then use only public read boundaries."""
        handle = await facade.submit_one_async(
            [circuit], shots=16, journal=journal, attempt_id=identity
        )
        results = await facade.wait_for_job_async(handle)
        assert results[0].counts == {"1": 16}
        with pytest.raises(ValueError, match="incomplete"):
            store_job_results(journal, identity, [])
        before = journal.snapshot(identity)
        first = export_provider_observation(journal, identity)
        second = export_provider_observation(journal, identity)
        assert first == second
        body = json.loads(first)["body"]
        payload = base64.b64decode(body["payload_base64"], validate=True)
        assert payload == before["payload"]
        assert hashlib.sha256(payload).hexdigest() == body["payload_sha256"]
        assert body["attempt_id"] == identity
        assert body["provider_job_id"] == handle.job_id
        assert body["target"] == "aer_simulator"
        assert body["local_state"] == "completed"
        assert body["provider_state"] is None
        assert body["billing"] == "unknown"
        assert body["events"] == before["events"]
        assert body["events"][-1]["observation"] == {"partial_results": []}
        cold = facade.recover_job(ProviderJobJournal(tmp_path / "journal"), identity)
        assert await facade.wait_for_job_async(cold) == results
        assert journal.snapshot(identity) == before

    asyncio.run(execute())


def test_unknown_dispatch_export_does_not_invent_a_provider_state(tmp_path: Path) -> None:
    """An unbound dispatch remains unknown across read-only export.

    Parameters
    ----------
    tmp_path
        Owned native request journal; no dispatch is performed.

    """
    journal = ProviderJobJournal(tmp_path)
    identity = str(uuid4())
    circuit = QuantumCircuit(1)
    circuit.measure_all()
    journal.prepare(
        identity,
        target="aer_simulator",
        payload=native_batch_payload([circuit]),
        shots=16,
        experiment="uncertain-dispatch",
        circuits=1,
    )
    journal.begin_submit(identity)
    before = journal.snapshot(identity)
    body = json.loads(export_provider_observation(journal, identity))["body"]
    assert body["local_state"] == "submission_unknown"
    assert body["provider_job_id"] is None
    assert body["provider_state"] is None
    assert body["no_submit"] is True
    assert journal.snapshot(identity) == before


def test_observation_preserves_provider_status_and_cancellation_intent(tmp_path: Path) -> None:
    """Status transport evidence stays separate from local cancellation intent.

    Parameters
    ----------
    tmp_path
        Owned journal containing original status observations, without sample claims.

    """
    journal = ProviderJobJournal(tmp_path)
    identity = str(uuid4())
    circuit = QuantumCircuit(1)
    circuit.measure_all()
    journal.prepare(
        identity,
        target="aer_simulator",
        payload=native_batch_payload([circuit]),
        shots=16,
        experiment="status-transport",
        circuits=1,
    )
    journal.begin_submit(identity)
    journal.bind_job(identity, "observed-existing-job")
    journal.record(identity, "submitted", {"provider_status": "RUNNING"})
    journal.record(identity, "cancellation_requested", {})
    before = journal.snapshot(identity)
    body = json.loads(export_provider_observation(journal, identity))["body"]
    assert body["local_state"] == "cancellation_requested"
    assert body["provider_state"] == "RUNNING"
    assert body["events"] == before["events"]
    assert body["billing"] == "unknown"
    assert journal.snapshot(identity) == before


@pytest.mark.parametrize("fault", ["status_type", "status_empty", "export_budget"])
def test_export_refusal_preserves_original_journal(tmp_path: Path, fault: str) -> None:
    """Malformed status or oversized transport evidence cannot mutate original custody.

    Parameters
    ----------
    tmp_path
        Owned SQLite journal containing an explicitly negative transport observation.
    fault
        Non-text status, empty status, or an over-budget raw transport record.

    """
    journal = ProviderJobJournal(tmp_path)
    identity = str(uuid4())
    circuit = QuantumCircuit(1)
    circuit.measure_all()
    journal.prepare(
        identity,
        target="aer_simulator",
        payload=native_batch_payload([circuit]),
        shots=16,
        experiment="negative-transport",
        circuits=1,
    )
    journal.begin_submit(identity)
    journal.bind_job(identity, "negative-transport-job")
    observation: dict[str, object]
    if fault == "export_budget":
        observation = {"native_runtime_json": "x" * MAX_OBSERVATION_BYTES}
    else:
        observation = {"provider_status": 1 if fault == "status_type" else ""}
    journal.record(identity, "submitted", observation)
    before = journal.snapshot(identity)
    with pytest.raises(ValueError, match="budget" if fault == "export_budget" else "status"):
        export_provider_observation(journal, identity)
    assert journal.snapshot(identity) == before


def test_unknown_attempt_export_does_not_prepare_or_dispatch(tmp_path: Path) -> None:
    """Retrieving an absent identity cannot become a new submission.

    Parameters
    ----------
    tmp_path
        Owned empty durable journal.

    """
    journal = ProviderJobJournal(tmp_path)
    identity = str(uuid4())
    with pytest.raises(KeyError, match="unknown provider attempt"):
        export_provider_observation(journal, identity)
    with pytest.raises(KeyError):
        journal.snapshot(identity)


def test_original_native_capture_replays_exact_cross_language_wire(tmp_path: Path) -> None:
    """Replay original custody through public SQLite APIs and compare every native wire byte.

    Parameters
    ----------
    tmp_path
        Owned journal replaying the exact captured requests and events. This
        checks metadata conformance; separate tests perform original SDK sampling.

    """
    fixture = Path(__file__).resolve().parents[1] / "data/studio/provider_observation_cases.json"
    cases = json.loads(fixture.read_text())["cases"]
    journal = ProviderJobJournal(tmp_path)
    for name in ("submission_unknown", "submitted"):
        body = json.loads(cases[name])["body"]
        identity = body["attempt_id"]
        journal.prepare(
            identity,
            target=body["target"],
            payload=base64.b64decode(body["payload_base64"], validate=True),
            shots=body["shots"],
            experiment=body["experiment"],
            circuits=body["circuits"],
        )
        journal.begin_submit(identity)
        if body["provider_job_id"] is not None:
            journal.bind_job(identity, body["provider_job_id"])
        assert export_provider_observation(journal, identity) == cases[name]
    retained = len(body["events"])
    for name in ("cancellation_requested", "completed", "late_partial"):
        body = json.loads(cases[name])["body"]
        for event in body["events"][retained:]:
            journal.record(identity, event["state"], event["observation"])
        retained = len(body["events"])
        assert export_provider_observation(journal, identity) == cases[name]
