# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native provider lifecycle tests
"""Qualify native request bytes and observable lifecycle/result custody."""

import io
from pathlib import Path
from uuid import uuid4

import pytest
from qiskit import QuantumCircuit, qpy

from scpn_quantum_control.hardware.provider_job_journal import (
    ProviderJobJournal,
    SubmissionUnknownError,
)
from scpn_quantum_control.hardware.provider_job_lifecycle import (
    cancel_durable_job,
    native_batch_payload,
    native_job_result,
    observe_durable_job,
    restore_job_results,
    retrieve_durable_job,
    store_job_results,
)
from scpn_quantum_control.hardware.runner import HardwareRunner, JobResult


def bound_attempt(root: Path) -> tuple[ProviderJobJournal, str]:
    """Prepare and bind one original request in a real SQLite journal."""
    journal = ProviderJobJournal(root)
    identity = str(uuid4())
    circuit = QuantumCircuit(1)
    circuit.measure_all()
    journal.prepare(
        identity,
        target="ibm_original",
        payload=native_batch_payload([circuit]),
        shots=16,
        experiment="native",
        circuits=1,
    )
    journal.begin_submit(identity)
    journal.bind_job(identity, "original_job")
    return journal, identity


class ObservedJob:
    """Inject only provider status/cancellation transport for negative contracts."""

    def __init__(self, status: str, identity: str = "original_job") -> None:
        self.observed = status
        self.identity = identity
        self.cancelled = 0

    def job_id(self) -> str:
        """Return the requested transport identity."""
        return self.identity

    def status(self) -> str:
        """Return an explicit transport observation, without invented results."""
        return self.observed

    def cancel(self) -> bool:
        """Acknowledge intent only, preserving the separate native observation."""
        self.cancelled += 1
        return True


def test_native_qpy_payload_preserves_source_and_measurement() -> None:
    """The real native decoder recovers exact original circuit instructions."""
    circuit = QuantumCircuit(2)
    circuit.x(1)
    circuit.measure_all()
    payload = native_batch_payload([circuit])
    restored = qpy.load(io.BytesIO(payload))
    assert restored == [circuit]
    assert native_batch_payload(restored) == payload


@pytest.mark.parametrize(
    "status,state",
    [
        ("RUNNING", "submitted"),
        ("DONE", "submitted"),
        ("CANCELLED", "cancelled"),
        ("ERROR", "failed"),
    ],
)
def test_native_status_remains_distinct_from_counts_and_billing(
    tmp_path: Path, status: str, state: str
) -> None:
    """A status cannot manufacture decoded completion evidence or a zero debit."""
    journal, identity = bound_attempt(tmp_path)
    snapshot = observe_durable_job(ObservedJob(status), journal, identity)
    assert snapshot["state"] == state
    assert snapshot["billing"] == "unknown"


def test_foreign_observation_and_cancel_leave_original_attempt_unchanged(tmp_path: Path) -> None:
    """A changed handle cannot cause cancellation of another provider job."""
    journal, identity = bound_attempt(tmp_path)
    foreign = ObservedJob("RUNNING", "foreign_job")
    before = journal.snapshot(identity)
    with pytest.raises(ValueError, match="handle differs"):
        observe_durable_job(foreign, journal, identity)
    with pytest.raises(ValueError, match="handle differs"):
        cancel_durable_job(foreign, journal, identity)
    assert foreign.cancelled == 0
    assert journal.snapshot(identity) == before


@pytest.mark.parametrize("counts", [None, {"1": 7}])
def test_partial_sample_totals_preserve_evidence_without_completion(
    tmp_path: Path, counts: dict[str, int] | None
) -> None:
    """An incomplete native sample receipt remains visible and nonterminal."""
    journal, identity = bound_attempt(tmp_path)
    result = JobResult(
        job_id="original_job",
        backend_name="ibm_original",
        experiment_name="native",
        counts=counts,
        wall_time_s=0.1,
        timestamp="2026-10-08T00:00:00Z",
        metadata={},
    )
    with pytest.raises(ValueError, match="sample total differs"):
        store_job_results(journal, identity, [result])
    snapshot = journal.snapshot(identity)
    assert snapshot["state"] == "submitted"
    events = snapshot["events"]
    assert isinstance(events, list)
    assert events[-1]["observation"]["partial_results"][0]["counts"] == counts


def test_empty_batch_retains_partial_receipt_and_refuses_completion(tmp_path: Path) -> None:
    """An absent publication cannot masquerade as a completed one-circuit job."""
    journal, identity = bound_attempt(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        store_job_results(journal, identity, [])
    assert journal.snapshot(identity)["state"] == "submitted"


def test_terminal_result_requires_retained_completion_evidence(tmp_path: Path) -> None:
    """A caller cannot restore uncommitted or merely status-complete results."""
    journal, identity = bound_attempt(tmp_path)
    with pytest.raises(ValueError, match="no retained results"):
        restore_job_results(journal.snapshot(identity))


def test_public_retrieval_refuses_unknown_handle_and_changed_route(tmp_path: Path) -> None:
    """Direct lifecycle callers get the same no-submit identity refusal as the facade."""
    journal, identity = bound_attempt(tmp_path / "journal")
    runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "results"))
    with pytest.raises(ValueError, match="handle differs"):
        retrieve_durable_job(runner, journal, identity, "foreign_job")
    with pytest.raises(ValueError, match="runner target differs"):
        retrieve_durable_job(runner, journal, identity)
    other = str(uuid4())
    journal.prepare(
        other,
        target="not_connected",
        payload=b"request",
        shots=16,
        experiment="unknown",
        circuits=1,
    )
    journal.begin_submit(other)
    with pytest.raises(SubmissionUnknownError):
        retrieve_durable_job(runner, journal, other)


def test_foreign_decoded_result_refuses_without_changing_custody(tmp_path: Path) -> None:
    """An alien publication cannot overwrite the bound job's retained state."""
    journal, identity = bound_attempt(tmp_path)
    before = journal.snapshot(identity)
    result = JobResult(
        job_id="foreign_job",
        backend_name="ibm_original",
        experiment_name="native",
        counts={"1": 16},
        wall_time_s=0.1,
        timestamp="2026-10-08T00:00:00Z",
        metadata={},
    )
    with pytest.raises(ValueError, match="identity differs"):
        store_job_results(journal, identity, [result])
    assert journal.snapshot(identity) == before


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -1.0, True, "one"])
def test_native_wait_rejects_malformed_bounds_before_result(timeout: object) -> None:
    """Malformed wait bounds cannot turn a local native job into unbounded polling."""
    with pytest.raises(ValueError, match="finite nonnegative"):
        native_job_result(None, timeout)


def test_local_native_status_lag_and_deadline_preserve_real_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale status and timeout retain the same actual original SDK job and samples."""
    from qiskit.primitives import StatevectorSampler
    from qiskit.providers import JobTimeoutError

    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.measure_all()
    job = StatevectorSampler(seed=47).run([circuit], shots=16)
    original = job.result()
    identity = job.job_id()
    statuses = iter([False, True])
    monkeypatch.setattr(job, "in_final_state", lambda: next(statuses))
    observed = native_job_result(job, 1.0)
    assert observed[0].data.meas.get_counts() == {"1": 16}
    assert job.job_id() == identity
    monkeypatch.setattr(job, "in_final_state", lambda: False)
    with pytest.raises(JobTimeoutError):
        native_job_result(job, 0.0)
    assert job.job_id() == identity
    assert original[0].data.meas.get_counts() == {"1": 16}
