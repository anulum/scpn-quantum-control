# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Qiskit Bridge Contract Edge Tests
"""Exercise fail-closed Qiskit bridge record and evidence-chain contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter as QiskitParameter

from scpn_quantum_control.phase import (
    QiskitCalibrationStatevectorComparisonArtifact,
    QiskitParameterShiftGradientResult,
    QiskitParameterShiftRecord,
    QiskitProviderGradientWorkflowArtifact,
    QiskitRawCountReplayArtifact,
    QiskitRuntimePrimitiveExecutionArtifact,
    QiskitRuntimeQPUExecutionArtifact,
    QiskitRuntimeQPUProviderEvidenceBundle,
)

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64
DIGEST_D = "sha256:" + "d" * 64


def _shift_record() -> QiskitParameterShiftRecord:
    circuit = QuantumCircuit(1)
    return QiskitParameterShiftRecord(
        parameter_index=0,
        shift_index=0,
        shift=np.pi / 2.0,
        coefficient=0.5,
        parameter_name="theta",
        plus_values=np.array([1.0]),
        minus_values=np.array([-1.0]),
        plus_circuit=circuit,
        minus_circuit=circuit.copy(),
    )


def _runtime_primitive(*, shots: int | None = 400) -> QiskitRuntimePrimitiveExecutionArtifact:
    return QiskitRuntimePrimitiveExecutionArtifact(
        artifact_id="runtime-estimator",
        provider_name="ibm_quantum",
        primitive_name="EstimatorV2",
        backend_name="ibm_simulator",
        job_id="job-1",
        session_id="session-1",
        circuit_fingerprint="circuit:v1",
        observable_fingerprint="observable:z",
        parameter_digest=DIGEST_A,
        result_digest=DIGEST_B,
        metadata_digest=DIGEST_C,
        shots=shots,
        raw_result_replay_artifact_id="replay-1",
    )


def _runtime_qpu(**changes: object) -> QiskitRuntimeQPUExecutionArtifact:
    fields: dict[str, object] = {
        "artifact_id": "runtime-qpu",
        "provider_name": "ibm_quantum",
        "primitive_name": "EstimatorV2",
        "backend_name": "ibm_brisbane",
        "job_id": "job-qpu",
        "session_id": "session-qpu",
        "circuit_fingerprint": "circuit:v1",
        "observable_fingerprint": "observable:z",
        "parameter_digest": DIGEST_A,
        "result_digest": DIGEST_B,
        "metadata_digest": DIGEST_C,
        "transpiled_circuit_digest": DIGEST_D,
        "live_execution_ticket": "ticket-1",
        "backend_allowlist_id": "allowlist-1",
        "shot_budget_id": "budget-1",
        "runtime_session_mode": "live_qpu_session",
        "shots": 4096,
        "hardware_execution": True,
    }
    fields.update(changes)
    return QiskitRuntimeQPUExecutionArtifact(**cast(Any, fields))


def _raw_counts(**changes: object) -> QiskitRawCountReplayArtifact:
    fields: dict[str, object] = {
        "artifact_id": "raw-counts",
        "provider_name": "ibm_quantum",
        "backend_name": "ibm_brisbane",
        "job_id": "job-qpu",
        "circuit_fingerprint": "circuit:v1",
        "counts_digest": DIGEST_A,
        "replay_digest": DIGEST_B,
        "shots": 4096,
        "measured_qubits": 1,
        "expectation_value": 0.5,
        "standard_error": 0.01,
        "hardware_execution": True,
        "live_ticket_id": "ticket-1",
    }
    fields.update(changes)
    return QiskitRawCountReplayArtifact(**cast(Any, fields))


def _calibration(**changes: object) -> QiskitCalibrationStatevectorComparisonArtifact:
    fields: dict[str, object] = {
        "artifact_id": "calibration-comparison",
        "provider_name": "ibm_quantum",
        "backend_name": "ibm_brisbane",
        "calibration_snapshot_id": "calibration-1",
        "statevector_reference_artifact_id": "statevector-1",
        "circuit_fingerprint": "circuit:v1",
        "calibration_digest": DIGEST_C,
        "comparison_digest": DIGEST_D,
        "max_abs_error": 0.01,
        "tolerance": 0.05,
        "hardware_execution": True,
        "live_ticket_id": "ticket-1",
    }
    fields.update(changes)
    return QiskitCalibrationStatevectorComparisonArtifact(**cast(Any, fields))


def _gradient_metadata() -> dict[str, object]:
    return {
        "method_schema": "parameter_shift_shift_rule",
        "method_artifact_id": "gradient-method-1",
        "workflow_version": "runtime-gradient-v1",
        "shift_rule_id": "two-term-shift",
        "shift_count": 2,
    }


def _gradient_workflow(**changes: object) -> QiskitProviderGradientWorkflowArtifact:
    fields: dict[str, object] = {
        "artifact_id": "provider-gradient",
        "provider_name": "ibm_quantum",
        "backend_name": "ibm_brisbane",
        "job_id": "gradient-job",
        "primitive_name": "EstimatorV2",
        "gradient_method": "parameter_shift",
        "circuit_fingerprint": "circuit:v1",
        "observable_fingerprint": "observable:z",
        "parameter_digest": DIGEST_A,
        "gradient_digest": DIGEST_B,
        "metadata_digest": DIGEST_C,
        "shots": 4096,
        "parameter_count": 1,
        "gradient_dimension": 1,
        "hardware_execution": True,
        "live_ticket_id": "ticket-1",
        "method_metadata": _gradient_metadata(),
    }
    fields.update(changes)
    return QiskitProviderGradientWorkflowArtifact(**cast(Any, fields))


def _bundle(**changes: object) -> QiskitRuntimeQPUProviderEvidenceBundle:
    fields: dict[str, object] = {
        "artifact_id": "provider-bundle",
        "runtime_qpu_execution_artifact": _runtime_qpu(),
        "raw_count_replay_artifact": _raw_counts(),
        "calibration_comparison_artifact": _calibration(),
        "captured_at_utc": "2026-06-19T18:00:00Z",
        "valid_until_utc": "2026-07-19T18:00:00Z",
    }
    fields.update(changes)
    return QiskitRuntimeQPUProviderEvidenceBundle(**cast(Any, fields))


def test_qiskit_parameter_shift_records_reject_remaining_invalid_fields() -> None:
    """Reject invalid indices, names, shapes, bound state, values and results."""
    record = _shift_record()
    parameter = QiskitParameter("theta")
    unbound = QuantumCircuit(1)
    unbound.ry(parameter, 0)
    record_cases: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: replace(record, parameter_index=-1), "parameter_index"),
        (lambda: replace(record, shift_index=-1), "shift_index"),
        (lambda: replace(record, parameter_name=""), "parameter_name"),
        (lambda: replace(record, minus_values=np.array([1.0, 2.0])), "matching shapes"),
        (lambda: replace(record, plus_circuit=unbound), "fully bound"),
    )
    gradient = QiskitParameterShiftGradientResult(
        value=0.5,
        gradient=np.array([0.1]),
        records=(record,),
        method="parameter_shift",
        evaluations=3,
        claim_boundary="local_statevector_only",
    )
    result_cases: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: replace(gradient, evaluations=0), "evaluations"),
        (lambda: replace(gradient, claim_boundary=""), "claim_boundary"),
        (lambda: replace(gradient, gradient=np.array([[0.1]])), "one-dimensional"),
        (lambda: replace(gradient, gradient=np.array([np.nan])), "finite values"),
        (lambda: replace(gradient, value=cast(Any, [0.5])), "real numeric scalar"),
        (lambda: replace(gradient, value=np.nan), "must be finite"),
    )

    for factory, match in (*record_cases, *result_cases):
        with pytest.raises(ValueError, match=match):
            factory()


def test_qiskit_runtime_primitive_rejects_remaining_optional_metadata_edges() -> None:
    """Reject empty primitive metadata while accepting an explicit analytic shot policy."""
    artifact = _runtime_primitive()
    cases: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: replace(artifact, artifact_id=""), "artifact_id"),
        (lambda: replace(artifact, session_id=""), "session_id"),
        (
            lambda: replace(artifact, raw_result_replay_artifact_id=""),
            "raw_result_replay_artifact_id",
        ),
    )
    for factory, match in cases:
        with pytest.raises(ValueError, match=match):
            factory()

    assert _runtime_primitive(shots=None).shots is None


def test_qiskit_raw_count_and_calibration_artifacts_reject_remaining_edges() -> None:
    """Reject invalid replay and calibration scalar or hardware evidence."""
    cases: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: _raw_counts(artifact_id=""), "artifact_id"),
        (lambda: _raw_counts(measured_qubits=0), "measured_qubits"),
        (lambda: _raw_counts(expectation_value=2.0), "expectation_value"),
        (lambda: _raw_counts(standard_error=-0.1), "standard_error"),
        (lambda: _calibration(artifact_id=""), "artifact_id"),
        (lambda: _calibration(max_abs_error=-0.1), "max_abs_error"),
        (lambda: _calibration(hardware_execution=False), "hardware execution"),
    )
    for factory, match in cases:
        with pytest.raises(ValueError, match=match):
            factory()


def test_qiskit_provider_gradient_artifact_rejects_remaining_contract_edges() -> None:
    """Reject missing observables, dimensions, hardware claims and invalid counts."""
    cases: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: _gradient_workflow(observable_fingerprint=None), "observable_fingerprint"),
        (lambda: _gradient_workflow(gradient_dimension=2), "gradient_dimension"),
        (lambda: _gradient_workflow(hardware_execution=False), "hardware execution"),
        (lambda: _gradient_workflow(parameter_count=0), "positive integer"),
    )
    for factory, match in cases:
        with pytest.raises(ValueError, match=match):
            factory()


def test_qiskit_metadata_and_evidence_chain_reject_remaining_defensive_edges() -> None:
    """Reject control text, timestamps, contradictory mappings, modes and shot drift."""

    class ContradictoryMetadata(Mapping[str, object]):
        def __init__(self) -> None:
            self._values = _gradient_metadata()

        def __getitem__(self, key: str) -> object:
            if key == "shift_rule_id":
                raise KeyError(key)
            return self._values[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self._values)

        def __len__(self) -> int:
            return len(self._values)

    cases: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: _runtime_qpu(artifact_id="bad\nid"), "control characters"),
        (lambda: _runtime_qpu(runtime_session_mode="live_simulator"), "simulator or replay"),
        (lambda: _bundle(captured_at_utc="not-a-time"), "ISO-8601"),
        (lambda: _bundle(captured_at_utc="2026-06-19T18:00:00"), "UTC offset"),
        (
            lambda: _gradient_workflow(method_metadata=ContradictoryMetadata()),
            "method_metadata.shift_rule_id is required",
        ),
        (
            lambda: _bundle(raw_count_replay_artifact=_raw_counts(shots=2048)),
            "shots must match",
        ),
    )
    for factory, match in cases:
        with pytest.raises(ValueError, match=match):
            factory()
