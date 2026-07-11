# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Qiskit Runtime Evidence Tests
"""Tests for Qiskit Runtime evidence and maturity orchestration."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.qiskit_bridge as qiskit_bridge
import scpn_quantum_control.phase.qiskit_runtime as qiskit_runtime
from scpn_quantum_control.phase import (
    QiskitCalibrationStatevectorComparisonArtifact,
    QiskitMaturityAuditResult,
    QiskitParameterShiftGradientResult,
    QiskitParameterShiftRecord,
    QiskitProviderGradientWorkflowArtifact,
    QiskitRawCountReplayArtifact,
    QiskitRuntimePrimitiveExecutionArtifact,
    QiskitRuntimeQPUExecutionArtifact,
    QiskitRuntimeQPUProviderEvidenceBundle,
    build_qiskit_provider_gradient_workflow_artifact,
    build_qiskit_runtime_qpu_execution_artifact,
    build_qiskit_runtime_qpu_provider_evidence_bundle,
    run_qiskit_maturity_audit,
)
from scpn_quantum_control.phase.provider_gradient import ProviderGradientExecutionResult
from scpn_quantum_control.phase.provider_hardware_gradient_audit import (
    ProviderHardwareGradientPreparationAuditResult,
)


def _single_rotation_problem() -> tuple[QuantumCircuit, tuple[Parameter, ...], SparsePauliOp]:
    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    observable = SparsePauliOp.from_list([("Z", 1.0)])
    return circuit, (theta,), observable


def test_qiskit_maturity_audit_records_local_evidence_and_provider_gaps() -> None:
    circuit, parameters, observable = _single_rotation_problem()

    result = run_qiskit_maturity_audit(
        circuit,
        observable,
        parameters,
        np.array([0.4], dtype=float),
        shots=400,
    )

    assert isinstance(result, QiskitMaturityAuditResult)
    assert result.local_gradient_ready
    assert not result.ready_for_provider_exceedance
    shifted_records = cast(
        tuple[QiskitParameterShiftRecord, ...], result.evidence["shifted_circuit_records"]
    )
    statevector_reference = cast(
        QiskitParameterShiftGradientResult,
        result.evidence["statevector_reference"],
    )
    finite_shot_surrogate = cast(
        ProviderGradientExecutionResult,
        result.evidence["finite_shot_surrogate"],
    )
    provider_preparation_audit = cast(
        ProviderHardwareGradientPreparationAuditResult,
        result.evidence["provider_preparation_audit"],
    )
    assert shifted_records[0].parameter_name == "theta"
    assert statevector_reference.method == "qiskit_statevector_parameter_shift"
    assert finite_shot_surrogate.backend == "finite_shot_simulator"
    assert provider_preparation_audit.passed
    assert result.required_capabilities["shifted_circuit_generation"] == "passed"
    assert result.required_capabilities["statevector_reference_comparison"] == "passed"
    assert result.required_capabilities["provider_hardware_preparation_policy"] == "passed"
    assert result.required_capabilities["raw_count_capture_replay_harness"] == "blocked"
    assert result.local_reference_metadata["shots"] == 400
    max_abs_error = cast(
        float, result.local_reference_metadata["statevector_finite_shot_max_abs_error"]
    )
    assert max_abs_error <= 1e-12
    assert "live_qpu_execution_ticket" in result.open_gaps
    assert "raw_count_capture_replay_harness" in result.open_gaps
    payload = cast(dict[str, Any], result.to_dict())
    assert payload["claim_boundary"] == "bounded_qiskit_provider_maturity_audit"
    local_reference_metadata = cast(dict[str, object], payload["local_reference_metadata"])
    assert local_reference_metadata["parameter_count"] == 1


def _qiskit_runtime_primitive_artifact() -> QiskitRuntimePrimitiveExecutionArtifact:
    return QiskitRuntimePrimitiveExecutionArtifact(
        artifact_id="qiskit-runtime-estimator-20260616",
        provider_name="ibm_quantum",
        primitive_name="EstimatorV2",
        backend_name="ibm_simulator",
        job_id="runtime-job-20260616",
        session_id="runtime-session-20260616",
        circuit_fingerprint="qiskit:ry(theta):z:v1",
        observable_fingerprint="SparsePauliOp:Z:v1",
        parameter_digest="sha256:" + "1" * 64,
        result_digest="sha256:" + "2" * 64,
        metadata_digest="sha256:" + "3" * 64,
        shots=400,
        hardware_execution=False,
        raw_result_replay_artifact_id="qiskit-runtime-replay-20260616",
    )


def _qiskit_runtime_qpu_execution_artifact(
    *,
    primitive_name: str = "EstimatorV2",
    runtime_session_mode: str = "live_qpu_session",
    observable_fingerprint: str | None = "SparsePauliOp:Z:v1",
    shots: int = 4096,
) -> QiskitRuntimeQPUExecutionArtifact:
    return QiskitRuntimeQPUExecutionArtifact(
        artifact_id="qiskit-runtime-qpu-20260619",
        provider_name="ibm_quantum",
        primitive_name=primitive_name,
        backend_name="ibm_brisbane",
        job_id="runtime-qpu-job-20260619",
        session_id="runtime-qpu-session-20260619",
        circuit_fingerprint="qiskit:ry(theta):z:v1",
        observable_fingerprint=observable_fingerprint,
        parameter_digest="sha256:" + "8" * 64,
        result_digest="sha256:" + "9" * 64,
        metadata_digest="sha256:" + "a" * 64,
        transpiled_circuit_digest="sha256:" + "b" * 64,
        live_execution_ticket="live-ticket-20260619",
        backend_allowlist_id="backend-allowlist-20260619",
        shot_budget_id="shot-budget-20260619",
        runtime_session_mode=runtime_session_mode,
        shots=shots,
        hardware_execution=True,
    )


def test_qiskit_maturity_audit_accepts_runtime_primitive_artifact_without_promotion() -> None:
    circuit, parameters, observable = _single_rotation_problem()
    artifact = _qiskit_runtime_primitive_artifact()

    result = run_qiskit_maturity_audit(
        circuit,
        observable,
        parameters,
        np.array([0.4], dtype=float),
        shots=400,
        runtime_primitive_artifact=artifact,
    )

    assert result.required_capabilities["runtime_primitive_execution_evidence"] == "passed"
    assert result.evidence["runtime_primitive_artifact"] is artifact
    assert result.local_reference_metadata["runtime_primitive_artifact_id"] == artifact.artifact_id
    assert not result.ready_for_provider_exceedance
    assert "runtime_primitive_execution_evidence" not in result.open_gaps
    assert "live_qpu_execution_ticket" in result.open_gaps
    assert "raw_count_capture_replay_harness" in result.open_gaps
    payload = cast(dict[str, Any], result.to_dict())
    evidence = cast(dict[str, object], payload["evidence"])
    primitive_payload = cast(dict[str, object], evidence["runtime_primitive_artifact"])
    assert primitive_payload["artifact_id"] == artifact.artifact_id
    assert primitive_payload["hardware_execution"] is False


def test_qiskit_maturity_audit_accepts_runtime_qpu_execution_without_promotion() -> None:
    circuit, parameters, observable = _single_rotation_problem()
    artifact = _qiskit_runtime_qpu_execution_artifact()

    result = run_qiskit_maturity_audit(
        circuit,
        observable,
        parameters,
        np.array([0.4], dtype=float),
        shots=400,
        runtime_qpu_execution_artifact=artifact,
    )

    assert result.required_capabilities["live_qpu_execution_ticket"] == "passed"
    assert result.evidence["runtime_qpu_execution_artifact"] is artifact
    assert result.local_reference_metadata["runtime_qpu_execution_artifact_id"] == (
        artifact.artifact_id
    )
    assert result.local_reference_metadata["runtime_qpu_primitive_name"] == "EstimatorV2"
    assert not result.ready_for_provider_exceedance
    assert "live_qpu_execution_ticket" not in result.open_gaps
    assert "raw_count_capture_replay_harness" in result.open_gaps
    assert "promotion_grade_isolated_benchmarks" in result.open_gaps
    payload = cast(dict[str, Any], result.to_dict())
    evidence = cast(dict[str, object], payload["evidence"])
    qpu_payload = cast(dict[str, object], evidence["runtime_qpu_execution_artifact"])
    assert qpu_payload["hardware_execution"] is True
    assert qpu_payload["primitive_name"] == "EstimatorV2"


def test_qiskit_runtime_qpu_execution_artifact_accepts_sampler_without_observable() -> None:
    artifact = _qiskit_runtime_qpu_execution_artifact(
        primitive_name="sampler",
        observable_fingerprint=None,
    )

    assert artifact.primitive_name == "SamplerV2"
    assert artifact.observable_fingerprint is None
    assert artifact.to_dict()["primitive_name"] == "SamplerV2"


def test_build_qiskit_runtime_qpu_execution_artifact_from_estimator_capture() -> None:
    artifact = build_qiskit_runtime_qpu_execution_artifact(
        artifact_id=" qiskit-runtime-qpu-builder-20260619 ",
        provider_name="ibm_quantum",
        primitive_name="estimator",
        backend_name="ibm_brisbane",
        job_id="runtime-qpu-job-20260619",
        session_id="runtime-qpu-session-20260619",
        circuit_fingerprint="qiskit:ry(theta):z:v1",
        observable_fingerprint="SparsePauliOp:Z:v1",
        parameter_digest="sha256:" + "8" * 64,
        result_digest="sha256:" + "9" * 64,
        metadata_digest="sha256:" + "a" * 64,
        transpiled_circuit_digest="sha256:" + "b" * 64,
        live_execution_ticket="live-ticket-20260619",
        backend_allowlist_id="backend-allowlist-20260619",
        shot_budget_id="shot-budget-20260619",
        runtime_session_mode="live_qpu_session",
        shots=4096,
    )

    assert artifact.artifact_id == "qiskit-runtime-qpu-builder-20260619"
    assert artifact.primitive_name == "EstimatorV2"
    assert artifact.hardware_execution is True
    assert artifact.observable_fingerprint == "SparsePauliOp:Z:v1"


def test_build_qiskit_runtime_qpu_execution_artifact_from_sampler_capture() -> None:
    artifact = build_qiskit_runtime_qpu_execution_artifact(
        artifact_id="qiskit-runtime-qpu-sampler-builder-20260619",
        provider_name="ibm_quantum",
        primitive_name="SamplerV2",
        backend_name="ibm_brisbane",
        job_id="runtime-qpu-sampler-job-20260619",
        session_id=None,
        circuit_fingerprint="qiskit:bell:meas:v1",
        observable_fingerprint=None,
        parameter_digest="sha256:" + "c" * 64,
        result_digest="sha256:" + "d" * 64,
        metadata_digest="sha256:" + "e" * 64,
        transpiled_circuit_digest="sha256:" + "f" * 64,
        live_execution_ticket="live-ticket-20260619",
        backend_allowlist_id="backend-allowlist-20260619",
        shot_budget_id="shot-budget-20260619",
        runtime_session_mode="backend_live_qpu_mode",
        shots=1024,
    )

    assert artifact.primitive_name == "SamplerV2"
    assert artifact.observable_fingerprint is None
    assert artifact.shots == 1024


def test_build_qiskit_runtime_qpu_execution_artifact_rejects_sampler_observable() -> None:
    with pytest.raises(ValueError, match="observable_fingerprint"):
        build_qiskit_runtime_qpu_execution_artifact(
            artifact_id="qiskit-runtime-qpu-sampler-builder-20260619",
            provider_name="ibm_quantum",
            primitive_name="SamplerV2",
            backend_name="ibm_brisbane",
            job_id="runtime-qpu-sampler-job-20260619",
            session_id=None,
            circuit_fingerprint="qiskit:bell:meas:v1",
            observable_fingerprint="SparsePauliOp:Z:v1",
            parameter_digest="sha256:" + "c" * 64,
            result_digest="sha256:" + "d" * 64,
            metadata_digest="sha256:" + "e" * 64,
            transpiled_circuit_digest="sha256:" + "f" * 64,
            live_execution_ticket="live-ticket-20260619",
            backend_allowlist_id="backend-allowlist-20260619",
            shot_budget_id="shot-budget-20260619",
            runtime_session_mode="backend_live_qpu_mode",
            shots=1024,
        )


def test_build_qiskit_runtime_qpu_execution_artifact_rejects_simulator_capture() -> None:
    with pytest.raises(ValueError, match="live QPU execution"):
        build_qiskit_runtime_qpu_execution_artifact(
            artifact_id="qiskit-runtime-qpu-builder-20260619",
            provider_name="ibm_quantum",
            primitive_name="EstimatorV2",
            backend_name="ibm_simulator",
            job_id="runtime-qpu-job-20260619",
            session_id="runtime-qpu-session-20260619",
            circuit_fingerprint="qiskit:ry(theta):z:v1",
            observable_fingerprint="SparsePauliOp:Z:v1",
            parameter_digest="sha256:" + "8" * 64,
            result_digest="sha256:" + "9" * 64,
            metadata_digest="sha256:" + "a" * 64,
            transpiled_circuit_digest="sha256:" + "b" * 64,
            live_execution_ticket="live-ticket-20260619",
            backend_allowlist_id="backend-allowlist-20260619",
            shot_budget_id="shot-budget-20260619",
            runtime_session_mode="simulator",
            shots=4096,
        )


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: _qiskit_runtime_qpu_execution_artifact(primitive_name="EstimatorV1"),
            "primitive_name",
        ),
        (
            lambda: _qiskit_runtime_qpu_execution_artifact(observable_fingerprint=None),
            "observable_fingerprint",
        ),
        (
            lambda: _qiskit_runtime_qpu_execution_artifact(
                primitive_name="SamplerV2",
                observable_fingerprint="SparsePauliOp:Z:v1",
            ),
            "observable_fingerprint",
        ),
        (
            lambda: _qiskit_runtime_qpu_execution_artifact(
                runtime_session_mode="offline_replay",
            ),
            "live QPU execution",
        ),
        (
            lambda: _qiskit_runtime_qpu_execution_artifact(shots=cast(Any, True)),
            "shots",
        ),
        (
            lambda: QiskitRuntimeQPUExecutionArtifact(
                artifact_id="qiskit-runtime-qpu-20260619",
                provider_name="ibm_quantum",
                primitive_name="EstimatorV2",
                backend_name="ibm_brisbane",
                job_id="runtime-qpu-job-20260619",
                session_id="runtime-qpu-session-20260619",
                circuit_fingerprint="qiskit:ry(theta):z:v1",
                observable_fingerprint="SparsePauliOp:Z:v1",
                parameter_digest="sha256:" + "8" * 63,
                result_digest="sha256:" + "9" * 64,
                metadata_digest="sha256:" + "a" * 64,
                transpiled_circuit_digest="sha256:" + "b" * 64,
                live_execution_ticket="live-ticket-20260619",
                backend_allowlist_id="backend-allowlist-20260619",
                shot_budget_id="shot-budget-20260619",
                runtime_session_mode="live_qpu_session",
                shots=4096,
                hardware_execution=True,
            ),
            "parameter_digest",
        ),
        (
            lambda: QiskitRuntimeQPUExecutionArtifact(
                artifact_id="qiskit-runtime-qpu-20260619",
                provider_name="ibm_quantum",
                primitive_name="EstimatorV2",
                backend_name="ibm_brisbane",
                job_id="runtime-qpu-job-20260619",
                session_id="runtime-qpu-session-20260619",
                circuit_fingerprint="qiskit:ry(theta):z:v1",
                observable_fingerprint="SparsePauliOp:Z:v1",
                parameter_digest="sha256:" + "8" * 64,
                result_digest="sha256:" + "9" * 64,
                metadata_digest="sha256:" + "a" * 64,
                transpiled_circuit_digest="sha256:" + "b" * 64,
                live_execution_ticket=" ",
                backend_allowlist_id="backend-allowlist-20260619",
                shot_budget_id="shot-budget-20260619",
                runtime_session_mode="live_qpu_session",
                shots=4096,
                hardware_execution=True,
            ),
            "live_execution_ticket",
        ),
        (
            lambda: QiskitRuntimeQPUExecutionArtifact(
                artifact_id="qiskit-runtime-qpu-20260619",
                provider_name="ibm_quantum",
                primitive_name="EstimatorV2",
                backend_name="ibm_brisbane",
                job_id="runtime-qpu-job-20260619",
                session_id="runtime-qpu-session-20260619",
                circuit_fingerprint="qiskit:ry(theta):z:v1",
                observable_fingerprint="SparsePauliOp:Z:v1",
                parameter_digest="sha256:" + "8" * 64,
                result_digest="sha256:" + "9" * 64,
                metadata_digest="sha256:" + "a" * 64,
                transpiled_circuit_digest="sha256:" + "b" * 64,
                live_execution_ticket="live-ticket-20260619",
                backend_allowlist_id="backend-allowlist-20260619",
                shot_budget_id="shot-budget-20260619",
                runtime_session_mode="live_qpu_session",
                shots=4096,
                hardware_execution=False,
            ),
            "hardware execution",
        ),
    ],
)
def test_qiskit_runtime_qpu_execution_artifact_rejects_malformed_evidence(
    factory: Callable[[], QiskitRuntimeQPUExecutionArtifact],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_qiskit_runtime_primitive_artifact_rejects_hardware_execution_claim() -> None:
    with pytest.raises(ValueError, match="must not claim hardware execution"):
        QiskitRuntimePrimitiveExecutionArtifact(
            artifact_id="qiskit-runtime-estimator-20260616",
            provider_name="ibm_quantum",
            primitive_name="EstimatorV2",
            backend_name="ibm_brisbane",
            job_id="runtime-job-20260616",
            session_id=None,
            circuit_fingerprint="qiskit:ry(theta):z:v1",
            observable_fingerprint="SparsePauliOp:Z:v1",
            parameter_digest="sha256:" + "1" * 64,
            result_digest="sha256:" + "2" * 64,
            metadata_digest="sha256:" + "3" * 64,
            shots=400,
            hardware_execution=True,
        )


def _qiskit_raw_count_replay_artifact(
    *,
    provider_name: str = "ibm_quantum",
    backend_name: str = "ibm_brisbane",
    job_id: str = "runtime-qpu-job-20260619",
    circuit_fingerprint: str = "qiskit:ry(theta):z:v1",
    shots: int = 4096,
    live_ticket_id: str = "live-ticket-20260619",
) -> QiskitRawCountReplayArtifact:
    return QiskitRawCountReplayArtifact(
        artifact_id="qiskit-raw-count-replay-20260616",
        provider_name=provider_name,
        backend_name=backend_name,
        job_id=job_id,
        circuit_fingerprint=circuit_fingerprint,
        counts_digest="sha256:" + "4" * 64,
        replay_digest="sha256:" + "5" * 64,
        shots=shots,
        measured_qubits=1,
        expectation_value=0.9210609940028851,
        standard_error=0.006,
        hardware_execution=True,
        live_ticket_id=live_ticket_id,
    )


def _qiskit_calibration_comparison_artifact(
    *,
    provider_name: str = "ibm_quantum",
    backend_name: str = "ibm_brisbane",
    circuit_fingerprint: str = "qiskit:ry(theta):z:v1",
    live_ticket_id: str = "live-ticket-20260619",
) -> QiskitCalibrationStatevectorComparisonArtifact:
    return QiskitCalibrationStatevectorComparisonArtifact(
        artifact_id="qiskit-calibration-comparison-20260616",
        provider_name=provider_name,
        backend_name=backend_name,
        calibration_snapshot_id="calibration-20260616",
        statevector_reference_artifact_id="statevector-reference-20260616",
        circuit_fingerprint=circuit_fingerprint,
        calibration_digest="sha256:" + "6" * 64,
        comparison_digest="sha256:" + "7" * 64,
        max_abs_error=0.012,
        tolerance=0.05,
        hardware_execution=True,
        live_ticket_id=live_ticket_id,
    )


def test_qiskit_maturity_audit_accepts_paired_qpu_raw_count_and_calibration_without_promotion() -> (
    None
):
    circuit, parameters, observable = _single_rotation_problem()
    qpu_artifact = _qiskit_runtime_qpu_execution_artifact()
    raw_count_artifact = _qiskit_raw_count_replay_artifact()
    calibration_artifact = _qiskit_calibration_comparison_artifact()

    result = run_qiskit_maturity_audit(
        circuit,
        observable,
        parameters,
        np.array([0.4], dtype=float),
        shots=400,
        runtime_qpu_execution_artifact=qpu_artifact,
        raw_count_replay_artifact=raw_count_artifact,
        calibration_comparison_artifact=calibration_artifact,
    )

    assert result.required_capabilities["live_qpu_execution_ticket"] == "passed"
    assert result.required_capabilities["raw_count_capture_replay_harness"] == "passed"
    assert (
        result.required_capabilities["live_backend_statevector_reference_comparison"] == "passed"
    )
    assert result.evidence["runtime_qpu_execution_artifact"] is qpu_artifact
    assert result.evidence["raw_count_replay_artifact"] is raw_count_artifact
    assert result.evidence["calibration_comparison_artifact"] is calibration_artifact
    assert result.local_reference_metadata["raw_count_replay_artifact_id"] == (
        raw_count_artifact.artifact_id
    )
    assert result.local_reference_metadata["calibration_comparison_artifact_id"] == (
        calibration_artifact.artifact_id
    )
    assert not result.ready_for_provider_exceedance
    assert "live_qpu_execution_ticket" not in result.open_gaps
    assert "raw_count_capture_replay_harness" not in result.open_gaps
    assert "live_backend_statevector_reference_comparison" not in result.open_gaps
    assert "promotion_grade_isolated_benchmarks" in result.open_gaps


def _qiskit_qpu_provider_evidence_bundle(
    *,
    isolated_benchmark_artifact_id: str | None = "isolated-qiskit-benchmark-20260619",
    valid_until_utc: str = "2026-07-19T00:00:00Z",
) -> QiskitRuntimeQPUProviderEvidenceBundle:
    return build_qiskit_runtime_qpu_provider_evidence_bundle(
        artifact_id="qiskit-provider-evidence-bundle-20260619",
        runtime_qpu_execution_artifact=_qiskit_runtime_qpu_execution_artifact(),
        raw_count_replay_artifact=_qiskit_raw_count_replay_artifact(),
        calibration_comparison_artifact=_qiskit_calibration_comparison_artifact(),
        captured_at_utc="2026-06-19T18:00:00Z",
        valid_until_utc=valid_until_utc,
        isolated_benchmark_artifact_id=isolated_benchmark_artifact_id,
    )


def _qiskit_provider_gradient_method_metadata(gradient_method: str) -> dict[str, object]:
    common_metadata: dict[str, object] = {
        "method_artifact_id": f"qiskit-gradient-{gradient_method}-provenance-20260619",
        "workflow_version": "qiskit-runtime-gradient-workflow-v1",
    }
    if gradient_method == "parameter_shift":
        return {
            **common_metadata,
            "method_schema": "parameter_shift_shift_rule",
            "shift_rule_id": "ry-half-pi-shift-rule-v1",
            "shift_count": 2,
        }
    if gradient_method == "finite_difference":
        return {
            **common_metadata,
            "method_schema": "finite_difference_stencil",
            "stencil": "central-two-point",
            "step_size": 0.001,
        }
    if gradient_method == "lcu":
        return {
            **common_metadata,
            "method_schema": "linear_combination_unitary",
            "generator_digest": "sha256:" + "d" * 64,
            "term_count": 1,
        }
    if gradient_method == "spsa":
        return {
            **common_metadata,
            "method_schema": "spsa_perturbation",
            "perturbation_seed": 0,
            "perturbation_count": 32,
        }
    if gradient_method == "qgt":
        return {
            **common_metadata,
            "method_schema": "quantum_geometric_tensor",
            "qgt_digest": "sha256:" + "e" * 64,
            "matrix_dimension": 1,
        }
    if gradient_method == "qfi":
        return {
            **common_metadata,
            "method_schema": "quantum_fisher_information",
            "qfi_digest": "sha256:" + "f" * 64,
            "matrix_dimension": 1,
        }
    raise ValueError(f"unsupported test gradient method: {gradient_method}")


def _qiskit_provider_gradient_workflow_artifact(
    gradient_method: str,
    *,
    artifact_id: str | None = None,
    primitive_name: str = "EstimatorV2",
    observable_fingerprint: str | None = "SparsePauliOp:Z:v1",
    parameter_digest: str = "sha256:" + "8" * 64,
    live_ticket_id: str = "live-ticket-20260619",
    method_metadata: dict[str, object] | None = None,
) -> QiskitProviderGradientWorkflowArtifact:
    metadata_payload = method_metadata
    if metadata_payload is None and gradient_method in (
        "parameter_shift",
        "finite_difference",
        "lcu",
        "spsa",
        "qgt",
        "qfi",
    ):
        metadata_payload = _qiskit_provider_gradient_method_metadata(gradient_method)
    return build_qiskit_provider_gradient_workflow_artifact(
        artifact_id=artifact_id or f"qiskit-gradient-workflow-{gradient_method}-20260619",
        provider_name="ibm_quantum",
        backend_name="ibm_brisbane",
        job_id="runtime-qpu-job-20260619",
        primitive_name=primitive_name,
        gradient_method=gradient_method,
        circuit_fingerprint="qiskit:ry(theta):z:v1",
        observable_fingerprint=observable_fingerprint,
        parameter_digest=parameter_digest,
        gradient_digest="sha256:" + "b" * 64,
        metadata_digest="sha256:" + "c" * 64,
        shots=4096,
        parameter_count=1,
        gradient_dimension=1,
        hardware_execution=True,
        live_ticket_id=live_ticket_id,
        method_metadata=metadata_payload,
        claim_boundary="qiskit_provider_gradient_workflow_capture",
    )


def _qiskit_provider_gradient_workflow_suite() -> tuple[
    QiskitProviderGradientWorkflowArtifact,
    ...,
]:
    return tuple(
        _qiskit_provider_gradient_workflow_artifact(method)
        for method in ("parameter_shift", "finite_difference", "lcu", "spsa", "qgt", "qfi")
    )


def test_qiskit_provider_evidence_bundle_keeps_gradient_workflow_gate_blocked() -> None:
    circuit, parameters, observable = _single_rotation_problem()
    bundle = _qiskit_qpu_provider_evidence_bundle()

    result = run_qiskit_maturity_audit(
        circuit,
        observable,
        parameters,
        np.array([0.4], dtype=float),
        shots=400,
        qpu_provider_evidence_bundle=bundle,
    )

    assert not result.ready_for_provider_exceedance
    assert "provider_gradient_workflow_evidence" in result.open_gaps
    assert result.evidence["qpu_provider_evidence_bundle"] is bundle
    assert result.required_capabilities["runtime_primitive_execution_evidence"] == "passed"
    assert result.required_capabilities["live_qpu_execution_ticket"] == "passed"
    assert result.required_capabilities["raw_count_capture_replay_harness"] == "passed"
    assert (
        result.required_capabilities["live_backend_statevector_reference_comparison"] == "passed"
    )
    assert result.required_capabilities["promotion_grade_isolated_benchmarks"] == "passed"
    assert result.required_capabilities["provider_gradient_workflow_evidence"] == "blocked"
    assert result.local_reference_metadata["qpu_provider_evidence_bundle_id"] == (
        bundle.artifact_id
    )
    assert result.local_reference_metadata["isolated_benchmark_artifact_id"] == (
        "isolated-qiskit-benchmark-20260619"
    )
    payload = bundle.to_dict()
    assert payload["artifact_id"] == bundle.artifact_id
    assert payload["valid_until_utc"] == "2026-07-19T00:00:00Z"
    assert payload["isolated_benchmark_artifact_id"] == "isolated-qiskit-benchmark-20260619"


def test_qiskit_gradient_workflow_suite_clears_provider_workflow_gate() -> None:
    circuit, parameters, observable = _single_rotation_problem()
    bundle = _qiskit_qpu_provider_evidence_bundle()
    gradient_artifacts = _qiskit_provider_gradient_workflow_suite()

    result = run_qiskit_maturity_audit(
        circuit,
        observable,
        parameters,
        np.array([0.4], dtype=float),
        shots=400,
        qpu_provider_evidence_bundle=bundle,
        provider_gradient_workflow_artifacts=gradient_artifacts,
    )

    assert result.ready_for_provider_exceedance
    assert not result.open_gaps
    assert result.evidence["provider_gradient_workflow_artifacts"] == gradient_artifacts
    assert result.required_capabilities["provider_gradient_workflow_evidence"] == "passed"
    assert result.local_reference_metadata["provider_gradient_workflow_methods"] == (
        "finite_difference",
        "lcu",
        "parameter_shift",
        "qfi",
        "qgt",
        "spsa",
    )
    assert result.local_reference_metadata["provider_gradient_workflow_artifact_count"] == 6
    assert result.local_reference_metadata["provider_gradient_workflow_method_schemas"] == (
        "finite_difference:finite_difference_stencil",
        "lcu:linear_combination_unitary",
        "parameter_shift:parameter_shift_shift_rule",
        "qfi:quantum_fisher_information",
        "qgt:quantum_geometric_tensor",
        "spsa:spsa_perturbation",
    )
    payload = gradient_artifacts[0].to_dict()
    assert payload["gradient_method"] == "parameter_shift"
    assert payload["hardware_execution"] is True
    method_metadata = cast(dict[str, object], payload["method_metadata"])
    assert method_metadata["method_schema"] == "parameter_shift_shift_rule"
    assert method_metadata["shift_rule_id"] == "ry-half-pi-shift-rule-v1"


def test_qiskit_gradient_workflow_artifact_requires_method_metadata() -> None:
    """Provider-gradient workflow artefacts must carry method-specific provenance."""
    with pytest.raises(ValueError, match="method_metadata"):
        build_qiskit_provider_gradient_workflow_artifact(
            artifact_id="qiskit-gradient-workflow-qgt-20260619",
            provider_name="ibm_quantum",
            backend_name="ibm_brisbane",
            job_id="runtime-qpu-job-20260619",
            primitive_name="EstimatorV2",
            gradient_method="qgt",
            circuit_fingerprint="qiskit:ry(theta):z:v1",
            observable_fingerprint="SparsePauliOp:Z:v1",
            parameter_digest="sha256:" + "8" * 64,
            gradient_digest="sha256:" + "b" * 64,
            metadata_digest="sha256:" + "c" * 64,
            shots=4096,
            parameter_count=1,
            gradient_dimension=1,
            hardware_execution=True,
            live_ticket_id="live-ticket-20260619",
        )


def test_qiskit_gradient_workflow_artifact_rejects_incomplete_method_metadata() -> None:
    """Provider-gradient provenance must include all required common fields."""
    method_metadata = _qiskit_provider_gradient_method_metadata("qgt")
    del method_metadata["method_artifact_id"]

    with pytest.raises(ValueError, match="missing required keys"):
        _qiskit_provider_gradient_workflow_artifact(
            "qgt",
            method_metadata=method_metadata,
        )


@pytest.mark.parametrize(
    ("gradient_method", "method_metadata", "match"),
    [
        (
            "parameter_shift",
            {
                **_qiskit_provider_gradient_method_metadata("parameter_shift"),
                "method_schema": "finite_difference_stencil",
            },
            "method_schema",
        ),
        (
            "finite_difference",
            {
                **_qiskit_provider_gradient_method_metadata("finite_difference"),
                "step_size": 0.0,
            },
            "step_size",
        ),
        (
            "lcu",
            {
                **_qiskit_provider_gradient_method_metadata("lcu"),
                "generator_digest": "sha256:not-a-digest",
            },
            "generator_digest",
        ),
        (
            "spsa",
            {
                **_qiskit_provider_gradient_method_metadata("spsa"),
                "perturbation_seed": -1,
            },
            "perturbation_seed",
        ),
        (
            "qgt",
            {
                **_qiskit_provider_gradient_method_metadata("qgt"),
                "matrix_dimension": 2,
            },
            "matrix_dimension",
        ),
        (
            "qfi",
            {
                **_qiskit_provider_gradient_method_metadata("qfi"),
                "unexpected_capture": "ambiguous",
            },
            "unsupported keys",
        ),
    ],
)
def test_qiskit_gradient_workflow_artifact_rejects_invalid_method_metadata(
    gradient_method: str,
    method_metadata: dict[str, object],
    match: str,
) -> None:
    """Provider-gradient workflow metadata stays method-specific and fail-closed."""
    with pytest.raises(ValueError, match=match):
        _qiskit_provider_gradient_workflow_artifact(
            gradient_method,
            method_metadata=method_metadata,
        )


def test_qiskit_gradient_workflow_suite_rejects_missing_method() -> None:
    circuit, parameters, observable = _single_rotation_problem()
    bundle = _qiskit_qpu_provider_evidence_bundle()

    with pytest.raises(ValueError, match="provider gradient workflow methods"):
        run_qiskit_maturity_audit(
            circuit,
            observable,
            parameters,
            np.array([0.4], dtype=float),
            shots=400,
            qpu_provider_evidence_bundle=bundle,
            provider_gradient_workflow_artifacts=_qiskit_provider_gradient_workflow_suite()[:-1],
        )


def test_qiskit_gradient_workflow_artifact_rejects_mismatched_qpu_chain() -> None:
    """Provider-gradient evidence must cite the same live-ticket chain."""
    circuit, parameters, observable = _single_rotation_problem()
    bundle = _qiskit_qpu_provider_evidence_bundle()
    gradient_artifacts = (
        _qiskit_provider_gradient_workflow_artifact(
            "parameter_shift",
            live_ticket_id="other-ticket",
        ),
        *tuple(
            _qiskit_provider_gradient_workflow_artifact(method)
            for method in ("finite_difference", "lcu", "spsa", "qgt", "qfi")
        ),
    )

    with pytest.raises(ValueError, match="provider_gradient_workflow_artifact.live_ticket_id"):
        run_qiskit_maturity_audit(
            circuit,
            observable,
            parameters,
            np.array([0.4], dtype=float),
            shots=400,
            qpu_provider_evidence_bundle=bundle,
            provider_gradient_workflow_artifacts=gradient_artifacts,
        )


def test_qiskit_gradient_workflow_suite_rejects_duplicate_artifact_ids() -> None:
    """Provider-gradient methods must not reuse a single evidence artifact ID."""
    circuit, parameters, observable = _single_rotation_problem()
    bundle = _qiskit_qpu_provider_evidence_bundle()
    gradient_artifacts = tuple(
        _qiskit_provider_gradient_workflow_artifact(
            method,
            artifact_id="qiskit-gradient-workflow-duplicate-20260619",
        )
        for method in ("parameter_shift", "finite_difference", "lcu", "spsa", "qgt", "qfi")
    )

    with pytest.raises(ValueError, match="artifact_id"):
        run_qiskit_maturity_audit(
            circuit,
            observable,
            parameters,
            np.array([0.4], dtype=float),
            shots=400,
            qpu_provider_evidence_bundle=bundle,
            provider_gradient_workflow_artifacts=gradient_artifacts,
        )


def test_qiskit_gradient_workflow_artifact_rejects_primitive_drift() -> None:
    """Provider-gradient workflows must keep the Runtime primitive family aligned."""
    circuit, parameters, observable = _single_rotation_problem()
    bundle = _qiskit_qpu_provider_evidence_bundle()
    gradient_artifacts = (
        _qiskit_provider_gradient_workflow_artifact("parameter_shift"),
        _qiskit_provider_gradient_workflow_artifact("finite_difference"),
        _qiskit_provider_gradient_workflow_artifact("lcu"),
        _qiskit_provider_gradient_workflow_artifact(
            "spsa",
            primitive_name="SamplerV2",
            observable_fingerprint=None,
        ),
        _qiskit_provider_gradient_workflow_artifact("qgt"),
        _qiskit_provider_gradient_workflow_artifact("qfi"),
    )

    with pytest.raises(ValueError, match="primitive_name"):
        run_qiskit_maturity_audit(
            circuit,
            observable,
            parameters,
            np.array([0.4], dtype=float),
            shots=400,
            qpu_provider_evidence_bundle=bundle,
            provider_gradient_workflow_artifacts=gradient_artifacts,
        )


def test_qiskit_gradient_workflow_artifact_rejects_observable_drift() -> None:
    """Provider-gradient workflows must cite the Runtime QPU observable."""
    circuit, parameters, observable = _single_rotation_problem()
    bundle = _qiskit_qpu_provider_evidence_bundle()
    gradient_artifacts = (
        _qiskit_provider_gradient_workflow_artifact("parameter_shift"),
        _qiskit_provider_gradient_workflow_artifact("finite_difference"),
        _qiskit_provider_gradient_workflow_artifact(
            "lcu",
            observable_fingerprint="SparsePauliOp:X:v1",
        ),
        _qiskit_provider_gradient_workflow_artifact("spsa"),
        _qiskit_provider_gradient_workflow_artifact("qgt"),
        _qiskit_provider_gradient_workflow_artifact("qfi"),
    )

    with pytest.raises(ValueError, match="observable_fingerprint"):
        run_qiskit_maturity_audit(
            circuit,
            observable,
            parameters,
            np.array([0.4], dtype=float),
            shots=400,
            qpu_provider_evidence_bundle=bundle,
            provider_gradient_workflow_artifacts=gradient_artifacts,
        )


def test_qiskit_gradient_workflow_artifact_rejects_parameter_digest_drift() -> None:
    """Provider-gradient workflows must use the Runtime QPU parameter payload."""
    circuit, parameters, observable = _single_rotation_problem()
    bundle = _qiskit_qpu_provider_evidence_bundle()
    gradient_artifacts = (
        _qiskit_provider_gradient_workflow_artifact("parameter_shift"),
        _qiskit_provider_gradient_workflow_artifact("finite_difference"),
        _qiskit_provider_gradient_workflow_artifact("lcu"),
        _qiskit_provider_gradient_workflow_artifact("spsa"),
        _qiskit_provider_gradient_workflow_artifact(
            "qgt",
            parameter_digest="sha256:" + "d" * 64,
        ),
        _qiskit_provider_gradient_workflow_artifact("qfi"),
    )

    with pytest.raises(ValueError, match="parameter_digest"):
        run_qiskit_maturity_audit(
            circuit,
            observable,
            parameters,
            np.array([0.4], dtype=float),
            shots=400,
            qpu_provider_evidence_bundle=bundle,
            provider_gradient_workflow_artifacts=gradient_artifacts,
        )


def test_qiskit_gradient_workflow_artifact_rejects_sampler_observable() -> None:
    with pytest.raises(ValueError, match="observable_fingerprint"):
        _qiskit_provider_gradient_workflow_artifact(
            "spsa",
            primitive_name="SamplerV2",
            observable_fingerprint="SparsePauliOp:Z:v1",
        )


def test_qiskit_gradient_workflow_artifact_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="gradient_method"):
        _qiskit_provider_gradient_workflow_artifact("adjoint")


def test_qiskit_provider_evidence_bundle_keeps_benchmark_gate_blocked_without_benchmark() -> None:
    circuit, parameters, observable = _single_rotation_problem()
    bundle = _qiskit_qpu_provider_evidence_bundle(isolated_benchmark_artifact_id=None)

    result = run_qiskit_maturity_audit(
        circuit,
        observable,
        parameters,
        np.array([0.4], dtype=float),
        shots=400,
        qpu_provider_evidence_bundle=bundle,
    )

    assert not result.ready_for_provider_exceedance
    assert "promotion_grade_isolated_benchmarks" in result.open_gaps
    assert result.required_capabilities["promotion_grade_isolated_benchmarks"] == "blocked"


def test_qiskit_provider_evidence_bundle_rejects_mismatched_raw_count_chain() -> None:
    with pytest.raises(ValueError, match="raw_count_replay_artifact.job_id"):
        build_qiskit_runtime_qpu_provider_evidence_bundle(
            artifact_id="qiskit-provider-evidence-bundle-20260619",
            runtime_qpu_execution_artifact=_qiskit_runtime_qpu_execution_artifact(),
            raw_count_replay_artifact=_qiskit_raw_count_replay_artifact(
                job_id="runtime-qpu-job-other"
            ),
            calibration_comparison_artifact=_qiskit_calibration_comparison_artifact(),
            captured_at_utc="2026-06-19T18:00:00Z",
            valid_until_utc="2026-07-19T00:00:00Z",
            isolated_benchmark_artifact_id="isolated-qiskit-benchmark-20260619",
        )


def test_qiskit_provider_evidence_bundle_rejects_inverted_freshness_window() -> None:
    """Provider bundles require a valid capture-to-expiry evidence window."""
    with pytest.raises(ValueError, match="valid_until_utc"):
        _qiskit_qpu_provider_evidence_bundle(valid_until_utc="2026-06-18T00:00:00Z")


def test_qiskit_maturity_audit_rejects_stale_provider_evidence_bundle() -> None:
    """Qiskit provider-exceedance audits reject expired provider evidence bundles."""
    circuit, parameters, observable = _single_rotation_problem()
    bundle = _qiskit_qpu_provider_evidence_bundle(valid_until_utc="2026-06-27T00:00:00Z")

    with pytest.raises(ValueError, match="qpu_provider_evidence_bundle.valid_until_utc"):
        run_qiskit_maturity_audit(
            circuit,
            observable,
            parameters,
            np.array([0.4], dtype=float),
            shots=400,
            qpu_provider_evidence_bundle=bundle,
            provider_gradient_workflow_artifacts=_qiskit_provider_gradient_workflow_suite(),
        )


def test_qiskit_maturity_audit_rejects_bundle_mixed_with_individual_qpu_artifacts() -> None:
    circuit, parameters, observable = _single_rotation_problem()

    with pytest.raises(ValueError, match="qpu_provider_evidence_bundle"):
        run_qiskit_maturity_audit(
            circuit,
            observable,
            parameters,
            np.array([0.4], dtype=float),
            shots=400,
            qpu_provider_evidence_bundle=_qiskit_qpu_provider_evidence_bundle(),
            runtime_qpu_execution_artifact=_qiskit_runtime_qpu_execution_artifact(),
        )


def test_qiskit_maturity_audit_rejects_unpaired_raw_count_replay_artifact() -> None:
    circuit, parameters, observable = _single_rotation_problem()

    with pytest.raises(ValueError, match="raw-count replay artefact requires"):
        run_qiskit_maturity_audit(
            circuit,
            observable,
            parameters,
            np.array([0.4], dtype=float),
            shots=400,
            raw_count_replay_artifact=_qiskit_raw_count_replay_artifact(),
        )


def test_qiskit_maturity_audit_rejects_mismatched_raw_count_replay_artifact() -> None:
    circuit, parameters, observable = _single_rotation_problem()

    with pytest.raises(ValueError, match="raw_count_replay_artifact.job_id"):
        run_qiskit_maturity_audit(
            circuit,
            observable,
            parameters,
            np.array([0.4], dtype=float),
            shots=400,
            runtime_qpu_execution_artifact=_qiskit_runtime_qpu_execution_artifact(),
            raw_count_replay_artifact=_qiskit_raw_count_replay_artifact(
                job_id="runtime-qpu-job-other"
            ),
        )


def test_qiskit_maturity_audit_rejects_unpaired_calibration_comparison_artifact() -> None:
    circuit, parameters, observable = _single_rotation_problem()

    with pytest.raises(ValueError, match="calibration comparison artefact requires"):
        run_qiskit_maturity_audit(
            circuit,
            observable,
            parameters,
            np.array([0.4], dtype=float),
            shots=400,
            calibration_comparison_artifact=_qiskit_calibration_comparison_artifact(),
        )


def test_qiskit_maturity_audit_rejects_mismatched_calibration_comparison_artifact() -> None:
    circuit, parameters, observable = _single_rotation_problem()

    with pytest.raises(ValueError, match="calibration_comparison_artifact.circuit_fingerprint"):
        run_qiskit_maturity_audit(
            circuit,
            observable,
            parameters,
            np.array([0.4], dtype=float),
            shots=400,
            runtime_qpu_execution_artifact=_qiskit_runtime_qpu_execution_artifact(),
            calibration_comparison_artifact=_qiskit_calibration_comparison_artifact(
                circuit_fingerprint="qiskit:other:v1"
            ),
        )


def test_qiskit_raw_count_replay_artifact_rejects_non_live_counts() -> None:
    with pytest.raises(ValueError, match="must cite hardware execution"):
        QiskitRawCountReplayArtifact(
            artifact_id="qiskit-raw-count-replay-20260616",
            provider_name="ibm_quantum",
            backend_name="ibm_brisbane",
            job_id="runtime-job-20260616",
            circuit_fingerprint="qiskit:ry(theta):z:v1",
            counts_digest="sha256:" + "4" * 64,
            replay_digest="sha256:" + "5" * 64,
            shots=4096,
            measured_qubits=1,
            expectation_value=0.9,
            standard_error=0.006,
            hardware_execution=False,
            live_ticket_id="live-ticket-20260616",
        )


def test_qiskit_calibration_comparison_artifact_rejects_failed_tolerance() -> None:
    with pytest.raises(ValueError, match="max_abs_error must be within tolerance"):
        QiskitCalibrationStatevectorComparisonArtifact(
            artifact_id="qiskit-calibration-comparison-20260616",
            provider_name="ibm_quantum",
            backend_name="ibm_brisbane",
            calibration_snapshot_id="calibration-20260616",
            statevector_reference_artifact_id="statevector-reference-20260616",
            circuit_fingerprint="qiskit:ry(theta):z:v1",
            calibration_digest="sha256:" + "6" * 64,
            comparison_digest="sha256:" + "7" * 64,
            max_abs_error=0.08,
            tolerance=0.05,
            hardware_execution=True,
            live_ticket_id="live-ticket-20260616",
        )


RUNTIME_FUNCTIONS = (
    "build_qiskit_runtime_qpu_execution_artifact",
    "build_qiskit_provider_gradient_workflow_artifact",
    "build_qiskit_runtime_qpu_provider_evidence_bundle",
    "run_qiskit_maturity_audit",
)
PRIVATE_RUNTIME_HELPERS = (
    "_validate_qiskit_provider_evidence_bundle_freshness",
    "_normalise_provider_gradient_workflow_artifacts",
    "_validate_provider_gradient_workflow_chain",
    "_require_matching_optional_evidence_field",
)


def test_qiskit_runtime_leaf_has_no_bridge_back_edge() -> None:
    """Keep Runtime evidence orchestration independent from the facade."""
    tree = ast.parse(inspect.getsource(qiskit_runtime))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "qiskit_bridge" not in relative_imports
    assert "__init__" not in relative_imports


def test_qiskit_runtime_keeps_leaf_bridge_and_phase_identity() -> None:
    """Re-export every public Runtime route as the same function object."""
    for name in RUNTIME_FUNCTIONS:
        leaf_function = getattr(qiskit_runtime, name)
        assert getattr(qiskit_bridge, name) is leaf_function
        assert getattr(phase, name) is leaf_function


def test_qiskit_private_runtime_helpers_remain_exact_bridge_aliases() -> None:
    """Keep freshness and evidence-chain helpers stable."""
    for name in PRIVATE_RUNTIME_HELPERS:
        assert getattr(qiskit_bridge, name) is getattr(qiskit_runtime, name)


def test_qiskit_compatibility_facade_defines_no_functions() -> None:
    """Keep the completed Qiskit facade free of executable definitions."""
    tree = ast.parse(inspect.getsource(qiskit_bridge))
    assert not [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
