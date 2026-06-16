# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Qiskit Bridge
"""Tests for Qiskit parameter-shift circuit generation and local gradients."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    QiskitCalibrationStatevectorComparisonArtifact,
    QiskitMaturityAuditResult,
    QiskitParameterShiftGradientResult,
    QiskitParameterShiftRecord,
    QiskitRawCountReplayArtifact,
    QiskitRuntimePrimitiveExecutionArtifact,
    execute_qiskit_finite_shot_parameter_shift,
    execute_qiskit_statevector_parameter_shift,
    generate_qiskit_parameter_shift_circuits,
    multi_frequency_parameter_shift_rule,
    parameter_shift_phase_qnode_gradient,
    plan_phase_qnode_parameter_shift_evaluations,
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


def test_generate_qiskit_parameter_shift_circuits_binds_plus_minus_values() -> None:
    circuit, parameters, _observable = _single_rotation_problem()

    records = generate_qiskit_parameter_shift_circuits(
        circuit,
        parameters,
        np.array([0.4], dtype=float),
    )

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, QiskitParameterShiftRecord)
    assert record.parameter_index == 0
    assert record.parameter_name == "theta"
    assert record.plus_values[0] == pytest.approx(0.4 + np.pi / 2.0)
    assert record.minus_values[0] == pytest.approx(0.4 - np.pi / 2.0)
    assert record.plus_circuit.num_parameters == 0
    assert record.minus_circuit.num_parameters == 0
    assert record.to_dict()["parameter_name"] == "theta"


def test_execute_qiskit_statevector_parameter_shift_matches_analytic_reference() -> None:
    circuit, parameters, observable = _single_rotation_problem()

    result = execute_qiskit_statevector_parameter_shift(
        circuit,
        observable,
        parameters,
        np.array([0.4], dtype=float),
    )

    assert isinstance(result, QiskitParameterShiftGradientResult)
    assert result.method == "qiskit_statevector_parameter_shift"
    assert result.evaluations == 3
    assert result.value == pytest.approx(np.cos(0.4), abs=1e-12)
    assert result.claim_boundary.startswith("local Qiskit Statevector")
    np.testing.assert_allclose(result.gradient, np.array([-np.sin(0.4)]), atol=1e-12)
    assert result.to_dict()["evaluations"] == 3


def test_execute_qiskit_finite_shot_parameter_shift_reports_uncertainty() -> None:
    circuit, parameters, observable = _single_rotation_problem()
    shots = 400

    result = execute_qiskit_finite_shot_parameter_shift(
        circuit,
        observable,
        parameters,
        np.array([0.4], dtype=float),
        shots=shots,
    )

    expected_gradient = -np.sin(0.4)
    expected_variance = np.cos(0.4) ** 2
    expected_standard_error = 0.5 * np.sqrt(expected_variance / shots + expected_variance / shots)
    assert result.backend == "finite_shot_simulator"
    assert result.method == "stochastic_parameter_shift"
    assert result.total_evaluations == 2
    assert result.total_shots == 2 * shots
    np.testing.assert_allclose(result.gradient, np.array([expected_gradient]), atol=1e-12)
    np.testing.assert_allclose(result.standard_error, np.array([expected_standard_error]))
    assert result.records[0].plus.shots == shots
    assert result.records[0].minus.shots == shots


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


def _qiskit_raw_count_replay_artifact() -> QiskitRawCountReplayArtifact:
    return QiskitRawCountReplayArtifact(
        artifact_id="qiskit-raw-count-replay-20260616",
        provider_name="ibm_quantum",
        backend_name="ibm_brisbane",
        job_id="runtime-job-20260616",
        circuit_fingerprint="qiskit:ry(theta):z:v1",
        counts_digest="sha256:" + "4" * 64,
        replay_digest="sha256:" + "5" * 64,
        shots=4096,
        measured_qubits=1,
        expectation_value=0.9210609940028851,
        standard_error=0.006,
        hardware_execution=True,
        live_ticket_id="live-ticket-20260616",
    )


def _qiskit_calibration_comparison_artifact() -> QiskitCalibrationStatevectorComparisonArtifact:
    return QiskitCalibrationStatevectorComparisonArtifact(
        artifact_id="qiskit-calibration-comparison-20260616",
        provider_name="ibm_quantum",
        backend_name="ibm_brisbane",
        calibration_snapshot_id="calibration-20260616",
        statevector_reference_artifact_id="statevector-reference-20260616",
        circuit_fingerprint="qiskit:ry(theta):z:v1",
        calibration_digest="sha256:" + "6" * 64,
        comparison_digest="sha256:" + "7" * 64,
        max_abs_error=0.012,
        tolerance=0.05,
        hardware_execution=True,
        live_ticket_id="live-ticket-20260616",
    )


def test_qiskit_maturity_audit_accepts_raw_count_and_calibration_artifacts_without_promotion() -> (
    None
):
    circuit, parameters, observable = _single_rotation_problem()
    raw_count_artifact = _qiskit_raw_count_replay_artifact()
    calibration_artifact = _qiskit_calibration_comparison_artifact()

    result = run_qiskit_maturity_audit(
        circuit,
        observable,
        parameters,
        np.array([0.4], dtype=float),
        shots=400,
        raw_count_replay_artifact=raw_count_artifact,
        calibration_comparison_artifact=calibration_artifact,
    )

    assert result.required_capabilities["raw_count_capture_replay_harness"] == "passed"
    assert (
        result.required_capabilities["live_backend_statevector_reference_comparison"] == "passed"
    )
    assert result.evidence["raw_count_replay_artifact"] is raw_count_artifact
    assert result.evidence["calibration_comparison_artifact"] is calibration_artifact
    assert result.local_reference_metadata["raw_count_replay_artifact_id"] == (
        raw_count_artifact.artifact_id
    )
    assert result.local_reference_metadata["calibration_comparison_artifact_id"] == (
        calibration_artifact.artifact_id
    )
    assert not result.ready_for_provider_exceedance
    assert "raw_count_capture_replay_harness" not in result.open_gaps
    assert "live_backend_statevector_reference_comparison" not in result.open_gaps
    assert "live_qpu_execution_ticket" in result.open_gaps
    assert "promotion_grade_isolated_benchmarks" in result.open_gaps


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


def test_qiskit_statevector_supports_multi_frequency_parameter_shift() -> None:
    theta = Parameter("theta")
    circuit = QuantumCircuit(2)
    circuit.ry(theta, 0)
    circuit.ry(2.0 * theta, 1)
    observable = SparsePauliOp.from_list([("IZ", 1.0), ("ZI", 0.1)])
    values = np.array([0.4], dtype=float)
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    records = generate_qiskit_parameter_shift_circuits(
        circuit,
        (theta,),
        values,
        rule=rule,
    )
    result = execute_qiskit_statevector_parameter_shift(
        circuit,
        observable,
        (theta,),
        values,
        rule=rule,
    )

    expected_gradient = -np.sin(values[0]) - 0.2 * np.sin(2.0 * values[0])
    assert len(records) == len(rule.terms)
    assert [record.shift_index for record in records] == [0, 1]
    assert records[0].to_dict()["shift_index"] == 0
    assert result.method == "qiskit_statevector_multi_frequency_parameter_shift"
    assert result.evaluations == 1 + 2 * len(rule.terms)
    np.testing.assert_allclose(result.gradient, np.array([expected_gradient]), atol=1e-12)


def test_phase_qnode_gate_aware_plan_matches_qiskit_tied_parameter_count() -> None:
    theta = Parameter("theta")
    qiskit_circuit = QuantumCircuit(1)
    qiskit_circuit.h(0)
    qiskit_circuit.rz(theta, 0)
    qiskit_circuit.rz(theta, 0)
    observable = SparsePauliOp.from_list([("X", 1.0)])
    phase_circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("h", (0,)), ("rz", (0,), 0), ("rz", (0,), 0)),
        observable=PauliTerm(1.0, ((0, "x"),)),
    )
    values = np.array([0.37], dtype=float)
    rule = multi_frequency_parameter_shift_rule([2.0])

    phase_plan = plan_phase_qnode_parameter_shift_evaluations(phase_circuit, values)
    phase_gradient = parameter_shift_phase_qnode_gradient(phase_circuit, values)
    qiskit_gradient = execute_qiskit_statevector_parameter_shift(
        qiskit_circuit,
        observable,
        (theta,),
        values,
        rule=rule,
    )

    assert phase_plan.planned_shifted_evaluations == 2
    assert phase_plan.operation_level_naive_evaluations == 4
    assert len(qiskit_gradient.records) == phase_plan.planned_shifted_evaluations // 2
    assert qiskit_gradient.evaluations == 1 + phase_plan.planned_shifted_evaluations
    np.testing.assert_allclose(
        phase_gradient.gradient,
        qiskit_gradient.gradient,
        atol=1e-12,
    )


def test_qiskit_finite_shot_supports_multi_frequency_parameter_shift() -> None:
    theta = Parameter("theta")
    circuit = QuantumCircuit(2)
    circuit.ry(theta, 0)
    circuit.ry(2.0 * theta, 1)
    observable = SparsePauliOp.from_list([("IZ", 1.0), ("ZI", 0.1)])
    values = np.array([0.4], dtype=float)
    shots = 300
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    result = execute_qiskit_finite_shot_parameter_shift(
        circuit,
        observable,
        (theta,),
        values,
        shots=shots,
        rule=rule,
    )

    expected_gradient = -np.sin(values[0]) - 0.2 * np.sin(2.0 * values[0])
    assert result.method == "multi_frequency_stochastic_parameter_shift"
    assert result.plan.shift_terms == len(rule.terms)
    assert result.total_evaluations == 2 * len(rule.terms)
    assert result.total_shots == 2 * len(rule.terms) * shots
    assert [record.shift_index for record in result.records] == [0, 1]
    np.testing.assert_allclose(result.gradient, np.array([expected_gradient]), atol=1e-12)
    assert result.standard_error[0] > 0.0


def test_execute_qiskit_statevector_parameter_shift_handles_two_parameters() -> None:
    theta_0 = Parameter("theta_0")
    theta_1 = Parameter("theta_1")
    circuit = QuantumCircuit(2)
    circuit.ry(theta_0, 0)
    circuit.ry(theta_1, 1)
    observable = SparsePauliOp.from_list([("IZ", 1.0), ("ZI", 0.25)])
    values = np.array([0.2, -0.4], dtype=float)

    result = execute_qiskit_statevector_parameter_shift(
        circuit,
        observable,
        (theta_0, theta_1),
        values,
    )

    np.testing.assert_allclose(
        result.gradient,
        np.array([-np.sin(values[0]), -0.25 * np.sin(values[1])], dtype=float),
        atol=1e-12,
    )
    assert result.evaluations == 5
    assert len(result.records) == 2


def test_qiskit_parameter_shift_rejects_unbound_or_bad_inputs() -> None:
    circuit, parameters, observable = _single_rotation_problem()
    extra = Parameter("extra")
    bad_circuit = circuit.copy()
    bad_circuit.rz(extra, 0)

    with pytest.raises(ValueError, match="all circuit parameters"):
        execute_qiskit_statevector_parameter_shift(
            bad_circuit,
            observable,
            parameters,
            np.array([0.4], dtype=float),
        )
    with pytest.raises(ValueError, match="values"):
        generate_qiskit_parameter_shift_circuits(
            circuit,
            parameters,
            np.array([0.4, 0.5], dtype=float),
        )
    with pytest.raises(ValueError, match="parameters"):
        generate_qiskit_parameter_shift_circuits(circuit, (), np.array([], dtype=float))
    with pytest.raises(ValueError, match="shots"):
        execute_qiskit_finite_shot_parameter_shift(
            circuit,
            observable,
            parameters,
            np.array([0.4], dtype=float),
            shots=0,
        )
