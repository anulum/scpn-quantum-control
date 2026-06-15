# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Qiskit Bridge
"""Tests for Qiskit parameter-shift circuit generation and local gradients."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    QiskitMaturityAuditResult,
    QiskitParameterShiftGradientResult,
    QiskitParameterShiftRecord,
    execute_qiskit_finite_shot_parameter_shift,
    execute_qiskit_statevector_parameter_shift,
    generate_qiskit_parameter_shift_circuits,
    multi_frequency_parameter_shift_rule,
    parameter_shift_phase_qnode_gradient,
    plan_phase_qnode_parameter_shift_evaluations,
    run_qiskit_maturity_audit,
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
    assert result.evidence["shifted_circuit_records"][0].parameter_name == "theta"
    assert result.evidence["statevector_reference"].method == "qiskit_statevector_parameter_shift"
    assert result.evidence["finite_shot_surrogate"].backend == "finite_shot_simulator"
    assert result.evidence["provider_preparation_audit"].passed
    assert result.required_capabilities["shifted_circuit_generation"] == "passed"
    assert result.required_capabilities["statevector_reference_comparison"] == "passed"
    assert result.required_capabilities["provider_hardware_preparation_policy"] == "passed"
    assert result.required_capabilities["raw_count_capture_replay_harness"] == "blocked"
    assert result.local_reference_metadata["shots"] == 400
    assert result.local_reference_metadata["statevector_finite_shot_max_abs_error"] <= 1e-12
    assert "live_qpu_execution_ticket" in result.open_gaps
    assert "raw_count_capture_replay_harness" in result.open_gaps
    payload = result.to_dict()
    assert payload["claim_boundary"] == "bounded_qiskit_provider_maturity_audit"
    assert payload["local_reference_metadata"]["parameter_count"] == 1


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
