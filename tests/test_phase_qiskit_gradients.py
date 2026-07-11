# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Qiskit Local Gradient Tests
"""Tests for Qiskit shifted-circuit generation and local gradients."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.qiskit_bridge as qiskit_bridge
import scpn_quantum_control.phase.qiskit_gradients as qiskit_gradients
from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    QiskitParameterShiftGradientResult,
    QiskitParameterShiftRecord,
    execute_qiskit_finite_shot_parameter_shift,
    execute_qiskit_statevector_parameter_shift,
    generate_qiskit_parameter_shift_circuits,
    multi_frequency_parameter_shift_rule,
    parameter_shift_phase_qnode_gradient,
    plan_phase_qnode_parameter_shift_evaluations,
)

GRADIENT_FUNCTIONS = (
    "generate_qiskit_parameter_shift_circuits",
    "execute_qiskit_statevector_parameter_shift",
    "execute_qiskit_finite_shot_parameter_shift",
)
PRIVATE_GRADIENT_HELPERS = (
    "_expectation",
    "_expectation_and_variance",
    "_bind_circuit",
    "_validate_circuit_parameters",
    "_normalise_parameters",
    "_parameter_shift_terms",
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
    plus_metadata = result.records[0].plus.metadata
    assert plus_metadata is not None
    assert plus_metadata["source_class"] == "local_simulator"
    assert plus_metadata["sample_seed"] == "deterministic-statevector-surrogate"
    assert plus_metadata["shift_direction"] == "plus"


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
    minus_metadata = result.records[1].minus.metadata
    assert minus_metadata is not None
    assert minus_metadata["shift_index"] == 1
    assert minus_metadata["shift_direction"] == "minus"
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


def test_qiskit_gradient_leaf_has_no_bridge_back_edge() -> None:
    """Keep local gradients independent from Runtime orchestration."""
    tree = ast.parse(inspect.getsource(qiskit_gradients))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "qiskit_bridge" not in relative_imports
    assert "__init__" not in relative_imports


def test_qiskit_gradients_keep_leaf_bridge_and_phase_identity() -> None:
    """Re-export every public local gradient route as the same function object."""
    for name in GRADIENT_FUNCTIONS:
        leaf_function = getattr(qiskit_gradients, name)
        assert getattr(qiskit_bridge, name) is leaf_function
        assert getattr(phase, name) is leaf_function


def test_qiskit_private_gradient_helpers_remain_exact_bridge_aliases() -> None:
    """Keep circuit, parameter, and expectation helpers stable."""
    for name in PRIVATE_GRADIENT_HELPERS:
        assert getattr(qiskit_bridge, name) is getattr(qiskit_gradients, name)


def test_qiskit_bridge_defines_no_duplicate_gradient_functions() -> None:
    """Prevent local gradient functions from drifting back into the bridge."""
    tree = ast.parse(inspect.getsource(qiskit_bridge))
    bridge_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert bridge_functions.isdisjoint(GRADIENT_FUNCTIONS)
    assert bridge_functions.isdisjoint(PRIVATE_GRADIENT_HELPERS)
