# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for controlled-rotation shift rules and U3 coverage
"""Controlled-rotation four-term parameter shift and U3/general-unitary support.

Covers the corrected controlled-rotation parameter-shift rule (the two-term rule
is only valid for single-Pauli generators; controlled rotations carry the
``{1/2, 1}`` spectrum and need the four-term rule) and the ZYZ decomposition that
gives U3 and arbitrary single-qubit unitaries exact, differentiable coverage.
"""

import numpy as np
import pytest

from scpn_quantum_control.phase.general_unitary import build_u3_operations, su2_zyz_angles
from scpn_quantum_control.phase.qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    _group_generator_frequencies,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)


def _value(circuit, params):
    return execute_phase_qnode_circuit(circuit, np.asarray(params, dtype=float)).value


def _five_point_gradient(circuit, params):
    params = np.asarray(params, dtype=float)
    h = 1e-4
    grad = np.zeros_like(params)
    for k in range(params.size):
        plus2, plus1, minus1, minus2 = (params.copy() for _ in range(4))
        plus2[k] += 2 * h
        plus1[k] += h
        minus1[k] -= h
        minus2[k] -= 2 * h
        grad[k] = (
            -_value(circuit, plus2)
            + 8 * _value(circuit, plus1)
            - 8 * _value(circuit, minus1)
            + _value(circuit, minus2)
        ) / (12 * h)
    return grad


# --------------------------------------------------------------------------- #
# Generator frequency spectrum
# --------------------------------------------------------------------------- #
def test_single_pauli_frequencies():
    op = PhaseQNodeOperation("ry", (0,), parameter_index=0)
    assert _group_generator_frequencies((op,)) == (1.0,)


@pytest.mark.parametrize("gate", ["crx", "cry", "crz"])
def test_controlled_rotation_frequencies(gate):
    op = PhaseQNodeOperation(gate, (0, 1), parameter_index=0)
    assert _group_generator_frequencies((op,)) == (0.5, 1.0)


def test_collapsible_controlled_rotation_frequencies_scale():
    op = PhaseQNodeOperation("cry", (0, 1), parameter_index=0)
    assert _group_generator_frequencies((op, op)) == (1.0, 2.0)


def test_collapsible_single_pauli_frequencies_scale():
    op = PhaseQNodeOperation("rx", (0,), parameter_index=0)
    assert _group_generator_frequencies((op, op, op)) == (3.0,)


# --------------------------------------------------------------------------- #
# Controlled-rotation gradient correctness (the four-term rule)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gate", ["crx", "cry", "crz"])
def test_controlled_rotation_gradient_control_coupling_observable(gate):
    # An X on the control qubit couples the control-on/off sectors and excites
    # the 1/2 frequency that the two-term rule mishandles.
    ops = (
        PhaseQNodeOperation("h", (0,)),
        PhaseQNodeOperation("h", (1,)),
        PhaseQNodeOperation(gate, (0, 1), parameter_index=0),
    )
    circuit = PhaseQNodeCircuit(
        n_qubits=2, operations=ops, observable=PauliTerm(1.0, ((0, "x"), (1, "z")))
    )
    for theta in np.linspace(0.1, 3.0, 10):
        analytic = parameter_shift_phase_qnode_gradient(circuit, np.array([theta])).gradient
        numeric = _five_point_gradient(circuit, [theta])
        assert np.allclose(analytic, numeric, atol=1e-6)


@pytest.mark.parametrize("gate", ["crx", "cry", "crz"])
def test_controlled_rotation_gradient_target_observable(gate):
    ops = (
        PhaseQNodeOperation("h", (0,)),
        PhaseQNodeOperation("h", (1,)),
        PhaseQNodeOperation(gate, (0, 1), parameter_index=0),
    )
    circuit = PhaseQNodeCircuit(n_qubits=2, operations=ops, observable=PauliTerm(1.0, ((1, "z"),)))
    for theta in np.linspace(0.1, 3.0, 8):
        analytic = parameter_shift_phase_qnode_gradient(circuit, np.array([theta])).gradient
        numeric = _five_point_gradient(circuit, [theta])
        assert np.allclose(analytic, numeric, atol=1e-6)


def test_collapsible_controlled_rotation_gradient():
    ops = (
        PhaseQNodeOperation("h", (0,)),
        PhaseQNodeOperation("h", (1,)),
        PhaseQNodeOperation("cry", (0, 1), parameter_index=0),
        PhaseQNodeOperation("cry", (0, 1), parameter_index=0),
    )
    circuit = PhaseQNodeCircuit(
        n_qubits=2, operations=ops, observable=PauliTerm(1.0, ((0, "x"), (1, "z")))
    )
    for theta in np.linspace(0.1, 3.0, 8):
        analytic = parameter_shift_phase_qnode_gradient(circuit, np.array([theta])).gradient
        numeric = _five_point_gradient(circuit, [theta])
        assert np.allclose(analytic, numeric, atol=1e-6)


def test_single_pauli_rotation_gradient_unchanged():
    ops = (
        PhaseQNodeOperation("rx", (0,), parameter_index=0),
        PhaseQNodeOperation("ry", (0,), parameter_index=1),
    )
    circuit = PhaseQNodeCircuit(n_qubits=1, operations=ops, observable=PauliTerm(1.0, ((0, "z"),)))
    params = np.array([0.6, 1.1])
    analytic = parameter_shift_phase_qnode_gradient(circuit, params).gradient
    numeric = _five_point_gradient(circuit, params)
    assert np.allclose(analytic, numeric, atol=1e-6)


# --------------------------------------------------------------------------- #
# U3 / general-unitary ZYZ decomposition
# --------------------------------------------------------------------------- #
def _random_unitary(rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    q, r = np.linalg.qr(z)
    return q @ np.diag(np.diagonal(r) / np.abs(np.diagonal(r)))


def test_su2_zyz_roundtrip_statevector():
    rng = np.random.default_rng(0)
    ops = build_u3_operations(0, (0, 1, 2))
    circuit = PhaseQNodeCircuit(n_qubits=1, operations=ops, observable=PauliTerm(1.0, ((0, "z"),)))
    for _ in range(50):
        unitary = _random_unitary(rng)
        phi, theta, lam = su2_zyz_angles(unitary)
        result = execute_phase_qnode_circuit(circuit, np.array([theta, phi, lam]))
        target = unitary @ np.array([1.0, 0.0], dtype=complex)
        global_phase = np.vdot(target, result.state)
        global_phase /= abs(global_phase)
        assert np.allclose(result.state, target * global_phase, atol=1e-10)


@pytest.mark.parametrize(
    "gate_matrix",
    [
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),  # X (theta = pi)
        np.eye(2, dtype=complex),  # identity (theta = 0)
        np.array([[1.0, 0.0], [0.0, 1j]], dtype=complex),  # S gate
    ],
)
def test_su2_zyz_special_cases(gate_matrix):
    phi, theta, lam = su2_zyz_angles(gate_matrix)
    ops = build_u3_operations(0, (0, 1, 2))
    circuit = PhaseQNodeCircuit(n_qubits=1, operations=ops, observable=PauliTerm(1.0, ((0, "z"),)))
    state = execute_phase_qnode_circuit(circuit, np.array([theta, phi, lam])).state
    target = gate_matrix @ np.array([1.0, 0.0], dtype=complex)
    global_phase = np.vdot(target, state)
    if abs(global_phase) > 1e-12:
        global_phase /= abs(global_phase)
        assert np.allclose(state, target * global_phase, atol=1e-9)


def test_u3_decomposition_is_differentiable():
    ops = build_u3_operations(0, (0, 1, 2))
    circuit = PhaseQNodeCircuit(n_qubits=1, operations=ops, observable=PauliTerm(1.0, ((0, "x"),)))
    params = np.array([0.7, 1.3, -0.4])
    analytic = parameter_shift_phase_qnode_gradient(circuit, params).gradient
    numeric = _five_point_gradient(circuit, params)
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_su2_zyz_rejects_bad_matrix():
    with pytest.raises(ValueError):
        su2_zyz_angles(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        su2_zyz_angles(np.array([[1.0, 2.0], [3.0, 4.0]]))


@pytest.mark.parametrize(
    "args",
    [
        (-1, (0, 1, 2)),
        (0, (0, 1)),
        (0, (0, 1, 1)),
        (0, (0, -1, 2)),
    ],
)
def test_build_u3_operations_rejects_bad_args(args):
    with pytest.raises(ValueError):
        build_u3_operations(*args)
