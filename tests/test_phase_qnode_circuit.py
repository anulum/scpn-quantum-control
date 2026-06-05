# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNode Circuit Registry
"""Tests for phase/qnode_circuit.py registered circuit execution."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase.qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeSupportError,
    SparsePauliHamiltonian,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
    phase_qnode_support_report,
    registered_phase_qnode_gates,
    registered_phase_qnode_observables,
)


def test_phase_qnode_registered_gate_family_executes_with_pauli_observables() -> None:
    params = np.linspace(0.11, 0.91, 10)
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("h", (0,)),
            ("x", (1,)),
            ("y", (0,)),
            ("z", (1,)),
            ("s", (0,)),
            ("t", (1,)),
            ("sx", (0,)),
            ("rx", (0,), 0),
            ("ry", (1,), 1),
            ("rz", (0,), 2),
            ("phase", (1,), 3),
            ("cnot", (0, 1)),
            ("cz", (1, 0)),
            ("cy", (0, 1)),
            ("swap", (0, 1)),
            ("crx", (0, 1), 4),
            ("cry", (1, 0), 5),
            ("crz", (0, 1), 6),
            ("rxx", (0, 1), 7),
            ("ryy", (0, 1), 8),
            ("rzz", (0, 1), 9),
        ),
        observable=SparsePauliHamiltonian(
            (
                PauliTerm(0.5, ((0, "x"),)),
                PauliTerm(-0.25, ((1, "y"),)),
                PauliTerm(0.75, ((0, "z"), (1, "z"))),
            )
        ),
    )

    result = execute_phase_qnode_circuit(circuit, params)

    assert np.isfinite(result.value)
    assert result.state.shape == (4,)
    np.testing.assert_allclose(np.vdot(result.state, result.state).real, 1.0, atol=1e-12)
    assert result.support_report.supported
    assert set(registered_phase_qnode_gates()) >= {
        "rx",
        "ry",
        "rz",
        "phase",
        "h",
        "x",
        "y",
        "z",
        "s",
        "t",
        "sx",
        "cnot",
        "cz",
        "cy",
        "swap",
        "crx",
        "cry",
        "crz",
        "rxx",
        "ryy",
        "rzz",
    }
    assert set(registered_phase_qnode_observables()) >= {
        "pauli_x",
        "pauli_y",
        "pauli_z",
        "weighted_pauli_sum",
        "pauli_product",
        "sparse_pauli_hamiltonian",
    }


def test_phase_qnode_parameter_shift_matches_finite_difference_for_registered_generators() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("ry", (0,), 0),
            ("cnot", (0, 1)),
            ("rzz", (0, 1), 1),
            ("rx", (1,), 2),
        ),
        observable=SparsePauliHamiltonian((PauliTerm(1.0, ((0, "z"), (1, "x"))),)),
    )
    params = np.array([0.31, -0.27, 0.43], dtype=float)

    gradient = parameter_shift_phase_qnode_gradient(circuit, params)
    finite_difference = np.zeros_like(params)
    eps = 1e-6
    for index in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[index] += eps
        minus[index] -= eps
        finite_difference[index] = (
            execute_phase_qnode_circuit(circuit, plus).value
            - execute_phase_qnode_circuit(circuit, minus).value
        ) / (2.0 * eps)

    np.testing.assert_allclose(gradient.gradient, finite_difference, atol=1e-6)
    assert gradient.support_report.differentiable_parameters == (0, 1, 2)
    assert gradient.parameter_shift_evaluations == 6


def test_phase_qnode_unsupported_routes_fail_with_structured_support_report() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("u3", (0,), 0),),
        observable="pauli_z",
    )

    report = phase_qnode_support_report(circuit, np.array([0.2], dtype=float))

    assert not report.supported
    assert report.unsupported_gates == ("u3",)
    assert "u3" in report.failure_reason
    with pytest.raises(PhaseQNodeSupportError) as exc_info:
        execute_phase_qnode_circuit(circuit, np.array([0.2], dtype=float))
    assert exc_info.value.report == report
