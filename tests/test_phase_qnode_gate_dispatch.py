# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Registered gate dispatch tests
"""Exercise the shared gate dispatcher on a NumPy backend built from the gate leaves."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.phase.qnode_circuit import registered_phase_qnode_gates
from scpn_quantum_control.phase.qnode_circuit_execution import _gate_matrix
from scpn_quantum_control.phase_qnode_gate_dispatch import (
    GateOperatorFactories,
    dispatch_gate_operator,
)
from scpn_quantum_control.phase_qnode_gate_numerics import (
    controlled_z_operator,
    hadamard_operator,
    pauli_rotation,
    quarter_turn_phase,
    sqrt_x_operator,
)
from scpn_quantum_control.phase_qnode_native_operators import (
    pauli_operator_entries,
    phase_operator_entries,
    two_qubit_operator_entries,
)


def _complex(values: object) -> Any:
    return np.asarray(values, dtype=np.complex128)


def _controlled(target: Any) -> Any:
    """Return the block-diagonal operator adding one control qubit to ``target``."""
    operator = np.eye(2 * target.shape[0], dtype=np.complex128)
    operator[target.shape[0] :, target.shape[0] :] = target
    return operator


def _transposition(first: int, second: int) -> Any:
    """Return the eight-dimensional permutation swapping two basis states."""
    operator = np.eye(8, dtype=np.complex128)
    operator[[first, second]] = operator[[second, first]]
    return operator


def _numpy_factories(calls: list[str]) -> GateOperatorFactories:
    """Return NumPy constructions over the shared leaves, recording recursive lookups."""

    def phase(angle: Any) -> Any:
        return np.diag(np.array([1.0, np.exp(1.0j * angle)], dtype=np.complex128))

    def gate(name: str, angle: Any) -> Any:
        calls.append(name)
        return dispatch_gate_operator(name, angle, _numpy_factories(calls))

    def unsupported(name: str) -> Any:
        raise ValueError(f"unsupported NumPy Phase-QNode gate: {name}")

    return GateOperatorFactories(
        identity=lambda: np.eye(2, dtype=np.complex128),
        x=lambda: _complex(pauli_operator_entries("x")),
        y=lambda: _complex(pauli_operator_entries("y")),
        z=lambda: _complex(pauli_operator_entries("z")),
        h=lambda: hadamard_operator(
            np, real_tensor=lambda values: np.asarray(values), complex_tensor=_complex
        ),
        s=lambda: _complex(phase_operator_entries(1.0j)),
        t=lambda: _complex(phase_operator_entries(quarter_turn_phase(lambda: np))),
        sx=lambda: sqrt_x_operator(complex_tensor=_complex),
        rotation=lambda angle, identity, operator: pauli_rotation(
            np, angle, _complex(1.0j), identity=identity, operator=operator
        ),
        phase=phase,
        two_qubit=lambda name: _complex(two_qubit_operator_entries(name)),
        controlled_z=lambda controls: controlled_z_operator(np, controls, complex_tensor=_complex),
        controlled=lambda: _controlled,
        ccnot=lambda: _transposition(6, 7),
        cswap=lambda: _transposition(5, 6),
        two_qubit_identity=lambda: np.eye(4, dtype=np.complex128),
        kron=lambda: np.kron,
        gate=gate,
        unsupported=unsupported,
    )


@pytest.mark.parametrize("gate", registered_phase_qnode_gates())
def test_every_registered_gate_matches_the_numpy_execution_reference(gate: str) -> None:
    """Dispatch each registered gate and compare with the circuit executor's matrix.

    Parameters
    ----------
    gate
        Registered Phase-QNode gate name.

    """
    calls: list[str] = []
    actual = dispatch_gate_operator(gate, 0.37, _numpy_factories(calls))
    np.testing.assert_allclose(actual, _gate_matrix(gate, 0.37), rtol=0, atol=1e-15)
    assert calls == ([gate[1:]] if gate in ("crx", "cry", "crz") else [])


def test_unregistered_gate_reaches_the_callers_refusal() -> None:
    """Leave the refusal of an unknown gate to the supplied factory."""
    with pytest.raises(ValueError, match="unsupported NumPy Phase-QNode gate: toffoli"):
        dispatch_gate_operator("toffoli", 0.0, _numpy_factories([]))
