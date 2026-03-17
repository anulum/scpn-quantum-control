# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""SSGF <> quantum bridge: geometry matrices to Hamiltonians, states to circuits."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from .knm_hamiltonian import knm_to_hamiltonian


def ssgf_w_to_hamiltonian(W: np.ndarray, omega: np.ndarray) -> SparsePauliOp:
    """Convert SSGF geometry matrix W to Pauli Hamiltonian.

    W has the same structure as K_nm (symmetric, non-negative, zero diagonal),
    so the existing knm_to_hamiltonian compiler applies directly.
    """
    return knm_to_hamiltonian(W, omega)


def ssgf_state_to_quantum(state_dict: dict) -> QuantumCircuit:
    """Encode SSGF oscillator phases into qubit XY-plane rotations.

    state_dict must contain 'theta': array of oscillator phases.
    Each qubit i gets Ry(pi/2)*Rz(theta_i), producing (|0>+e^{i*theta}|1>)/sqrt(2).
    This preserves phase in <X>=cos(theta), <Y>=sin(theta) for roundtrip recovery.
    """
    theta = np.asarray(state_dict["theta"], dtype=np.float64)
    n = len(theta)
    qc = QuantumCircuit(n)
    for i, t in enumerate(theta):
        qc.ry(np.pi / 2, i)
        qc.rz(float(t), i)
    return qc


def quantum_to_ssgf_state(sv: Statevector, n_osc: int) -> dict:
    """Extract oscillator phases and coherence from statevector.

    Per-qubit: theta_i = atan2(<Y_i>, <X_i>).
    R_global = |mean(exp(i*theta))|.
    """
    theta = np.zeros(n_osc)
    for i in range(n_osc):
        exp_x = _qubit_expectation(sv, n_osc, i, "X")
        exp_y = _qubit_expectation(sv, n_osc, i, "Y")
        theta[i] = np.arctan2(exp_y, exp_x)

    z = np.mean(np.exp(1j * theta))
    R_global = float(np.abs(z))
    return {"theta": theta, "R_global": R_global}


def _qubit_expectation(sv: Statevector, n: int, qubit: int, pauli: str) -> float:
    """Single-qubit Pauli expectation <psi|P_qubit|psi>."""
    label = ["I"] * n
    label[qubit] = pauli
    op = SparsePauliOp("".join(reversed(label)))
    return float(sv.expectation_value(op).real)
