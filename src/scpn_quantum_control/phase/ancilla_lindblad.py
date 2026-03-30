# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Single-Ancilla Lindblad Circuit
"""Simulate open-system Kuramoto-XY dynamics on quantum hardware using
a single ancilla qubit.

Instead of representing the full density matrix (exponentially expensive),
this method uses repeated interactions with a single environment qubit
that is reset after each interaction step. Each step applies:

  1. System-ancilla entangling gate (controlled Ry with angle ∝ √γ·dt)
  2. Measure and reset ancilla
  3. Continue with system in partially decohered state

This implements amplitude damping at rate γ per qubit per step.

Reference: Cattaneo et al., PRR 6, 043321 (2024);
Hu et al., arXiv:2312.05371 (2023).
"""

from __future__ import annotations

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


def build_ancilla_lindblad_circuit(
    K: np.ndarray,
    omega: np.ndarray,
    t: float = 0.1,
    trotter_reps: int = 5,
    gamma: float = 0.05,
    n_dissipation_steps: int = 3,
) -> QuantumCircuit:
    """Build a circuit for open-system Kuramoto-XY with single ancilla.

    Parameters
    ----------
    K : array (n, n)
        Coupling matrix.
    omega : array (n,)
        Natural frequencies.
    t : float
        Total evolution time.
    trotter_reps : int
        Trotter steps for coherent evolution.
    gamma : float
        Amplitude damping rate.
    n_dissipation_steps : int
        Number of dissipation rounds (ancilla reset cycles).

    Returns
    -------
    QuantumCircuit
        Circuit with n system qubits + 1 ancilla qubit.
    """
    n = K.shape[0]
    sys_reg = QuantumRegister(n, "sys")
    anc_reg = QuantumRegister(1, "anc")
    cl_reg = ClassicalRegister(n, "meas")
    qc = QuantumCircuit(sys_reg, anc_reg, cl_reg)

    # Initial state preparation
    for i in range(n):
        qc.ry(float(omega[i]) % (2 * np.pi), sys_reg[i])

    dt_coherent = t / n_dissipation_steps
    dt_trotter = dt_coherent / trotter_reps

    for _diss_step in range(n_dissipation_steps):
        # Coherent evolution (Trotter)
        for _rep in range(trotter_reps):
            # Z terms
            for i in range(n):
                if abs(omega[i]) > 1e-15:
                    qc.rz(omega[i] * dt_trotter, sys_reg[i])
            # XY coupling terms
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(K[i, j]) > 1e-15:
                        qc.cx(sys_reg[i], sys_reg[j])
                        qc.rx(2 * K[i, j] * dt_trotter, sys_reg[j])
                        qc.cx(sys_reg[i], sys_reg[j])

        # Dissipation: interact each system qubit with ancilla
        angle = 2 * np.arcsin(np.sqrt(gamma * dt_coherent))
        angle = min(angle, np.pi)  # clamp

        for i in range(n):
            qc.cry(angle, sys_reg[i], anc_reg[0])
            qc.reset(anc_reg[0])

    # Final measurement
    qc.measure(sys_reg, cl_reg)

    return qc


def ancilla_circuit_stats(
    K: np.ndarray,
    omega: np.ndarray,
    t: float = 0.1,
    trotter_reps: int = 5,
    gamma: float = 0.05,
    n_dissipation_steps: int = 3,
) -> dict:
    """Return circuit statistics without building full circuit.

    Returns
    -------
    dict with: n_qubits, n_system, n_ancilla, estimated_depth, n_cx_gates,
               n_resets, total_gates
    """
    n = K.shape[0]
    n_couplings = sum(1 for i in range(n) for j in range(i + 1, n) if abs(K[i, j]) > 1e-15)

    cx_per_trotter = 2 * n_couplings
    rz_per_trotter = n
    rx_per_trotter = n_couplings

    gates_per_trotter = rz_per_trotter + cx_per_trotter + rx_per_trotter
    gates_coherent = gates_per_trotter * trotter_reps * n_dissipation_steps
    gates_dissipation = 2 * n * n_dissipation_steps  # cry + reset per qubit per step

    return {
        "n_qubits": n + 1,
        "n_system": n,
        "n_ancilla": 1,
        "n_cx_gates": cx_per_trotter * trotter_reps * n_dissipation_steps,
        "n_resets": n * n_dissipation_steps,
        "total_gates": gates_coherent + gates_dissipation + n,  # +n for initial Ry
        "n_dissipation_steps": n_dissipation_steps,
        "gamma": gamma,
    }
