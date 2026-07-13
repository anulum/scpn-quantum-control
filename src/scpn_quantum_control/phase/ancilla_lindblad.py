# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Single-Ancilla Lindblad Circuit
"""Simulate open-system Kuramoto-XY dynamics with one ancilla qubit.

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

from typing import TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

FloatArray: TypeAlias = NDArray[np.float64]


class AncillaCircuitStats(TypedDict):
    """Resource estimate for a single-ancilla Lindblad circuit."""

    n_qubits: int
    n_system: int
    n_ancilla: int
    n_cx_gates: int
    n_resets: int
    total_gates: int
    n_dissipation_steps: int
    gamma: float


def _validate_parameters(
    t: float, trotter_reps: int, gamma: float, n_dissipation_steps: int
) -> None:
    if not np.isfinite(t):
        raise ValueError("t must be finite")
    if t < 0.0:
        raise ValueError("t must be non-negative")
    if trotter_reps <= 0:
        raise ValueError("trotter_reps must be positive")
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    if n_dissipation_steps <= 0:
        raise ValueError("n_dissipation_steps must be positive")


def _amplitude_damping_angle(gamma: float, dt: float) -> float:
    """Return the finite-time single-qubit amplitude-damping rotation angle."""
    decay_probability = -np.expm1(-gamma * dt)
    decay_probability = min(max(float(decay_probability), 0.0), 1.0)
    return float(2.0 * np.arcsin(np.sqrt(decay_probability)))


def build_ancilla_lindblad_circuit(
    K: FloatArray,
    omega: FloatArray,
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

    Raises
    ------
    ValueError
        If the evolution time or damping rate is non-finite or negative, or if
        either step count is non-positive.

    """
    _validate_parameters(t, trotter_reps, gamma, n_dissipation_steps)
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

        # Dissipation: interact each system qubit with ancilla.  The finite-time
        # amplitude-damping probability is p = 1 - exp(-gamma * dt), so the
        # controlled rotation angle is 2 asin(sqrt(p)).
        angle = _amplitude_damping_angle(gamma, dt_coherent)

        for i in range(n):
            qc.cry(angle, sys_reg[i], anc_reg[0])
            qc.reset(anc_reg[0])

    # Final measurement
    qc.measure(sys_reg, cl_reg)

    return qc


def ancilla_circuit_stats(
    K: FloatArray,
    omega: FloatArray,
    t: float = 0.1,
    trotter_reps: int = 5,
    gamma: float = 0.05,
    n_dissipation_steps: int = 3,
) -> AncillaCircuitStats:
    """Estimate circuit resources without building the full circuit.

    Parameters
    ----------
    K : array (n, n)
        Coupling matrix whose non-zero upper-triangle entries define exchange
        interactions.
    omega : array (n,)
        Natural-frequency vector retained for parity with the circuit builder.
    t : float
        Total evolution time. It is validated but does not change gate counts.
    trotter_reps : int
        Trotter steps per dissipation round.
    gamma : float
        Amplitude-damping rate recorded in the returned estimate.
    n_dissipation_steps : int
        Number of dissipation rounds and ancilla-reset cycles.

    Returns
    -------
    AncillaCircuitStats
        System/ancilla qubit counts, controlled-X and reset counts, total gate
        estimate, dissipation-step count, and damping rate.

    Raises
    ------
    ValueError
        If the evolution time or damping rate is non-finite or negative, or if
        either step count is non-positive.

    """
    _validate_parameters(t, trotter_reps, gamma, n_dissipation_steps)
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
