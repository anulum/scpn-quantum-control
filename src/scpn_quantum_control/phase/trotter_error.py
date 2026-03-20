# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Trotter error analysis for the XY Kuramoto Hamiltonian.

Provides both empirical (Frobenius norm) and analytical (commutator bound)
error estimates for Lie-Trotter and Suzuki-Trotter decompositions.

Analytical bound (Childs et al., PRX 11, 011020, 2021):
    ||U_exact - U_trotter|| <= (t²/2r) × ||[H_XY, H_Z]||

For the Kuramoto XY Hamiltonian:
    [H_XY, H_Z] = 2i Σ_{i<j} K_ij(ω_j - ω_i)(Y_iX_j - X_iY_j)

Key insight: Trotter error vanishes when all frequencies are equal.
Error scales with K × Δω, not K alone.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..bridge.knm_hamiltonian import knm_to_hamiltonian


def trotter_error_norm(
    K: np.ndarray,
    omega: np.ndarray,
    t: float,
    reps: int,
) -> float:
    """||U_exact - U_trotter||_F for a single (t, reps) pair.

    Raises ValueError for n > 10 (2^10 = 1024, matrices get large fast).
    """
    n = len(omega)
    if n > 10:
        raise ValueError(f"n={n} too large for exact unitary comparison (max 10)")

    H_op = knm_to_hamiltonian(K, omega)
    H_mat = np.array(H_op.to_matrix())

    U_exact = expm(-1j * H_mat * t)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import Operator
    from qiskit.synthesis import LieTrotter

    qc = QuantumCircuit(n)
    evo = PauliEvolutionGate(H_op, time=t, synthesis=LieTrotter(reps=reps))
    qc.append(evo, range(n))
    # Decompose so Operator sees individual gates, not the exact PauliEvolutionGate
    qc_decomposed = qc.decompose(reps=2)
    U_trotter = Operator(qc_decomposed).data

    return float(np.linalg.norm(U_exact - U_trotter, "fro"))


def trotter_error_sweep(
    K: np.ndarray,
    omega: np.ndarray,
    t_values: list[float],
    reps_values: list[int],
) -> dict:
    """2D sweep of Trotter error over (t, reps) pairs.

    Returns dict with keys 't_values', 'reps_values', 'errors' (2D list).
    """
    errors = []
    for t in t_values:
        row = []
        for reps in reps_values:
            row.append(trotter_error_norm(K, omega, t, reps))
        errors.append(row)

    return {
        "t_values": t_values,
        "reps_values": reps_values,
        "errors": errors,
    }


def commutator_norm_bound(K: np.ndarray, omega: np.ndarray) -> float:
    """Compute ||[H_XY, H_Z]|| analytically.

    For the XY-Z split:
        [H_XY, H_Z] = 2i Σ_{i<j} K_ij(ω_j - ω_i)(Y_iX_j - X_iY_j)

    Since ||Y_iX_j - X_iY_j|| = 2 (operator norm), the triangle inequality gives:
        ||[H_XY, H_Z]|| ≤ 4 Σ_{i<j} |K_ij| × |ω_j - ω_i|

    Childs et al., PRX 11, 011020 (2021), Theorem 1.
    """
    n = len(omega)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += abs(K[i, j]) * abs(omega[j] - omega[i])
    return 4.0 * total


def trotter_error_bound(
    K: np.ndarray,
    omega: np.ndarray,
    t: float,
    reps: int,
    order: int = 1,
) -> float:
    """Analytical upper bound on Trotter error.

    For first-order (Lie-Trotter, order=1):
        error ≤ (t²/2r) × ||[H_XY, H_Z]||

    For second-order (Suzuki-Trotter, order=2):
        error ≤ (t³/12r²) × ||[H_XY, [H_XY, H_Z]]|| + ||[H_Z, [H_XY, H_Z]]||
        (approximated as commutator_norm² / ||[H_XY, H_Z]|| for simplicity)

    Returns the upper bound on ||U_exact - U_trotter||.
    """
    gamma = commutator_norm_bound(K, omega)
    if order == 1:
        return (t * t / (2.0 * reps)) * gamma
    if order == 2:
        # Second-order bound: O(t³/r²) with nested commutator
        # Approximate nested commutator norm as gamma² / max(K) for typical cases
        max_k = float(np.max(np.abs(K)))
        gamma_nested = gamma * gamma / max(max_k, 1e-10)
        return (t**3 / (12.0 * reps * reps)) * gamma_nested
    raise ValueError(f"order must be 1 or 2, got {order}")


def optimal_dt(
    K: np.ndarray,
    omega: np.ndarray,
    epsilon: float,
    t_total: float,
    order: int = 1,
) -> dict:
    """Compute optimal Trotter step size for target error epsilon.

    Returns dict with dt, n_steps, error_bound, and the commutator norm Gamma.
    """
    gamma = commutator_norm_bound(K, omega)
    if order == 1:
        # error = (t²/2r) × gamma ≤ epsilon → r ≥ t² × gamma / (2 × epsilon)
        r = max(1, int(np.ceil(t_total * t_total * gamma / (2.0 * epsilon))))
    elif order == 2:
        max_k = float(np.max(np.abs(K)))
        gamma_nested = gamma * gamma / max(max_k, 1e-10)
        r = max(1, int(np.ceil(np.sqrt(t_total**3 * gamma_nested / (12.0 * epsilon)))))
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")

    dt = t_total / r
    bound = trotter_error_bound(K, omega, t_total, r, order)

    return {
        "dt": dt,
        "n_steps": r,
        "error_bound": bound,
        "commutator_norm": gamma,
        "order": order,
        "t_total": t_total,
        "epsilon": epsilon,
    }


def frequency_heterogeneity(omega: np.ndarray) -> float:
    """Measure of frequency spread: mean |ω_i - ω_j| over all pairs.

    Zero when all frequencies are equal (Trotter error vanishes).
    Large when frequencies are heterogeneous (Trotter error grows).
    """
    n = len(omega)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += abs(omega[j] - omega[i])
            count += 1
    return total / max(count, 1)
