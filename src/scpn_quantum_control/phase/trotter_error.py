# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Trotter Error
"""Trotter error analysis for the XY Kuramoto Hamiltonian.

Provides both empirical and analytical spectral-norm error estimates for the
two-group (``H_XY`` then ``H_Z``) Lie-Trotter and symmetric Suzuki-Trotter
product formulas. The empirical measurement and the analytical bound use the
*same* operator splitting and the *same* (spectral / induced 2-) norm, so the
analytical bound rigorously upper-bounds the empirical error.

Analytical first-order bound (Childs et al., PRX 11, 011020, 2021):
    ||U_exact - U_trotter||_2 <= (t²/2r) × ||[H_XY, H_Z]||_2

Analytical second-order (symmetric) bound:
    ||U_exact - U_trotter||_2 <= (t³/12r²) × (||[H_XY,[H_XY,H_Z]]||_2
                                              + ||[H_Z,[H_XY,H_Z]]||_2)

For the Kuramoto XY Hamiltonian ``H = -Σ_{i<j} K_ij(X_iX_j + Y_iY_j) - Σ_i ω_i Z_i``
the inner commutator is
    [H_XY, H_Z] = 2i Σ_{i<j} K_ij(ω_j - ω_i)(Y_iX_j - X_iY_j),

so ``||[H_XY, H_Z]||_2 <= 4 Σ_{i<j} |K_ij| |ω_j - ω_i|`` by the triangle
inequality with ``||Y_iX_j - X_iY_j||_2 = 2``.

Key insight: Trotter error vanishes when all frequencies are equal. Error
scales with K × Δω, not K alone.

Norm choice: the spectral norm is the algorithm-error norm in which the
product-formula bounds of Childs et al. are stated; measuring the empirical
error in the same norm is what makes the bound a genuine upper bound. The
Frobenius norm of the same difference is larger by up to a factor ``√(2^n)``
and is not bounded by these commutator estimates.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from ..bridge.knm_hamiltonian import knm_to_dense_matrix
from ..dense_budget import require_dense_allocation

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
TrotterSweepResult: TypeAlias = dict[str, object]
OptimalStepResult: TypeAlias = dict[str, object]


def _two_group_unitary(
    h_xy: ComplexArray,
    h_z: ComplexArray,
    t: float,
    reps: int,
    order: int,
) -> ComplexArray:
    """Dense unitary of the two-group product formula over ``reps`` steps.

    order 1: ``(e^{-i H_XY τ} e^{-i H_Z τ})^r`` (Lie-Trotter).
    order 2: ``(e^{-i H_XY τ/2} e^{-i H_Z τ} e^{-i H_XY τ/2})^r`` (symmetric Strang).
    """
    tau = t / reps
    if order == 1:
        step = expm(-1j * h_xy * tau) @ expm(-1j * h_z * tau)
    else:
        half_xy = expm(-1j * h_xy * (tau / 2.0))
        full_z = expm(-1j * h_z * tau)
        step = half_xy @ full_z @ half_xy
    return np.linalg.matrix_power(step, reps)


def trotter_error_norm(
    K: FloatArray,
    omega: FloatArray,
    t: float,
    reps: int,
    order: int = 1,
    *,
    max_dense_gib: float | None = None,
) -> float:
    """``||U_exact - U_trotter||_2`` for the two-group product formula.

    Measures the spectral-norm distance between the exact propagator and the
    two-group product formula of the requested ``order`` (1 = Lie-Trotter,
    2 = symmetric Suzuki-Trotter). This is the same splitting and norm assumed
    by :func:`trotter_error_bound`, so that bound upper-bounds this value.

    Raises ValueError for n > 10 (2^10 = 1024, dense matrices grow fast),
    reps < 1, or order not in {1, 2}.
    """
    K_arr: FloatArray = np.asarray(K, dtype=np.float64)
    omega_arr: FloatArray = np.asarray(omega, dtype=np.float64)
    _validate_k_omega(K_arr, omega_arr)
    n = omega_arr.shape[0]
    if n > 10:
        raise ValueError(f"n={n} too large for exact unitary comparison (max 10)")
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")

    require_dense_allocation(
        n,
        dtype=np.complex128,
        rank=2,
        object_count=5,
        max_gib=max_dense_gib,
        label="Trotter dense unitary comparison workspace",
    )
    h_full: ComplexArray = np.asarray(
        knm_to_dense_matrix(K_arr, omega_arr, max_dense_gib=max_dense_gib),
        dtype=np.complex128,
    )
    h_xy: ComplexArray = np.asarray(
        knm_to_dense_matrix(K_arr, np.zeros_like(omega_arr), max_dense_gib=max_dense_gib),
        dtype=np.complex128,
    )
    h_z: ComplexArray = np.asarray(
        knm_to_dense_matrix(np.zeros_like(K_arr), omega_arr, max_dense_gib=max_dense_gib),
        dtype=np.complex128,
    )

    u_exact = expm(-1j * h_full * t)
    u_trotter = _two_group_unitary(h_xy, h_z, t, reps, order)

    return float(np.linalg.norm(u_exact - u_trotter, 2))


def trotter_error_sweep(
    K: FloatArray,
    omega: FloatArray,
    t_values: list[float],
    reps_values: list[int],
    order: int = 1,
    *,
    max_dense_gib: float | None = None,
) -> TrotterSweepResult:
    """2D sweep of two-group Trotter error over (t, reps) pairs.

    Returns dict with keys 't_values', 'reps_values', 'order', 'errors' (2D list).
    """
    errors: list[list[float]] = []
    for t in t_values:
        row: list[float] = []
        for reps in reps_values:
            row.append(trotter_error_norm(K, omega, t, reps, order, max_dense_gib=max_dense_gib))
        errors.append(row)

    return {
        "t_values": t_values,
        "reps_values": reps_values,
        "order": order,
        "errors": errors,
    }


def commutator_norm_bound(K: FloatArray, omega: FloatArray) -> float:
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


def nested_commutator_norm_bound(
    K: FloatArray,
    omega: FloatArray,
    *,
    exact_qubit_limit: int = 8,
    max_dense_gib: float | None = None,
) -> float:
    """Bound the second-order Suzuki nested-commutator contribution.

    The second-order product-formula bound depends on
    ``||[H_XY,[H_XY,H_Z]]|| + ||[H_Z,[H_XY,H_Z]]||``. For small systems this
    function computes that quantity exactly with the spectral norm. For larger
    systems it returns the rigorous submultiplicative upper bound
    ``2 (||H_XY|| + ||H_Z||) ||[H_XY,H_Z]||`` using Pauli coefficient-norm
    bounds, avoiding unsafe dense ``2^n`` allocation.
    """
    K_arr: FloatArray = np.asarray(K, dtype=np.float64)
    omega_arr: FloatArray = np.asarray(omega, dtype=np.float64)
    _validate_k_omega(K_arr, omega_arr)
    n = omega_arr.shape[0]
    if exact_qubit_limit < 0:
        raise ValueError("exact_qubit_limit must be non-negative")

    if n <= exact_qubit_limit:
        require_dense_allocation(
            n,
            dtype=np.complex128,
            rank=2,
            object_count=5,
            max_gib=max_dense_gib,
            label="Trotter nested dense commutator workspace",
        )
        h_xy: ComplexArray = np.asarray(
            knm_to_dense_matrix(K_arr, np.zeros_like(omega_arr), max_dense_gib=max_dense_gib),
            dtype=np.complex128,
        )
        h_z: ComplexArray = np.asarray(
            knm_to_dense_matrix(np.zeros_like(K_arr), omega_arr, max_dense_gib=max_dense_gib),
            dtype=np.complex128,
        )
        comm = h_xy @ h_z - h_z @ h_xy
        nested_xy = h_xy @ comm - comm @ h_xy
        nested_z = h_z @ comm - comm @ h_z
        return float(np.linalg.norm(nested_xy, 2) + np.linalg.norm(nested_z, 2))

    gamma = commutator_norm_bound(K_arr, omega_arr)
    h_xy_norm_bound = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            h_xy_norm_bound += 2.0 * abs(K_arr[i, j])
    h_z_norm_bound = float(np.sum(np.abs(omega_arr)))
    return 2.0 * (h_xy_norm_bound + h_z_norm_bound) * gamma


def trotter_error_bound(
    K: FloatArray,
    omega: FloatArray,
    t: float,
    reps: int,
    order: int = 1,
    *,
    max_dense_gib: float | None = None,
) -> float:
    """Analytical spectral-norm upper bound on two-group Trotter error.

    For first-order (Lie-Trotter, order=1):
        error ≤ (t²/2r) × ||[H_XY, H_Z]||₂

    For second-order (symmetric Suzuki-Trotter, order=2):
        error ≤ (t³/12r²) × ( ||[H_XY,[H_XY,H_Z]]||₂ + ||[H_Z,[H_XY,H_Z]]||₂ )

    Returns the upper bound on ||U_exact - U_trotter||₂ for the matching
    product formula measured by :func:`trotter_error_norm`.
    """
    gamma = commutator_norm_bound(K, omega)
    if order == 1:
        return (t * t / (2.0 * reps)) * gamma
    if order == 2:
        gamma_nested = nested_commutator_norm_bound(K, omega, max_dense_gib=max_dense_gib)
        return (t**3 / (12.0 * reps * reps)) * gamma_nested
    raise ValueError(f"order must be 1 or 2, got {order}")


def optimal_dt(
    K: FloatArray,
    omega: FloatArray,
    epsilon: float,
    t_total: float,
    order: int = 1,
    *,
    max_dense_gib: float | None = None,
) -> OptimalStepResult:
    """Compute optimal Trotter step size for target error epsilon.

    Returns dict with dt, n_steps, error_bound, and the commutator norm Gamma.
    """
    gamma = commutator_norm_bound(K, omega)
    if order == 1:
        # error = (t²/2r) × gamma ≤ epsilon → r ≥ t² × gamma / (2 × epsilon)
        r = max(1, int(np.ceil(t_total * t_total * gamma / (2.0 * epsilon))))
    elif order == 2:
        gamma_nested = nested_commutator_norm_bound(K, omega, max_dense_gib=max_dense_gib)
        r = max(1, int(np.ceil(np.sqrt(t_total**3 * gamma_nested / (12.0 * epsilon)))))
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")

    dt = t_total / r
    bound = trotter_error_bound(K, omega, t_total, r, order, max_dense_gib=max_dense_gib)

    return {
        "dt": dt,
        "n_steps": r,
        "error_bound": bound,
        "commutator_norm": gamma,
        "order": order,
        "t_total": t_total,
        "epsilon": epsilon,
    }


def frequency_heterogeneity(omega: FloatArray) -> float:
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


def _validate_k_omega(K: FloatArray, omega: FloatArray) -> None:
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be a square 2-D matrix, got shape {K.shape}")
    if omega.ndim != 1 or omega.shape[0] != K.shape[0]:
        raise ValueError(f"omega must be 1-D with length {K.shape[0]}, got shape {omega.shape}")
    if not np.all(np.isfinite(K)):
        raise ValueError("K contains non-finite entries")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega contains non-finite entries")
