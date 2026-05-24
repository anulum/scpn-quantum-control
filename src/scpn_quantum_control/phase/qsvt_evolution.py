# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Qsvt Evolution
"""QSVT-based Hamiltonian simulation for the Kuramoto-XY model.

Quantum Singular Value Transformation (Gilyén et al., STOC 2019)
achieves optimal Hamiltonian simulation with query complexity:

    O(α × |t| + log(1/ε))

vs product-formula (Trotter) complexity:

    O((α × |t|)^{1+1/2p} / ε^{1/2p})

where α = ||H||, t = simulation time, ε = error, p = Trotter order.

For the Kuramoto-XY Hamiltonian with n oscillators:
    α = Σ_{ij} |K_ij| + Σ_i |ω_i|  (1-norm of Pauli coefficients)

QSVT achieves this via polynomial approximation of cos(Ht) and sin(Ht)
using Chebyshev polynomials, embedded into a quantum signal processing
(QSP) circuit.

This module provides:
    1. Spectral norm and 1-norm computation for the XY Hamiltonian
    2. QSVT resource estimation (gate count, circuit depth)
    3. Comparison with Trotter at matched error budget
    4. Phase angle computation for QSP convention

Note: Full QSVT circuit construction requires a block-encoding oracle,
which is hardware-dependent. This module provides the resource estimates
and phase angles; circuit compilation is deferred to hardware backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Any

import numpy as np

from ..bridge.knm_hamiltonian import (
    knm_to_dense_matrix,
    knm_to_hamiltonian,
    knm_to_sparse_matrix,
)
from ..dense_budget import require_dense_allocation


@dataclass
class QSVTResourceEstimate:
    """QSVT resource estimation for Hamiltonian simulation."""

    alpha: float  # 1-norm of Hamiltonian (block-encoding normalisation)
    spectral_norm: float  # ||H|| (largest eigenvalue magnitude)
    simulation_time: float
    target_error: float
    qsvt_queries: int  # number of block-encoding queries
    trotter1_steps: int  # equivalent first-order Trotter steps
    trotter2_steps: int  # equivalent second-order Trotter steps
    speedup_vs_trotter1: float
    speedup_vs_trotter2: float
    n_qubits: int
    n_ancilla_qsvt: int  # ancilla qubits for block encoding


def _as_real_numeric_array(name: str, values: object) -> np.ndarray:
    """Return a real numeric array without implicit string/bool/object coercion."""
    try:
        raw = np.asarray(values)
    except ValueError as exc:
        raise ValueError(f"{name} must be a rectangular numeric array.") from exc

    if raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must contain real numeric scalars.")
    try:
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric scalars.") from exc


def _as_real_scalar(name: str, value: object) -> float:
    """Return an explicit real numeric scalar without string/bool coercion."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real numeric scalar.")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must be a real numeric scalar.")
    return float(raw)


def _validate_problem_inputs(K: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    K_arr = _as_real_numeric_array("K", K)
    omega_arr = _as_real_numeric_array("omega", omega)
    if K_arr.ndim != 2 or K_arr.shape[0] != K_arr.shape[1]:
        raise ValueError("K must be a square two-dimensional coupling matrix.")
    if omega_arr.ndim != 1:
        raise ValueError("omega must be a one-dimensional natural-frequency vector.")
    if omega_arr.shape[0] != K_arr.shape[0]:
        raise ValueError("omega length must match the coupling matrix dimension.")
    if not np.all(np.isfinite(K_arr)) or not np.all(np.isfinite(omega_arr)):
        raise ValueError("K and omega must contain only finite values.")
    if not np.allclose(K_arr, K_arr.T, rtol=1e-10, atol=1e-12):
        raise ValueError("K must be symmetric for a Hermitian Kuramoto-XY Hamiltonian.")
    return K_arr, omega_arr


def _validate_resource_budget(
    alpha: float, t: float, epsilon: float
) -> tuple[float, float, float]:
    alpha_value = _as_real_scalar("alpha", alpha)
    time_value = _as_real_scalar("simulation time", t)
    epsilon_value = _as_real_scalar("epsilon", epsilon)
    if not np.isfinite(alpha_value) or alpha_value <= 0.0:
        raise ValueError("alpha must be finite and strictly positive.")
    if not np.isfinite(time_value) or time_value < 0.0:
        raise ValueError("simulation time must be finite and non-negative.")
    if not np.isfinite(epsilon_value) or not 0.0 < epsilon_value < 1.0:
        raise ValueError("epsilon must be finite and satisfy 0 < epsilon < 1.")
    return alpha_value, time_value, epsilon_value


def hamiltonian_1norm(K: np.ndarray, omega: np.ndarray) -> float:
    """1-norm of the Kuramoto-XY Hamiltonian: Σ |c_i| over Pauli terms.

    Computed directly from the SparsePauliOp for exactness.
    """
    K, omega = _validate_problem_inputs(K, omega)
    H_op = knm_to_hamiltonian(K, omega)
    return float(np.sum(np.abs(H_op.coeffs)))


def hamiltonian_spectral_norm(
    K: np.ndarray,
    omega: np.ndarray,
    *,
    max_dense_gib: float | None = None,
) -> float:
    """Spectral norm ||H|| = max |eigenvalue|.

    Uses sparse eigsh for n >= 14 to avoid dense 2^n x 2^n allocation.
    """
    from scipy.sparse.linalg import eigsh

    K, omega = _validate_problem_inputs(K, omega)
    n = K.shape[0]

    if n >= 14:
        H_sparse = knm_to_sparse_matrix(K, omega)
        # Get largest and smallest eigenvalues
        eig_max = eigsh(H_sparse, k=1, which="LA", return_eigenvectors=False)
        eig_min = eigsh(H_sparse, k=1, which="SA", return_eigenvectors=False)
        return float(max(abs(eig_max[0]), abs(eig_min[0])))

    knm_to_hamiltonian(K, omega)
    require_dense_allocation(
        n,
        rank=2,
        object_count=2,
        max_gib=max_dense_gib,
        label="QSVT dense spectral norm",
    )
    H_mat = knm_to_dense_matrix(K, omega, max_dense_gib=max_dense_gib)
    eigenvalues = np.linalg.eigvalsh(H_mat)
    return float(np.max(np.abs(eigenvalues)))


def qsvt_query_count(alpha: float, t: float, epsilon: float) -> int:
    """Number of block-encoding queries for QSVT simulation.

    Q = O(α|t| + log(1/ε)) — optimal (Gilyén et al.)
    Using the concrete bound: Q = ceil(e × α × |t| + ln(2/ε) / ln(e))
    """
    alpha, t, epsilon = _validate_resource_budget(alpha, t, epsilon)
    main_term = np.e * alpha * t
    log_term = np.log(2.0 / epsilon)
    return max(int(np.ceil(main_term + log_term)), 1)


def trotter1_step_count(alpha: float, t: float, epsilon: float) -> int:
    """Number of first-order Trotter steps for same error.

    r = ceil((α|t|)² / ε) — first-order product formula.
    """
    alpha, t, epsilon = _validate_resource_budget(alpha, t, epsilon)
    return max(int(np.ceil((alpha * t) ** 2 / epsilon)), 1)


def trotter2_step_count(alpha: float, t: float, epsilon: float) -> int:
    """Number of second-order Trotter steps for same error.

    r = ceil((α|t|)^{3/2} / sqrt(ε)) — second-order product formula.
    """
    alpha, t, epsilon = _validate_resource_budget(alpha, t, epsilon)
    return max(int(np.ceil((alpha * t) ** 1.5 / np.sqrt(epsilon))), 1)


def qsvt_resource_estimate(
    K: np.ndarray,
    omega: np.ndarray,
    t: float = 1.0,
    epsilon: float = 0.01,
    *,
    max_dense_gib: float | None = None,
) -> QSVTResourceEstimate:
    """Full QSVT vs Trotter resource comparison.

    Args:
        K: coupling matrix
        omega: natural frequencies
        t: simulation time
        epsilon: target error
    """
    K, omega = _validate_problem_inputs(K, omega)
    _, t, epsilon = _validate_resource_budget(1.0, t, epsilon)
    n = K.shape[0]
    alpha = hamiltonian_1norm(K, omega)
    spec_norm = hamiltonian_spectral_norm(K, omega, max_dense_gib=max_dense_gib)

    q_qsvt = qsvt_query_count(alpha, t, epsilon)
    r_t1 = trotter1_step_count(alpha, t, epsilon)
    r_t2 = trotter2_step_count(alpha, t, epsilon)

    speedup_t1 = r_t1 / max(q_qsvt, 1)
    speedup_t2 = r_t2 / max(q_qsvt, 1)

    # Block encoding requires 1 ancilla qubit + log2(n_terms) selection qubits
    n_terms = n * (n - 1) + n  # XX+YY pairs + Z fields
    n_ancilla = 1 + max(int(np.ceil(np.log2(max(n_terms, 1)))), 1)

    return QSVTResourceEstimate(
        alpha=alpha,
        spectral_norm=spec_norm,
        simulation_time=t,
        target_error=epsilon,
        qsvt_queries=q_qsvt,
        trotter1_steps=r_t1,
        trotter2_steps=r_t2,
        speedup_vs_trotter1=speedup_t1,
        speedup_vs_trotter2=speedup_t2,
        n_qubits=n,
        n_ancilla_qsvt=n_ancilla,
    )


def qsp_phase_angles(degree: int, *, allow_initial_guess: bool = False) -> np.ndarray:
    """Return QSP phase angles for a cosine polynomial only when explicit.

    Production QSP phase synthesis requires a complementary-polynomial
    optimisation/verification routine. That implementation is not wired here,
    so the function fails by default rather than returning unverified seed angles.

    Set ``allow_initial_guess=True`` only when a caller needs the historical
    symmetric seed angles for an offline optimiser. Those angles are not valid
    compiled QSP phases and must not be used for resource or hardware claims.
    """
    degree_value = _validate_non_negative_integer(degree, "degree")
    if not allow_initial_guess:
        raise NotImplementedError(
            "QSP phase synthesis is not implemented. Pass allow_initial_guess=True "
            "only to obtain non-production seed angles for an external optimiser."
        )

    # Symmetric phase angles for even polynomial (cosine)
    phases = np.zeros(degree_value + 1)
    for k in range(degree_value + 1):
        phases[k] = np.pi / 4 * (-1) ** k
    # Correct first and last for QSP convention
    phases[0] = np.pi / 4
    phases[-1] = np.pi / 4
    result: np.ndarray = phases
    return result


def _validate_non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        integer_value = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if integer_value < 0:
        raise ValueError(f"{name} must be non-negative, got {integer_value}")
    return int(integer_value)
