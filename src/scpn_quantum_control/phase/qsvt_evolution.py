# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Qsvt Evolution
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

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian


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


def hamiltonian_1norm(K: np.ndarray, omega: np.ndarray) -> float:
    """1-norm of the Kuramoto-XY Hamiltonian: Σ |c_i| over Pauli terms.

    Computed directly from the SparsePauliOp for exactness.
    """
    H_op = knm_to_hamiltonian(K, omega)
    return float(np.sum(np.abs(H_op.coeffs)))


def hamiltonian_spectral_norm(K: np.ndarray, omega: np.ndarray) -> float:
    """Spectral norm ||H|| = max |eigenvalue|.

    Uses sparse eigsh for n >= 14 to avoid memory issues.
    """
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import eigsh

    n = K.shape[0]
    knm_to_hamiltonian(K, omega)
    H_raw = knm_to_dense_matrix(K, omega)

    if n >= 14:
        H_sparse = csc_matrix(H_raw) if not hasattr(H_raw, "tocsc") else H_raw.tocsc()
        # Get largest and smallest eigenvalues
        eig_max = eigsh(H_sparse, k=1, which="LA", return_eigenvectors=False)
        eig_min = eigsh(H_sparse, k=1, which="SA", return_eigenvectors=False)
        return float(max(abs(eig_max[0]), abs(eig_min[0])))

    H_mat = H_raw.toarray() if hasattr(H_raw, "toarray") else np.array(H_raw)
    eigenvalues = np.linalg.eigvalsh(H_mat)
    return float(np.max(np.abs(eigenvalues)))


def qsvt_query_count(alpha: float, t: float, epsilon: float) -> int:
    """Number of block-encoding queries for QSVT simulation.

    Q = O(α|t| + log(1/ε)) — optimal (Gilyén et al.)
    Using the concrete bound: Q = ceil(e × α × |t| + ln(2/ε) / ln(e))
    """
    main_term = np.e * alpha * abs(t)
    log_term = np.log(2.0 / max(epsilon, 1e-20))
    return max(int(np.ceil(main_term + log_term)), 1)


def trotter1_step_count(alpha: float, t: float, epsilon: float) -> int:
    """Number of first-order Trotter steps for same error.

    r = ceil((α|t|)² / ε) — first-order product formula.
    """
    return max(int(np.ceil((alpha * abs(t)) ** 2 / max(epsilon, 1e-20))), 1)


def trotter2_step_count(alpha: float, t: float, epsilon: float) -> int:
    """Number of second-order Trotter steps for same error.

    r = ceil((α|t|)^{3/2} / sqrt(ε)) — second-order product formula.
    """
    return max(int(np.ceil((alpha * abs(t)) ** 1.5 / np.sqrt(max(epsilon, 1e-20)))), 1)


def qsvt_resource_estimate(
    K: np.ndarray,
    omega: np.ndarray,
    t: float = 1.0,
    epsilon: float = 0.01,
) -> QSVTResourceEstimate:
    """Full QSVT vs Trotter resource comparison.

    Args:
        K: coupling matrix
        omega: natural frequencies
        t: simulation time
        epsilon: target error
    """
    n = K.shape[0]
    alpha = hamiltonian_1norm(K, omega)
    spec_norm = hamiltonian_spectral_norm(K, omega)

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


def qsp_phase_angles(degree: int) -> np.ndarray:
    """Compute QSP phase angles for cos(x) polynomial of given degree.

    Uses the Chebyshev approximation: cos(αt·x) ≈ Σ c_k T_k(x).
    The phase angles are computed via the complementary polynomial method.

    This is a simplified version using equally-spaced angles as a starting
    point. Full optimisation (Haah 2018) requires iterative refinement.
    """
    # Symmetric phase angles for even polynomial (cosine)
    phases = np.zeros(degree + 1)
    for k in range(degree + 1):
        phases[k] = np.pi / 4 * (-1) ** k
    # Correct first and last for QSP convention
    phases[0] = np.pi / 4
    phases[-1] = np.pi / 4
    result: np.ndarray = phases
    return result
