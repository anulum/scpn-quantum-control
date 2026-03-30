# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Sync Witness
"""Quantum synchronization witness operators.

A synchronization witness W is a Hermitian observable such that:

    ⟨W⟩ < 0  →  system is synchronized (collective phase coherence)
    ⟨W⟩ ≥ 0  →  system is incoherent

This is analogous to entanglement witnesses (Horodecki et al., 1996)
but detects collective synchronization instead of quantum correlations.

Three witness constructions are provided:

1. **Correlation witness** W_corr = R_c·I - (1/N²)Σ_{ij}(X_iX_j + Y_iY_j)
   Threshold R_c separates synchronized from incoherent.
   Measurable with 2-qubit correlators (no tomography needed).

2. **Fiedler witness** W_F = λ₂_c·I - L̃(ρ)
   Based on the algebraic connectivity (2nd smallest eigenvalue of
   the quantum correlation Laplacian). λ₂ > 0 indicates connected
   synchronization; the witness fires when λ₂ exceeds threshold.

3. **Topological witness** W_top = p_c·I - P̂_H1
   Based on persistent homology H1 cycle count. Fires when the
   fraction of persistent 1-cycles exceeds threshold, indicating
   vortex-free (synchronized) topology.

All three witnesses are:
- Hermitian (self-adjoint)
- Efficiently measurable on NISQ hardware
- Calibratable against classical Kuramoto simulations

Reference: Prior quantum sync measures: Ameri et al., PRA 91, 012301 (2015);
Ma et al., arXiv:2005.09001 (2020). Entanglement witnesses: Horodecki et al.,
PLA 223, 1 (1996). Sync-entanglement: Galve et al., Sci. Rep. 3, 1 (2013).
This module's contribution: NISQ-hardware-ready witness trio with calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WitnessResult:
    """Evaluation result of a synchronization witness."""

    witness_name: str
    expectation_value: float
    threshold: float
    is_synchronized: bool
    raw_observable: float
    n_qubits: int


# ---------------------------------------------------------------------------
# Witness 1: Correlation witness from XY correlators
# ---------------------------------------------------------------------------


def correlation_witness_from_counts(
    x_counts: dict[str, int],
    y_counts: dict[str, int],
    n_qubits: int,
    threshold: float = 0.0,
) -> WitnessResult:
    """Evaluate correlation-based synchronization witness from hardware counts.

    W_corr = R_c - (1/M) Σ_{i<j} (⟨X_iX_j⟩ + ⟨Y_iY_j⟩)

    where M = N(N-1)/2 is the number of pairs and R_c is the threshold.
    The mean two-point XY correlator measures collective phase alignment.

    When oscillators are phase-locked, ⟨X_iX_j⟩ + ⟨Y_iY_j⟩ → 2cos(θ_i-θ_j)
    which is large and positive. The witness fires (⟨W⟩ < 0) when the
    average correlator exceeds threshold.
    """
    xx_corr = _two_point_correlator(x_counts, n_qubits)
    yy_corr = _two_point_correlator(y_counts, n_qubits)

    n_pairs = n_qubits * (n_qubits - 1) // 2
    if n_pairs == 0:
        return WitnessResult("correlation", threshold, threshold, False, 0.0, n_qubits)

    mean_xy_corr = (np.sum(np.triu(xx_corr, k=1)) + np.sum(np.triu(yy_corr, k=1))) / n_pairs
    witness_val = threshold - mean_xy_corr

    return WitnessResult(
        witness_name="correlation",
        expectation_value=witness_val,
        threshold=threshold,
        is_synchronized=witness_val < 0,
        raw_observable=float(mean_xy_corr),
        n_qubits=n_qubits,
    )


def _two_point_correlator(counts: dict[str, int], n_qubits: int) -> np.ndarray:
    """Compute ⟨Z_iZ_j⟩ correlator matrix from measurement counts.

    Since we measure in a rotated basis (X or Y), Z_iZ_j in that
    basis gives the XX or YY correlator in the original basis.
    """
    total = sum(counts.values())
    corr = np.zeros((n_qubits, n_qubits))

    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        vals = np.array([1 - 2 * int(bits[-(q + 1)]) for q in range(min(n_qubits, len(bits)))])
        corr += count * np.outer(vals, vals)

    corr /= total
    result: np.ndarray = corr
    return result


# ---------------------------------------------------------------------------
# Witness 2: Fiedler (algebraic connectivity) witness
# ---------------------------------------------------------------------------


def fiedler_witness_from_correlator(
    corr_matrix: np.ndarray,
    threshold: float = 0.0,
) -> WitnessResult:
    """Evaluate Fiedler-based synchronization witness.

    W_F = λ₂_c - λ₂(L)

    where L is the Laplacian of the correlation matrix and λ₂ is the
    algebraic connectivity (Fiedler eigenvalue). λ₂ > 0 indicates
    a connected correlation graph; larger λ₂ means stronger collective
    synchronization.

    The correlation matrix C_ij = ⟨X_iX_j⟩ + ⟨Y_iY_j⟩ is interpreted
    as an adjacency matrix. The Laplacian L = D - C where D_ii = Σ_j C_ij.
    """
    n = corr_matrix.shape[0]
    # Ensure non-negative weights for Laplacian
    adj = np.maximum(corr_matrix, 0.0)
    np.fill_diagonal(adj, 0.0)
    degree = np.diag(np.sum(adj, axis=1))
    laplacian = degree - adj

    eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
    lambda_2 = float(eigenvalues[1]) if n > 1 else 0.0

    witness_val = threshold - lambda_2

    return WitnessResult(
        witness_name="fiedler",
        expectation_value=witness_val,
        threshold=threshold,
        is_synchronized=witness_val < 0,
        raw_observable=lambda_2,
        n_qubits=n,
    )


def fiedler_witness_from_counts(
    x_counts: dict[str, int],
    y_counts: dict[str, int],
    n_qubits: int,
    threshold: float = 0.0,
) -> WitnessResult:
    """Convenience: build Fiedler witness directly from hardware counts."""
    xx = _two_point_correlator(x_counts, n_qubits)
    yy = _two_point_correlator(y_counts, n_qubits)
    corr_matrix = xx + yy
    return fiedler_witness_from_correlator(corr_matrix, threshold)


# ---------------------------------------------------------------------------
# Witness 3: Topological witness via persistent homology
# ---------------------------------------------------------------------------


def topological_witness_from_correlator(
    corr_matrix: np.ndarray,
    threshold: float = 0.5,
    max_dim: int = 1,
) -> WitnessResult:
    """Evaluate topological synchronization witness via persistent H1.

    W_top = p_c - p_H1

    where p_H1 = (# persistent 1-cycles) / (max possible 1-cycles)
    computed from the Vietoris-Rips complex of the correlation distance
    matrix d_ij = 1 - |C_ij|.

    In the synchronized phase, the correlation matrix is nearly
    rank-1 (all-to-all connected), so there are no persistent 1-cycles
    (p_H1 ≈ 0). In the incoherent phase, partial correlations create
    holes (p_H1 > 0). The witness fires when p_H1 drops below threshold.

    Note: inverted polarity — low p_H1 = synchronized. So:
    W_top = p_H1 - p_c → negative when p_H1 < p_c (synchronized).
    """
    n = corr_matrix.shape[0]

    try:
        import ripser
    except ImportError:
        return WitnessResult("topological", float("nan"), threshold, False, float("nan"), n)

    # Distance matrix from correlations
    abs_corr = np.abs(corr_matrix)
    np.fill_diagonal(abs_corr, 1.0)
    dist = 1.0 - abs_corr

    result = ripser.ripser(dist, maxdim=max_dim, distance_matrix=True)
    dgms = result["dgms"]

    if len(dgms) > 1 and len(dgms[1]) > 0:
        h1 = dgms[1]
        # Persistent = death - birth > median lifetime
        lifetimes = h1[:, 1] - h1[:, 0]
        median_lt = np.median(lifetimes) if len(lifetimes) > 0 else 0.0
        n_persistent = int(np.sum(lifetimes > median_lt))
        max_h1 = n * (n - 1) // 2  # upper bound on 1-cycles
        p_h1 = n_persistent / max(max_h1, 1)
    else:
        p_h1 = 0.0

    # Low p_H1 = synchronized → witness = p_H1 - threshold → negative when synced
    witness_val = p_h1 - threshold

    return WitnessResult(
        witness_name="topological",
        expectation_value=witness_val,
        threshold=threshold,
        is_synchronized=witness_val < 0,
        raw_observable=p_h1,
        n_qubits=n,
    )


# ---------------------------------------------------------------------------
# Combined witness evaluation
# ---------------------------------------------------------------------------


def evaluate_all_witnesses(
    x_counts: dict[str, int],
    y_counts: dict[str, int],
    n_qubits: int,
    corr_threshold: float = 0.0,
    fiedler_threshold: float = 0.0,
    topo_threshold: float = 0.5,
) -> dict[str, WitnessResult]:
    """Evaluate all three synchronization witnesses from hardware counts.

    Returns dict keyed by witness name.
    """
    xx = _two_point_correlator(x_counts, n_qubits)
    yy = _two_point_correlator(y_counts, n_qubits)
    corr_matrix = xx + yy

    return {
        "correlation": correlation_witness_from_counts(
            x_counts, y_counts, n_qubits, corr_threshold
        ),
        "fiedler": fiedler_witness_from_correlator(corr_matrix, fiedler_threshold),
        "topological": topological_witness_from_correlator(corr_matrix, topo_threshold),
    }


def calibrate_thresholds(
    K: np.ndarray,
    omega: np.ndarray,
    K_base_range: np.ndarray | None = None,
    n_samples: int = 20,
) -> dict[str, float]:
    """Calibrate witness thresholds from classical Kuramoto simulation.

    Runs classical ODE at multiple coupling strengths and finds the
    threshold value of each observable at the synchronization transition.
    The transition point is where the order parameter R crosses 0.5.

    Returns dict of {witness_name: optimal_threshold}.
    """
    from ..hardware.classical import classical_kuramoto_reference

    n = K.shape[0]
    if K_base_range is None:
        K_base_range = np.linspace(0.0, 2.0, n_samples)

    corr_vals = []
    fiedler_vals = []
    R_vals = []

    for k_base in K_base_range:
        K_scaled = K * k_base
        result = classical_kuramoto_reference(n, t_max=2.0, dt=0.1, K=K_scaled, omega=omega)
        R_final = float(result["R"][-1])
        R_vals.append(R_final)

        theta_final = result["theta"][-1]
        cos_diff = np.cos(np.subtract.outer(theta_final, theta_final))
        corr_matrix = cos_diff

        n_pairs = n * (n - 1) // 2
        mean_corr = np.sum(np.triu(corr_matrix, k=1)) / max(n_pairs, 1)
        corr_vals.append(mean_corr)

        adj = np.maximum(corr_matrix, 0.0)
        np.fill_diagonal(adj, 0.0)
        degree = np.diag(np.sum(adj, axis=1))
        lap = degree - adj
        eigs = np.sort(np.linalg.eigvalsh(lap))
        fiedler_vals.append(float(eigs[1]) if n > 1 else 0.0)

    R_arr: np.ndarray = np.array(R_vals)

    # Find transition index (R crosses 0.5)
    above = R_arr >= 0.5
    if np.any(above) and not np.all(above):
        trans_idx = int(np.argmax(above))
    else:
        trans_idx = len(R_vals) // 2

    return {
        "correlation": float(corr_vals[trans_idx]),
        "fiedler": float(fiedler_vals[trans_idx]),
        "topological": 0.5,  # empirical default for p_H1
    }
