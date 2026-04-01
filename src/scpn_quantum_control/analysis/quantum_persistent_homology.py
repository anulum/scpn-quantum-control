# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Persistent Homology
"""Persistent homology on quantum measurement data.

Bridges quantum hardware → topological data analysis. Extracts a
correlation distance matrix from measurement counts and computes
persistent homology to detect the synchronization transition.

The classical version of this pipeline (PH on Kuramoto simulations)
was published in Scientific Reports (2025), s41598-025-27083-w.
The quantum version — PH on quantum measurement outcomes — is new.

Pipeline:
    1. Hardware measurement counts (X, Y bases) → correlation matrix
    2. Correlation matrix → distance matrix d_ij = 1 - |C_ij|
    3. Distance matrix → Vietoris-Rips persistent homology (ripser)
    4. H1 persistence diagram → p_h1 (synchronization indicator)
    5. Compare quantum p_h1 vs classical p_h1 at same parameters

When the system is synchronized:
    - Correlation matrix is nearly rank-1 (all-to-all)
    - Distance matrix has small entries (close to 0)
    - Few persistent 1-cycles (vortices)
    - p_h1 ≈ 0

When the system is incoherent:
    - Correlation matrix has partial structure
    - Distance matrix has varied entries
    - Many persistent 1-cycles
    - p_h1 > 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ripser import ripser  # type: ignore[import-untyped]

    _RIPSER_AVAILABLE = True
except ImportError:
    _RIPSER_AVAILABLE = False


@dataclass
class QuantumPHResult:
    """Persistent homology result from quantum measurement data."""

    n_qubits: int
    p_h1: float
    n_h1_persistent: int
    n_h1_total: int
    h1_lifetimes: list[float]
    h0_components: int
    correlation_matrix: np.ndarray
    distance_matrix: np.ndarray


def correlation_matrix_from_counts(
    x_counts: dict[str, int],
    y_counts: dict[str, int],
    n_qubits: int,
) -> np.ndarray:
    """Build qubit-qubit correlation matrix from X and Y basis measurements.

    C_ij = ⟨X_iX_j⟩ + ⟨Y_iY_j⟩

    This equals 2cos(θ_i - θ_j) for states with well-defined phases,
    making it the natural quantum analog of the classical phase
    correlation used in persistent homology.
    """
    xx = _correlator_from_counts(x_counts, n_qubits)
    yy = _correlator_from_counts(y_counts, n_qubits)
    result: np.ndarray = xx + yy
    return result


def _correlator_from_counts(counts: dict[str, int], n_qubits: int) -> np.ndarray:
    """Compute ⟨Z_iZ_j⟩ from measurement counts in a given basis."""
    total = sum(counts.values())
    if total == 0:
        zeros: np.ndarray = np.zeros((n_qubits, n_qubits))
        return zeros

    corr: np.ndarray = np.zeros((n_qubits, n_qubits))
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        n = min(n_qubits, len(bits))
        vals = np.array([1 - 2 * int(bits[-(q + 1)]) for q in range(n)])
        corr += count * np.outer(vals, vals)
    corr /= total
    result: np.ndarray = corr
    return result


def correlation_to_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix for PH.

    d_ij = 1 - |C_ij| / max(|C|)

    Normalized so fully correlated pairs → distance 0,
    uncorrelated pairs → distance 1.
    """
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)
    max_corr: float = float(np.max(abs_corr))
    if max_corr < 1e-15:
        ones_minus_eye: np.ndarray = np.ones_like(corr) - np.eye(corr.shape[0])
        return ones_minus_eye

    dist: np.ndarray = 1.0 - abs_corr / max_corr
    np.fill_diagonal(dist, 0.0)
    return dist


def quantum_persistent_homology(
    x_counts: dict[str, int],
    y_counts: dict[str, int],
    n_qubits: int,
    persistence_threshold: float = 0.1,
) -> QuantumPHResult:
    """Full pipeline: hardware counts → persistent homology.

    Args:
        x_counts: Measurement counts in X basis.
        y_counts: Measurement counts in Y basis.
        n_qubits: Number of qubits.
        persistence_threshold: Minimum H1 lifetime to count as persistent.

    Returns:
        QuantumPHResult with p_h1 and full persistence data.
    """
    if not _RIPSER_AVAILABLE:
        raise ImportError("ripser not installed: pip install ripser")

    corr = correlation_matrix_from_counts(x_counts, y_counts, n_qubits)
    dist = correlation_to_distance(corr)

    result = ripser(dist, maxdim=1, distance_matrix=True)

    # H0
    h0_dgm = result["dgms"][0]
    h0_components = len(h0_dgm)

    # H1
    h1_dgm = result["dgms"][1]
    h1_lifetimes_all = [float(d - b) for b, d in h1_dgm if np.isfinite(d)]
    h1_persistent = [lt for lt in h1_lifetimes_all if lt > persistence_threshold]

    max_h1 = max((n_qubits - 1) * (n_qubits - 2) // 2, 1)
    p_h1 = len(h1_persistent) / max_h1

    return QuantumPHResult(
        n_qubits=n_qubits,
        p_h1=float(p_h1),
        n_h1_persistent=len(h1_persistent),
        n_h1_total=len(h1_lifetimes_all),
        h1_lifetimes=h1_lifetimes_all,
        h0_components=h0_components,
        correlation_matrix=corr,
        distance_matrix=dist,
    )


def compare_quantum_classical_ph(
    x_counts: dict[str, int],
    y_counts: dict[str, int],
    n_qubits: int,
    K: np.ndarray,
    omega: np.ndarray,
    t: float = 1.0,
    persistence_threshold: float = 0.1,
) -> dict:
    """Compare quantum vs classical persistent homology at same parameters.

    Runs the quantum PH pipeline on hardware counts and the classical
    PH pipeline on exact Kuramoto evolution at the same K, omega, t.

    Returns dict with both results and the delta.
    """
    from ..hardware.classical import classical_kuramoto_reference
    from .persistent_homology import compute_persistence

    # Quantum path
    q_result = quantum_persistent_homology(x_counts, y_counts, n_qubits, persistence_threshold)

    # Classical path: evolve Kuramoto ODE, extract final phases
    cl = classical_kuramoto_reference(n_qubits, t_max=t, dt=0.01, K=K, omega=omega)
    theta_final = cl["theta"][-1]
    cl_result = compute_persistence(theta_final, persistence_threshold)

    return {
        "quantum_p_h1": q_result.p_h1,
        "quantum_n_h1": q_result.n_h1_persistent,
        "classical_p_h1": cl_result.p_h1,
        "classical_n_h1": cl_result.n_h1,
        "delta_p_h1": q_result.p_h1 - cl_result.p_h1,
        "quantum_result": q_result,
        "classical_result": cl_result,
    }


def ph_sync_scan(
    x_counts_list: list[dict[str, int]],
    y_counts_list: list[dict[str, int]],
    n_qubits: int,
    K_base_values: np.ndarray,
    persistence_threshold: float = 0.1,
) -> dict:
    """Scan p_h1 across coupling strengths from hardware data.

    Takes lists of measurement counts at different K_base values
    (from sync_threshold experiment or similar) and computes p_h1
    at each coupling strength.

    Returns dict with K_base values and corresponding p_h1 values,
    suitable for plotting the topological phase diagram.
    """
    p_h1_values = []
    n_h1_values = []

    for x_counts, y_counts in zip(x_counts_list, y_counts_list):
        result = quantum_persistent_homology(x_counts, y_counts, n_qubits, persistence_threshold)
        p_h1_values.append(result.p_h1)
        n_h1_values.append(result.n_h1_persistent)

    return {
        "K_base": list(K_base_values),
        "p_h1": p_h1_values,
        "n_h1": n_h1_values,
        "n_qubits": n_qubits,
    }
