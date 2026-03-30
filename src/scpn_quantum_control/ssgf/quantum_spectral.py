# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Quantum Spectral
"""Spectral bridge: Fiedler value from quantum phase estimation.

The Fiedler eigenvalue λ_2 of the coupling-weighted graph Laplacian
measures algebraic connectivity — how well-connected the coupling
topology is. In the SSGF, λ_2 determines entrainment stability:
    λ_2 > Δω → synchronisation possible

Classical: O(n³) via dense eigendecomposition.
Quantum: O(poly(n) × 1/ε) via quantum phase estimation (QPE) on
the Laplacian Hamiltonian H_L = L_K (already Hermitian PSD).

This module:
    1. Encodes the graph Laplacian as a Hamiltonian
    2. Estimates λ_2 via QPE (simulated classically for now)
    3. Computes the entrainment stability criterion λ_2 vs Δω
    4. Provides resource estimates for hardware QPE
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis.bkt_analysis import coupling_laplacian, fiedler_eigenvalue


@dataclass
class SpectralBridgeResult:
    """Spectral bridge analysis result."""

    fiedler_value: float  # λ_2
    frequency_spread: float  # max(ω) - min(ω)
    entrainment_stable: bool  # λ_2 > frequency_spread
    stability_margin: float  # λ_2 - Δω (positive = stable)
    laplacian_spectrum: np.ndarray  # all eigenvalues of L_K
    qpe_bits_needed: int  # precision bits for ε resolution
    qpe_circuit_depth: int  # estimated QPE depth


def laplacian_spectrum(K: np.ndarray) -> np.ndarray:
    """Full eigenvalue spectrum of the coupling-weighted Laplacian."""
    L = coupling_laplacian(K)
    eigenvalues: np.ndarray = np.sort(np.linalg.eigvalsh(L))
    return eigenvalues


def entrainment_criterion(
    K: np.ndarray,
    omega: np.ndarray,
) -> tuple[bool, float]:
    """Check if λ_2 > Δω (synchronisation possible).

    Returns (stable, margin) where margin = λ_2 - Δω.
    """
    lam2 = fiedler_eigenvalue(K)
    delta_omega = float(np.max(omega) - np.min(omega))
    margin = lam2 - delta_omega
    return margin > 0, margin


def qpe_resource_estimate(
    K: np.ndarray,
    epsilon: float = 0.01,
) -> tuple[int, int]:
    """Estimate QPE resources for Fiedler eigenvalue extraction.

    Args:
        K: coupling matrix
        epsilon: target precision for λ_2

    Returns:
        (n_bits, circuit_depth) where n_bits = ceil(log2(1/ε))
        and depth = O(2^n_bits × n²) for Hamiltonian simulation queries.
    """
    n = K.shape[0]
    n_bits = max(int(np.ceil(np.log2(1.0 / epsilon))), 1)
    # QPE depth: 2^n_bits controlled-U applications, each U costs O(n²) gates
    depth = (2**n_bits) * n * n
    return n_bits, depth


def spectral_bridge_analysis(
    K: np.ndarray,
    omega: np.ndarray,
    epsilon: float = 0.01,
) -> SpectralBridgeResult:
    """Full spectral bridge analysis."""
    lam2 = fiedler_eigenvalue(K)
    spectrum = laplacian_spectrum(K)
    delta_omega = float(np.max(omega) - np.min(omega))
    stable, margin = entrainment_criterion(K, omega)
    n_bits, depth = qpe_resource_estimate(K, epsilon)

    return SpectralBridgeResult(
        fiedler_value=lam2,
        frequency_spread=delta_omega,
        entrainment_stable=stable,
        stability_margin=margin,
        laplacian_spectrum=spectrum,
        qpe_bits_needed=n_bits,
        qpe_circuit_depth=depth,
    )


def spectral_bridge_vs_coupling(
    omega: np.ndarray,
    k_values: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Scan Fiedler value and stability margin vs coupling strength."""
    from ..bridge.knm_hamiltonian import build_knm_paper27

    if k_values is None:
        k_values = np.linspace(0.01, 3.0, 20)

    n = len(omega)
    results: dict[str, list[float]] = {
        "k_base": [],
        "fiedler": [],
        "stability_margin": [],
        "entrainment_stable": [],
    }

    for kb in k_values:
        K = build_knm_paper27(L=n, K_base=float(kb))
        lam2 = fiedler_eigenvalue(K)
        _stable, margin = entrainment_criterion(K, omega)
        results["k_base"].append(float(kb))
        results["fiedler"].append(lam2)
        results["stability_margin"].append(margin)
        results["entrainment_stable"].append(1.0 if margin > 0 else 0.0)

    return results
