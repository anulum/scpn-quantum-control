# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Spectral
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
    4. Provides coarse asymptotic QPE resource estimates, not hardware evidence
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..analysis.bkt_analysis import coupling_laplacian, fiedler_eigenvalue


@dataclass
class SpectralBridgeResult:
    """Spectral bridge analysis result.

    Attributes
    ----------
    fiedler_value
        Algebraic-connectivity eigenvalue of the weighted Laplacian.
    frequency_spread
        Difference between the largest and smallest natural frequencies.
    entrainment_stable
        Whether the Fiedler value exceeds the frequency spread.
    stability_margin
        Fiedler value minus the frequency spread.
    laplacian_spectrum
        Sorted eigenvalues of the weighted Laplacian.
    qpe_bits_needed
        Coarse phase-estimation precision-bit estimate.
    qpe_circuit_depth
        Coarse asymptotic circuit-depth estimate, not a compiled-circuit count.

    """

    fiedler_value: float  # λ_2
    frequency_spread: float  # max(ω) - min(ω)
    entrainment_stable: bool  # λ_2 > frequency_spread
    stability_margin: float  # λ_2 - Δω (positive = stable)
    laplacian_spectrum: NDArray[np.float64]  # all eigenvalues of L_K
    qpe_bits_needed: int  # precision bits for ε resolution
    qpe_circuit_depth: int  # estimated QPE depth


def laplacian_spectrum(K: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the full spectrum of the coupling-weighted Laplacian.

    Parameters
    ----------
    K
        Oscillator coupling matrix.

    Returns
    -------
    NDArray[np.float64]
        Laplacian eigenvalues in ascending order.

    """
    L = coupling_laplacian(K)
    eigenvalues: NDArray[np.float64] = np.sort(np.linalg.eigvalsh(L)).astype(np.float64)
    return eigenvalues


def entrainment_criterion(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> tuple[bool, float]:
    """Check if λ_2 > Δω (synchronisation possible).

    Parameters
    ----------
    K
        Oscillator coupling matrix.
    omega
        Natural oscillator frequencies.

    Returns
    -------
    tuple[bool, float]
        Stability decision and margin ``λ_2 - Δω``.

    """
    lam2 = fiedler_eigenvalue(K)
    delta_omega = float(np.max(omega) - np.min(omega))
    margin = lam2 - delta_omega
    return margin > 0, margin


def qpe_resource_estimate(
    K: NDArray[np.float64],
    epsilon: float = 0.01,
) -> tuple[int, int]:
    """Estimate QPE resources for Fiedler eigenvalue extraction.

    This asymptotic estimate is not a compiled-circuit, device-readiness, or
    hardware-performance claim.

    Parameters
    ----------
    K
        Oscillator coupling matrix.
    epsilon
        Target Fiedler-value resolution.

    Returns
    -------
    tuple[int, int]
        Precision-bit estimate ``ceil(log2(1/epsilon))`` and coarse depth
        ``2**n_bits * n**2`` for Hamiltonian-simulation queries.

    """
    n = K.shape[0]
    n_bits = max(int(np.ceil(np.log2(1.0 / epsilon))), 1)
    # QPE depth: 2^n_bits controlled-U applications, each U costs O(n²) gates
    depth = (2**n_bits) * n * n
    return n_bits, depth


def spectral_bridge_analysis(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    epsilon: float = 0.01,
) -> SpectralBridgeResult:
    """Run the full spectral bridge analysis.

    Parameters
    ----------
    K
        Oscillator coupling matrix.
    omega
        Natural oscillator frequencies.
    epsilon
        Target resolution for the coarse QPE resource estimate.

    Returns
    -------
    SpectralBridgeResult
        Connectivity, stability, spectrum, and diagnostic resource estimates.

    """
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
    omega: NDArray[np.float64],
    k_values: NDArray[np.float64] | None = None,
) -> dict[str, list[float]]:
    """Scan Fiedler value and stability margin against coupling strength.

    Parameters
    ----------
    omega
        Natural oscillator frequencies.
    k_values
        Base coupling strengths. Uses a fixed 20-point grid when omitted.

    Returns
    -------
    dict[str, list[float]]
        Base couplings, Fiedler values, stability margins, and numeric stable
        indicators in input order.

    """
    from ..bridge.knm_hamiltonian import build_knm_paper27

    if k_values is None:
        k_values = np.linspace(0.01, 3.0, 20, dtype=np.float64)

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
