# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Bkt Analysis
"""BKT phase transition analysis for Kuramoto-XY systems.

Computes the BKT critical coupling K_c for the XY model on a finite
graph defined by the coupling matrix K_nm. Relates the synchronization
transition to the vortex binding/unbinding transition.

The BKT transition temperature T_BKT for the 2D XY model satisfies:
    T_BKT = (π/2) × J_eff

where J_eff is the effective coupling (spin stiffness / helicity modulus).

For our finite graph with coupling matrix K:
    J_eff = λ_2(L_K) / (2 × n)

where λ_2 is the Fiedler eigenvalue of the coupling-weighted Laplacian.

The consciousness gate threshold p_h1 = 0.72 corresponds to a specific
point on the BKT phase diagram. This module tests whether 0.72 can be
derived from the BKT universal numbers.

BKT universals:
    η(T_BKT) = 1/4  (critical exponent)
    Jump in stiffness: ρ_s(T_BKT⁻) = (2/π) × T_BKT  (Nelson-Kosterlitz)
    Correlation length: ξ ~ exp(b / sqrt(T - T_BKT))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BKTResult:
    """BKT transition analysis result."""

    fiedler_value: float
    effective_coupling: float
    t_bkt_estimate: float
    critical_ratio: float
    eta_critical: float
    n_oscillators: int
    stiffness_jump: float
    p_h1_predicted: float | None


def coupling_laplacian(K: np.ndarray) -> np.ndarray:
    """Coupling-weighted graph Laplacian: L = D - K where D_ii = Σ_j |K_ij|."""
    K_abs = np.abs(K)
    np.fill_diagonal(K_abs, 0.0)
    D = np.diag(np.sum(K_abs, axis=1))
    L: np.ndarray = D - K_abs
    return L


def fiedler_eigenvalue(K: np.ndarray) -> float:
    """Second-smallest eigenvalue of the coupling-weighted Laplacian.

    The Fiedler value measures algebraic connectivity of the coupling graph.
    λ_2 = 0 means disconnected; larger = more connected.
    """
    if K.shape[0] < 2:
        return 0.0
    L = coupling_laplacian(K)
    eigenvalues = np.sort(np.linalg.eigvalsh(L))
    return float(eigenvalues[1])


def estimate_t_bkt(K: np.ndarray) -> float:
    """Estimate BKT transition temperature from coupling matrix.

    Uses T_BKT = (π/2) × J_eff where J_eff = λ_2(L) / (2n).
    This is an approximation valid for the mean-field regime.
    """
    n = K.shape[0]
    lam2 = fiedler_eigenvalue(K)
    j_eff = lam2 / (2.0 * n)
    return float((np.pi / 2.0) * j_eff)


def bkt_analysis(
    K: np.ndarray,
    omega: np.ndarray | None = None,
) -> BKTResult:
    """Full BKT transition analysis for a coupling matrix.

    Computes:
    1. Fiedler eigenvalue (algebraic connectivity)
    2. Effective coupling J_eff
    3. Estimated BKT temperature T_BKT
    4. Critical ratio K/T_BKT (synchronization criterion)
    5. Nelson-Kosterlitz stiffness jump
    6. Predicted p_h1 from BKT universal amplitude ratio

    The key question: does the BKT framework predict p_h1 = 0.72?
    """
    n = K.shape[0]
    lam2 = fiedler_eigenvalue(K)
    j_eff = lam2 / (2.0 * n)
    t_bkt = (np.pi / 2.0) * j_eff

    # Mean coupling as "temperature" proxy
    K_abs = np.abs(K)
    np.fill_diagonal(K_abs, 0.0)
    k_mean = float(np.mean(K_abs[K_abs > 0])) if np.any(K_abs > 0) else 0.0
    critical_ratio = k_mean / max(t_bkt, 1e-15)

    # Nelson-Kosterlitz universal stiffness jump
    stiffness_jump = (2.0 / np.pi) * t_bkt

    # BKT critical exponent
    eta = 0.25  # universal value at T_BKT

    # Attempt to predict p_h1 from BKT universals
    # The persistent homology threshold corresponds to the vortex pair
    # binding probability. At T_BKT, the probability of finding a bound
    # vortex pair at distance r scales as r^{-eta} = r^{-1/4}.
    # For a finite system of size L = sqrt(n), the fraction of bound pairs:
    #   P_bound = integral_a^L r^{-1/4} dr / integral_a^L dr
    #   = (4/3)(L^{3/4} - a^{3/4}) / (L - a)
    # For L >> a (large system): P_bound → (4/3) × L^{-1/4}
    # For n=16 (L=4): P_bound ≈ (4/3) × 4^{-1/4} ≈ (4/3) × 0.707 ≈ 0.943
    # For n=16 but with finite-size corrections and noise:
    # P_bound ≈ 0.72 ± 0.1 is plausible but NOT exact.
    L_eff = np.sqrt(n)
    if L_eff > 1:
        a = 1.0  # lattice cutoff
        p_bound = (4.0 / 3.0) * (L_eff**0.75 - a**0.75) / max(L_eff - a, 1e-10)
        p_h1_pred = min(p_bound, 1.0)
    else:
        p_h1_pred = None

    return BKTResult(
        fiedler_value=lam2,
        effective_coupling=j_eff,
        t_bkt_estimate=t_bkt,
        critical_ratio=critical_ratio,
        eta_critical=eta,
        n_oscillators=n,
        stiffness_jump=stiffness_jump,
        p_h1_predicted=p_h1_pred,
    )


def scan_synchronization_transition(
    K_base_values: np.ndarray,
    alpha: float = 0.3,
    n: int = 16,
) -> dict:
    """Scan BKT observables across coupling strength K_base.

    Returns dict with K_base values and corresponding:
    - T_BKT estimates
    - Fiedler values
    - Critical ratios
    - Predicted p_h1 values
    """
    from ..bridge.knm_hamiltonian import build_knm_paper27

    results: dict[str, list[float]] = {
        "K_base": [],
        "T_BKT": [],
        "fiedler": [],
        "critical_ratio": [],
        "p_h1_predicted": [],
    }

    for kb in K_base_values:
        K = build_knm_paper27(L=n, K_base=kb, K_alpha=alpha)
        bkt = bkt_analysis(K)
        results["K_base"].append(float(kb))
        results["T_BKT"].append(bkt.t_bkt_estimate)
        results["fiedler"].append(bkt.fiedler_value)
        results["critical_ratio"].append(bkt.critical_ratio)
        results["p_h1_predicted"].append(bkt.p_h1_predicted if bkt.p_h1_predicted else 0.0)

    return results
