# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Robustness
"""Adiabatic robustness certificate for identity binding.

The energy gap Δ = E_1 - E_0 of the identity Hamiltonian H(K_nm)
provides a quantitative stability guarantee: perturbations below
Δ cannot change the ground state identity.

Adiabatic theorem (Jansen, Ruskai, Seiler 2007):
    If H(s) varies along path s in [0,1] with minimum gap g_min,
    and ||dH/ds|| <= J, then the transition probability satisfies:
        P_transition <= (J / g_min²)²

For constant Hamiltonian with noise perturbation δH:
    P_transition = 0  if ||δH|| < Δ/2  (exact for 2-level)
    P_transition ~ (||δH|| / Δ)²      (perturbative regime)
    P_transition ~ 1                    (||δH|| >> Δ, identity lost)

Decoherence connection: T2 dephasing at rate γ gives effective
perturbation ||δH_eff|| ~ γ. The identity survives if γ < Δ/2,
i.e., T2 > 2/Δ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..hardware.classical import classical_exact_diag


@dataclass
class RobustnessCertificate:
    """Quantitative robustness bounds for an identity binding."""

    energy_gap: float
    max_safe_perturbation: float  # ||δH|| below which identity is preserved
    min_t2_for_stability: float  # T2 (μs) needed to protect identity
    transition_probability: float  # P_transition for given noise level
    adiabatic_bound: float  # Jansen-Ruskai-Seiler bound
    n_oscillators: int
    eigenvalues: list[float]  # first few eigenvalues


def compute_robustness_certificate(
    K: np.ndarray,
    omega: np.ndarray,
    noise_strength: float = 0.01,
    sweep_rate: float = 0.1,
) -> RobustnessCertificate:
    """Compute adiabatic robustness certificate from coupling matrix.

    Args:
        K: coupling matrix
        omega: natural frequencies
        noise_strength: ||δH|| of the perturbation
        sweep_rate: ||dH/ds|| for adiabatic bound
    """
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    eigenvalues = exact["eigenvalues"]

    gap = float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) >= 2 else 0.0

    # Max safe perturbation: ||δH|| < Δ/2 preserves ground state
    max_safe = gap / 2.0

    # Min T2 for stability: γ = 1/T2 < Δ/2 → T2 > 2/Δ
    min_t2 = 2.0 / max(gap, 1e-15)

    # Transition probability in perturbative regime
    if gap > 1e-15:
        p_transition = min((noise_strength / gap) ** 2, 1.0)
    else:
        p_transition = 1.0

    # Adiabatic bound: P <= (J / g_min²)²
    if gap > 1e-15:
        adiabatic = min((sweep_rate / gap**2) ** 2, 1.0)
    else:
        adiabatic = 1.0

    return RobustnessCertificate(
        energy_gap=gap,
        max_safe_perturbation=max_safe,
        min_t2_for_stability=min_t2,
        transition_probability=p_transition,
        adiabatic_bound=adiabatic,
        n_oscillators=n,
        eigenvalues=eigenvalues[: min(6, len(eigenvalues))].tolist(),
    )


def perturbation_fidelity(
    K: np.ndarray,
    omega: np.ndarray,
    delta_K: np.ndarray,
) -> float:
    """Ground state overlap |<ψ_0(K)|ψ_0(K+δK)>|² under coupling perturbation.

    Direct numerical check: solve both Hamiltonians and compute overlap.
    """
    n = K.shape[0]
    exact_orig = classical_exact_diag(n, K=K, omega=omega)
    exact_pert = classical_exact_diag(n, K=K + delta_K, omega=omega)

    psi_0 = exact_orig["ground_state"]
    psi_pert = exact_pert["ground_state"]

    overlap = abs(np.dot(psi_0.conj(), psi_pert)) ** 2
    return float(overlap)


def gap_vs_perturbation_scan(
    K: np.ndarray,
    omega: np.ndarray,
    noise_range: np.ndarray | None = None,
    n_samples: int = 20,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Scan transition probability vs perturbation strength.

    Returns dict with noise_strength, p_transition, fidelity columns.
    """
    if noise_range is None:
        noise_range = np.linspace(0.001, 0.5, n_samples)

    rng = np.random.default_rng(seed)
    n = K.shape[0]

    cert = compute_robustness_certificate(K, omega)
    gap = cert.energy_gap

    results: dict[str, list[float]] = {
        "noise_strength": [],
        "p_transition_theory": [],
        "fidelity_numerical": [],
    }

    for eps in noise_range:
        delta = rng.normal(0, eps, size=(n, n))
        delta = (delta + delta.T) / 2.0
        np.fill_diagonal(delta, 0.0)

        fid = perturbation_fidelity(K, omega, delta)

        if gap > 1e-15:
            p_theory = min((eps / gap) ** 2, 1.0)
        else:
            p_theory = 1.0

        results["noise_strength"].append(float(eps))
        results["p_transition_theory"].append(p_theory)
        results["fidelity_numerical"].append(fid)

    return results
