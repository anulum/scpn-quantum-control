# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Entanglement Spectrum
"""Entanglement spectrum analysis at the synchronization transition.

Measures bipartite entanglement entropy S(ρ_A) as a function of
subsystem size |A| and coupling strength K. The scaling law reveals
the quantum phase structure:

    - Area law: S(|A|) ~ O(1) — gapped, non-critical
    - Log correction: S(|A|) ~ (c/3) log(|A|) — CFT at criticality (c=1 for XY)
    - Volume law: S(|A|) ~ |A| — maximally entangled

At the Kuramoto synchronization transition (K ≈ K_c), the XY model
is in the BKT universality class with CFT central charge c=1. The
entanglement entropy should show logarithmic scaling at criticality
(Calabrese & Cardy 2004, JSTAT P06002).

This module scans K and measures:
    1. Half-chain entanglement entropy S(n/2) vs K
    2. Entanglement entropy scaling S(|A|) vs |A| at fixed K
    3. Entanglement spectrum (eigenvalues of ρ_A) at criticality
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..hardware.classical import classical_exact_diag
from .quantum_phi import partial_trace, von_neumann_entropy


@dataclass
class EntanglementResult:
    """Entanglement spectrum analysis result."""

    n_qubits: int
    half_chain_entropy: float  # S(n/2)
    entanglement_spectrum: NDArray[np.float64]  # eigenvalues of ρ_A
    entropy_vs_subsystem: list[float]  # S(|A|) for |A| = 1..n/2
    cft_central_charge: float | None  # extracted c from log fit


def entanglement_entropy_half_chain(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> float:
    """Bipartite entanglement entropy for half-chain partition."""
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi = exact["ground_state"]
    rho = np.outer(psi, psi.conj())

    half = n // 2
    keep = list(range(half))
    rho_a = partial_trace(rho, keep, n)
    return von_neumann_entropy(rho_a)


def entanglement_spectrum_half_chain(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Eigenvalue spectrum of the reduced density matrix ρ_A (half chain)."""
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi = exact["ground_state"]
    rho = np.outer(psi, psi.conj())

    half = n // 2
    rho_a = partial_trace(rho, list(range(half)), n)
    eigenvalues: NDArray[np.float64] = np.sort(np.linalg.eigvalsh(rho_a)).astype(np.float64)[::-1]
    return eigenvalues


def entropy_vs_subsystem_size(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> list[float]:
    """S(|A|) for |A| = 1 to n//2."""
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi = exact["ground_state"]
    rho = np.outer(psi, psi.conj())

    entropies: list[float] = []
    for size in range(1, n // 2 + 1):
        rho_a = partial_trace(rho, list(range(size)), n)
        entropies.append(von_neumann_entropy(rho_a))
    return entropies


def fit_cft_central_charge(
    subsystem_sizes: NDArray[np.float64],
    entropies: NDArray[np.float64],
    n_total: int,
) -> float | None:
    """Extract CFT central charge c from S(l) = (c/3) log(l) + const.

    Uses chord length for finite systems:
        S(l) = (c/3) log((n/π) sin(πl/n)) + const
    (Calabrese & Cardy 2004)
    """
    sizes = np.asarray(subsystem_sizes, dtype=np.float64)
    entropy_values = np.asarray(entropies, dtype=np.float64)
    if sizes.ndim != 1 or entropy_values.ndim != 1:
        raise ValueError("subsystem_sizes and entropies must be one-dimensional arrays")
    if sizes.shape[0] != entropy_values.shape[0]:
        raise ValueError("subsystem_sizes and entropies must have the same length")
    if not isinstance(n_total, int) or n_total < 2:
        raise ValueError("n_total must be an integer at least 2")
    if len(sizes) < 3:
        return None
    if not np.all(np.isfinite(sizes)) or not np.all(np.isfinite(entropy_values)):
        raise ValueError("subsystem_sizes and entropies must contain only finite values")
    if np.any(sizes <= 0):
        raise ValueError("subsystem_sizes must be positive")
    if np.any(sizes >= n_total):
        raise ValueError("subsystem_sizes must lie inside the chain")

    # Chord length: x(l) = (n/π) sin(πl/n)
    x = (n_total / np.pi) * np.sin(np.pi * sizes / n_total)
    log_x = np.log(x)

    # Linear fit: S = (c/3) × log(x) + const
    A = np.vstack([log_x, np.ones_like(log_x)]).T
    result = np.linalg.lstsq(A, entropy_values, rcond=None)
    slope = result[0][0]
    c = 3.0 * slope
    return float(c)


def entanglement_analysis(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> EntanglementResult:
    """Full entanglement spectrum analysis at given coupling."""
    n = K.shape[0]
    s_half = entanglement_entropy_half_chain(K, omega)
    spectrum = entanglement_spectrum_half_chain(K, omega)
    s_vs_l = entropy_vs_subsystem_size(K, omega)

    sizes = np.arange(1, n // 2 + 1, dtype=float)
    c = fit_cft_central_charge(sizes, np.array(s_vs_l), n)

    return EntanglementResult(
        n_qubits=n,
        half_chain_entropy=s_half,
        entanglement_spectrum=spectrum,
        entropy_vs_subsystem=s_vs_l,
        cft_central_charge=c,
    )


def entropy_vs_coupling_scan(
    omega: NDArray[np.float64],
    k_base_values: NDArray[np.float64] | None = None,
) -> dict[str, list[float]]:
    """Scan half-chain entropy S(n/2) vs coupling K_base.

    The peak/inflection of S vs K marks the entanglement transition:
    area law (low K) → log correction (K_c) → volume law (high K).
    """
    from ..bridge.knm_hamiltonian import build_knm_paper27

    if k_base_values is None:
        k_base_values = np.linspace(0.01, 3.0, 30, dtype=np.float64)

    n = len(omega)
    results: dict[str, list[float]] = {
        "k_base": [],
        "half_chain_entropy": [],
        "cft_central_charge": [],
    }

    for kb in k_base_values:
        K = build_knm_paper27(L=n, K_base=float(kb))
        analysis = entanglement_analysis(K, omega)
        results["k_base"].append(float(kb))
        results["half_chain_entropy"].append(analysis.half_chain_entropy)
        results["cft_central_charge"].append(
            analysis.cft_central_charge if analysis.cft_central_charge is not None else 0.0
        )

    return results
