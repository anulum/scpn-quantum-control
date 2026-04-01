# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Spectral Form Factor
"""Spectral Form Factor (SFF) at the synchronization transition.

The SFF K(t) = |Tr(e^{-iHt})|² / Z² is the diagnostic fingerprint
of quantum chaos:
- Integrable systems (Poisson): K(t) ≈ 1 (no dip)
- Chaotic systems (RMT): dip → ramp → plateau structure

At the BKT synchronization critical point K_c:
- Does the SFF transition from Poisson to RMT?
- Is the synchronization transition also a chaos transition?
- The BKT essential singularity may produce anomalous SFF behavior

Prior art: Joshi et al. PRL 2025 (SFF on hardware for chaos/MBL).
Andersen et al. Nature 2025 (BKT on hardware, no SFF).
SFF + synchronization: never combined.

The level spacing ratio r̄ (mean of min(δ_n, δ_{n+1})/max(...))
distinguishes:
- Poisson: r̄ ≈ 0.386 (integrable)
- GOE: r̄ ≈ 0.530 (chaotic, time-reversal symmetric)
- GUE: r̄ ≈ 0.603 (chaotic, time-reversal broken)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian


@dataclass
class SFFResult:
    """SFF computation at a single coupling strength."""

    K_base: float
    times: np.ndarray
    sff: np.ndarray  # K(t) = |Tr(e^{-iHt})|² / Z²
    level_spacing_ratio: float  # r̄ — chaos diagnostic
    spectral_gap: float


@dataclass
class SFFScanResult:
    """SFF scan across coupling strength."""

    k_values: np.ndarray
    level_spacing_ratios: np.ndarray  # r̄ at each K
    spectral_gaps: np.ndarray
    sff_dip_depth: np.ndarray  # min(K(t))/K(0) — deeper dip = more chaotic
    chaos_onset_K: float | None  # K where r̄ first exceeds Poisson threshold


def _level_spacing_ratio(eigenvalues: np.ndarray) -> float:
    """Mean level spacing ratio r̄ = ⟨min(δ_n, δ_{n+1})/max(δ_n, δ_{n+1})⟩.

    Poisson: 0.386, GOE: 0.530, GUE: 0.603.
    """
    spacings = np.diff(eigenvalues)
    spacings = spacings[spacings > 1e-15]
    if len(spacings) < 2:
        return 0.0
    ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return float(np.mean(ratios))


def compute_sff(
    K: np.ndarray,
    omega: np.ndarray,
    t_max: float = 20.0,
    n_times: int = 200,
) -> SFFResult:
    """Compute the Spectral Form Factor K(t) from eigenvalues."""
    knm_to_hamiltonian(K, omega)
    H_mat = knm_to_dense_matrix(K, omega)
    eigenvalues = np.linalg.eigvalsh(H_mat)

    dim = len(eigenvalues)
    gap = float(eigenvalues[1] - eigenvalues[0])
    r_bar = _level_spacing_ratio(eigenvalues)

    # SFF: K(t) = |Σ_n exp(-iE_n t)|² / d²
    times = np.linspace(0, t_max, n_times)
    sff: np.ndarray = np.zeros(n_times)

    for idx, t in enumerate(times):
        trace_val: complex = complex(np.sum(np.exp(-1j * eigenvalues * t)))
        sff[idx] = float(abs(trace_val) ** 2) / dim**2

    k_base = float(np.max(np.abs(K)))

    return SFFResult(
        K_base=k_base,
        times=times,
        sff=sff,
        level_spacing_ratio=r_bar,
        spectral_gap=gap,
    )


def sff_vs_coupling(
    omega: np.ndarray,
    K_topology: np.ndarray,
    k_range: np.ndarray | None = None,
    t_max: float = 20.0,
    n_times: int = 100,
) -> SFFScanResult:
    """Scan SFF diagnostics across coupling strength.

    At K_c, look for r̄ transition from Poisson (0.386) toward GOE (0.530).
    """
    if k_range is None:
        k_range = np.linspace(0.5, 5.0, 15)

    n_k = len(k_range)
    r_bars = np.zeros(n_k)
    gaps = np.zeros(n_k)
    dip_depths = np.zeros(n_k)

    for idx, kb in enumerate(k_range):
        K = float(kb) * K_topology
        result = compute_sff(K, omega, t_max, n_times)
        r_bars[idx] = result.level_spacing_ratio
        gaps[idx] = result.spectral_gap
        # Dip depth: minimum of K(t) for t > 0 relative to K(0)=1
        if len(result.sff) > 1:
            dip_depths[idx] = float(np.min(result.sff[1:]))
        else:
            dip_depths[idx] = 1.0

    # Chaos onset: where r̄ first exceeds midpoint between Poisson and GOE
    # Poisson: 0.386, GOE: 0.530. Threshold: 0.458
    chaos_threshold = 0.458
    chaos_k = None
    for i, r in enumerate(r_bars):
        if r > chaos_threshold:
            chaos_k = float(k_range[i])
            break

    return SFFScanResult(
        k_values=k_range,
        level_spacing_ratios=r_bars,
        spectral_gaps=gaps,
        sff_dip_depth=dip_depths,
        chaos_onset_K=chaos_k,
    )
