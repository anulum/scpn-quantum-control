# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""ITER synthetic data benchmark: tokamak MHD mode coupling.

Tokamak plasmas exhibit coupled MHD oscillation modes that follow
Kuramoto-like dynamics. The coupling between modes depends on:
    - Toroidal mode numbers (n, m)
    - Plasma rotation profile
    - Magnetic island overlap (Chirikov parameter)

Synthetic ITER-like coupling matrix based on resistive wall mode
(RWM) and neoclassical tearing mode (NTM) interaction patterns.

MHD mode coupling → Kuramoto mapping:
    K_ij = magnetic coupling between modes i, j
    ω_i = mode rotation frequency
    Synchronisation = mode locking = disruption precursor

If SCPN K_nm topology matches MHD coupling patterns, this validates
Gap 1 for the fusion application.

Ref: La Haye, Physics of Plasmas 13, 055501 (2006) — NTM dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

# Synthetic ITER MHD mode parameters (8 modes)
# Based on published NTM/RWM mode structure for ITER-scale tokamak
ITER_MODE_LABELS = [
    "m2n1_NTM",  # (2,1) NTM — most dangerous
    "m3n2_NTM",  # (3,2) NTM
    "m1n1_kink",  # (1,1) internal kink
    "m2n1_RWM",  # (2,1) RWM
    "m3n1_ELM",  # (3,1) ELM-related
    "m4n3_NTM",  # (4,3) NTM
    "m1n1_sawtooth",  # (1,1) sawtooth
    "m5n4_NTM",  # (5,4) NTM
]

# Mode rotation frequencies (kHz) — typical ITER L-mode
ITER_MODE_FREQ = np.array([2.5, 4.0, 8.0, 1.5, 6.0, 5.5, 12.0, 7.0])

# Mode coupling matrix (arbitrary units, symmetric)
# Strong coupling between same-n modes, weak for different-n
ITER_MODE_COUPLING = np.array(
    [
        [0.0, 0.30, 0.15, 0.80, 0.10, 0.05, 0.12, 0.03],  # m2n1_NTM
        [0.30, 0.0, 0.10, 0.20, 0.08, 0.40, 0.05, 0.25],  # m3n2_NTM
        [0.15, 0.10, 0.0, 0.10, 0.05, 0.03, 0.70, 0.02],  # m1n1_kink
        [0.80, 0.20, 0.10, 0.0, 0.15, 0.08, 0.05, 0.04],  # m2n1_RWM
        [0.10, 0.08, 0.05, 0.15, 0.0, 0.12, 0.03, 0.06],  # m3n1_ELM
        [0.05, 0.40, 0.03, 0.08, 0.12, 0.0, 0.02, 0.35],  # m4n3_NTM
        [0.12, 0.05, 0.70, 0.05, 0.03, 0.02, 0.0, 0.01],  # m1n1_sawtooth
        [0.03, 0.25, 0.02, 0.04, 0.06, 0.35, 0.01, 0.0],  # m5n4_NTM
    ]
)


@dataclass
class ITERBenchmarkResult:
    """ITER MHD vs SCPN comparison result."""

    n_modes: int
    topology_correlation: float
    frequency_correlation: float
    coupling_ratio: float
    mode_locking_risk: float  # fraction of strongly coupled pairs
    summary: str


def iter_coupling_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Get ITER synthetic MHD mode coupling and frequencies."""
    return ITER_MODE_COUPLING.copy(), ITER_MODE_FREQ.copy()


def iter_benchmark(
    K_scpn: np.ndarray,
    omega_scpn: np.ndarray,
) -> ITERBenchmarkResult:
    """Compare SCPN K_nm with ITER MHD mode coupling."""
    K_iter, omega_iter = iter_coupling_matrix()
    n_iter = K_iter.shape[0]
    n_scpn = K_scpn.shape[0]
    n = min(n_iter, n_scpn)

    K_i = K_iter[:n, :n]
    K_s = K_scpn[:n, :n]
    omega_i = omega_iter[:n]
    omega_s = omega_scpn[:n]

    triu_idx = np.triu_indices(n, k=1)
    i_flat = K_i[triu_idx]
    s_flat = K_s[triu_idx]

    if len(i_flat) >= 3:
        topo_corr = float(spearmanr(i_flat, s_flat).statistic)
    else:
        topo_corr = 0.0

    if n >= 3:
        freq_corr = float(np.corrcoef(omega_i, omega_s)[0, 1])
        if np.isnan(freq_corr):
            freq_corr = 0.0
    else:
        freq_corr = 0.0

    i_mean = float(np.mean(i_flat[i_flat > 0])) if np.any(i_flat > 0) else 0.0
    s_mean = float(np.mean(s_flat[s_flat > 0])) if np.any(s_flat > 0) else 0.0
    ratio = i_mean / max(s_mean, 1e-15)

    # Mode locking risk: fraction of pairs with coupling > 0.5
    strong_pairs: int = int(np.sum(i_flat > 0.5))
    total_pairs = len(i_flat)
    locking_risk = strong_pairs / max(total_pairs, 1)

    summary = (
        f"SCPN vs ITER MHD: topology ρ={topo_corr:.3f}, "
        f"freq r={freq_corr:.3f}, locking risk={locking_risk:.2f}"
    )

    return ITERBenchmarkResult(
        n_modes=n,
        topology_correlation=topo_corr,
        frequency_correlation=freq_corr,
        coupling_ratio=ratio,
        mode_locking_risk=locking_risk,
        summary=summary,
    )
