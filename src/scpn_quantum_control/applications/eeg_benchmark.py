# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""EEG neural oscillator coupling benchmark.

The brain's neural oscillators synchronise across frequency bands
(delta 1-4 Hz, theta 4-8, alpha 8-13, beta 13-30, gamma 30-100).
Inter-region coupling follows the Kuramoto model:

    dφ_i/dt = ω_i + Σ_j K_ij sin(φ_j - φ_i)

where φ_i = phase of oscillator at brain region i,
ω_i = natural frequency, K_ij = functional connectivity.

EEG coupling matrices (functional connectivity) are estimated from:
    - Phase Locking Value (PLV): K_ij = |<exp(i(φ_j - φ_i))>|
    - Coherence: K_ij = |C_xy(f)|²
    - Weighted Phase Lag Index (wPLI)

Canonical EEG coupling patterns:
    - Resting state alpha: strong occipital-parietal, weak frontal
    - Attention beta: strong fronto-parietal network
    - Sleep delta: global synchronisation

This module provides synthetic EEG coupling matrices based on
published functional connectivity patterns for comparison with
SCPN K_nm.

Ref: Breakspear, Nature Neuroscience 20, 340 (2017).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

# Canonical 8-channel EEG layout (10-20 system subset)
EEG_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]

# Alpha-band (8-13 Hz) resting state PLV matrix (8×8, synthetic but
# based on published patterns: strong occipital, moderate parietal)
# Ref: Stam et al., Clinical Neurophysiology 118, 2317 (2007)
EEG_ALPHA_PLV = np.array(
    [
        [0.0, 0.65, 0.40, 0.35, 0.20, 0.15, 0.10, 0.08],
        [0.65, 0.0, 0.35, 0.40, 0.15, 0.20, 0.08, 0.10],
        [0.40, 0.35, 0.0, 0.50, 0.45, 0.30, 0.20, 0.15],
        [0.35, 0.40, 0.50, 0.0, 0.30, 0.45, 0.15, 0.20],
        [0.20, 0.15, 0.45, 0.30, 0.0, 0.55, 0.40, 0.35],
        [0.15, 0.20, 0.30, 0.45, 0.55, 0.0, 0.35, 0.40],
        [0.10, 0.08, 0.20, 0.15, 0.40, 0.35, 0.0, 0.70],
        [0.08, 0.10, 0.15, 0.20, 0.35, 0.40, 0.70, 0.0],
    ]
)

# Natural frequencies (Hz) — alpha band peak per region
EEG_ALPHA_FREQ = np.array([9.5, 9.5, 10.0, 10.0, 10.5, 10.5, 11.0, 11.0])


@dataclass
class EEGBenchmarkResult:
    """EEG vs SCPN comparison result."""

    n_channels: int
    topology_correlation: float
    frequency_correlation: float
    coupling_ratio: float
    eeg_band: str
    summary: str


def eeg_coupling_matrix(band: str = "alpha") -> tuple[np.ndarray, np.ndarray]:
    """Get synthetic EEG coupling matrix for given frequency band."""
    if band == "alpha":
        return EEG_ALPHA_PLV.copy(), EEG_ALPHA_FREQ.copy()
    raise ValueError(f"Unknown EEG band: {band}. Available: alpha")


def eeg_benchmark(
    K_scpn: np.ndarray,
    omega_scpn: np.ndarray,
    band: str = "alpha",
) -> EEGBenchmarkResult:
    """Compare SCPN K_nm with EEG functional connectivity."""
    K_eeg, omega_eeg = eeg_coupling_matrix(band)
    n_eeg = K_eeg.shape[0]
    n_scpn = K_scpn.shape[0]
    n = min(n_eeg, n_scpn)

    K_e = K_eeg[:n, :n]
    K_s = K_scpn[:n, :n]
    omega_e = omega_eeg[:n]
    omega_s = omega_scpn[:n]

    triu_idx = np.triu_indices(n, k=1)
    e_flat = K_e[triu_idx]
    s_flat = K_s[triu_idx]

    if len(e_flat) >= 3:
        topo_corr = float(spearmanr(e_flat, s_flat).statistic)
    else:
        topo_corr = 0.0

    if n >= 3:
        freq_corr = float(np.corrcoef(omega_e, omega_s)[0, 1])
        if np.isnan(freq_corr):
            freq_corr = 0.0
    else:
        freq_corr = 0.0

    e_mean = float(np.mean(e_flat[e_flat > 0])) if np.any(e_flat > 0) else 0.0
    s_mean = float(np.mean(s_flat[s_flat > 0])) if np.any(s_flat > 0) else 0.0
    ratio = e_mean / max(s_mean, 1e-15)

    summary = (
        f"SCPN vs EEG ({band}): topology ρ={topo_corr:.3f}, "
        f"freq r={freq_corr:.3f}, coupling ratio={ratio:.3f}"
    )

    return EEGBenchmarkResult(
        n_channels=n,
        topology_correlation=topo_corr,
        frequency_correlation=freq_corr,
        coupling_ratio=ratio,
        eeg_band=band,
        summary=summary,
    )
