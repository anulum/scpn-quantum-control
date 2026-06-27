# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Eeg Benchmark
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
from numpy.typing import NDArray
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
    """EEG vs SCPN structural-comparison result."""

    n_channels: int
    topology_correlation: float
    frequency_correlation: float
    coupling_ratio: float
    eeg_band: str
    summary: str
    source_mode: str
    publication_safe: bool

    @property
    def topology_similarity_proxy(self) -> float:
        """Spearman PLV-vs-K_nm similarity proxy, not a neural model reproduction."""
        return self.topology_correlation


def _validated_square_matrix(
    matrix: NDArray[np.float64],
    name: str,
    *,
    require_plv: bool = False,
) -> NDArray[np.float64]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix.")
    if values.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two coupled channels.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    if require_plv:
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} values must be in [0, 1].")
        if not np.allclose(values, values.T, atol=1e-12):
            raise ValueError(f"{name} must be symmetric.")
        if not np.allclose(np.diag(values), 0.0, atol=1e-12):
            raise ValueError(f"{name} diagonal must be zero.")
    return values


def _validated_frequency_vector(
    frequencies: NDArray[np.float64],
    n_channels: int,
    name: str,
    matrix_name: str,
) -> NDArray[np.float64]:
    values = np.asarray(frequencies, dtype=float)
    if values.ndim != 1 or values.shape != (n_channels,):
        raise ValueError(f"{name} must match {matrix_name} channel count.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _finite_correlation(value: float) -> float:
    if np.isnan(value):
        return 0.0
    return float(value)


def eeg_coupling_matrix(
    band: str = "alpha",
    *,
    allow_builtin_reference: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get the built-in EEG coupling reference for a frequency band."""
    if not allow_builtin_reference:
        raise RuntimeError(
            "Refusing built-in EEG reference matrix without allow_builtin_reference=True. "
            "Pass measured EEG PLV/coherence matrices to eeg_benchmark for publication-safe claims."
        )
    if band == "alpha":
        return EEG_ALPHA_PLV.copy(), EEG_ALPHA_FREQ.copy()
    raise ValueError(f"Unknown EEG band: {band}. Available: alpha")


def eeg_benchmark(
    K_scpn: NDArray[np.float64],
    omega_scpn: NDArray[np.float64],
    band: str = "alpha",
    *,
    eeg_coupling: NDArray[np.float64] | None = None,
    eeg_frequencies: NDArray[np.float64] | None = None,
    allow_builtin_reference: bool = False,
) -> EEGBenchmarkResult:
    """Compare SCPN K_nm with EEG functional connectivity."""
    K_scpn = _validated_square_matrix(K_scpn, "K_scpn")
    omega_scpn = _validated_frequency_vector(
        omega_scpn,
        K_scpn.shape[0],
        "omega_scpn",
        "K_scpn",
    )
    if eeg_coupling is None or eeg_frequencies is None:
        if eeg_coupling is not None or eeg_frequencies is not None:
            raise ValueError("eeg_coupling and eeg_frequencies must be supplied together.")
        K_eeg, omega_eeg = eeg_coupling_matrix(
            band, allow_builtin_reference=allow_builtin_reference
        )
        source_mode = "builtin_literature_shape"
        publication_safe = False
    else:
        K_eeg = _validated_square_matrix(eeg_coupling, "eeg_coupling", require_plv=True)
        omega_eeg = _validated_frequency_vector(
            eeg_frequencies,
            K_eeg.shape[0],
            "eeg_frequencies",
            "eeg_coupling",
        )
        source_mode = "measured"
        publication_safe = True

    K_eeg = _validated_square_matrix(K_eeg, "eeg_coupling", require_plv=True)
    omega_eeg = _validated_frequency_vector(
        omega_eeg,
        K_eeg.shape[0],
        "eeg_frequencies",
        "eeg_coupling",
    )

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
        topo_corr = _finite_correlation(float(spearmanr(e_flat, s_flat).statistic))
    else:
        topo_corr = 0.0

    if n >= 3:
        freq_corr = _finite_correlation(float(np.corrcoef(omega_e, omega_s)[0, 1]))
    else:
        freq_corr = 0.0

    e_mean = float(np.mean(e_flat[e_flat > 0])) if np.any(e_flat > 0) else 0.0
    s_mean = float(np.mean(s_flat[s_flat > 0])) if np.any(s_flat > 0) else 0.0
    ratio = e_mean / max(s_mean, 1e-15)

    summary = (
        f"SCPN vs EEG ({band}): topology similarity proxy ρ={topo_corr:.3f}, "
        f"freq r={freq_corr:.3f}, coupling ratio={ratio:.3f}"
    )

    return EEGBenchmarkResult(
        n_channels=n,
        topology_correlation=topo_corr,
        frequency_correlation=freq_corr,
        coupling_ratio=ratio,
        eeg_band=band,
        summary=summary,
        source_mode=source_mode,
        publication_safe=publication_safe,
    )
