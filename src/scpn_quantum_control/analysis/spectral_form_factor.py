# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Spectral Form Factor
"""Finite-size spectral form-factor and adjacent-gap diagnostics.

The normalized SFF ``K(t) = |Tr(exp(-iHt))|² / d²`` and adjacent-gap ratio
are commonly used spectral diagnostics. Their interpretation depends on
symmetry resolution, ensemble/energy-window choices, unfolding conventions,
and finite-size scaling.
- Integrable systems (Poisson): K(t) ≈ 1 (no dip)
- Chaotic systems (RMT): dip → ramp → plateau structure

This module reports exact finite-system values. It does not certify quantum
chaos, a Poisson-to-RMT transition, a BKT transition, or a coincidence between
chaos and synchronization.

The level spacing ratio r̄ (mean of min(δ_n, δ_{n+1})/max(...))
distinguishes:
- Poisson: r̄ ≈ 0.386 (integrable)
- GOE: r̄ ≈ 0.530 (chaotic, time-reversal symmetric)
- GUE: r̄ ≈ 0.603 (chaotic, time-reversal broken)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian
from ..dense_budget import require_dense_eigensolver_workspace
from .magnetisation_sectors import level_spacing_by_magnetisation
from .symmetry_sectors import level_spacing_by_sector

LevelSpacingBasis = Literal["magnetisation", "parity", "full"]


@dataclass
class SFFResult:
    """Finite-spectrum SFF and level-spacing result at one coupling.

    The selected symmetry basis and sector metadata are part of the result so
    callers cannot silently present a mixed-sector ratio as chaos evidence.

    Attributes
    ----------
    K_base
        Maximum absolute coupling in the evaluated matrix.
    times
        Inclusive finite-time evaluation grid.
    sff
        Normalized spectral form factor at each time.
    level_spacing_ratio
        Adjacent-gap ratio in the selected symmetry basis.
    spectral_gap
        Full-spectrum gap between the two lowest eigenvalues.
    level_spacing_basis
        Symmetry basis used for the reported adjacent-gap ratio.
    level_spacing_sector
        Selected magnetisation or parity sector, when applicable.
    level_spacing_sector_dim
        Hilbert-space dimension of the selected sector.
    full_spectrum_level_spacing_ratio
        Adjacent-gap ratio before symmetry-sector resolution.

    """

    K_base: float
    times: NDArray[np.float64]
    sff: NDArray[np.float64]  # K(t) = |Tr(e^{-iHt})|² / Z²
    level_spacing_ratio: float  # r̄ — chaos diagnostic
    spectral_gap: float
    level_spacing_basis: str = "magnetisation"
    level_spacing_sector: int | None = None
    level_spacing_sector_dim: int = 0
    full_spectrum_level_spacing_ratio: float = float("nan")


@dataclass
class SFFScanResult:
    """Finite-grid spectral diagnostic scan.

    ``chaos_onset_K`` is a heuristic first threshold crossing on the supplied
    grid, not a statistical or thermodynamic-limit certification.

    Attributes
    ----------
    k_values
        Evaluated coupling-scale grid.
    level_spacing_ratios
        Selected-sector adjacent-gap ratio at each coupling.
    spectral_gaps
        Full-spectrum gap at each coupling.
    sff_dip_depth
        Minimum nonzero-time SFF value at each coupling.
    chaos_onset_K
        First finite-grid threshold crossing, if one exists.

    """

    k_values: NDArray[np.float64]
    level_spacing_ratios: NDArray[np.float64]  # r̄ at each K
    spectral_gaps: NDArray[np.float64]
    sff_dip_depth: NDArray[np.float64]  # min(K(t))/K(0) — deeper dip = more chaotic
    chaos_onset_K: float | None  # K where r̄ first exceeds Poisson threshold


def _level_spacing_ratio(eigenvalues: NDArray[np.float64]) -> float:
    """Mean level spacing ratio r̄ = ⟨min(δ_n, δ_{n+1})/max(δ_n, δ_{n+1})⟩.

    Poisson: 0.386, GOE: 0.530, GUE: 0.603.
    """
    spacings = np.diff(eigenvalues)
    spacings = spacings[spacings > 1e-15]
    if len(spacings) < 2:
        return 0.0
    ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return float(np.mean(ratios))


def _sector_level_spacing_ratio(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    basis: LevelSpacingBasis,
    magnetisation: int | None = None,
    parity: int | None = None,
    full_eigenvalues: NDArray[np.float64],
    max_dense_gib: float | None = None,
) -> tuple[float, int | None, int]:
    if basis == "full":
        return _level_spacing_ratio(full_eigenvalues), None, len(full_eigenvalues)
    if basis == "magnetisation":
        sector = level_spacing_by_magnetisation(
            K,
            omega,
            M=magnetisation,
            max_dense_gib=max_dense_gib,
        )
        return float(sector["r_bar"]), int(sector["M"]), int(sector["dim"])
    if basis == "parity":
        if parity is not None and parity not in (0, 1):
            raise ValueError("parity must be 0, 1, or None for ground-parity selection.")
        sector = level_spacing_by_sector(
            K,
            omega,
            max_dense_gib=max_dense_gib,
        )
        selected_parity = int(sector["ground_parity"] if parity is None else parity)
        ratio_key = "r_bar_even" if selected_parity == 0 else "r_bar_odd"
        return (
            float(sector[ratio_key]),
            selected_parity,
            int(sector["dim_per_sector"]),
        )
    raise ValueError("level_spacing_basis must be 'magnetisation', 'parity', or 'full'.")


def compute_sff(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    t_max: float = 20.0,
    n_times: int = 200,
    *,
    level_spacing_basis: LevelSpacingBasis = "magnetisation",
    magnetisation: int | None = None,
    parity: int | None = None,
    max_dense_gib: float | None = None,
) -> SFFResult:
    """Compute normalized SFF values from exact finite-system eigenvalues.

    Parameters
    ----------
    K, omega
        Coupling matrix and natural-frequency vector.
    t_max, n_times
        Inclusive time horizon and number of grid points.
    level_spacing_basis
        ``"magnetisation"`` (default), ``"parity"``, or ``"full"``.
    magnetisation, parity
        Optional explicit sector selectors for their corresponding basis.
    max_dense_gib
        Optional fail-closed dense eigensolver budget.

    Returns
    -------
    SFFResult
        Full-spectrum SFF plus selected-sector and full-spectrum gap ratios.

    Notes
    -----
    The SFF itself uses the full finite-size spectrum. The reported
    level-spacing ratio defaults to a U(1) magnetisation sector because
    mixing independent symmetry sectors biases spectral diagnostics. The
    output is not a quantum-chaos certificate.

    """
    n = len(omega)
    require_dense_eigensolver_workspace(
        n,
        max_gib=max_dense_gib,
        label="SFF dense eigensolver",
    )
    knm_to_hamiltonian(K, omega)
    H_mat = knm_to_dense_matrix(K, omega, max_dense_gib=max_dense_gib)
    eigenvalues = np.linalg.eigvalsh(H_mat).astype(np.float64)

    dim = len(eigenvalues)
    gap = float(eigenvalues[1] - eigenvalues[0])
    full_r_bar = _level_spacing_ratio(eigenvalues)
    r_bar, sector, sector_dim = _sector_level_spacing_ratio(
        K,
        omega,
        basis=level_spacing_basis,
        magnetisation=magnetisation,
        parity=parity,
        full_eigenvalues=eigenvalues,
        max_dense_gib=max_dense_gib,
    )

    # SFF: K(t) = |Σ_n exp(-iE_n t)|² / d²
    times = np.linspace(0, t_max, n_times, dtype=np.float64)
    sff: NDArray[np.float64] = np.zeros(n_times)

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
        level_spacing_basis=level_spacing_basis,
        level_spacing_sector=sector,
        level_spacing_sector_dim=sector_dim,
        full_spectrum_level_spacing_ratio=full_r_bar,
    )


def sff_vs_coupling(
    omega: NDArray[np.float64],
    K_topology: NDArray[np.float64],
    k_range: NDArray[np.float64] | None = None,
    t_max: float = 20.0,
    n_times: int = 100,
    *,
    level_spacing_basis: LevelSpacingBasis = "magnetisation",
    magnetisation: int | None = None,
    parity: int | None = None,
    max_dense_gib: float | None = None,
) -> SFFScanResult:
    """Scan finite-size SFF diagnostics across a coupling grid.

    The compatibility field ``chaos_onset_K`` uses a fixed adjacent-gap-ratio
    threshold. It is a heuristic grid crossing only and does not establish a
    Poisson-to-GOE transition or critical coupling.

    Parameters
    ----------
    omega
        Natural-frequency vector.
    K_topology
        Unscaled coupling-topology matrix.
    k_range
        Optional coupling-scale grid; defaults to 15 points from 0.5 to 5.0.
    t_max
        Inclusive SFF time horizon.
    n_times
        Number of SFF time-grid points.
    level_spacing_basis
        ``"magnetisation"`` (default), ``"parity"``, or ``"full"``.
    magnetisation
        Optional magnetisation-sector selector.
    parity
        Optional parity-sector selector.
    max_dense_gib
        Optional fail-closed dense eigensolver budget.

    Returns
    -------
    SFFScanResult
        Finite-grid spacing ratios, gaps, dip depths, and heuristic crossing.

    """
    if k_range is None:
        k_range = np.linspace(0.5, 5.0, 15, dtype=np.float64)

    n_k = len(k_range)
    r_bars = np.zeros(n_k)
    gaps = np.zeros(n_k)
    dip_depths = np.zeros(n_k)

    for idx, kb in enumerate(k_range):
        K = float(kb) * K_topology
        result = compute_sff(
            K,
            omega,
            t_max,
            n_times,
            level_spacing_basis=level_spacing_basis,
            magnetisation=magnetisation,
            parity=parity,
            max_dense_gib=max_dense_gib,
        )
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
        if np.isfinite(r) and r > chaos_threshold:
            chaos_k = float(k_range[i])
            break

    return SFFScanResult(
        k_values=k_range,
        level_spacing_ratios=r_bars,
        spectral_gaps=gaps,
        sff_dip_depth=dip_depths,
        chaos_onset_K=chaos_k,
    )
