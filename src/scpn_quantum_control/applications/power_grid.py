# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Power Grid
"""Power grid synchronisation benchmark using IEEE test cases.

The Kuramoto model naturally describes power grid frequency
synchronisation (Dörfler & Bullo, Automatica 2014):

    dδ_i/dt = ω_i - (P_i / M_i) + Σ_j (V_i V_j B_ij / M_i) sin(δ_j - δ_i)

where δ_i = rotor angle, ω_i = nominal frequency deviation,
P_i = mechanical power, M_i = inertia, B_ij = susceptance.

The coupling matrix K_ij = V_i V_j B_ij / M_i maps directly to
the Kuramoto K_nm with:
    K_nm ↔ electrical susceptance (weighted by voltage and inertia)
    ω_nm ↔ frequency deviation from 50/60 Hz

IEEE test systems provide public benchmark coupling topologies:
    - IEEE 5-bus: 5 generators, textbook example
    - IEEE 14-bus: 5 generators + 9 loads (14 nodes total)

If the SCPN K_nm structure (exponential-decay all-to-all) matches
the power grid coupling pattern, Gap 1 is partially closed.

Ref: IEEE PES Test Feeder Working Group, public domain data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from scpn_quantum_control.bridge.qpu_data_artifact import (
    ALL_SOURCE_MODES,
    SYNTHETIC_SOURCE_MODES,
)

# IEEE 5-bus system (Stagg & El-Abiad, 5 generators)
# Susceptance matrix B_ij (per-unit, 100 MVA base)
IEEE_5BUS_SUSCEPTANCE = np.array(
    [
        [0.0, 3.81, 5.17, 0.0, 0.0],
        [3.81, 0.0, 0.0, 3.06, 0.0],
        [5.17, 0.0, 0.0, 0.0, 4.28],
        [0.0, 3.06, 0.0, 0.0, 6.58],
        [0.0, 0.0, 4.28, 6.58, 0.0],
    ]
)

# Generator inertia constants H (seconds), typical values
IEEE_5BUS_INERTIA = np.array([5.0, 4.0, 3.5, 4.5, 3.0])

# Voltage magnitudes (per-unit)
IEEE_5BUS_VOLTAGE = np.array([1.06, 1.04, 1.02, 1.03, 1.01])

# Frequency deviations (Hz) from nominal — small deviations
IEEE_5BUS_FREQ_DEV = np.array([0.0, 0.02, -0.01, 0.015, -0.005])


@dataclass
class PowerGridBenchmarkResult:
    """Power grid vs SCPN comparison result."""

    n_generators: int
    topology_correlation: float  # Spearman rho of coupling matrices
    coupling_ratio: float  # mean(K_grid) / mean(K_scpn)
    frequency_correlation: float  # correlation of frequency vectors
    grid_name: str
    summary: str
    source_mode: str
    publication_safe: bool


def _validated_square_matrix(
    matrix: np.ndarray,
    name: str,
    *,
    require_grid_coupling: bool = False,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix.")
    if values.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two coupled grid nodes.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    if require_grid_coupling:
        if np.any(values < 0.0):
            raise ValueError(f"{name} values must be non-negative.")
        if not np.allclose(values, values.T, atol=1e-12):
            raise ValueError(f"{name} must be symmetric.")
        if not np.allclose(np.diag(values), 0.0, atol=1e-12):
            raise ValueError(f"{name} diagonal must be zero.")
    return values


def _validated_frequency_vector(
    frequencies: np.ndarray,
    n_nodes: int,
    name: str,
    matrix_name: str,
) -> np.ndarray:
    values = np.asarray(frequencies, dtype=float)
    if values.ndim != 1 or values.shape != (n_nodes,):
        raise ValueError(f"{name} must match {matrix_name} node count.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _finite_correlation(value: float) -> float:
    if np.isnan(value):
        return 0.0
    return float(value)


def ieee_5bus_coupling_matrix(
    *,
    allow_builtin_reference: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Kuramoto coupling matrix from IEEE 5-bus data.

    K_ij = V_i × V_j × B_ij / (2 × H_i × ω_0)
    where ω_0 = 2π × 60 Hz (US standard).
    """
    if not allow_builtin_reference:
        raise RuntimeError(
            "Refusing built-in IEEE 5-bus reference without allow_builtin_reference=True. "
            "Pass curated grid_coupling and grid_frequencies to power_grid_benchmark "
            "for explicit provenance."
        )
    omega_0 = 2 * np.pi * 60.0
    n = 5
    K: np.ndarray = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if IEEE_5BUS_SUSCEPTANCE[i, j] > 0:
                K[i, j] = (
                    IEEE_5BUS_VOLTAGE[i]
                    * IEEE_5BUS_VOLTAGE[j]
                    * IEEE_5BUS_SUSCEPTANCE[i, j]
                    / (2.0 * IEEE_5BUS_INERTIA[i] * omega_0)
                )

    # Symmetrise (inertia makes it asymmetric; use geometric mean)
    K = (K + K.T) / 2.0
    omega = IEEE_5BUS_FREQ_DEV.copy()
    return K, omega


def power_grid_benchmark(
    K_scpn: np.ndarray,
    omega_scpn: np.ndarray,
    grid_name: str = "IEEE-5bus",
    *,
    grid_coupling: np.ndarray | None = None,
    grid_frequencies: np.ndarray | None = None,
    reference_source_mode: str = "curated",
    allow_builtin_reference: bool = False,
) -> PowerGridBenchmarkResult:
    """Compare SCPN coupling topology with power grid.

    Uses the smaller dimension (min(n_scpn, n_grid)) for comparison.
    """
    if grid_coupling is None or grid_frequencies is None:
        if grid_coupling is not None or grid_frequencies is not None:
            raise ValueError("grid_coupling and grid_frequencies must be supplied together.")
        if grid_name == "IEEE-5bus":
            K_grid, omega_grid = ieee_5bus_coupling_matrix(
                allow_builtin_reference=allow_builtin_reference
            )
            source_mode = "curated"
        else:
            raise ValueError(f"Unknown grid: {grid_name}")
    else:
        source_mode = str(reference_source_mode).strip()
        if source_mode not in ALL_SOURCE_MODES:
            raise ValueError(f"reference_source_mode must be one of {sorted(ALL_SOURCE_MODES)}")
        K_grid = _validated_square_matrix(
            grid_coupling,
            "grid_coupling",
            require_grid_coupling=True,
        )
        omega_grid = _validated_frequency_vector(
            grid_frequencies,
            K_grid.shape[0],
            "grid_frequencies",
            "grid_coupling",
        )
    publication_safe = source_mode not in SYNTHETIC_SOURCE_MODES

    K_scpn = _validated_square_matrix(K_scpn, "K_scpn")
    omega_scpn = _validated_frequency_vector(
        omega_scpn,
        K_scpn.shape[0],
        "omega_scpn",
        "K_scpn",
    )
    K_grid = _validated_square_matrix(
        K_grid,
        "grid_coupling",
        require_grid_coupling=True,
    )
    omega_grid = _validated_frequency_vector(
        omega_grid,
        K_grid.shape[0],
        "grid_frequencies",
        "grid_coupling",
    )

    n_grid = K_grid.shape[0]
    n_scpn = K_scpn.shape[0]
    n = min(n_grid, n_scpn)

    K_g = K_grid[:n, :n]
    K_s = K_scpn[:n, :n]
    omega_g = omega_grid[:n]
    omega_s = omega_scpn[:n]

    # Upper triangle elements for correlation
    triu_idx = np.triu_indices(n, k=1)
    g_flat = K_g[triu_idx]
    s_flat = K_s[triu_idx]

    if len(g_flat) < 3:
        topo_corr = 0.0
    else:
        topo_corr = _finite_correlation(float(spearmanr(g_flat, s_flat).statistic))

    # Coupling ratio
    g_mean = float(np.mean(g_flat[g_flat > 0])) if np.any(g_flat > 0) else 0.0
    s_mean = float(np.mean(s_flat[s_flat > 0])) if np.any(s_flat > 0) else 0.0
    ratio = g_mean / max(s_mean, 1e-15)

    # Frequency correlation
    if n >= 3:
        freq_corr = _finite_correlation(float(np.corrcoef(omega_g, omega_s)[0, 1]))
    else:
        freq_corr = 0.0

    summary = (
        f"SCPN vs {grid_name}: topology ρ={topo_corr:.3f}, "
        f"coupling ratio={ratio:.3f}, freq r={freq_corr:.3f}"
    )

    return PowerGridBenchmarkResult(
        n_generators=n,
        topology_correlation=topo_corr,
        coupling_ratio=ratio,
        frequency_correlation=freq_corr,
        grid_name=grid_name,
        summary=summary,
        source_mode=source_mode,
        publication_safe=publication_safe,
    )
