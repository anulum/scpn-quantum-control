# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Josephson Array
"""Josephson junction array mapping for the self-simulation narrative.

A Josephson junction array (JJA) is described by the same XY model
as the Kuramoto oscillators:

    H_JJA = -Σ_{ij} E_J cos(φ_j - φ_i) + (e²/2C) Σ_i n_i²

where φ_i = superconducting phase, n_i = Cooper pair number,
E_J = Josephson energy, C = junction capacitance.

The quantum XY Hamiltonian is:
    H_XY = -Σ_{ij} K_ij (X_i X_j + Y_i Y_j) - Σ_i ω_i Z_i

The mapping: K_ij ↔ E_J (Josephson energy), ω_i ↔ e²/2C (charging).

This creates the self-simulation narrative: a superconducting quantum
computer (which IS a JJA) simulating the XY model (which IS a JJA).
The quantum hardware is literally simulating itself.

Transmon qubits (used in IBM Heron) operate in the E_J >> E_C regime,
where the JJA reduces to the XY model. The coupling between transmons
via capacitors or resonators maps to K_nm.

Ref: Fazio & van der Zant, Physics Reports 355, 235 (2001).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

# Typical transmon parameters (IBM Heron r2, approximate)
TRANSMON_EJ = 15.0  # GHz, Josephson energy
TRANSMON_EC = 0.25  # GHz, charging energy
TRANSMON_EJ_EC_RATIO = TRANSMON_EJ / TRANSMON_EC  # ~60, deep transmon regime

# Nearest-neighbour coupling via bus resonator (typical)
TRANSMON_COUPLING = 0.015  # GHz, exchange coupling J


@dataclass
class JosephsonBenchmarkResult:
    """Josephson junction array vs SCPN comparison."""

    n_junctions: int
    topology_correlation: float
    ej_ec_ratio: float  # E_J/E_C for the mapped system
    coupling_ratio: float  # J_JJA / K_scpn
    frequency_correlation: float
    is_transmon_regime: bool  # E_J/E_C > 20
    summary: str


def jja_coupling_matrix(
    n: int,
    ej: float = TRANSMON_EJ,
    ec: float = TRANSMON_EC,
    j_coupling: float = TRANSMON_COUPLING,
    topology: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Build Kuramoto-equivalent coupling matrix from JJA parameters.

    Topologies:
        - "linear": nearest-neighbour chain
        - "heavy_hex": IBM heavy-hex connectivity (approximate)
        - "all_to_all": complete graph (capacitive bus)
    """
    K = np.zeros((n, n))
    omega = np.full(n, ec)  # charging energy as frequency

    if topology == "linear":
        for i in range(n - 1):
            K[i, i + 1] = K[i + 1, i] = j_coupling
    elif topology == "heavy_hex":
        # Approximate heavy-hex: each qubit coupled to 2-3 neighbours
        for i in range(n - 1):
            K[i, i + 1] = K[i + 1, i] = j_coupling
        for i in range(0, n - 2, 2):
            K[i, i + 2] = K[i + 2, i] = j_coupling * 0.5
    elif topology == "all_to_all":
        for i in range(n):
            for j in range(i + 1, n):
                dist = abs(j - i)
                K[i, j] = K[j, i] = j_coupling * np.exp(-0.3 * dist)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    return K, omega


def josephson_benchmark(
    K_scpn: np.ndarray,
    omega_scpn: np.ndarray,
    topology: str = "all_to_all",
) -> JosephsonBenchmarkResult:
    """Compare SCPN K_nm with Josephson junction array coupling."""
    n = K_scpn.shape[0]
    K_jja, omega_jja = jja_coupling_matrix(n, topology=topology)

    triu_idx = np.triu_indices(n, k=1)
    jja_flat = K_jja[triu_idx]
    scpn_flat = K_scpn[triu_idx]

    # Filter to non-zero pairs for correlation
    mask = (jja_flat > 0) | (scpn_flat > 0)
    if np.sum(mask) >= 3:
        topo_corr = float(spearmanr(jja_flat[mask], scpn_flat[mask]).statistic)
    else:
        topo_corr = 0.0

    jja_mean = float(np.mean(jja_flat[jja_flat > 0])) if np.any(jja_flat > 0) else 0.0
    scpn_mean = float(np.mean(scpn_flat[scpn_flat > 0])) if np.any(scpn_flat > 0) else 0.0
    ratio = jja_mean / max(scpn_mean, 1e-15)

    if n >= 3:
        freq_corr = float(np.corrcoef(omega_jja[:n], omega_scpn[:n])[0, 1])
        if np.isnan(freq_corr):
            freq_corr = 0.0
    else:
        freq_corr = 0.0

    ej_ec = TRANSMON_EJ / TRANSMON_EC
    is_transmon = ej_ec > 20

    summary = (
        f"SCPN vs JJA ({topology}): topology ρ={topo_corr:.3f}, "
        f"coupling ratio={ratio:.4f}, E_J/E_C={ej_ec:.1f} ({'transmon' if is_transmon else 'charge'})"
    )

    return JosephsonBenchmarkResult(
        n_junctions=n,
        topology_correlation=topo_corr,
        ej_ec_ratio=ej_ec,
        coupling_ratio=ratio,
        frequency_correlation=freq_corr,
        is_transmon_regime=is_transmon,
        summary=summary,
    )
