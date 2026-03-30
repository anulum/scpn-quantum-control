# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Fmo Benchmark
"""FMO photosynthetic complex benchmark against SCPN coupling.

The Fenna-Matthews-Olson complex is a 7-chromophore coupled oscillator
network with published coupling parameters. If our K_nm structure matches
FMO coupling topology, it validates the framework against a physical system.

FMO Hamiltonian (Adolphs & Renger, Biophysical Journal 91, 2778, 2006):
  H_FMO = Σ_n ε_n |n⟩⟨n| + Σ_{n≠m} J_nm |n⟩⟨m|

where ε_n are site energies (cm⁻¹) and J_nm are dipole-dipole couplings.

The mapping to our XY Hamiltonian:
  ω_n ↔ ε_n (natural frequencies ↔ site energies)
  K_nm ↔ J_nm (coupling matrix ↔ dipole couplings)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# FMO site energies (cm⁻¹, Adolphs & Renger 2006, Table 1)
# Relative to average: ε_n - ⟨ε⟩
FMO_SITE_ENERGIES = np.array(
    [
        215.0,  # BChl 1
        220.0,  # BChl 2
        0.0,  # BChl 3 (reference)
        125.0,  # BChl 4
        450.0,  # BChl 5
        330.0,  # BChl 6
        270.0,  # BChl 7
    ]
)

# FMO dipole-dipole coupling matrix (cm⁻¹, Adolphs & Renger 2006, Table 2)
# Symmetric, zero diagonal
FMO_COUPLING = np.array(
    [
        [0.0, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
        [-87.7, 0.0, 30.8, 8.2, 0.7, 11.8, 4.3],
        [5.5, 30.8, 0.0, -53.5, -2.2, -9.6, 6.0],
        [-5.9, 8.2, -53.5, 0.0, -70.7, -17.0, -63.3],
        [6.7, 0.7, -2.2, -70.7, 0.0, 81.1, -1.3],
        [-13.7, 11.8, -9.6, -17.0, 81.1, 0.0, 39.7],
        [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 0.0],
    ]
)


def fmo_coupling_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Return the FMO coupling matrix and site energies.

    Scaled to natural units: energies in rad/ps (divide cm⁻¹ by 5309).
    """
    # Convert cm⁻¹ to rad/ps: 1 cm⁻¹ ≈ 0.0001884 rad/ps × 2π ≈ 0.001884 rad/ps
    cm_to_radps = 2.0 * np.pi * 2.998e10 * 1e-12  # 2π × c(cm/s) × 10⁻¹² s/ps
    omega = FMO_SITE_ENERGIES * cm_to_radps
    K = np.abs(FMO_COUPLING) * cm_to_radps  # absolute values for XY mapping
    np.fill_diagonal(K, 0.0)
    return K, omega


@dataclass
class FMOBenchmarkResult:
    """Comparison between SCPN and FMO coupling structures."""

    topology_correlation: float
    frequency_correlation: float
    coupling_ratio: float
    frequency_ratio: float
    n_oscillators: int
    summary: str


def fmo_benchmark(
    K_scpn: np.ndarray,
    omega_scpn: np.ndarray,
) -> FMOBenchmarkResult:
    """Compare SCPN coupling structure (7-oscillator subset) against FMO.

    Computes:
    1. Topology correlation: Spearman rank correlation of off-diagonal K values
    2. Frequency correlation: Pearson correlation of natural frequencies
    3. Coupling ratio: mean(K_scpn) / mean(K_fmo)
    4. Frequency ratio: mean(ω_scpn) / mean(ω_fmo)
    """
    K_fmo, omega_fmo = fmo_coupling_matrix()

    n = min(len(omega_scpn), 7)
    K_s = K_scpn[:n, :n]
    omega_s = omega_scpn[:n]
    K_f = K_fmo[:n, :n]
    omega_f = omega_fmo[:n]

    # Flatten upper triangles for coupling comparison
    idx = np.triu_indices(n, k=1)
    k_s_flat = K_s[idx]
    k_f_flat = K_f[idx]

    # Topology correlation (Spearman rank)
    from scipy.stats import spearmanr

    rho_topo, _ = spearmanr(k_s_flat, k_f_flat)

    # Frequency correlation (Pearson)
    omega_s_norm = (omega_s - np.mean(omega_s)) / max(np.std(omega_s), 1e-10)
    omega_f_norm = (omega_f - np.mean(omega_f)) / max(np.std(omega_f), 1e-10)
    rho_freq = float(np.corrcoef(omega_s_norm, omega_f_norm)[0, 1])

    # Ratios
    coupling_ratio = float(np.mean(k_s_flat) / max(float(np.mean(k_f_flat)), 1e-10))
    freq_ratio = float(np.mean(np.abs(omega_s)) / max(float(np.mean(np.abs(omega_f))), 1e-10))

    # Interpret
    if abs(rho_topo) > 0.5:
        topo_verdict = "significant correlation"
    elif abs(rho_topo) > 0.3:
        topo_verdict = "weak correlation"
    else:
        topo_verdict = "no correlation"

    summary = (
        f"SCPN vs FMO ({n} oscillators): "
        f"topology rho={rho_topo:.3f} ({topo_verdict}), "
        f"frequency r={rho_freq:.3f}, "
        f"coupling scale {coupling_ratio:.2e}x, "
        f"frequency scale {freq_ratio:.2e}x"
    )

    return FMOBenchmarkResult(
        topology_correlation=float(rho_topo),
        frequency_correlation=rho_freq,
        coupling_ratio=coupling_ratio,
        frequency_ratio=freq_ratio,
        n_oscillators=n,
        summary=summary,
    )
