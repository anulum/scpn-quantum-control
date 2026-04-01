# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Cross Domain
"""Cross-domain validation: compare SCPN K_nm against all physical systems.

Runs all 5 physical system benchmarks and produces a summary table:
    1. FMO photosynthetic complex (7 chromophores)
    2. IEEE 5-bus power grid (5 generators)
    3. Josephson junction array (transmon coupling)
    4. EEG alpha-band neural oscillators (8 channels)
    5. ITER MHD mode coupling (8 modes)

If any system shows strong topology correlation (ρ > 0.5), Gap 1
is partially closed. If multiple show moderate correlation (ρ > 0.3),
the K_nm exponential-decay pattern is a universal feature of coupled
oscillator systems — which would be the actual finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from .eeg_benchmark import eeg_benchmark
from .fmo_benchmark import fmo_benchmark
from .iter_benchmark import iter_benchmark
from .josephson_array import josephson_benchmark
from .power_grid import power_grid_benchmark


@dataclass
class CrossDomainResult:
    """Cross-domain validation summary."""

    system_names: list[str]
    topology_correlations: list[float]
    frequency_correlations: list[float]
    best_system: str
    best_correlation: float
    mean_correlation: float
    n_above_threshold: int  # systems with |ρ| > 0.3


def run_cross_domain_validation(
    n_max: int = 16,
) -> CrossDomainResult:
    """Run all 5 physical system benchmarks against SCPN K_nm.

    Uses the appropriate oscillator count for each system.
    """
    results_topo: list[float] = []
    results_freq: list[float] = []
    names: list[str] = []

    # FMO (7 oscillators)
    K7 = build_knm_paper27(L=7)
    omega7 = OMEGA_N_16[:7]
    fmo = fmo_benchmark(K7, omega7)
    names.append("FMO (photosynthesis)")
    results_topo.append(fmo.topology_correlation)
    results_freq.append(fmo.frequency_correlation)

    # Power grid (5 oscillators)
    K5 = build_knm_paper27(L=5)
    omega5 = OMEGA_N_16[:5]
    grid = power_grid_benchmark(K5, omega5)
    names.append("IEEE 5-bus (power grid)")
    results_topo.append(grid.topology_correlation)
    results_freq.append(grid.frequency_correlation)

    # Josephson (5 oscillators, all-to-all)
    jja = josephson_benchmark(K5, omega5, topology="all_to_all")
    names.append("JJA (self-simulation)")
    results_topo.append(jja.topology_correlation)
    results_freq.append(jja.frequency_correlation)

    # EEG (8 oscillators)
    K8 = build_knm_paper27(L=8)
    omega8 = OMEGA_N_16[:8]
    eeg = eeg_benchmark(K8, omega8)
    names.append("EEG alpha (neuroscience)")
    results_topo.append(eeg.topology_correlation)
    results_freq.append(eeg.frequency_correlation)

    # ITER (8 modes)
    it = iter_benchmark(K8, omega8)
    names.append("ITER MHD (fusion)")
    results_topo.append(it.topology_correlation)
    results_freq.append(it.frequency_correlation)

    best_idx = int(np.argmax(np.abs(results_topo)))
    above = sum(1 for r in results_topo if abs(r) > 0.3)

    return CrossDomainResult(
        system_names=names,
        topology_correlations=results_topo,
        frequency_correlations=results_freq,
        best_system=names[best_idx],
        best_correlation=results_topo[best_idx],
        mean_correlation=float(np.mean(np.abs(results_topo))),
        n_above_threshold=above,
    )
