# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — U(1) Gauge Theory Observables
"""U(1) gauge theory observables for the Kuramoto-XY quantum model."""

from .cft_analysis import CFTResult, cft_analysis, extract_central_charge, find_critical_coupling
from .confinement import ConfinementResult, confinement_analysis, confinement_vs_coupling
from .universality import UniversalityResult, universality_analysis
from .vortex_detector import VortexResult, measure_vortex_density, vortex_density_vs_coupling
from .wilson_loop import WilsonLoopResult, compute_wilson_loops, wilson_loop_expectation

__all__ = [
    "CFTResult",
    "cft_analysis",
    "extract_central_charge",
    "find_critical_coupling",
    "ConfinementResult",
    "confinement_analysis",
    "confinement_vs_coupling",
    "UniversalityResult",
    "universality_analysis",
    "VortexResult",
    "measure_vortex_density",
    "vortex_density_vs_coupling",
    "WilsonLoopResult",
    "compute_wilson_loops",
    "wilson_loop_expectation",
]
