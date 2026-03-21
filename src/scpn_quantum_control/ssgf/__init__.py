# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""SSGF quantum extensions: gradient, cost, outer cycle."""

from .quantum_costs import QuantumCosts, compute_quantum_costs
from .quantum_gradient import QuantumGradientResult, compute_quantum_gradient, quantum_cost
from .quantum_outer_cycle import OuterCycleResult, quantum_outer_cycle
from .quantum_spectral import SpectralBridgeResult, spectral_bridge_analysis

__all__ = [
    "QuantumCosts",
    "compute_quantum_costs",
    "QuantumGradientResult",
    "compute_quantum_gradient",
    "quantum_cost",
    "OuterCycleResult",
    "quantum_outer_cycle",
    "SpectralBridgeResult",
    "spectral_bridge_analysis",
]
