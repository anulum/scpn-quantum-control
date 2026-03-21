# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""SSGF quantum extensions: gradient, cost, outer cycle."""

from .quantum_gradient import QuantumGradientResult, compute_quantum_gradient, quantum_cost

__all__ = [
    "QuantumGradientResult",
    "compute_quantum_gradient",
    "quantum_cost",
]
