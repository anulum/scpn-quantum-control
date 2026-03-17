# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Quantum advantage and scaling benchmarks."""

from .quantum_advantage import (
    AdvantageResult,
    classical_benchmark,
    estimate_crossover,
    quantum_benchmark,
    run_scaling_benchmark,
)

__all__ = [
    "AdvantageResult",
    "classical_benchmark",
    "estimate_crossover",
    "quantum_benchmark",
    "run_scaling_benchmark",
]
