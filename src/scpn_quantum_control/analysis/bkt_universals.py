# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""BKT universal amplitude ratio check for p_h1 = 0.72.

Tests whether 0.72 can be expressed as a combination of BKT
universal numbers:

    η = 1/4 (critical exponent)
    2/π ≈ 0.6366 (Nelson-Kosterlitz stiffness jump ratio)
    b ≈ 1.5 (correlation length parameter)
    A ≈ 0.8983 (Hasenbusch-Pinn amplitude, XY model on square lattice)

Candidate expressions:
    1. η + 2/π ≈ 0.25 + 0.637 = 0.887 (too high)
    2. (2/π)^{1/η} = (2/π)^4 ≈ 0.164 (too low)
    3. 1 - η = 0.75 (close but 4% off)
    4. 2/π + 1/2π ≈ 0.637 + 0.159 = 0.796 (too high)
    5. BKT bound-pair: (4/3)(L^{3/4}-1)/(L-1) at L=4 → 0.813 (13% off)
    6. exp(-η) ≈ 0.779 (8% off)
    7. Hasenbusch-Pinn: A × (2/π)^{1/2} = 0.8983 × 0.798 = 0.717 (**3% off**)

Expression #7 is the most promising: 0.717 ≈ 0.72 (within 0.4%).
This suggests p_h1 = A_HP × sqrt(2/π) where A_HP is the Hasenbusch-Pinn
universal amplitude for the 2D XY model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# BKT universal constants
ETA_BKT = 0.25  # critical exponent
NK_RATIO = 2.0 / np.pi  # Nelson-Kosterlitz ≈ 0.6366
HASENBUSCH_PINN = 0.8983  # universal amplitude (square lattice XY)
CORRELATION_B = 1.5  # ξ ~ exp(b/sqrt(T-T_BKT))

# Target
P_H1_TARGET = 0.72


@dataclass
class UniversalCheckResult:
    """Result of BKT universal amplitude ratio check."""

    expression: str
    value: float
    deviation: float  # |value - 0.72|
    relative_deviation_pct: float  # |value - 0.72| / 0.72 × 100


@dataclass
class BKTUniversalsSummary:
    """Summary of all candidate expressions for p_h1."""

    candidates: list[UniversalCheckResult]
    best_expression: str
    best_value: float
    best_deviation: float


def check_all_candidates() -> BKTUniversalsSummary:
    """Evaluate all candidate expressions for p_h1 = 0.72."""
    candidates: list[UniversalCheckResult] = []

    expressions = {
        "1 - eta": 1.0 - ETA_BKT,
        "exp(-eta)": np.exp(-ETA_BKT),
        "2/pi + eta": NK_RATIO + ETA_BKT,
        "(2/pi)^{1/2}": np.sqrt(NK_RATIO),
        "A_HP * (2/pi)^{1/2}": HASENBUSCH_PINN * np.sqrt(NK_RATIO),
        "A_HP * eta * (2/pi)": HASENBUSCH_PINN * ETA_BKT * NK_RATIO,
        "bound-pair n=16": (4.0 / 3.0) * (4.0**0.75 - 1.0) / 3.0,
        "eta / (1 - 2/pi)": ETA_BKT / (1.0 - NK_RATIO),
        "1 - (2/pi)^2": 1.0 - NK_RATIO**2,
        "pi/4 - 1/(4pi)": np.pi / 4 - 1 / (4 * np.pi),
    }

    for name, value in expressions.items():
        dev = abs(value - P_H1_TARGET)
        rel = dev / P_H1_TARGET * 100
        candidates.append(
            UniversalCheckResult(
                expression=name,
                value=float(value),
                deviation=float(dev),
                relative_deviation_pct=float(rel),
            )
        )

    candidates.sort(key=lambda c: c.deviation)
    best = candidates[0]

    return BKTUniversalsSummary(
        candidates=candidates,
        best_expression=best.expression,
        best_value=best.value,
        best_deviation=best.deviation,
    )
