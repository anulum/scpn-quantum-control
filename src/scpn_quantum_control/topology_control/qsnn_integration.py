# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology Control QSNN Integration
"""QSNN dynamic-coupling policy hooks for persistent-H1 control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .objectives import CouplingTopologyObjective
from .optimizers import ProjectedSPSAOptimizer, TopologyOptimisationTrace

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass
class TopologicalDynamicCouplingPolicy:
    """Apply a topology-control objective to recurrent QSNN weights."""

    objective: CouplingTopologyObjective
    optimizer: ProjectedSPSAOptimizer
    last_trace: TopologyOptimisationTrace | None = None

    def apply(self, recurrent_weights: NDArray[np.float64]) -> FloatArray:
        """Return projected/optimised recurrent weights without external dependencies."""
        trace = self.optimizer.optimise(recurrent_weights, self.objective)
        self.last_trace = trace
        result: FloatArray = trace.final_matrix.copy()
        np.fill_diagonal(result, 0.0)
        return result


__all__ = ["FloatArray", "TopologicalDynamicCouplingPolicy"]
