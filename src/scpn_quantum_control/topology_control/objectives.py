# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology Control Objectives
"""Degeneracy-safe persistent-H1 objectives for coupling graph control."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .complexes import H1Summary, PersistentHomologyBackend, build_coupling_distance_matrix
from .constraints import TopologyConstraintLedger, algebraic_connectivity

FloatArray: TypeAlias = NDArray[np.float64]


class DegeneracyMode(Enum):
    """Graph states that make unconstrained p_h1 minimisation scientifically invalid."""

    NONE = "none"
    ZERO_GRAPH = "zero_graph"
    COMPLETE_UNIFORM = "complete_uniform"
    DISCONNECTED = "disconnected"


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """Auditable scalar objective decomposition."""

    total: float
    terms: dict[str, float]
    h1_summary: H1Summary
    degeneracy_mode: DegeneracyMode
    projected_matrix: FloatArray
    constraint_violation_total: float


@dataclass(frozen=True)
class CouplingTopologyObjective:
    """Persistent-H1 objective with hard projection and anti-collapse penalties."""

    ph_backend: PersistentHomologyBackend
    ledger: TopologyConstraintLedger
    h1_target: float = 0.0
    h1_weight: float = 1.0
    source_matrix: np.ndarray | None = None
    source_distance_weight: float = 0.0
    connectivity_weight: float = 10.0
    constraint_weight: float = 100.0
    degeneracy_penalty: float = 1_000.0
    persistence_threshold: float = 0.1
    allow_degenerate: bool = False
    allow_approximate_ph_backend: bool = False
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.h1_weight < 0.0:
            raise ValueError("h1_weight must be non-negative")
        if self.source_distance_weight < 0.0:
            raise ValueError("source_distance_weight must be non-negative")
        if self.h1_target < 0.0:
            raise ValueError("h1_target must be non-negative")

    def evaluate(self, matrix: np.ndarray) -> ObjectiveBreakdown:
        """Evaluate projected coupling matrix and return decomposed score."""

        if (
            bool(getattr(self.ph_backend, "approximate", False))
            and not self.allow_approximate_ph_backend
        ):
            raise ValueError(
                "approximate persistent-homology backend requires explicit "
                "allow_approximate_ph_backend=True"
            )

        projected = self.ledger.project(matrix)
        distance = build_coupling_distance_matrix(projected)
        h1 = self.ph_backend.compute(distance, persistence_threshold=self.persistence_threshold)
        violations = self.ledger.violations(projected)
        degeneracy = classify_degeneracy(projected)

        terms: dict[str, float] = {}
        terms["h1"] = self.h1_weight * abs(float(h1.p_h1) - self.h1_target)
        terms["constraint_penalty"] = self.constraint_weight * violations.total
        terms["connectivity_penalty"] = self.connectivity_weight * max(
            self.ledger.algebraic_connectivity_min - algebraic_connectivity(projected),
            0.0,
        )
        terms["source_distance"] = self._source_distance(projected)
        if degeneracy is DegeneracyMode.NONE or self.allow_degenerate:
            terms["degeneracy_penalty"] = 0.0
        else:
            terms["degeneracy_penalty"] = self.degeneracy_penalty

        total = float(sum(terms.values()))
        return ObjectiveBreakdown(
            total=total,
            terms=terms,
            h1_summary=h1,
            degeneracy_mode=degeneracy,
            projected_matrix=projected,
            constraint_violation_total=violations.total,
        )

    def _source_distance(self, matrix: FloatArray) -> float:
        if self.source_matrix is None or self.source_distance_weight == 0.0:
            return 0.0
        source = self.ledger.project(self.source_matrix)
        denom = max(float(np.linalg.norm(source)), 1e-15)
        return float(self.source_distance_weight * np.linalg.norm(matrix - source) / denom)


def classify_degeneracy(matrix: np.ndarray) -> DegeneracyMode:
    """Classify trivial coupling graphs that can fake a low-H1 score."""

    K = np.asarray(matrix, dtype=np.float64)
    off_diag = K[~np.eye(K.shape[0], dtype=bool)]
    if np.all(np.abs(off_diag) <= 1e-12):
        return DegeneracyMode.ZERO_GRAPH
    positive = np.abs(off_diag[off_diag > 1e-12])
    if positive.size == off_diag.size and positive.size > 0 and float(np.std(positive)) <= 1e-12:
        return DegeneracyMode.COMPLETE_UNIFORM
    if algebraic_connectivity(K) <= 1e-12:
        return DegeneracyMode.DISCONNECTED
    return DegeneracyMode.NONE


def objective_sha256_payload(objective: CouplingTopologyObjective) -> dict[str, object]:
    """Stable serialisable subset used for objective digests."""

    source_shape: tuple[int, ...] | None = None
    if objective.source_matrix is not None:
        source_shape = tuple(int(x) for x in np.asarray(objective.source_matrix).shape)
    return {
        "backend": objective.ph_backend.name,
        "h1_target": objective.h1_target,
        "h1_weight": objective.h1_weight,
        "source_distance_weight": objective.source_distance_weight,
        "connectivity_weight": objective.connectivity_weight,
        "constraint_weight": objective.constraint_weight,
        "degeneracy_penalty": objective.degeneracy_penalty,
        "persistence_threshold": objective.persistence_threshold,
        "allow_degenerate": objective.allow_degenerate,
        "allow_approximate_ph_backend": objective.allow_approximate_ph_backend,
        "source_shape": source_shape,
        "metadata": dict(sorted(objective.metadata.items())),
    }


__all__ = [
    "CouplingTopologyObjective",
    "DegeneracyMode",
    "FloatArray",
    "ObjectiveBreakdown",
    "classify_degeneracy",
    "objective_sha256_payload",
]
