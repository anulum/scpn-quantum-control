# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology Control Constraints
"""Constraint ledgers and projection routines for coupling graph control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
Edge = tuple[int, int]
FrozenEdges = dict[Edge, float]
SignPolicy = Literal["nonnegative", "signed", "fixed_sign"]


def canonical_edge(i: int, j: int) -> Edge:
    """Return an undirected edge key."""

    if i == j:
        raise ValueError("self edges are not valid coupling edges")
    return (i, j) if i < j else (j, i)


@dataclass(frozen=True)
class CouplingGraphBounds:
    """Uniform lower/upper bounds for off-diagonal coupling values."""

    lower: float = 0.0
    upper: float = 1.0

    def __post_init__(self) -> None:
        if self.upper <= self.lower:
            raise ValueError("upper bound must exceed lower bound")


@dataclass(frozen=True)
class HardwareEmbeddingConstraint:
    """Allowed logical edges for a hardware layout."""

    edges: frozenset[Edge] = field(default_factory=frozenset)

    @classmethod
    def from_edges(
        cls, edges: set[Edge] | frozenset[Edge] | tuple[Edge, ...]
    ) -> HardwareEmbeddingConstraint:
        """Build a hardware embedding constraint from undirected edge pairs."""
        return cls(frozenset(canonical_edge(i, j) for i, j in edges))

    def mask(self, n_nodes: int) -> FloatArray:
        """Return a symmetric binary mask for allowed hardware edges."""

        mask = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        for i, j in self.edges:
            if i >= n_nodes or j >= n_nodes:
                raise ValueError("hardware edge index exceeds graph size")
            mask[i, j] = 1.0
            mask[j, i] = 1.0
        return cast(FloatArray, mask)


@dataclass(frozen=True)
class ConstraintViolation:
    """Per-constraint violation magnitudes."""

    total_weight: float = 0.0
    connectivity: float = 0.0
    frozen_edges: float = 0.0
    hardware_edges: float = 0.0
    bounds: float = 0.0

    @property
    def total(self) -> float:
        """Total scalar violation."""

        return float(
            self.total_weight
            + self.connectivity
            + self.frozen_edges
            + self.hardware_edges
            + self.bounds
        )


@dataclass(frozen=True)
class TopologyConstraintLedger:
    """Complete projection and validation policy for a coupling graph."""

    bounds: CouplingGraphBounds = field(default_factory=CouplingGraphBounds)
    sign_policy: SignPolicy = "nonnegative"
    total_weight: tuple[float, float] | None = None
    frozen_edges: FrozenEdges = field(default_factory=dict)
    hardware_edges: set[Edge] | frozenset[Edge] | None = None
    algebraic_connectivity_min: float = 0.0
    fixed_sign_reference: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.sign_policy not in {"nonnegative", "signed", "fixed_sign"}:
            raise ValueError("invalid sign_policy")
        if self.total_weight is not None:
            low, high = self.total_weight
            if low < 0.0 or high < low:
                raise ValueError("total_weight must be an ordered non-negative interval")
        if self.algebraic_connectivity_min < 0.0:
            raise ValueError("algebraic_connectivity_min must be non-negative")

    def _as_matrix(self, matrix: NDArray[np.float64]) -> FloatArray:
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("coupling matrix must be square")
        if not np.all(np.isfinite(arr)):
            raise ValueError("coupling matrix must contain only finite values")
        out: FloatArray = arr.copy()
        return out

    def project(self, matrix: NDArray[np.float64]) -> FloatArray:
        """Project a candidate matrix onto hard shape, sign, edge, and budget constraints."""

        projected = self._as_matrix(matrix)
        projected = (projected + projected.T) / 2.0
        np.fill_diagonal(projected, 0.0)

        if self.sign_policy == "nonnegative":
            projected = np.maximum(projected, 0.0)
        elif self.sign_policy == "fixed_sign":
            if self.fixed_sign_reference is None:
                raise ValueError("fixed_sign policy requires fixed_sign_reference")
            reference = self._as_matrix(self.fixed_sign_reference)
            projected = np.sign(reference) * np.abs(projected)

        projected = np.clip(projected, self.bounds.lower, self.bounds.upper)
        np.fill_diagonal(projected, 0.0)

        if self.hardware_edges is not None:
            mask = HardwareEmbeddingConstraint.from_edges(self.hardware_edges).mask(
                projected.shape[0]
            )
            projected *= mask

        for (i, j), value in self.frozen_edges.items():
            edge = canonical_edge(i, j)
            projected[edge[0], edge[1]] = float(value)
            projected[edge[1], edge[0]] = float(value)

        if self.total_weight is not None:
            projected = self._project_total_weight(projected)

        symmetric: FloatArray = (projected + projected.T) / 2.0
        np.fill_diagonal(symmetric, 0.0)
        return symmetric

    def _project_total_weight(self, matrix: FloatArray) -> FloatArray:
        if self.total_weight is None:
            return matrix
        low, high = self.total_weight
        total = float(np.sum(matrix))
        if low <= total <= high or total <= 1e-15:
            return matrix
        target = low if total < low else high

        adjustable_mask = np.ones_like(matrix, dtype=bool)
        np.fill_diagonal(adjustable_mask, False)
        if self.hardware_edges is not None:
            hardware_mask = (
                HardwareEmbeddingConstraint.from_edges(self.hardware_edges).mask(matrix.shape[0])
                > 0
            )
            adjustable_mask &= hardware_mask
        for i, j in self.frozen_edges:
            edge = canonical_edge(i, j)
            adjustable_mask[edge[0], edge[1]] = False
            adjustable_mask[edge[1], edge[0]] = False

        adjustable_sum = float(np.sum(matrix[adjustable_mask]))
        frozen_sum = total - adjustable_sum
        target_adjustable = max(target - frozen_sum, 0.0)
        if adjustable_sum <= 1e-15:
            return matrix
        scaled = matrix.copy()
        scaled[adjustable_mask] *= target_adjustable / adjustable_sum
        return cast(FloatArray, np.clip(scaled, self.bounds.lower, self.bounds.upper))

    def violations(self, matrix: NDArray[np.float64]) -> ConstraintViolation:
        """Return violation magnitudes without mutating the candidate matrix."""

        K = self._as_matrix(matrix)
        bounds_low = np.maximum(self.bounds.lower - K, 0.0)
        bounds_high = np.maximum(K - self.bounds.upper, 0.0)
        bounds_violation = float(np.sum(bounds_low + bounds_high))

        total_violation = 0.0
        if self.total_weight is not None:
            low, high = self.total_weight
            total = float(np.sum(K))
            total_violation = max(low - total, total - high, 0.0)

        frozen_violation = 0.0
        for (i, j), value in self.frozen_edges.items():
            edge = canonical_edge(i, j)
            frozen_violation += abs(float(K[edge[0], edge[1]]) - float(value))
            frozen_violation += abs(float(K[edge[1], edge[0]]) - float(value))

        hardware_violation = 0.0
        if self.hardware_edges is not None:
            mask = HardwareEmbeddingConstraint.from_edges(self.hardware_edges).mask(K.shape[0])
            hardware_violation = float(
                np.sum(np.abs(K) * (1.0 - mask) * (1.0 - np.eye(K.shape[0])))
            )

        connectivity_violation = 0.0
        if self.algebraic_connectivity_min > 0.0:
            lambda2 = algebraic_connectivity(K)
            connectivity_violation = max(self.algebraic_connectivity_min - lambda2, 0.0)

        return ConstraintViolation(
            total_weight=float(total_violation),
            connectivity=float(connectivity_violation),
            frozen_edges=float(frozen_violation),
            hardware_edges=float(hardware_violation),
            bounds=float(bounds_violation),
        )


def algebraic_connectivity(matrix: NDArray[np.float64]) -> float:
    """Weighted graph algebraic connectivity using absolute edge weights."""

    K = np.asarray(matrix, dtype=np.float64)
    graph = nx.Graph()
    n = K.shape[0]
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            weight = abs(float(K[i, j]))
            if weight > 1e-15:
                graph.add_edge(i, j, weight=weight)
    if n < 2 or not nx.is_connected(graph):
        return 0.0
    laplacian = nx.laplacian_matrix(graph, weight="weight").toarray()
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return float(np.sort(eigenvalues)[1])


__all__ = [
    "CouplingGraphBounds",
    "ConstraintViolation",
    "Edge",
    "FrozenEdges",
    "HardwareEmbeddingConstraint",
    "SignPolicy",
    "TopologyConstraintLedger",
    "algebraic_connectivity",
    "canonical_edge",
]
