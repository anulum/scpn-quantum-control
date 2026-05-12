# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — TCBO coupling-weighted complex
"""Coupling-weighted simplicial complex for TCBO p_h1 reconstruction.

The TCBO roadmap calls for a construction whose edge filtration is built from

    w_ij = K_ij * |cos(theta_j - theta_i)|

rather than from delay-embedded Vietoris-Rips point clouds. This module builds
that weighted graph, lifts it to the 2-dimensional flag complex, and computes
the first Betti number over GF(2). It deliberately does not promote
``p_h1 = 0.72``; promotion requires a preregistered dataset plus uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class TCBOWeightedComplexResult:
    """Single threshold result for the TCBO coupling-weighted flag complex."""

    threshold: float
    beta_0: int
    beta_1: int
    p_h1: float
    n_nodes: int
    n_edges: int
    n_triangles: int
    max_h1: int
    edge_weights: np.ndarray


@dataclass(frozen=True)
class TCBOWeightedThresholdScan:
    """Threshold scan summary for the TCBO coupling-weighted complex."""

    results: tuple[TCBOWeightedComplexResult, ...]
    target_p_h1: float
    best_threshold: float
    best_p_h1: float
    best_abs_error: float
    promotes_target: bool


def _validated_coupling_matrix(K: np.ndarray) -> np.ndarray:
    coupling = np.asarray(K, dtype=float)
    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
        raise ValueError("K must be a square 2-D coupling matrix.")
    if coupling.shape[0] < 2:
        raise ValueError("K must contain at least two nodes.")
    if not np.all(np.isfinite(coupling)):
        raise ValueError("K must contain only finite values.")
    if not np.allclose(coupling, coupling.T, atol=1e-12):
        raise ValueError("K must be symmetric.")
    if not np.allclose(np.diag(coupling), 0.0, atol=1e-12):
        raise ValueError("K diagonal must be zero.")
    return coupling


def _validated_phase_vector(theta: np.ndarray, n_nodes: int) -> np.ndarray:
    phases = np.asarray(theta, dtype=float)
    if phases.ndim != 1 or phases.shape != (n_nodes,):
        raise ValueError("theta must match K node count as a 1-D vector.")
    if not np.all(np.isfinite(phases)):
        raise ValueError("theta must contain only finite values.")
    return phases


def _validated_threshold(threshold: float) -> float:
    value = float(threshold)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    return value


def coupling_weighted_edge_matrix(
    K: np.ndarray,
    theta: np.ndarray,
    *,
    normalise: bool = True,
) -> np.ndarray:
    """Return edge weights ``K_ij * |cos(theta_j - theta_i)|``.

    Args:
        K: symmetric zero-diagonal coupling matrix.
        theta: oscillator phase vector matching ``K``.
        normalise: divide by the maximum positive edge weight when present.
    """
    coupling = _validated_coupling_matrix(K)
    phases = _validated_phase_vector(theta, coupling.shape[0])

    phase_delta = phases[None, :] - phases[:, None]
    weights = np.asarray(coupling * np.abs(np.cos(phase_delta)), dtype=float)
    np.fill_diagonal(weights, 0.0)

    if normalise:
        max_weight = float(np.max(weights))
        if max_weight > 0.0:
            weights = np.asarray(weights / max_weight, dtype=float)
    return weights


def _active_edges(weights: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    n_nodes = weights.shape[0]
    return [
        (i, j)
        for i in range(n_nodes)
        for j in range(i + 1, n_nodes)
        if weights[i, j] > 0.0 and weights[i, j] >= threshold
    ]


def _connected_components(n_nodes: int, edges: list[tuple[int, int]]) -> int:
    parent = list(range(n_nodes))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, j in edges:
        union(i, j)

    return len({find(node) for node in range(n_nodes)})


def _triangles_from_edges(
    n_nodes: int, edges: list[tuple[int, int]]
) -> list[tuple[int, int, int]]:
    edge_set = set(edges)
    triangles: list[tuple[int, int, int]] = []
    for i, j, k in combinations(range(n_nodes), 3):
        if (i, j) in edge_set and (i, k) in edge_set and (j, k) in edge_set:
            triangles.append((i, j, k))
    return triangles


def _gf2_rank(matrix: np.ndarray) -> int:
    work = np.asarray(matrix, dtype=np.uint8).copy() % 2
    if work.size == 0:
        return 0

    n_rows, n_cols = work.shape
    rank = 0
    for col in range(n_cols):
        pivot_rows = np.flatnonzero(work[rank:, col])
        if pivot_rows.size == 0:
            continue
        pivot = int(pivot_rows[0] + rank)
        if pivot != rank:
            work[[rank, pivot], :] = work[[pivot, rank], :]
        for row in range(n_rows):
            if row != rank and work[row, col]:
                work[row, :] ^= work[rank, :]
        rank += 1
        if rank == n_rows:
            break
    return rank


def _boundary_2_rank(
    edges: list[tuple[int, int]],
    triangles: list[tuple[int, int, int]],
) -> int:
    if not edges or not triangles:
        return 0

    edge_index = {edge: idx for idx, edge in enumerate(edges)}
    boundary = np.zeros((len(edges), len(triangles)), dtype=np.uint8)
    for col, (i, j, k) in enumerate(triangles):
        for edge in ((i, j), (i, k), (j, k)):
            boundary[edge_index[edge], col] = 1
    return _gf2_rank(boundary)


def tcbo_weighted_complex(
    K: np.ndarray,
    theta: np.ndarray,
    *,
    threshold: float = 0.72,
) -> TCBOWeightedComplexResult:
    """Compute beta-1 for the TCBO coupling-weighted flag complex."""
    threshold = _validated_threshold(threshold)
    weights = coupling_weighted_edge_matrix(K, theta, normalise=True)
    n_nodes = weights.shape[0]
    edges = _active_edges(weights, threshold)
    triangles = _triangles_from_edges(n_nodes, edges)

    beta_0 = _connected_components(n_nodes, edges)
    cycle_rank = len(edges) - n_nodes + beta_0
    beta_1 = max(cycle_rank - _boundary_2_rank(edges, triangles), 0)
    max_h1 = max((n_nodes - 1) * (n_nodes - 2) // 2, 1)

    return TCBOWeightedComplexResult(
        threshold=threshold,
        beta_0=beta_0,
        beta_1=beta_1,
        p_h1=float(beta_1 / max_h1),
        n_nodes=n_nodes,
        n_edges=len(edges),
        n_triangles=len(triangles),
        max_h1=max_h1,
        edge_weights=weights,
    )


def tcbo_weighted_threshold_scan(
    K: np.ndarray,
    theta: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    target_p_h1: float = 0.72,
    promotion_tolerance: float | None = None,
) -> TCBOWeightedThresholdScan:
    """Scan thresholded coupling-weighted complexes against a target p_h1."""
    threshold_values = (
        np.asarray(thresholds, dtype=float)
        if thresholds is not None
        else np.linspace(0.0, 1.0, 41)
    )
    if threshold_values.ndim != 1 or threshold_values.size == 0:
        raise ValueError("thresholds must be a non-empty 1-D array.")

    results = tuple(
        tcbo_weighted_complex(K, theta, threshold=float(threshold))
        for threshold in threshold_values
    )
    target = float(target_p_h1)
    if not np.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("target_p_h1 must be in [0, 1].")

    errors = np.array([abs(result.p_h1 - target) for result in results], dtype=float)
    best_idx = int(np.argmin(errors))
    best = results[best_idx]
    tolerance = None if promotion_tolerance is None else float(promotion_tolerance)
    promotes = tolerance is not None and errors[best_idx] <= tolerance

    return TCBOWeightedThresholdScan(
        results=results,
        target_p_h1=target,
        best_threshold=best.threshold,
        best_p_h1=best.p_h1,
        best_abs_error=float(errors[best_idx]),
        promotes_target=bool(promotes),
    )
