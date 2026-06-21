# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology Control Complexes
"""Persistent-H1 complex builders and backend abstractions for topology control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import networkx as nx
import numpy as np
from numpy.typing import NDArray

try:
    from ripser import ripser

    RIPSER_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional topology extra.
    RIPSER_AVAILABLE = False

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class PersistenceDiagram:
    """Finite persistence bars for one homology dimension."""

    dimension: int
    births: tuple[float, ...]
    deaths: tuple[float, ...]

    @property
    def lifetimes(self) -> tuple[float, ...]:
        """Finite death-birth lifetimes."""

        return tuple(float(death - birth) for birth, death in zip(self.births, self.deaths))


@dataclass(frozen=True)
class H1Summary:
    """Persistent-H1 summary used by coupling objectives."""

    p_h1: float
    n_h1_persistent: int
    n_h1_total: int
    lifetimes: list[float]
    max_h1: int
    backend: str
    distance_matrix: FloatArray


class PersistentHomologyBackend(Protocol):
    """Backend protocol for persistent-H1 summaries."""

    name: str
    approximate: bool

    def compute(
        self, distance_matrix: NDArray[np.float64], *, persistence_threshold: float = 0.1
    ) -> H1Summary:
        """Compute persistent-H1 summary from a precomputed distance matrix."""


def _as_square_symmetric_matrix(name: str, matrix: NDArray[np.float64]) -> FloatArray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(arr, arr.T, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    if np.any(arr < -1e-12):
        raise ValueError(f"{name} must be non-negative")
    result: FloatArray = (arr + arr.T) / 2.0
    np.fill_diagonal(result, 0.0)
    return result


def max_h1_for_vertices(n_vertices: int) -> int:
    """Maximum independent one-cycles in a complete graph skeleton."""

    if n_vertices < 3:
        return 1
    return max((n_vertices - 1) * (n_vertices - 2) // 2, 1)


def build_coupling_distance_matrix(coupling: NDArray[np.float64]) -> FloatArray:
    """Convert a coupling matrix into a Vietoris-Rips distance matrix.

    Stronger absolute couplings become shorter distances. A zero-coupling graph
    maps to unit off-diagonal distances so PH consumers can still classify it.
    """

    K = _as_square_symmetric_matrix("coupling", coupling)
    weights = np.abs(K)
    np.fill_diagonal(weights, 0.0)
    max_weight = float(np.max(weights))
    if max_weight <= 1e-15:
        distance = np.ones_like(K) - np.eye(K.shape[0])
        return distance
    distance = 1.0 - weights / max_weight
    np.fill_diagonal(distance, 0.0)
    return distance


def build_correlation_distance_matrix(correlation: NDArray[np.float64]) -> FloatArray:
    """Convert a correlation matrix into a distance matrix."""

    corr = _as_square_symmetric_matrix("correlation", np.abs(correlation))
    max_corr = float(np.max(corr))
    if max_corr <= 1e-15:
        distance = np.ones_like(corr) - np.eye(corr.shape[0])
        return distance
    distance = 1.0 - corr / max_corr
    np.fill_diagonal(distance, 0.0)
    return distance


def spike_trace_correlation_distance(spike_traces: NDArray[np.float64]) -> FloatArray:
    """Build a neuron-neuron distance matrix from spike or membrane traces.

    Input shape is ``(steps, nodes)``. Constant columns are treated as zero
    correlation to avoid NaNs.
    """

    traces = np.asarray(spike_traces, dtype=np.float64)
    if traces.ndim != 2:
        raise ValueError("spike_traces must have shape (steps, nodes)")
    if traces.shape[1] < 2:
        raise ValueError("spike_traces must contain at least two nodes")
    if not np.all(np.isfinite(traces)):
        raise ValueError("spike_traces must contain only finite values")
    centred = traces - np.mean(traces, axis=0, keepdims=True)
    norm = np.linalg.norm(centred, axis=0)
    safe = np.where(norm > 1e-15, norm, 1.0)
    corr = (centred.T @ centred) / np.outer(safe, safe)
    corr = np.where(np.outer(norm > 1e-15, norm > 1e-15), corr, 0.0)
    return build_correlation_distance_matrix(corr)


class NetworkCycleBackend:
    """Deterministic graph-cycle H1 approximation for tests and no-extra installs."""

    name = "network_cycle"
    approximate = True

    def __init__(self, threshold: float = 0.5) -> None:
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        self.threshold = float(threshold)

    def compute(
        self,
        distance_matrix: NDArray[np.float64],
        *,
        persistence_threshold: float = 0.1,
    ) -> H1Summary:
        """Compute an H1 summary from a thresholded graph cycle basis."""
        distance = _as_square_symmetric_matrix("distance_matrix", distance_matrix)
        graph = nx.Graph()
        n = distance.shape[0]
        graph.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if float(distance[i, j]) <= self.threshold:
                    graph.add_edge(i, j)

        n_edges = graph.number_of_edges()
        n_components = nx.number_connected_components(graph) if n > 0 else 0
        cycle_rank = max(n_edges - n + n_components, 0)
        max_h1 = max_h1_for_vertices(n)
        persistent = [1.0] * cycle_rank
        thresholded = [life for life in persistent if life > persistence_threshold]
        return H1Summary(
            p_h1=float(len(thresholded) / max_h1),
            n_h1_persistent=len(thresholded),
            n_h1_total=cycle_rank,
            lifetimes=thresholded,
            max_h1=max_h1,
            backend=self.name,
            distance_matrix=distance,
        )


class RipserPHBackend:
    """Ripser-backed Vietoris-Rips persistent-H1 backend."""

    name = "ripser"
    approximate = False

    def __init__(self, maxdim: int = 1) -> None:
        if maxdim < 1:
            raise ValueError("maxdim must be >= 1")
        self.maxdim = int(maxdim)

    def compute(
        self,
        distance_matrix: NDArray[np.float64],
        *,
        persistence_threshold: float = 0.1,
    ) -> H1Summary:
        """Compute a Vietoris-Rips persistent-H1 summary with ripser."""
        if not RIPSER_AVAILABLE:
            raise ImportError("ripser not installed: pip install 'scpn-quantum-control[topology]'")
        distance = _as_square_symmetric_matrix("distance_matrix", distance_matrix)
        result = ripser(distance, maxdim=self.maxdim, distance_matrix=True)
        h1 = result["dgms"][1]
        lifetimes = [float(death - birth) for birth, death in h1 if np.isfinite(death)]
        persistent = [life for life in lifetimes if life > persistence_threshold]
        max_h1 = max_h1_for_vertices(distance.shape[0])
        return H1Summary(
            p_h1=float(len(persistent) / max_h1),
            n_h1_persistent=len(persistent),
            n_h1_total=len(lifetimes),
            lifetimes=persistent,
            max_h1=max_h1,
            backend=self.name,
            distance_matrix=distance,
        )


__all__ = [
    "FloatArray",
    "H1Summary",
    "NetworkCycleBackend",
    "PersistenceDiagram",
    "PersistentHomologyBackend",
    "RIPSER_AVAILABLE",
    "RipserPHBackend",
    "build_correlation_distance_matrix",
    "build_coupling_distance_matrix",
    "max_h1_for_vertices",
    "spike_trace_correlation_distance",
]
