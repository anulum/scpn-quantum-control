# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for topology-control persistent-homology complexes
"""Tests for the persistent-H1 complex builders and backends.

Covers the distance-matrix constructors (coupling, correlation, spike-trace),
the square-symmetric input validator and its rejection paths, the analytic
``max_h1_for_vertices`` bound, the deterministic ``NetworkCycleBackend`` cycle
counter, and the ripser-backed ``RipserPHBackend`` including its fail-closed
import guard.
"""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control.topology_control.complexes as complexes_mod
from scpn_quantum_control.topology_control.complexes import (
    NetworkCycleBackend,
    PersistenceDiagram,
    RipserPHBackend,
    build_correlation_distance_matrix,
    build_coupling_distance_matrix,
    max_h1_for_vertices,
    spike_trace_correlation_distance,
)


class TestPersistenceDiagram:
    """The persistence-bar container and its derived lifetimes."""

    def test_lifetimes_are_death_minus_birth(self) -> None:
        """Lifetimes pair births and deaths element-wise."""
        diagram = PersistenceDiagram(dimension=1, births=(0.0, 0.2), deaths=(0.5, 1.0))
        assert diagram.lifetimes == (0.5, 0.8)

    def test_lifetimes_empty_when_no_bars(self) -> None:
        """An empty diagram has no lifetimes."""
        diagram = PersistenceDiagram(dimension=1, births=(), deaths=())
        assert diagram.lifetimes == ()


class TestMaxH1ForVertices:
    """Analytic upper bound on independent one-cycles."""

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_small_graphs_clamp_to_one(self, n: int) -> None:
        """Fewer than three vertices cannot host a cycle; the bound clamps to 1."""
        assert max_h1_for_vertices(n) == 1

    def test_triangle_allows_one_cycle(self) -> None:
        """A triangle (n=3) admits a single independent cycle."""
        assert max_h1_for_vertices(3) == 1

    def test_grows_quadratically(self) -> None:
        """The bound follows (n-1)(n-2)/2 for n>=3."""
        assert max_h1_for_vertices(5) == (4 * 3) // 2


class TestSquareSymmetricValidation:
    """Rejection paths of the shared square-symmetric matrix validator.

    The validator is private; exercise it through the public coupling-distance
    builder that delegates to it.
    """

    def test_rejects_non_square(self) -> None:
        """A non-square matrix is rejected."""
        with pytest.raises(ValueError, match="square"):
            build_coupling_distance_matrix(np.zeros((2, 3), dtype=np.float64))

    def test_rejects_non_finite(self) -> None:
        """A matrix containing inf/nan is rejected."""
        bad = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="finite"):
            build_coupling_distance_matrix(bad)

    def test_rejects_asymmetric(self) -> None:
        """An asymmetric matrix is rejected."""
        bad = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="symmetric"):
            build_coupling_distance_matrix(bad)

    def test_rejects_negative(self) -> None:
        """A matrix with negative entries is rejected."""
        bad = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="non-negative"):
            build_coupling_distance_matrix(bad)


class TestBuildCouplingDistanceMatrix:
    """Coupling-to-distance conversion."""

    def test_strong_coupling_maps_to_short_distance(self) -> None:
        """The strongest coupling becomes the shortest (zero) off-diagonal distance."""
        coupling = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)
        distance = build_coupling_distance_matrix(coupling)
        assert distance[0, 1] == pytest.approx(0.0)
        assert np.allclose(np.diag(distance), 0.0)

    def test_zero_coupling_maps_to_unit_offdiagonal(self) -> None:
        """A zero-coupling graph maps to unit off-diagonal distances."""
        coupling = np.zeros((3, 3), dtype=np.float64)
        distance = build_coupling_distance_matrix(coupling)
        assert distance[0, 1] == pytest.approx(1.0)
        assert distance[2, 2] == pytest.approx(0.0)

    def test_partial_coupling_scales_between_zero_and_one(self) -> None:
        """Weaker couplings produce larger distances bounded by one."""
        coupling = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        distance = build_coupling_distance_matrix(coupling)
        assert 0.0 <= distance[0, 1] <= 1.0


class TestBuildCorrelationDistanceMatrix:
    """Correlation-to-distance conversion."""

    def test_full_correlation_maps_to_zero_distance(self) -> None:
        """Maximal correlation becomes zero distance."""
        corr = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        distance = build_correlation_distance_matrix(corr)
        assert distance[0, 1] == pytest.approx(0.0)

    def test_zero_correlation_maps_to_unit_offdiagonal(self) -> None:
        """A vanishing correlation matrix maps to unit off-diagonal distances."""
        corr = np.zeros((3, 3), dtype=np.float64)
        distance = build_correlation_distance_matrix(corr)
        assert distance[0, 1] == pytest.approx(1.0)
        assert np.allclose(np.diag(distance), 0.0)


class TestSpikeTraceCorrelationDistance:
    """Neuron-neuron distance from time-series traces."""

    def test_correlated_traces_have_small_distance(self) -> None:
        """Two identical traces are maximally correlated (zero distance)."""
        steps = np.linspace(0.0, 1.0, 50, dtype=np.float64)
        traces = np.stack([steps, steps], axis=1)
        distance = spike_trace_correlation_distance(traces)
        assert distance.shape == (2, 2)
        assert distance[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_rejects_wrong_dimension(self) -> None:
        """A non-2D trace array is rejected."""
        with pytest.raises(ValueError, match="shape"):
            spike_trace_correlation_distance(np.zeros(5, dtype=np.float64))

    def test_rejects_single_node(self) -> None:
        """Fewer than two nodes cannot form a distance matrix."""
        with pytest.raises(ValueError, match="at least two nodes"):
            spike_trace_correlation_distance(np.zeros((5, 1), dtype=np.float64))

    def test_rejects_non_finite(self) -> None:
        """Non-finite samples are rejected."""
        bad = np.array([[0.0, 1.0], [np.nan, 1.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="finite"):
            spike_trace_correlation_distance(bad)

    def test_constant_column_is_treated_as_zero_correlation(self) -> None:
        """A constant (zero-variance) column yields a finite, defined distance."""
        varying = np.linspace(0.0, 1.0, 20, dtype=np.float64)
        constant = np.full(20, 0.7, dtype=np.float64)
        traces = np.stack([varying, constant], axis=1)
        distance = spike_trace_correlation_distance(traces)
        assert np.all(np.isfinite(distance))
        assert distance[0, 1] == pytest.approx(1.0)


class TestNetworkCycleBackend:
    """Deterministic graph-cycle H1 approximation."""

    def test_metadata_attributes(self) -> None:
        """The backend advertises its name and approximate status."""
        backend = NetworkCycleBackend()
        assert backend.name == "network_cycle"
        assert backend.approximate is True

    def test_rejects_negative_threshold(self) -> None:
        """A negative threshold is rejected at construction."""
        with pytest.raises(ValueError, match="non-negative"):
            NetworkCycleBackend(threshold=-0.1)

    def test_triangle_has_one_cycle(self) -> None:
        """A fully connected triangle of short distances yields one cycle."""
        distance = 1.0 - np.eye(3, dtype=np.float64)
        distance = distance * 0.1
        backend = NetworkCycleBackend(threshold=0.5)
        summary = backend.compute(distance, persistence_threshold=0.5)
        assert summary.n_h1_total == 1
        assert summary.backend == "network_cycle"
        assert summary.max_h1 == max_h1_for_vertices(3)

    def test_tree_has_no_cycles(self) -> None:
        """A path graph (no closed loop) yields zero cycles."""
        distance = np.array(
            [[0.0, 0.1, 0.9], [0.1, 0.0, 0.1], [0.9, 0.1, 0.0]],
            dtype=np.float64,
        )
        backend = NetworkCycleBackend(threshold=0.5)
        summary = backend.compute(distance)
        assert summary.n_h1_total == 0
        assert summary.p_h1 == pytest.approx(0.0)

    def test_persistence_threshold_filters_short_bars(self) -> None:
        """Cycle bars below the persistence threshold are dropped from the count."""
        distance = 0.1 * (1.0 - np.eye(3, dtype=np.float64))
        backend = NetworkCycleBackend(threshold=0.5)
        summary = backend.compute(distance, persistence_threshold=2.0)
        assert summary.n_h1_persistent == 0


class TestRipserPHBackend:
    """Ripser-backed Vietoris-Rips persistent-H1 backend."""

    def test_rejects_maxdim_below_one(self) -> None:
        """maxdim must be at least one to reach H1."""
        with pytest.raises(ValueError, match="maxdim"):
            RipserPHBackend(maxdim=0)

    @pytest.mark.skipif(
        not complexes_mod.RIPSER_AVAILABLE, reason="ripser optional extra not installed"
    )
    def test_compute_returns_summary(self) -> None:
        """A square ring distance matrix yields a ripser H1 summary."""
        ring = np.array(
            [
                [0.0, 1.0, 2.0, 1.0],
                [1.0, 0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0, 1.0],
                [1.0, 2.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        backend = RipserPHBackend(maxdim=1)
        summary = backend.compute(ring, persistence_threshold=0.0)
        assert summary.backend == "ripser"
        assert summary.n_h1_total >= 1
        assert summary.max_h1 == max_h1_for_vertices(4)
        assert summary.distance_matrix.shape == (4, 4)

    def test_compute_without_ripser_is_fail_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the optional extra absent, compute raises a clear ImportError."""
        monkeypatch.setattr(complexes_mod, "RIPSER_AVAILABLE", False)
        backend = RipserPHBackend(maxdim=1)
        with pytest.raises(ImportError, match="ripser not installed"):
            backend.compute(np.zeros((3, 3), dtype=np.float64))
