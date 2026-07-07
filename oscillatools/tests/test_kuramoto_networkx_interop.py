# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the NetworkX coupling-matrix interop
r"""Tests for :mod:`oscillatools.accel.kuramoto_networkx_interop`.

The ingestion is anchored against NetworkX's own reference conversion (``networkx.to_numpy_array``,
transposed to our row-receives convention) on undirected, directed and multigraph topologies, and is
additionally exercised through a duck-typed stand-in graph so the no-dependency contract is enforced
(NetworkX must never be required for ingestion). The export direction is checked for round-tripping,
symmetric/directed detection and thresholding, and every validation branch is covered.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillatools.accel.kuramoto_networkx_interop import (
    coupling_from_networkx,
    graph_from_networked_coupling,
)
from oscillatools.accel.networked_kuramoto import networked_kuramoto_force


class _FakeGraph:
    """A minimal duck-typed stand-in satisfying the documented Graph API subset."""

    def __init__(
        self,
        nodes: list[object],
        edges: list[tuple[object, object, dict[str, object]]],
        *,
        directed: bool = False,
        multigraph: bool = False,
    ) -> None:
        self._nodes = nodes
        self._edges = edges
        self._directed = directed
        self._multigraph = multigraph

    def is_directed(self) -> bool:
        return self._directed

    def is_multigraph(self) -> bool:
        return self._multigraph

    def nodes(self) -> list[object]:
        return list(self._nodes)

    def edges(self, data: bool = False) -> list[tuple[object, object, dict[str, object]]]:
        assert data, "the interop always requests edge attributes"
        return list(self._edges)


# --------------------------------------------------------------------------- duck-typed ingestion


def test_fake_graph_ingests_without_networkx() -> None:
    """A duck-typed graph object is enough — NetworkX is not needed for ingestion."""
    graph = _FakeGraph(["a", "b", "c"], [("a", "b", {"weight": 2.0}), ("b", "c", {})])
    coupling = coupling_from_networkx(graph)
    expected = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    np.testing.assert_array_equal(coupling, expected)


def test_directed_edge_lands_in_the_receiving_row() -> None:
    """A directed edge ``u → v`` drives ``v``: it must land in ``K[v, u]`` only."""
    graph = _FakeGraph([0, 1], [(0, 1, {"weight": 3.0})], directed=True)
    coupling = coupling_from_networkx(graph)
    np.testing.assert_array_equal(coupling, np.array([[0.0, 0.0], [3.0, 0.0]]))


def test_default_weight_applies_to_unweighted_edges() -> None:
    graph = _FakeGraph([0, 1], [(0, 1, {})])
    coupling = coupling_from_networkx(graph, default_weight=0.25)
    np.testing.assert_array_equal(coupling, np.array([[0.0, 0.25], [0.25, 0.0]]))


def test_self_loop_lands_once_on_the_diagonal() -> None:
    """An undirected self-loop contributes its weight once (it is dynamically inert anyway)."""
    graph = _FakeGraph([0, 1], [(0, 0, {"weight": 5.0}), (0, 1, {"weight": 1.0})])
    coupling = coupling_from_networkx(graph)
    np.testing.assert_array_equal(coupling, np.array([[5.0, 1.0], [1.0, 0.0]]))


def test_parallel_multigraph_edges_sum() -> None:
    graph = _FakeGraph(
        [0, 1],
        [(0, 1, {"weight": 1.5}), (0, 1, {"weight": 0.5})],
        multigraph=True,
    )
    coupling = coupling_from_networkx(graph)
    np.testing.assert_array_equal(coupling, np.array([[0.0, 2.0], [2.0, 0.0]]))


def test_nodelist_permutation_reorders_the_matrix() -> None:
    graph = _FakeGraph(["a", "b"], [("a", "b", {"weight": 4.0})])
    coupling = coupling_from_networkx(graph, nodelist=["b", "a"])
    np.testing.assert_array_equal(coupling, np.array([[0.0, 4.0], [4.0, 0.0]]))
    # Index 0 is now "b": the same single edge, seen from the permuted ordering.
    direct = coupling_from_networkx(graph)
    np.testing.assert_array_equal(coupling, direct[::-1, ::-1])


# --------------------------------------------------------------------------- validation branches


def test_rejects_empty_graph() -> None:
    with pytest.raises(ValueError, match="graph must contain at least one node"):
        coupling_from_networkx(_FakeGraph([], []))


def test_rejects_nodelist_that_is_not_a_permutation() -> None:
    graph = _FakeGraph([0, 1], [(0, 1, {})])
    with pytest.raises(ValueError, match="permutation of the graph's nodes"):
        coupling_from_networkx(graph, nodelist=[0, 2])


def test_rejects_nodelist_with_wrong_length() -> None:
    graph = _FakeGraph([0, 1], [(0, 1, {})])
    with pytest.raises(ValueError, match="permutation of the graph's nodes"):
        coupling_from_networkx(graph, nodelist=[0])


def test_rejects_nodelist_with_duplicates() -> None:
    graph = _FakeGraph([0, 1], [(0, 1, {})])
    with pytest.raises(ValueError, match="must not contain duplicate nodes"):
        coupling_from_networkx(graph, nodelist=[0, 0])


def test_rejects_non_numeric_weight() -> None:
    graph = _FakeGraph([0, 1], [(0, 1, {"weight": "heavy"})])
    with pytest.raises(ValueError, match="non-numeric 'weight' attribute"):
        coupling_from_networkx(graph)


def test_rejects_boolean_weight() -> None:
    graph = _FakeGraph([0, 1], [(0, 1, {"weight": True})])
    with pytest.raises(ValueError, match="non-numeric 'weight' attribute"):
        coupling_from_networkx(graph)


def test_rejects_non_finite_weight() -> None:
    graph = _FakeGraph([0, 1], [(0, 1, {"weight": float("inf")})])
    with pytest.raises(ValueError, match="non-finite weight"):
        coupling_from_networkx(graph)


def test_export_rejects_non_square_coupling() -> None:
    pytest.importorskip("networkx")
    with pytest.raises(ValueError, match="non-empty square matrix"):
        graph_from_networked_coupling(np.zeros((2, 3)))


def test_export_rejects_negative_threshold() -> None:
    pytest.importorskip("networkx")
    with pytest.raises(ValueError, match="threshold must be non-negative"):
        graph_from_networked_coupling(np.zeros((2, 2)), threshold=-1.0)


# --------------------------------------------------------------------------- networkx ground truth


def test_undirected_ingestion_matches_networkx_reference() -> None:
    """Our matrix equals NetworkX's own ``to_numpy_array`` (transposed) on a weighted graph."""
    networkx = pytest.importorskip("networkx")
    rng = np.random.default_rng(0)
    graph = networkx.gnm_random_graph(12, 30, seed=3)
    for source, target in graph.edges():
        graph[source][target]["weight"] = float(rng.uniform(0.1, 2.0))
    coupling = coupling_from_networkx(graph)
    reference = networkx.to_numpy_array(graph, nodelist=list(graph.nodes())).T
    np.testing.assert_allclose(coupling, reference)


def test_directed_ingestion_matches_networkx_reference() -> None:
    networkx = pytest.importorskip("networkx")
    rng = np.random.default_rng(1)
    graph = networkx.gnp_random_graph(10, 0.3, seed=5, directed=True)
    for source, target in graph.edges():
        graph[source][target]["weight"] = float(rng.uniform(0.1, 2.0))
    coupling = coupling_from_networkx(graph)
    reference = networkx.to_numpy_array(graph, nodelist=list(graph.nodes())).T
    np.testing.assert_allclose(coupling, reference)


def test_multigraph_ingestion_matches_networkx_reference() -> None:
    networkx = pytest.importorskip("networkx")
    graph = networkx.MultiGraph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(0, 1, weight=2.5)
    graph.add_edge(1, 2, weight=0.5)
    coupling = coupling_from_networkx(graph)
    reference = networkx.to_numpy_array(graph, nodelist=[0, 1, 2]).T
    np.testing.assert_allclose(coupling, reference)


# --------------------------------------------------------------------------- export + round trip


def test_symmetric_coupling_exports_as_undirected_graph_and_round_trips() -> None:
    networkx = pytest.importorskip("networkx")
    rng = np.random.default_rng(2)
    coupling = rng.uniform(0.1, 1.0, (6, 6))
    coupling = 0.5 * (coupling + coupling.T)
    np.fill_diagonal(coupling, 0.0)
    graph = graph_from_networked_coupling(coupling)
    assert isinstance(graph, networkx.Graph) and not graph.is_directed()
    np.testing.assert_allclose(coupling_from_networkx(graph), coupling)


def test_asymmetric_coupling_exports_as_directed_graph_and_round_trips() -> None:
    networkx = pytest.importorskip("networkx")
    coupling = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.5, 0.0, 0.0]])
    graph = graph_from_networked_coupling(coupling)
    assert graph.is_directed()
    assert isinstance(graph, networkx.DiGraph)
    np.testing.assert_allclose(coupling_from_networkx(graph), coupling)


def test_threshold_drops_weak_entries() -> None:
    pytest.importorskip("networkx")
    coupling = np.array([[0.0, 0.05], [0.05, 0.0]])
    graph = graph_from_networked_coupling(coupling, threshold=0.1)
    assert graph.number_of_edges() == 0
    assert graph.number_of_nodes() == 2


def test_self_loops_round_trip_through_the_export() -> None:
    pytest.importorskip("networkx")
    coupling = np.array([[0.7, 1.0], [1.0, 0.0]])
    graph = graph_from_networked_coupling(coupling)
    np.testing.assert_allclose(coupling_from_networkx(graph), coupling)


# --------------------------------------------------------------------------- toolkit integration


def test_ingested_coupling_feeds_the_networked_force() -> None:
    """The produced matrix drives the dispatched networked force with the expected sign structure."""
    graph = _FakeGraph([0, 1], [(0, 1, {"weight": 1.0})])
    coupling = coupling_from_networkx(graph)
    theta = np.array([0.0, 0.5])
    force = networked_kuramoto_force(theta, coupling)
    expected = np.array([np.sin(0.5 - 0.0), np.sin(0.0 - 0.5)])
    np.testing.assert_allclose(force, expected, atol=1e-12)
