# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Internal-branch tests for biological surface diagnostics
"""Branch tests for the biological surface-code diagnostics helpers.

Covers the optional python-louvain community path and its empty-partition
shortcut, the inter-domain coupling matrix's skip of unannotated nodes, and the
edgeless-graph guard in the public diagnostics entry point.
"""

from __future__ import annotations

import sys
import types
from typing import Any, cast

import networkx as nx
import pytest

from scpn_quantum_control.qec.biological_diagnostics import (
    _compute_modularity_partition,
    _domain_matrix,
    analyse_biological_surface_code,
)
from scpn_quantum_control.qec.biological_surface_code import BiologicalSurfaceCode


def _triangle() -> nx.Graph:
    """A weighted triangle graph for community/modularity tests."""
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=1.0)
    graph.add_edge(0, 2, weight=1.0)
    return graph


def _install_fake_louvain(monkeypatch: pytest.MonkeyPatch, assignment: dict[int, int]) -> None:
    """Install a fake python-louvain module returning a fixed node→community map."""

    def best_partition(graph: Any, weight: str) -> dict[int, int]:
        return assignment

    fake = types.SimpleNamespace(
        community_louvain=types.SimpleNamespace(best_partition=best_partition)
    )
    monkeypatch.setitem(sys.modules, "community", cast(Any, fake))


def test_modularity_partition_uses_louvain_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When python-louvain is importable its partition is grouped by community id."""
    _install_fake_louvain(monkeypatch, {0: 0, 1: 0, 2: 1})
    communities, modularity = _compute_modularity_partition(_triangle())
    assert {frozenset(c) for c in communities} == {frozenset({0, 1}), frozenset({2})}
    assert isinstance(modularity, float)


def test_modularity_partition_empty_assignment_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty louvain assignment yields no communities and zero modularity."""
    _install_fake_louvain(monkeypatch, {})
    communities, modularity = _compute_modularity_partition(_triangle())
    assert communities == []
    assert modularity == 0.0


def test_domain_matrix_skips_unannotated_nodes() -> None:
    """Edges touching an unannotated node are skipped, not crashed on."""
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=2.0)
    graph.add_edge(1, 2, weight=3.0)
    matrix = _domain_matrix(graph, {0: "a", 1: "a"})
    assert matrix == {"a": {"a": 2.0}}


def test_analyse_rejects_edgeless_graph() -> None:
    """A surface-code graph with no edges has no defined diagnostics."""
    code = cast(BiologicalSurfaceCode, types.SimpleNamespace(G=nx.Graph()))
    with pytest.raises(ValueError, match="no edges"):
        analyse_biological_surface_code(code)
