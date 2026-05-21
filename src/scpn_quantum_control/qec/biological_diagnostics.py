# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Biological Surface-Code Diagnostics
"""Biology-oriented topology diagnostics for Biological Surface Code graphs.

This module augments QEC code construction with graph biomarkers useful for
analysing biological coupling substrates:
- weighted degree and betweenness criticality
- edge and cycle burden statistics
- community-modularity profile
- inter-domain coupling matrix from optional domain annotations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from .biological_surface_code import BiologicalSurfaceCode


@dataclass(frozen=True)
class BiologicalSurfaceDiagnostics:
    """Structured diagnostics for biological coupling topology under QEC."""

    n_nodes: int
    n_edges: int
    n_cycles: int
    mean_weighted_degree: float
    max_weighted_degree_node: int
    max_weighted_degree: float
    max_betweenness_node: int
    max_betweenness: float
    modularity: float
    n_communities: int
    cycle_length_mean: float
    cycle_length_max: int
    inter_domain_coupling: dict[str, dict[str, float]]
    metadata: dict[str, Any]


def _compute_modularity_partition(graph: nx.Graph) -> tuple[list[set[int]], float]:
    """Return community partition and modularity using available implementation."""
    try:
        from community import community_louvain  # type: ignore[import-not-found]

        assignment = community_louvain.best_partition(graph, weight="weight")
        groups: dict[int, set[int]] = {}
        for node, cid in assignment.items():
            groups.setdefault(int(cid), set()).add(int(node))
        communities = list(groups.values())
    except Exception:
        communities = list(
            nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
        )

    if not communities:
        return [], 0.0
    modularity = float(nx.algorithms.community.modularity(graph, communities, weight="weight"))
    return communities, modularity


def _domain_matrix(
    graph: nx.Graph,
    node_domains: dict[int, str] | None,
) -> dict[str, dict[str, float]]:
    """Aggregate absolute edge coupling by domain pairs."""
    if node_domains is None:
        return {}
    domains = sorted(set(node_domains.values()))
    matrix = {d1: {d2: 0.0 for d2 in domains} for d1 in domains}
    for u, v, data in graph.edges(data=True):
        du = node_domains.get(int(u))
        dv = node_domains.get(int(v))
        if du is None or dv is None:
            continue
        w = float(data.get("weight", 0.0))
        matrix[du][dv] += abs(w)
        if du != dv:
            matrix[dv][du] += abs(w)
    return matrix


def analyse_biological_surface_code(
    code: BiologicalSurfaceCode,
    node_domains: dict[int, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> BiologicalSurfaceDiagnostics:
    """Compute biological graph diagnostics for a surface-code substrate."""
    graph = code.G
    weighted_degree = dict(graph.degree(weight="weight"))
    if not weighted_degree:
        raise ValueError("Biological surface code graph has no edges; diagnostics unavailable.")

    max_deg_node, max_deg = max(weighted_degree.items(), key=lambda kv: float(kv[1]))
    betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    max_btw_node, max_btw = max(betweenness.items(), key=lambda kv: float(kv[1]))

    cycles = nx.cycle_basis(graph)
    cycle_lengths = [len(cycle) for cycle in cycles]
    cycle_length_mean = float(np.mean(cycle_lengths)) if cycle_lengths else 0.0
    cycle_length_max = int(max(cycle_lengths)) if cycle_lengths else 0

    communities, modularity = _compute_modularity_partition(graph)
    inter_domain = _domain_matrix(graph, node_domains)

    return BiologicalSurfaceDiagnostics(
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        n_cycles=len(cycles),
        mean_weighted_degree=float(np.mean(list(weighted_degree.values()))),
        max_weighted_degree_node=int(max_deg_node),
        max_weighted_degree=float(max_deg),
        max_betweenness_node=int(max_btw_node),
        max_betweenness=float(max_btw),
        modularity=float(modularity),
        n_communities=len(communities),
        cycle_length_mean=cycle_length_mean,
        cycle_length_max=cycle_length_max,
        inter_domain_coupling=inter_domain,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "BiologicalSurfaceDiagnostics",
    "analyse_biological_surface_code",
]
