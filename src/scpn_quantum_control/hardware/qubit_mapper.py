# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Dynamic Topology-Agnostic Qubit Mapper
"""DynQ-inspired qubit placement via quality-weighted community detection.

Models the QPU as a weighted graph where edge weights reflect two-qubit
gate fidelity. Louvain community detection partitions into high-quality
execution regions, replacing Qiskit's default layout heuristic.

Ref: Liu et al., arXiv:2601.19635 (2026) — DynQ
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    _HAS_NX = True
except ImportError:
    _HAS_NX = False


EPSILON_0 = 1e-6  # regularisation for zero-error edges


@dataclass
class ExecutionRegion:
    """A candidate execution region on the QPU."""

    qubits: frozenset[int]
    quality_score: float
    connectivity: float
    mean_gate_fidelity: float
    n_qubits: int


@dataclass
class QubitMappingResult:
    """Result of DynQ-style qubit mapping."""

    selected_region: ExecutionRegion
    all_regions: list[ExecutionRegion]
    initial_layout: list[int]
    resolution: float


def build_calibration_graph(
    gate_errors: dict[tuple[int, int], float],
    readout_errors: dict[int, float] | None = None,
) -> nx.Graph:
    """Build quality-weighted graph from calibration data.

    Edge weight: w_ij = 1 / (e_ij + ε₀) where e_ij is two-qubit gate error.
    Node weight: 1 / (readout_error + ε₀) if available.

    Args:
        gate_errors: {(i, j): error_rate} for each two-qubit gate
        readout_errors: {i: error_rate} for each qubit (optional)
    """
    if not _HAS_NX:
        raise ImportError("networkx required for qubit mapping")

    G = nx.Graph()
    qubits: set[int] = set()
    for (i, j), e in gate_errors.items():
        w = 1.0 / (e + EPSILON_0)
        G.add_edge(i, j, weight=w, error=e)
        qubits.update([i, j])

    if readout_errors:
        for q in qubits:
            ro = readout_errors.get(q, 0.0)
            G.nodes[q]["readout_weight"] = 1.0 / (ro + EPSILON_0)
            G.nodes[q]["readout_error"] = ro

    return G


def detect_execution_regions(
    G: nx.Graph,
    min_qubits: int = 3,
    resolution: float = 1.0,
    seed: int | None = None,
) -> list[ExecutionRegion]:
    """Partition QPU graph into execution regions via Louvain.

    Args:
        G: quality-weighted QPU graph from build_calibration_graph
        min_qubits: minimum region size (DynQ uses 3)
        resolution: Louvain resolution parameter (higher → smaller communities)
        seed: random seed for reproducibility
    """
    if not _HAS_NX:
        raise ImportError("networkx required for qubit mapping")

    communities = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)

    regions = []
    for comm in communities:
        if len(comm) < min_qubits:
            continue

        qubits = frozenset(comm)
        subgraph = G.subgraph(comm)

        # Connectivity score: edge density
        n = len(comm)
        n_edges = subgraph.number_of_edges()
        max_edges = n * (n - 1) / 2
        connectivity = (n_edges / max_edges) if max_edges > 0 else 0.0

        # Mean gate fidelity within region
        errors = [d["error"] for _, _, d in subgraph.edges(data=True) if "error" in d]
        mean_fidelity = 1.0 - np.mean(errors) if errors else 0.0

        # Composite quality (DynQ Eq. 8): connectivity × fidelity
        quality = connectivity * mean_fidelity

        regions.append(
            ExecutionRegion(
                qubits=qubits,
                quality_score=quality,
                connectivity=connectivity,
                mean_gate_fidelity=mean_fidelity,
                n_qubits=n,
            )
        )

    regions.sort(key=lambda r: r.quality_score, reverse=True)
    return regions


def select_best_region(
    regions: list[ExecutionRegion],
    circuit_width: int,
) -> ExecutionRegion | None:
    """Select the highest-quality region that fits the circuit.

    Args:
        regions: sorted list from detect_execution_regions
        circuit_width: number of qubits the circuit requires
    """
    for region in regions:
        if region.n_qubits >= circuit_width:
            return region
    return None


def dynq_initial_layout(
    gate_errors: dict[tuple[int, int], float],
    circuit_width: int,
    readout_errors: dict[int, float] | None = None,
    resolution: float = 1.0,
    min_qubits: int = 3,
    seed: int | None = None,
) -> QubitMappingResult | None:
    """Full DynQ pipeline: calibration → community detection → layout.

    Returns None if no suitable region found.
    """
    G = build_calibration_graph(gate_errors, readout_errors)
    regions = detect_execution_regions(G, min_qubits=min_qubits, resolution=resolution, seed=seed)

    selected = select_best_region(regions, circuit_width)
    if selected is None:
        return None

    # Sort qubits within region by readout quality (best first)
    sorted_qubits = sorted(selected.qubits)
    if readout_errors:
        sorted_qubits = sorted(
            selected.qubits,
            key=lambda q: readout_errors.get(q, 0.0),
        )

    layout = sorted_qubits[:circuit_width]

    return QubitMappingResult(
        selected_region=selected,
        all_regions=regions,
        initial_layout=layout,
        resolution=resolution,
    )
