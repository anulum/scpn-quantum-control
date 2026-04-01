# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Biological Surface Code
"""Topological Quantum Error Correction: Biological Surface Code.

Maps the stabilizers and error syndromes to the actual biological SCPN
coupling architecture rather than a generic 2D square lattice.

In this Biological Surface Code:
- Data Qubits: Edges of the K_nm coupling graph (synaptic connections).
- Vertex (X) Stabilizers: Nodes of the K_nm graph (layers/oscillators).
- Plaquette (Z) Stabilizers: Cycle basis of the K_nm graph (topological loops).

This uses the physical biological network to optimize quantum error
correction, creating a fully native topological code for the system.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


class BiologicalSurfaceCode:
    """A Surface Code defined directly on a biological coupling graph."""

    def __init__(self, K: np.ndarray, threshold: float = 1e-5):
        self.K = K
        self.threshold = threshold
        self.n_nodes = K.shape[0]

        self.G = nx.Graph()

        # Data qubits are the active edges (couplings)
        self.edges = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if abs(K[i, j]) >= threshold:
                    self.edges.append((i, j))
                    # Weight inversely proportional to coupling strength for MWPM
                    # Stronger biological coupling = shorter distance = preferred error path
                    self.G.add_edge(i, j, weight=1.0 / (abs(K[i, j]) + 1e-5))

        self.num_data = len(self.edges)
        self.edge_to_idx = {e: i for i, e in enumerate(self.edges)}

        if self.num_data == 0:
            raise ValueError("Coupling matrix K has no edges above threshold.")

        # X Stabilizers: One per node (Star operators)
        self.num_x_stabs = self.n_nodes
        self.Hx = np.zeros((self.num_x_stabs, self.num_data), dtype=np.int8)
        for v in range(self.n_nodes):
            for neighbor in self.G.neighbors(v):
                e = (min(v, neighbor), max(v, neighbor))
                idx = self.edge_to_idx[e]
                self.Hx[v, idx] = 1

        # Z Stabilizers: Cycle basis of the graph (Plaquette operators)
        cycles = nx.cycle_basis(self.G)
        self.num_z_stabs = len(cycles)
        self.Hz = np.zeros((self.num_z_stabs, self.num_data), dtype=np.int8)
        for c_idx, cycle in enumerate(cycles):
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                idx = self.edge_to_idx[e]
                self.Hz[c_idx, idx] = 1

    def verify_css_commutation(self) -> bool:
        """Verify that all X and Z stabilizers commute (Hx @ Hz.T == 0 mod 2)."""
        if self.num_z_stabs == 0:
            return True
        comm_matrix = (self.Hx @ self.Hz.T) % 2
        return bool(np.all(comm_matrix == 0))


class BiologicalMWPMDecoder:
    """Minimum Weight Perfect Matching decoder for the Biological Surface Code."""

    def __init__(self, code: BiologicalSurfaceCode):
        self.code = code
        self.G = code.G

    def decode_z_errors(self, syndrome_x: np.ndarray) -> np.ndarray:
        """Decode X-syndromes to find the optimal Z-error correction.

        Z errors on edges flip the X stabilizers at their endpoint nodes.
        We match the excited nodes (defects) through the biological graph.
        """
        defects = np.where(syndrome_x == 1)[0]
        correction = np.zeros(self.code.num_data, dtype=np.int8)

        if len(defects) == 0:
            return correction

        # If odd number of defects (should only happen with open boundaries),
        # we drop one for simplicity in this implementation, though a true
        # fault-tolerant code would map it to a rough boundary.
        if len(defects) % 2 != 0:
            defects = defects[:-1]

        matching_graph = nx.Graph()
        paths = {}

        for i in range(len(defects)):
            for j in range(i + 1, len(defects)):
                u, v = defects[i], defects[j]
                try:
                    path = nx.shortest_path(self.G, source=u, target=v, weight="weight")
                    dist = nx.shortest_path_length(self.G, source=u, target=v, weight="weight")
                    # networkx max_weight_matching maximizes, so we use negative distance
                    matching_graph.add_edge(u, v, weight=-dist)
                    paths[(u, v)] = path
                    paths[(v, u)] = path
                except nx.NetworkXNoPath:
                    pass

        matching = nx.max_weight_matching(matching_graph, maxcardinality=True)

        for u, v in matching:
            path = paths[(u, v)]
            for i in range(len(path) - 1):
                n1, n2 = path[i], path[i + 1]
                e = (min(n1, n2), max(n1, n2))
                idx = self.code.edge_to_idx[e]
                correction[idx] ^= 1

        return correction
