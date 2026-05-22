# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Biological Surface Code
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

from scpn_quantum_control.accel.rust_import import optional_rust_engine


class BiologicalSurfaceCode:
    """A Surface Code defined directly on a biological coupling graph."""

    def __init__(self, K: np.ndarray, threshold: float = 1e-5):
        K_array = np.asarray(K, dtype=float)
        if K_array.ndim != 2 or K_array.shape[0] != K_array.shape[1]:
            raise ValueError("Coupling matrix K must be a square 2D matrix.")
        if not np.all(np.isfinite(K_array)):
            raise ValueError("Coupling matrix K must contain only finite values.")
        if not np.allclose(K_array, K_array.T, rtol=1e-12, atol=1e-12):
            raise ValueError("Coupling matrix K must be symmetric for an undirected graph code.")
        if not np.allclose(np.diag(K_array), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("Coupling matrix K must have a zero diagonal.")

        threshold_value = float(threshold)
        if not np.isfinite(threshold_value) or threshold_value < 0.0:
            raise ValueError("threshold must be a finite non-negative value.")

        self.K = K_array
        self.threshold = threshold_value
        self.n_nodes = K_array.shape[0]

        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n_nodes))

        # Data qubits are the active edges (couplings)
        self.edges = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if abs(self.K[i, j]) >= self.threshold:
                    self.edges.append((i, j))
                    # Weight inversely proportional to coupling strength for MWPM
                    # Stronger biological coupling = shorter distance = preferred error path
                    self.G.add_edge(i, j, weight=1.0 / (abs(self.K[i, j]) + 1e-5))

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

    def estimate_logical_qubits(self) -> int:
        """Estimate k from CSS rank relation: k = n - rank(Hx) - rank(Hz)."""
        rank_hx = int(np.linalg.matrix_rank(self.Hx.astype(np.float64)))
        if self.Hz.size == 0:
            rank_hz = 0
        else:
            rank_hz = int(np.linalg.matrix_rank(self.Hz.astype(np.float64)))
        logical_qubits = self.num_data - rank_hx - rank_hz
        return max(0, int(logical_qubits))

    def code_summary(self) -> dict[str, int | bool]:
        """Return structural code parameters and commutation status."""
        return {
            "n_data_qubits": self.num_data,
            "n_x_stabilisers": self.num_x_stabs,
            "n_z_stabilisers": self.num_z_stabs,
            "n_logical_qubits_estimate": self.estimate_logical_qubits(),
            "css_commutes": self.verify_css_commutation(),
        }

    def x_syndrome_from_z_errors(self, z_errors: np.ndarray) -> np.ndarray:
        """Compute X stabiliser syndrome induced by edge-local Z errors."""
        errors = np.asarray(z_errors)
        if errors.ndim != 1 or errors.shape[0] != self.num_data:
            raise ValueError(
                f"z_errors length must equal number of data qubits ({self.num_data})."
            )
        if not np.all((errors == 0) | (errors == 1)):
            raise ValueError("z_errors must be a binary vector with entries 0 or 1.")
        return ((self.Hx @ errors.astype(np.int8, copy=False)) % 2).astype(np.int8)

    def apply_z_correction(self, z_errors: np.ndarray, correction: np.ndarray) -> np.ndarray:
        """Apply edge correction modulo 2 and return residual Z-error pattern."""
        errors = np.asarray(z_errors)
        corr = np.asarray(correction)
        if errors.ndim != 1 or errors.shape[0] != self.num_data:
            raise ValueError(
                f"z_errors length must equal number of data qubits ({self.num_data})."
            )
        if corr.ndim != 1 or corr.shape[0] != self.num_data:
            raise ValueError(
                f"correction length must equal number of data qubits ({self.num_data})."
            )
        if not np.all((errors == 0) | (errors == 1)):
            raise ValueError("z_errors must be binary.")
        if not np.all((corr == 0) | (corr == 1)):
            raise ValueError("correction must be binary.")
        return (
            (errors.astype(np.int8, copy=False) + corr.astype(np.int8, copy=False)) % 2
        ).astype(np.int8)


class BiologicalMWPMDecoder:
    """Minimum Weight Perfect Matching decoder for the Biological Surface Code."""

    def __init__(self, code: BiologicalSurfaceCode):
        self.code = code
        self.G = code.G
        self._rust_engine = optional_rust_engine()
        self._rust_exact_mwpm_defect_limit = 24
        self._last_decoder_backend = "uninitialised"

    @property
    def last_decoder_backend(self) -> str:
        """Return backend used by most recent decode call."""
        return self._last_decoder_backend

    def decode_z_errors(self, syndrome_x: np.ndarray) -> np.ndarray:
        """Decode X-syndromes to find the optimal Z-error correction.

        Z errors on edges flip the X stabilizers at their endpoint nodes.
        We match the excited nodes (defects) through the biological graph.
        """
        syndrome = np.asarray(syndrome_x)
        if syndrome.ndim != 1 or syndrome.shape[0] != self.code.num_x_stabs:
            raise ValueError(
                "syndrome_x length must equal the number of X stabilizers "
                f"({self.code.num_x_stabs})."
            )
        if not np.all((syndrome == 0) | (syndrome == 1)):
            raise ValueError("syndrome_x must be a binary vector with entries 0 or 1.")

        syndrome = syndrome.astype(np.int8, copy=False)
        defects = np.where(syndrome == 1)[0]

        if self._rust_engine is not None and defects.size <= self._rust_exact_mwpm_defect_limit:
            rust_decode = getattr(self._rust_engine, "biological_decode_z_errors", None)
            if callable(rust_decode):
                edge_u = np.fromiter((u for u, _ in self.code.edges), dtype=np.int64)
                edge_v = np.fromiter((v for _, v in self.code.edges), dtype=np.int64)
                edge_w = np.fromiter(
                    (float(self.G[u][v]["weight"]) for u, v in self.code.edges),
                    dtype=np.float64,
                )
                try:
                    correction = np.asarray(
                        rust_decode(
                            edge_u,
                            edge_v,
                            edge_w,
                            int(self.code.n_nodes),
                            syndrome,
                        ),
                        dtype=np.int8,
                    )
                    self._last_decoder_backend = "rust_exact_mwpm"
                    return correction
                except ValueError as exc:
                    if "defect count exceeds exact-MWPM limit" not in str(exc):
                        raise

        if defects.size > self._rust_exact_mwpm_defect_limit:
            self._last_decoder_backend = "python_fallback_high_defect"
        else:
            self._last_decoder_backend = "python_mwpm"

        correction = np.zeros(self.code.num_data, dtype=np.int8)

        if len(defects) == 0:
            return correction

        defect_set = set(int(node) for node in defects)
        for component in nx.connected_components(self.G):
            n_component_defects = len(defect_set.intersection(component))
            if n_component_defects % 2 != 0:
                raise ValueError(
                    "syndrome_x has odd syndrome parity in a connected component; "
                    "this graph decoder has no explicit rough-boundary model."
                )

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
        if len(matching) * 2 != len(defects):
            raise ValueError(
                "syndrome_x defects cannot be perfectly matched on the biological graph; "
                "check connectivity, thresholding, or boundary model assumptions."
            )

        for u, v in matching:
            path = paths[(u, v)]
            for i in range(len(path) - 1):
                n1, n2 = path[i], path[i + 1]
                e = (min(n1, n2), max(n1, n2))
                idx = self.code.edge_to_idx[e]
                correction[idx] ^= 1

        return correction

    def decode_and_apply(self, z_errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode from a Z-error pattern and return (correction, residual)."""
        syndrome = self.code.x_syndrome_from_z_errors(z_errors)
        correction = self.decode_z_errors(syndrome)
        residual = self.code.apply_z_correction(z_errors, correction)
        return correction, residual
