"""QEC for quantum control signals using toric surface code + MWPM decoder.

Extends the scpn-quantum MWPM decoder with physics-aware weighting:
Knm graph distance instead of lattice distance for edge weights.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


class SurfaceCode:
    """Toric surface code with distance d.

    Data qubits: N = 2*d^2 (edges of d x d torus).
    Vertex (X) stabilizers: d^2.
    Plaquette (Z) stabilizers: d^2.
    Edge indexing: h(r,c) = 2*(r*d+c), v(r,c) = 2*(r*d+c)+1.
    """

    def __init__(self, distance: int = 3):
        """Build toric code check matrices for given code distance."""
        self.d = distance
        self.num_data = 2 * distance**2
        self.Hx, self.Hz = self._build_checks()

    def _build_checks(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.d
        N = self.num_data
        Hx: np.ndarray = np.zeros((d * d, N), dtype=np.int8)
        Hz: np.ndarray = np.zeros((d * d, N), dtype=np.int8)

        def h(r, c):
            return 2 * (r * d + c)

        def v(r, c):
            return 2 * (r * d + c) + 1

        for r in range(d):
            for c in range(d):
                stab = r * d + c
                # Vertex (X) stabilizer: 4 edges around vertex
                Hx[stab, h(r, c)] = 1
                Hx[stab, h(r, (c - 1) % d)] = 1
                Hx[stab, v(r, c)] = 1
                Hx[stab, v((r - 1) % d, c)] = 1

                # Plaquette (Z) stabilizer: 4 edges around face
                Hz[stab, h(r, c)] = 1
                Hz[stab, h((r + 1) % d, c)] = 1
                Hz[stab, v(r, c)] = 1
                Hz[stab, v(r, (c + 1) % d)] = 1

        return Hx, Hz


class MWPMDecoder:
    """Minimum Weight Perfect Matching decoder for toric codes.

    Manhattan distance on toric lattice, optionally weighted by Knm graph distance.
    """

    def __init__(self, distance: int, knm_weights: np.ndarray | None = None):
        """Optional knm_weights rescales Manhattan distance by 1/(1+K[u,v])."""
        self.d = distance
        self.knm_weights = knm_weights

    def decode(self, syndrome: np.ndarray, dual: bool = False) -> np.ndarray:
        """Decode syndrome -> correction vector of length 2*d^2.

        dual=False for vertex syndromes (X error correction),
        dual=True for plaquette syndromes (Z error correction).
        """
        defects = np.where(syndrome == 1)[0]
        if len(defects) == 0:
            return np.zeros(2 * self.d**2, dtype=np.int8)
        if len(defects) % 2 == 1:
            defects = np.append(defects, defects[0])

        G = nx.Graph()
        for i in range(len(defects)):
            for j in range(i + 1, len(defects)):
                dist = self._distance(defects[i], defects[j])
                G.add_edge(i, j, weight=-dist)

        matching = nx.max_weight_matching(G, maxcardinality=True)

        correction: np.ndarray = np.zeros(2 * self.d**2, dtype=np.int8)
        for i, j in matching:
            path = self._shortest_path(defects[i], defects[j], dual=dual)
            for qubit in path:
                correction[qubit] ^= 1

        return correction

    def _distance(self, u: int, v: int) -> int:
        d = self.d
        r1, c1 = divmod(u, d)
        r2, c2 = divmod(v, d)
        dr = min(abs(r1 - r2), d - abs(r1 - r2))
        dc = min(abs(c1 - c2), d - abs(c1 - c2))
        base_dist = dr + dc

        if (
            self.knm_weights is not None
            and u < len(self.knm_weights)
            and v < len(self.knm_weights)
        ):
            knm_factor = 1.0 / (1.0 + self.knm_weights[u, v])
            return max(1, int(base_dist * knm_factor))
        return base_dist

    def _shortest_path(self, u: int, v: int, dual: bool = False) -> list[int]:
        """Qubit indices along Manhattan shortest path on torus.

        dual=False: vertex path (vertical edges for row moves, horizontal for column).
        dual=True: plaquette path (horizontal edges for row moves, vertical for column).
        On the toric code, plaquette (r,c) and plaquette ((r+1)%d, c) share
        horizontal edge h((r+1)%d, c), so row movement in the dual uses h-edges.
        """
        d = self.d
        r1, c1 = divmod(u, d)
        r2, c2 = divmod(v, d)
        path: list[int] = []

        dr_fwd = (r2 - r1) % d
        dr_bwd = (r1 - r2) % d
        r = r1
        if dr_fwd <= dr_bwd:
            for _ in range(dr_fwd):
                if dual:
                    r = (r + 1) % d
                    path.append(2 * (r * d + c1))  # horizontal edge h(r, c1)
                else:
                    path.append(2 * (r * d + c1) + 1)  # vertical edge v(r, c1)
                    r = (r + 1) % d
        else:
            for _ in range(dr_bwd):
                if dual:
                    path.append(2 * (r * d + c1))  # horizontal edge h(r, c1)
                    r = (r - 1) % d
                else:
                    r = (r - 1) % d
                    path.append(2 * (r * d + c1) + 1)  # vertical edge v(r, c1)

        dc_fwd = (c2 - c1) % d
        dc_bwd = (c1 - c2) % d
        c = c1
        if dc_fwd <= dc_bwd:
            for _ in range(dc_fwd):
                if dual:
                    c = (c + 1) % d
                    path.append(2 * (r2 * d + c) + 1)  # vertical edge v(r2, c)
                else:
                    path.append(2 * (r2 * d + c))  # horizontal edge h(r2, c)
                    c = (c + 1) % d
        else:
            for _ in range(dc_bwd):
                if dual:
                    path.append(2 * (r2 * d + c) + 1)  # vertical edge v(r2, c)
                    c = (c - 1) % d
                else:
                    c = (c - 1) % d
                    path.append(2 * (r2 * d + c))  # horizontal edge h(r2, c)

        return path


class ControlQEC:
    """QEC wrapper for protecting quantum control signals.

    Combines SurfaceCode + MWPMDecoder with optional Knm-weighted edges.
    """

    def __init__(self, distance: int = 3, knm_weights: np.ndarray | None = None):
        """Assemble SurfaceCode + MWPMDecoder for the given distance."""
        self.code = SurfaceCode(distance)
        self.decoder = MWPMDecoder(distance, knm_weights)

    def simulate_errors(
        self, p_error: float, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate independent X and Z errors with probability p_error."""
        if rng is None:
            rng = np.random.default_rng()
        N = self.code.num_data
        err_x = rng.binomial(1, p_error, N).astype(np.int8)
        err_z = rng.binomial(1, p_error, N).astype(np.int8)
        return err_x, err_z

    def get_syndrome(self, err_x: np.ndarray, err_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute syndromes: syn_z = Hx @ err_x mod 2, syn_x = Hz @ err_z mod 2."""
        syn_z = (self.code.Hx @ err_x) % 2
        syn_x = (self.code.Hz @ err_z) % 2
        return syn_z, syn_x

    def decode_and_correct(self, err_x: np.ndarray, err_z: np.ndarray) -> bool:
        """Full decode cycle. Returns True if no logical error remains.

        Checks both syndrome clearance and non-trivial homology on the torus.
        A residual with zero syndrome can still be a logical operator if it wraps
        around a non-contractible cycle.
        """
        syn_z, syn_x = self.get_syndrome(err_x, err_z)

        corr_x = self.decoder.decode(syn_z, dual=False)
        corr_z = self.decoder.decode(syn_x, dual=True)

        residual_x = err_x ^ corr_x
        residual_z = err_z ^ corr_z

        new_syn_z = (self.code.Hx @ residual_x) % 2
        new_syn_x = (self.code.Hz @ residual_z) % 2

        if not (np.all(new_syn_z == 0) and np.all(new_syn_x == 0)):
            return False

        return not self._has_logical_error(residual_x, residual_z)

    def _has_logical_error(self, residual_x: np.ndarray, residual_z: np.ndarray) -> bool:
        """Check if residual errors form non-trivial homology cycles.

        A residual with zero syndrome may still be a logical operator if it
        wraps around a non-contractible cycle of the torus. We detect winding
        by counting edge crossings at a seam:
        - Horizontal winding: parity of h-edges crossing the vertical seam
          at column 0, i.e. h(r, 0) = 2*r*d for each row r.
        - Vertical winding: parity of v-edges crossing the horizontal seam
          at row 0, i.e. v(0, c) = 2*c + 1 for each column c.
        """
        d = self.code.d
        for residual in (residual_x, residual_z):
            # Horizontal winding: h-edges crossing column 0→1 boundary
            if sum(int(residual[2 * r * d]) for r in range(d)) % 2 == 1:
                return True
            # Vertical winding: v-edges crossing row 0→1 boundary
            if sum(int(residual[2 * c + 1]) for c in range(d)) % 2 == 1:
                return True
        return False
