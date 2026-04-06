# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — U(1) Lattice Gauge Field
"""Compact U(1) lattice gauge theory on arbitrary graph topologies.

Link variables U_ij = exp(iA_ij) live on edges of the graph.
The Wilson plaquette action S = −β Σ_plaq Re(U_plaq) governs dynamics.
On general graphs, plaquettes are minimal cycles (triangles, squares)
detected from the adjacency structure.

Hybrid Monte Carlo (HMC) provides exact sampling: the conjugate
momentum π_ij is refreshed from N(0,1), then leapfrog integration
evolves (A, π) along the Hamiltonian H = π²/2 + S(A), and a
Metropolis accept/reject step corrects discretisation error.

Ref:
    - Wilson, Phys. Rev. D 10, 2445 (1974)
    - Creutz, "Quarks, Gluons and Lattices" (1983)
    - Rothe, "Lattice Gauge Theories" (2012), Ch. 3-4
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scpn_quantum_engine import (
        gauge_force_batch as _force_rust,
    )
    from scpn_quantum_engine import (
        plaquette_action_batch as _plaq_rust,
    )

    _HAS_RUST_GAUGE = True
except ImportError:
    _HAS_RUST_GAUGE = False


@dataclass
class PlaquetteResult:
    """Plaquette action measurement."""

    mean_plaquette: float  # ⟨Re(U_plaq)⟩ averaged over all plaquettes
    n_plaquettes: int
    action: float  # S = −β × n_plaq × mean_plaquette
    beta: float


class U1LatticGauge:
    """Compact U(1) lattice gauge field on arbitrary graph.

    Link variables A_ij ∈ [−π, π) with U_ij = exp(iA_ij).
    The gauge field is stored as a dict {(i,j): A_ij} for edges
    where i < j. Reversing gives A_ji = −A_ij.

    Plaquettes are precomputed from the graph adjacency.
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        beta: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """Initialise U(1) gauge field.

        Args:
            adjacency: n×n symmetric adjacency/coupling matrix.
                       Non-zero entries define edges.
            beta: inverse coupling constant (β = 1/g²).
            seed: RNG seed for reproducibility.
        """
        self.n = adjacency.shape[0]
        self.beta = beta
        self.rng = np.random.default_rng(seed)

        # Extract edges (i < j where adj[i,j] != 0)
        self.edges: list[tuple[int, int]] = []
        self.edge_weights: dict[tuple[int, int], float] = {}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(adjacency[i, j]) > 1e-15:
                    self.edges.append((i, j))
                    self.edge_weights[(i, j)] = float(adjacency[i, j])

        self.n_edges = len(self.edges)
        self.edge_index = {e: idx for idx, e in enumerate(self.edges)}

        # Link variables: A[idx] for edge self.edges[idx]
        self.links = self.rng.uniform(-np.pi, np.pi, self.n_edges)

        # Precompute plaquettes (minimal 3-cycles = triangles)
        self.plaquettes = self._find_triangles(adjacency)

        # Flat arrays for Rust acceleration
        self._tri_flat, self._tri_signs = self._build_flat_triangles()

    def _find_triangles(self, adjacency: np.ndarray) -> list[list[tuple[int, int]]]:
        """Find all triangles in the graph.

        A triangle (i,j,k) contributes plaquette
        U_ij × U_jk × U_ki = exp(i(A_ij + A_jk + A_ki)).
        """
        adj_bool = np.abs(adjacency) > 1e-15
        triangles: list[list[tuple[int, int]]] = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if not adj_bool[i, j]:
                    continue
                for k in range(j + 1, self.n):
                    if adj_bool[j, k] and adj_bool[i, k]:
                        # Triangle (i,j,k): edges (i,j), (j,k), (i,k)
                        triangles.append([(i, j), (j, k), (i, k)])
        return triangles

    def _build_flat_triangles(self) -> tuple[np.ndarray, np.ndarray]:
        """Build flat arrays of triangle edge indices and signs for Rust."""
        tri_flat = []
        tri_signs = []
        for plaq in self.plaquettes:
            i, j = plaq[0]
            _, k = plaq[1]
            # Edges: (i,j) +1, (j,k) +1, (i,k) −1
            e_ij = self.edge_index.get((min(i, j), max(i, j)))
            e_jk = self.edge_index.get((min(j, k), max(j, k)))
            e_ik = self.edge_index.get((min(i, k), max(i, k)))
            if e_ij is None or e_jk is None or e_ik is None:
                continue
            tri_flat.extend([e_ij, e_jk, e_ik])
            s_ij = 1.0 if i < j else -1.0
            s_jk = 1.0 if j < k else -1.0
            s_ik = -(1.0 if i < k else -1.0)  # reversed in plaquette
            tri_signs.extend([s_ij, s_jk, s_ik])
        return (
            np.array(tri_flat, dtype=np.int64),
            np.array(tri_signs, dtype=np.float64),
        )

    def _edge_link(self, i: int, j: int) -> float:
        """Get A_ij (signed: A_ji = −A_ij)."""
        if i < j:
            return float(self.links[self.edge_index[(i, j)]])
        elif j < i:
            return float(-self.links[self.edge_index[(j, i)]])
        return 0.0

    def plaquette_phase(self, plaq: list[tuple[int, int]]) -> float:
        """Compute plaquette phase Σ A_link around a closed path."""
        phase = 0.0
        for i, j in plaq:
            phase += self._edge_link(i, j)
        # Close the loop: last edge connects back
        # For triangles (i,j), (j,k), (i,k): phase = A_ij + A_jk - A_ik
        # The third edge is (i,k) but traversed as k→i, so sign flips
        return phase

    def plaquette_action_value(self, plaq: list[tuple[int, int]]) -> float:
        """Re(U_plaq) = cos(plaquette_phase)."""
        # Triangle (i,j), (j,k), (i,k): circulation = A_ij + A_jk − A_ik
        i1, j1 = plaq[0]  # (i, j)
        i2, j2 = plaq[1]  # (j, k)
        i3, j3 = plaq[2]  # (i, k)
        # Oriented: i→j→k→i
        phase = (
            self._edge_link(i1, j1)
            + self._edge_link(j2, i2 if j2 != i2 else j2)
            - self._edge_link(i3, j3)
        )
        # Actually simpler: sum with proper orientation
        phase = self._edge_link(i1, j1)  # i→j
        phase += self._edge_link(j1, j2)  # j→k (j1=j, j2=k)
        phase -= self._edge_link(i1, j3)  # k→i = −(i→k), but plaq[2]=(i,k), so −A_ik
        return float(np.cos(phase))

    def total_action(self) -> float:
        """S = −β × Σ_plaq Re(U_plaq). Uses Rust when available."""
        if _HAS_RUST_GAUGE and len(self.plaquettes) > 0:
            _, action = _plaq_rust(
                self.links,
                self._tri_flat,
                self._tri_signs,
                len(self.plaquettes),
                self.beta,
            )
            return float(action)

        s = 0.0
        for plaq in self.plaquettes:
            s -= self.beta * self.plaquette_action_value(plaq)
        return s

    def measure_plaquettes(self) -> PlaquetteResult:
        """Measure average plaquette value."""
        if not self.plaquettes:
            return PlaquetteResult(0.0, 0, 0.0, self.beta)

        total = sum(self.plaquette_action_value(p) for p in self.plaquettes)
        n_plaq = len(self.plaquettes)
        mean = total / n_plaq
        return PlaquetteResult(
            mean_plaquette=mean,
            n_plaquettes=n_plaq,
            action=-self.beta * total,
            beta=self.beta,
        )

    def force(self) -> np.ndarray:
        """Compute dS/dA for each link. Uses Rust when available."""
        if _HAS_RUST_GAUGE and len(self.plaquettes) > 0:
            return np.asarray(
                _force_rust(
                    self.links,
                    self._tri_flat,
                    self._tri_signs,
                    len(self.plaquettes),
                    self.n_edges,
                    self.beta,
                )
            )

        f = np.zeros(self.n_edges)
        for plaq in self.plaquettes:
            # Compute oriented phase for this plaquette
            i, j = plaq[0]
            _, k = plaq[1]
            phase = self._edge_link(i, j) + self._edge_link(j, k) - self._edge_link(i, k)
            sin_phase = np.sin(phase)

            # dS/dA_ij = −β × sin(phase) × (orientation of ij in plaq)
            # Edge (i,j): +1 orientation
            if (i, j) in self.edge_index:
                f[self.edge_index[(i, j)]] += self.beta * sin_phase
            # Edge (j,k): +1 orientation
            if j < k and (j, k) in self.edge_index:
                f[self.edge_index[(j, k)]] += self.beta * sin_phase
            elif k < j and (k, j) in self.edge_index:
                f[self.edge_index[(k, j)]] -= self.beta * sin_phase
            # Edge (i,k): −1 orientation (reversed in plaquette)
            if i < k and (i, k) in self.edge_index:
                f[self.edge_index[(i, k)]] -= self.beta * sin_phase
            elif k < i and (k, i) in self.edge_index:
                f[self.edge_index[(k, i)]] += self.beta * sin_phase

        return f


def hmc_update(
    gauge: U1LatticGauge,
    n_leapfrog: int = 10,
    step_size: float = 0.1,
) -> tuple[bool, float]:
    """Single HMC update step.

    1. Refresh momenta π ~ N(0,1)
    2. Leapfrog integration of (A, π) for n_leapfrog steps
    3. Metropolis accept/reject

    Returns (accepted, delta_H).
    """
    # Save old state
    old_links = gauge.links.copy()
    old_action = gauge.total_action()

    # Refresh momenta
    pi = gauge.rng.standard_normal(gauge.n_edges)
    old_kinetic = 0.5 * np.sum(pi**2)
    old_H = old_action + old_kinetic

    # Leapfrog
    force = gauge.force()
    pi -= 0.5 * step_size * force  # half-step momentum

    for _step in range(n_leapfrog - 1):
        gauge.links += step_size * pi  # full-step position
        # Wrap to [−π, π)
        gauge.links = (gauge.links + np.pi) % (2 * np.pi) - np.pi
        force = gauge.force()
        pi -= step_size * force  # full-step momentum

    gauge.links += step_size * pi  # final position step
    gauge.links = (gauge.links + np.pi) % (2 * np.pi) - np.pi
    pi -= 0.5 * step_size * force  # final half-step momentum

    # Accept/reject
    new_action = gauge.total_action()
    new_kinetic = 0.5 * np.sum(pi**2)
    new_H = new_action + new_kinetic
    delta_H = new_H - old_H

    if delta_H < 0 or gauge.rng.random() < np.exp(-delta_H):
        return True, delta_H
    else:
        gauge.links = old_links
        return False, delta_H
