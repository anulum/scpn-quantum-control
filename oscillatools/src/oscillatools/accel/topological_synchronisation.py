# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topological (Hodge-Laplacian) synchronisation on simplicial complexes
r"""Topological synchronisation of phase oscillators living on the edges of a simplicial complex.

Where the rest of the toolkit places oscillators on the *nodes* of a graph, topological
synchronisation places them on higher-order simplices — here the *edges* — and couples them through
the **Hodge Laplacian** built from the complex's boundary operators (Millán, Torres & Bianconi, 2020;
Ghorbanchian et al., 2021). This is genuinely distinct from the toolkit's simplex / hyperedge *forces*,
which are node dynamics with higher-order coupling: here the dynamical variable itself is a topological
signal on the edges.

For a simplicial complex the node–edge boundary operator ``B_1`` and the edge–triangle boundary
operator ``B_2`` satisfy the defining identity ``B_1 B_2 = 0`` (the boundary of a boundary is empty).
The 1-Hodge Laplacian

.. math::

    L_1 = B_1^{\mathsf T} B_1 + B_2 B_2^{\mathsf T} = L_1^{\downarrow} + L_1^{\uparrow}

splits the edge space orthogonally into a *gradient* part (``\operatorname{im} B_1^{\mathsf T}``,
node-induced), a *curl* part (``\operatorname{im} B_2``, triangle-induced) and a *harmonic* part
(``\ker L_1``) whose dimension is the first Betti number — the number of independent cycles. The
topological Kuramoto flow on the edge phases ``\theta`` is

.. math::

    \dot\theta = \omega - \sigma_\downarrow\,B_1^{\mathsf T}\sin(B_1\theta)
                          - \sigma_\uparrow\,B_2\sin(B_2^{\mathsf T}\theta),

which drives the gradient and curl projections towards coherence. It adds no compute kernel.

**Provenance.** This capability was included as a Phase-8 frontier on a strategic, user-directed
commitment (CEO directive, 2026-06-29), not on an evidence-backed gap analysis: a structured sweep of the
external literature returned zero verified claims establishing topological (Hodge-Laplacian)
synchronisation as a required capability gap for this toolkit. The model above is literature-grounded (the
cited Hodge-Laplacian simplicial construction); the decision to prioritise it as a frontier is a strategic
choice, and any paper or public write-up must present it as such — not as an evidence-backed gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HodgeStructure:
    """The boundary operators and Hodge Laplacian of a simplicial complex.

    Attributes
    ----------
    node_boundary : numpy.ndarray
        The ``(N_nodes, N_edges)`` node–edge boundary operator ``B_1``.
    edge_boundary : numpy.ndarray
        The ``(N_edges, N_triangles)`` edge–triangle boundary operator ``B_2``.
    """

    node_boundary: NDArray[np.float64]
    edge_boundary: NDArray[np.float64]

    @property
    def down_laplacian(self) -> NDArray[np.float64]:
        """The lower (gradient) Laplacian ``B_1^T B_1``."""
        return np.asarray(self.node_boundary.T @ self.node_boundary, dtype=np.float64)

    @property
    def up_laplacian(self) -> NDArray[np.float64]:
        """The upper (curl) Laplacian ``B_2 B_2^T``."""
        return np.asarray(self.edge_boundary @ self.edge_boundary.T, dtype=np.float64)

    @property
    def hodge_laplacian(self) -> NDArray[np.float64]:
        """The 1-Hodge Laplacian ``L_1 = B_1^T B_1 + B_2 B_2^T``."""
        return np.asarray(self.down_laplacian + self.up_laplacian, dtype=np.float64)

    @property
    def betti_number(self) -> int:
        """The first Betti number ``dim ker L_1`` (the number of independent cycles)."""
        edges = self.node_boundary.shape[1]
        return int(edges - np.linalg.matrix_rank(self.hodge_laplacian))


@dataclass(frozen=True)
class HodgeComponents:
    """The orthogonal Hodge decomposition of an edge signal.

    Attributes
    ----------
    gradient : numpy.ndarray
        The gradient (node-induced) part in ``im B_1^T``.
    curl : numpy.ndarray
        The curl (triangle-induced) part in ``im B_2``.
    harmonic : numpy.ndarray
        The harmonic part in ``ker L_1``.
    """

    gradient: NDArray[np.float64]
    curl: NDArray[np.float64]
    harmonic: NDArray[np.float64]


@dataclass(frozen=True)
class TopologicalKuramotoTrajectory:
    """A topological Kuramoto trajectory of edge phases.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(n_steps + 1,)`` sample times.
    phases : numpy.ndarray
        The ``(n_steps + 1, N_edges)`` edge phases.
    """

    times: NDArray[np.float64]
    phases: NDArray[np.float64]

    @property
    def final_phases(self) -> NDArray[np.float64]:
        """The edge phases at the final step."""
        return np.ascontiguousarray(self.phases[-1], dtype=np.float64)


def simplicial_hodge_structure(
    n_nodes: int, edges: NDArray[np.int_], triangles: NDArray[np.int_]
) -> HodgeStructure:
    r"""Build the boundary operators ``B_1``, ``B_2`` of an oriented simplicial complex.

    Edges are oriented low-to-high (``u < v``) and triangles low-to-high (``a < b < c``) with the
    canonical boundary ``\partial[a,b,c] = [b,c] - [a,c] + [a,b]``.

    Parameters
    ----------
    n_nodes : int
        The number of nodes (``≥ 2``).
    edges : numpy.ndarray
        The ``(N_edges, 2)`` integer edge list, each row ``(u, v)`` with ``u < v``.
    triangles : numpy.ndarray
        The ``(N_triangles, 3)`` integer triangle list, each row ``(a, b, c)`` with ``a < b < c``;
        every triangle edge must be present in ``edges``. May be empty.

    Returns
    -------
    HodgeStructure
        The boundary operators.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be at least two, got {n_nodes}")
    edge_array = np.ascontiguousarray(edges, dtype=np.int_)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2 or edge_array.shape[0] < 1:
        raise ValueError("edges must be a non-empty (N_edges, 2) integer array")
    if np.any(edge_array < 0) or np.any(edge_array >= n_nodes):
        raise ValueError("edge endpoints must be valid node indices")
    if np.any(edge_array[:, 0] >= edge_array[:, 1]):
        raise ValueError("each edge must be oriented low-to-high (u < v)")
    edge_count = edge_array.shape[0]
    index = {(int(u), int(v)): k for k, (u, v) in enumerate(edge_array)}
    if len(index) != edge_count:
        raise ValueError("edges must be distinct")

    node_boundary = np.zeros((n_nodes, edge_count), dtype=np.float64)
    node_boundary[edge_array[:, 0], np.arange(edge_count)] = -1.0
    node_boundary[edge_array[:, 1], np.arange(edge_count)] = 1.0

    triangle_array = np.ascontiguousarray(triangles, dtype=np.int_).reshape(-1, 3)
    if triangle_array.size and np.any(triangle_array[:, :-1] >= triangle_array[:, 1:]):
        raise ValueError("each triangle must be oriented low-to-high (a < b < c)")
    edge_boundary = np.zeros((edge_count, triangle_array.shape[0]), dtype=np.float64)
    for column, (a, b, c) in enumerate(triangle_array):
        try:
            edge_boundary[index[(int(a), int(b))], column] += 1.0
            edge_boundary[index[(int(b), int(c))], column] += 1.0
            edge_boundary[index[(int(a), int(c))], column] -= 1.0
        except KeyError as error:
            raise ValueError(f"triangle ({a}, {b}, {c}) has an edge missing from edges") from error
    return HodgeStructure(node_boundary=node_boundary, edge_boundary=edge_boundary)


def hodge_decomposition(
    edge_signal: NDArray[np.float64], structure: HodgeStructure
) -> HodgeComponents:
    r"""Split an edge signal into its orthogonal gradient, curl and harmonic parts.

    Parameters
    ----------
    edge_signal : numpy.ndarray
        The ``(N_edges,)`` edge signal.
    structure : HodgeStructure
        The complex's boundary operators.

    Returns
    -------
    HodgeComponents
        The gradient, curl and harmonic components (summing to the signal).

    Raises
    ------
    ValueError
        If the signal length does not match the number of edges.
    """
    signal = np.ascontiguousarray(edge_signal, dtype=np.float64)
    boundary = structure.node_boundary
    edge_count = boundary.shape[1]
    if signal.ndim != 1 or signal.shape[0] != edge_count:
        raise ValueError(f"edge_signal must have length {edge_count}, got {signal.shape}")
    gradient = boundary.T @ (np.linalg.pinv(boundary @ boundary.T) @ (boundary @ signal))
    up = structure.edge_boundary
    if up.shape[1]:
        curl = up @ (np.linalg.pinv(up.T @ up) @ (up.T @ signal))
    else:
        curl = np.zeros(edge_count, dtype=np.float64)
    harmonic = signal - gradient - curl
    return HodgeComponents(
        gradient=np.ascontiguousarray(gradient, dtype=np.float64),
        curl=np.ascontiguousarray(curl, dtype=np.float64),
        harmonic=np.ascontiguousarray(harmonic, dtype=np.float64),
    )


def topological_order_parameter(phases: NDArray[np.float64]) -> float:
    r"""The edge order parameter ``|N^{-1}\sum_e e^{iθ_e}|`` (coherence of the edge phases)."""
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    if angle.ndim != 1 or angle.size < 1:
        raise ValueError("phases must be a non-empty one-dimensional array")
    return float(np.abs(np.mean(np.exp(1j * angle))))


def topological_kuramoto_field(
    phases: NDArray[np.float64],
    natural_frequencies: NDArray[np.float64],
    structure: HodgeStructure,
    *,
    down_coupling: float,
    up_coupling: float,
) -> NDArray[np.float64]:
    r"""The topological Kuramoto field ``ω - σ↓ B_1^T sin(B_1 θ) - σ↑ B_2 sin(B_2^T θ)``.

    Parameters
    ----------
    phases : numpy.ndarray
        The edge phases ``θ`` (length ``N_edges``).
    natural_frequencies : numpy.ndarray
        The edge natural frequencies ``ω`` (length ``N_edges``).
    structure : HodgeStructure
        The complex's boundary operators.
    down_coupling, up_coupling : float
        The lower (gradient) and upper (curl) coupling strengths.

    Returns
    -------
    numpy.ndarray
        The phase velocity (length ``N_edges``).

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    angle, frequencies = _validate_edges(phases, natural_frequencies, structure)
    if not (np.isfinite(down_coupling) and np.isfinite(up_coupling)):
        raise ValueError("down_coupling and up_coupling must be finite")
    return _field(angle, frequencies, structure, down_coupling, up_coupling)


def _field(
    angle: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    structure: HodgeStructure,
    down_coupling: float,
    up_coupling: float,
) -> NDArray[np.float64]:
    down = structure.node_boundary.T @ np.sin(structure.node_boundary @ angle)
    up = structure.edge_boundary @ np.sin(structure.edge_boundary.T @ angle)
    return np.asarray(frequencies - down_coupling * down - up_coupling * up, dtype=np.float64)


def _validate_edges(
    phases: NDArray[np.float64],
    natural_frequencies: NDArray[np.float64],
    structure: HodgeStructure,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(natural_frequencies, dtype=np.float64)
    edge_count = structure.node_boundary.shape[1]
    if angle.shape != (edge_count,):
        raise ValueError(f"phases must have length {edge_count}, got {angle.shape}")
    if frequencies.shape != (edge_count,):
        raise ValueError(
            f"natural_frequencies must have length {edge_count}, got {frequencies.shape}"
        )
    if not (np.all(np.isfinite(angle)) and np.all(np.isfinite(frequencies))):
        raise ValueError("phases and natural_frequencies must be finite")
    return angle, frequencies


def integrate_topological_kuramoto(
    initial_phases: NDArray[np.float64],
    natural_frequencies: NDArray[np.float64],
    structure: HodgeStructure,
    dt: float,
    n_steps: int,
    *,
    down_coupling: float,
    up_coupling: float,
) -> TopologicalKuramotoTrajectory:
    r"""Integrate the topological Kuramoto flow on the edge phases by RK4.

    Parameters
    ----------
    initial_phases, natural_frequencies, structure, down_coupling, up_coupling
        As for :func:`topological_kuramoto_field`.
    dt : float
        The RK4 step (finite, ``> 0``).
    n_steps : int
        The number of steps (``≥ 1``).

    Returns
    -------
    TopologicalKuramotoTrajectory
        The edge-phase trajectory.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    angle, frequencies = _validate_edges(initial_phases, natural_frequencies, structure)
    if not (np.isfinite(down_coupling) and np.isfinite(up_coupling)):
        raise ValueError("down_coupling and up_coupling must be finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    edge_count = angle.size
    trajectory = np.empty((n_steps + 1, edge_count), dtype=np.float64)
    trajectory[0] = angle
    current = angle
    for step in range(n_steps):
        k1 = _field(current, frequencies, structure, down_coupling, up_coupling)
        k2 = _field(current + 0.5 * dt * k1, frequencies, structure, down_coupling, up_coupling)
        k3 = _field(current + 0.5 * dt * k2, frequencies, structure, down_coupling, up_coupling)
        k4 = _field(current + dt * k3, frequencies, structure, down_coupling, up_coupling)
        current = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[step + 1] = current
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    return TopologicalKuramotoTrajectory(times=times, phases=trajectory)


__all__ = [
    "HodgeComponents",
    "HodgeStructure",
    "TopologicalKuramotoTrajectory",
    "hodge_decomposition",
    "integrate_topological_kuramoto",
    "simplicial_hodge_structure",
    "topological_kuramoto_field",
    "topological_order_parameter",
]
