# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Structural hypergraph Kuramoto force and Jacobian, generic hyperedge list
r"""Structural hypergraph Kuramoto coupling — a generic hyperedge list of any arity.

A hyperedge ``e = \{n_0, …, n_p\}`` of ``p + 1`` nodes couples its members through the
higher-order phase-difference interaction: a member ``i`` of the edge feels

.. math::

    K_e\,\sin\!\Bigl(\sum_{k \in e,\, k \ne i} θ_k - p\,θ_i\Bigr)
      = K_e\,\sin\!\Bigl(\sum_{k \in e} θ_k - |e|\,θ_i\Bigr),

with ``K_e`` the weight of that hyperedge. The total force on an oscillator is the sum over every
hyperedge that contains it, ``F_i = Σ_{e \ni i} K_e \sin(\cdots)``. Unlike the mean-field simplex
force this is a *structural* model: the coupling is given by an explicit list of hyperedges, so
the network may be sparse and mix orders (pairwise links, triangles, tetrahedra) in one list.

The order reduces to the lower-order Kuramoto couplings:

- arity-2 hyperedges ``\{i, j\}`` give the pairwise force ``K_{e}\sin(θ_j - θ_i)`` — for an
  undirected edge list with symmetric weights this is the networked Kuramoto force;
- arity-3 hyperedges are 2-simplex (triadic) triangles, arity-4 are 3-simplex tetrahedra, and so on.

The Jacobian (:func:`hyperedge_jacobian`) differentiates each edge term; because the interaction
depends only on phase differences, every row sums to zero (the global-phase Goldstone mode). This
is an analysis layer evaluated directly from the hyperedge list, so it adds no compute kernel.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def _resolve_hyperedges(hyperedges: Sequence[Sequence[int]], count: int) -> list[NDArray[np.intp]]:
    """Validate the hyperedge list and return each edge as an index array."""
    resolved: list[NDArray[np.intp]] = []
    for position, edge in enumerate(hyperedges):
        members = np.asarray(edge, dtype=np.intp)
        if members.ndim != 1 or members.size < 2:
            raise ValueError(
                f"hyperedge {position} must list at least two node indices, got {list(edge)}"
            )
        if np.unique(members).size != members.size:
            raise ValueError(f"hyperedge {position} has a repeated node index: {list(edge)}")
        if members.min() < 0 or members.max() >= count:
            raise ValueError(
                f"hyperedge {position} has a node index outside [0, {count}): {list(edge)}"
            )
        resolved.append(members)
    return resolved


def _resolve_weights(coupling: float | Sequence[float], edge_count: int) -> NDArray[np.float64]:
    """Broadcast a scalar coupling, or validate a per-edge weight sequence, to length ``edge_count``."""
    if isinstance(coupling, int | float):
        return np.full(edge_count, float(coupling), dtype=np.float64)
    weights = np.ascontiguousarray(coupling, dtype=np.float64)
    if weights.shape != (edge_count,):
        raise ValueError(
            f"coupling must be a scalar or a length-{edge_count} sequence, got shape {weights.shape}"
        )
    return weights


def hyperedge_force(
    theta: NDArray[np.float64],
    hyperedges: Sequence[Sequence[int]],
    coupling: float | Sequence[float],
) -> NDArray[np.float64]:
    r"""Return the structural hypergraph force ``F_i = Σ_{e∋i} K_e sin(Σ_{k∈e} θ_k − |e| θ_i)``.

    Each hyperedge contributes the higher-order phase-difference coupling to each of its members.

    Parameters
    ----------
    theta : numpy.ndarray
        The oscillator phases ``θ`` (one-dimensional, length ``N``).
    hyperedges : sequence of sequence of int
        The hyperedge list; each hyperedge is a sequence of at least two distinct node indices in
        ``[0, N)``. Edges may mix arities.
    coupling : float or sequence of float
        The hyperedge weight ``K_e``: a scalar applied to every edge, or one weight per edge.

    Returns
    -------
    numpy.ndarray
        The force on each oscillator (length ``N``).

    Raises
    ------
    ValueError
        If ``theta`` is not a non-empty vector, a hyperedge is malformed, or the weights mismatch.
    """
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    if phases.ndim != 1 or phases.size < 1:
        raise ValueError("theta must be a non-empty one-dimensional array")
    edges = _resolve_hyperedges(hyperedges, phases.size)
    weights = _resolve_weights(coupling, len(edges))
    force = np.zeros(phases.size, dtype=np.float64)
    for members, weight in zip(edges, weights, strict=True):
        total = phases[members].sum()
        contribution = weight * np.sin(total - members.size * phases[members])
        np.add.at(force, members, contribution)
    return force


def hyperedge_jacobian(
    theta: NDArray[np.float64],
    hyperedges: Sequence[Sequence[int]],
    coupling: float | Sequence[float],
) -> NDArray[np.float64]:
    r"""Return the ``(N, N)`` Jacobian of the structural hypergraph force.

    For a hyperedge ``e`` of size ``m = |e|`` and a member ``i`` with argument
    ``α_{e,i} = Σ_{k∈e} θ_k − m θ_i``, the contribution to ``∂F_i/∂θ_l`` is
    ``K_e \cos(α_{e,i})`` for each other member ``l ∈ e \setminus \{i\}`` and
    ``-(m-1) K_e \cos(α_{e,i})`` for ``l = i``. The matrix is non-symmetric for a directed /
    weighted edge list but every row sums to zero (the global-phase Goldstone mode).

    Parameters
    ----------
    theta : numpy.ndarray
        The oscillator phases ``θ`` (one-dimensional, length ``N``).
    hyperedges : sequence of sequence of int
        The hyperedge list (see :func:`hyperedge_force`).
    coupling : float or sequence of float
        The hyperedge weight ``K_e`` (scalar or per-edge).

    Returns
    -------
    numpy.ndarray
        The ``(N, N)`` Jacobian matrix.

    Raises
    ------
    ValueError
        If ``theta`` is not a non-empty vector, a hyperedge is malformed, or the weights mismatch.
    """
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    if phases.ndim != 1 or phases.size < 1:
        raise ValueError("theta must be a non-empty one-dimensional array")
    edges = _resolve_hyperedges(hyperedges, phases.size)
    weights = _resolve_weights(coupling, len(edges))
    jacobian = np.zeros((phases.size, phases.size), dtype=np.float64)
    for members, weight in zip(edges, weights, strict=True):
        size = members.size
        total = phases[members].sum()
        sensitivity = weight * np.cos(total - size * phases[members])  # one per member (row)
        rows = np.repeat(members, size)
        columns = np.tile(members, size)
        np.add.at(jacobian, (rows, columns), np.repeat(sensitivity, size))
        np.add.at(jacobian, (members, members), -size * sensitivity)
    return jacobian


__all__ = [
    "hyperedge_force",
    "hyperedge_jacobian",
]
