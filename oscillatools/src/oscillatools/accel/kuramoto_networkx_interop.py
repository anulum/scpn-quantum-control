# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — NetworkX graph interop for Kuramoto coupling matrices
r"""Build Kuramoto coupling matrices from NetworkX graphs (and export them back).

Practitioners hold network topology in `NetworkX <https://networkx.org>`_ graph objects; the toolkit
consumes an ``(N, N)`` coupling matrix ``K`` whose row ``j`` weights the influence of every neighbour
``k`` on oscillator ``j`` (the networked force is :math:`F_j = \sum_k K_{jk}\sin(\theta_k -
\theta_j)`). This module adapts between the two representations:

* :func:`coupling_from_networkx` reads any NetworkX-compatible graph into a coupling matrix. The
  ingestion is **duck-typed on the documented Graph API** (``is_directed`` / ``is_multigraph`` /
  ``nodes`` / ``edges(data=True)``), so NetworkX never has to be imported — and is **not** a
  dependency of this package. A directed edge ``u → v`` means *u drives v* and lands in ``K[v, u]``;
  an undirected edge couples both ways; parallel edges of a multigraph sum (the
  ``networkx.to_numpy_array`` convention).
* :func:`graph_from_networked_coupling` exports a coupling matrix back to a NetworkX graph for graph
  algorithms and drawing. Only this direction constructs graph objects, so it lazily imports
  ``networkx`` and raises a clear install hint when the library is absent.

This is a facade ADAPT slice — pure input/output adaptation with no compute kernel and no hot loop —
so it carries no Rust/Julia counterpart and no tier benchmark (the matrices it builds feed the
polyglot-dispatched forces and integrators unchanged).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class GraphLike(Protocol):
    """The duck-typed subset of the NetworkX ``Graph`` API the ingestion consumes.

    Any object exposing these four members works — a real :class:`networkx.Graph` /
    :class:`networkx.DiGraph` / :class:`networkx.MultiGraph`, or a lightweight stand-in.
    """

    def is_directed(self) -> bool:
        """Whether edges are one-way (``u → v``) rather than symmetric."""
        ...

    def is_multigraph(self) -> bool:
        """Whether parallel edges may repeat between the same node pair."""
        ...

    def nodes(self) -> Any:
        """The nodes in insertion order (iterable of hashables)."""
        ...

    def edges(self, data: bool = ...) -> Any:
        """The edges as ``(u, v, attributes)`` triples when ``data=True``."""
        ...


def coupling_from_networkx(
    graph: GraphLike,
    *,
    weight: str = "weight",
    default_weight: float = 1.0,
    nodelist: Sequence[Hashable] | None = None,
) -> NDArray[np.float64]:
    r"""Return the ``(N, N)`` Kuramoto coupling matrix encoded by a NetworkX-compatible graph.

    Row ``j`` of the returned matrix weights the influence of every neighbour on oscillator ``j``
    (the convention of :func:`~oscillatools.accel.networked_kuramoto.networked_kuramoto_force`): a
    directed edge ``u → v`` with weight ``w`` sets ``K[v, u] = w``, an undirected edge sets both
    ``K[u, v]`` and ``K[v, u]``, and parallel edges of a multigraph sum their weights. Self-loops
    land on the diagonal (they are dynamically inert, since ``sin(θ_j − θ_j) = 0``).

    The graph is consumed through the documented Graph API only (duck typing), so NetworkX is not
    imported and is not a dependency — any object satisfying :class:`GraphLike` works.

    Parameters
    ----------
    graph : GraphLike
        The topology: a :class:`networkx.Graph` / ``DiGraph`` / ``MultiGraph`` / ``MultiDiGraph``,
        or any object exposing ``is_directed`` / ``is_multigraph`` / ``nodes`` / ``edges(data=True)``.
    weight : str, optional
        The edge-attribute key holding the coupling weight; defaults to ``"weight"``.
    default_weight : float, optional
        The weight used for edges that lack the attribute; defaults to ``1``.
    nodelist : sequence of hashable, optional
        The oscillator ordering. Must be a permutation of the graph's nodes; defaults to the graph's
        own node insertion order. Index ``i`` of the returned matrix corresponds to ``nodelist[i]``.

    Returns
    -------
    numpy.ndarray
        The ``(N, N)`` float64 coupling matrix ``K``.

    Raises
    ------
    ValueError
        If the graph has no nodes, ``nodelist`` is not a permutation of the graph's nodes, or an
        edge weight is non-numeric or non-finite.
    """
    graph_nodes = list(graph.nodes())
    if not graph_nodes:
        raise ValueError("graph must contain at least one node")
    ordering = graph_nodes if nodelist is None else list(nodelist)
    index = {node: position for position, node in enumerate(ordering)}
    count = len(ordering)
    if len(index) != count:
        raise ValueError("nodelist must not contain duplicate nodes")
    if set(index) != set(graph_nodes):
        raise ValueError("nodelist must be a permutation of the graph's nodes")

    directed = bool(graph.is_directed())
    coupling = np.zeros((count, count), dtype=np.float64)
    for source, target, attributes in graph.edges(data=True):
        raw = attributes.get(weight, default_weight)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(
                f"edge ({source!r}, {target!r}) has a non-numeric {weight!r} attribute: {raw!r}"
            )
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(f"edge ({source!r}, {target!r}) has a non-finite weight: {value}")
        row, column = index[target], index[source]
        # A multigraph's parallel edges accumulate; a plain graph reports each edge once.
        coupling[row, column] += value
        if not directed and row != column:
            coupling[column, row] += value
    return coupling


def graph_from_networked_coupling(
    coupling: NDArray[np.float64],
    *,
    threshold: float = 0.0,
    weight: str = "weight",
    symmetry_tolerance: float = 1e-12,
) -> Any:
    r"""Export an ``(N, N)`` coupling matrix as a NetworkX graph.

    Entries with ``|K[j, k]| > threshold`` become edges carrying the coupling as an edge attribute,
    inverting the :func:`coupling_from_networkx` convention (``K[v, u]`` becomes the edge ``u → v``).
    A matrix that is symmetric to ``symmetry_tolerance`` exports as an undirected
    :class:`networkx.Graph` (one edge per pair); anything else exports as a
    :class:`networkx.DiGraph`. Diagonal entries export as self-loops.

    This is the only direction that constructs graph objects, so it imports ``networkx`` lazily and
    raises an actionable hint when the optional library is missing.

    Parameters
    ----------
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K`` (row ``j`` weights the influence on oscillator ``j``).
    threshold : float, optional
        The absolute magnitude below which an entry is treated as no edge; defaults to ``0``.
    weight : str, optional
        The edge-attribute key to write the coupling into; defaults to ``"weight"``.
    symmetry_tolerance : float, optional
        The absolute tolerance for detecting a symmetric matrix; defaults to ``1e-12``.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        The exported topology, with nodes ``0 … N-1``.

    Raises
    ------
    ImportError
        If ``networkx`` is not installed (install with ``pip install networkx``).
    ValueError
        If ``coupling`` is not a square two-dimensional matrix or ``threshold`` is negative.
    """
    try:
        import networkx
    except ImportError as error:  # pragma: no cover - exercised only without networkx installed
        raise ImportError(
            "graph_from_networked_coupling requires the optional networkx library; "
            "install it with `pip install networkx`"
        ) from error

    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"coupling must be a non-empty square matrix, got shape {matrix.shape}")
    if threshold < 0.0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")

    symmetric = bool(np.allclose(matrix, matrix.T, atol=symmetry_tolerance, rtol=0.0))
    graph = networkx.Graph() if symmetric else networkx.DiGraph()
    count = matrix.shape[0]
    graph.add_nodes_from(range(count))
    for row in range(count):
        columns = range(row, count) if symmetric else range(count)
        for column in columns:
            value = float(matrix[row, column])
            if abs(value) > threshold:
                # K[row, column] is the influence of `column` on `row`: the edge column → row.
                graph.add_edge(column, row, **{weight: value})
    return graph


__all__ = [
    "GraphLike",
    "coupling_from_networkx",
    "graph_from_networked_coupling",
]
