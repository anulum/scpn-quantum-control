# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-cluster and partial-synchronisation metrics
r"""Phase-cluster and partial-synchronisation metrics from a coherence matrix.

Between full synchrony and incoherence a Kuramoto ensemble breaks into *clusters* — groups of
mutually coherent oscillators. Given a pairwise coherence matrix (the signed coherence
:math:`C_{jk} = \langle\cos(\theta_j-\theta_k)\rangle_t` or the phase-locking value
:math:`\rho_{jk}` of the companion module) the clusters are recovered by thresholding the matrix
into an adjacency graph — an edge wherever the coherence reaches ``threshold`` — and taking its
connected components. Two oscillators land in the same cluster when a chain of strongly coherent
pairs links them.

The notion of "cluster" follows the matrix supplied: the signed coherence groups oscillators by
*phase value* (an antiphase pair of groups stays two clusters), while the phase-locking value
groups by *locking* (the same antiphase pair merges into one, since their relative phase is
constant). The metrics report the cluster count, the sizes (largest first) and the mean
intra-cluster coherence of each cluster — one for a tight cluster and for the singleton
convention of an isolated oscillator.

This is a pure-Python analysis layer over a connected-components pass on the coherence matrix; it
adds no compute kernel. Thresholding is discrete, so — unlike the order-parameter observables —
no gradient is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components


def _validate_square(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a contiguous non-empty square coherence matrix after validating its shape."""
    array = np.ascontiguousarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] == 0:
        raise ValueError(
            f"matrix must be a non-empty square (N, N) array, got shape {array.shape}"
        )
    return array


def phase_clusters(matrix: NDArray[np.float64], *, threshold: float = 0.9) -> NDArray[np.int_]:
    r"""Cluster label of each oscillator from a thresholded coherence matrix.

    Builds the adjacency graph ``matrix >= threshold`` and labels its connected components, so
    two oscillators share a label when a chain of pairwise coherences at or above ``threshold``
    links them. Labels are assigned by descending cluster size (label ``0`` is the largest
    cluster), with ties broken by the smallest member index.

    Parameters
    ----------
    matrix : numpy.ndarray
        A symmetric ``(N, N)`` coherence matrix, e.g. from
        :func:`~scpn_quantum_control.accel.kuramoto_coherence_matrix.mean_coherence_matrix` or
        :func:`~scpn_quantum_control.accel.kuramoto_coherence_matrix.phase_locking_matrix`.
    threshold : float, optional
        The coherence at or above which a pair is joined. Defaults to ``0.9``.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``(N,)`` integer array of cluster labels in ``[0, K)`` for ``K`` clusters.

    Raises
    ------
    ValueError
        If ``matrix`` is not a non-empty square array.
    """
    array = _validate_square(matrix)
    adjacency = array >= threshold
    component_count, raw = connected_components(adjacency, directed=False)
    sizes = np.array([(raw == label).sum() for label in range(component_count)])
    order = np.argsort(-sizes, kind="stable")
    remap = np.empty(component_count, dtype=np.int_)
    remap[order] = np.arange(component_count)
    return np.ascontiguousarray(remap[raw], dtype=np.int_)


def cluster_count(matrix: NDArray[np.float64], *, threshold: float = 0.9) -> int:
    """Number of phase clusters in a thresholded coherence matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        A symmetric ``(N, N)`` coherence matrix.
    threshold : float, optional
        The coherence at or above which a pair is joined. Defaults to ``0.9``.

    Returns
    -------
    int
        The cluster count; ``1`` for a fully synchronised ensemble.

    Raises
    ------
    ValueError
        If ``matrix`` is not a non-empty square array.
    """
    return int(phase_clusters(matrix, threshold=threshold).max()) + 1


@dataclass(frozen=True)
class ClusterPartition:
    """The cluster structure of a coherence matrix.

    Attributes
    ----------
    labels : numpy.ndarray
        The ``(N,)`` cluster label of each oscillator, by descending cluster size.
    count : int
        The number of clusters.
    sizes : numpy.ndarray
        The ``(count,)`` cluster sizes, largest first.
    coherences : numpy.ndarray
        The ``(count,)`` mean intra-cluster coherence of each cluster; ``1.0`` for a singleton.
    """

    labels: NDArray[np.int_]
    count: int
    sizes: NDArray[np.int_]
    coherences: NDArray[np.float64]


def cluster_partition(matrix: NDArray[np.float64], *, threshold: float = 0.9) -> ClusterPartition:
    r"""Full cluster partition of a coherence matrix: labels, count, sizes and coherences.

    Labels the oscillators with :func:`phase_clusters`, then reports the cluster count, the
    sizes (largest first) and the mean intra-cluster coherence of each cluster — the average of
    the off-diagonal coherences among its members, or ``1.0`` for a single-oscillator cluster.

    Parameters
    ----------
    matrix : numpy.ndarray
        A symmetric ``(N, N)`` coherence matrix.
    threshold : float, optional
        The coherence at or above which a pair is joined. Defaults to ``0.9``.

    Returns
    -------
    ClusterPartition
        The labels, cluster count, descending sizes and per-cluster mean coherence.

    Raises
    ------
    ValueError
        If ``matrix`` is not a non-empty square array.
    """
    array = _validate_square(matrix)
    labels = phase_clusters(array, threshold=threshold)
    count = int(labels.max()) + 1
    sizes = np.empty(count, dtype=np.int_)
    coherences = np.empty(count, dtype=np.float64)
    for label in range(count):
        members = np.where(labels == label)[0]
        sizes[label] = members.size
        if members.size == 1:
            coherences[label] = 1.0
        else:
            block = array[np.ix_(members, members)]
            upper = np.triu_indices(members.size, 1)
            coherences[label] = float(block[upper].mean())
    return ClusterPartition(
        labels=labels,
        count=count,
        sizes=np.ascontiguousarray(sizes, dtype=np.int_),
        coherences=np.ascontiguousarray(coherences, dtype=np.float64),
    )


__all__ = [
    "ClusterPartition",
    "cluster_count",
    "cluster_partition",
    "phase_clusters",
]
