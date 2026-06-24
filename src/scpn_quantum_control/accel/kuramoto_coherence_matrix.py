# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Pairwise phase-coherence matrices and their spectral structure
r"""Pairwise phase-coherence matrices for a Kuramoto ensemble and their spectral structure.

Where the order parameter collapses the whole ensemble to one number, the pairwise coherence
matrix keeps the full relational picture — which oscillators move together. Three views are
provided:

* the instantaneous coherence :math:`Q_{jk} = \cos(\theta_j - \theta_k) \in [-1, 1]`, a
  snapshot whose ``+1`` / ``-1`` blocks are in-phase / antiphase pairs;
* the time-averaged coherence :math:`C_{jk} = \langle \cos(\theta_j - \theta_k)\rangle_t \in
  [-1, 1]`, which keeps the signed block structure of persistent clusters;
* the phase-locking value :math:`\rho_{jk} = |\langle e^{i(\theta_j - \theta_k)}\rangle_t| \in
  [0, 1]`, which measures how *constant* the relative phase is regardless of its offset — it is
  one for any rigidly locked pair (in-phase or antiphase) and zero for independent oscillators.

All three are symmetric with a unit diagonal. The signed coherence matrices are positive
semidefinite combinations of the phase vectors, so their spectral structure is informative: the
eigenvector of the largest eigenvalue assigns oscillators to clusters by sign (a single block
gives a uniform-sign eigenvector; two antiphase clusters split it into opposite signs), which is
the basis for the cluster metrics of the companion module.

This is a pure-Python analysis layer over NumPy linear algebra; it adds no compute kernel.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_snapshot(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a contiguous non-empty one-dimensional phase vector."""
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    if phases.ndim != 1 or phases.size == 0:
        raise ValueError("theta must be a non-empty one-dimensional array of phases")
    return phases


def _validate_trajectory(phases: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a contiguous ``(T, N)`` trajectory after validating its shape."""
    trajectory = np.ascontiguousarray(phases, dtype=np.float64)
    if trajectory.ndim != 2:
        raise ValueError(
            f"phases must be a two-dimensional (T, N) trajectory, got ndim {trajectory.ndim}"
        )
    if trajectory.shape[0] < 1 or trajectory.shape[1] < 1:
        raise ValueError("phases must hold at least one time sample and one oscillator")
    return trajectory


def coherence_matrix(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Instantaneous pairwise coherence :math:`Q_{jk} = \cos(\theta_j - \theta_k)`.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of ``N`` oscillator phases in radians.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(N, N)`` float64 matrix in ``[-1, 1]`` with a unit diagonal; ``+1`` blocks
        are in-phase pairs and ``-1`` blocks antiphase pairs.

    Raises
    ------
    ValueError
        If ``theta`` is not a non-empty one-dimensional array.
    """
    phases = _validate_snapshot(theta)
    cos = np.cos(phases)
    sin = np.sin(phases)
    return np.ascontiguousarray(np.outer(cos, cos) + np.outer(sin, sin), dtype=np.float64)


def mean_coherence_matrix(phases: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Time-averaged coherence :math:`C_{jk} = \langle \cos(\theta_j - \theta_k)\rangle_t`.

    The temporal mean of the instantaneous coherence over a trajectory; it keeps the signed
    block structure of persistent clusters (``+1`` within a cluster, ``-1`` across an antiphase
    pair of clusters).

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(N, N)`` float64 matrix in ``[-1, 1]`` with a unit diagonal.

    Raises
    ------
    ValueError
        If ``phases`` is not a ``(T, N)`` array with ``T ≥ 1`` and ``N ≥ 1``.
    """
    trajectory = _validate_trajectory(phases)
    samples = trajectory.shape[0]
    cos = np.cos(trajectory)
    sin = np.sin(trajectory)
    matrix = (cos.T @ cos + sin.T @ sin) / samples
    return np.ascontiguousarray(matrix, dtype=np.float64)


def phase_locking_matrix(phases: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Pairwise phase-locking value :math:`\rho_{jk} = |\langle e^{i(\theta_j-\theta_k)}\rangle_t|`.

    The modulus of the time-averaged relative phase — one for a rigidly locked pair whatever its
    constant offset, zero for statistically independent oscillators. Unlike the signed coherence,
    it does not distinguish in-phase from antiphase locking; it measures only locking strength.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(N, N)`` float64 matrix in ``[0, 1]`` with a unit diagonal.

    Raises
    ------
    ValueError
        If ``phases`` is not a ``(T, N)`` array with ``T ≥ 1`` and ``N ≥ 1``.
    """
    trajectory = _validate_trajectory(phases)
    samples = trajectory.shape[0]
    rotors = np.exp(1j * trajectory)  # (T, N)
    matrix = np.abs(rotors.conj().T @ rotors) / samples
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _validate_square(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a contiguous square coherence matrix after validating its shape."""
    array = np.ascontiguousarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] == 0:
        raise ValueError(
            f"matrix must be a non-empty square (N, N) array, got shape {array.shape}"
        )
    return array


def coherence_spectrum(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Eigenvalues and eigenvectors of a symmetric coherence matrix, largest first.

    Diagonalises the (assumed symmetric) coherence matrix and returns the eigenvalues in
    descending order together with the matching eigenvectors as columns. The leading eigenvector
    is the cluster indicator — oscillators of one block share its sign.

    Parameters
    ----------
    matrix : numpy.ndarray
        A symmetric ``(N, N)`` coherence matrix, as returned by :func:`coherence_matrix`,
        :func:`mean_coherence_matrix` or :func:`phase_locking_matrix`.

    Returns
    -------
    eigenvalues : numpy.ndarray
        One-dimensional ``(N,)`` float64 array of eigenvalues, sorted in descending order.
    eigenvectors : numpy.ndarray
        ``(N, N)`` float64 array whose column ``i`` is the unit eigenvector of
        ``eigenvalues[i]``. Each eigenvector's overall sign is arbitrary.

    Raises
    ------
    ValueError
        If ``matrix`` is not a non-empty square array.
    """
    array = _validate_square(matrix)
    values, vectors = np.linalg.eigh(array)
    order = np.argsort(values)[::-1]
    return (
        np.ascontiguousarray(values[order], dtype=np.float64),
        np.ascontiguousarray(vectors[:, order], dtype=np.float64),
    )


def leading_coherence_eigenvector(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Eigenvector of the largest eigenvalue of a symmetric coherence matrix.

    The dominant synchronisation mode — its sign pattern assigns oscillators to coherent
    clusters. The overall sign of the vector is arbitrary.

    Parameters
    ----------
    matrix : numpy.ndarray
        A symmetric ``(N, N)`` coherence matrix.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``(N,)`` float64 unit eigenvector of the largest eigenvalue.

    Raises
    ------
    ValueError
        If ``matrix`` is not a non-empty square array.
    """
    _, vectors = coherence_spectrum(matrix)
    return np.ascontiguousarray(vectors[:, 0], dtype=np.float64)


__all__ = [
    "coherence_matrix",
    "coherence_spectrum",
    "leading_coherence_eigenvector",
    "mean_coherence_matrix",
    "phase_locking_matrix",
]
