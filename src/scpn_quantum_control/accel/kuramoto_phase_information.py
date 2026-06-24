# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Information-theoretic phase diagnostics for a Kuramoto ensemble
r"""Information-theoretic phase diagnostics: distribution entropy and mutual information.

The spread of a Kuramoto ensemble can be read as disorder. Binning the oscillator phases
into ``B`` equal arcs of the circle gives an empirical distribution :math:`p_b`, whose Shannon
entropy

.. math::

    H = -\sum_{b} p_b \ln p_b \in [0, \ln B]

is maximal :math:`(\ln B)` for the incoherent state — phases uniform on the circle — and
minimal :math:`(0)` for full synchrony, where every oscillator falls in one arc. The
normalised entropy :math:`H/\ln B` reports the same on :math:`[0, 1]`.

The pairwise mutual information

.. math::

    I(\theta_i, \theta_j) = \sum_{a, b} p_{ab} \ln \frac{p_{ab}}{p_a\,p_b}

is estimated from the joint histogram of two phase time series. It is zero when the two
oscillators are statistically independent and rises to the marginal entropy when they are
phase-locked (one phase determines the other), so the mutual-information matrix exposes the
information-sharing structure of a network — the synchronised clusters light up as blocks.

Both estimators are the plug-in (maximum-likelihood) form: the entropy is biased slightly
*downward* and the mutual information slightly *upward* by :math:`O((B-1)/N)` for ``N``
samples, so an independent pair shows a small positive residual rather than an exact zero.
The bias is negligible at the extremes the diagnostics target. Binning makes these quantities
piecewise-constant and hence non-differentiable, so — unlike the order-parameter observables —
no gradient is exposed.

This is a pure-Python analysis layer over NumPy histograms; it adds no compute kernel.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_TWO_PI = 2.0 * np.pi


def _validate_bins(bins: int) -> int:
    """Return ``bins`` after checking it is at least two."""
    if bins < 2:
        raise ValueError(f"bins must be at least 2, got {bins}")
    return bins


def _validate_angles(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a contiguous non-empty one-dimensional angle array."""
    values = np.ascontiguousarray(angles, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("angles must be a non-empty one-dimensional array")
    return values


def _distribution(angles: NDArray[np.float64], bins: int) -> NDArray[np.float64]:
    """Return the empirical phase distribution over ``bins`` equal arcs of the circle."""
    counts, _ = np.histogram(np.mod(angles, _TWO_PI), bins=bins, range=(0.0, _TWO_PI))
    total = counts.sum()
    return np.ascontiguousarray(counts / total, dtype=np.float64)


def _entropy_of(probabilities: NDArray[np.float64]) -> float:
    """Shannon entropy (nats) of a probability vector, summing only the positive entries."""
    positive = probabilities[probabilities > 0.0]
    return float(-(positive * np.log(positive)).sum())


def phase_entropy(angles: NDArray[np.float64], *, bins: int = 36) -> float:
    r"""Shannon entropy (nats) of the phase distribution.

    Bins the angles into ``bins`` equal arcs of :math:`[0, 2\pi)` and returns
    :math:`-\sum_b p_b \ln p_b`. The value is maximal :math:`(\ln \text{bins})` for phases
    spread uniformly around the circle and zero when every angle falls in one arc.

    Parameters
    ----------
    angles : numpy.ndarray
        One-dimensional array of angles in radians (an ensemble snapshot or a time series).
    bins : int, optional
        The number of equal arcs of the circle; must be at least two. Defaults to ``36``.

    Returns
    -------
    float
        The entropy in nats.

    Raises
    ------
    ValueError
        If ``angles`` is not a non-empty one-dimensional array or ``bins`` is below two.
    """
    values = _validate_angles(angles)
    return _entropy_of(_distribution(values, _validate_bins(bins)))


def normalised_phase_entropy(angles: NDArray[np.float64], *, bins: int = 36) -> float:
    r"""Phase entropy scaled to :math:`[0, 1]` by its maximum :math:`\ln \text{bins}`.

    Returns ``1`` for a uniform (incoherent) phase distribution and ``0`` for full synchrony,
    independent of the bin count.

    Parameters
    ----------
    angles : numpy.ndarray
        One-dimensional array of angles in radians.
    bins : int, optional
        The number of equal arcs of the circle; must be at least two. Defaults to ``36``.

    Returns
    -------
    float
        The normalised entropy in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``angles`` is not a non-empty one-dimensional array or ``bins`` is below two.
    """
    resolved = _validate_bins(bins)
    return phase_entropy(angles, bins=resolved) / float(np.log(resolved))


def phase_entropy_series(phases: NDArray[np.float64], *, bins: int = 36) -> NDArray[np.float64]:
    r"""Per-time ensemble phase entropy along a trajectory.

    Applies :func:`phase_entropy` to the oscillator phases at each time sample of a ``(T, N)``
    trajectory, returning the entropy as a function of time — a disorder time series that falls
    as the ensemble orders and rises as it disperses.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians.
    bins : int, optional
        The number of equal arcs of the circle; must be at least two. Defaults to ``36``.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array of ``T`` entropies in nats.

    Raises
    ------
    ValueError
        If ``phases`` is not a ``(T, N)`` array with ``T ≥ 1`` and ``N ≥ 1`` or ``bins`` is
        below two.
    """
    trajectory = np.ascontiguousarray(phases, dtype=np.float64)
    if trajectory.ndim != 2:
        raise ValueError(
            f"phases must be a two-dimensional (T, N) trajectory, got ndim {trajectory.ndim}"
        )
    if trajectory.shape[0] < 1 or trajectory.shape[1] < 1:
        raise ValueError("phases must hold at least one time sample and one oscillator")
    resolved = _validate_bins(bins)
    series = [_entropy_of(_distribution(row, resolved)) for row in trajectory]
    return np.ascontiguousarray(series, dtype=np.float64)


def pairwise_mutual_information(
    first: NDArray[np.float64], second: NDArray[np.float64], *, bins: int = 36
) -> float:
    r"""Mutual information (nats) between two phase time series.

    Estimates :math:`I = \sum_{a,b} p_{ab}\ln[p_{ab}/(p_a p_b)]` from the joint histogram of
    the paired angles. It is near zero for statistically independent oscillators and rises to
    the marginal entropy when one phase determines the other (phase locking).

    Parameters
    ----------
    first, second : numpy.ndarray
        One-dimensional angle series of equal length, in radians.
    bins : int, optional
        The number of equal arcs per axis; must be at least two. Defaults to ``36``.

    Returns
    -------
    float
        The mutual information in nats (non-negative up to the plug-in bias).

    Raises
    ------
    ValueError
        If the series are not equal-length non-empty one-dimensional arrays or ``bins`` is
        below two.
    """
    left = _validate_angles(first)
    right = _validate_angles(second)
    if left.size != right.size:
        raise ValueError(f"series must have equal length, got {left.size} and {right.size}")
    resolved = _validate_bins(bins)
    joint, _, _ = np.histogram2d(
        np.mod(left, _TWO_PI),
        np.mod(right, _TWO_PI),
        bins=resolved,
        range=[[0.0, _TWO_PI], [0.0, _TWO_PI]],
    )
    pij = joint / joint.sum()
    marginal_left = pij.sum(axis=1)
    marginal_right = pij.sum(axis=0)
    mask = pij > 0.0
    outer = marginal_left[:, None] * marginal_right[None, :]
    return float((pij[mask] * np.log(pij[mask] / outer[mask])).sum())


def mutual_information_matrix(
    phases: NDArray[np.float64], *, bins: int = 36
) -> NDArray[np.float64]:
    r"""Symmetric matrix of pairwise mutual information across a trajectory.

    For a ``(T, N)`` trajectory returns the ``(N, N)`` matrix whose entry :math:`(i, j)` is the
    mutual information between oscillators ``i`` and ``j`` estimated over the ``T`` time
    samples. The diagonal is each oscillator's marginal phase entropy (its self-information),
    and synchronised clusters appear as high-information blocks.

    Parameters
    ----------
    phases : numpy.ndarray
        Two-dimensional ``(T, N)`` array of oscillator phases in radians; ``T`` should comfortably
        exceed ``bins**2`` for the joint histogram to be populated.
    bins : int, optional
        The number of equal arcs per axis; must be at least two. Defaults to ``36``.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` symmetric float64 mutual-information matrix.

    Raises
    ------
    ValueError
        If ``phases`` is not a ``(T, N)`` array with ``T ≥ 1`` and ``N ≥ 1`` or ``bins`` is
        below two.
    """
    trajectory = np.ascontiguousarray(phases, dtype=np.float64)
    if trajectory.ndim != 2:
        raise ValueError(
            f"phases must be a two-dimensional (T, N) trajectory, got ndim {trajectory.ndim}"
        )
    if trajectory.shape[0] < 1 or trajectory.shape[1] < 1:
        raise ValueError("phases must hold at least one time sample and one oscillator")
    resolved = _validate_bins(bins)
    count = trajectory.shape[1]
    matrix = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        matrix[i, i] = _entropy_of(_distribution(trajectory[:, i], resolved))
        for j in range(i + 1, count):
            value = pairwise_mutual_information(trajectory[:, i], trajectory[:, j], bins=resolved)
            matrix[i, j] = value
            matrix[j, i] = value
    return np.ascontiguousarray(matrix, dtype=np.float64)


__all__ = [
    "mutual_information_matrix",
    "normalised_phase_entropy",
    "pairwise_mutual_information",
    "phase_entropy",
    "phase_entropy_series",
]
