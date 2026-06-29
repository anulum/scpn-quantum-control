# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Sparse identification of phase-oscillator network dynamics
r"""Sparse, network-aware discovery of the governing equations of phase-oscillator dynamics.

Where the coupling-function and Bayesian estimators *assume* the Kuramoto structure and fit it, this
module *discovers* the structure from data: it does not presume which interactions are present or what
functional form they take. From a candidate library of trigonometric interaction terms it selects, by
sparse regression, the few terms that actually generate the observed dynamics — recovering both the
*network topology* (which pairs couple) and the *functional form* (which harmonics, sine versus
cosine), with the absent terms driven to exact zero.

For each node ``i`` the candidate library spans the natural pairwise harmonics

.. math::

    \dot\theta_i \;=\; \omega_i \;+\; \sum_{j\neq i}\sum_{m=1}^{M}
        \bigl[a^{(m)}_{ij}\,\sin\!\bigl(m(\theta_j-\theta_i)\bigr)
            + b^{(m)}_{ij}\,\cos\!\bigl(m(\theta_j-\theta_i)\bigr)\bigr],

and the coefficients are found by sequentially-thresholded least squares (Brunton–Proctor–Kutz
SINDy): an ordinary least-squares fit, then repeatedly zero every coefficient below the sparsity
threshold and refit on the surviving terms until the active set stabilises. A classic Kuramoto network
is recovered as ``a^{(1)}_{ij} = K_{ij}`` with every other coefficient exactly zero; a
Sakaguchi–Kuramoto network appears as a paired ``a^{(1)}, b^{(1)}`` (``K\cos\beta, -K\sin\beta``); a
biharmonic (Hansel–Daido) coupling lights up the ``m = 2`` terms. The hard zeros are the discovery:
they reconstruct the directed adjacency and reject spurious couplings outright, rather than returning
a dense matrix of small numbers.

This is the network-aware data-driven model-discovery layer (PI-NDSR/SINDy family) complementing the
inference modules. It adds no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SparseDynamicsModel:
    """A sparse discovered model of a phase-oscillator network.

    Attributes
    ----------
    frequencies : numpy.ndarray
        The ``(N,)`` discovered natural frequencies ``ω`` (the constant library term per node).
    sine_coupling : numpy.ndarray
        The ``(M, N, N)`` sine coefficients; ``sine_coupling[m - 1, i, j]`` multiplies
        ``sin(m(θ_j − θ_i))`` in node ``i``'s equation (the ``m = 1`` slice is the directed Kuramoto
        coupling ``K``). The diagonal ``j = i`` is zero.
    cosine_coupling : numpy.ndarray
        The ``(M, N, N)`` cosine coefficients; ``cosine_coupling[m - 1, i, j]`` multiplies
        ``cos(m(θ_j − θ_i))``. The diagonal is zero.
    residual : float
        The root-mean-square residual of the discovered model against the supplied derivatives.
    """

    frequencies: NDArray[np.float64]
    sine_coupling: NDArray[np.float64]
    cosine_coupling: NDArray[np.float64]
    residual: float

    @property
    def active_terms(self) -> int:
        """The number of non-zero terms in the discovered model (its sparsity)."""
        return int(
            np.count_nonzero(self.frequencies)
            + np.count_nonzero(self.sine_coupling)
            + np.count_nonzero(self.cosine_coupling)
        )

    def field(self, phases: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the discovered vector field ``θ̇`` at the phase state ``phases``."""
        state = np.ascontiguousarray(phases, dtype=np.float64)
        difference = state[None, :] - state[:, None]
        velocity = np.array(self.frequencies, dtype=np.float64)
        for harmonic in range(self.sine_coupling.shape[0]):
            scaled = (harmonic + 1) * difference
            velocity = velocity + np.sum(
                self.sine_coupling[harmonic] * np.sin(scaled)
                + self.cosine_coupling[harmonic] * np.cos(scaled),
                axis=1,
            )
        return np.asarray(velocity, dtype=np.float64)


def _node_library(
    phases: NDArray[np.float64],
    node: int,
    n_harmonics: int,
) -> tuple[NDArray[np.float64], list[int]]:
    """Build node ``i``'s candidate library and the source-node index of each column block."""
    count = phases.shape[1]
    others = [other for other in range(count) if other != node]
    columns: list[NDArray[np.float64]] = [np.ones(phases.shape[0], dtype=np.float64)]
    for other in others:
        difference = phases[:, other] - phases[:, node]
        for harmonic in range(1, n_harmonics + 1):
            columns.append(np.asarray(np.sin(harmonic * difference), dtype=np.float64))
            columns.append(np.asarray(np.cos(harmonic * difference), dtype=np.float64))
    return np.column_stack(columns), others


def _sequentially_thresholded_least_squares(
    library: NDArray[np.float64],
    target: NDArray[np.float64],
    threshold: float,
    max_iterations: int,
) -> NDArray[np.float64]:
    """Sparse-regression coefficients by sequentially-thresholded least squares (SINDy)."""
    coefficients = np.linalg.lstsq(library, target, rcond=None)[0]
    for _ in range(max_iterations):
        small = np.abs(coefficients) < threshold
        previous_small = small.copy()
        coefficients[small] = 0.0
        active = ~small
        if active.any():
            coefficients[active] = np.linalg.lstsq(library[:, active], target, rcond=None)[0]
        next_small = np.abs(coefficients) < threshold
        if np.array_equal(next_small, previous_small):
            break
    coefficients[np.abs(coefficients) < threshold] = 0.0
    return np.asarray(coefficients, dtype=np.float64)


def discover_phase_dynamics(
    phases: NDArray[np.float64],
    derivatives: NDArray[np.float64],
    *,
    n_harmonics: int,
    threshold: float,
    max_iterations: int = 10,
) -> SparseDynamicsModel:
    r"""Discover the sparse governing equations of a phase-oscillator network from data.

    Fits each node's phase velocity against a trigonometric candidate library by
    sequentially-thresholded least squares, returning the sparse discovered model: the active terms
    reconstruct the directed topology and the functional form, with absent terms exactly zero.

    Parameters
    ----------
    phases : numpy.ndarray
        The ``(n_samples, N)`` observed phase snapshots (``n_samples ≥ 1``, ``N ≥ 2``).
    derivatives : numpy.ndarray
        The ``(n_samples, N)`` observed phase velocities ``θ̇``.
    n_harmonics : int
        The highest harmonic ``M`` in the candidate library (``≥ 1``).
    threshold : float
        The sparsity threshold; coefficients below it (in magnitude) are set to zero (``> 0``).
    max_iterations : int, optional
        The maximum sequential-thresholding iterations; defaults to ``10``.

    Returns
    -------
    SparseDynamicsModel
        The discovered frequencies and sparse sine / cosine coupling tensors.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    snapshots = np.ascontiguousarray(phases, dtype=np.float64)
    velocity = np.ascontiguousarray(derivatives, dtype=np.float64)
    if snapshots.ndim != 2 or snapshots.shape[0] < 1 or snapshots.shape[1] < 2:
        raise ValueError("phases must be a (n_samples >= 1, N >= 2) array")
    if velocity.shape != snapshots.shape:
        raise ValueError(f"derivatives must match phases shape, got {velocity.shape}")
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be positive, got {n_harmonics}")
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be positive, got {max_iterations}")

    count = snapshots.shape[1]
    frequencies = np.zeros(count, dtype=np.float64)
    sine_coupling = np.zeros((n_harmonics, count, count), dtype=np.float64)
    cosine_coupling = np.zeros((n_harmonics, count, count), dtype=np.float64)
    total_squared = 0.0
    for node in range(count):
        library, others = _node_library(snapshots, node, n_harmonics)
        coefficients = _sequentially_thresholded_least_squares(
            library, velocity[:, node], threshold, max_iterations
        )
        frequencies[node] = coefficients[0]
        for position, other in enumerate(others):
            base = 1 + position * 2 * n_harmonics
            for harmonic in range(n_harmonics):
                sine_coupling[harmonic, node, other] = coefficients[base + 2 * harmonic]
                cosine_coupling[harmonic, node, other] = coefficients[base + 2 * harmonic + 1]
        total_squared += float(np.sum((library @ coefficients - velocity[:, node]) ** 2))

    residual = float(np.sqrt(total_squared / velocity.size))
    return SparseDynamicsModel(
        frequencies=frequencies,
        sine_coupling=sine_coupling,
        cosine_coupling=cosine_coupling,
        residual=residual,
    )


__all__ = [
    "SparseDynamicsModel",
    "discover_phase_dynamics",
]
