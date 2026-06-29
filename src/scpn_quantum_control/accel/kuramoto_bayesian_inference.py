# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Dynamical Bayesian inference and network reconstruction
r"""Dynamical Bayesian inference of directed coupling and network reconstruction.

Given a noisy phase time series from a stochastic phase-oscillator network
``φ̇_i = ω_i + Σ_j K_{ij} \sin(φ_j - φ_i) + ξ_i`` (independent white noise ``⟨ξ_i ξ_j⟩ = E_i δ_{ij}``),
this module reconstructs the *directed* coupling matrix ``K`` and the natural frequencies ``ω`` —
with calibrated uncertainty — by dynamical Bayesian inference (Stankovski/Duggento). The drift is
linear in the unknowns, so with a Gaussian prior the posterior is Gaussian and available in closed
form: for each node the Euler–Maruyama transition likelihood (the exact likelihood of the toolkit's
own noisy integrator, ``φ_{n+1} - φ_n = h\,f(φ_n) + \sqrt{E\,h}\,z_n``) gives the conjugate update

.. math::

    \Xi = \Xi_{\text{prior}} + \frac{h}{E}\,\sum_n p_n p_n^{\mathsf T}, \qquad
    \bar c = \Xi^{-1}\Bigl(\Xi_{\text{prior}}\,\bar c_{\text{prior}}
              + \frac{1}{E}\sum_n p_n\,(φ_{n+1}-φ_n)\Bigr),

with the per-node design ``p_n = [1, \sin(φ_j - φ_i)]_{j\neq i}`` and the noise ``E`` estimated from
the residuals. The posterior covariance ``\Xi^{-1}`` quantifies the reconstruction uncertainty;
because the estimator is the maximum-likelihood estimator of the generating model it is consistent
(the error shrinks as the record lengthens) and its credible intervals are calibrated. The recovered
``K`` is *directed* — ``K_{ij}`` need not equal ``K_{ji}`` — which is exactly the directionality of
influence.

Time-varying coupling is tracked by the Bayesian window propagation of Stankovski: the series is cut
into windows, each window's posterior becomes the next window's prior with its covariance inflated to
admit parameter drift, so a slowly changing coupling is followed window by window.

This complements the coupling-*function* inference (which learns the interaction *shape* ``Γ``) and
the trajectory-match coupling-*matrix* system identification (which is deterministic and gives no
uncertainty): here the data are stochastic and the output is a full posterior over the directed
network. It adds no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DynamicalBayesianPosterior:
    """The posterior over the directed network and frequencies.

    Attributes
    ----------
    frequencies : numpy.ndarray
        The ``(N,)`` posterior-mean natural frequencies ``ω``.
    coupling : numpy.ndarray
        The ``(N, N)`` posterior-mean directed coupling; ``coupling[i, j]`` is the influence of
        oscillator ``j`` on oscillator ``i`` (the diagonal is zero).
    frequency_std : numpy.ndarray
        The ``(N,)`` posterior standard deviations of the frequencies.
    coupling_std : numpy.ndarray
        The ``(N, N)`` posterior standard deviations of the coupling entries (diagonal zero).
    noise : numpy.ndarray
        The ``(N,)`` estimated per-node noise intensities ``E_i``.
    """

    frequencies: NDArray[np.float64]
    coupling: NDArray[np.float64]
    frequency_std: NDArray[np.float64]
    coupling_std: NDArray[np.float64]
    noise: NDArray[np.float64]


@dataclass(frozen=True)
class TimeVaryingCouplingHistory:
    """A window-by-window reconstruction of a time-varying directed coupling.

    Attributes
    ----------
    window_times : numpy.ndarray
        The ``(W,)`` start time of each window.
    frequencies : numpy.ndarray
        The ``(W, N)`` posterior-mean frequencies per window.
    coupling : numpy.ndarray
        The ``(W, N, N)`` posterior-mean directed coupling per window.
    coupling_std : numpy.ndarray
        The ``(W, N, N)`` posterior standard deviations per window.
    """

    window_times: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    coupling: NDArray[np.float64]
    coupling_std: NDArray[np.float64]


def _node_design(phases: NDArray[np.float64], node: int) -> NDArray[np.float64]:
    """Build the ``(n_steps, N)`` design ``[1, sin(φ_j − φ_i)]`` at the step start points for ``node``."""
    starts = phases[:-1]
    count = phases.shape[1]
    columns: list[NDArray[np.float64]] = [np.ones(starts.shape[0], dtype=np.float64)]
    columns.extend(
        np.asarray(np.sin(starts[:, other] - starts[:, node]), dtype=np.float64)
        for other in range(count)
        if other != node
    )
    return np.column_stack(columns)


def _node_posterior(
    design: NDArray[np.float64],
    increments: NDArray[np.float64],
    dt: float,
    prior_mean: NDArray[np.float64],
    prior_precision: NDArray[np.float64],
    noise_iterations: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Closed-form Gaussian posterior ``(mean, covariance, noise)`` for one node's parameters."""
    gram = design.T @ design
    cross = design.T @ increments
    prior_term = prior_precision @ prior_mean
    mean = np.linalg.solve(gram * dt + np.eye(design.shape[1]) * 1e-12, cross)
    precision = prior_precision
    for _ in range(noise_iterations):
        residual = increments - dt * (design @ mean)
        noise = float(np.mean(residual**2) / dt)
        precision = prior_precision + (dt / noise) * gram
        mean = np.linalg.solve(precision, prior_term + cross / noise)
    covariance = np.linalg.inv(precision)
    return (
        np.asarray(mean, dtype=np.float64),
        np.asarray(covariance, dtype=np.float64),
        noise,
    )


def _validate(phases: NDArray[np.float64], dt: float) -> int:
    if phases.ndim != 2 or phases.shape[0] < 2 or phases.shape[1] < 2:
        raise ValueError("phases must be a (n_samples >= 2, N >= 2) array")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    return int(phases.shape[1])


def _assemble(
    means: list[NDArray[np.float64]],
    covariances: list[NDArray[np.float64]],
    noises: list[float],
    count: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Pack the per-node posteriors into frequency / coupling mean and standard-deviation arrays."""
    frequencies = np.array([means[node][0] for node in range(count)], dtype=np.float64)
    frequency_std = np.array(
        [np.sqrt(covariances[node][0, 0]) for node in range(count)], dtype=np.float64
    )
    coupling = np.zeros((count, count), dtype=np.float64)
    coupling_std = np.zeros((count, count), dtype=np.float64)
    for node in range(count):
        others = [other for other in range(count) if other != node]
        deviations = np.sqrt(np.diag(covariances[node]))
        for position, other in enumerate(others):
            coupling[node, other] = means[node][1 + position]
            coupling_std[node, other] = deviations[1 + position]
    return frequencies, coupling, frequency_std, coupling_std, np.array(noises, dtype=np.float64)


def infer_network_bayesian(
    phases: NDArray[np.float64],
    dt: float,
    *,
    prior_precision: float = 1e-6,
    noise_iterations: int = 4,
) -> DynamicalBayesianPosterior:
    r"""Reconstruct the directed coupling and frequencies with calibrated uncertainty.

    Performs the per-node conjugate Gaussian update on the Euler–Maruyama transition likelihood and
    returns the posterior over the directed coupling matrix and the natural frequencies.

    Parameters
    ----------
    phases : numpy.ndarray
        The ``(n_samples, N)`` observed phase time series (continuous, not wrapped; ``n_samples ≥ 2``,
        ``N ≥ 2``), sampled at interval ``dt``.
    dt : float
        The sampling interval ``> 0``.
    prior_precision : float, optional
        The isotropic Gaussian-prior precision on every parameter (a weak ``1e-6`` by default, which
        is an almost-flat prior recovering the maximum-likelihood estimate).
    noise_iterations : int, optional
        The number of noise-estimation refinement iterations (``≥ 1``); defaults to ``4``.

    Returns
    -------
    DynamicalBayesianPosterior
        The posterior means and standard deviations of the directed coupling and frequencies, and the
        estimated per-node noise.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    series = np.ascontiguousarray(phases, dtype=np.float64)
    count = _validate(series, dt)
    if prior_precision <= 0.0:
        raise ValueError(f"prior_precision must be positive, got {prior_precision}")
    if noise_iterations < 1:
        raise ValueError(f"noise_iterations must be positive, got {noise_iterations}")

    prior_matrix = prior_precision * np.eye(count, dtype=np.float64)
    prior_mean = np.zeros(count, dtype=np.float64)
    means, covariances, noises = [], [], []
    for node in range(count):
        design = _node_design(series, node)
        increments = series[1:, node] - series[:-1, node]
        mean, covariance, noise = _node_posterior(
            design, increments, dt, prior_mean, prior_matrix, noise_iterations
        )
        means.append(mean)
        covariances.append(covariance)
        noises.append(noise)

    frequencies, coupling, frequency_std, coupling_std, noise_array = _assemble(
        means, covariances, noises, count
    )
    return DynamicalBayesianPosterior(
        frequencies=frequencies,
        coupling=coupling,
        frequency_std=frequency_std,
        coupling_std=coupling_std,
        noise=noise_array,
    )


def track_time_varying_coupling(
    phases: NDArray[np.float64],
    dt: float,
    window_size: int,
    *,
    prior_precision: float = 1e-6,
    propagation_inflation: float = 10.0,
    noise_iterations: int = 4,
) -> TimeVaryingCouplingHistory:
    r"""Track a time-varying directed coupling by Bayesian window propagation.

    Cuts the series into consecutive windows of ``window_size`` steps; each window's posterior becomes
    the next window's prior with its covariance multiplied by ``propagation_inflation`` (so the
    coupling is allowed to drift between windows), recovering a window-by-window reconstruction.

    Parameters
    ----------
    phases : numpy.ndarray
        The ``(n_samples, N)`` observed phase time series.
    dt : float
        The sampling interval ``> 0``.
    window_size : int
        The number of steps per window (``≥ 1``).
    prior_precision : float, optional
        The isotropic precision of the first window's prior; defaults to ``1e-6``.
    propagation_inflation : float, optional
        The factor by which each posterior covariance is inflated before becoming the next prior
        (``> 0``); defaults to ``10``.
    noise_iterations : int, optional
        The number of noise-estimation iterations per window; defaults to ``4``.

    Returns
    -------
    TimeVaryingCouplingHistory
        The per-window start times, frequencies, directed coupling and coupling standard deviations.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    series = np.ascontiguousarray(phases, dtype=np.float64)
    count = _validate(series, dt)
    if prior_precision <= 0.0:
        raise ValueError(f"prior_precision must be positive, got {prior_precision}")
    if propagation_inflation <= 0.0:
        raise ValueError(f"propagation_inflation must be positive, got {propagation_inflation}")
    if noise_iterations < 1:
        raise ValueError(f"noise_iterations must be positive, got {noise_iterations}")
    n_steps = series.shape[0] - 1
    if window_size < 1 or window_size > n_steps:
        raise ValueError(f"window_size must be in [1, {n_steps}], got {window_size}")

    n_windows = n_steps // window_size
    prior_means: list[NDArray[np.float64]] = [
        np.zeros(count, dtype=np.float64) for _ in range(count)
    ]
    prior_precisions: list[NDArray[np.float64]] = [
        prior_precision * np.eye(count, dtype=np.float64) for _ in range(count)
    ]

    window_times = []
    coupling_blocks = []
    coupling_std_blocks = []
    frequency_blocks = []
    for window in range(n_windows):
        start = window * window_size
        block = series[start : start + window_size + 1]
        means, covariances, noises = [], [], []
        for node in range(count):
            design = _node_design(block, node)
            increments = block[1:, node] - block[:-1, node]
            mean, covariance, noise = _node_posterior(
                design, increments, dt, prior_means[node], prior_precisions[node], noise_iterations
            )
            means.append(mean)
            covariances.append(covariance)
            noises.append(noise)
            prior_means[node] = mean
            prior_precisions[node] = np.asarray(
                np.linalg.inv(covariance * propagation_inflation), dtype=np.float64
            )
        frequencies, coupling, _, coupling_std, _ = _assemble(means, covariances, noises, count)
        window_times.append(start * dt)
        frequency_blocks.append(frequencies)
        coupling_blocks.append(coupling)
        coupling_std_blocks.append(coupling_std)

    return TimeVaryingCouplingHistory(
        window_times=np.array(window_times, dtype=np.float64),
        frequencies=np.array(frequency_blocks, dtype=np.float64),
        coupling=np.array(coupling_blocks, dtype=np.float64),
        coupling_std=np.array(coupling_std_blocks, dtype=np.float64),
    )


__all__ = [
    "DynamicalBayesianPosterior",
    "TimeVaryingCouplingHistory",
    "infer_network_bayesian",
    "track_time_varying_coupling",
]
