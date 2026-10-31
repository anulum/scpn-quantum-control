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

Identifiability is reported, not patched. A direction in parameter space whose data precision
``(h/E) vᵀ(Σ p_n p_nᵀ)v`` is at most :data:`IDENTIFIABILITY_PRIOR_RATIO` times the prior
precision keeps (essentially) its prior; such directions are returned in
:attr:`DynamicalBayesianPosterior.unidentified_directions` and set ``identifiable`` to ``False``
(for example, two oscillators with identical phases give two identical design columns, so only
the sum of their couplings is learned). A series that is noise-free to rounding
(:data:`NOISE_FREE_RATIO`) is refused, since the transition likelihood is then degenerate.

Held-out evaluation uses contiguous, disjoint segments: :func:`fit_with_held_out` fits on the
first segment only, then scores one-step predictive intervals on a calibration and a test segment
that the fit never reads (:func:`evaluate_held_out`).

This complements the coupling-*function* inference (which learns the interaction *shape* ``Γ``) and
the trajectory-match coupling-*matrix* system identification (which is deterministic and gives no
uncertainty): here the data are stochastic and the output is a full posterior over the directed
network. It adds no compute kernel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

IDENTIFIABILITY_PRIOR_RATIO = 0.1
"""A direction is unidentified when its data precision is at most this times the prior precision.

Then its posterior variance stays within ``1 / (1 + ratio)`` (here 91 %) of the prior variance:
the posterior along it is essentially the prior. The decision compares data with prior, so it does
not change when a design column is rescaled.
"""
NOISE_FREE_RATIO = 1e-10
"""A residual RMS at most this times the increment RMS is a noise-free series, which is refused."""


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
    parameter_covariances : tuple of numpy.ndarray
        Per node, the posterior covariance of ``[ω_i, K_{ij} (j ≠ i, ascending)]``.
    data_rank : numpy.ndarray
        Per node, the number of parameter directions the data inform.
    unidentified_directions : tuple of numpy.ndarray
        Per node, an ``(N, d_i)`` orthonormal basis (in the same parameter order) of the directions
        the data do not inform; empty when ``d_i = 0``. Along them the posterior is the prior.
    identifiable : bool
        Whether every node's parameters are informed by the data.

    """

    frequencies: NDArray[np.float64]
    coupling: NDArray[np.float64]
    frequency_std: NDArray[np.float64]
    coupling_std: NDArray[np.float64]
    noise: NDArray[np.float64]
    parameter_covariances: tuple[NDArray[np.float64], ...] = ()
    data_rank: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    unidentified_directions: tuple[NDArray[np.float64], ...] = ()
    identifiable: bool = True


@dataclass(frozen=True)
class HeldOutEvaluation:
    """One-step predictive scores of a posterior on data the fit did not read.

    Attributes
    ----------
    samples : int
        Number of scored increments (all nodes).
    coverage_95 : float
        Fraction of increments inside their central 95 % predictive interval.
    standardized_rms : float
        Root mean square of the standardised residuals; about one when the intervals are calibrated.
    mean_log_predictive_density : float
        Mean Gaussian log predictive density per increment.

    """

    samples: int
    coverage_95: float
    standardized_rms: float
    mean_log_predictive_density: float


@dataclass(frozen=True)
class InverseProblemSplit:
    """A fit on a training segment scored on later calibration and test segments.

    Attributes
    ----------
    boundaries : tuple of int
        Sample indices ``(0, a, b, n)``: training ``[0, a]``, calibration ``[a, b]``, test ``[b, n]``.
        Consecutive segments share only their boundary sample; their increments are disjoint.
    posterior : DynamicalBayesianPosterior
        The posterior from the training segment alone.
    calibration : HeldOutEvaluation
        Scores on the calibration segment.
    test : HeldOutEvaluation
        Scores on the test segment.

    """

    boundaries: tuple[int, int, int, int]
    posterior: DynamicalBayesianPosterior
    calibration: HeldOutEvaluation
    test: HeldOutEvaluation


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
    # Minimum-norm least squares for the first noise estimate: no ridge hides a rank deficiency.
    mean = np.linalg.lstsq(design * dt, increments, rcond=None)[0]
    precision = prior_precision
    # Residuals at rounding level relative to the increments mean the series is noise-free.
    floor = (NOISE_FREE_RATIO**2) * float(np.mean(increments**2)) / dt
    for _ in range(noise_iterations):
        residual = increments - dt * (design @ mean)
        noise = float(np.mean(residual**2) / dt)
        if not noise > floor or not math.isfinite(noise):
            raise ValueError(
                "the residual noise is zero to rounding or not finite; the Euler-Maruyama "
                "likelihood is degenerate and no posterior is issued"
            )
        precision = prior_precision + (dt / noise) * gram
        mean = np.linalg.solve(precision, prior_term + cross / noise)
    covariance = np.linalg.inv(precision)
    return (
        np.asarray(mean, dtype=np.float64),
        np.asarray(covariance, dtype=np.float64),
        noise,
    )


def _unidentified(
    design: NDArray[np.float64], dt: float, noise: float, prior_precision: float
) -> NDArray[np.float64]:
    """Return an orthonormal basis of the directions whose posterior is essentially the prior.

    The data precision ``(dt / noise) Pᵀ P`` is compared with the isotropic prior precision along
    each of its eigenvectors.
    """
    values, vectors = np.linalg.eigh((dt / noise) * (design.T @ design))
    mask = values <= IDENTIFIABILITY_PRIOR_RATIO * prior_precision
    return np.asarray(vectors[:, mask], dtype=np.float64)


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
    means, covariances, noises, unidentified = [], [], [], []
    for node in range(count):
        design = _node_design(series, node)
        increments = series[1:, node] - series[:-1, node]
        mean, covariance, noise = _node_posterior(
            design, increments, dt, prior_mean, prior_matrix, noise_iterations
        )
        means.append(mean)
        covariances.append(covariance)
        noises.append(noise)
        unidentified.append(_unidentified(design, dt, noise, prior_precision))

    frequencies, coupling, frequency_std, coupling_std, noise_array = _assemble(
        means, covariances, noises, count
    )
    return DynamicalBayesianPosterior(
        frequencies=frequencies,
        coupling=coupling,
        frequency_std=frequency_std,
        coupling_std=coupling_std,
        noise=noise_array,
        parameter_covariances=tuple(covariances),
        data_rank=np.array([count - basis.shape[1] for basis in unidentified], dtype=np.int64),
        unidentified_directions=tuple(unidentified),
        identifiable=all(basis.shape[1] == 0 for basis in unidentified),
    )


def evaluate_held_out(
    posterior: DynamicalBayesianPosterior,
    phases: NDArray[np.float64],
    dt: float,
) -> HeldOutEvaluation:
    r"""Score one-step predictive intervals of a posterior on a held-out phase series.

    Each increment ``φ_{n+1,i} - φ_{n,i}`` is predicted as ``h\,p_nᵀ \bar c_i`` with variance
    ``E_i h + h² p_nᵀ Σ_i p_n`` (noise plus parameter uncertainty), from the posterior alone.

    Parameters
    ----------
    posterior : DynamicalBayesianPosterior
        A posterior from :func:`infer_network_bayesian` (it must carry ``parameter_covariances``).
    phases : numpy.ndarray
        The ``(n_samples, N)`` held-out series, sampled at ``dt``.
    dt : float
        The sampling interval ``> 0``.

    Returns
    -------
    HeldOutEvaluation
        Coverage of the 95 % intervals, the standardised RMS and the mean log predictive density.

    Raises
    ------
    ValueError
        If the series does not match the posterior or the posterior has no covariances.

    """
    series = np.ascontiguousarray(phases, dtype=np.float64)
    count = _validate(series, dt)
    if count != posterior.frequencies.size or len(posterior.parameter_covariances) != count:
        raise ValueError("the held-out series and the posterior must describe the same N nodes")
    standardized: list[NDArray[np.float64]] = []
    log_density = 0.0
    for node in range(count):
        design = _node_design(series, node)
        others = [other for other in range(count) if other != node]
        mean = np.concatenate(([posterior.frequencies[node]], posterior.coupling[node, others]))
        covariance = posterior.parameter_covariances[node]
        increments = series[1:, node] - series[:-1, node]
        variance = posterior.noise[node] * dt + dt**2 * np.einsum(
            "ni,ij,nj->n", design, covariance, design
        )
        residual = increments - dt * (design @ mean)
        standardized.append(residual / np.sqrt(variance))
        log_density += float(
            np.sum(-0.5 * (np.log(2.0 * np.pi * variance) + residual**2 / variance))
        )
    scores = np.concatenate(standardized)
    return HeldOutEvaluation(
        samples=int(scores.size),
        coverage_95=float(np.mean(np.abs(scores) <= 1.959963984540054)),
        standardized_rms=float(np.sqrt(np.mean(scores**2))),
        mean_log_predictive_density=log_density / scores.size,
    )


def fit_with_held_out(
    phases: NDArray[np.float64],
    dt: float,
    *,
    fractions: Sequence[float] = (0.6, 0.2, 0.2),
    prior_precision: float = 1e-6,
    noise_iterations: int = 4,
) -> InverseProblemSplit:
    """Fit on the first segment only and score the posterior on later, disjoint segments.

    Parameters
    ----------
    phases : numpy.ndarray
        The ``(n_samples, N)`` phase series, in time order.
    dt : float
        The sampling interval ``> 0``.
    fractions : sequence of float, optional
        Positive training, calibration and test fractions of the increments, summing to one.
    prior_precision, noise_iterations
        As for :func:`infer_network_bayesian`, applied to the training segment.

    Returns
    -------
    InverseProblemSplit
        The segment boundaries, the training posterior and the calibration and test scores.

    Raises
    ------
    ValueError
        If the fractions are invalid or a segment would have fewer than two samples.

    """
    series = np.ascontiguousarray(phases, dtype=np.float64)
    _validate(series, dt)
    if len(fractions) != 3 or any(not part > 0.0 for part in fractions):
        raise ValueError("fractions must be three positive numbers")
    if abs(sum(fractions) - 1.0) > 1e-12:
        raise ValueError("fractions must sum to one")
    steps = series.shape[0] - 1
    first = round(steps * fractions[0])
    second = first + round(steps * fractions[1])
    boundaries = (0, first, second, steps)
    if min(first, second - first, steps - second) < 1:
        raise ValueError("every segment needs at least one increment")
    posterior = infer_network_bayesian(
        series[: first + 1], dt, prior_precision=prior_precision, noise_iterations=noise_iterations
    )
    return InverseProblemSplit(
        boundaries=boundaries,
        posterior=posterior,
        calibration=evaluate_held_out(posterior, series[first : second + 1], dt),
        test=evaluate_held_out(posterior, series[second:], dt),
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
    "IDENTIFIABILITY_PRIOR_RATIO",
    "NOISE_FREE_RATIO",
    "DynamicalBayesianPosterior",
    "HeldOutEvaluation",
    "InverseProblemSplit",
    "TimeVaryingCouplingHistory",
    "evaluate_held_out",
    "fit_with_held_out",
    "infer_network_bayesian",
    "track_time_varying_coupling",
]
