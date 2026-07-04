# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Stochastic (noisy) Kuramoto model, Euler–Maruyama integration
r"""Noisy Kuramoto model — additive white-noise dynamics by the Euler–Maruyama scheme.

Each oscillator obeys the Langevin equation ``dθ_j = (ω_j + F_j(θ)) dt + √(2D)\,dW_j`` with
independent Wiener increments and a common diffusion (noise intensity) ``D``: thermal phase
noise on top of any Kuramoto coupling force ``F``. The Euler–Maruyama discretisation advances
the phases by ``θ ← θ + (ω + F(θ)) dt + √(2 D dt)\,ξ`` with ``ξ`` standard normal, the
order-1/2 strong scheme for additive noise.

Noise raises the onset of synchronisation: for a Lorentzian frequency spread of half-width
``γ`` the critical coupling shifts from ``2γ`` to ``2(γ + D)`` — the noise adds to the
frequency width. That mean-field / Fokker–Planck onset and the stationary order parameter live
in :mod:`oscillatools.accel.kuramoto_noisy_mean_field`; this module is the matching
finite-population simulation. The run is reproducible: the Wiener increments are drawn from a
seeded :class:`numpy.random.Generator`, so the same seed gives the same trajectory.

This is an analysis layer over the synchronisation dynamics: the step composes any of the
polyglot Kuramoto forces (the mean-field, triadic or networked force re-exported from
:mod:`oscillatools.accel`) and reads the coherence with the accelerated
:func:`~oscillatools.accel.order_parameter_observables.order_parameter`, so the module
adds no compute kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .order_parameter_observables import order_parameter

#: A phase-coupling force ``F(θ)`` already closed over its coupling parameters, mapping the phase
#: vector to a force vector of the same length (e.g. ``lambda theta: mean_field_force(theta, K)``
#: or ``lambda theta: networked_kuramoto_force(theta, coupling_matrix)``).
StochasticForce = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class NoisyKuramotoRun:
    """The outcome of a seeded Euler–Maruyama run of the noisy Kuramoto model.

    Attributes
    ----------
    order_parameter_series : numpy.ndarray
        The Kuramoto order parameter ``r(t)`` sampled after every step (length ``n_steps``).
    terminal_phases : numpy.ndarray
        The phase vector at the final step.
    mean_order_parameter : float
        The order parameter averaged over the trailing settle window (the stationary estimate).
    order_parameter_std : float
        The standard deviation of the order parameter over the settle window (its fluctuation).
    diffusion : float
        The diffusion / noise intensity ``D`` used for the run.
    settle_steps : int
        The number of trailing steps averaged for ``mean_order_parameter``.
    """

    order_parameter_series: NDArray[np.float64]
    terminal_phases: NDArray[np.float64]
    mean_order_parameter: float
    order_parameter_std: float
    diffusion: float
    settle_steps: int


def noisy_kuramoto_step(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    force: StochasticForce,
    diffusion: float,
    dt: float,
    noise: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Advance the noisy Kuramoto phases by one Euler–Maruyama step.

    Returns ``θ + (ω + F(θ)) dt + √(2 D dt)\,ξ`` for the supplied standard-normal increment
    ``ξ = noise``. Taking the increment as an argument keeps the step a pure function (the
    randomness is injected by the caller), so it is deterministic and directly testable.

    Parameters
    ----------
    phases : numpy.ndarray
        The current phases ``θ`` (one-dimensional, length ``N``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    force : callable
        The phase-coupling force ``F(θ)`` (see :data:`StochasticForce`).
    diffusion : float
        The diffusion / noise intensity ``D`` (``≥ 0``).
    dt : float
        The time step (``> 0``).
    noise : numpy.ndarray
        A standard-normal Wiener increment ``ξ`` (length ``N``).

    Returns
    -------
    numpy.ndarray
        The phases after one step.

    Raises
    ------
    ValueError
        If the arrays are mismatched or ``diffusion``/``dt`` are out of range.
    """
    theta = np.ascontiguousarray(phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    increment = np.ascontiguousarray(noise, dtype=np.float64)
    _validate_step(theta, frequencies, increment, diffusion, dt)
    drift = (frequencies + force(theta)) * dt
    return np.asarray(theta + drift + np.sqrt(2.0 * diffusion * dt) * increment, dtype=np.float64)


def _validate_step(
    theta: NDArray[np.float64],
    omega: NDArray[np.float64],
    noise: NDArray[np.float64],
    diffusion: float,
    dt: float,
) -> None:
    """Validate the shared shape and range constraints of a noisy step / run."""
    if omega.ndim != 1 or omega.size < 1:
        raise ValueError("omega must be a non-empty one-dimensional array")
    if theta.shape != omega.shape:
        raise ValueError(f"phases must have shape {omega.shape}, got {theta.shape}")
    if noise.shape != omega.shape:
        raise ValueError(f"noise must have shape {omega.shape}, got {noise.shape}")
    if diffusion < 0.0:
        raise ValueError(f"diffusion must be non-negative, got {diffusion}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")


def integrate_noisy_kuramoto(
    initial_phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    force: StochasticForce,
    *,
    diffusion: float,
    dt: float,
    n_steps: int,
    seed: int,
    settle_steps: int | None = None,
) -> NoisyKuramotoRun:
    r"""Integrate the noisy Kuramoto model by a seeded Euler–Maruyama scheme.

    The phases are advanced by :func:`noisy_kuramoto_step` for ``n_steps`` steps, the Wiener
    increments drawn from a :class:`numpy.random.Generator` seeded with ``seed`` (so the run is
    reproducible). The order parameter is recorded after every step and averaged over the trailing
    ``settle_steps`` to estimate the stationary coherence and its fluctuation.

    Parameters
    ----------
    initial_phases : numpy.ndarray
        The seed phases ``θ(0)`` (one-dimensional, length ``N``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    force : callable
        The phase-coupling force ``F(θ)`` (see :data:`StochasticForce`).
    diffusion : float
        The diffusion / noise intensity ``D`` (``≥ 0``).
    dt : float
        The Euler–Maruyama time step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``).
    seed : int
        The seed of the noise generator; identical seeds reproduce the run.
    settle_steps : int, optional
        The trailing window averaged for the stationary order parameter; defaults to the final
        half of ``n_steps``. Must lie in ``[1, n_steps]``.

    Returns
    -------
    NoisyKuramotoRun
        The order-parameter series, terminal phases and the settle-window coherence statistics.

    Raises
    ------
    ValueError
        If any input is malformed (see the shape and range checks).
    """
    theta = np.ascontiguousarray(initial_phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    resolved_settle = settle_steps if settle_steps is not None else max(1, n_steps // 2)
    if not 1 <= resolved_settle <= n_steps:
        raise ValueError(f"settle_steps must be in [1, {n_steps}], got {resolved_settle}")
    # A zero increment validates the shapes / ranges once before the loop.
    _validate_step(theta, frequencies, np.zeros_like(frequencies), diffusion, dt)

    generator = np.random.default_rng(seed)
    scale = np.sqrt(2.0 * diffusion * dt)
    series = np.empty(n_steps, dtype=np.float64)
    for step in range(n_steps):
        increment = generator.standard_normal(theta.size)
        theta = theta + (frequencies + force(theta)) * dt + scale * increment
        series[step] = order_parameter(theta)
    settle = series[n_steps - resolved_settle :]
    return NoisyKuramotoRun(
        order_parameter_series=series,
        terminal_phases=theta,
        mean_order_parameter=float(settle.mean()),
        order_parameter_std=float(settle.std()),
        diffusion=diffusion,
        settle_steps=resolved_settle,
    )


__all__ = [
    "NoisyKuramotoRun",
    "StochasticForce",
    "integrate_noisy_kuramoto",
    "noisy_kuramoto_step",
]
