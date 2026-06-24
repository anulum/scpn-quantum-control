# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Time-delayed Kuramoto dynamics, method-of-steps delay-aware integration
r"""Time-delayed Kuramoto model — finite propagation delay in the coupling.

When the interaction between oscillators is not instantaneous — a signal takes a time ``τ`` to
travel — each oscillator feels the *delayed* phases of its neighbours while its own response is
instantaneous:

.. math::

    \dot{θ}_j(t) = ω_j + \sum_k K_{jk}\,\sin\!\bigl(θ_k(t-τ) - θ_j(t)\bigr).

The delay turns the system into a delay-differential equation: the state is no longer a point
but a *history* — the phases on the whole interval ``[-τ, 0]``. This module integrates it by the
**method of steps** with a delay-aware fixed-step RK4: the running phase grid doubles as the
history buffer, and the delayed term at any RK sub-stage time ``t + c\,dt`` is read from the
buffer at grid position ``t + c\,dt - τ`` (linearly interpolated for the half-step stages),
which is always an already-computed sample when ``τ`` is an integer multiple of ``dt``.

The delay reshapes synchronisation: for the mean-field coupling of *identical* oscillators the
collective frequency ``Ω`` of the phase-locked state no longer equals the natural frequency but
solves the Yeung–Strogatz self-consistency ``Ω = ω₀ - K sin(Ω τ)``, which at large ``K τ`` has
several stable roots — coexisting synchronised branches at different frequencies
(delay-induced multistability). That self-consistency and its stable branches live in
:mod:`scpn_quantum_control.accel.kuramoto_delayed_mean_field`; this module is the matching
simulation, and a run started from different constant initial-history frequencies settles onto
different branches.

This is an analysis layer over the synchronisation dynamics: the delayed forces are the
phase-difference couplings of the standard Kuramoto interaction (mean-field and networked)
evaluated across the *two* phase vectors ``θ(t)`` and ``θ(t-τ)``, and the coherence is read with
the accelerated :func:`~scpn_quantum_control.accel.order_parameter_observables.order_parameter`,
so the module adds no compute kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .order_parameter_observables import order_parameter

#: A delayed phase-coupling force ``F(θ(t), θ(t-τ))`` already closed over its coupling
#: parameters. It maps the current phases ``θ(t)`` and the delayed phases ``θ(t-τ)`` (both length
#: ``N``) to a force vector of length ``N``. Bind a delayed force to its coupling to obtain one,
#: e.g. ``lambda cur, lag: delayed_mean_field_force(cur, lag, K)`` or
#: ``lambda cur, lag: delayed_networked_force(cur, lag, coupling_matrix)``.
DelayedForce = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


def delayed_mean_field_force(
    current_phases: NDArray[np.float64],
    delayed_phases: NDArray[np.float64],
    coupling: float,
) -> NDArray[np.float64]:
    r"""Return the delayed mean-field Kuramoto force ``F_j = (K/N) Σ_k sin(θ_k(t-τ) − θ_j(t))``.

    The all-to-all coupling reads the delayed phases ``θ(t-τ)`` for every neighbour while the
    self-phase ``θ_j(t)`` is instantaneous, so it is the standard mean-field force evaluated
    across the two phase vectors.

    Parameters
    ----------
    current_phases : numpy.ndarray
        The current phases ``θ(t)`` (one-dimensional, length ``N``).
    delayed_phases : numpy.ndarray
        The delayed phases ``θ(t-τ)`` (length ``N``).
    coupling : float
        The global coupling strength ``K``.

    Returns
    -------
    numpy.ndarray
        The force on each oscillator (length ``N``).

    Raises
    ------
    ValueError
        If the two phase vectors are not matching non-empty 1-D arrays.
    """
    current = np.ascontiguousarray(current_phases, dtype=np.float64)
    delayed = np.ascontiguousarray(delayed_phases, dtype=np.float64)
    _validate_phase_pair(current, delayed)
    differences = delayed[np.newaxis, :] - current[:, np.newaxis]
    return np.asarray(coupling * np.sin(differences).mean(axis=1), dtype=np.float64)


def delayed_networked_force(
    current_phases: NDArray[np.float64],
    delayed_phases: NDArray[np.float64],
    coupling_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Return the delayed networked Kuramoto force ``F_j = Σ_k K_{jk} sin(θ_k(t-τ) − θ_j(t))``.

    The weighted sum over neighbours reads the delayed phases ``θ(t-τ)`` while the self-phase
    ``θ_j(t)`` is instantaneous, generalising the networked Kuramoto force to a finite coupling
    delay.

    Parameters
    ----------
    current_phases : numpy.ndarray
        The current phases ``θ(t)`` (one-dimensional, length ``N``).
    delayed_phases : numpy.ndarray
        The delayed phases ``θ(t-τ)`` (length ``N``).
    coupling_matrix : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K_{jk}`` (row ``j`` weights the influence of ``k``).

    Returns
    -------
    numpy.ndarray
        The force on each oscillator (length ``N``).

    Raises
    ------
    ValueError
        If the phase vectors mismatch or the coupling matrix is not ``(N, N)``.
    """
    current = np.ascontiguousarray(current_phases, dtype=np.float64)
    delayed = np.ascontiguousarray(delayed_phases, dtype=np.float64)
    _validate_phase_pair(current, delayed)
    coupling = np.ascontiguousarray(coupling_matrix, dtype=np.float64)
    count = current.size
    if coupling.shape != (count, count):
        raise ValueError(
            f"coupling_matrix must have shape ({count}, {count}), got {coupling.shape}"
        )
    differences = delayed[np.newaxis, :] - current[:, np.newaxis]
    return np.asarray((coupling * np.sin(differences)).sum(axis=1), dtype=np.float64)


def _validate_phase_pair(current: NDArray[np.float64], delayed: NDArray[np.float64]) -> None:
    """Validate that ``current`` and ``delayed`` are matching non-empty 1-D phase vectors."""
    if current.ndim != 1 or current.size < 1:
        raise ValueError("current_phases must be a non-empty one-dimensional array")
    if delayed.shape != current.shape:
        raise ValueError(f"delayed_phases must have shape {current.shape}, got {delayed.shape}")


@dataclass(frozen=True)
class DelayedTrajectory:
    """A trajectory of the time-delayed Kuramoto model sampled at every integration step.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(n_steps + 1,)`` sample times ``0, dt, …, n_steps·dt`` of the integrated span
        (the initial history on ``[-τ, 0]`` is not included).
    phases : numpy.ndarray
        The ``(n_steps + 1, N)`` phase trajectory ``θ(t)``, left unwrapped so the collective
        frequency is recoverable from the phase increment.
    order_parameter_series : numpy.ndarray
        The Kuramoto order parameter ``r(t)`` at every sample (length ``n_steps + 1``).
    delay : float
        The coupling delay ``τ`` used to generate the trajectory.
    delay_steps : int
        The number of grid steps in one delay, ``τ / dt``.
    """

    times: NDArray[np.float64]
    phases: NDArray[np.float64]
    order_parameter_series: NDArray[np.float64]
    delay: float
    delay_steps: int

    @property
    def terminal_phases(self) -> NDArray[np.float64]:
        """The final phase vector of the trajectory."""
        return np.asarray(self.phases[-1], dtype=np.float64)

    def collective_frequency(self, *, fraction: float = 0.25) -> float:
        r"""Estimate the collective frequency ``Ω`` from the trailing ``fraction`` of the run.

        The mean phase ``(1/N) Σ_j θ_j(t)`` of the (unwrapped) trajectory advances linearly once
        the run has settled onto a synchronised branch; ``Ω`` is its slope, recovered by a
        least-squares line fit over the trailing window. For a fully phase-locked state this is
        the branch frequency that solves the Yeung–Strogatz self-consistency.

        Parameters
        ----------
        fraction : float, optional
            The trailing fraction of the trajectory used for the fit, in ``(0, 1]``; defaults to
            the final quarter.

        Returns
        -------
        float
            The estimated collective frequency ``Ω``.

        Raises
        ------
        ValueError
            If ``fraction`` is outside ``(0, 1]`` or the window has fewer than two samples.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        total = self.times.size
        window = max(2, int(round(fraction * total)))
        if window > total:
            window = total
        if window < 2:
            raise ValueError("trajectory too short to estimate a collective frequency")
        tail_times = self.times[total - window :]
        mean_phase = self.phases[total - window :].mean(axis=1)
        slope = np.polyfit(tail_times, mean_phase, 1)[0]
        return float(slope)


def integrate_delayed_kuramoto(
    initial_history: NDArray[np.float64],
    omega: NDArray[np.float64],
    force: DelayedForce,
    *,
    delay: float,
    dt: float,
    n_steps: int,
    delay_tolerance: float = 1e-9,
) -> DelayedTrajectory:
    r"""Integrate the time-delayed Kuramoto model by a delay-aware method-of-steps RK4.

    The phases ``θ`` obey the delay-differential equation ``θ̇(t) = ω + F(θ(t), θ(t-τ))``. The
    ``initial_history`` supplies ``θ`` on the interval ``[-τ, 0]`` sampled on the integration
    grid (``delay_steps + 1`` rows, the last row being ``θ(0)``). Each step advances the state by
    the classical four-stage Runge–Kutta rule; the delayed argument at a sub-stage time
    ``t + c\,dt`` is read from the running phase grid at position ``t + c\,dt - τ`` (linearly
    interpolated for the ``c = 1/2`` stages), which is always an earlier, already-computed sample
    because ``τ`` is an integer number of steps. The phases are left unwrapped so the collective
    frequency is recoverable from the trajectory.

    Parameters
    ----------
    initial_history : numpy.ndarray
        The phase history on ``[-τ, 0]`` as a ``(delay_steps + 1, N)`` array; row ``s``
        corresponds to time ``-(delay_steps - s)·dt`` and the last row is ``θ(0)``.
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    force : callable
        The delayed phase-coupling force ``F(θ(t), θ(t-τ))`` (see :data:`DelayedForce`).
    delay : float
        The coupling delay ``τ`` (``> 0``); must be an integer multiple of ``dt``.
    dt : float
        The integration time step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``); the trajectory has ``n_steps + 1`` samples.
    delay_tolerance : float, optional
        The absolute tolerance for ``τ`` being an integer multiple of ``dt``; defaults to ``1e-9``.

    Returns
    -------
    DelayedTrajectory
        The sampled phase trajectory, the order-parameter series and the delay used.

    Raises
    ------
    ValueError
        If ``delay``/``dt``/``n_steps`` are out of range, ``τ`` is not an integer multiple of
        ``dt``, or ``initial_history``/``omega`` are malformed.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if delay <= 0.0:
        raise ValueError(f"delay must be positive, got {delay}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    delay_steps = int(round(delay / dt))
    if delay_steps < 1 or abs(delay - delay_steps * dt) > delay_tolerance:
        raise ValueError(
            f"delay must be a positive integer multiple of dt, got delay={delay}, dt={dt}"
        )
    history = np.ascontiguousarray(initial_history, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size < 1:
        raise ValueError("omega must be a non-empty one-dimensional array")
    count = frequencies.size
    if history.shape != (delay_steps + 1, count):
        raise ValueError(
            f"initial_history must have shape ({delay_steps + 1}, {count}), got {history.shape}"
        )

    # The running phase grid: grid index g holds θ at time (g - delay_steps)·dt, so the initial
    # history fills g = 0 … delay_steps (g = delay_steps is θ(0)) and integration appends g > delay_steps.
    buffer: list[NDArray[np.float64]] = [np.array(row, dtype=np.float64) for row in history]

    def delayed_at(position: float) -> NDArray[np.float64]:
        """The phase sample at fractional grid index ``position`` (linear interpolation)."""
        lower = int(np.floor(position))
        weight = position - lower
        if weight == 0.0:
            return buffer[lower]
        return (1.0 - weight) * buffer[lower] + weight * buffer[lower + 1]

    def rhs(current: NDArray[np.float64], lagged: NDArray[np.float64]) -> NDArray[np.float64]:
        return frequencies + force(current, lagged)

    phase_history = np.empty((n_steps + 1, count), dtype=np.float64)
    series = np.empty(n_steps + 1, dtype=np.float64)
    phase_history[0] = buffer[delay_steps]
    series[0] = order_parameter(buffer[delay_steps])
    for step in range(n_steps):
        theta = buffer[delay_steps + step]
        # Delayed grid position for a sub-stage at t_n + c·dt is (step + c); see the module note.
        k1 = rhs(theta, delayed_at(step))
        k2 = rhs(theta + 0.5 * dt * k1, delayed_at(step + 0.5))
        k3 = rhs(theta + 0.5 * dt * k2, delayed_at(step + 0.5))
        k4 = rhs(theta + dt * k3, delayed_at(step + 1.0))
        nxt = theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        buffer.append(nxt)
        phase_history[step + 1] = nxt
        series[step + 1] = order_parameter(nxt)
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    return DelayedTrajectory(times, phase_history, series, delay, delay_steps)


__all__ = [
    "DelayedForce",
    "DelayedTrajectory",
    "delayed_mean_field_force",
    "delayed_networked_force",
    "integrate_delayed_kuramoto",
]
