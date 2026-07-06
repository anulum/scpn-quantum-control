# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Polyglot time-delayed (method-of-steps) networked-Kuramoto trajectory
r"""Polyglot time-delayed (method-of-steps) networked-Kuramoto forward trajectory.

Integrates the delay-differential equation ``θ̇_j(t) = ω_j + \sum_k K_{jk}\sin(θ_k(t-τ) - θ_j(t))`` —
the time-delayed Kuramoto model with a finite coupling delay ``τ`` — for the *networked* coupling
matrix, by a delay-aware fixed-step RK4. The running phase grid doubles as the history buffer and
the delayed argument at a Runge–Kutta sub-stage time ``t + c·dt`` is read from the grid at position
``t + c·dt - τ`` (linearly interpolated for the ``c = 1/2`` stages), always an already-computed
sample because ``τ`` is an integer multiple of ``dt``.

This is the accelerated, networked-force special case of the general
:func:`~oscillatools.accel.kuramoto_delayed.integrate_delayed_kuramoto`: where that function
composes an arbitrary delayed Python force callable (and so runs on the pure-Python floor, keeping
gradients propagating through it), this function pins the force to the networked coupling matrix so
the whole method-of-steps hot loop can run in a compiled tier. The forward integration dispatches
across a Rust → Julia → Python floor tier chain (measured fastest first, recorded on
:func:`last_networked_delayed_trajectory_tier_used`). The three tiers share the same RK4 arithmetic
and delay interpolation and differ only in the coupling-force summation order — a scalar loop in the
compiled tiers versus NumPy's vectorised reduction in the floor — so they are *tolerance-parity*
(~1e-11), not bit-identical. It parallels the instantaneous-coupling accelerated integrators
(:func:`~oscillatools.accel.diff_kuramoto_rk4.kuramoto_rk4_trajectory` and
:func:`~oscillatools.accel.networked_inertial.networked_inertial_trajectory`), extending the polyglot
integrator family to the delay-differential model.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from . import dispatcher
from .dispatcher import MultiLangDispatcher, register_dispatcher
from .kuramoto_delayed import (
    DelayedTrajectory,
    delayed_networked_force,
    integrate_delayed_kuramoto,
)

_Forward = tuple[NDArray[np.float64], NDArray[np.float64]]


def _order_parameter_series(phases: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the Kuramoto order parameter ``r(t) = |⟨e^{iθ}⟩|`` at every sample row of ``phases``.

    This is the vectorised, row-wise form of the pure-Python floor order parameter, so the returned
    series is identical whichever tier produced ``phases``.
    """
    magnitudes = np.abs(np.mean(np.exp(1j * np.asarray(phases, dtype=np.float64)), axis=1))
    return np.asarray(magnitudes, dtype=np.float64)


def _validate_delayed_state(
    initial_history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    delay: float,
    dt: float,
    n_steps: int,
    delay_tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int, int]:
    """Return contiguous ``(history, omega, coupling)`` with ``(count, delay_steps)`` after checks.

    Mirrors the validation of :func:`~oscillatools.accel.kuramoto_delayed.integrate_delayed_kuramoto`
    (positive ``dt``/``delay``/``n_steps``, ``τ`` an integer multiple of ``dt``, matching
    ``initial_history`` history block) and adds the square-coupling check of the networked force.
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
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size < 1:
        raise ValueError("omega must be a non-empty one-dimensional array")
    count = frequencies.size
    if history.shape != (delay_steps + 1, count):
        raise ValueError(
            f"initial_history must have shape ({delay_steps + 1}, {count}), got {history.shape}"
        )
    if matrix.shape != (count, count):
        raise ValueError(f"coupling must be a square matrix of order {count}, got {matrix.shape}")
    return history, frequencies, matrix, count, delay_steps


def _rust_networked_delayed_trajectory(
    initial_history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    delay: float,
    dt: float,
    n_steps: int,
    delay_tolerance: float = 1e-9,
) -> _Forward:
    history, frequencies, matrix, _count, _delay_steps = _validate_delayed_state(
        initial_history, omega, coupling, delay, dt, n_steps, delay_tolerance
    )
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_forward = getattr(engine, "kuramoto_delayed_trajectory", None)
    if not callable(rust_forward):
        raise ImportError("scpn_quantum_engine.kuramoto_delayed_trajectory is unavailable")
    times, path = rust_forward(history, frequencies, matrix, float(dt), int(n_steps))
    return (
        np.ascontiguousarray(times, dtype=np.float64),
        np.ascontiguousarray(path, dtype=np.float64),
    )


def _julia_networked_delayed_trajectory(
    initial_history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    delay: float,
    dt: float,
    n_steps: int,
    delay_tolerance: float = 1e-9,
) -> _Forward:
    history, frequencies, matrix, _count, _delay_steps = _validate_delayed_state(
        initial_history, omega, coupling, delay, dt, n_steps, delay_tolerance
    )
    from .julia import kuramoto_delayed_trajectory as julia_forward

    times, path = julia_forward(history, frequencies, matrix, float(dt), int(n_steps))
    return (
        np.ascontiguousarray(times, dtype=np.float64),
        np.ascontiguousarray(path, dtype=np.float64),
    )


def _python_networked_delayed_trajectory(
    initial_history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    delay: float,
    dt: float,
    n_steps: int,
    delay_tolerance: float = 1e-9,
) -> _Forward:
    """Pure-Python floor: the general method-of-steps integrator bound to the networked force."""
    history, frequencies, matrix, _count, _delay_steps = _validate_delayed_state(
        initial_history, omega, coupling, delay, dt, n_steps, delay_tolerance
    )
    trajectory = integrate_delayed_kuramoto(
        history,
        frequencies,
        lambda current, lagged: delayed_networked_force(current, lagged, matrix),
        delay=delay,
        dt=dt,
        n_steps=n_steps,
        delay_tolerance=delay_tolerance,
    )
    return (
        np.ascontiguousarray(trajectory.times, dtype=np.float64),
        np.ascontiguousarray(trajectory.phases, dtype=np.float64),
    )


_NETWORKED_DELAYED_TRAJECTORY_CHAIN: list[tuple[str, Callable[..., _Forward]]] = [
    ("rust", _rust_networked_delayed_trajectory),
    ("julia", _julia_networked_delayed_trajectory),
    ("python", _python_networked_delayed_trajectory),
]
_networked_delayed_trajectory_dispatcher = MultiLangDispatcher(_NETWORKED_DELAYED_TRAJECTORY_CHAIN)


def networked_delayed_trajectory(
    initial_history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    delay: float,
    dt: float,
    n_steps: int,
    delay_tolerance: float = 1e-9,
) -> DelayedTrajectory:
    r"""Integrate the time-delayed networked Kuramoto model by a method-of-steps RK4.

    Advances the delay-differential equation ``θ̇(t) = ω + F(θ(t), θ(t-τ))`` with the networked
    delayed coupling force :math:`F_j = \sum_k K_{jk}\sin(\theta_k(t-\tau) - \theta_j(t))`, sampling
    ``θ`` at every step and reading the delayed argument from the running phase grid (linearly
    interpolated at the half-step Runge–Kutta stages). This is the accelerated, networked-force
    special case of :func:`~oscillatools.accel.kuramoto_delayed.integrate_delayed_kuramoto`; the
    forward integration dispatches across a Rust → Julia → Python floor tier chain (measured fastest
    first, recorded on :func:`last_networked_delayed_trajectory_tier_used`). The tiers are
    tolerance-parity: they share the RK4 arithmetic and delay interpolation and differ only in the
    coupling-force summation order, agreeing to ~1e-11.

    Parameters
    ----------
    initial_history : numpy.ndarray
        The phase history on ``[-τ, 0]`` as a ``(delay_steps + 1, N)`` array (``delay_steps = τ/dt``);
        row ``s`` is time ``-(delay_steps - s)·dt`` and the last row is ``θ(0)``.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies ``ω``.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.
    delay : float
        The coupling delay ``τ`` (``> 0``); must be an integer multiple of ``dt``.
    dt : float
        The fixed RK4 time step (``> 0``).
    n_steps : int
        The number of RK4 steps (``≥ 1``); the trajectory has ``n_steps + 1`` samples.
    delay_tolerance : float, optional
        The absolute tolerance for ``τ`` being an integer multiple of ``dt``; defaults to ``1e-9``.

    Returns
    -------
    DelayedTrajectory
        The sampled phase trajectory, the order-parameter series and the delay used.

    Raises
    ------
    ValueError
        If ``delay``/``dt``/``n_steps`` are out of range, ``τ`` is not an integer multiple of ``dt``,
        or ``initial_history``/``omega``/``coupling`` are malformed.
    """
    _history, _frequencies, _matrix, _count, delay_steps = _validate_delayed_state(
        initial_history, omega, coupling, delay, dt, n_steps, delay_tolerance
    )
    times, phases = _networked_delayed_trajectory_dispatcher(
        initial_history, omega, coupling, delay, dt, n_steps, delay_tolerance
    )
    series = _order_parameter_series(phases)
    return DelayedTrajectory(times, phases, series, float(delay), delay_steps)


def last_networked_delayed_trajectory_tier_used() -> str | None:
    """The tier (``'rust'``/``'julia'``/``'python'``) that served the last dispatched forward.

    ``None`` before the first dispatched call.
    """
    return _networked_delayed_trajectory_dispatcher.last_tier


register_dispatcher("networked_delayed_trajectory", _networked_delayed_trajectory_dispatcher)


__all__ = [
    "last_networked_delayed_trajectory_tier_used",
    "networked_delayed_trajectory",
]
