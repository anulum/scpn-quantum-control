# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Sensitivity of the time-delayed Kuramoto terminal state to the coupling delay
r"""Sensitivity of the time-delayed Kuramoto terminal state to the coupling delay ``τ``.

The differentiable delayed integrator in
:mod:`~oscillatools.accel.diff_kuramoto_delayed` gives the gradient of a terminal objective
with respect to the initial history, the frequencies and the coupling — but *not* with respect to the
delay ``τ`` itself, which enters only as an integer number of grid steps. The derivative
``∂θ_N/∂τ`` is the genuinely hard direction: a continuous-adjoint delay sensitivity is unavailable
even in the established sensitivity toolchains (it is unsupported in SciMLSensitivity.jl), because the
delay changes the *structure* of a method-of-steps scheme, not just its inputs.

This module provides it in closed, exact forward-mode. The delay is treated as a continuous parameter
by reading the lagged state ``θ(t − τ)`` from a **linear interpolation** of the running trajectory
(the same interpolation the production method-of-steps already uses at its half-steps), so ``θ_N(τ)``
is a well-defined, differentiable function of ``τ`` away from the grid nodes. A tangent buffer
``σ(t) = ∂θ(t)/∂τ`` is co-integrated with the phases by the identical RK4 stages; the lagged term
contributes its total ``τ``-derivative

``d/dτ[θ_k(t − τ)] = σ_k(t − τ) − (θ_k(t − τ))'``,

where the second term is the **discrete slope** ``(θ_k[i+1] − θ_k[i])/dt`` of the bracketing grid
interval — the exact derivative of the linear interpolant, so ``σ_N`` differentiates the discrete map
and matches a central finite difference of ``θ_N(τ)`` to machine precision. On a constant history
(``θ(s) = θ_0`` for ``s ≤ 0``) both terms vanish while the lag is negative, so the sensitivity is
seeded at zero and accumulates entirely from the interior dynamics.

Where the derivative does not exist — when a Runge–Kutta stage lag lands exactly on a grid node, i.e.
when ``τ`` is a half-integer multiple of ``dt`` — the interpolant has a kink and only one-sided
derivatives exist; the functions below reject such ``τ`` rather than return a one-sided value as if it
were the derivative. The forward map itself is faithful: at an *integer* multiple of ``dt`` it
reproduces :func:`~oscillatools.accel.kuramoto_delayed.integrate_delayed_kuramoto` on a
constant history exactly.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .diff_kuramoto_delayed import DelayedTerminalObjective, DelayedTerminalObjectiveGrad

#: Absolute tolerance for detecting that ``τ`` is a (half-)integer multiple of ``dt``.
_NODE_TOLERANCE = 1e-9


def _validate(
    history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    delay: float,
    dt: float,
    n_steps: int,
) -> int:
    r"""Validate the delay-sensitivity inputs and return the oscillator count ``N``.

    Raises
    ------
    ValueError
        If the history/frequency/coupling shapes are inconsistent, ``dt`` or ``delay`` is not
        positive, ``delay`` is smaller than ``dt`` (the method-of-steps lag would reference the
        current step), ``n_steps`` is not positive, or ``delay`` is a half-integer multiple of ``dt``
        (a grid node, where ``∂θ_N/∂τ`` does not exist).
    """
    if history.ndim != 1:
        raise ValueError(
            f"history must be a one-dimensional constant history, got shape {history.shape}"
        )
    count = int(history.size)
    if count == 0:
        raise ValueError("history must contain at least one oscillator")
    if omega.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {omega.shape}")
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {coupling.shape}")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if not math.isfinite(delay) or delay <= 0.0:
        raise ValueError(f"delay must be finite and positive, got {delay}")
    if delay < dt:
        raise ValueError(
            f"delay must be at least dt for the method-of-steps, got delay={delay}, dt={dt}"
        )
    if n_steps < 1:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps}")
    half_steps = 2.0 * delay / dt
    if abs(half_steps - round(half_steps)) < _NODE_TOLERANCE:
        raise ValueError(
            "delay must not be a half-integer multiple of dt: the method-of-steps interpolant is "
            f"non-differentiable in τ at grid nodes, got delay={delay}, dt={dt} (2·delay/dt={half_steps})"
        )
    return count


def _force(
    current: NDArray[np.float64],
    lagged: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Delayed networked Kuramoto field ``θ̇_j = ω_j + Σ_k K_jk sin(θ_k(t−τ) − θ_j(t))``."""
    difference = lagged[np.newaxis, :] - current[:, np.newaxis]
    return np.asarray(omega + (coupling * np.sin(difference)).sum(axis=1), dtype=np.float64)


def _tangent_field(
    current: NDArray[np.float64],
    lagged: NDArray[np.float64],
    current_tangent: NDArray[np.float64],
    lagged_total_derivative: NDArray[np.float64],
    coupling: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Tangent of :func:`_force` in the direction ``τ``.

    ``σ̇_j = Σ_k K_jk cos(θ_k(t−τ) − θ_j(t)) · (D_k − σ_j)`` where ``D_k`` is the total ``τ``-derivative
    of the lagged phase ``θ_k(t−τ)`` and ``σ_j = ∂θ_j(t)/∂τ`` is the current tangent.
    """
    difference = lagged[np.newaxis, :] - current[:, np.newaxis]
    weighted_cos = coupling * np.cos(difference)
    return np.asarray(
        (
            weighted_cos
            * (lagged_total_derivative[np.newaxis, :] - current_tangent[:, np.newaxis])
        ).sum(axis=1),
        dtype=np.float64,
    )


def _lag(
    phases: NDArray[np.float64],
    tangents: NDArray[np.float64],
    history: NDArray[np.float64],
    dt: float,
    lag_time: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Interpolate the lagged ``(phase, total τ-derivative, tangent)`` at continuous time ``lag_time``.

    The phase and tangent are linear interpolants of the stored grids; the total ``τ``-derivative of
    the lagged phase is ``σ_lag − slope``, with ``slope`` the discrete grid slope of the bracketing
    interval (the exact derivative of the linear interpolant). A negative ``lag_time`` reads the
    constant history, whose phase is fixed and whose derivatives are zero.

    The caller guarantees ``delay > dt`` (``delay = dt`` is a rejected grid node), so every
    non-negative stage lag brackets two already-stored grid rows and ``phases[index + 1]`` is always
    a written row.
    """
    if lag_time < 0.0:
        zero = np.zeros_like(history)
        return history, zero.copy(), zero
    position = lag_time / dt
    index = int(math.floor(position + 1e-12))
    fraction = position - index
    lower_phase, upper_phase = phases[index], phases[index + 1]
    lower_tangent, upper_tangent = tangents[index], tangents[index + 1]
    lagged_phase = (1.0 - fraction) * lower_phase + fraction * upper_phase
    lagged_tangent = (1.0 - fraction) * lower_tangent + fraction * upper_tangent
    slope = (upper_phase - lower_phase) / dt
    total_derivative = lagged_tangent - slope
    return (
        np.ascontiguousarray(lagged_phase),
        np.ascontiguousarray(total_derivative),
        np.ascontiguousarray(lagged_tangent),
    )


def _stage_rates(
    phase: NDArray[np.float64],
    tangent: NDArray[np.float64],
    stage_time: float,
    *,
    phases: NDArray[np.float64],
    tangents: NDArray[np.float64],
    history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    delay: float,
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Return the ``(phase rate, tangent rate)`` of one RK4 stage at ``stage_time``."""
    lagged_phase, lagged_total_derivative, _ = _lag(
        phases, tangents, history, dt, stage_time - delay
    )
    force = _force(phase, lagged_phase, omega, coupling)
    tangent_rate = _tangent_field(phase, lagged_phase, tangent, lagged_total_derivative, coupling)
    return force, tangent_rate


def delayed_delay_sensitivity(
    history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    delay: float,
    dt: float,
    n_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Terminal phases and their delay sensitivity ``∂θ_N/∂τ`` of the time-delayed Kuramoto model.

    Integrates ``θ̇_j = ω_j + Σ_k K_jk sin(θ_k(t−τ) − θ_j(t))`` from a **constant** history
    ``θ(s) = θ_0`` (``s ≤ 0``) by a continuous-delay method-of-steps RK4, co-integrating the tangent
    ``σ(t) = ∂θ(t)/∂τ`` by the identical stages, and returns the final phases and ``σ_N = ∂θ_N/∂τ``.

    Parameters
    ----------
    history : numpy.ndarray
        The constant history phases ``θ_0`` as a one-dimensional array of length ``N``.
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` networked coupling matrix ``K``.
    delay : float
        The coupling delay ``τ`` (``≥ dt`` and not a half-integer multiple of ``dt``).
    dt : float
        The integration step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``).

    Returns
    -------
    tuple of numpy.ndarray
        ``(theta_final, dtheta_dtau)``, each a length-``N`` float64 array.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound (see :func:`_validate`).
    """
    history = np.ascontiguousarray(history, dtype=np.float64)
    omega = np.ascontiguousarray(omega, dtype=np.float64)
    coupling = np.ascontiguousarray(coupling, dtype=np.float64)
    count = _validate(history, omega, coupling, delay, dt, n_steps)

    phases = np.empty((n_steps + 1, count), dtype=np.float64)
    tangents = np.zeros((n_steps + 1, count), dtype=np.float64)
    phases[0] = history
    # tangents[0] = 0: the constant history is independent of τ.

    for step in range(n_steps):
        time = step * dt
        current_phase = phases[step]
        current_tangent = tangents[step]

        f1, s1 = _stage_rates(
            current_phase,
            current_tangent,
            time,
            phases=phases,
            tangents=tangents,
            history=history,
            omega=omega,
            coupling=coupling,
            delay=delay,
            dt=dt,
        )
        f2, s2 = _stage_rates(
            current_phase + 0.5 * dt * f1,
            current_tangent + 0.5 * dt * s1,
            time + 0.5 * dt,
            phases=phases,
            tangents=tangents,
            history=history,
            omega=omega,
            coupling=coupling,
            delay=delay,
            dt=dt,
        )
        f3, s3 = _stage_rates(
            current_phase + 0.5 * dt * f2,
            current_tangent + 0.5 * dt * s2,
            time + 0.5 * dt,
            phases=phases,
            tangents=tangents,
            history=history,
            omega=omega,
            coupling=coupling,
            delay=delay,
            dt=dt,
        )
        f4, s4 = _stage_rates(
            current_phase + dt * f3,
            current_tangent + dt * s3,
            time + dt,
            phases=phases,
            tangents=tangents,
            history=history,
            omega=omega,
            coupling=coupling,
            delay=delay,
            dt=dt,
        )

        phases[step + 1] = current_phase + (dt / 6.0) * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
        tangents[step + 1] = current_tangent + (dt / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4)

    return (
        np.ascontiguousarray(phases[n_steps]),
        np.ascontiguousarray(tangents[n_steps]),
    )


def delayed_delay_gradient(
    history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    delay: float,
    dt: float,
    n_steps: int,
    objective: DelayedTerminalObjective,
    objective_grad: DelayedTerminalObjectiveGrad,
) -> tuple[float, float]:
    r"""Value and delay sensitivity ``dL/dτ`` of a terminal objective through the delayed Kuramoto model.

    Evaluates ``L(θ_N)`` and contracts the terminal cotangent ``∂L/∂θ_N`` with the delay sensitivity
    ``∂θ_N/∂τ`` to return the scalar ``dL/dτ`` — the derivative of a control or inference objective
    with respect to the coupling delay itself.

    Parameters
    ----------
    history, omega, coupling, delay, dt, n_steps
        As in :func:`delayed_delay_sensitivity`.
    objective : callable
        The terminal objective ``L(θ_N) → float``.
    objective_grad : callable
        Its gradient ``θ_N → ∂L/∂θ_N`` (length ``N``).

    Returns
    -------
    tuple
        ``(value, d_value_d_delay)``.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound, or ``objective_grad`` returns a cotangent
        of the wrong shape.
    """
    theta_final, dtheta_dtau = delayed_delay_sensitivity(
        history, omega, coupling, delay=delay, dt=dt, n_steps=n_steps
    )
    count = int(theta_final.size)
    value = float(objective(theta_final))
    cotangent = np.ascontiguousarray(objective_grad(theta_final), dtype=np.float64)
    if cotangent.shape != (count,):
        raise ValueError(
            f"objective_grad must return a ({count},) cotangent, got {cotangent.shape}"
        )
    return value, float(cotangent @ dtheta_dtau)


__all__ = [
    "delayed_delay_gradient",
    "delayed_delay_sensitivity",
]
