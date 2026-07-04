# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable time-delayed Kuramoto integrator
r"""Differentiable time-delayed Kuramoto integrator via the method-of-steps sensitivity.

The time-delayed Kuramoto DDE ``θ̇(t) = ω + F(θ(t), θ(t-τ))`` integrated by
:func:`~oscillatools.accel.kuramoto_delayed.integrate_delayed_kuramoto` (a delay-aware
method-of-steps RK4) carried no gradient path. Differentiating through a delay differential
equation is the hardest of the variant integrators: the "initial condition" is not a point but
the whole history ``θ`` on ``[-τ, 0]``, and every step's force reads a *lagged* state, so the
sensitivity of the lagged term references the sensitivity stored earlier in the run. Continuous
adjoints do not apply to DDEs — only a discretise-then-optimise method does, and even the leading
differentiable-equation library offers only that for delays — so this module differentiates the
discrete method-of-steps map directly, which is exact and is the right tool for the delay case.

Method — method-of-steps forward-mode sensitivity
-------------------------------------------------
A sensitivity buffer ``S`` is carried alongside the phase buffer, one ``(N, P)`` block per grid
index, so the lagged term's tangent is read from the stored history exactly as the lagged phase
is. Each RK4 stage propagates the tangent with the two analytic Jacobian blocks of the delayed
networked force — ``∂F_j/∂θ_j(t) = −Σ_k K_{jk}\cos(θ_k(t-τ) − θ_j(t))`` (diagonal in the current
state) and ``∂F_j/∂θ_k(t-τ) = K_{jk}\cos(θ_k(t-τ) − θ_j(t))`` (the lagged block) — plus the
explicit injections ``∂(ω+F)/∂ω = I`` and ``∂F_j/∂K_{jk} = \sin(θ_k(t-τ) − θ_j(t))``. The
history rows seed the buffer with identity sensitivity blocks. Differentiating the discrete
stages exactly reproduces the derivative of the discrete map, so the gradients match a central
finite difference of the integrator. The channels are the full initial history, the frequencies
and the coupling (the delay ``τ`` is structural — an integer number of steps — and is not a
differentiable parameter). The module adds no compute kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: A terminal objective ``L(θ_N)`` on the final phases of a delayed run.
DelayedTerminalObjective = Callable[[NDArray[np.float64]], float]

#: The gradient of a :data:`DelayedTerminalObjective`, returning ``∂L/∂θ_N``.
DelayedTerminalObjectiveGrad = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class DelayedGradients:
    """Gradients of a terminal objective through the time-delayed Kuramoto integrator.

    Attributes
    ----------
    initial_history : numpy.ndarray
        ``∂L/∂θ(history)`` (shape ``(delay_steps + 1, N)``, matching the integrator's history).
    omega : numpy.ndarray
        ``∂L/∂ω`` (length ``N``).
    coupling : numpy.ndarray
        ``∂L/∂K`` (shape ``(N, N)``).
    """

    initial_history: NDArray[np.float64]
    omega: NDArray[np.float64]
    coupling: NDArray[np.float64]


def _validate(
    initial_history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    delay: float,
    dt: float,
    n_steps: int,
    delay_tolerance: float,
) -> tuple[int, int]:
    """Validate the delayed differentiable problem; return ``(count, delay_steps)``."""
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
    if omega.ndim != 1 or omega.size < 1:
        raise ValueError("omega must be a non-empty one-dimensional array")
    count = int(omega.size)
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape {(count, count)}, got {coupling.shape}")
    if initial_history.shape != (delay_steps + 1, count):
        raise ValueError(
            f"initial_history must have shape {(delay_steps + 1, count)}, "
            f"got {initial_history.shape}"
        )
    return count, delay_steps


def _stage(
    current: NDArray[np.float64],
    current_sensitivity: NDArray[np.float64],
    lagged: NDArray[np.float64],
    lagged_sensitivity: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    count: int,
    omega_start: int,
    coupling_start: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Return ``(k, dk)`` for one RK4 stage of the delayed networked field.

    ``k = ω + F(current, lagged)`` and ``dk`` is its tangent given the current/lagged tangents,
    including the explicit ``∂/∂ω`` and ``∂/∂K`` injections.
    """
    difference = lagged[np.newaxis, :] - current[:, np.newaxis]  # Δ_jk = θ_k(t-τ) − θ_j(t)
    sine = np.sin(difference)
    cosine = np.cos(difference)
    force = (coupling * sine).sum(axis=1)
    k = omega + force

    weighted_cos = coupling * cosine
    current_jacobian_diagonal = -weighted_cos.sum(axis=1)  # ∂F_j/∂θ_j(t)
    # dF = diag(∂F/∂θ(t)) · dcurrent + (∂F/∂θ(t-τ)) · dlagged
    dk = current_jacobian_diagonal[:, np.newaxis] * current_sensitivity + (
        weighted_cos @ lagged_sensitivity
    )
    dk[:, omega_start : omega_start + count] += np.eye(count)  # ∂(ω+F)/∂ω = I
    for j in range(count):
        base = coupling_start + j * count
        dk[j, base : base + count] += sine[j]  # ∂F_j/∂K_{jk} = sin(Δ_jk)
    return k, dk


def delayed_phase_sensitivity(
    initial_history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    delay: float,
    dt: float,
    n_steps: int,
    delay_tolerance: float = 1e-9,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Integrate the delayed method-of-steps RK4 with forward-mode sensitivity of the final state.

    Returns the final phases ``θ_N`` and the sensitivity matrix ``S_N = ∂θ_N/∂p`` of shape
    ``(N, P)`` for the channel layout ``[history ((delay_steps+1)·N), ω (N), K (N²)]`` (so
    ``P = (delay_steps + 1)·N + N + N²``).

    Parameters
    ----------
    initial_history : numpy.ndarray
        The phase history on ``[-τ, 0]`` as a ``(delay_steps + 1, N)`` array (the last row is
        ``θ(0)``), matching :func:`~oscillatools.accel.kuramoto_delayed.integrate_delayed_kuramoto`.
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` networked coupling matrix ``K``.
    delay : float
        The coupling delay ``τ`` (``> 0``, an integer multiple of ``dt``).
    dt : float
        The integration step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``).
    delay_tolerance : float, optional
        Absolute tolerance for ``τ`` being an integer multiple of ``dt``; defaults to ``1e-9``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(theta_final, sensitivity)``.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    history = np.ascontiguousarray(initial_history, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    coupling_matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count, delay_steps = _validate(
        history, frequencies, coupling_matrix, delay, dt, n_steps, delay_tolerance
    )
    history_width = (delay_steps + 1) * count
    width = history_width + count + count * count
    omega_start = history_width
    coupling_start = history_width + count

    buffer: list[NDArray[np.float64]] = [np.array(row, dtype=np.float64) for row in history]
    sensitivity_buffer: list[NDArray[np.float64]] = []
    for grid_index in range(delay_steps + 1):
        block = np.zeros((count, width), dtype=np.float64)
        block[:, grid_index * count : (grid_index + 1) * count] = np.eye(count)
        sensitivity_buffer.append(block)

    for step in range(n_steps):
        theta = buffer[delay_steps + step]
        theta_sensitivity = sensitivity_buffer[delay_steps + step]
        lag_full = buffer[step]
        lag_full_sensitivity = sensitivity_buffer[step]
        lag_half = 0.5 * (buffer[step] + buffer[step + 1])
        lag_half_sensitivity = 0.5 * (sensitivity_buffer[step] + sensitivity_buffer[step + 1])
        lag_next = buffer[step + 1]
        lag_next_sensitivity = sensitivity_buffer[step + 1]

        k1, d1 = _stage(
            theta,
            theta_sensitivity,
            lag_full,
            lag_full_sensitivity,
            frequencies,
            coupling_matrix,
            count,
            omega_start,
            coupling_start,
        )
        k2, d2 = _stage(
            theta + 0.5 * dt * k1,
            theta_sensitivity + 0.5 * dt * d1,
            lag_half,
            lag_half_sensitivity,
            frequencies,
            coupling_matrix,
            count,
            omega_start,
            coupling_start,
        )
        k3, d3 = _stage(
            theta + 0.5 * dt * k2,
            theta_sensitivity + 0.5 * dt * d2,
            lag_half,
            lag_half_sensitivity,
            frequencies,
            coupling_matrix,
            count,
            omega_start,
            coupling_start,
        )
        k4, d4 = _stage(
            theta + dt * k3,
            theta_sensitivity + dt * d3,
            lag_next,
            lag_next_sensitivity,
            frequencies,
            coupling_matrix,
            count,
            omega_start,
            coupling_start,
        )
        buffer.append(theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        sensitivity_buffer.append(theta_sensitivity + (dt / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4))

    return buffer[delay_steps + n_steps], sensitivity_buffer[delay_steps + n_steps]


def delayed_terminal_value_and_grad(
    initial_history: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    delay: float,
    dt: float,
    n_steps: int,
    objective: DelayedTerminalObjective,
    objective_grad: DelayedTerminalObjectiveGrad,
    delay_tolerance: float = 1e-9,
) -> tuple[float, DelayedGradients]:
    r"""Differentiate a terminal objective through the time-delayed Kuramoto integrator.

    Evaluates ``L(θ_N)`` and returns its gradients with respect to the full initial history, the
    frequencies and the coupling by contracting the final-phase cotangent ``∂L/∂θ_N`` with the
    method-of-steps sensitivity ``S_N``.

    Parameters
    ----------
    initial_history : numpy.ndarray
        The phase history on ``[-τ, 0]`` as a ``(delay_steps + 1, N)`` array.
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` networked coupling matrix ``K``.
    delay : float
        The coupling delay ``τ`` (``> 0``, an integer multiple of ``dt``).
    dt : float
        The integration step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``).
    objective : callable
        The terminal objective ``L(θ_N) → float``.
    objective_grad : callable
        Its gradient ``θ_N → ∂L/∂θ_N`` (length ``N``).
    delay_tolerance : float, optional
        Absolute tolerance for ``τ`` being an integer multiple of ``dt``; defaults to ``1e-9``.

    Returns
    -------
    tuple
        ``(value, DelayedGradients)``.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound, or ``objective_grad`` returns a
        cotangent of the wrong shape.
    """
    theta_final, sensitivity = delayed_phase_sensitivity(
        initial_history,
        omega,
        coupling,
        delay=delay,
        dt=dt,
        n_steps=n_steps,
        delay_tolerance=delay_tolerance,
    )
    count = int(theta_final.size)
    delay_steps = int(round(delay / dt))
    value = float(objective(theta_final))
    cotangent = np.ascontiguousarray(objective_grad(theta_final), dtype=np.float64)
    if cotangent.shape != (count,):
        raise ValueError(
            f"objective_grad must return a ({count},) cotangent, got {cotangent.shape}"
        )

    flat = cotangent @ sensitivity
    history_width = (delay_steps + 1) * count
    return value, DelayedGradients(
        initial_history=np.ascontiguousarray(
            flat[:history_width].reshape(delay_steps + 1, count), dtype=np.float64
        ),
        omega=np.ascontiguousarray(flat[history_width : history_width + count], dtype=np.float64),
        coupling=np.ascontiguousarray(
            flat[history_width + count :].reshape(count, count), dtype=np.float64
        ),
    )


__all__ = [
    "DelayedGradients",
    "DelayedTerminalObjective",
    "DelayedTerminalObjectiveGrad",
    "delayed_phase_sensitivity",
    "delayed_terminal_value_and_grad",
]
