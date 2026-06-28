# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable inertial (second-order) Kuramoto integrator
r"""Differentiable inertial (second-order) Kuramoto integrator via forward-mode sensitivity.

The inertial Kuramoto / power-grid swing model ``m θ̈ + γ θ̇ = ω + F(θ)`` integrated by
:func:`~scpn_quantum_control.accel.kuramoto_inertial.integrate_inertial` carried no gradient
path: unlike the first-order Euler/RK4/DOPRI integrators it had no adjoint, so an objective
on a swing-equation trajectory could not be optimised over the initial state, the power
injection, the coupling, the inertia, or the damping. This module closes that gap for the
**networked** inertial model.

Method — discrete forward-mode sensitivity
------------------------------------------
The fixed-step RK4 over the concatenated phase-space state ``z = [θ, v]`` is a smooth map
``z_{n+1} = Φ(z_n; ω, K, m, γ)``. Its exact derivative with respect to every input is
obtained by propagating the sensitivity matrix ``S = ∂z/∂p`` through the *same* four RK4
stages: at a stage evaluated at state ``w`` the tangent obeys

.. math::

    \dot S = J(w)\,S + E(w),

with :math:`J(w)` the analytic :func:`~scpn_quantum_control.accel.kuramoto_inertial.inertial_jacobian`
block matrix and :math:`E(w) = \partial f/\partial p` the explicit parameter injection of the
inertial vector field. Differentiating the RK4 stages exactly reproduces the derivative of the
*discrete* map (not merely the continuous flow), so the returned gradients match a central
finite difference of the integrator to truncation order rather than to the integrator's global
error. For the small-to-moderate oscillator counts this swing model targets, forward-mode
sensitivity is the efficient choice (state + parameters below the adjoint cross-over); a
reverse path is the right tool only once ``N`` is large enough that the ``N²`` coupling
sensitivities dominate.

The explicit injections for the networked force are analytic: ``∂F_p/∂K_{pq} = sin(θ_q − θ_p)``,
``∂F/∂θ`` is the networked Jacobian, ``∂(v̇)/∂ω = I/m``, ``∂(v̇)/∂γ = −v/m`` and
``∂(v̇)/∂m = −(ω + F(θ) − γv)/m²``. The module adds no compute kernel; it composes the
polyglot networked force and Jacobian.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .kuramoto_inertial import inertial_jacobian
from .networked_kuramoto import networked_kuramoto_force, networked_kuramoto_jacobian

#: A terminal objective ``L(θ_N, v_N)`` on the final phase-space state.
InertialTerminalObjective = Callable[[NDArray[np.float64], NDArray[np.float64]], float]

#: The gradient of a :data:`InertialTerminalObjective`, returning ``(∂L/∂θ_N, ∂L/∂v_N)``.
InertialTerminalObjectiveGrad = Callable[
    [NDArray[np.float64], NDArray[np.float64]],
    "tuple[NDArray[np.float64], NDArray[np.float64]]",
]


@dataclass(frozen=True)
class InertialGradients:
    """Gradients of a terminal objective through the inertial Kuramoto integrator.

    Attributes
    ----------
    initial_phases : numpy.ndarray
        ``∂L/∂θ_0`` (length ``N``).
    initial_velocities : numpy.ndarray
        ``∂L/∂v_0`` (length ``N``).
    omega : numpy.ndarray
        ``∂L/∂ω`` (length ``N``).
    coupling : numpy.ndarray
        ``∂L/∂K`` (shape ``(N, N)``).
    mass : float
        ``∂L/∂m``.
    damping : float
        ``∂L/∂γ``.
    """

    initial_phases: NDArray[np.float64]
    initial_velocities: NDArray[np.float64]
    omega: NDArray[np.float64]
    coupling: NDArray[np.float64]
    mass: float
    damping: float


def _validate(
    phases: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    mass: float,
    damping: float,
    dt: float,
    n_steps: int,
) -> int:
    """Validate the inertial differentiable problem and return the oscillator count."""
    if omega.ndim != 1 or omega.size < 1:
        raise ValueError("omega must be a non-empty one-dimensional array")
    count = int(omega.size)
    if phases.shape != omega.shape:
        raise ValueError(f"phases must have shape {omega.shape}, got {phases.shape}")
    if velocities.shape != omega.shape:
        raise ValueError(f"velocities must have shape {omega.shape}, got {velocities.shape}")
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape {(count, count)}, got {coupling.shape}")
    if mass <= 0.0:
        raise ValueError(f"mass must be positive, got {mass}")
    if damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {damping}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    return count


def _explicit_injection(
    theta_w: NDArray[np.float64],
    velocity_w: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    mass: float,
    damping: float,
    count: int,
    width: int,
) -> NDArray[np.float64]:
    r"""Return the explicit parameter injection ``E(w) = ∂f/∂p`` at a stage state.

    The columns are laid out ``[θ₀ (N), v₀ (N), ω (N), K (N²), m, γ]``; only the explicit
    field-parameter blocks (ω, K, m, γ) are non-zero, and they populate the velocity half of
    the ``2N`` phase-space derivative.
    """
    injection = np.zeros((2 * count, width), dtype=np.float64)
    velocity_block = injection[count:]
    # omega columns: ∂(v̇)/∂ω_j = e_j / m
    omega_start = 2 * count
    velocity_block[:, omega_start : omega_start + count] = np.eye(count) / mass
    # coupling columns: ∂(v̇_p)/∂K_pq = sin(θ_q − θ_p) / m
    coupling_start = omega_start + count
    phase_delta = theta_w[None, :] - theta_w[:, None]
    sin_delta = np.sin(phase_delta) / mass
    for p in range(count):
        base = coupling_start + p * count
        velocity_block[p, base : base + count] = sin_delta[p]
    # mass column: ∂(v̇)/∂m = −(ω + F(θ) − γ v) / m² = −a / m
    acceleration = (
        omega + networked_kuramoto_force(theta_w, coupling) - damping * velocity_w
    ) / mass
    mass_col = coupling_start + count * count
    velocity_block[:, mass_col] = -acceleration / mass
    # damping column: ∂(v̇)/∂γ = −v / m
    velocity_block[:, mass_col + 1] = -velocity_w / mass
    return injection


def _stage_derivative(
    state: NDArray[np.float64],
    sensitivity: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    mass: float,
    damping: float,
    count: int,
    width: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(f(state), J(state)·S + E(state))`` for one RK4 stage."""
    theta = state[:count]
    velocity = state[count:]
    force = networked_kuramoto_force(theta, coupling)
    acceleration = (omega + force - damping * velocity) / mass
    field = np.concatenate([velocity, acceleration])

    jacobian = inertial_jacobian(
        theta, lambda t: networked_kuramoto_jacobian(t, coupling), mass, damping=damping
    )
    injection = _explicit_injection(theta, velocity, omega, coupling, mass, damping, count, width)
    sensitivity_derivative = jacobian @ sensitivity + injection
    return field, sensitivity_derivative


def _initial_sensitivity(count: int, width: int) -> NDArray[np.float64]:
    """Return ``S_0 = ∂z_0/∂p`` for the channel layout ``[θ₀, v₀, ω, K, m, γ]``."""
    sensitivity = np.zeros((2 * count, width), dtype=np.float64)
    sensitivity[:count, :count] = np.eye(count)  # ∂θ₀/∂θ₀ = I
    sensitivity[count:, count : 2 * count] = np.eye(count)  # ∂v₀/∂v₀ = I
    return sensitivity


def inertial_state_sensitivity(
    phases: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    mass: float,
    *,
    damping: float = 1.0,
    dt: float,
    n_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Integrate the inertial RK4 with forward-mode sensitivity of the final state.

    Returns the final phases ``θ_N``, the final velocities ``v_N`` and the sensitivity matrix
    ``S_N = ∂[θ_N, v_N]/∂p`` of shape ``(2N, P)`` for the channel layout
    ``[θ₀ (N), v₀ (N), ω (N), K (N²), m, γ]`` (so ``P = 3N + N² + 2``).

    Parameters
    ----------
    phases, velocities, omega : numpy.ndarray
        Initial phases, initial velocities and natural frequencies (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` networked coupling matrix ``K``.
    mass : float
        The inertia ``m`` (``> 0``).
    damping : float, optional
        The damping ``γ`` (``≥ 0``); defaults to ``1``.
    dt : float
        The RK4 step (``> 0``).
    n_steps : int
        The number of RK4 steps (``≥ 1``).

    Returns
    -------
    tuple of numpy.ndarray
        ``(theta_final, velocity_final, sensitivity)``.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    theta0 = np.ascontiguousarray(phases, dtype=np.float64)
    velocity0 = np.ascontiguousarray(velocities, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    coupling_matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = _validate(theta0, velocity0, frequencies, coupling_matrix, mass, damping, dt, n_steps)
    width = 3 * count + count * count + 2

    state = np.concatenate([theta0, velocity0])
    sensitivity = _initial_sensitivity(count, width)

    for _ in range(n_steps):
        k1, s1 = _stage_derivative(
            state, sensitivity, frequencies, coupling_matrix, mass, damping, count, width
        )
        k2, s2 = _stage_derivative(
            state + 0.5 * dt * k1,
            sensitivity + 0.5 * dt * s1,
            frequencies,
            coupling_matrix,
            mass,
            damping,
            count,
            width,
        )
        k3, s3 = _stage_derivative(
            state + 0.5 * dt * k2,
            sensitivity + 0.5 * dt * s2,
            frequencies,
            coupling_matrix,
            mass,
            damping,
            count,
            width,
        )
        k4, s4 = _stage_derivative(
            state + dt * k3,
            sensitivity + dt * s3,
            frequencies,
            coupling_matrix,
            mass,
            damping,
            count,
            width,
        )
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        sensitivity = sensitivity + (dt / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4)

    return state[:count], state[count:], sensitivity


def inertial_terminal_value_and_grad(
    phases: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    mass: float,
    *,
    damping: float = 1.0,
    dt: float,
    n_steps: int,
    objective: InertialTerminalObjective,
    objective_grad: InertialTerminalObjectiveGrad,
) -> tuple[float, InertialGradients]:
    r"""Differentiate a terminal objective through the inertial Kuramoto integrator.

    Evaluates ``L(θ_N, v_N)`` and returns its gradients with respect to every input
    (``θ_0, v_0, ω, K, m, γ``) by contracting the final-state cotangent
    ``(∂L/∂θ_N, ∂L/∂v_N)`` with the forward-mode sensitivity ``S_N``.

    Parameters
    ----------
    phases, velocities, omega : numpy.ndarray
        Initial phases, initial velocities and natural frequencies (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` networked coupling matrix ``K``.
    mass : float
        The inertia ``m`` (``> 0``).
    damping : float, optional
        The damping ``γ`` (``≥ 0``); defaults to ``1``.
    dt : float
        The RK4 step (``> 0``).
    n_steps : int
        The number of RK4 steps (``≥ 1``).
    objective : callable
        The terminal objective ``L(θ_N, v_N) → float``.
    objective_grad : callable
        Its gradient ``(θ_N, v_N) → (∂L/∂θ_N, ∂L/∂v_N)`` (each length ``N``).

    Returns
    -------
    tuple
        ``(value, InertialGradients)``.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound, or ``objective_grad`` returns a
        cotangent of the wrong shape.
    """
    theta_final, velocity_final, sensitivity = inertial_state_sensitivity(
        phases, velocities, omega, coupling, mass, damping=damping, dt=dt, n_steps=n_steps
    )
    count = int(theta_final.size)
    value = float(objective(theta_final, velocity_final))
    grad_theta_n, grad_velocity_n = objective_grad(theta_final, velocity_final)
    grad_theta_n = np.ascontiguousarray(grad_theta_n, dtype=np.float64)
    grad_velocity_n = np.ascontiguousarray(grad_velocity_n, dtype=np.float64)
    if grad_theta_n.shape != (count,) or grad_velocity_n.shape != (count,):
        raise ValueError(
            f"objective_grad must return two ({count},) cotangents, got "
            f"{grad_theta_n.shape} and {grad_velocity_n.shape}"
        )

    cotangent = np.concatenate([grad_theta_n, grad_velocity_n])
    flat = cotangent @ sensitivity  # ∂L/∂p over the channel layout

    omega_start = 2 * count
    coupling_start = omega_start + count
    mass_index = coupling_start + count * count
    return value, InertialGradients(
        initial_phases=np.ascontiguousarray(flat[:count], dtype=np.float64),
        initial_velocities=np.ascontiguousarray(flat[count : 2 * count], dtype=np.float64),
        omega=np.ascontiguousarray(flat[omega_start:coupling_start], dtype=np.float64),
        coupling=np.ascontiguousarray(
            flat[coupling_start:mass_index].reshape(count, count), dtype=np.float64
        ),
        mass=float(flat[mass_index]),
        damping=float(flat[mass_index + 1]),
    )


__all__ = [
    "InertialGradients",
    "InertialTerminalObjective",
    "InertialTerminalObjectiveGrad",
    "inertial_state_sensitivity",
    "inertial_terminal_value_and_grad",
]
