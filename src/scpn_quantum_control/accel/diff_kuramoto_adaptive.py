# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable adaptive (plastic-coupling) Kuramoto integrator
r"""Differentiable adaptive (Hebbian plastic-coupling) Kuramoto integrator.

The adaptive Kuramoto model co-evolves the phases and the coupling weights:
``θ̇_i = ω_i + Σ_j K_{ij} sin(θ_j − θ_i)`` with the Seliger–Young–Tsimring Hebbian rule
``K̇_{ij} = ε (cos(θ_j − θ_i) − K_{ij})``. The forward integrator
:func:`~scpn_quantum_control.accel.kuramoto_adaptive.integrate_adaptive_kuramoto` carried no
gradient path, so a learning objective on the self-organised connectivity could not be
optimised over the initial state, the natural frequencies, or the plasticity rate. This module
closes that gap for the canonical networked–Hebbian pairing.

Method — discrete forward-mode sensitivity
------------------------------------------
The fixed-step RK4 over the joint state ``z = [θ, vec(K)]`` (the coupling flattened row-major,
length ``N + N²``) is a smooth map ``z_{n+1} = Φ(z_n; ω, ε)``. Its exact derivative is obtained
by propagating the sensitivity matrix ``S = ∂z/∂p`` through the *same* four RK4 stages with the
analytic coupled Jacobian
:func:`~scpn_quantum_control.accel.kuramoto_adaptive.hebbian_adaptive_jacobian` and the explicit
parameter injection ``E = ∂f/∂p``. Differentiating the discrete stages exactly reproduces the
derivative of the discrete map, so the returned gradients match a central finite difference of
the integrator rather than the integrator's global error. Forward-mode is the efficient choice
in the small-to-moderate-``N`` regime this model targets; the explicit injections are analytic
(``∂θ̇/∂ω = I``, ``∂K̇/∂ε = cos(θ_j − θ_i) − K_{ij}``), and the module adds no compute kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .kuramoto_adaptive import hebbian_adaptive_jacobian
from .networked_kuramoto import networked_kuramoto_force

#: A terminal objective ``L(θ_N, K_N)`` on the final phases and coupling.
AdaptiveTerminalObjective = Callable[[NDArray[np.float64], NDArray[np.float64]], float]

#: The gradient of an :data:`AdaptiveTerminalObjective`, returning ``(∂L/∂θ_N, ∂L/∂K_N)``.
AdaptiveTerminalObjectiveGrad = Callable[
    [NDArray[np.float64], NDArray[np.float64]],
    "tuple[NDArray[np.float64], NDArray[np.float64]]",
]


@dataclass(frozen=True)
class AdaptiveGradients:
    """Gradients of a terminal objective through the adaptive Kuramoto integrator.

    Attributes
    ----------
    initial_phases : numpy.ndarray
        ``∂L/∂θ_0`` (length ``N``).
    initial_coupling : numpy.ndarray
        ``∂L/∂K_0`` (shape ``(N, N)``).
    omega : numpy.ndarray
        ``∂L/∂ω`` (length ``N``).
    plasticity_rate : float
        ``∂L/∂ε``.
    """

    initial_phases: NDArray[np.float64]
    initial_coupling: NDArray[np.float64]
    omega: NDArray[np.float64]
    plasticity_rate: float


def _validate(
    phases: NDArray[np.float64],
    coupling: NDArray[np.float64],
    omega: NDArray[np.float64],
    plasticity_rate: float,
    dt: float,
    n_steps: int,
) -> int:
    """Validate the adaptive differentiable problem and return the oscillator count."""
    if phases.ndim != 1 or phases.size < 1:
        raise ValueError("phases must be a non-empty one-dimensional array")
    count = int(phases.size)
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape {(count, count)}, got {coupling.shape}")
    if omega.shape != (count,):
        raise ValueError(f"omega must have shape {(count,)}, got {omega.shape}")
    if plasticity_rate < 0.0:
        raise ValueError(f"plasticity_rate must be non-negative, got {plasticity_rate}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    return count


def _field(
    state: NDArray[np.float64],
    omega: NDArray[np.float64],
    plasticity_rate: float,
    count: int,
) -> NDArray[np.float64]:
    """Return the joint vector field ``[θ̇, vec(K̇)]`` at a flattened state ``z = [θ, vec(K)]``."""
    theta = state[:count]
    coupling = state[count:].reshape(count, count)
    phase_velocity = omega + networked_kuramoto_force(theta, coupling)
    difference = theta[np.newaxis, :] - theta[:, np.newaxis]
    coupling_velocity = plasticity_rate * (np.cos(difference) - coupling)
    return np.concatenate([phase_velocity, coupling_velocity.reshape(-1)])


def _explicit_injection(
    state: NDArray[np.float64],
    plasticity_rate: float,
    count: int,
    width: int,
) -> NDArray[np.float64]:
    r"""Return the explicit parameter injection ``E = ∂f/∂p`` at a stage state.

    Columns are laid out ``[θ₀ (N), K₀ (N²), ω (N), ε]``; only the explicit field parameters
    (ω, ε) are non-zero. ``∂θ̇/∂ω = I`` populates the phase half; ``∂K̇/∂ε = cos(θ_j − θ_i) − K``
    populates the coupling half.
    """
    theta = state[:count]
    coupling = state[count:].reshape(count, count)
    injection = np.zeros((count + count * count, width), dtype=np.float64)
    omega_start = count + count * count
    injection[:count, omega_start : omega_start + count] = np.eye(count)
    difference = theta[np.newaxis, :] - theta[:, np.newaxis]
    eps_col = omega_start + count
    injection[count:, eps_col] = (np.cos(difference) - coupling).reshape(-1)
    return injection


def _stage_derivative(
    state: NDArray[np.float64],
    sensitivity: NDArray[np.float64],
    omega: NDArray[np.float64],
    plasticity_rate: float,
    count: int,
    width: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(f(state), J(state)·S + E(state))`` for one RK4 stage."""
    field = _field(state, omega, plasticity_rate, count)
    coupling = state[count:].reshape(count, count)
    jacobian = hebbian_adaptive_jacobian(state[:count], coupling, plasticity_rate=plasticity_rate)
    injection = _explicit_injection(state, plasticity_rate, count, width)
    return field, jacobian @ sensitivity + injection


def _initial_sensitivity(count: int, width: int) -> NDArray[np.float64]:
    """Return ``S_0 = ∂z_0/∂p`` for the layout ``[θ₀ (N), K₀ (N²), ω (N), ε]``."""
    dimension = count + count * count
    sensitivity = np.zeros((dimension, width), dtype=np.float64)
    sensitivity[:dimension, :dimension] = np.eye(dimension)  # θ₀ and K₀ initial conditions
    return sensitivity


def adaptive_state_sensitivity(
    phases: NDArray[np.float64],
    coupling: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    plasticity_rate: float,
    dt: float,
    n_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Integrate the adaptive RK4 with forward-mode sensitivity of the final state.

    Returns the final phases ``θ_N``, the final coupling ``K_N`` and the sensitivity matrix
    ``S_N = ∂[θ_N, vec(K_N)]/∂p`` of shape ``(N + N², P)`` for the channel layout
    ``[θ₀ (N), K₀ (N²), ω (N), ε]`` (so ``P = 2N + N² + 1``).

    Parameters
    ----------
    phases : numpy.ndarray
        The initial phases ``θ(0)`` (length ``N``).
    coupling : numpy.ndarray
        The initial coupling ``K(0)`` (shape ``(N, N)``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    plasticity_rate : float
        The Hebbian plasticity rate ``ε`` (``≥ 0``).
    dt : float
        The RK4 step (``> 0``).
    n_steps : int
        The number of RK4 steps (``≥ 1``).

    Returns
    -------
    tuple of numpy.ndarray
        ``(theta_final, coupling_final, sensitivity)``.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    theta0 = np.ascontiguousarray(phases, dtype=np.float64)
    coupling0 = np.ascontiguousarray(coupling, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    count = _validate(theta0, coupling0, frequencies, plasticity_rate, dt, n_steps)
    width = 2 * count + count * count + 1

    state = np.concatenate([theta0, coupling0.reshape(-1)])
    sensitivity = _initial_sensitivity(count, width)

    for _ in range(n_steps):
        k1, s1 = _stage_derivative(state, sensitivity, frequencies, plasticity_rate, count, width)
        k2, s2 = _stage_derivative(
            state + 0.5 * dt * k1,
            sensitivity + 0.5 * dt * s1,
            frequencies,
            plasticity_rate,
            count,
            width,
        )
        k3, s3 = _stage_derivative(
            state + 0.5 * dt * k2,
            sensitivity + 0.5 * dt * s2,
            frequencies,
            plasticity_rate,
            count,
            width,
        )
        k4, s4 = _stage_derivative(
            state + dt * k3,
            sensitivity + dt * s3,
            frequencies,
            plasticity_rate,
            count,
            width,
        )
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        sensitivity = sensitivity + (dt / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4)

    theta_final = state[:count]
    coupling_final = state[count:].reshape(count, count)
    return theta_final, coupling_final, sensitivity


def adaptive_terminal_value_and_grad(
    phases: NDArray[np.float64],
    coupling: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    plasticity_rate: float,
    dt: float,
    n_steps: int,
    objective: AdaptiveTerminalObjective,
    objective_grad: AdaptiveTerminalObjectiveGrad,
) -> tuple[float, AdaptiveGradients]:
    r"""Differentiate a terminal objective through the adaptive Kuramoto integrator.

    Evaluates ``L(θ_N, K_N)`` and returns its gradients with respect to every input
    (``θ_0, K_0, ω, ε``) by contracting the final-state cotangent ``(∂L/∂θ_N, ∂L/∂K_N)`` with
    the forward-mode sensitivity ``S_N``.

    Parameters
    ----------
    phases : numpy.ndarray
        The initial phases ``θ(0)`` (length ``N``).
    coupling : numpy.ndarray
        The initial coupling ``K(0)`` (shape ``(N, N)``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    plasticity_rate : float
        The Hebbian plasticity rate ``ε`` (``≥ 0``).
    dt : float
        The RK4 step (``> 0``).
    n_steps : int
        The number of RK4 steps (``≥ 1``).
    objective : callable
        The terminal objective ``L(θ_N, K_N) → float``.
    objective_grad : callable
        Its gradient ``(θ_N, K_N) → (∂L/∂θ_N, ∂L/∂K_N)`` with shapes ``(N,)`` and ``(N, N)``.

    Returns
    -------
    tuple
        ``(value, AdaptiveGradients)``.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound, or ``objective_grad`` returns a
        cotangent of the wrong shape.
    """
    theta_final, coupling_final, sensitivity = adaptive_state_sensitivity(
        phases, coupling, omega, plasticity_rate=plasticity_rate, dt=dt, n_steps=n_steps
    )
    count = int(theta_final.size)
    value = float(objective(theta_final, coupling_final))
    grad_theta_n, grad_coupling_n = objective_grad(theta_final, coupling_final)
    grad_theta_n = np.ascontiguousarray(grad_theta_n, dtype=np.float64)
    grad_coupling_n = np.ascontiguousarray(grad_coupling_n, dtype=np.float64)
    if grad_theta_n.shape != (count,) or grad_coupling_n.shape != (count, count):
        raise ValueError(
            f"objective_grad must return a ({count},) phase cotangent and a "
            f"({count}, {count}) coupling cotangent, got {grad_theta_n.shape} and "
            f"{grad_coupling_n.shape}"
        )

    cotangent = np.concatenate([grad_theta_n, grad_coupling_n.reshape(-1)])
    flat = cotangent @ sensitivity  # ∂L/∂p over the channel layout

    coupling_start = count
    omega_start = count + count * count
    eps_index = omega_start + count
    return value, AdaptiveGradients(
        initial_phases=np.ascontiguousarray(flat[:count], dtype=np.float64),
        initial_coupling=np.ascontiguousarray(
            flat[coupling_start:omega_start].reshape(count, count), dtype=np.float64
        ),
        omega=np.ascontiguousarray(flat[omega_start:eps_index], dtype=np.float64),
        plasticity_rate=float(flat[eps_index]),
    )


__all__ = [
    "AdaptiveGradients",
    "AdaptiveTerminalObjective",
    "AdaptiveTerminalObjectiveGrad",
    "adaptive_state_sensitivity",
    "adaptive_terminal_value_and_grad",
]
