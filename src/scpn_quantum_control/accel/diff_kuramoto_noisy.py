# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable noisy (stochastic) Kuramoto integrator
r"""Differentiable noisy Kuramoto integrator via the pathwise (fixed-noise) sensitivity.

The noisy Kuramoto Langevin model ``dθ_j = (ω_j + F_j(θ)) dt + √(2D)\,dW_j`` integrated by
:func:`~scpn_quantum_control.accel.kuramoto_noisy.integrate_noisy_kuramoto` (seeded
Euler–Maruyama) carried no gradient path, so a stochastic-control or noise-calibration objective
could not be optimised over the initial state, the frequencies, the coupling, or the noise
intensity. This module closes that gap for the **networked** model.

Method — pathwise (reparameterised) forward-mode sensitivity
------------------------------------------------------------
For a *fixed* Brownian path the Euler–Maruyama recursion
``θ_{n+1} = θ_n + (ω + F(θ_n)) dt + √(2 D dt)\,ξ_n`` is a deterministic map of its inputs, so the
pathwise derivative is well defined and finite-difference checkable. Crucially the diffusion here
is **additive** (the noise coefficient ``√(2D)`` does not depend on the state), so the Itô and
Stratonovich interpretations coincide and the stochastic adjoint carries no Itô–Stratonovich
correction term — the gradient is the ordinary derivative of the discrete recursion at the frozen
noise realisation. The sensitivity matrix ``S = ∂θ/∂p`` is propagated alongside the state with the
networked Jacobian and the explicit per-step parameter injection
(``∂/∂ω = dt\,I``, ``∂F_p/∂K_{pq} = dt\sin(θ_q − θ_p)``, ``∂/∂D = √(2dt)\,ξ_n / (2√D)``), and the
identical noise increments are regenerated from the same ``seed`` so the differentiated path is
exactly the forward path. The module adds no compute kernel.

The ``D`` channel requires ``D > 0`` (the noise-scale derivative ``∝ 1/√D`` is singular at the
deterministic limit); the phase, frequency and coupling channels are valid for any ``D ≥ 0``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .networked_kuramoto import networked_kuramoto_force, networked_kuramoto_jacobian

#: A terminal objective ``L(θ_N)`` on the final phases of a fixed-noise run.
NoisyTerminalObjective = Callable[[NDArray[np.float64]], float]

#: The gradient of a :data:`NoisyTerminalObjective`, returning ``∂L/∂θ_N``.
NoisyTerminalObjectiveGrad = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class NoisyGradients:
    """Pathwise gradients of a terminal objective through the noisy Kuramoto integrator.

    Attributes
    ----------
    initial_phases : numpy.ndarray
        ``∂L/∂θ_0`` (length ``N``).
    omega : numpy.ndarray
        ``∂L/∂ω`` (length ``N``).
    coupling : numpy.ndarray
        ``∂L/∂K`` (shape ``(N, N)``).
    diffusion : float
        ``∂L/∂D`` for the frozen noise realisation.
    """

    initial_phases: NDArray[np.float64]
    omega: NDArray[np.float64]
    coupling: NDArray[np.float64]
    diffusion: float


def _validate(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    diffusion: float,
    dt: float,
    n_steps: int,
) -> int:
    """Validate the noisy differentiable problem and return the oscillator count."""
    if omega.ndim != 1 or omega.size < 1:
        raise ValueError("omega must be a non-empty one-dimensional array")
    count = int(omega.size)
    if phases.shape != (count,):
        raise ValueError(f"phases must have shape {(count,)}, got {phases.shape}")
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape {(count, count)}, got {coupling.shape}")
    if diffusion <= 0.0:
        raise ValueError(f"diffusion must be positive for the gradient, got {diffusion}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    return count


def noisy_phase_sensitivity(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    diffusion: float,
    dt: float,
    n_steps: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Integrate the noisy Euler–Maruyama with pathwise forward-mode sensitivity.

    Regenerates the seeded Wiener increments identically to
    :func:`~scpn_quantum_control.accel.kuramoto_noisy.integrate_noisy_kuramoto`, so the
    differentiated path is exactly the forward path, and returns the final phases ``θ_N`` and the
    sensitivity matrix ``S_N = ∂θ_N/∂p`` of shape ``(N, P)`` for the channel layout
    ``[θ₀ (N), ω (N), K (N²), D]`` (so ``P = 2N + N² + 1``).

    Parameters
    ----------
    phases : numpy.ndarray
        The initial phases ``θ(0)`` (length ``N``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` networked coupling matrix ``K``.
    diffusion : float
        The diffusion / noise intensity ``D`` (``> 0`` for the gradient).
    dt : float
        The Euler–Maruyama step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``).
    seed : int
        The noise seed; must match the forward run for the pathwise derivative to apply.

    Returns
    -------
    tuple of numpy.ndarray
        ``(theta_final, sensitivity)``.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    theta = np.ascontiguousarray(phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    coupling_matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = _validate(theta, frequencies, coupling_matrix, diffusion, dt, n_steps)
    width = 2 * count + count * count + 1

    omega_start = count
    coupling_start = 2 * count
    diffusion_index = coupling_start + count * count

    sensitivity: NDArray[np.float64] = np.zeros((count, width), dtype=np.float64)
    sensitivity[:, :count] = np.eye(count)  # ∂θ₀/∂θ₀ = I

    generator = np.random.default_rng(seed)
    scale = np.sqrt(2.0 * diffusion * dt)
    scale_derivative = np.sqrt(2.0 * dt) / (2.0 * np.sqrt(diffusion))
    identity = np.eye(count, dtype=np.float64)

    for _ in range(n_steps):
        increment = generator.standard_normal(count)
        jacobian = networked_kuramoto_jacobian(theta, coupling_matrix)

        injection = np.zeros((count, width), dtype=np.float64)
        injection[:, omega_start:coupling_start] = dt * identity
        phase_delta = theta[np.newaxis, :] - theta[:, np.newaxis]
        sin_delta = dt * np.sin(phase_delta)
        for p in range(count):
            base = coupling_start + p * count
            injection[p, base : base + count] = sin_delta[p]
        injection[:, diffusion_index] = scale_derivative * increment

        sensitivity = sensitivity + dt * (jacobian @ sensitivity) + injection
        drift = (frequencies + networked_kuramoto_force(theta, coupling_matrix)) * dt
        theta = np.asarray(theta + drift + scale * increment, dtype=np.float64)

    return theta, sensitivity


def noisy_terminal_value_and_grad(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    diffusion: float,
    dt: float,
    n_steps: int,
    seed: int,
    objective: NoisyTerminalObjective,
    objective_grad: NoisyTerminalObjectiveGrad,
) -> tuple[float, NoisyGradients]:
    r"""Differentiate a terminal objective through the noisy Kuramoto integrator (fixed noise).

    Evaluates ``L(θ_N)`` for the frozen ``seed`` noise realisation and returns its pathwise
    gradients with respect to ``θ_0, ω, K, D`` by contracting the final-phase cotangent
    ``∂L/∂θ_N`` with the pathwise sensitivity ``S_N``. The returned gradient is the derivative for
    *this* noise realisation; average over seeds for an expectation gradient.

    Parameters
    ----------
    phases : numpy.ndarray
        The initial phases ``θ(0)`` (length ``N``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` networked coupling matrix ``K``.
    diffusion : float
        The diffusion / noise intensity ``D`` (``> 0``).
    dt : float
        The Euler–Maruyama step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``).
    seed : int
        The noise seed; must match the forward run.
    objective : callable
        The terminal objective ``L(θ_N) → float``.
    objective_grad : callable
        Its gradient ``θ_N → ∂L/∂θ_N`` (length ``N``).

    Returns
    -------
    tuple
        ``(value, NoisyGradients)``.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound, or ``objective_grad`` returns a
        cotangent of the wrong shape.
    """
    theta_final, sensitivity = noisy_phase_sensitivity(
        phases, omega, coupling, diffusion=diffusion, dt=dt, n_steps=n_steps, seed=seed
    )
    count = int(theta_final.size)
    value = float(objective(theta_final))
    cotangent = np.ascontiguousarray(objective_grad(theta_final), dtype=np.float64)
    if cotangent.shape != (count,):
        raise ValueError(
            f"objective_grad must return a ({count},) cotangent, got {cotangent.shape}"
        )

    flat = cotangent @ sensitivity  # ∂L/∂p over the channel layout
    omega_start = count
    coupling_start = 2 * count
    diffusion_index = coupling_start + count * count
    return value, NoisyGradients(
        initial_phases=np.ascontiguousarray(flat[:count], dtype=np.float64),
        omega=np.ascontiguousarray(flat[omega_start:coupling_start], dtype=np.float64),
        coupling=np.ascontiguousarray(
            flat[coupling_start:diffusion_index].reshape(count, count), dtype=np.float64
        ),
        diffusion=float(flat[diffusion_index]),
    )


__all__ = [
    "NoisyGradients",
    "NoisyTerminalObjective",
    "NoisyTerminalObjectiveGrad",
    "noisy_phase_sensitivity",
    "noisy_terminal_value_and_grad",
]
