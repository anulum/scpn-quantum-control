# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Independent JAX reference forwards for the differentiable Kuramoto integrators
r"""Independent JAX reference forwards for the differentiable Kuramoto integrators.

The differentiable Kuramoto integrators in
:mod:`scpn_quantum_control.accel.diff_kuramoto_euler`,
:mod:`~scpn_quantum_control.accel.diff_kuramoto_rk4`,
:mod:`~scpn_quantum_control.accel.diff_kuramoto_dopri`,
:mod:`~scpn_quantum_control.accel.diff_kuramoto_adaptive`,
:mod:`~scpn_quantum_control.accel.diff_kuramoto_delayed`,
:mod:`~scpn_quantum_control.accel.diff_kuramoto_inertial`,
:mod:`~scpn_quantum_control.accel.diff_kuramoto_noisy` and
:mod:`~scpn_quantum_control.accel.kuramoto_ott_antonsen` all validate their
hand-derived reverse-mode adjoints and forward-mode sensitivities against a
central finite difference of the same integrator. A finite difference shares no
derivation with the analytic gradient, but it is a *numerical* witness limited to
roughly ``√eps`` precision, so a subtly wrong adjoint can still pass at the loose
tolerances a finite difference forces.

This module supplies the missing *second* witness: it re-implements each
integrator's forward map in JAX and lets JAX's reverse-mode autodiff differentiate
it. Crucially it re-implements only the **forward map** — the Dormand–Prince
Butcher tableau is imported from the production module so the two forwards are
byte-for-byte the same map — and leaves the differentiation entirely to JAX. The
gradient a test then compares is therefore produced by a mature, independent
differentiation engine, at (double-precision) machine tolerance rather than
finite-difference tolerance. Agreement between the hand-derived adjoint and the
JAX gradient is evidence that the adjoint is the exact transpose of the realised
forward map, not merely close to a coarse numerical estimate.

The module is a test helper, not shipped surface: JAX is an optional dependency,
so importing this module skips the collecting test when JAX is absent (matching
the framework-overlay lane that provisions ``jax[cpu]``). Every forward here
enables 64-bit JAX so the reference matches the float64 production integrators.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scpn_quantum_control.accel.diff_kuramoto_dopri import (
    _A as _DOPRI_A,
    _B5 as _DOPRI_B5,
    _STAGE_COUNT as _DOPRI_STAGE_COUNT,
)

# The production integrators run in float64; match that so the reference gradient is
# limited by the shared arithmetic, not by JAX's default single precision.
jax.config.update("jax_enable_x64", True)

# JAX arrays are dynamically typed at runtime; the helper takes and returns them as ``Any``.
Array = jax.Array


def _force(theta: Array, coupling: Array) -> Array:
    r"""Networked coupling force ``F_j = Σ_k K_jk sin(θ_k − θ_j)`` in JAX.

    Mirrors :func:`scpn_quantum_control.accel.networked_kuramoto.networked_kuramoto_force`
    with ``difference[j, k] = θ_k − θ_j``.
    """
    difference = theta[jnp.newaxis, :] - theta[:, jnp.newaxis]
    return (coupling * jnp.sin(difference)).sum(axis=1)


def euler_terminal(theta0: Array, omega: Array, coupling: Array, dt: float, n_steps: int) -> Array:
    r"""Terminal phase of the explicit-Euler Kuramoto forward, ``θ_{n+1} = θ_n + dt(ω + F)``."""
    theta = theta0
    for _ in range(n_steps):
        theta = theta + dt * (omega + _force(theta, coupling))
    return theta


def rk4_terminal(theta0: Array, omega: Array, coupling: Array, dt: float, n_steps: int) -> Array:
    r"""Terminal phase of the classical fourth-order Runge–Kutta Kuramoto forward."""
    theta = theta0
    for _ in range(n_steps):
        k1 = omega + _force(theta, coupling)
        k2 = omega + _force(theta + 0.5 * dt * k1, coupling)
        k3 = omega + _force(theta + 0.5 * dt * k2, coupling)
        k4 = omega + _force(theta + dt * k3, coupling)
        theta = theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return theta


def _dopri_stage_derivatives(
    theta: Array, omega: Array, coupling: Array, step: float
) -> list[Array]:
    """Return the seven Dormand–Prince stage derivatives, using the production tableau ``_A``."""
    derivatives: list[Array] = []
    for stage in range(_DOPRI_STAGE_COUNT):
        increment = theta
        for previous in range(stage):
            increment = increment + step * _DOPRI_A[stage][previous] * derivatives[previous]
        derivatives.append(omega + _force(increment, coupling))
    return derivatives


def dopri_terminal(theta0: Array, omega: Array, coupling: Array, steps: Sequence[float]) -> Array:
    r"""Terminal phase of the *fixed-grid* Dormand–Prince map over the realised ``steps``.

    Replays exactly the step sequence the adaptive forward realised — the same fixed grid the
    production discrete adjoint reverses over — so differentiating this map reproduces the
    adjoint. The Butcher tableau (``_A``, ``_B5``) is the production module's, so only the
    differentiation differs.
    """
    theta = theta0
    for raw_step in steps:
        step = float(raw_step)
        derivatives = _dopri_stage_derivatives(theta, omega, coupling, step)
        weighted = sum(
            float(_DOPRI_B5[stage]) * derivatives[stage] for stage in range(_DOPRI_STAGE_COUNT)
        )
        theta = theta + step * weighted
    return theta


def _adaptive_field(
    theta: Array, coupling: Array, omega: Array, plasticity_rate: float
) -> tuple[Array, Array]:
    r"""Joint adaptive field ``(θ̇, K̇)`` with Hebbian ``K̇ = ε(cos(θ_j − θ_i) − K)``."""
    phase_velocity = omega + _force(theta, coupling)
    difference = theta[jnp.newaxis, :] - theta[:, jnp.newaxis]
    coupling_velocity = plasticity_rate * (jnp.cos(difference) - coupling)
    return phase_velocity, coupling_velocity


def adaptive_terminal(
    theta0: Array,
    coupling0: Array,
    omega: Array,
    plasticity_rate: float,
    dt: float,
    n_steps: int,
) -> tuple[Array, Array]:
    r"""Terminal ``(θ_N, K_N)`` of the joint RK4 over the plastic-coupling adaptive Kuramoto model."""
    theta, coupling = theta0, coupling0
    for _ in range(n_steps):
        k1t, k1k = _adaptive_field(theta, coupling, omega, plasticity_rate)
        k2t, k2k = _adaptive_field(
            theta + 0.5 * dt * k1t, coupling + 0.5 * dt * k1k, omega, plasticity_rate
        )
        k3t, k3k = _adaptive_field(
            theta + 0.5 * dt * k2t, coupling + 0.5 * dt * k2k, omega, plasticity_rate
        )
        k4t, k4k = _adaptive_field(theta + dt * k3t, coupling + dt * k3k, omega, plasticity_rate)
        theta = theta + (dt / 6.0) * (k1t + 2.0 * k2t + 2.0 * k3t + k4t)
        coupling = coupling + (dt / 6.0) * (k1k + 2.0 * k2k + 2.0 * k3k + k4k)
    return theta, coupling


def _delayed_force(current: Array, lagged: Array, coupling: Array) -> Array:
    r"""Delayed networked force ``F_j = Σ_k K_jk sin(θ_k(t−τ) − θ_j(t))``."""
    difference = lagged[jnp.newaxis, :] - current[:, jnp.newaxis]
    return (coupling * jnp.sin(difference)).sum(axis=1)


def delayed_terminal(
    history: Array,
    omega: Array,
    coupling: Array,
    delay_steps: int,
    dt: float,
    n_steps: int,
) -> Array:
    r"""Terminal phase of the method-of-steps delayed Kuramoto forward.

    Mirrors :func:`scpn_quantum_control.accel.diff_kuramoto_delayed.delayed_phase_sensitivity`:
    the lagged term is read from the phase buffer (``lag_full`` for stage 1, the midpoint
    ``lag_half`` for stages 2 and 3, ``lag_next`` for stage 4).
    """
    buffer: list[Array] = [history[index] for index in range(delay_steps + 1)]
    for step in range(n_steps):
        theta = buffer[delay_steps + step]
        lag_full = buffer[step]
        lag_half = 0.5 * (buffer[step] + buffer[step + 1])
        lag_next = buffer[step + 1]
        k1 = omega + _delayed_force(theta, lag_full, coupling)
        k2 = omega + _delayed_force(theta + 0.5 * dt * k1, lag_half, coupling)
        k3 = omega + _delayed_force(theta + 0.5 * dt * k2, lag_half, coupling)
        k4 = omega + _delayed_force(theta + dt * k3, lag_next, coupling)
        buffer.append(theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
    return buffer[delay_steps + n_steps]


def _inertial_field(
    theta: Array, velocity: Array, omega: Array, coupling: Array, mass: float, damping: float
) -> tuple[Array, Array]:
    r"""Inertial (swing) field ``(θ̇, v̇)`` with ``v̇ = (ω + F(θ) − γv)/m``."""
    acceleration = (omega + _force(theta, coupling) - damping * velocity) / mass
    return velocity, acceleration


def inertial_terminal(
    theta0: Array,
    velocity0: Array,
    omega: Array,
    coupling: Array,
    mass: float,
    damping: float,
    dt: float,
    n_steps: int,
) -> tuple[Array, Array]:
    r"""Terminal ``(θ_N, v_N)`` of the second-order (inertial) Kuramoto RK4 forward."""
    theta, velocity = theta0, velocity0
    for _ in range(n_steps):
        k1t, k1v = _inertial_field(theta, velocity, omega, coupling, mass, damping)
        k2t, k2v = _inertial_field(
            theta + 0.5 * dt * k1t, velocity + 0.5 * dt * k1v, omega, coupling, mass, damping
        )
        k3t, k3v = _inertial_field(
            theta + 0.5 * dt * k2t, velocity + 0.5 * dt * k2v, omega, coupling, mass, damping
        )
        k4t, k4v = _inertial_field(
            theta + dt * k3t, velocity + dt * k3v, omega, coupling, mass, damping
        )
        theta = theta + (dt / 6.0) * (k1t + 2.0 * k2t + 2.0 * k3t + k4t)
        velocity = velocity + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return theta, velocity


def frozen_noise_increments(seed: int, n_steps: int, count: int) -> NDArray[np.float64]:
    r"""Regenerate the seeded Euler–Maruyama increments identically to the production integrator.

    Draws ``standard_normal(count)`` once per step from ``numpy.random.default_rng(seed)``, in the
    same order as :func:`scpn_quantum_control.accel.diff_kuramoto_noisy.noisy_phase_sensitivity`,
    so the differentiated path is exactly the frozen forward path.
    """
    generator = np.random.default_rng(seed)
    return np.stack([generator.standard_normal(count) for _ in range(n_steps)]).astype(np.float64)


def noisy_terminal(
    theta0: Array,
    omega: Array,
    coupling: Array,
    diffusion: float,
    increments: NDArray[np.float64],
    dt: float,
) -> Array:
    r"""Terminal phase of the frozen-noise Euler–Maruyama forward, ``θ_{n+1} = θ_n + (ω+F)dt + √(2Ddt)ξ_n``.

    ``increments`` are the frozen ξ from :func:`frozen_noise_increments`; ``diffusion`` is the
    differentiable channel, entering only through the additive noise scale ``√(2Ddt)``.
    """
    theta = theta0
    scale = jnp.sqrt(2.0 * diffusion * dt)
    for increment in increments:
        theta = theta + (omega + _force(theta, coupling)) * dt + scale * jnp.asarray(increment)
    return theta


def _ott_antonsen_field(
    x: Array, y: Array, coupling: float, half_width: float, centre: float
) -> tuple[Array, Array]:
    r"""Real-coordinate Ott–Antonsen field ``(ẋ, ẏ)`` for ``ż = (iω₀ − Δ + K/2)z − (K/2)|z|²z``."""
    linear = 0.5 * coupling - half_width
    radius_squared = x * x + y * y
    cubic = 0.5 * coupling * radius_squared
    dx = linear * x - centre * y - cubic * x
    dy = linear * y + centre * x - cubic * y
    return dx, dy


def ott_antonsen_terminal_radius(
    x0: float,
    y0: float,
    coupling: float,
    half_width: float,
    centre: float,
    dt: float,
    n_steps: int,
) -> Array:
    r"""Terminal order-parameter modulus ``r(T) = |z(T)|`` of the reduced Ott–Antonsen RK4 flow.

    Integrates the real ``(x, y)`` coordinates exactly as
    :func:`scpn_quantum_control.accel.kuramoto_ott_antonsen.ott_antonsen_terminal_order_parameter_value_and_grad`,
    so ``jax.grad`` with respect to ``(coupling, half_width)`` is an independent witness for
    ``(∂r/∂K, ∂r/∂Δ)``.
    """
    x = jnp.asarray(x0, dtype=jnp.float64)
    y = jnp.asarray(y0, dtype=jnp.float64)
    for _ in range(n_steps):
        k1x, k1y = _ott_antonsen_field(x, y, coupling, half_width, centre)
        k2x, k2y = _ott_antonsen_field(
            x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, coupling, half_width, centre
        )
        k3x, k3y = _ott_antonsen_field(
            x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, coupling, half_width, centre
        )
        k4x, k4y = _ott_antonsen_field(x + dt * k3x, y + dt * k3y, coupling, half_width, centre)
        x = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    return jnp.sqrt(x * x + y * y)
