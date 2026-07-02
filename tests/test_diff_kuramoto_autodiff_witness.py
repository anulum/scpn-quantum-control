# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Independent-autodiff witness for the differentiable Kuramoto adjoints
r"""Independent-autodiff witness tests for every differentiable Kuramoto gradient path.

Every differentiable Kuramoto integrator validates its hand-derived reverse-mode adjoint (Euler,
RK4, Dormand–Prince) or forward-mode sensitivity (adaptive, delayed, inertial, noisy, Ott–Antonsen)
against a central finite difference of the same integrator. A finite difference is an independent
*numerical* witness, but only to roughly ``√eps`` precision, so the production suites assert those
gradients at loose tolerances (``1e-5``–``1e-7``).

These tests add the missing *second* witness: they differentiate the identical forward map with
JAX reverse-mode autodiff — a mature, independent differentiation engine — and require the
hand-derived gradient to agree with the JAX gradient at (double-precision) machine tolerance, three
to five orders of magnitude tighter than the finite-difference checks. Agreement is evidence the
adjoint is the exact transpose of the realised forward map, not merely close to a coarse estimate.
The forward maps in :mod:`tests.kuramoto_autodiff_witness` re-implement only the integrator step
(the Dormand–Prince tableau is imported from production), so the sole quantity under test is the
gradient.

JAX is an optional dependency; importing the witness module skips this file when JAX is absent, and
the ``differentiable-frameworks`` CI lane provisions ``jax[cpu]`` so the witness runs there.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scpn_quantum_control.accel.diff_kuramoto_adaptive import adaptive_terminal_value_and_grad
from scpn_quantum_control.accel.diff_kuramoto_delayed import delayed_terminal_value_and_grad
from scpn_quantum_control.accel.diff_kuramoto_dopri import (
    kuramoto_dopri_trajectory,
    kuramoto_dopri_vjp,
)
from scpn_quantum_control.accel.diff_kuramoto_euler import (
    kuramoto_euler_trajectory,
    kuramoto_euler_vjp,
)
from scpn_quantum_control.accel.diff_kuramoto_inertial import inertial_terminal_value_and_grad
from scpn_quantum_control.accel.diff_kuramoto_noisy import noisy_terminal_value_and_grad
from scpn_quantum_control.accel.diff_kuramoto_rk4 import (
    kuramoto_rk4_trajectory,
    kuramoto_rk4_vjp,
)
from scpn_quantum_control.accel.kuramoto_ott_antonsen import (
    ott_antonsen_terminal_order_parameter_value_and_grad,
)
from tests import kuramoto_autodiff_witness as witness

# Machine-tolerance parity: the JAX gradient and the hand-derived adjoint differentiate the
# byte-identical float64 forward map, so they agree far tighter than the finite-difference checks
# the production suites use (which sit at 1e-5..1e-7).
_ATOL = 1e-9
_RTOL = 1e-7


def _problem(count: int = 4, seed: int = 20260702) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a reproducible ``(theta0, omega, coupling)`` networked-Kuramoto problem."""
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-np.pi, np.pi, count)
    omega = rng.normal(0.0, 1.0, count)
    coupling = rng.normal(0.0, 0.6, (count, count))
    np.fill_diagonal(coupling, 0.0)
    return theta0, omega, coupling


def test_euler_vjp_matches_jax_reverse_mode_at_machine_tolerance() -> None:
    """The Euler adjoint equals the JAX reverse-mode gradient of the same forward map."""
    theta0, omega, coupling = _problem()
    dt, n_steps = 0.05, 8
    rng = np.random.default_rng(1)
    cotangent = rng.normal(0.0, 1.0, theta0.size)

    trajectory = kuramoto_euler_trajectory(theta0, omega, coupling, dt, n_steps)
    grad_theta0, grad_omega, grad_coupling = kuramoto_euler_vjp(
        trajectory, coupling, dt, cotangent
    )

    # Independent forward: the witness terminal phase reproduces the production trajectory endpoint.
    terminal = np.asarray(witness.euler_terminal(theta0, omega, coupling, dt, n_steps))
    assert np.allclose(terminal, trajectory[-1], atol=1e-11, rtol=0.0)

    def loss(th0: jax.Array, om: jax.Array, cpl: jax.Array) -> jax.Array:
        return jnp.dot(jnp.asarray(cotangent), witness.euler_terminal(th0, om, cpl, dt, n_steps))

    jax_grads = jax.grad(loss, argnums=(0, 1, 2))(
        jnp.asarray(theta0), jnp.asarray(omega), jnp.asarray(coupling)
    )
    # First-order Euler over eight steps is nearly bit-exact between the two engines.
    assert np.allclose(grad_theta0, np.asarray(jax_grads[0]), atol=1e-10, rtol=1e-8)
    assert np.allclose(grad_omega, np.asarray(jax_grads[1]), atol=1e-10, rtol=1e-8)
    assert np.allclose(grad_coupling, np.asarray(jax_grads[2]), atol=1e-10, rtol=1e-8)


def test_rk4_vjp_matches_jax_reverse_mode() -> None:
    """The RK4 four-stage adjoint equals the JAX reverse-mode gradient of the same forward map."""
    theta0, omega, coupling = _problem(seed=7)
    dt, n_steps = 0.05, 8
    rng = np.random.default_rng(2)
    cotangent = rng.normal(0.0, 1.0, theta0.size)

    trajectory = kuramoto_rk4_trajectory(theta0, omega, coupling, dt, n_steps)
    grad_theta0, grad_omega, grad_coupling = kuramoto_rk4_vjp(
        trajectory, omega, coupling, dt, cotangent
    )

    terminal = np.asarray(witness.rk4_terminal(theta0, omega, coupling, dt, n_steps))
    assert np.allclose(terminal, trajectory[-1], atol=1e-11, rtol=0.0)

    def loss(th0: jax.Array, om: jax.Array, cpl: jax.Array) -> jax.Array:
        return jnp.dot(jnp.asarray(cotangent), witness.rk4_terminal(th0, om, cpl, dt, n_steps))

    jax_grads = jax.grad(loss, argnums=(0, 1, 2))(
        jnp.asarray(theta0), jnp.asarray(omega), jnp.asarray(coupling)
    )
    assert np.allclose(grad_theta0, np.asarray(jax_grads[0]), atol=_ATOL, rtol=_RTOL)
    assert np.allclose(grad_omega, np.asarray(jax_grads[1]), atol=_ATOL, rtol=_RTOL)
    assert np.allclose(grad_coupling, np.asarray(jax_grads[2]), atol=_ATOL, rtol=_RTOL)


def test_dopri_fixed_grid_vjp_matches_jax_reverse_mode() -> None:
    """The Dormand–Prince fixed-grid adjoint equals the JAX gradient over the realised step grid."""
    theta0, omega, coupling = _problem(seed=11)
    rng = np.random.default_rng(3)
    cotangent = rng.normal(0.0, 1.0, theta0.size)

    trajectory = kuramoto_dopri_trajectory(
        theta0, omega, coupling, t_end=0.5, rtol=1e-6, atol=1e-9
    )
    grad_theta0, grad_omega, grad_coupling = kuramoto_dopri_vjp(
        trajectory.phases, trajectory.steps, omega, coupling, cotangent
    )
    assert trajectory.steps.size >= 2  # a genuinely multi-step realised grid

    terminal = np.asarray(witness.dopri_terminal(theta0, omega, coupling, trajectory.steps))
    assert np.allclose(terminal, trajectory.terminal_phases, atol=1e-10, rtol=0.0)

    def loss(th0: jax.Array, om: jax.Array, cpl: jax.Array) -> jax.Array:
        return jnp.dot(
            jnp.asarray(cotangent), witness.dopri_terminal(th0, om, cpl, trajectory.steps)
        )

    jax_grads = jax.grad(loss, argnums=(0, 1, 2))(
        jnp.asarray(theta0), jnp.asarray(omega), jnp.asarray(coupling)
    )
    assert np.allclose(grad_theta0, np.asarray(jax_grads[0]), atol=_ATOL, rtol=_RTOL)
    assert np.allclose(grad_omega, np.asarray(jax_grads[1]), atol=_ATOL, rtol=_RTOL)
    assert np.allclose(grad_coupling, np.asarray(jax_grads[2]), atol=_ATOL, rtol=_RTOL)


def test_adaptive_sensitivity_matches_jax_reverse_mode() -> None:
    """The adaptive forward-mode sensitivity matches the JAX gradient over ``(θ₀, K₀, ω, ε)``."""
    theta0, omega, coupling = _problem(seed=5)
    dt, n_steps, plasticity_rate = 0.05, 6, 0.4

    def objective(theta: np.ndarray, matrix: np.ndarray) -> float:
        return float(np.sum(np.sin(theta)) + 0.5 * np.sum(matrix**2))

    def objective_grad(theta: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.cos(theta), matrix

    value, grads = adaptive_terminal_value_and_grad(
        theta0,
        coupling,
        omega,
        plasticity_rate=plasticity_rate,
        dt=dt,
        n_steps=n_steps,
        objective=objective,
        objective_grad=objective_grad,
    )

    def loss(th0: jax.Array, cpl0: jax.Array, om: jax.Array, eps: jax.Array) -> jax.Array:
        theta_n, coupling_n = witness.adaptive_terminal(th0, cpl0, om, eps, dt, n_steps)
        return jnp.sum(jnp.sin(theta_n)) + 0.5 * jnp.sum(coupling_n**2)

    jax_value, jax_grads = jax.value_and_grad(loss, argnums=(0, 1, 2, 3))(
        jnp.asarray(theta0),
        jnp.asarray(coupling),
        jnp.asarray(omega),
        jnp.asarray(plasticity_rate, dtype=jnp.float64),
    )
    assert value == pytest.approx(float(jax_value), abs=1e-10)
    assert np.allclose(grads.initial_phases, np.asarray(jax_grads[0]), atol=_ATOL, rtol=_RTOL)
    assert np.allclose(grads.initial_coupling, np.asarray(jax_grads[1]), atol=_ATOL, rtol=_RTOL)
    assert np.allclose(grads.omega, np.asarray(jax_grads[2]), atol=_ATOL, rtol=_RTOL)
    assert grads.plasticity_rate == pytest.approx(float(jax_grads[3]), abs=_ATOL, rel=_RTOL)


def test_delayed_method_of_steps_sensitivity_matches_jax_reverse_mode() -> None:
    """The delayed method-of-steps sensitivity matches the JAX gradient over history, ω and K."""
    _, omega, coupling = _problem(seed=13)
    dt, n_steps, delay_steps = 0.05, 6, 3
    delay = delay_steps * dt
    rng = np.random.default_rng(4)
    history = rng.uniform(-np.pi, np.pi, (delay_steps + 1, omega.size))

    def objective(theta: np.ndarray) -> float:
        return float(np.sum(np.sin(theta)))

    def objective_grad(theta: np.ndarray) -> np.ndarray:
        return np.cos(theta)

    value, grads = delayed_terminal_value_and_grad(
        history,
        omega,
        coupling,
        delay=delay,
        dt=dt,
        n_steps=n_steps,
        objective=objective,
        objective_grad=objective_grad,
    )

    def loss(hist: jax.Array, om: jax.Array, cpl: jax.Array) -> jax.Array:
        theta_n = witness.delayed_terminal(hist, om, cpl, delay_steps, dt, n_steps)
        return jnp.sum(jnp.sin(theta_n))

    jax_value, jax_grads = jax.value_and_grad(loss, argnums=(0, 1, 2))(
        jnp.asarray(history), jnp.asarray(omega), jnp.asarray(coupling)
    )
    assert value == pytest.approx(float(jax_value), abs=1e-10)
    assert np.allclose(grads.initial_history, np.asarray(jax_grads[0]), atol=_ATOL, rtol=_RTOL)
    assert np.allclose(grads.omega, np.asarray(jax_grads[1]), atol=_ATOL, rtol=_RTOL)
    assert np.allclose(grads.coupling, np.asarray(jax_grads[2]), atol=_ATOL, rtol=_RTOL)


def test_inertial_sensitivity_matches_jax_reverse_mode() -> None:
    """The inertial (swing) sensitivity matches the JAX gradient over ``(θ₀, v₀, ω, K, m, γ)``."""
    theta0, omega, coupling = _problem(seed=17)
    rng = np.random.default_rng(6)
    velocity0 = rng.normal(0.0, 0.3, theta0.size)
    dt, n_steps, mass, damping = 0.05, 6, 1.3, 0.4

    def objective(theta: np.ndarray, velocity: np.ndarray) -> float:
        return float(np.sum(np.sin(theta)) + 0.5 * np.sum(velocity**2))

    def objective_grad(theta: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.cos(theta), velocity

    value, grads = inertial_terminal_value_and_grad(
        theta0,
        velocity0,
        omega,
        coupling,
        mass,
        damping=damping,
        dt=dt,
        n_steps=n_steps,
        objective=objective,
        objective_grad=objective_grad,
    )

    def loss(
        th0: jax.Array,
        vel0: jax.Array,
        om: jax.Array,
        cpl: jax.Array,
        mss: jax.Array,
        dmp: jax.Array,
    ) -> jax.Array:
        theta_n, velocity_n = witness.inertial_terminal(th0, vel0, om, cpl, mss, dmp, dt, n_steps)
        return jnp.sum(jnp.sin(theta_n)) + 0.5 * jnp.sum(velocity_n**2)

    jax_value, jax_grads = jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5))(
        jnp.asarray(theta0),
        jnp.asarray(velocity0),
        jnp.asarray(omega),
        jnp.asarray(coupling),
        jnp.asarray(mass, dtype=jnp.float64),
        jnp.asarray(damping, dtype=jnp.float64),
    )
    assert value == pytest.approx(float(jax_value), abs=1e-10)
    assert np.allclose(grads.initial_phases, np.asarray(jax_grads[0]), atol=_ATOL, rtol=_RTOL)
    assert np.allclose(grads.initial_velocities, np.asarray(jax_grads[1]), atol=_ATOL, rtol=_RTOL)
    assert np.allclose(grads.omega, np.asarray(jax_grads[2]), atol=_ATOL, rtol=_RTOL)
    assert np.allclose(grads.coupling, np.asarray(jax_grads[3]), atol=_ATOL, rtol=_RTOL)
    assert grads.mass == pytest.approx(float(jax_grads[4]), abs=_ATOL, rel=_RTOL)
    assert grads.damping == pytest.approx(float(jax_grads[5]), abs=_ATOL, rel=_RTOL)


def test_noisy_pathwise_sensitivity_matches_jax_reverse_mode() -> None:
    """The noisy pathwise (frozen-noise) sensitivity matches the JAX gradient over ``(θ₀, ω, K, D)``."""
    theta0, omega, coupling = _problem(seed=23)
    dt, n_steps, diffusion, seed = 0.05, 6, 0.3, 424242
    increments = witness.frozen_noise_increments(seed, n_steps, theta0.size)

    def objective(theta: np.ndarray) -> float:
        return float(np.sum(np.sin(theta)))

    def objective_grad(theta: np.ndarray) -> np.ndarray:
        return np.cos(theta)

    value, grads = noisy_terminal_value_and_grad(
        theta0,
        omega,
        coupling,
        diffusion=diffusion,
        dt=dt,
        n_steps=n_steps,
        seed=seed,
        objective=objective,
        objective_grad=objective_grad,
    )

    def loss(th0: jax.Array, om: jax.Array, cpl: jax.Array, diff: jax.Array) -> jax.Array:
        theta_n = witness.noisy_terminal(th0, om, cpl, diff, increments, dt)
        return jnp.sum(jnp.sin(theta_n))

    jax_value, jax_grads = jax.value_and_grad(loss, argnums=(0, 1, 2, 3))(
        jnp.asarray(theta0),
        jnp.asarray(omega),
        jnp.asarray(coupling),
        jnp.asarray(diffusion, dtype=jnp.float64),
    )
    # Matching value confirms the frozen-noise path is regenerated identically.
    assert value == pytest.approx(float(jax_value), abs=1e-10)
    assert np.allclose(grads.initial_phases, np.asarray(jax_grads[0]), atol=_ATOL, rtol=_RTOL)
    assert np.allclose(grads.omega, np.asarray(jax_grads[1]), atol=_ATOL, rtol=_RTOL)
    assert np.allclose(grads.coupling, np.asarray(jax_grads[2]), atol=_ATOL, rtol=_RTOL)
    assert grads.diffusion == pytest.approx(float(jax_grads[3]), abs=_ATOL, rel=_RTOL)


def test_ott_antonsen_gradient_matches_jax_reverse_mode() -> None:
    """The Ott–Antonsen terminal-order-parameter gradient matches the JAX gradient over ``(K, Δ)``."""
    z0 = complex(0.4, 0.2)
    coupling, half_width, centre = 1.6, 0.5, 0.3
    dt, n_steps = 0.05, 12

    radius, grad_coupling, grad_half_width = ott_antonsen_terminal_order_parameter_value_and_grad(
        z0, coupling, half_width, dt, n_steps, centre=centre
    )

    def terminal_radius(cpl: jax.Array, delta: jax.Array) -> jax.Array:
        return witness.ott_antonsen_terminal_radius(
            z0.real, z0.imag, cpl, delta, centre, dt, n_steps
        )

    jax_radius, jax_grads = jax.value_and_grad(terminal_radius, argnums=(0, 1))(
        jnp.asarray(coupling, dtype=jnp.float64), jnp.asarray(half_width, dtype=jnp.float64)
    )
    assert radius == pytest.approx(float(jax_radius), abs=1e-11)
    assert grad_coupling == pytest.approx(float(jax_grads[0]), abs=_ATOL, rel=_RTOL)
    assert grad_half_width == pytest.approx(float(jax_grads[1]), abs=_ATOL, rel=_RTOL)
