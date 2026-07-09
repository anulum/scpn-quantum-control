# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the JAX autodiff Kuramoto RK4 tier
"""Contract tests for the JAX autodiff accelerator tier of the networked-Kuramoto RK4 integrator.

These exercise real JAX on the accelerator JAX selected (a CUDA GPU when present), so the module skips
without the optional ``[jax]`` extra. They pin the two claims the tier is built on — the forward is
faithful to the production Rust integrator at 64-bit precision, and the autodiff gradient matches the
hand-derived reverse-mode adjoint to machine precision — plus the residency, fallback, validation and
determinism contracts.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel import jax_kuramoto as jk
from oscillatools.accel import jax_kuramoto_integrators as ji
from oscillatools.accel.diff_kuramoto_dopri import kuramoto_dopri_trajectory
from oscillatools.accel.diff_kuramoto_euler import kuramoto_euler_trajectory
from oscillatools.accel.diff_kuramoto_rk4 import (
    kuramoto_rk4_trajectory,
    kuramoto_rk4_vjp,
)
from oscillatools.accel.networked_inertial import networked_inertial_trajectory
from oscillatools.accel.networked_noisy import networked_noisy_trajectory
from oscillatools.accel.networked_symplectic_inertial import (
    networked_symplectic_inertial_trajectory,
)

pytest.importorskip("jax")

_DT = 0.05
_STEPS = 40


def _network(
    n: int, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    omega = np.asarray(rng.normal(0.0, 0.7, size=n), dtype=np.float64)
    coupling = np.full((n, n), 1.8 / n, dtype=np.float64)
    np.fill_diagonal(coupling, 0.0)
    theta0 = np.asarray(rng.uniform(0.0, 2.0 * np.pi, size=n), dtype=np.float64)
    return theta0, omega, coupling


@pytest.mark.parametrize("n", [6, 32, 64])
def test_forward_matches_rust_tier_at_x64(n: int) -> None:
    theta0, omega, coupling = _network(n, seed=n)
    reference = kuramoto_rk4_trajectory(theta0, omega, coupling, _DT, _STEPS)
    jax_trajectory = jk.jax_kuramoto_rk4_trajectory(theta0, omega, coupling, _DT, _STEPS)
    assert jax_trajectory.shape == (_STEPS + 1, n)
    assert jax_trajectory.dtype == np.float64
    # 64-bit JAX matches the production integrator; parity is asserted under a tolerance because GPU
    # reduction ordering need not be bit-identical to NumPy (observed here at machine precision).
    assert np.max(np.abs(jax_trajectory - reference)) < 1e-11


def test_euler_forward_matches_production_tier() -> None:
    theta0, omega, coupling = _network(20, seed=41)
    reference = kuramoto_euler_trajectory(theta0, omega, coupling, _DT, 25)
    jax_trajectory = ji.jax_kuramoto_euler_trajectory(theta0, omega, coupling, _DT, 25)
    assert jax_trajectory.shape == reference.shape
    assert np.max(np.abs(jax_trajectory - reference)) < 1e-11


def test_dopri_forward_matches_production_python_grid() -> None:
    theta0, omega, coupling = _network(10, seed=42)
    reference = kuramoto_dopri_trajectory(
        theta0,
        omega,
        coupling,
        t_end=0.35,
        rtol=1e-7,
        atol=1e-10,
        first_step=0.035,
        max_steps=200,
    )
    jax_trajectory = ji.jax_kuramoto_dopri_trajectory(
        theta0,
        omega,
        coupling,
        t_end=0.35,
        rtol=1e-7,
        atol=1e-10,
        first_step=0.035,
        max_steps=200,
    )
    assert np.allclose(jax_trajectory.times, reference.times, atol=1e-13)
    assert np.allclose(jax_trajectory.steps, reference.steps, atol=1e-13)
    assert np.max(np.abs(jax_trajectory.phases - reference.phases)) < 1e-10


def test_networked_inertial_forward_matches_production_tier() -> None:
    theta0, omega, coupling = _network(12, seed=43)
    velocities = np.asarray(np.linspace(-0.2, 0.2, theta0.size), dtype=np.float64)
    reference = networked_inertial_trajectory(
        theta0,
        velocities,
        omega,
        coupling,
        mass=1.7,
        damping=0.3,
        dt=0.02,
        n_steps=24,
    )
    jax_trajectory = ji.jax_networked_inertial_trajectory(
        theta0,
        velocities,
        omega,
        coupling,
        mass=1.7,
        damping=0.3,
        dt=0.02,
        n_steps=24,
    )
    assert np.array_equal(jax_trajectory.times, reference.times)
    assert np.max(np.abs(jax_trajectory.phases - reference.phases)) < 1e-10
    assert np.max(np.abs(jax_trajectory.velocities - reference.velocities)) < 1e-10


def test_networked_symplectic_forward_matches_production_tier() -> None:
    theta0, omega, coupling = _network(12, seed=44)
    velocities = np.asarray(np.linspace(0.15, -0.15, theta0.size), dtype=np.float64)
    reference = networked_symplectic_inertial_trajectory(
        theta0,
        velocities,
        omega,
        coupling,
        mass=1.2,
        damping=0.1,
        dt=0.015,
        n_steps=28,
    )
    jax_trajectory = ji.jax_networked_symplectic_inertial_trajectory(
        theta0,
        velocities,
        omega,
        coupling,
        mass=1.2,
        damping=0.1,
        dt=0.015,
        n_steps=28,
    )
    assert np.array_equal(jax_trajectory.times, reference.times)
    assert np.max(np.abs(jax_trajectory.phases - reference.phases)) < 1e-10
    assert np.max(np.abs(jax_trajectory.velocities - reference.velocities)) < 1e-10


def test_networked_noisy_forward_matches_production_tier_with_shared_seed() -> None:
    theta0, omega, coupling = _network(14, seed=45)
    reference = networked_noisy_trajectory(
        theta0,
        omega,
        coupling,
        diffusion=0.04,
        dt=0.01,
        n_steps=35,
        seed=4545,
        settle_steps=10,
    )
    jax_run = ji.jax_networked_noisy_trajectory(
        theta0,
        omega,
        coupling,
        diffusion=0.04,
        dt=0.01,
        n_steps=35,
        seed=4545,
        settle_steps=10,
    )
    assert np.max(np.abs(jax_run.terminal_phases - reference.terminal_phases)) < 1e-10
    assert (
        np.max(np.abs(jax_run.order_parameter_series - reference.order_parameter_series)) < 1e-10
    )
    assert abs(jax_run.mean_order_parameter - reference.mean_order_parameter) < 1e-12
    assert abs(jax_run.order_parameter_std - reference.order_parameter_std) < 1e-12


@pytest.mark.parametrize("n", [6, 32, 64])
def test_gradient_matches_hand_derived_vjp(n: int) -> None:
    theta0, omega, coupling = _network(n, seed=n + 100)
    trajectory = kuramoto_rk4_trajectory(theta0, omega, coupling, _DT, _STEPS)
    rng = np.random.default_rng(n)
    cotangent = rng.normal(0.0, 1.0, size=n)
    hand = kuramoto_rk4_vjp(trajectory, omega, coupling, _DT, cotangent)
    autodiff = jk.jax_kuramoto_rk4_gradient(theta0, omega, coupling, _DT, _STEPS, cotangent)
    for hand_grad, autodiff_grad in zip(hand, autodiff, strict=True):
        assert hand_grad.shape == autodiff_grad.shape
        assert np.max(np.abs(hand_grad - autodiff_grad)) < 1e-9


def test_trajectory_runs_on_the_jax_default_device() -> None:
    import jax

    theta0, omega, coupling = _network(16, seed=7)
    backend = jk._load_backend()
    jnp = backend.jnp
    result = backend.trajectory(
        jnp.asarray(theta0), jnp.asarray(omega), jnp.asarray(coupling), _DT, _STEPS
    )
    # the jitted solve lands on the accelerator JAX selected (a CUDA GPU when one is present)
    assert result.device == jax.devices()[0]


def test_n_steps_zero_returns_the_initial_state() -> None:
    theta0, omega, coupling = _network(8, seed=1)
    trajectory = jk.jax_kuramoto_rk4_trajectory(theta0, omega, coupling, _DT, 0)
    assert trajectory.shape == (1, 8)
    assert np.allclose(trajectory[0], theta0)


def test_forward_rejects_inconsistent_shapes() -> None:
    theta0, omega, coupling = _network(6, seed=2)
    with pytest.raises(ValueError, match="omega"):
        jk.jax_kuramoto_rk4_trajectory(theta0, omega[:-1], coupling, _DT, _STEPS)
    with pytest.raises(ValueError, match="coupling"):
        jk.jax_kuramoto_rk4_trajectory(theta0, omega, coupling[:-1], _DT, _STEPS)
    with pytest.raises(ValueError, match="n_steps"):
        jk.jax_kuramoto_rk4_trajectory(theta0, omega, coupling, _DT, -1)


def test_new_integrators_reject_invalid_inputs() -> None:
    theta0, omega, coupling = _network(6, seed=46)
    velocities = np.zeros_like(theta0)
    with pytest.raises(ValueError, match="omega"):
        ji.jax_kuramoto_euler_trajectory(theta0, omega[:-1], coupling, _DT, 4)
    with pytest.raises(ValueError, match="t_end"):
        ji.jax_kuramoto_dopri_trajectory(theta0, omega, coupling, t_end=0.0)
    with pytest.raises(ValueError, match="max_steps"):
        ji.jax_kuramoto_dopri_trajectory(theta0, omega, coupling, t_end=0.1, max_steps=0)
    with pytest.raises(ValueError, match="mass"):
        ji.jax_networked_inertial_trajectory(
            theta0, velocities, omega, coupling, mass=0.0, dt=0.01, n_steps=2
        )
    with pytest.raises(ValueError, match="damping"):
        ji.jax_networked_symplectic_inertial_trajectory(
            theta0, velocities, omega, coupling, mass=1.0, damping=-0.1, dt=0.01, n_steps=2
        )
    with pytest.raises(ValueError, match="settle_steps"):
        ji.jax_networked_noisy_trajectory(
            theta0,
            omega,
            coupling,
            diffusion=0.1,
            dt=0.01,
            n_steps=2,
            seed=1,
            settle_steps=3,
        )


def test_gradient_rejects_non_positive_steps_and_bad_cotangent() -> None:
    theta0, omega, coupling = _network(6, seed=3)
    cotangent = np.ones(6, dtype=np.float64)
    with pytest.raises(ValueError, match="n_steps must be positive"):
        jk.jax_kuramoto_rk4_gradient(theta0, omega, coupling, _DT, 0, cotangent)
    with pytest.raises(ValueError, match="cotangent"):
        jk.jax_kuramoto_rk4_gradient(theta0, omega, coupling, _DT, _STEPS, cotangent[:-1])


def test_missing_jax_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> tuple[object, object, object]:
        raise ImportError("the JAX Kuramoto tier requires JAX; install oscillatools[jax]")

    monkeypatch.setattr(jk, "_load_backend", _raise)
    theta0, omega, coupling = _network(6, seed=4)
    with pytest.raises(ImportError, match="oscillatools\\[jax\\]"):
        jk.jax_kuramoto_rk4_trajectory(theta0, omega, coupling, _DT, _STEPS)
    with pytest.raises(ImportError, match="oscillatools\\[jax\\]"):
        jk.jax_kuramoto_rk4_gradient(
            theta0, omega, coupling, _DT, _STEPS, np.ones(6, dtype=np.float64)
        )


def test_forward_and_gradient_are_deterministic() -> None:
    theta0, omega, coupling = _network(12, seed=5)
    first = jk.jax_kuramoto_rk4_trajectory(theta0, omega, coupling, _DT, _STEPS)
    second = jk.jax_kuramoto_rk4_trajectory(theta0, omega, coupling, _DT, _STEPS)
    assert np.array_equal(first, second)
    cotangent = np.asarray(np.linspace(-1.0, 1.0, 12), dtype=np.float64)
    grad_a = jk.jax_kuramoto_rk4_gradient(theta0, omega, coupling, _DT, _STEPS, cotangent)
    grad_b = jk.jax_kuramoto_rk4_gradient(theta0, omega, coupling, _DT, _STEPS, cotangent)
    for first_grad, second_grad in zip(grad_a, grad_b, strict=True):
        assert np.array_equal(first_grad, second_grad)


def _ensemble(
    n: int, batch: int, seed: int
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    rng = np.random.default_rng(seed)
    omega = np.asarray(rng.normal(0.0, 0.7, size=n), dtype=np.float64)
    coupling = np.full((n, n), 1.8 / n, dtype=np.float64)
    np.fill_diagonal(coupling, 0.0)
    theta0_batch = np.asarray(rng.uniform(0.0, 2.0 * np.pi, size=(batch, n)), dtype=np.float64)
    cotangent_batch = np.asarray(rng.normal(0.0, 1.0, size=(batch, n)), dtype=np.float64)
    return theta0_batch, omega, coupling, cotangent_batch


def test_ensemble_forward_matches_each_single_member() -> None:
    batch, n = 6, 32
    theta0_batch, omega, coupling, _ = _ensemble(n, batch, seed=21)
    ensemble = jk.jax_kuramoto_rk4_ensemble(theta0_batch, omega, coupling, _DT, _STEPS)
    assert ensemble.shape == (batch, _STEPS + 1, n)
    for member in range(batch):
        single = jk.jax_kuramoto_rk4_trajectory(theta0_batch[member], omega, coupling, _DT, _STEPS)
        # the vmap batches the identical solve, so each member is bit-for-bit the single call
        assert np.array_equal(ensemble[member], single)


def test_ensemble_gradient_matches_each_single_member() -> None:
    batch, n = 6, 32
    theta0_batch, omega, coupling, cotangent_batch = _ensemble(n, batch, seed=22)
    grad_theta0, grad_omega, grad_coupling = jk.jax_kuramoto_rk4_ensemble_gradient(
        theta0_batch, omega, coupling, _DT, _STEPS, cotangent_batch
    )
    assert grad_theta0.shape == (batch, n)
    assert grad_omega.shape == (batch, n)
    assert grad_coupling.shape == (batch, n, n)
    for member in range(batch):
        single = jk.jax_kuramoto_rk4_gradient(
            theta0_batch[member], omega, coupling, _DT, _STEPS, cotangent_batch[member]
        )
        assert np.max(np.abs(grad_theta0[member] - single[0])) < 1e-9
        assert np.max(np.abs(grad_omega[member] - single[1])) < 1e-9
        assert np.max(np.abs(grad_coupling[member] - single[2])) < 1e-9


def test_ensemble_runs_on_the_jax_default_device() -> None:
    import jax

    theta0_batch, omega, coupling, _ = _ensemble(16, 4, seed=23)
    backend = jk._load_backend()
    jnp = backend.jnp
    result = backend.ensemble_trajectory(
        jnp.asarray(theta0_batch), jnp.asarray(omega), jnp.asarray(coupling), _DT, _STEPS
    )
    # the batched solve lands on the accelerator JAX selected (a CUDA GPU when one is present)
    assert result.device == jax.devices()[0]


def test_ensemble_rejects_bad_shapes() -> None:
    theta0_batch, omega, coupling, cotangent_batch = _ensemble(6, 4, seed=24)
    with pytest.raises(ValueError, match="two-dimensional"):
        jk.jax_kuramoto_rk4_ensemble(theta0_batch[0], omega, coupling, _DT, _STEPS)
    with pytest.raises(ValueError, match="B >= 1"):
        jk.jax_kuramoto_rk4_ensemble(theta0_batch[:0], omega, coupling, _DT, _STEPS)
    with pytest.raises(ValueError, match="n_steps must be positive"):
        jk.jax_kuramoto_rk4_ensemble_gradient(
            theta0_batch, omega, coupling, _DT, 0, cotangent_batch
        )
    with pytest.raises(ValueError, match="cotangent_batch"):
        jk.jax_kuramoto_rk4_ensemble_gradient(
            theta0_batch, omega, coupling, _DT, _STEPS, cotangent_batch[:, :-1]
        )


def test_new_integrators_are_reexported_through_public_facades() -> None:
    import oscillatools
    from oscillatools import accel

    assert accel.jax_kuramoto_euler_trajectory is ji.jax_kuramoto_euler_trajectory
    assert accel.jax_kuramoto_dopri_trajectory is ji.jax_kuramoto_dopri_trajectory
    assert accel.jax_networked_inertial_trajectory is ji.jax_networked_inertial_trajectory
    assert accel.jax_networked_noisy_trajectory is ji.jax_networked_noisy_trajectory
    assert (
        accel.jax_networked_symplectic_inertial_trajectory
        is ji.jax_networked_symplectic_inertial_trajectory
    )
    assert oscillatools.jax_kuramoto_euler_trajectory is ji.jax_kuramoto_euler_trajectory
    assert oscillatools.jax_kuramoto_dopri_trajectory is ji.jax_kuramoto_dopri_trajectory
    assert oscillatools.jax_networked_inertial_trajectory is ji.jax_networked_inertial_trajectory
    assert oscillatools.jax_networked_noisy_trajectory is ji.jax_networked_noisy_trajectory
    assert (
        oscillatools.jax_networked_symplectic_inertial_trajectory
        is ji.jax_networked_symplectic_inertial_trajectory
    )
