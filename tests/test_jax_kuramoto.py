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

pytest.importorskip("jax")

from scpn_quantum_control.accel import jax_kuramoto as jk  # noqa: E402
from scpn_quantum_control.accel.diff_kuramoto_rk4 import (  # noqa: E402
    kuramoto_rk4_trajectory,
    kuramoto_rk4_vjp,
)

_DT = 0.05
_STEPS = 40


def _network(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, 0.7, size=n)
    coupling = np.full((n, n), 1.8 / n, dtype=np.float64)
    np.fill_diagonal(coupling, 0.0)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=n)
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


def test_gradient_rejects_non_positive_steps_and_bad_cotangent() -> None:
    theta0, omega, coupling = _network(6, seed=3)
    cotangent = np.ones(6, dtype=np.float64)
    with pytest.raises(ValueError, match="n_steps must be positive"):
        jk.jax_kuramoto_rk4_gradient(theta0, omega, coupling, _DT, 0, cotangent)
    with pytest.raises(ValueError, match="cotangent"):
        jk.jax_kuramoto_rk4_gradient(theta0, omega, coupling, _DT, _STEPS, cotangent[:-1])


def test_missing_jax_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> tuple[object, object, object]:
        raise ImportError("the JAX Kuramoto tier requires JAX; install scpn-quantum-control[jax]")

    monkeypatch.setattr(jk, "_load_backend", _raise)
    theta0, omega, coupling = _network(6, seed=4)
    with pytest.raises(ImportError, match="scpn-quantum-control\\[jax\\]"):
        jk.jax_kuramoto_rk4_trajectory(theta0, omega, coupling, _DT, _STEPS)
    with pytest.raises(ImportError, match="scpn-quantum-control\\[jax\\]"):
        jk.jax_kuramoto_rk4_gradient(theta0, omega, coupling, _DT, _STEPS, np.ones(6))


def test_forward_and_gradient_are_deterministic() -> None:
    theta0, omega, coupling = _network(12, seed=5)
    first = jk.jax_kuramoto_rk4_trajectory(theta0, omega, coupling, _DT, _STEPS)
    second = jk.jax_kuramoto_rk4_trajectory(theta0, omega, coupling, _DT, _STEPS)
    assert np.array_equal(first, second)
    cotangent = np.linspace(-1.0, 1.0, 12)
    grad_a = jk.jax_kuramoto_rk4_gradient(theta0, omega, coupling, _DT, _STEPS, cotangent)
    grad_b = jk.jax_kuramoto_rk4_gradient(theta0, omega, coupling, _DT, _STEPS, cotangent)
    for first_grad, second_grad in zip(grad_a, grad_b, strict=True):
        assert np.array_equal(first_grad, second_grad)


def _ensemble(
    n: int, batch: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, 0.7, size=n)
    coupling = np.full((n, n), 1.8 / n, dtype=np.float64)
    np.fill_diagonal(coupling, 0.0)
    theta0_batch = rng.uniform(0.0, 2.0 * np.pi, size=(batch, n))
    cotangent_batch = rng.normal(0.0, 1.0, size=(batch, n))
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
