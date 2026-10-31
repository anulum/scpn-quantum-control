# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the JAX autodiff delayed-Kuramoto tier
"""Contract tests for the JAX autodiff tier of the time-delayed method-of-steps Kuramoto integrator.

These exercise real JAX on the accelerator JAX selected (a CUDA GPU when present), so the module skips
without the optional ``[jax]`` extra. They pin the two claims the tier is built on — the forward is
faithful to the production NumPy delayed integrator at 64-bit precision, and the autodiff gradient
matches the hand-derived method-of-steps sensitivity to machine precision (and a central finite
difference independently) — plus the residency, fallback, validation and determinism contracts, and the
``vmap`` ensemble equivalence over a batch of initial histories.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from oscillatools.accel import jax_kuramoto_delayed as jkd  # noqa: E402
from oscillatools.accel.diff_kuramoto_delayed import (  # noqa: E402
    delayed_terminal_value_and_grad,
)
from oscillatools.accel.kuramoto_delayed import (  # noqa: E402
    delayed_networked_force,
    integrate_delayed_kuramoto,
)

_DT = 0.05
_STEPS = 40


def _delayed_case(
    n: int, delay_steps: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``(initial_history, omega, coupling, delay)`` for a networked delayed problem."""
    rng = np.random.default_rng(seed)
    history = rng.uniform(-0.6, 0.6, size=(delay_steps + 1, n))
    omega = rng.normal(0.0, 0.4, size=n)
    coupling = rng.uniform(0.0, 0.9, size=(n, n))
    np.fill_diagonal(coupling, 0.0)
    return history, omega, coupling, delay_steps * _DT


def _reference_phases(
    history: np.ndarray, omega: np.ndarray, coupling: np.ndarray, delay: float
) -> np.ndarray:
    trajectory = integrate_delayed_kuramoto(
        history,
        omega,
        lambda current, lagged: delayed_networked_force(current, lagged, coupling),
        delay=delay,
        dt=_DT,
        n_steps=_STEPS,
    )
    return trajectory.phases


@pytest.mark.parametrize(("n", "delay_steps"), [(6, 3), (8, 1), (5, 5)])
def test_forward_matches_numpy_delayed_at_x64(n: int, delay_steps: int) -> None:
    history, omega, coupling, delay = _delayed_case(n, delay_steps, seed=n * 10 + delay_steps)
    reference = _reference_phases(history, omega, coupling, delay)
    jax_trajectory = jkd.jax_kuramoto_delayed_trajectory(
        history, omega, coupling, delay=delay, dt=_DT, n_steps=_STEPS
    )
    assert jax_trajectory.shape == (_STEPS + 1, n)
    assert jax_trajectory.dtype == np.float64
    # 64-bit JAX reproduces the method-of-steps map; parity under a tolerance (GPU reduction ordering
    # need not be bit-identical to NumPy — observed here at machine precision).
    assert np.max(np.abs(jax_trajectory - reference)) < 1e-11


@pytest.mark.parametrize(("n", "delay_steps"), [(6, 3), (8, 1), (5, 5)])
def test_gradient_matches_hand_derived_sensitivity(n: int, delay_steps: int) -> None:
    history, omega, coupling, delay = _delayed_case(n, delay_steps, seed=n * 10 + delay_steps + 1)
    rng = np.random.default_rng(n + delay_steps)
    cotangent = rng.normal(0.0, 1.0, size=n)
    _, hand = delayed_terminal_value_and_grad(
        history,
        omega,
        coupling,
        delay=delay,
        dt=_DT,
        n_steps=_STEPS,
        objective=lambda phases: float(cotangent @ phases),
        objective_grad=lambda phases: cotangent,
    )
    autodiff = jkd.jax_kuramoto_delayed_gradient(
        history, omega, coupling, delay=delay, dt=_DT, n_steps=_STEPS, cotangent=cotangent
    )
    grad_history, grad_omega, grad_coupling = autodiff
    assert grad_history.shape == (delay_steps + 1, n)
    assert grad_omega.shape == (n,)
    assert grad_coupling.shape == (n, n)
    assert np.max(np.abs(grad_history - hand.initial_history)) < 1e-9
    assert np.max(np.abs(grad_omega - hand.omega)) < 1e-9
    assert np.max(np.abs(grad_coupling - hand.coupling)) < 1e-9


def test_gradient_matches_central_finite_difference() -> None:
    n, delay_steps = 4, 2
    history, omega, coupling, delay = _delayed_case(n, delay_steps, seed=99)
    cotangent = np.sin(np.arange(n) + 1.0)
    _, _, grad_coupling = jkd.jax_kuramoto_delayed_gradient(
        history, omega, coupling, delay=delay, dt=_DT, n_steps=_STEPS, cotangent=cotangent
    )
    eps = 1e-6
    finite_difference = np.zeros_like(coupling)
    for row in range(n):
        for column in range(n):
            plus = coupling.copy()
            plus[row, column] += eps
            minus = coupling.copy()
            minus[row, column] -= eps
            forward = cotangent @ _reference_phases(history, omega, plus, delay)[-1]
            backward = cotangent @ _reference_phases(history, omega, minus, delay)[-1]
            finite_difference[row, column] = (forward - backward) / (2.0 * eps)
    # an independent anchor: the autodiff coupling gradient matches a central finite difference of the
    # production integrator to finite-difference precision.
    assert np.max(np.abs(grad_coupling - finite_difference)) < 1e-6


def test_trajectory_runs_on_the_jax_default_device() -> None:
    import jax

    history, omega, coupling, _ = _delayed_case(10, 3, seed=7)
    backend = jkd._load_backend()
    jnp = backend.jnp
    result = backend.trajectory(
        jnp.asarray(history), jnp.asarray(omega), jnp.asarray(coupling), _DT, _STEPS
    )
    # the jitted delayed solve lands on the accelerator JAX selected (a CUDA GPU when one is present)
    assert result.device == jax.devices()[0]


def test_forward_rejects_out_of_range_and_malformed() -> None:
    history, omega, coupling, delay = _delayed_case(6, 3, seed=2)
    with pytest.raises(ValueError, match="n_steps must be positive"):
        jkd.jax_kuramoto_delayed_trajectory(
            history, omega, coupling, delay=delay, dt=_DT, n_steps=0
        )
    with pytest.raises(ValueError, match="integer multiple of dt"):
        jkd.jax_kuramoto_delayed_trajectory(
            history, omega, coupling, delay=delay + 0.5 * _DT, dt=_DT, n_steps=_STEPS
        )
    with pytest.raises(ValueError, match="integer multiple of dt"):
        jkd.jax_kuramoto_delayed_trajectory(
            history, omega, coupling, delay=0.5 * _DT, dt=_DT, n_steps=_STEPS
        )
    with pytest.raises(ValueError, match="initial_history must have shape"):
        jkd.jax_kuramoto_delayed_trajectory(
            history[:-1], omega, coupling, delay=delay, dt=_DT, n_steps=_STEPS
        )
    with pytest.raises(ValueError, match="coupling must have shape"):
        jkd.jax_kuramoto_delayed_trajectory(
            history, omega, coupling[:-1], delay=delay, dt=_DT, n_steps=_STEPS
        )
    with pytest.raises(ValueError, match="omega must be a non-empty one-dimensional array"):
        jkd.jax_kuramoto_delayed_trajectory(
            history, omega.reshape(-1, 1), coupling, delay=delay, dt=_DT, n_steps=_STEPS
        )


def test_gradient_rejects_bad_cotangent() -> None:
    history, omega, coupling, delay = _delayed_case(6, 3, seed=3)
    with pytest.raises(ValueError, match="cotangent"):
        jkd.jax_kuramoto_delayed_gradient(
            history, omega, coupling, delay=delay, dt=_DT, n_steps=_STEPS, cotangent=np.ones(5)
        )


def test_missing_jax_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> object:
        raise ImportError("the JAX delayed-Kuramoto tier requires JAX; install oscillatools[jax]")

    monkeypatch.setattr(jkd, "_load_backend", _raise)
    history, omega, coupling, delay = _delayed_case(6, 3, seed=4)
    with pytest.raises(ImportError, match="oscillatools\\[jax\\]"):
        jkd.jax_kuramoto_delayed_trajectory(
            history, omega, coupling, delay=delay, dt=_DT, n_steps=_STEPS
        )
    with pytest.raises(ImportError, match="oscillatools\\[jax\\]"):
        jkd.jax_kuramoto_delayed_gradient(
            history, omega, coupling, delay=delay, dt=_DT, n_steps=_STEPS, cotangent=np.ones(6)
        )


def test_forward_and_gradient_are_deterministic() -> None:
    history, omega, coupling, delay = _delayed_case(12, 4, seed=5)
    first = jkd.jax_kuramoto_delayed_trajectory(
        history, omega, coupling, delay=delay, dt=_DT, n_steps=_STEPS
    )
    second = jkd.jax_kuramoto_delayed_trajectory(
        history, omega, coupling, delay=delay, dt=_DT, n_steps=_STEPS
    )
    assert np.array_equal(first, second)
    cotangent = np.linspace(-1.0, 1.0, 12)
    grad_a = jkd.jax_kuramoto_delayed_gradient(
        history, omega, coupling, delay=delay, dt=_DT, n_steps=_STEPS, cotangent=cotangent
    )
    grad_b = jkd.jax_kuramoto_delayed_gradient(
        history, omega, coupling, delay=delay, dt=_DT, n_steps=_STEPS, cotangent=cotangent
    )
    for first_grad, second_grad in zip(grad_a, grad_b, strict=True):
        assert np.array_equal(first_grad, second_grad)


def _delayed_ensemble(
    n: int, delay_steps: int, batch: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    history_batch = rng.uniform(-0.6, 0.6, size=(batch, delay_steps + 1, n))
    omega = rng.normal(0.0, 0.4, size=n)
    coupling = rng.uniform(0.0, 0.9, size=(n, n))
    np.fill_diagonal(coupling, 0.0)
    cotangent_batch = rng.normal(0.0, 1.0, size=(batch, n))
    return history_batch, omega, coupling, delay_steps * _DT, cotangent_batch


def test_ensemble_forward_matches_each_single_member() -> None:
    batch, n, delay_steps = 5, 6, 3
    history_batch, omega, coupling, delay, _ = _delayed_ensemble(n, delay_steps, batch, seed=21)
    ensemble = jkd.jax_kuramoto_delayed_ensemble(
        history_batch, omega, coupling, delay=delay, dt=_DT, n_steps=_STEPS
    )
    assert ensemble.shape == (batch, _STEPS + 1, n)
    for member in range(batch):
        single = jkd.jax_kuramoto_delayed_trajectory(
            history_batch[member], omega, coupling, delay=delay, dt=_DT, n_steps=_STEPS
        )
        # the vmap batches the identical solve, so each member is bit-for-bit the single call
        assert np.array_equal(ensemble[member], single)


def test_ensemble_gradient_matches_each_single_member() -> None:
    batch, n, delay_steps = 5, 6, 3
    history_batch, omega, coupling, delay, cotangent_batch = _delayed_ensemble(
        n, delay_steps, batch, seed=22
    )
    grad_history, grad_omega, grad_coupling = jkd.jax_kuramoto_delayed_ensemble_gradient(
        history_batch,
        omega,
        coupling,
        delay=delay,
        dt=_DT,
        n_steps=_STEPS,
        cotangent_batch=cotangent_batch,
    )
    assert grad_history.shape == (batch, delay_steps + 1, n)
    assert grad_omega.shape == (batch, n)
    assert grad_coupling.shape == (batch, n, n)
    for member in range(batch):
        single = jkd.jax_kuramoto_delayed_gradient(
            history_batch[member],
            omega,
            coupling,
            delay=delay,
            dt=_DT,
            n_steps=_STEPS,
            cotangent=cotangent_batch[member],
        )
        assert np.max(np.abs(grad_history[member] - single[0])) < 1e-9
        assert np.max(np.abs(grad_omega[member] - single[1])) < 1e-9
        assert np.max(np.abs(grad_coupling[member] - single[2])) < 1e-9


def test_ensemble_runs_on_the_jax_default_device() -> None:
    import jax

    history_batch, omega, coupling, _, _ = _delayed_ensemble(10, 3, 4, seed=23)
    backend = jkd._load_backend()
    jnp = backend.jnp
    result = backend.ensemble_trajectory(
        jnp.asarray(history_batch), jnp.asarray(omega), jnp.asarray(coupling), _DT, _STEPS
    )
    # the batched delayed solve lands on the accelerator JAX selected (a CUDA GPU when one is present)
    assert result.device == jax.devices()[0]


def test_ensemble_rejects_bad_shapes() -> None:
    history_batch, omega, coupling, delay, cotangent_batch = _delayed_ensemble(6, 3, 4, seed=24)
    with pytest.raises(ValueError, match="three-dimensional"):
        jkd.jax_kuramoto_delayed_ensemble(
            history_batch[0], omega, coupling, delay=delay, dt=_DT, n_steps=_STEPS
        )
    with pytest.raises(ValueError, match="B >= 1"):
        jkd.jax_kuramoto_delayed_ensemble(
            history_batch[:0], omega, coupling, delay=delay, dt=_DT, n_steps=_STEPS
        )
    with pytest.raises(ValueError, match="cotangent_batch must have shape"):
        jkd.jax_kuramoto_delayed_ensemble_gradient(
            history_batch,
            omega,
            coupling,
            delay=delay,
            dt=_DT,
            n_steps=_STEPS,
            cotangent_batch=cotangent_batch[:, :-1],
        )


def _cosine_objective() -> tuple[object, object, object]:
    """Return ``L = Σ cos θ`` with its gradient and Hessian."""

    def value(theta: np.ndarray) -> float:
        return float(np.sum(np.cos(theta)))

    def gradient(theta: np.ndarray) -> np.ndarray:
        return -np.sin(theta)

    def hessian(theta: np.ndarray) -> np.ndarray:
        return np.diag(-np.cos(theta))

    return value, gradient, hessian


def test_hessian_vector_product_matches_a_difference_of_hand_derived_gradients() -> None:
    """The second-order product equals a central difference of the hand-written gradient."""
    history, omega, coupling, delay = _delayed_case(4, 3, seed=11)
    value, gradient, hessian = _cosine_objective()
    rng = np.random.default_rng(5)
    direction = (
        rng.normal(size=history.shape),
        rng.normal(size=omega.shape),
        rng.normal(size=coupling.shape),
    )
    product = jkd.jax_kuramoto_delayed_hessian_vector_product(
        history,
        omega,
        coupling,
        delay=delay,
        dt=_DT,
        n_steps=_STEPS,
        objective_grad=gradient,
        objective_hessian=hessian,
        direction=direction,
    )
    step = 1e-5

    def hand(sign: float) -> list[np.ndarray]:
        _, grads = delayed_terminal_value_and_grad(
            history + sign * step * direction[0],
            omega + sign * step * direction[1],
            coupling + sign * step * direction[2],
            delay=delay,
            dt=_DT,
            n_steps=_STEPS,
            objective=value,
            objective_grad=gradient,
        )
        return [grads.initial_history, grads.omega, grads.coupling]

    for exact, plus, minus in zip(product, hand(1.0), hand(-1.0), strict=True):
        difference = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(exact, difference, rtol=1e-6, atol=1e-8)


def test_hessian_vector_product_refuses_malformed_inputs() -> None:
    """Directions, gradients and Hessians of the wrong shape or non-finite are refused."""
    history, omega, coupling, delay = _delayed_case(3, 2, seed=2)
    _, gradient, hessian = _cosine_objective()
    good = (np.zeros_like(history), np.zeros_like(omega), np.zeros_like(coupling))
    common = {"delay": delay, "dt": _DT, "n_steps": 5}
    with pytest.raises(ValueError, match="omega direction"):
        jkd.jax_kuramoto_delayed_hessian_vector_product(
            history,
            omega,
            coupling,
            **common,
            objective_grad=gradient,
            objective_hessian=hessian,
            direction=(good[0], np.zeros(2), good[2]),
        )
    with pytest.raises(ValueError, match="history direction"):
        jkd.jax_kuramoto_delayed_hessian_vector_product(
            history,
            omega,
            coupling,
            **common,
            objective_grad=gradient,
            objective_hessian=hessian,
            direction=(np.full_like(history, np.nan), good[1], good[2]),
        )
    with pytest.raises(ValueError, match="objective_grad"):
        jkd.jax_kuramoto_delayed_hessian_vector_product(
            history,
            omega,
            coupling,
            **common,
            objective_grad=lambda theta: np.zeros(2),
            objective_hessian=hessian,
            direction=good,
        )
    with pytest.raises(ValueError, match="objective_hessian"):
        jkd.jax_kuramoto_delayed_hessian_vector_product(
            history,
            omega,
            coupling,
            **common,
            objective_grad=gradient,
            objective_hessian=lambda theta: np.full((3, 3), np.inf),
            direction=good,
        )
