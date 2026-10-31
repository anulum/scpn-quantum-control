# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — qualification of delayed, stochastic and second-order Kuramoto sensitivities
"""Each sensitivity is checked against an oracle that does not share its derivation.

* Continuous delay: perturb the delay, re-solve with the same history and
  interpolation rule, and difference; refine the step to show the discrete
  derivative converges.
* Integer delay: the step-count delay is structural and has no derivative.
* Noise: one path's derivative against a difference on the same path (common
  random numbers); the expectation gradient against a closed form, with
  independent replicates and their standard errors.
* Second order: the delayed Hessian-vector product against a difference of
  the hand-derived gradient.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel import (
    EXPECTATION_ESTIMAND,
    delayed_delay_sensitivity,
    delayed_phase_sensitivity,
    delayed_terminal_value_and_grad,
    integrate_delayed_kuramoto,
    noisy_expectation_value_and_grad,
    noisy_terminal_value_and_grad,
)
from oscillatools.accel.kuramoto_delayed import delayed_networked_force


def _network(
    count: int, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return a reproducible ``(history, omega, coupling)`` with a constant history."""
    rng = np.random.default_rng(seed)
    coupling = rng.uniform(0.2, 1.0, size=(count, count))
    np.fill_diagonal(coupling, 0.0)
    return rng.uniform(-1.0, 1.0, size=count), rng.normal(1.0, 0.3, size=count), coupling


def _cosine(theta: NDArray[np.float64]) -> float:
    """Terminal objective ``Σ cos θ``."""
    return float(np.sum(np.cos(theta)))


def _cosine_gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gradient of :func:`_cosine`."""
    return np.asarray(-np.sin(theta), dtype=np.float64)


def test_delayed_stochastic_sensitivities_01_continuous_delay_by_perturb_and_resolve() -> None:
    """``∂θ_N/∂τ`` equals re-solving at ``τ ± h``, and converges as the step is refined."""
    history, omega, coupling = _network(3, seed=4)
    delay, horizon = 0.3725, 2.0
    for dt in (0.02, 0.01):
        steps = round(horizon / dt)
        _, sensitivity = delayed_delay_sensitivity(
            history, omega, coupling, delay=delay, dt=dt, n_steps=steps
        )
        step = 1e-6
        plus, _ = delayed_delay_sensitivity(
            history, omega, coupling, delay=delay + step, dt=dt, n_steps=steps
        )
        minus, _ = delayed_delay_sensitivity(
            history, omega, coupling, delay=delay - step, dt=dt, n_steps=steps
        )
        np.testing.assert_allclose(sensitivity, (plus - minus) / (2 * step), rtol=1e-6, atol=1e-9)
    # Refinement: the discrete derivative approaches one limit as dt shrinks.
    estimates = []
    for dt in (0.04, 0.02, 0.01):
        _, sensitivity = delayed_delay_sensitivity(
            history, omega, coupling, delay=delay, dt=dt, n_steps=round(horizon / dt)
        )
        estimates.append(sensitivity)
    coarse = float(np.max(np.abs(estimates[1] - estimates[0])))
    fine = float(np.max(np.abs(estimates[2] - estimates[1])))
    assert fine < 0.5 * coarse
    # A frozen integer delay would report no change at all; the continuous delay moves the state.
    assert float(np.max(np.abs(estimates[2]))) > 1e-3


def test_delayed_stochastic_sensitivities_01_frozen_delay_is_caught_by_the_structural_map() -> (
    None
):
    """Moving the integer delay by one step changes the state by about ``dt · ∂θ_N/∂τ``.

    A derivative that treats the step-count delay as a frozen constant would
    predict no change; the structural map's one-step difference agrees with
    the continuous sensitivity instead, to first order in ``dt``.
    """
    history, omega, coupling = _network(3, seed=9)
    dt, steps, delay_steps = 0.01, 200, 30

    def structural(count: int) -> NDArray[np.float64]:
        trajectory = integrate_delayed_kuramoto(
            np.tile(history, (count + 1, 1)),
            omega,
            lambda current, lagged: delayed_networked_force(current, lagged, coupling),
            delay=count * dt,
            dt=dt,
            n_steps=steps,
        )
        return np.asarray(trajectory.phases[-1], dtype=np.float64)

    one_step = (structural(delay_steps + 1) - structural(delay_steps)) / dt
    _, sensitivity = delayed_delay_sensitivity(
        history, omega, coupling, delay=(delay_steps + 0.25) * dt, dt=dt, n_steps=steps
    )
    assert float(np.max(np.abs(one_step))) > 1e-3
    np.testing.assert_allclose(one_step, sensitivity, rtol=0.05, atol=1e-3)


def test_delayed_stochastic_sensitivities_02_integer_delay_has_no_continuous_derivative() -> None:
    """A delay on the step grid is refused, and the structural sensitivity has no delay channel."""
    history, omega, coupling = _network(2, seed=1)
    dt = 0.05
    for multiple in (3.0, 3.5):
        with pytest.raises(ValueError, match="half-integer multiple of dt"):
            delayed_delay_sensitivity(
                history, omega, coupling, delay=multiple * dt, dt=dt, n_steps=10
            )
    delay_steps, count = 3, 2
    _, sensitivity = delayed_phase_sensitivity(
        np.tile(history, (delay_steps + 1, 1)),
        omega,
        coupling,
        delay=delay_steps * dt,
        dt=dt,
        n_steps=10,
    )
    # Channels: the history rows, the frequencies and the coupling; none for τ.
    assert sensitivity.shape == (count, (delay_steps + 1) * count + count + count * count)
    with pytest.raises(ValueError):
        delayed_phase_sensitivity(
            np.tile(history, (delay_steps + 1, 1)),
            omega,
            coupling,
            delay=3.3 * dt,
            dt=dt,
            n_steps=10,
        )


def test_delayed_stochastic_sensitivities_03_pathwise_and_expectation_are_distinct() -> None:
    """A common random path checks one path's derivative; replicates check the expectation."""
    phases, omega, coupling = _network(3, seed=2)
    diffusion, dt, steps, seed = 0.15, 0.01, 80, 11
    _, pathwise = noisy_terminal_value_and_grad(
        phases,
        omega,
        coupling,
        diffusion=diffusion,
        dt=dt,
        n_steps=steps,
        seed=seed,
        objective=_cosine,
        objective_grad=_cosine_gradient,
    )
    step = 1e-6
    for index in range(3):
        shift = np.zeros(3)
        shift[index] = step
        plus, _ = noisy_terminal_value_and_grad(
            phases,
            omega + shift,
            coupling,
            diffusion=diffusion,
            dt=dt,
            n_steps=steps,
            seed=seed,
            objective=_cosine,
            objective_grad=_cosine_gradient,
        )
        minus, _ = noisy_terminal_value_and_grad(
            phases,
            omega - shift,
            coupling,
            diffusion=diffusion,
            dt=dt,
            n_steps=steps,
            seed=seed,
            objective=_cosine,
            objective_grad=_cosine_gradient,
        )
        assert pathwise.omega[index] == pytest.approx((plus - minus) / (2 * step), rel=1e-6)
    # Expectation: without coupling the terminal phases are Gaussian and E[Σ cos θ_N] is exact.
    free = np.zeros((3, 3))
    horizon = dt * steps
    angle = phases + omega * horizon
    damping = np.exp(-diffusion * horizon)
    estimate = noisy_expectation_value_and_grad(
        phases,
        omega,
        free,
        diffusion=diffusion,
        dt=dt,
        n_steps=steps,
        seeds=range(500),
        objective=_cosine,
        objective_grad=_cosine_gradient,
    )
    exact_omega = -horizon * np.sin(angle) * damping
    exact_diffusion = -horizon * float(np.sum(np.cos(angle))) * damping
    assert np.all(
        np.abs(estimate.gradients.omega - exact_omega) <= 4 * estimate.standard_errors.omega
    )
    assert (
        abs(estimate.gradients.diffusion - exact_diffusion)
        <= 4 * estimate.standard_errors.diffusion
    )
    assert estimate.estimand == EXPECTATION_ESTIMAND
    # A second, disjoint replicate set is an independent interval for the same estimand.
    other = noisy_expectation_value_and_grad(
        phases,
        omega,
        free,
        diffusion=diffusion,
        dt=dt,
        n_steps=steps,
        seeds=range(500, 1000),
        objective=_cosine,
        objective_grad=_cosine_gradient,
    )
    assert np.all(np.abs(other.gradients.omega - exact_omega) <= 4 * other.standard_errors.omega)
    _, single = noisy_terminal_value_and_grad(
        phases,
        omega,
        free,
        diffusion=diffusion,
        dt=dt,
        n_steps=steps,
        seed=seed,
        objective=_cosine,
        objective_grad=_cosine_gradient,
    )
    assert np.any(np.abs(single.omega - exact_omega) > 4 * estimate.standard_errors.omega)


def test_delayed_hessian_vector_product_has_its_own_oracle() -> None:
    """The second-order product equals a central difference of the hand-derived gradient."""
    pytest.importorskip("jax", reason="the JAX tier is optional and not installed")
    from oscillatools.accel import jax_kuramoto_delayed_hessian_vector_product

    phases, omega, coupling = _network(3, seed=6)
    dt, delay_steps, steps = 0.05, 4, 40
    history = np.tile(phases, (delay_steps + 1, 1))
    rng = np.random.default_rng(8)
    direction = (
        rng.normal(size=history.shape),
        rng.normal(size=omega.shape),
        rng.normal(size=coupling.shape),
    )
    product = jax_kuramoto_delayed_hessian_vector_product(
        history,
        omega,
        coupling,
        delay=delay_steps * dt,
        dt=dt,
        n_steps=steps,
        objective_grad=_cosine_gradient,
        objective_hessian=lambda theta: np.diag(-np.cos(theta)),
        direction=direction,
    )
    step = 1e-5

    def hand(sign: float) -> list[NDArray[np.float64]]:
        _, grads = delayed_terminal_value_and_grad(
            history + sign * step * direction[0],
            omega + sign * step * direction[1],
            coupling + sign * step * direction[2],
            delay=delay_steps * dt,
            dt=dt,
            n_steps=steps,
            objective=_cosine,
            objective_grad=_cosine_gradient,
        )
        return [grads.initial_history, grads.omega, grads.coupling]

    for exact, plus, minus in zip(product, hand(1.0), hand(-1.0), strict=True):
        np.testing.assert_allclose(exact, (plus - minus) / (2 * step), rtol=1e-6, atol=1e-8)
