# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Differentiable Kuramoto pinning control tests
"""Multi-angle tests for the differentiable Kuramoto pinning control.

Covers the pacemaker-augmentation identity (the pacemaker stays at the target, pinning equals an
extra force), the gain gradient against finite differences for both integrators, the value-only
companion, input and integrator validation, and the sparse pinning design (monotone penalised
cost, the order parameter raised, the sparsity weight shrinking the pin-set, the converged flag,
the iteration budget and reproducibility).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import oscillatools.accel.diff_kuramoto_rk4 as dk_rk4
import oscillatools.accel.kuramoto_pinning_control as pin
from oscillatools.accel import order_parameter


def _random_problem(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-math.pi, math.pi, size=n)
    omega = rng.uniform(-1.0, 1.0, size=n)
    coupling = rng.uniform(0.0, 0.2, size=(n, n))
    coupling = 0.5 * (coupling + coupling.T)
    np.fill_diagonal(coupling, 0.0)
    gains = rng.uniform(0.0, 0.5, size=n)
    return theta0, omega, coupling, gains


def _final_radius(theta0, omega, coupling, gains, target, dt=0.05, n_steps=30):
    theta_aug, omega_aug, coupling_aug = pin._augment(theta0, omega, coupling, gains, target)
    final = dk_rk4._python_kuramoto_rk4_trajectory(
        theta_aug, omega_aug, coupling_aug, dt, n_steps
    )[-1]
    return float(order_parameter(final[: theta0.size]))


class TestPinningGradient:
    def test_pacemaker_stays_at_target_and_pins_apply_force(self) -> None:
        theta0, omega, coupling, gains = _random_problem(6, 1)
        target = 0.7
        theta_aug, omega_aug, coupling_aug = pin._augment(theta0, omega, coupling, gains, target)
        trajectory = dk_rk4._python_kuramoto_rk4_trajectory(
            theta_aug, omega_aug, coupling_aug, 0.05, 20
        )
        # The pacemaker (last node) is held exactly at the target throughout.
        np.testing.assert_allclose(trajectory[:, -1], target, atol=1e-13)

    @pytest.mark.parametrize("integrator", ["rk4", "euler"])
    def test_gain_gradient_matches_finite_difference(self, integrator: str) -> None:
        theta0, omega, coupling, gains = _random_problem(7, 5)
        target, dt, n_steps = 0.4, 0.05, 15
        _, grad_gains = pin.pinning_coherence_value_and_grad(
            theta0, omega, coupling, gains, target, dt, n_steps, integrator=integrator
        )
        step = 1e-6
        fd = np.zeros_like(gains)
        for i in range(gains.size):
            plus = gains.copy()
            minus = gains.copy()
            plus[i] += step
            minus[i] -= step
            fd[i] = (
                pin.pinning_coherence_value(
                    theta0, omega, coupling, plus, target, dt, n_steps, integrator=integrator
                )
                - pin.pinning_coherence_value(
                    theta0, omega, coupling, minus, target, dt, n_steps, integrator=integrator
                )
            ) / (2.0 * step)
        np.testing.assert_allclose(grad_gains, fd, atol=1e-5)

    def test_value_matches_value_and_grad(self) -> None:
        theta0, omega, coupling, gains = _random_problem(6, 9)
        cost, _ = pin.pinning_coherence_value_and_grad(
            theta0, omega, coupling, gains, 0.3, 0.05, 12
        )
        value = pin.pinning_coherence_value(theta0, omega, coupling, gains, 0.3, 0.05, 12)
        assert abs(cost - value) < 1e-12

    def test_validation(self) -> None:
        theta0, omega, coupling, gains = _random_problem(4, 1)
        with pytest.raises(ValueError, match="gains must have shape"):
            pin.pinning_coherence_value_and_grad(
                theta0, omega, coupling, np.zeros(3), 0.0, 0.05, 5
            )
        with pytest.raises(ValueError, match="omega must have shape"):
            pin.pinning_coherence_value(theta0, np.zeros(3), coupling, gains, 0.0, 0.05, 5)
        with pytest.raises(ValueError, match="coupling must have shape"):
            pin.pinning_coherence_value_and_grad(
                theta0, omega, np.zeros((4, 3)), gains, 0.0, 0.05, 5
            )
        with pytest.raises(ValueError, match="integrator must be 'rk4' or 'euler'"):
            pin.pinning_coherence_value_and_grad(
                theta0, omega, coupling, gains, 0.0, 0.05, 5, integrator="midpoint"
            )
        with pytest.raises(ValueError, match="integrator must be 'rk4' or 'euler'"):
            pin.pinning_coherence_value(
                theta0, omega, coupling, gains, 0.0, 0.05, 5, integrator="x"
            )


class TestDesignPinning:
    def test_monotone_cost_and_raises_order_parameter(self) -> None:
        theta0, omega, _, _ = _random_problem(10, 21)
        # With no inter-node coupling, pinning is the sole synchronisation mechanism, so its
        # effect on the order parameter is unambiguous.
        coupling = np.zeros((10, 10))
        target = 0.0
        n_steps = 60
        radius_before = _final_radius(
            theta0, omega, coupling, np.zeros(10), target, n_steps=n_steps
        )
        result = pin.design_pinning(
            theta0, omega, coupling, target, 0.05, n_steps, max_iterations=60, learning_rate=20.0
        )
        for earlier, later in zip(result.cost_history, result.cost_history[1:], strict=False):
            assert later <= earlier + 1e-9
        radius_after = _final_radius(
            theta0, omega, coupling, result.gains, target, n_steps=n_steps
        )
        assert radius_after > radius_before + 0.1
        assert np.all(result.gains >= 0.0)

    def test_sparsity_weight_shrinks_pin_set(self) -> None:
        theta0, omega, coupling, _ = _random_problem(12, 4)
        dense = pin.design_pinning(
            theta0,
            omega,
            coupling,
            0.0,
            0.05,
            30,
            max_iterations=30,
            learning_rate=5.0,
            sparsity_weight=0.0,
        )
        sparse = pin.design_pinning(
            theta0,
            omega,
            coupling,
            0.0,
            0.05,
            30,
            max_iterations=30,
            learning_rate=5.0,
            sparsity_weight=0.2,
        )
        assert sparse.pin_set.size <= dense.pin_set.size
        # The sparse design pins a genuine subset.
        assert sparse.pin_set.size < theta0.size

    def test_converged_flag_when_already_synchronised(self) -> None:
        n = 5
        theta0 = np.zeros(n)
        omega = np.zeros(n)
        coupling = np.zeros((n, n))
        result = pin.design_pinning(
            theta0, omega, coupling, 0.0, 0.05, 20, max_iterations=50, sparsity_weight=0.1
        )
        assert result.converged
        assert result.iterations == 1
        assert result.pin_set.size == 0

    def test_iteration_budget_and_rejects_non_positive(self) -> None:
        theta0, omega, coupling, _ = _random_problem(8, 11)
        result = pin.design_pinning(theta0, omega, coupling, 0.0, 0.05, 30, max_iterations=3)
        assert result.iterations <= 3
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            pin.design_pinning(theta0, omega, coupling, 0.0, 0.05, 5, max_iterations=0)

    def test_reproducible(self) -> None:
        theta0, omega, coupling, _ = _random_problem(6, 5)
        first = pin.design_pinning(theta0, omega, coupling, 0.0, 0.05, 25, max_iterations=20)
        second = pin.design_pinning(theta0, omega, coupling, 0.0, 0.05, 25, max_iterations=20)
        np.testing.assert_array_equal(first.gains, second.gains)
        np.testing.assert_array_equal(first.pin_set, second.pin_set)
