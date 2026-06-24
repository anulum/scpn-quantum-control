# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Differentiable Kuramoto optimal coupling design tests
"""Multi-angle tests for the differentiable Kuramoto optimal coupling design.

Covers the symmetric-non-negative projection (idempotence and the three constraints), the
projected-gradient-descent optimiser (monotone cost history, the converged flag at a local
optimum, the iteration budget, validation, the unconstrained path) and the synchronising-design
wrapper (it raises the order parameter substantially and respects the projection), plus
reproducibility.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import scpn_quantum_control.accel.diff_kuramoto_rk4 as dk_rk4
import scpn_quantum_control.accel.kuramoto_control as kc
import scpn_quantum_control.accel.kuramoto_coupling_design as design
from scpn_quantum_control.accel import order_parameter


def _random_problem(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-math.pi, math.pi, size=n)
    omega = rng.uniform(-1.0, 1.0, size=n)
    coupling = rng.uniform(0.0, 0.3, size=(n, n))
    coupling = 0.5 * (coupling + coupling.T)
    np.fill_diagonal(coupling, 0.0)
    return theta0, omega, coupling


def _final_radius(theta0: np.ndarray, omega: np.ndarray, coupling: np.ndarray) -> float:
    final = dk_rk4._python_kuramoto_rk4_trajectory(theta0, omega, coupling, 0.05, 30)[-1]
    return float(order_parameter(final))


class TestProjection:
    def test_enforces_symmetry_nonnegativity_zero_diagonal(self) -> None:
        rng = np.random.default_rng(1)
        matrix = rng.uniform(-1.0, 1.0, size=(6, 6))
        projected = design.symmetric_nonnegative_projection(matrix)
        np.testing.assert_allclose(projected, projected.T, atol=1e-15)
        assert np.all(projected >= 0.0)
        np.testing.assert_array_equal(np.diag(projected), np.zeros(6))

    def test_is_idempotent(self) -> None:
        rng = np.random.default_rng(2)
        matrix = rng.uniform(-1.0, 1.0, size=(5, 5))
        once = design.symmetric_nonnegative_projection(matrix)
        twice = design.symmetric_nonnegative_projection(once)
        np.testing.assert_allclose(once, twice, atol=1e-15)


class TestOptimiser:
    def test_cost_history_is_monotone_non_increasing(self) -> None:
        theta0, omega, coupling = _random_problem(8, 7)
        result = design.optimise_coupling(
            kc.coherence_objective,
            theta0,
            omega,
            coupling,
            0.05,
            30,
            max_iterations=30,
            learning_rate=5.0,
            projection=design.symmetric_nonnegative_projection,
        )
        history = result.cost_history
        assert len(history) >= 1
        for earlier, later in zip(history, history[1:], strict=False):
            assert later <= earlier + 1e-12

    def test_converged_flag_at_local_optimum(self) -> None:
        # Starting fully synchronised with zero detuning, the coherence cost is already 0 and the
        # line search cannot improve, so the optimiser converges immediately.
        n = 5
        theta0 = np.zeros(n)
        omega = np.zeros(n)
        coupling = np.zeros((n, n))
        result = design.optimise_coupling(
            kc.coherence_objective,
            theta0,
            omega,
            coupling,
            0.05,
            20,
            max_iterations=50,
            learning_rate=5.0,
            projection=design.symmetric_nonnegative_projection,
        )
        assert result.converged
        assert result.iterations == 1

    def test_iteration_budget_is_respected(self) -> None:
        theta0, omega, coupling = _random_problem(8, 11)
        result = design.optimise_coupling(
            kc.coherence_objective,
            theta0,
            omega,
            coupling,
            0.05,
            30,
            max_iterations=3,
            learning_rate=5.0,
            projection=design.symmetric_nonnegative_projection,
        )
        assert result.iterations <= 3
        assert len(result.cost_history) <= 3

    def test_unconstrained_path_runs(self) -> None:
        theta0, omega, coupling = _random_problem(5, 13)
        result = design.optimise_coupling(
            kc.coherence_objective, theta0, omega, coupling, 0.05, 20, max_iterations=5
        )
        assert isinstance(result, design.CouplingDesignResult)
        assert result.coupling.shape == (5, 5)

    def test_rejects_non_positive_iterations(self) -> None:
        theta0, omega, coupling = _random_problem(4, 1)
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            design.optimise_coupling(
                kc.coherence_objective, theta0, omega, coupling, 0.05, 5, max_iterations=0
            )

    def test_invalid_integrator_propagates(self) -> None:
        theta0, omega, coupling = _random_problem(4, 1)
        with pytest.raises(ValueError, match="integrator must be 'rk4' or 'euler'"):
            design.optimise_coupling(
                kc.coherence_objective,
                theta0,
                omega,
                coupling,
                0.05,
                5,
                integrator="midpoint",
            )

    def test_reproducible(self) -> None:
        theta0, omega, coupling = _random_problem(6, 5)
        first = design.optimise_coupling(
            kc.coherence_objective, theta0, omega, coupling, 0.05, 25, max_iterations=20
        )
        second = design.optimise_coupling(
            kc.coherence_objective, theta0, omega, coupling, 0.05, 25, max_iterations=20
        )
        np.testing.assert_array_equal(first.coupling, second.coupling)
        assert first.cost_history == second.cost_history


class TestSynchronisingDesign:
    def test_raises_order_parameter_and_respects_projection(self) -> None:
        theta0, omega, coupling = _random_problem(10, 21)
        radius_before = _final_radius(theta0, omega, coupling)
        result = design.design_synchronising_coupling(
            theta0, omega, coupling, 0.05, 30, max_iterations=40, learning_rate=5.0
        )
        radius_after = _final_radius(theta0, omega, result.coupling)
        assert radius_after > radius_before + 0.1
        # The default projection keeps the optimised coupling symmetric, non-negative, diagonal-free.
        np.testing.assert_allclose(result.coupling, result.coupling.T, atol=1e-12)
        assert np.all(result.coupling >= 0.0)
        np.testing.assert_array_equal(np.diag(result.coupling), np.zeros(10))

    def test_default_projection_is_symmetric_nonnegative(self) -> None:
        theta0, omega, coupling = _random_problem(5, 2)
        # An asymmetric, signed start is projected before optimisation begins.
        rng = np.random.default_rng(99)
        asymmetric = rng.uniform(-0.5, 0.5, size=(5, 5))
        result = design.design_synchronising_coupling(
            theta0, omega, asymmetric, 0.05, 20, max_iterations=3
        )
        np.testing.assert_allclose(result.coupling, result.coupling.T, atol=1e-12)
        assert np.all(result.coupling >= 0.0)
