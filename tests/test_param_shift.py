# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Parameter-Shift Gradient Rule
"""Tests for the parameter-shift gradient module.

Covers:
    - Analytic gradient correctness against finite differences
    - Known gradients for simple cost functions
    - VQE convergence for quadratic cost
    - Edge cases: single parameter, zero gradient at minimum
    - Custom shift values
    - Gradient norm decrease during optimisation
"""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.phase.param_shift import (
    parameter_shift_gradient,
    vqe_with_param_shift,
)


def _cost_sin(p: np.ndarray) -> float:
    return float(np.sin(p[0]))


def _cost_cos(p: np.ndarray) -> float:
    return float(np.cos(p[0]))


def _cost_sin_cos_multi(p: np.ndarray) -> float:
    return float(np.sin(p[0]) + np.cos(p[1]))


def _cost_sin_cos_prod(p: np.ndarray) -> float:
    return float(np.sin(p[0]) * np.cos(p[1]) + np.sin(p[2]))


def _cost_quadratic(p: np.ndarray) -> float:
    return float(p[0] ** 2)


def _cost_sum_sin(p: np.ndarray) -> float:
    return float(np.sum(np.sin(p)))


def _cost_sum_cos(p: np.ndarray) -> float:
    return float(np.sum(np.cos(p)))


def _cost_sum_sq(p: np.ndarray) -> float:
    return float(np.sum(p**2))


def _cost_zero(p: np.ndarray) -> float:
    return 0.0


# ── Gradient correctness ──────────────────────────────────────────────


class TestParameterShiftGradient:
    def test_sin_gradient(self):
        """∂sin(θ)/∂θ = cos(θ), exact for parameter-shift with shift=π/2."""
        params = np.array([0.7])
        grad = parameter_shift_gradient(_cost_sin, params)
        np.testing.assert_allclose(grad[0], np.cos(0.7), atol=1e-12)

    def test_cos_gradient(self):
        """∂cos(θ)/∂θ = -sin(θ)."""
        params = np.array([1.3])
        grad = parameter_shift_gradient(_cost_cos, params)
        np.testing.assert_allclose(grad[0], -np.sin(1.3), atol=1e-12)

    def test_multivariate_gradient(self):
        """f(θ₀, θ₁) = sin(θ₀) + cos(θ₁), gradient is [cos(θ₀), -sin(θ₁)]."""
        params = np.array([0.5, 1.2])
        grad = parameter_shift_gradient(_cost_sin_cos_multi, params)
        np.testing.assert_allclose(grad, [np.cos(0.5), -np.sin(1.2)], atol=1e-12)

    def test_zero_gradient_at_extremum(self):
        """∂sin(θ)/∂θ = 0 at θ = π/2."""
        params = np.array([np.pi / 2])
        grad = parameter_shift_gradient(_cost_sin, params)
        np.testing.assert_allclose(grad[0], 0.0, atol=1e-12)

    def test_matches_finite_difference(self):
        """Parameter-shift should match central finite differences for sin-based cost."""
        params = np.array([0.3, 0.8, 1.5])
        grad_ps = parameter_shift_gradient(_cost_sin_cos_prod, params)

        eps = 1e-7
        grad_fd = np.zeros(3)
        for k in range(3):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[k] += eps
            p_minus[k] -= eps
            grad_fd[k] = (_cost_sin_cos_prod(p_plus) - _cost_sin_cos_prod(p_minus)) / (2 * eps)

        np.testing.assert_allclose(grad_ps, grad_fd, atol=1e-5)

    def test_custom_shift(self):
        """Non-standard shift value still gives correct gradient for sin."""
        shift = np.pi / 4
        params = np.array([0.9])
        grad = parameter_shift_gradient(_cost_sin, params, shift=shift)
        expected = (np.sin(0.9 + shift) - np.sin(0.9 - shift)) / (2 * np.sin(shift))
        np.testing.assert_allclose(grad[0], expected, atol=1e-12)

    def test_single_parameter(self):
        """Works correctly for single parameter."""
        params = np.array([0.0])
        grad = parameter_shift_gradient(_cost_quadratic, params)
        # At 0, gradient of x² via param-shift: ((π/2)² - (-π/2)²)/(2sin(π/2)) = 0
        np.testing.assert_allclose(grad[0], 0.0, atol=1e-12)

    def test_gradient_shape(self):
        """Output shape matches input shape."""
        for n in [1, 3, 7, 15]:
            params = np.zeros(n)
            grad = parameter_shift_gradient(_cost_sum_sin, params)
            assert grad.shape == (n,), f"n={n}: shape {grad.shape}"


# ── VQE convergence ──────────────────────────────────────────────────


class TestVQEWithParamShift:
    def test_converges_to_minimum(self):
        """VQE should find minimum of sum-of-cosines cost."""
        result = vqe_with_param_shift(
            _cost_sum_cos, n_params=2, learning_rate=0.3, n_iterations=100, seed=42
        )
        assert result["energy"] < -1.8, f"energy={result['energy']}"

    def test_energy_history_length(self):
        """Energy history has n_iterations + 1 entries."""
        result = vqe_with_param_shift(_cost_sum_sq, 3, n_iterations=50, seed=1)
        assert len(result["energy_history"]) == 51

    def test_grad_norms_length(self):
        """Gradient norms has n_iterations entries."""
        result = vqe_with_param_shift(_cost_sum_sq, 3, n_iterations=50, seed=1)
        assert len(result["grad_norms"]) == 50

    def test_output_keys(self):
        """Result dict has all expected keys."""
        result = vqe_with_param_shift(_cost_zero, 2, n_iterations=5, seed=0)
        assert set(result.keys()) == {
            "optimal_params",
            "energy",
            "energy_history",
            "grad_norms",
        }

    def test_optimal_params_shape(self):
        """optimal_params has correct shape."""
        result = vqe_with_param_shift(_cost_sum_sin, n_params=5, n_iterations=10, seed=7)
        assert result["optimal_params"].shape == (5,)

    def test_seed_reproducibility(self):
        """Same seed gives identical results."""
        r1 = vqe_with_param_shift(_cost_sum_cos, 3, n_iterations=20, seed=42)
        r2 = vqe_with_param_shift(_cost_sum_cos, 3, n_iterations=20, seed=42)
        np.testing.assert_array_equal(r1["optimal_params"], r2["optimal_params"])
        assert r1["energy"] == r2["energy"]

    def test_gradient_norm_decreases(self):
        """For a well-behaved cost, gradient norm should generally decrease."""
        result = vqe_with_param_shift(
            _cost_sum_cos, 2, learning_rate=0.2, n_iterations=80, seed=42
        )
        norms = result["grad_norms"]
        assert np.mean(norms[-10:]) < np.mean(norms[:10])

    def test_single_iteration(self):
        """Works with n_iterations=1."""
        result = vqe_with_param_shift(_cost_sum_sq, 2, n_iterations=1, seed=0)
        assert len(result["energy_history"]) == 2
        assert len(result["grad_norms"]) == 1
