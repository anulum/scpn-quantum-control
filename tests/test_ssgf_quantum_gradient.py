# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Ssgf Quantum Gradient
"""Tests for SSGF quantum gradient."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.ssgf.quantum_gradient import (
    QuantumGradientResult,
    _w_from_z,
    compute_quantum_gradient,
    quantum_cost,
)


class TestWFromZ:
    """Exercise latent-vector to geometry-matrix construction."""

    def test_shape(self) -> None:
        """The matrix dimension matches the oscillator count."""
        W = _w_from_z(np.zeros(6), 4)
        assert W.shape == (4, 4)

    def test_symmetric(self) -> None:
        """The generated coupling matrix is symmetric."""
        z = np.random.default_rng(42).normal(size=6)
        W = _w_from_z(z, 4)
        np.testing.assert_allclose(W, W.T, atol=1e-12)

    def test_non_negative(self) -> None:
        """Softplus parameterization produces non-negative couplings."""
        z = np.random.default_rng(42).normal(size=6)
        W = _w_from_z(z, 4)
        assert np.all(W >= 0)

    def test_zero_diagonal(self) -> None:
        """The geometry does not contain self-coupling."""
        z = np.random.default_rng(42).normal(size=6)
        W = _w_from_z(z, 4)
        np.testing.assert_allclose(np.diag(W), 0.0)

    def test_larger_z_stronger_coupling(self) -> None:
        """Larger latent values yield stronger aggregate coupling."""
        W_weak = _w_from_z(np.full(6, -2.0), 4)
        W_strong = _w_from_z(np.full(6, 2.0), 4)
        assert np.sum(W_strong) > np.sum(W_weak)


class TestQuantumCost:
    """Exercise the evolved synchronization cost."""

    def test_cost_bounded(self) -> None:
        """Quantum synchronization cost remains in the unit interval."""
        W = np.array([[0, 0.5], [0.5, 0]])
        theta = np.array([0.0, 0.1])
        c = quantum_cost(W, theta)
        assert 0 <= c <= 1.0

    def test_synchronized_state_low_cost(self) -> None:
        """All phases aligned → high R → low cost."""
        W = np.array([[0, 1.0], [1.0, 0]])
        theta = np.array([0.0, 0.0])
        c = quantum_cost(W, theta, dt=0.01)
        assert c < 0.5

    def test_cost_with_omega(self) -> None:
        """Explicit natural frequencies produce a scalar cost."""
        W = np.array([[0, 0.5], [0.5, 0]])
        theta = np.array([0.0, 0.5])
        omega = np.array([1.0, 1.5])
        c = quantum_cost(W, theta, omega=omega)
        assert isinstance(c, float)

    def test_4_oscillators(self) -> None:
        """The cost supports a four-oscillator coupling matrix."""
        n = 4
        W = np.ones((n, n)) * 0.3
        np.fill_diagonal(W, 0.0)
        theta = np.linspace(0, np.pi / 2, n)
        c = quantum_cost(W, theta)
        assert 0 <= c <= 1.0


class TestComputeQuantumGradient:
    """Exercise the complete central finite-difference gradient."""

    def test_returns_result(self) -> None:
        """The public computation returns its result record."""
        z = np.zeros(3)
        result = compute_quantum_gradient(z, n_osc=3)
        assert isinstance(result, QuantumGradientResult)

    def test_gradient_shape(self) -> None:
        """The gradient shape matches the latent-vector shape."""
        z = np.zeros(6)
        result = compute_quantum_gradient(z, n_osc=4)
        assert result.gradient.shape == (6,)

    def test_r_global_bounded(self) -> None:
        """The central global order parameter remains bounded."""
        z = np.ones(3)
        result = compute_quantum_gradient(z, n_osc=3)
        assert 0 <= result.r_global <= 1.0

    def test_cost_matches_r(self) -> None:
        """Central cost equals one minus the global order parameter."""
        z = np.ones(3)
        result = compute_quantum_gradient(z, n_osc=3)
        assert result.cost == pytest.approx(1.0 - result.r_global, abs=1e-10)

    def test_n_evaluations(self) -> None:
        """1 center + 2 per parameter."""
        n_osc = 3
        n_upper = n_osc * (n_osc - 1) // 2  # 3
        z = np.zeros(n_upper)
        result = compute_quantum_gradient(z, n_osc=n_osc)
        assert result.n_evaluations == 1 + 2 * n_upper

    def test_gradient_nonzero_for_asymmetric_z(self) -> None:
        """An asymmetric latent geometry yields a nonzero gradient."""
        z = np.array([1.0, -1.0, 0.5])
        result = compute_quantum_gradient(z, n_osc=3)
        assert np.any(np.abs(result.gradient) > 1e-10)

    def test_custom_theta_init(self) -> None:
        """Explicit initial phases are accepted by the gradient computation."""
        z = np.zeros(3)
        theta = np.array([0.0, np.pi / 4, np.pi / 2])
        result = compute_quantum_gradient(z, n_osc=3, theta_init=theta)
        assert isinstance(result.cost, float)
