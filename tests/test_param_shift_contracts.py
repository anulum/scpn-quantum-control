# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Parameter-shift contract tests
"""Contract tests for finite-dimensional parameter-shift gradients and VQE optimisation behaviour."""

from __future__ import annotations

import numpy as np


def _system(n: int = 4):
    """Standard heterogeneous Kuramoto-XY system."""
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


def _zero_coupling(n: int = 4):
    """Decoupled system — K=0, eigenstates are product states."""
    K = np.zeros((n, n))
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


class TestParamShift:
    """Tests for parameter-shift gradient rule and VQE."""

    def test_gradient_of_quadratic(self):
        from scpn_quantum_control.phase.param_shift import parameter_shift_gradient

        def quadratic(x):
            return float(x[0] ** 2 + x[1] ** 2)

        grad = parameter_shift_gradient(quadratic, np.array([1.0, 2.0]), shift=0.01)
        np.testing.assert_allclose(grad, [2.0, 4.0], atol=0.1)

    def test_gradient_of_sinusoidal(self):
        """Parameter-shift is exact for sinusoidal functions."""
        from scpn_quantum_control.phase.param_shift import parameter_shift_gradient

        def sinusoidal(x):
            return float(np.sin(x[0]) + np.cos(x[1]))

        params = np.array([0.5, 1.0])
        grad = parameter_shift_gradient(sinusoidal, params, shift=np.pi / 2)
        expected = np.array([np.cos(0.5), -np.sin(1.0)])
        np.testing.assert_allclose(grad, expected, atol=1e-10)

    def test_gradient_zero_at_minimum(self):
        from scpn_quantum_control.phase.param_shift import parameter_shift_gradient

        def cost(x):
            return float(x[0] ** 2)

        grad = parameter_shift_gradient(cost, np.array([0.0]), shift=0.01)
        assert abs(grad[0]) < 0.01

    def test_gradient_shape_matches_params(self):
        from scpn_quantum_control.phase.param_shift import parameter_shift_gradient

        for n_params in [1, 3, 5, 10]:
            grad = parameter_shift_gradient(
                lambda x: float(np.sum(x**2)),
                np.random.randn(n_params),
                shift=0.01,
            )
            assert grad.shape == (n_params,)

    def test_vqe_converges_to_minimum(self):
        from scpn_quantum_control.phase.param_shift import vqe_with_param_shift

        def cost(params):
            return float((params[0] - 1.0) ** 2 + (params[1] + 0.5) ** 2)

        result = vqe_with_param_shift(
            cost,
            n_params=2,
            learning_rate=0.05,
            n_iterations=100,
            seed=42,
        )
        assert result["energy"] < 0.1, "Should converge near minimum"

    def test_vqe_energy_monotonically_decreases(self):
        """Energy should generally decrease (allow small fluctuations)."""
        from scpn_quantum_control.phase.param_shift import vqe_with_param_shift

        result = vqe_with_param_shift(
            lambda x: float(sum(x**2)),
            n_params=3,
            learning_rate=0.05,
            n_iterations=50,
            seed=42,
        )
        # First energy should be higher than last
        assert result["energy_history"][0] > result["energy_history"][-1]

    def test_vqe_output_keys_and_types(self):
        from scpn_quantum_control.phase.param_shift import vqe_with_param_shift

        result = vqe_with_param_shift(
            lambda x: float(sum(x**2)),
            n_params=3,
            n_iterations=5,
            seed=42,
        )
        assert set(result.keys()) == {
            "optimal_params",
            "energy",
            "energy_history",
            "grad_norms",
        }
        assert isinstance(result["optimal_params"], np.ndarray)
        assert isinstance(result["energy"], float)
        assert len(result["energy_history"]) >= 5
        assert len(result["grad_norms"]) >= 5

    def test_vqe_reproducible_with_seed(self):
        from scpn_quantum_control.phase.param_shift import vqe_with_param_shift

        cost = lambda x: float(sum(x**2))  # noqa: E731
        r1 = vqe_with_param_shift(cost, n_params=3, n_iterations=10, seed=42)
        r2 = vqe_with_param_shift(cost, n_params=3, n_iterations=10, seed=42)
        np.testing.assert_array_equal(r1["optimal_params"], r2["optimal_params"])
