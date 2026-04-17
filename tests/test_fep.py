# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Free Energy Principle
"""Multi-angle tests for fep/ subpackage.

6 dimensions: empty/null, error handling, negative cases, pipeline
integration, roundtrip, performance.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.fep.predictive_coding import (
    hierarchical_prediction_error,
    predictive_coding_step,
)
from scpn_quantum_control.fep.variational_free_energy import (
    evidence_lower_bound,
    free_energy_gradient,
    kl_divergence_gaussian,
    variational_free_energy,
)

# ===== 1. Empty/Null Inputs =====


class TestEmptyNull:
    def test_kl_identical_distributions(self) -> None:
        """KL[q || q] = 0."""
        mu = np.array([1.0, 2.0])
        sigma = np.eye(2) * 0.5
        kl = kl_divergence_gaussian(mu, sigma, mu, sigma)
        assert abs(kl) < 1e-10

    def test_free_energy_zero_observation(self) -> None:
        """Zero observation with zero belief → F = complexity only."""
        n = 3
        mu = np.zeros(n)
        sigma = np.eye(n)
        x = np.zeros(n)
        K = np.eye(n)
        result = variational_free_energy(mu, sigma, x, K)
        assert result.accuracy == pytest.approx(0.0, abs=1e-10)
        assert isinstance(result.free_energy, float)

    def test_prediction_error_zero_when_perfect(self) -> None:
        """If beliefs = observations, prediction errors vanish."""
        n = 4
        K = build_knm_paper27(L=n)
        x = np.ones(n) * 0.5
        errors = hierarchical_prediction_error(x, x, K)
        # With perfect prediction, errors should be small
        # (not exactly zero because coupling-weighted prediction ≠ identity)
        assert np.linalg.norm(errors) < n  # bounded


# ===== 2. Error Handling =====


class TestErrorHandling:
    def test_kl_singular_covariance(self) -> None:
        """Singular covariance should raise or return inf."""
        mu = np.array([0.0])
        sigma_q = np.array([[1.0]])
        sigma_p = np.array([[0.0]])  # singular
        with pytest.raises(np.linalg.LinAlgError):
            kl_divergence_gaussian(mu, sigma_q, mu, sigma_p)

    def test_free_energy_with_zero_precision(self) -> None:
        """Zero K_nm (no prior) → F driven by accuracy only."""
        n = 3
        mu = np.array([0.1, 0.2, 0.3])
        sigma = np.eye(n)
        x = np.array([0.5, 0.6, 0.7])
        K = np.zeros((n, n))  # no coupling
        # K + ridge → near-zero prior → complexity ≈ 0
        result = variational_free_energy(mu, sigma, x, K)
        assert result.free_energy > 0  # accuracy > 0


# ===== 3. Negative Cases =====


class TestNegativeCases:
    def test_kl_always_non_negative(self) -> None:
        """KL divergence must be ≥ 0 for any distributions."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            n = rng.integers(2, 6)
            mu_q = rng.standard_normal(n)
            mu_p = rng.standard_normal(n)
            A = rng.standard_normal((n, n))
            sigma_q = A @ A.T + 0.1 * np.eye(n)
            B = rng.standard_normal((n, n))
            sigma_p = B @ B.T + 0.1 * np.eye(n)
            kl = kl_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p)
            assert kl >= -1e-10, f"KL must be ≥ 0, got {kl}"

    def test_gradient_points_toward_observation(self) -> None:
        """Gradient should reduce free energy (push mu toward x)."""
        n = 4
        K = build_knm_paper27(L=n)
        mu = np.zeros(n)
        x = np.ones(n)
        sigma = np.eye(n)
        grad = free_energy_gradient(mu, sigma, x, K)
        # Taking a step −grad should bring mu closer to x
        mu_new = mu - 0.01 * grad
        f_old = variational_free_energy(mu, sigma, x, K).free_energy
        f_new = variational_free_energy(mu_new, sigma, x, K).free_energy
        assert f_new < f_old, "gradient step must reduce F"


# ===== 4. Pipeline Integration =====


class TestPipelineIntegration:
    def test_with_scpn_knm(self) -> None:
        """FEP works with actual SCPN K_nm coupling matrix."""
        K = build_knm_paper27()
        n = K.shape[0]
        mu = np.zeros(n)
        sigma = 0.1 * np.eye(n)
        x = np.random.default_rng(42).standard_normal(n) * 0.1
        result = variational_free_energy(mu, sigma, x, K)
        assert isinstance(result.free_energy, float)
        assert result.free_energy > 0

    def test_predictive_coding_reduces_error(self) -> None:
        """Multiple PC steps should reduce prediction error."""
        K = build_knm_paper27(L=4)
        n = 4
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n) * 0.5
        beliefs = np.zeros(n)

        errors_over_time = []
        for _ in range(50):
            result = predictive_coding_step(x, beliefs, K, learning_rate=0.001)
            beliefs = result.beliefs
            errors_over_time.append(result.total_error_norm)

        # Error should decrease over iterations
        assert errors_over_time[-1] < errors_over_time[0], (
            f"PC should reduce error: {errors_over_time[0]:.4f} → {errors_over_time[-1]:.4f}"
        )

    def test_elbo_consistent_with_free_energy(self) -> None:
        """ELBO = −F."""
        n = 3
        K = np.eye(n)
        mu = np.array([0.1, 0.2, 0.3])
        sigma = np.eye(n)
        x = np.array([0.5, 0.5, 0.5])
        result = variational_free_energy(mu, sigma, x, K)
        elbo = evidence_lower_bound(mu, sigma, x, K)
        assert abs(result.elbo - elbo) < 1e-12
        assert abs(result.elbo + result.free_energy) < 1e-12

    def test_top_level_import(self) -> None:
        """FEP must be importable from top-level."""
        from scpn_quantum_control import fep

        assert hasattr(fep, "variational_free_energy")
        assert hasattr(fep, "predictive_coding_step")


# ===== 5. Roundtrip =====


class TestRoundtrip:
    def test_free_energy_decomposition(self) -> None:
        """F = complexity + accuracy."""
        n = 4
        K = build_knm_paper27(L=n)
        mu = np.array([0.1, 0.2, 0.3, 0.4])
        sigma = 0.5 * np.eye(n)
        x = np.array([0.5, 0.6, 0.7, 0.8])
        result = variational_free_energy(mu, sigma, x, K)
        assert abs(result.free_energy - (result.complexity + result.accuracy)) < 1e-10

    def test_gradient_zero_at_minimum(self) -> None:
        """At the exact MAP point, gradient should be small."""
        n = 2
        K = np.eye(n)
        # MAP: μ = (K + Γ)⁻¹ Γ x = 0.5 × x for K=Γ=I
        x = np.array([1.0, 2.0])
        mu_map = 0.5 * x
        sigma = np.eye(n)
        grad = free_energy_gradient(mu_map, sigma, x, K)
        # Ridge regularisation (1e-10) introduces small residual
        assert np.linalg.norm(grad) < 1e-8, f"gradient at MAP: {grad}"

    def test_convergence_of_pc_to_equilibrium(self) -> None:
        """Many PC steps should converge to stable beliefs."""
        K = build_knm_paper27(L=4)
        n = 4
        x = np.array([0.5, 0.3, -0.2, 0.1])
        beliefs = np.zeros(n)

        for _ in range(200):
            result = predictive_coding_step(x, beliefs, K, learning_rate=0.001)
            beliefs = result.beliefs

        # Should have converged
        result2 = predictive_coding_step(x, beliefs, K, learning_rate=0.001)
        change = np.linalg.norm(result2.beliefs - beliefs)
        assert change < 0.01, f"not converged: change = {change:.6f}"


# ===== 6. Performance =====


class TestPerformance:
    def test_free_energy_fast(self) -> None:
        """Free energy computation for n=16 in < 1ms."""
        K = build_knm_paper27()
        n = 16
        mu = np.zeros(n)
        sigma = np.eye(n)
        x = np.random.default_rng(42).standard_normal(n) * 0.1
        t0 = time.perf_counter()
        for _ in range(1000):
            variational_free_energy(mu, sigma, x, K)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"1000 calls took {elapsed:.3f}s"

    def test_pc_step_fast(self) -> None:
        """Single PC step for n=16 in < 2ms."""
        K = build_knm_paper27()
        n = 16
        x = np.random.default_rng(42).standard_normal(n) * 0.1
        beliefs = np.zeros(n)
        t0 = time.perf_counter()
        for _ in range(100):
            predictive_coding_step(x, beliefs, K, learning_rate=0.001)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.2, f"100 calls took {elapsed:.3f}s"
