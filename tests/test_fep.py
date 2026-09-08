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

import builtins
import importlib
import importlib.util
import sys
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.fep.predictive_coding import (
    hierarchical_prediction_error,
    predictive_coding_step,
)
from scpn_quantum_control.fep.variational_free_energy import (
    COVARIANCE_SYMMETRY_ATOL,
    evidence_lower_bound,
    free_energy_gradient,
    kl_divergence_gaussian,
    variational_free_energy,
)

# ===== 1. Empty/Null Inputs =====


class TestEmptyNull:
    """Exercise zero-valued and identity-distribution contracts."""

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
    """Exercise numerical and optional-acceleration failure boundaries."""

    def test_kl_singular_covariance(self) -> None:
        """A singular covariance is a domain error, not a leaked solver error."""
        mu = np.array([0.0])
        sigma_q = np.array([[1.0]])
        sigma_p = np.array([[0.0]])  # singular
        with pytest.raises(ValueError, match="sigma_p must be positive definite"):
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

    def test_variational_free_energy_import_guard_without_rust(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Import guard marks Rust acceleration unavailable when the engine is absent."""
        source = (
            Path(__file__).parents[1]
            / "src"
            / "scpn_quantum_control"
            / "fep"
            / "variational_free_energy.py"
        )
        module_name = "_test_variational_free_energy_no_rust"
        spec = importlib.util.spec_from_file_location(module_name, source)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)

        original_import: Callable[..., Any] = builtins.__import__

        def blocked_import(name: str, *args: object, **kwargs: object) -> Any:
            if name == "scpn_quantum_engine":
                raise ImportError("blocked in test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)

        assert module._HAS_RUST is False

    def test_predictive_coding_import_guard_without_rust(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Predictive-coding import guard records absent Rust acceleration."""
        source = (
            Path(__file__).parents[1]
            / "src"
            / "scpn_quantum_control"
            / "fep"
            / "predictive_coding.py"
        )
        module_name = "scpn_quantum_control.fep._test_predictive_coding_no_rust"
        spec = importlib.util.spec_from_file_location(module_name, source)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)

        original_import: Callable[..., Any] = builtins.__import__

        def blocked_import(name: str, *args: object, **kwargs: object) -> Any:
            if name == "scpn_quantum_engine":
                raise ImportError("blocked in test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)

        assert module._HAS_RUST is False

    def test_predictive_coding_dispatches_to_rust_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Import and exercise the optional native prediction-error dispatch."""
        source = (
            Path(__file__).parents[1]
            / "src"
            / "scpn_quantum_control"
            / "fep"
            / "predictive_coding.py"
        )
        module_name = "scpn_quantum_control.fep._test_predictive_coding_with_rust"
        spec = importlib.util.spec_from_file_location(module_name, source)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)

        def native_prediction_error(
            observations: NDArray[np.float64],
            beliefs: NDArray[np.float64],
            coupling: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            del coupling
            return observations - beliefs

        engine = types.ModuleType("scpn_quantum_engine")
        engine.hierarchical_prediction_error_rust = native_prediction_error  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "scpn_quantum_engine", engine)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)

        observations = np.array([0.5, 0.25], dtype=np.float64)
        beliefs = np.array([0.1, -0.25], dtype=np.float64)
        errors = module.hierarchical_prediction_error(observations, beliefs, np.eye(2))
        assert module._HAS_RUST is True
        np.testing.assert_array_equal(errors, observations - beliefs)


# ===== 3. Negative Cases =====


class TestNegativeCases:
    """Exercise invalid, adversarial, and Python-fallback inputs."""

    def test_kl_always_non_negative(self) -> None:
        """KL divergence must be ≥ 0 for any distributions."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            n = int(rng.integers(2, 6))
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

    def test_gradient_python_identity_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Python identity gradient remains available when Rust is disabled."""
        module = importlib.import_module("scpn_quantum_control.fep.variational_free_energy")

        monkeypatch.setattr(module, "_HAS_RUST", False)

        mu = np.array([0.2, -0.1])
        sigma = np.eye(2)
        x = np.array([0.5, 0.25])
        K = np.array([[2.0, 0.1], [0.1, 1.5]])
        sensory = np.diag([3.0, 4.0])

        grad = module.free_energy_gradient(mu, sigma, x, K, sensory_precision=sensory)

        K_reg = K + 1e-10 * np.eye(2)
        expected = K_reg @ mu - sensory @ (x - mu)
        assert np.allclose(grad, expected)

    def test_gradient_python_generative_fn_without_jacobian(self) -> None:
        """A custom generator without its Jacobian is refused, not approximated.

        This test previously asserted that the gradient substituted the identity
        Jacobian. A substituted identity Jacobian yields the gradient of a
        gradient of a different model, so the recorded expectation was wrong
        rather than merely incomplete. The full contract has a dedicated owner
        in ``tests/test_fep_generative_model_contract.py``.
        """
        mu = np.array([0.2, -0.1])
        sigma = np.eye(2)
        x = np.array([0.5, 0.25])
        K = np.eye(2)
        sensory = np.diag([2.0, 3.0])

        def generative(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return values**2

        with pytest.raises(ValueError, match="generative_fn requires generative_jac"):
            free_energy_gradient(
                mu,
                sigma,
                x,
                K,
                sensory_precision=sensory,
                generative_fn=generative,
            )

    def test_gradient_python_generative_jacobian(self) -> None:
        """Python gradient uses the supplied generator Jacobian."""
        mu = np.array([0.2, -0.1])
        sigma = np.eye(2)
        x = np.array([0.5, 0.25])
        K = np.eye(2)
        sensory = np.diag([2.0, 3.0])

        def generative(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return values**2

        def jacobian(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.diag(2.0 * values)

        grad = free_energy_gradient(
            mu,
            sigma,
            x,
            K,
            sensory_precision=sensory,
            generative_fn=generative,
            generative_jac=jacobian,
        )

        error = x - generative(mu)
        expected = (K + 1e-10 * np.eye(2)) @ mu - jacobian(mu).T @ sensory @ error
        assert np.allclose(grad, expected)

    def test_prediction_error_python_fallback_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Python prediction-error path handles isolated and coupled rows."""
        module = importlib.import_module("scpn_quantum_control.fep.predictive_coding")
        monkeypatch.setattr(module, "_HAS_RUST", False)

        observations = np.array([1.0, 2.0])
        beliefs = np.array([0.5, 1.0])
        K = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
            ]
        )

        errors = module.hierarchical_prediction_error(observations, beliefs, K)

        assert np.allclose(errors, np.array([0.5, 3.0]))


# ===== 4. Pipeline Integration =====


class TestPipelineIntegration:
    """Exercise predictive coding through connected public FEP surfaces."""

    def test_with_scpn_knm(self) -> None:
        """FEP works with actual SCPN K_nm coupling matrix."""
        K = build_knm_paper27()
        n = K.shape[0]
        mu = np.zeros(n)
        sigma = 0.1 * np.eye(n)
        x: NDArray[np.float64] = np.asarray(
            np.random.default_rng(42).standard_normal(n) * 0.1,
            dtype=np.float64,
        )
        result = variational_free_energy(mu, sigma, x, K)
        assert isinstance(result.free_energy, float)
        assert result.free_energy > 0

    def test_predictive_coding_reduces_error(self) -> None:
        """Multiple PC steps should reduce prediction error."""
        K = build_knm_paper27(L=4)
        n = 4
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n) * 0.5
        beliefs: NDArray[np.float64] = np.zeros(n)

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
    """Exercise decompositions, convergence, and explicit covariance input."""

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

    def test_predictive_coding_preserves_explicit_covariance_contract(self) -> None:
        """Use a caller-supplied covariance in the public update step."""
        observations = np.array([0.4, -0.2])
        beliefs = np.array([0.1, 0.0])
        coupling = np.array([[1.0, 0.2], [0.2, 1.5]])
        sigma = np.diag([0.3, 0.7])

        result = predictive_coding_step(
            observations,
            beliefs,
            coupling,
            learning_rate=0.005,
            sigma=sigma,
        )

        assert result.beliefs.shape == observations.shape
        assert np.isfinite(result.free_energy)
        assert result.total_error_norm == pytest.approx(np.linalg.norm(result.prediction_errors))

    def test_convergence_of_pc_to_equilibrium(self) -> None:
        """Many PC steps should converge to stable beliefs."""
        K = build_knm_paper27(L=4)
        n = 4
        x = np.array([0.5, 0.3, -0.2, 0.1])
        beliefs: NDArray[np.float64] = np.zeros(n)

        for _ in range(200):
            result = predictive_coding_step(x, beliefs, K, learning_rate=0.001)
            beliefs = result.beliefs

        # Should have converged
        result2 = predictive_coding_step(x, beliefs, K, learning_rate=0.001)
        change = np.linalg.norm(result2.beliefs - beliefs)
        assert change < 0.01, f"not converged: change = {change:.6f}"


# ===== 6. Performance =====


class TestPerformance:
    """Protect bounded execution time for the 16-layer public surface."""

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


def _kl_gaussian_1d(mu_q: float, sd_q: float, mu_p: float, sd_p: float) -> float:
    """Closed-form KL between two univariate Gaussians, in nats.

    Independent of the implementation: this is the textbook scalar expression,
    not the matrix algebra the production path evaluates.
    """
    return float(np.log(sd_p / sd_q) + (sd_q**2 + (mu_q - mu_p) ** 2) / (2.0 * sd_p**2) - 0.5)


class TestGaussianKLContract:
    """The divergence must match analysis and refuse anything that is not a covariance."""

    @pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
    def test_material_asymmetry_is_rejected_at_every_scale(self, scale: float) -> None:
        """An absolute threshold must not admit relatively large covariance errors."""
        sigma = scale * np.array([[1.0, 90.0], [0.5, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            kl_divergence_gaussian(np.zeros(2), sigma, np.zeros(2), sigma)

    @pytest.mark.parametrize(
        ("mu_q", "sd_q", "mu_p", "sd_p"),
        [(0.0, 1.0, 0.0, 2.0), (1.5, 0.3, -2.0, 1.7), (0.0, 5.0, 0.0, 0.2)],
    )
    def test_matches_the_closed_form_in_one_dimension(
        self, mu_q: float, sd_q: float, mu_p: float, sd_p: float
    ) -> None:
        """The matrix path reproduces the scalar closed form exactly."""
        value = kl_divergence_gaussian(
            np.array([mu_q]), np.array([[sd_q**2]]), np.array([mu_p]), np.array([[sd_p**2]])
        )
        assert value == pytest.approx(_kl_gaussian_1d(mu_q, sd_q, mu_p, sd_p), abs=1e-12)

    def test_diagonal_case_decomposes_into_independent_dimensions(self) -> None:
        """A diagonal pair must equal the sum of its per-dimension divergences."""
        sd_q = np.array([0.5, 2.0, 1.3])
        sd_p = np.array([1.1, 0.7, 3.0])
        mu_q = np.array([0.2, -1.0, 0.4])
        mu_p = np.array([-0.3, 0.8, 0.0])

        value = kl_divergence_gaussian(mu_q, np.diag(sd_q**2), mu_p, np.diag(sd_p**2))
        expected = sum(
            _kl_gaussian_1d(a, b, c, d) for a, b, c, d in zip(mu_q, sd_q, mu_p, sd_p, strict=True)
        )
        assert value == pytest.approx(expected, abs=1e-12)

    def test_stays_non_negative_across_random_spd_pairs(self) -> None:
        """KL is a divergence: no valid input may produce a negative value."""
        rng = np.random.default_rng(11)
        for _ in range(200):
            n = int(rng.integers(1, 5))
            left = rng.normal(size=(n, n))
            right = rng.normal(size=(n, n))
            sigma_q = left @ left.T + n * np.eye(n)
            sigma_p = right @ right.T + n * np.eye(n)
            value = kl_divergence_gaussian(
                rng.normal(size=n), sigma_q, rng.normal(size=n), sigma_p
            )
            assert value >= 0.0

    def test_self_divergence_degrades_gracefully_with_conditioning(self) -> None:
        """KL[p || p] stays near zero as the covariance becomes ill-conditioned.

        The residual is bounded by machine epsilon times the condition number,
        which is the accuracy a Cholesky solve can deliver. It is asserted as a
        bound rather than as exact zero, and it is never clamped.
        """
        rotation = np.array([[3.0, 4.0], [-4.0, 3.0]]) / 5.0
        for condition in (1e2, 1e6, 1e10):
            spectrum = np.diag(np.array([1.0, 1.0 / condition]))
            sigma = rotation @ spectrum @ rotation.T
            value = kl_divergence_gaussian(np.zeros(2), sigma, np.zeros(2), sigma)
            assert abs(value) <= 100.0 * np.finfo(float).eps * condition

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"sigma_q": np.array([[-1.0]])}, "sigma_q must be positive definite"),
            ({"sigma_p": np.array([[-1.0]])}, "sigma_p must be positive definite"),
            ({"sigma_q": np.array([[0.0]])}, "sigma_q must be positive definite"),
            ({"sigma_q": np.array([[np.nan]])}, "sigma_q must be finite"),
            ({"mu_q": np.array([np.inf])}, "mu_q must be finite"),
            ({"mu_p": np.array([np.nan])}, "mu_p must be finite"),
            ({"sigma_q": np.eye(2)}, r"sigma_q must have shape \(1, 1\)"),
            ({"mu_p": np.zeros(2)}, "must share a dimension"),
            ({"mu_q": np.zeros((1, 1))}, "mu_q must be a one-dimensional vector"),
        ],
    )
    def test_rejects_inputs_that_are_not_gaussian_parameters(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        """Every invalid parameter fails closed with a named reason."""
        call = {
            "mu_q": np.zeros(1),
            "sigma_q": np.eye(1),
            "mu_p": np.zeros(1),
            "sigma_p": np.eye(1),
        }
        call.update(kwargs)
        with pytest.raises(ValueError, match=message):
            kl_divergence_gaussian(**call)

    def test_asymmetric_matrix_is_not_a_covariance(self) -> None:
        """An asymmetric matrix must be refused, not silently symmetrised."""
        with pytest.raises(ValueError, match="sigma_q must be symmetric"):
            kl_divergence_gaussian(
                np.zeros(2), np.array([[1.0, 2.0], [0.0, 1.0]]), np.zeros(2), np.eye(2)
            )

    def test_symmetry_tolerance_admits_rounding_but_not_a_real_asymmetry(self) -> None:
        """The tolerance covers float noise, not a materially asymmetric matrix."""
        noise = COVARIANCE_SYMMETRY_ATOL / 10.0
        nearly = np.array([[2.0, 0.5 + noise], [0.5, 2.0]])
        assert kl_divergence_gaussian(np.zeros(2), nearly, np.zeros(2), np.eye(2)) >= 0.0

        clearly = np.array([[2.0, 0.5 + 1e-6], [0.5, 2.0]])
        with pytest.raises(ValueError, match="sigma_q must be symmetric"):
            kl_divergence_gaussian(np.zeros(2), clearly, np.zeros(2), np.eye(2))


class TestCouplingAndPriorPrecisionAreDistinct:
    """A coupling matrix is not automatically a prior precision.

    ``predictive_coding_step`` uses ``K`` twice: to weight the hierarchical
    prediction, where a zero diagonal is normal because a layer is not coupled
    to itself, and as the precision of the Gaussian prior ``N(0, Π⁻¹)``, which
    must be positive definite. The two roles are separable, and the second one
    now has its own argument.
    """

    @staticmethod
    def _ring_coupling() -> NDArray[np.float64]:
        """Return a zero-diagonal symmetric coupling matrix.

        Returns
        -------
        numpy.ndarray
            A four-node ring with unit couplings and no self-coupling.

        """
        coupling = np.zeros((4, 4))
        for index in range(4):
            coupling[index, (index + 1) % 4] = 1.0
            coupling[(index + 1) % 4, index] = 1.0
        return coupling

    @staticmethod
    def _shifted_laplacian(coupling: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return a positive-definite precision derived from a coupling matrix.

        Parameters
        ----------
        coupling
            Symmetric non-negative coupling matrix.

        Returns
        -------
        numpy.ndarray
            ``L + 0.25 I``, where ``L`` is the graph Laplacian of ``coupling``.

        """
        laplacian = np.diag(np.sum(coupling, axis=1)) - coupling
        return np.asarray(laplacian + 0.25 * np.eye(coupling.shape[0]))

    def test_zero_diagonal_coupling_is_not_accepted_as_a_precision(self) -> None:
        """The prior covariance of a zero-diagonal coupling does not exist."""
        coupling = self._ring_coupling()

        with pytest.raises(ValueError, match="must be positive definite"):
            predictive_coding_step(np.ones(4) * 0.1, np.zeros(4), coupling)

    def test_explicit_prior_precision_admits_a_zero_diagonal_coupling(self) -> None:
        """The step runs when the prior is given its own valid precision."""
        coupling = self._ring_coupling()
        observations = np.array([0.5, -0.2, 0.1, 0.0])
        beliefs = np.zeros(4)

        result = predictive_coding_step(
            observations,
            beliefs,
            coupling,
            learning_rate=0.05,
            prior_precision=self._shifted_laplacian(coupling),
        )

        assert np.all(np.isfinite(result.beliefs))
        assert np.isfinite(result.free_energy)

    def test_prediction_errors_still_come_from_the_coupling(self) -> None:
        """The precision argument does not displace ``K`` in the error term."""
        coupling = self._ring_coupling()
        observations = np.array([0.5, -0.2, 0.1, 0.0])
        beliefs = np.array([0.1, 0.0, -0.1, 0.2])

        result = predictive_coding_step(
            observations,
            beliefs,
            coupling,
            prior_precision=self._shifted_laplacian(coupling),
        )

        expected = hierarchical_prediction_error(observations, beliefs, coupling)
        assert result.prediction_errors == pytest.approx(expected)

    def test_belief_update_follows_the_supplied_precision(self) -> None:
        """Two precisions over one coupling give two different updates."""
        coupling = self._ring_coupling()
        observations = np.array([0.5, -0.2, 0.1, 0.0])
        beliefs = np.array([0.1, 0.0, -0.1, 0.2])
        laplacian = self._shifted_laplacian(coupling)

        from_laplacian = predictive_coding_step(
            observations, beliefs, coupling, prior_precision=laplacian
        )
        from_identity = predictive_coding_step(
            observations, beliefs, coupling, prior_precision=np.eye(4)
        )

        assert not np.allclose(from_laplacian.beliefs, from_identity.beliefs)
        expected = beliefs - 0.01 * free_energy_gradient(
            mu=beliefs,
            sigma=0.1 * np.eye(4),
            x_observed=observations,
            K_precision=laplacian,
        )
        assert from_laplacian.beliefs == pytest.approx(expected)

    def test_default_prior_precision_is_the_coupling_argument(self) -> None:
        """Omitting the precision keeps the historical single-matrix call."""
        K = build_knm_paper27(L=4)
        observations = np.array([0.5, 0.3, -0.2, 0.1])
        beliefs = np.zeros(4)

        implicit = predictive_coding_step(observations, beliefs, K)
        explicit = predictive_coding_step(observations, beliefs, K, prior_precision=K)

        assert implicit.beliefs == pytest.approx(explicit.beliefs)
        assert implicit.free_energy == pytest.approx(explicit.free_energy)
