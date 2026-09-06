# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — FEP Generative Model Contract Tests
"""Generative model and Jacobian contract for the free energy gradient.

The gradient of the variational free energy is
``∂F/∂μ = Π_z μ − Jᵀ Γ (x − g(μ))``. It is well defined only when the Jacobian
``J`` belongs to the model ``g`` that produced the prediction. Substituting the
identity for a missing Jacobian returns the gradient of a different model, and
because the result is a finite vector of the right length, nothing downstream
can tell it apart from the gradient that was asked for.

Every positive case below is checked against two independent oracles: a
hand-derived analytic gradient for the small system, and a central difference of
the public :func:`variational_free_energy`. The models are smooth, real and
first order, so finite differences qualify here; they are used alongside the
analytic form rather than instead of it.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.fep import free_energy_gradient as package_gradient
from scpn_quantum_control.fep.variational_free_energy import (
    PRECISION_RIDGE,
    free_energy_gradient,
    variational_free_energy,
)

FD_STEP = 1e-6
"""Central-difference step, chosen so truncation and rounding are both small."""


def _central_difference(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    x_observed: NDArray[np.float64],
    K_precision: NDArray[np.float64],
    sensory_precision: NDArray[np.float64] | None = None,
    generative_fn: Any = None,
) -> NDArray[np.float64]:
    """Differentiate the public free energy coordinate by coordinate.

    Parameters
    ----------
    mu
        Belief mean the derivative is taken at.
    sigma, x_observed, K_precision, sensory_precision, generative_fn
        Passed through to :func:`variational_free_energy` unchanged.

    Returns
    -------
    numpy.ndarray
        Central-difference estimate of ``∂F/∂μ`` with shape ``mu.shape``.

    """
    gradient = np.zeros_like(mu)
    for index in range(mu.size):
        step = np.zeros_like(mu)
        step[index] = FD_STEP
        forward = variational_free_energy(
            mu + step, sigma, x_observed, K_precision, sensory_precision, generative_fn
        ).free_energy
        backward = variational_free_energy(
            mu - step, sigma, x_observed, K_precision, sensory_precision, generative_fn
        ).free_energy
        gradient[index] = (forward - backward) / (2.0 * FD_STEP)
    return gradient


def _untyped_model(model: Callable[[NDArray[np.float64]], object]) -> Any:
    """Present a deliberately non-conforming model as the declared callable type.

    A caller can violate the static contract at runtime — from a notebook, an
    untyped plugin or a dynamically built model — so the refusal has to come
    from the function rather than only from the type checker. This helper states
    that intent instead of suppressing the resulting type error.

    Parameters
    ----------
    model
        Callable whose return type does not satisfy the declared model type.

    Returns
    -------
    Any
        The same callable, typed so it can be passed to the public functions.

    """
    return model


class _CallRecorder:
    """Callable that records how often it was invoked.

    Used to establish that an incomplete model pair is refused before either
    half of it runs, so no belief state can be derived from a partial contract.
    """

    def __init__(self, result: Any) -> None:
        """Store the value to return and start the call count at zero.

        Parameters
        ----------
        result
            Value returned by every call.

        """
        self.result = result
        self.calls = 0

    def __call__(self, values: NDArray[np.float64]) -> Any:
        """Record the call and return the stored value.

        Parameters
        ----------
        values
            Belief mean supplied by the gradient; unused.

        Returns
        -------
        Any
            The value given to the constructor.

        """
        self.calls += 1
        return self.result


class TestReviewedDefect:
    """A Jacobian supplied without its model differentiates a different model."""

    def test_recorded_case_now_matches_its_finite_difference(self) -> None:
        """g(mu)=2mu at mu=0.4 gives 2.0, the value the card derived."""
        mu = np.array([0.4])
        sigma = np.eye(1)
        x = np.zeros(1)
        K = np.eye(1)

        def model(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return 2.0 * values

        def model_jacobian(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return 2.0 * np.eye(values.size)

        gradient = free_energy_gradient(
            mu, sigma, x, K, generative_fn=model, generative_jac=model_jacobian
        )

        # Analytic: Π_z μ − Jᵀ Γ (x − g(μ)) = 0.4 − 2·(0 − 0.8) = 2.0
        assert gradient == pytest.approx(np.array([2.0]), abs=1e-9)
        difference = _central_difference(mu, sigma, x, K, generative_fn=model)
        assert gradient == pytest.approx(difference, abs=1e-6)

    def test_identity_substitution_is_gone(self) -> None:
        """The 1.2 the identity Jacobian produced is no longer reachable."""
        mu = np.array([0.4])

        def model(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return 2.0 * values

        with pytest.raises(ValueError) as excinfo:
            free_energy_gradient(mu, np.eye(1), np.zeros(1), np.eye(1), generative_fn=model)

        assert str(excinfo.value) == (
            "generative_fn requires generative_jac: no derivative is inferred, and "
            "an identity Jacobian returns the gradient of a different model"
        )


class TestIncompletePairs:
    """An incomplete model pair is refused before either half is called."""

    def test_model_without_jacobian_is_refused_before_the_model_runs(self) -> None:
        """The forward model is never evaluated on an incomplete pair."""
        model = _CallRecorder(np.zeros(2))

        with pytest.raises(ValueError, match="generative_fn requires generative_jac"):
            free_energy_gradient(
                np.zeros(2), np.eye(2), np.zeros(2), np.eye(2), generative_fn=model
            )

        assert model.calls == 0

    def test_jacobian_without_model_is_refused_before_the_jacobian_runs(self) -> None:
        """A Jacobian alone would differentiate the identity prediction."""
        model_jacobian = _CallRecorder(np.eye(2))

        with pytest.raises(ValueError) as excinfo:
            free_energy_gradient(
                np.zeros(2), np.eye(2), np.zeros(2), np.eye(2), generative_jac=model_jacobian
            )

        assert str(excinfo.value) == (
            "generative_jac requires generative_fn: a Jacobian without its model "
            "would differentiate the identity prediction"
        )
        assert model_jacobian.calls == 0

    def test_refusal_leaves_the_belief_mean_untouched(self) -> None:
        """A rejected call has no side effect on its inputs."""
        mu = np.array([0.3, -0.7])
        before = mu.copy()

        with pytest.raises(ValueError):
            free_energy_gradient(
                mu, np.eye(2), np.zeros(2), np.eye(2), generative_fn=lambda values: values
            )

        assert np.array_equal(mu, before)


class TestAffineModels:
    """Non-identity affine models agree with both oracles."""

    def test_scalar_multiple_model(self) -> None:
        """g(μ) = 3μ on two coordinates with an anisotropic precision."""
        mu = np.array([0.2, -0.1])
        sigma = np.eye(2)
        x = np.array([0.5, 0.25])
        K = np.array([[2.0, 0.1], [0.1, 1.5]])
        sensory = np.diag([2.0, 3.0])

        def model(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return 3.0 * values

        def model_jacobian(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return 3.0 * np.eye(values.size)

        gradient = free_energy_gradient(
            mu, sigma, x, K, sensory, generative_fn=model, generative_jac=model_jacobian
        )

        ridged = K + PRECISION_RIDGE * np.eye(2)
        analytic = ridged @ mu - 3.0 * np.eye(2) @ sensory @ (x - 3.0 * mu)
        assert gradient == pytest.approx(analytic, abs=1e-12)
        difference = _central_difference(mu, sigma, x, K, sensory, model)
        assert gradient == pytest.approx(difference, abs=1e-6)

    def test_asymmetric_matrix_model(self) -> None:
        """A non-symmetric A makes Jᵀ distinguishable from J."""
        mu = np.array([0.4, -0.2, 0.1])
        sigma = np.eye(3)
        x = np.array([0.1, 0.0, -0.3])
        K = np.eye(3) * 1.5
        matrix = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [4.0, 0.0, 1.0]])

        def model(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return matrix @ values

        def model_jacobian(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return matrix

        gradient = free_energy_gradient(
            mu, sigma, x, K, generative_fn=model, generative_jac=model_jacobian
        )

        analytic = (K + PRECISION_RIDGE * np.eye(3)) @ mu - matrix.T @ (x - matrix @ mu)
        assert gradient == pytest.approx(analytic, abs=1e-12)
        difference = _central_difference(mu, sigma, x, K, generative_fn=model)
        assert gradient == pytest.approx(difference, abs=1e-6)
        # The transpose matters: using J instead of Jᵀ would give another vector.
        assert not np.allclose(analytic, (K @ mu - matrix @ (x - matrix @ mu)))


class TestNonlinearModels:
    """Nonlinear models agree with both oracles."""

    def test_sine_model(self) -> None:
        """g(μ) = sin(μ), the model the reference page documents."""
        mu = np.array([0.3, -0.8, 1.2, 0.05])
        sigma = np.eye(4)
        x = np.ones(4) * 0.5
        K = np.eye(4)

        def model(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.sin(values)

        def model_jacobian(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.diag(np.cos(values))

        gradient = free_energy_gradient(
            mu, sigma, x, K, generative_fn=model, generative_jac=model_jacobian
        )

        analytic = (K + PRECISION_RIDGE * np.eye(4)) @ mu - np.diag(np.cos(mu)) @ (x - np.sin(mu))
        assert gradient == pytest.approx(analytic, abs=1e-12)
        difference = _central_difference(mu, sigma, x, K, generative_fn=model)
        assert gradient == pytest.approx(difference, abs=1e-6)

    def test_square_model_with_anisotropic_precision(self) -> None:
        """g(μ) = μ², the model whose identity-Jacobian result was wrong."""
        mu = np.array([0.2, -0.1])
        sigma = np.eye(2)
        x = np.array([0.5, 0.25])
        K = np.eye(2)
        sensory = np.diag([2.0, 3.0])

        def model(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return values**2

        def model_jacobian(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.diag(2.0 * values)

        gradient = free_energy_gradient(
            mu, sigma, x, K, sensory, generative_fn=model, generative_jac=model_jacobian
        )

        analytic = (K + PRECISION_RIDGE * np.eye(2)) @ mu - np.diag(2.0 * mu) @ sensory @ (
            x - mu**2
        )
        assert gradient == pytest.approx(analytic, abs=1e-12)
        difference = _central_difference(mu, sigma, x, K, sensory, model)
        assert gradient == pytest.approx(difference, abs=1e-6)
        # The identity Jacobian this call used to substitute gives another vector.
        substituted = (K + PRECISION_RIDGE * np.eye(2)) @ mu - sensory @ (x - mu**2)
        assert not np.allclose(analytic, substituted)


class TestRectangularModels:
    """A model whose output dimension differs from the belief dimension."""

    def test_three_predictions_from_two_beliefs(self) -> None:
        """g: R² → R³ keeps a length-2 gradient and an (3, 3) precision."""
        mu = np.array([0.4, 0.1])
        sigma = np.eye(2)
        x = np.zeros(3)
        K = np.eye(2)
        sensory = np.diag([1.0, 2.0, 0.5])
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        def model(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return matrix @ values

        def model_jacobian(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return matrix

        gradient = free_energy_gradient(
            mu, sigma, x, K, sensory, generative_fn=model, generative_jac=model_jacobian
        )

        assert gradient.shape == (2,)
        analytic = (K + PRECISION_RIDGE * np.eye(2)) @ mu - matrix.T @ sensory @ (x - matrix @ mu)
        assert gradient == pytest.approx(analytic, abs=1e-12)
        difference = _central_difference(mu, sigma, x, K, sensory, model)
        assert gradient == pytest.approx(difference, abs=1e-6)

    def test_rectangular_jacobian_of_the_wrong_orientation_is_refused(self) -> None:
        """A (n, m) Jacobian where (m, n) is required does not pass."""
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        with pytest.raises(ValueError, match=r"must return shape \(3, 2\), got \(2, 3\)"):
            free_energy_gradient(
                np.array([0.4, 0.1]),
                np.eye(2),
                np.zeros(3),
                np.eye(2),
                np.eye(3),
                generative_fn=lambda values: matrix @ values,
                generative_jac=lambda values: matrix.T,
            )


class TestMalformedModelOutput:
    """Model and Jacobian returns are rejected by contract, not by broadcasting."""

    @staticmethod
    def _gradient_with(model: Any, model_jacobian: Any) -> NDArray[np.float64]:
        """Call the gradient on a fixed two-dimensional system.

        Parameters
        ----------
        model
            Forward model passed as ``generative_fn``.
        model_jacobian
            Jacobian passed as ``generative_jac``.

        Returns
        -------
        numpy.ndarray
            The computed gradient, for calls that are expected to succeed.

        """
        return free_energy_gradient(
            np.array([0.4, 0.1]),
            np.eye(2),
            np.zeros(2),
            np.eye(2),
            generative_fn=model,
            generative_jac=model_jacobian,
        )

    def test_scalar_prediction_is_refused(self) -> None:
        """A scalar return would broadcast across every observation."""
        with pytest.raises(ValueError, match=r"one-dimensional vector, got shape \(\)"):
            self._gradient_with(lambda values: 1.0, lambda values: np.eye(2))

    def test_column_vector_prediction_is_refused(self) -> None:
        """An (n, 1) return would form an outer product, not an error vector."""
        with pytest.raises(ValueError, match=r"one-dimensional vector, got shape \(2, 1\)"):
            self._gradient_with(lambda values: values.reshape(-1, 1), lambda values: np.eye(2))

    def test_wrong_length_prediction_is_refused(self) -> None:
        """A prediction must explain exactly the observations supplied."""
        with pytest.raises(ValueError, match="must return 2 values, got 3"):
            self._gradient_with(lambda values: np.ones(3), lambda values: np.eye(2))

    def test_non_finite_prediction_is_refused(self) -> None:
        """A NaN prediction would return a NaN gradient and poison the belief."""
        with pytest.raises(ValueError, match=r"generative_fn\(mu\) must return finite values"):
            self._gradient_with(lambda values: values * np.nan, lambda values: np.eye(2))

    def test_one_dimensional_jacobian_is_refused(self) -> None:
        """A vector Jacobian contracts into a plausible but wrong gradient."""
        with pytest.raises(ValueError, match=r"must return shape \(2, 2\), got \(2,\)"):
            self._gradient_with(lambda values: 2.0 * values, lambda values: np.array([2.0, 2.0]))

    def test_non_finite_jacobian_is_refused(self) -> None:
        """An infinite Jacobian entry is rejected before the product."""
        with pytest.raises(ValueError, match=r"generative_jac\(mu\) must return finite values"):
            self._gradient_with(
                lambda values: 2.0 * values, lambda values: np.full((2, 2), np.inf)
            )

    def test_valid_pair_still_passes_the_same_guard(self) -> None:
        """The shared helper accepts the well-formed pair it rejects others by."""
        gradient = self._gradient_with(lambda values: 2.0 * values, lambda values: 2.0 * np.eye(2))
        assert gradient == pytest.approx(np.array([2.0, 0.5]), abs=1e-9)


class TestArgumentDomain:
    """Arguments outside the admitted domain are named in the refusal."""

    def test_non_finite_belief_is_refused(self) -> None:
        """An infinite belief mean cannot start a gradient step."""
        with pytest.raises(ValueError, match="mu must be finite"):
            free_energy_gradient(np.array([np.inf, 0.0]), np.eye(2), np.zeros(2), np.eye(2))

    def test_two_dimensional_belief_is_refused(self) -> None:
        """The belief mean is a vector, not a matrix."""
        with pytest.raises(ValueError, match="mu must be a one-dimensional vector"):
            free_energy_gradient(np.zeros((2, 1)), np.eye(2), np.zeros(2), np.eye(2))

    def test_non_finite_observation_is_refused(self) -> None:
        """A non-finite observation has no prediction error."""
        with pytest.raises(ValueError, match="x_observed must be finite"):
            free_energy_gradient(np.zeros(2), np.eye(2), np.array([np.nan, 0.0]), np.eye(2))

    def test_identity_model_requires_matching_observations(self) -> None:
        """Without a model, observations and beliefs share one space."""
        with pytest.raises(ValueError, match="x_observed must have length 2"):
            free_energy_gradient(np.zeros(2), np.eye(2), np.zeros(3), np.eye(2))

    def test_prior_precision_shape_is_checked(self) -> None:
        """A mis-shaped prior precision is named rather than left to matmul."""
        with pytest.raises(
            ValueError, match=r"K_precision must have shape \(2, 2\), got \(3, 3\)"
        ):
            free_energy_gradient(np.zeros(2), np.eye(2), np.zeros(2), np.eye(3))

    def test_non_finite_prior_precision_is_refused(self) -> None:
        """A non-finite prior precision cannot weight the belief."""
        with pytest.raises(ValueError, match="K_precision must be finite"):
            free_energy_gradient(np.zeros(2), np.eye(2), np.zeros(2), np.full((2, 2), np.nan))

    def test_sensory_precision_shape_follows_the_prediction(self) -> None:
        """For g: R² → R³ the precision is (3, 3), not (2, 2)."""
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        with pytest.raises(
            ValueError, match=r"sensory_precision must have shape \(3, 3\), got \(2, 2\)"
        ):
            free_energy_gradient(
                np.array([0.4, 0.1]),
                np.eye(2),
                np.zeros(3),
                np.eye(2),
                np.eye(2),
                generative_fn=lambda values: matrix @ values,
                generative_jac=lambda values: matrix,
            )

    def test_non_finite_sensory_precision_is_refused(self) -> None:
        """A non-finite precision cannot weight the prediction error."""
        with pytest.raises(ValueError, match="sensory_precision must be finite"):
            free_energy_gradient(
                np.zeros(2), np.eye(2), np.zeros(2), np.eye(2), np.full((2, 2), np.inf)
            )


class TestIdentityModelUnchanged:
    """The identity path keeps the semantics the Rust engine implements."""

    def test_identity_gradient_matches_its_analytic_form(self) -> None:
        """Π_z μ − Γ (x − μ) with the documented ridge."""
        mu = np.array([0.2, -0.1])
        x = np.array([0.5, 0.25])
        K = np.array([[2.0, 0.1], [0.1, 1.5]])
        sensory = np.diag([3.0, 4.0])

        gradient = free_energy_gradient(mu, np.eye(2), x, K, sensory_precision=sensory)

        analytic = (K + PRECISION_RIDGE * np.eye(2)) @ mu - sensory @ (x - mu)
        assert gradient == pytest.approx(analytic, abs=1e-12)

    def test_identity_gradient_matches_its_finite_difference(self) -> None:
        """The identity path is differentiated by the same public energy."""
        mu = np.array([0.2, -0.1])
        sigma = np.eye(2)
        x = np.array([0.5, 0.25])
        K = np.array([[2.0, 0.1], [0.1, 1.5]])
        sensory = np.diag([3.0, 4.0])

        gradient = free_energy_gradient(mu, sigma, x, K, sensory_precision=sensory)
        difference = _central_difference(mu, sigma, x, K, sensory)

        assert gradient == pytest.approx(difference, abs=1e-6)

    def test_explicit_identity_pair_reproduces_the_identity_path(self) -> None:
        """Supplying the identity model and its Jacobian changes nothing."""
        mu = np.array([0.2, -0.1])
        x = np.array([0.5, 0.25])
        K = np.eye(2)

        implicit = free_energy_gradient(mu, np.eye(2), x, K)
        explicit = free_energy_gradient(
            mu,
            np.eye(2),
            x,
            K,
            generative_fn=lambda values: values,
            generative_jac=lambda values: np.eye(values.size),
        )

        assert implicit == pytest.approx(explicit, abs=1e-12)


class TestRustDispatch:
    """Identity dispatch to the native engine, with the same admitted domain."""

    def test_native_engine_receives_the_validated_identity_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The engine is reached only after the arguments are validated."""
        # The package attribute of this name is the function, not the module.
        module = importlib.import_module("scpn_quantum_control.fep.variational_free_energy")

        seen: dict[str, Any] = {}

        def fake_gradient(
            mu: NDArray[np.float64],
            x_observed: NDArray[np.float64],
            k_precision: NDArray[np.float64],
            sensory_precision: NDArray[np.float64],
            ridge: float,
        ) -> NDArray[np.float64]:
            seen.update(
                mu=mu,
                x_observed=x_observed,
                k_precision=k_precision,
                sensory_precision=sensory_precision,
                ridge=ridge,
            )
            return np.array([7.0, 8.0])

        monkeypatch.setattr(module, "_HAS_RUST", True)
        monkeypatch.setattr(module, "_grad_rust", fake_gradient, raising=False)

        gradient = module.free_energy_gradient(
            np.array([0.2, -0.1]), np.eye(2), np.array([0.5, 0.25]), np.eye(2)
        )

        assert gradient == pytest.approx(np.array([7.0, 8.0]))
        assert seen["ridge"] == PRECISION_RIDGE
        assert seen["sensory_precision"] == pytest.approx(np.eye(2))
        assert seen["mu"].dtype == np.float64

    def test_native_engine_is_not_reached_by_a_custom_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The engine implements the identity model only."""
        # The package attribute of this name is the function, not the module.
        module = importlib.import_module("scpn_quantum_control.fep.variational_free_energy")

        def fail(*args: Any, **kwargs: Any) -> NDArray[np.float64]:
            raise AssertionError("the identity engine must not receive a custom model")

        monkeypatch.setattr(module, "_HAS_RUST", True)
        monkeypatch.setattr(module, "_grad_rust", fail, raising=False)

        gradient = module.free_energy_gradient(
            np.array([0.4]),
            np.eye(1),
            np.zeros(1),
            np.eye(1),
            generative_fn=lambda values: 2.0 * values,
            generative_jac=lambda values: 2.0 * np.eye(values.size),
        )

        assert gradient == pytest.approx(np.array([2.0]), abs=1e-9)

    def test_native_engine_is_not_reached_by_an_invalid_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validation precedes dispatch, so both tiers see one domain."""
        # The package attribute of this name is the function, not the module.
        module = importlib.import_module("scpn_quantum_control.fep.variational_free_energy")

        def fail(*args: Any, **kwargs: Any) -> NDArray[np.float64]:
            raise AssertionError("the engine must not receive an unvalidated call")

        monkeypatch.setattr(module, "_HAS_RUST", True)
        monkeypatch.setattr(module, "_grad_rust", fail, raising=False)

        with pytest.raises(ValueError, match="x_observed must be finite"):
            module.free_energy_gradient(np.zeros(2), np.eye(2), np.array([np.nan, 0.0]), np.eye(2))


class TestFreeEnergySharesTheContract:
    """The energy and its gradient admit the same models."""

    def test_energy_rejects_a_scalar_prediction(self) -> None:
        """A scalar prediction used to be accepted and broadcast."""
        with pytest.raises(ValueError, match=r"one-dimensional vector, got shape \(\)"):
            variational_free_energy(
                np.array([0.4, 0.1]),
                np.eye(2),
                np.zeros(2),
                np.eye(2),
                generative_fn=_untyped_model(lambda values: 1.0),
            )

    def test_energy_rejects_mismatched_identity_observations(self) -> None:
        """Without a model the observation space is the belief space."""
        with pytest.raises(ValueError, match="x_observed must have length 2"):
            variational_free_energy(np.zeros(2), np.eye(2), np.zeros(3), np.eye(2))

    def test_energy_accepts_a_rectangular_model(self) -> None:
        """The energy follows the model output dimension, like the gradient."""
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        mu = np.array([0.4, 0.1])

        result = variational_free_energy(
            mu,
            np.eye(2),
            np.zeros(3),
            np.eye(2),
            np.eye(3),
            generative_fn=lambda values: matrix @ values,
        )

        predicted = matrix @ mu
        assert result.accuracy == pytest.approx(0.5 * float(predicted @ predicted), abs=1e-12)


class TestPackageSurface:
    """The repaired function is reachable from the package."""

    def test_package_export_is_the_module_function(self) -> None:
        """``scpn_quantum_control.fep`` exposes the gradient it documents."""
        assert package_gradient is free_energy_gradient

    def test_package_export_enforces_the_pair(self) -> None:
        """The contract holds through the package entry point."""
        with pytest.raises(ValueError, match="generative_fn requires generative_jac"):
            package_gradient(
                np.zeros(2), np.eye(2), np.zeros(2), np.eye(2), generative_fn=lambda v: v
            )
