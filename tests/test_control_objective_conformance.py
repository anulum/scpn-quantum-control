# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Control objective conformance
"""Exercise control costs, Gaussian beliefs and real native refusal together."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.hardware.classical as classical_module
import scpn_quantum_engine as engine
from scpn_quantum_control.control.qaoa_mpc import QAOA_MPC
from scpn_quantum_control.fep import (
    PRECISION_RIDGE,
    free_energy_gradient,
    kl_divergence_gaussian,
    predictive_coding_step,
    variational_free_energy,
)
from scpn_quantum_control.hardware.classical import classical_brute_mpc


@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize(
    ("matrix", "target"),
    [
        (np.eye(2), np.array([-1.0, -0.5])),
        (np.eye(2), np.array([1.0, 0.5])),
        (np.array([[0.6, -0.8], [0.8, 0.6]]), np.array([-0.2, 1.4])),
        (np.array([[0.6, -0.8], [0.8, 0.6]]), np.array([1.4, -0.2])),
        (np.zeros((2, 2)), np.array([0.3, -0.4])),
    ],
)
def test_control_objective_conformance_01(
    matrix: NDArray[np.float64],
    target: NDArray[np.float64],
    horizon: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare complete binary landscapes and unapplied optimal proposals.

    Parameters
    ----------
    matrix
        Signed two-dimensional actuation map, including a rotation and a tie.
    target
        Target vector in the actuation vector's units.
    horizon
        Number of binary timesteps; at most eight sequences are enumerated.
    monkeypatch
        Restore accelerator discovery after additionally exercising its absence.

    """
    original_matrix, original_target = matrix.copy(), target.copy()
    sequences = np.array(
        [[(index >> step) & 1 for step in range(horizon)] for index in range(2**horizon)],
        dtype=np.int64,
    )
    actuation = matrix @ np.ones(2)
    residuals = sequences[:, :, None] * actuation - target
    expected = np.einsum("bti,bti->b", residuals, residuals)
    diagonal = np.diag(
        QAOA_MPC(matrix, target, horizon, p_layers=1).build_cost_hamiltonian().to_matrix()
    )
    public = classical_brute_mpc(matrix, target, horizon)
    actions, minimum, landscape, evaluated = engine.brute_mpc(matrix.ravel(), target, 2, horizon)
    with monkeypatch.context() as absent_accelerator:
        absent_accelerator.setattr(classical_module, "optional_rust_engine", lambda: None)
        fallback = classical_brute_mpc(matrix, target, horizon)
    np.testing.assert_allclose(diagonal.real, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(diagonal.imag, np.zeros(expected.size))
    np.testing.assert_allclose(public["all_costs"], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(landscape, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fallback["all_costs"], expected, rtol=1e-12, atol=1e-12)
    expected_actions = sequences[int(np.argmin(expected))]
    np.testing.assert_array_equal(actions, expected_actions)
    np.testing.assert_array_equal(public["optimal_actions"], expected_actions)
    np.testing.assert_array_equal(fallback["optimal_actions"], expected_actions)
    assert minimum == pytest.approx(float(np.min(expected)), rel=1e-12, abs=1e-12)
    assert public["optimal_cost"] == pytest.approx(minimum, rel=1e-12, abs=1e-12)
    assert fallback["optimal_cost"] == pytest.approx(minimum, rel=1e-12, abs=1e-12)
    assert evaluated == public["n_evaluated"] == 2**horizon
    assert fallback["n_evaluated"] == evaluated
    assert set(np.asarray(actions).tolist()) <= {0, 1}
    np.testing.assert_array_equal(matrix, original_matrix)
    np.testing.assert_array_equal(target, original_target)


def test_control_objective_conformance_02() -> None:
    """Retain off-diagonal covariance, full determinants and a rectangular model."""
    mean = np.array([0.2, -0.3])
    covariance = np.array([[1.4, 0.3], [0.3, 0.8]])
    prior = np.array([[2.0, 0.4], [0.4, 1.2]])
    observation = np.array([0.4, -0.2, 0.6])
    model = np.array([[2.0, -0.5], [0.2, 1.0], [-0.7, 0.4]])
    sensory = np.array([[1.5, 0.2, 0.1], [0.2, 1.1, -0.1], [0.1, -0.1, 0.9]])
    prior_regularised = prior + PRECISION_RIDGE * np.eye(2)
    prior_covariance = np.asarray(np.linalg.inv(prior_regularised), dtype=np.float64)
    expected_kl = 0.5 * (
        np.trace(prior_regularised @ covariance)
        + mean @ prior_regularised @ mean
        - 2
        + np.linalg.slogdet(prior_covariance)[1]
        - np.linalg.slogdet(covariance)[1]
    )
    actual_kl = kl_divergence_gaussian(mean, covariance, np.zeros(2), prior_covariance)
    assert actual_kl == pytest.approx(expected_kl, rel=1e-12, abs=1e-12)

    def predict(values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the explicit three-output linear model.

        Parameters
        ----------
        values
            Two belief coordinates.

        Returns
        -------
        numpy.ndarray
            Three predicted observations.

        """
        return model @ values

    def jacobian(values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the complete constant model Jacobian.

        Parameters
        ----------
        values
            Belief point, retained by the callable model interface.

        Returns
        -------
        numpy.ndarray
            Three-by-two model Jacobian.

        """
        return model.copy()

    original = tuple(v.copy() for v in (mean, covariance, observation, prior, sensory))
    residual = observation - model @ mean
    expected_accuracy = 0.5 * residual @ sensory @ residual
    expected_gradient = prior_regularised @ mean - model.T @ sensory @ residual
    result = variational_free_energy(mean, covariance, observation, prior, sensory, predict)
    gradient = free_energy_gradient(
        mean, covariance, observation, prior, sensory, predict, jacobian
    )
    assert result.complexity == pytest.approx(expected_kl, rel=1e-12, abs=1e-12)
    assert result.accuracy == pytest.approx(expected_accuracy, rel=1e-12, abs=1e-12)
    assert result.free_energy == pytest.approx(
        expected_kl + expected_accuracy, rel=1e-12, abs=1e-12
    )
    assert result.elbo == -result.free_energy
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    for actual, saved in zip(
        (mean, covariance, observation, prior, sensory), original, strict=True
    ):
        np.testing.assert_array_equal(actual, saved)


@pytest.mark.parametrize(
    "covariance",
    [
        np.array([[1.0, 0.3], [0.0, 1.0]]),
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        np.zeros((2, 2)),
        np.array([[1.0, 0.0], [0.0, np.nan]]),
    ],
)
def test_control_objective_conformance_03(covariance: NDArray[np.float64]) -> None:
    """Reject invalid beliefs before returning a proposal, preserving caller data.

    Parameters
    ----------
    covariance
        Asymmetric, indefinite, singular or non-finite candidate covariance.

    """
    beliefs, observation, coupling = np.array([0.2, -0.3]), np.array([0.7, -0.1]), np.eye(2)
    saved = tuple(v.copy() for v in (beliefs, observation, coupling, covariance))
    with pytest.raises(ValueError):
        kl_divergence_gaussian(beliefs, covariance, np.zeros(2), np.eye(2))
    with pytest.raises(ValueError):
        predictive_coding_step(observation, beliefs, coupling, sigma=covariance)
    for actual, original in zip((beliefs, observation, coupling, covariance), saved, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_missing_model_jacobian_refuses_before_callback() -> None:
    """Reject an incomplete model before evaluating it or changing beliefs."""
    beliefs = np.array([0.2, -0.3])
    saved = beliefs.copy()
    evaluations: list[NDArray[np.float64]] = []

    def model(values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate a real affine observation model and record its input.

        Parameters
        ----------
        values
            Two belief coordinates.

        Returns
        -------
        numpy.ndarray
            Scaled observations.

        """
        evaluations.append(values.copy())
        return 2.0 * values

    with pytest.raises(ValueError, match="requires generative_jac"):
        free_energy_gradient(beliefs, np.eye(2), np.zeros(2), np.eye(2), generative_fn=model)
    assert evaluations == []
    np.testing.assert_array_equal(beliefs, saved)


@pytest.mark.parametrize("observation", [np.zeros(1), np.zeros(3), np.array([0.0, np.inf])])
def test_native_shape_refusal_returns_no_energy_or_gradient(
    observation: NDArray[np.float64],
) -> None:
    """Reject malformed observations through both actual compiled public exports.

    Parameters
    ----------
    observation
        Short, long or non-finite observation vector for two beliefs.

    """
    beliefs, prior, sensory = np.array([0.2, -0.3]), np.eye(2), np.eye(2)
    saved = tuple(v.copy() for v in (beliefs, observation, prior, sensory))
    with pytest.raises(ValueError):
        engine.free_energy_gradient_rust(beliefs, observation, prior, sensory, PRECISION_RIDGE)
    with pytest.raises(ValueError):
        engine.variational_free_energy_rust(
            beliefs, observation, prior, sensory, 0.2, PRECISION_RIDGE
        )
    for actual, original in zip((beliefs, observation, prior, sensory), saved, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_native_strided_identity_model_matches_public_belief_objective() -> None:
    """Compare real strided FFI inputs within the native isotropic-covariance domain."""
    backing = np.array([0.2, np.nan, -0.3, np.nan])
    beliefs = backing[::2]
    observation = np.array([0.7, -0.1])
    prior = np.asfortranarray([[2.0, 0.4], [0.4, 1.2]])
    sensory = np.array([[1.5, 0.2], [0.2, 0.9]]).T
    covariance = 0.2 * np.eye(2)
    saved = tuple(v.copy() for v in (backing, observation, prior, sensory))
    expected = variational_free_energy(beliefs, covariance, observation, prior, sensory)
    expected_gradient = (prior + PRECISION_RIDGE * np.eye(2)) @ beliefs - sensory @ (
        observation - beliefs
    )
    value, complexity, accuracy = engine.variational_free_energy_rust(
        beliefs, observation, prior, sensory, 0.2, PRECISION_RIDGE
    )
    gradient = engine.free_energy_gradient_rust(
        beliefs, observation, prior, sensory, PRECISION_RIDGE
    )
    public_gradient = free_energy_gradient(beliefs, covariance, observation, prior, sensory)
    np.testing.assert_allclose(
        [value, complexity, accuracy],
        [expected.free_energy, expected.complexity, expected.accuracy],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(public_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    for actual, original in zip((backing, observation, prior, sensory), saved, strict=True):
        np.testing.assert_array_equal(actual, original)
