# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Score-Function Gradient Tests
"""Tests for materialised likelihood-ratio gradient estimation."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control import Parameter
from scpn_quantum_control.differentiable import (
    ScoreFunctionGradientResult,
    score_function_gradient_estimate,
)


def test_score_function_gradient_estimate_uses_likelihood_ratio_samples() -> None:
    rewards = np.array([2.0, 0.0, 4.0], dtype=np.float64)
    score_vectors = np.array(
        [
            [1.0, 2.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    result = score_function_gradient_estimate(
        rewards,
        score_vectors,
        baseline=1.0,
        confidence_z=2.0,
        parameters=(Parameter("theta"), Parameter("phi")),
    )

    sample_gradients = (rewards[:, None] - 1.0) * score_vectors
    expected_gradient = np.mean(sample_gradients, axis=0)
    expected_covariance = np.cov(sample_gradients, rowvar=False, ddof=1) / rewards.size

    assert isinstance(result, ScoreFunctionGradientResult)
    assert result.method == "score_function_likelihood_ratio"
    assert result.parameter_names == ("theta", "phi")
    assert result.trainable == (True, True)
    assert result.sample_count == 3
    assert result.baseline == 1.0
    assert result.hardware_execution is False
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.covariance, expected_covariance, atol=1e-12)
    np.testing.assert_allclose(result.standard_error, np.sqrt(np.diag(expected_covariance)))
    np.testing.assert_allclose(result.confidence_radius, 2.0 * result.standard_error)
    assert result.records[0].weighted_score.tolist() == [1.0, 2.0]
    assert result.to_dict()["method"] == "score_function_likelihood_ratio"


def test_score_function_gradient_estimate_honours_frozen_parameters() -> None:
    result = score_function_gradient_estimate(
        [2.0, 0.0, 4.0],
        [[1.0, 2.0], [-1.0, 0.0], [0.0, 1.0]],
        baseline=1.0,
        parameters=(Parameter("active"), Parameter("frozen", trainable=False)),
    )

    np.testing.assert_allclose(result.gradient[1], 0.0, atol=0.0)
    np.testing.assert_allclose(result.standard_error[1], 0.0, atol=0.0)
    np.testing.assert_allclose(result.covariance[1], [0.0, 0.0], atol=0.0)
    np.testing.assert_allclose(result.covariance[:, 1], [0.0, 0.0], atol=0.0)


def test_score_function_gradient_estimate_fails_closed_for_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="at least two"):
        score_function_gradient_estimate([1.0], [[0.5]])
    with pytest.raises(ValueError, match="score_vectors"):
        score_function_gradient_estimate([1.0, 2.0], [[0.5]])
    with pytest.raises(ValueError, match="baseline"):
        score_function_gradient_estimate([1.0, 2.0], [[0.5], [0.25]], baseline=np.inf)
    with pytest.raises(ValueError, match="trainable"):
        score_function_gradient_estimate(
            [1.0, 2.0],
            [[0.5], [0.25]],
            parameters=(Parameter("frozen", trainable=False),),
        )


def test_score_function_exports_from_package_root() -> None:
    assert scpn.ScoreFunctionGradientResult is ScoreFunctionGradientResult
    assert scpn.score_function_gradient_estimate is score_function_gradient_estimate
