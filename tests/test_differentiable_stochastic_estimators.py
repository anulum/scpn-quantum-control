# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable stochastic estimators tests
# scpn-quantum-control -- stochastic estimator extraction tests
"""Tests for extracted stochastic differentiable estimators."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control as scpn
import scpn_quantum_control.differentiable as differentiable
from scpn_quantum_control.differentiable import (
    Parameter,
    ParameterShiftRule,
    ScoreFunctionGradientResult,
    ShotAllocationResult,
    SPSAGradientResult,
    SPSAObjectiveSample,
    multi_frequency_parameter_shift_rule,
)
from scpn_quantum_control.differentiable_stochastic_estimators import (
    allocate_parameter_shift_shots,
    score_function_gradient_estimate,
    spsa_gradient_estimate,
)


def _assert_allclose(actual: object, expected: object, *, atol: float = 1.0e-12) -> None:
    """Assert NumPy-close equality while preserving strict test typing."""

    cast(Any, np.testing.assert_allclose)(actual, expected, atol=atol)


def _linear_objective(values: NDArray[np.float64]) -> float:
    """Return a deterministic scalar objective for SPSA tests."""

    return float(1.2 * values[0] - 0.4 * values[1])


def test_facade_and_package_root_reuse_extracted_stochastic_estimators() -> None:
    """Facade and package-root exports should point at the extracted helpers."""

    assert differentiable.spsa_gradient_estimate is spsa_gradient_estimate
    assert differentiable.score_function_gradient_estimate is score_function_gradient_estimate
    assert differentiable.allocate_parameter_shift_shots is allocate_parameter_shift_shots
    assert scpn.spsa_gradient_estimate is spsa_gradient_estimate
    assert scpn.score_function_gradient_estimate is score_function_gradient_estimate
    assert scpn.allocate_parameter_shift_shots is allocate_parameter_shift_shots


def test_spsa_gradient_estimate_records_seeded_probe_pairs() -> None:
    """SPSA should produce deterministic probe records and frozen-parameter zeros."""

    first = spsa_gradient_estimate(
        _linear_objective,
        [0.5, -0.25],
        perturbation_radius=0.125,
        repetitions=6,
        seed=19,
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
    )
    second = spsa_gradient_estimate(
        _linear_objective,
        [0.5, -0.25],
        perturbation_radius=0.125,
        repetitions=6,
        seed=19,
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
    )

    assert isinstance(first, SPSAGradientResult)
    assert first.method == "spsa"
    assert first.total_shots is None
    assert first.trainable == (True, False)
    assert len(first.records) == 6
    _assert_allclose(first.gradient, [1.2, 0.0])
    _assert_allclose(first.gradient, second.gradient)
    _assert_allclose(first.records[0].perturbation[1], 0.0)


def test_spsa_gradient_estimate_accepts_exact_objective_samples_without_shots() -> None:
    """Exact SPSA objectives may return sample objects without shot metadata."""

    def objective(values: NDArray[np.float64]) -> SPSAObjectiveSample:
        return SPSAObjectiveSample(value=float(0.75 * values[0]))

    result = spsa_gradient_estimate(
        objective,
        [0.2],
        perturbation_radius=0.5,
        repetitions=1,
        seed=3,
    )

    assert result.method == "spsa"
    assert result.records[0].plus.variance is None
    _assert_allclose(result.gradient, [0.75])


def test_spsa_gradient_estimate_propagates_finite_shot_samples() -> None:
    """Finite-shot SPSA should fill missing sample shot counts and propagate variance."""

    def objective(values: NDArray[np.float64], shots: int | None) -> SPSAObjectiveSample:
        assert shots == 200
        return SPSAObjectiveSample(
            value=float(0.5 * values[0] + 0.25 * values[1]),
            variance=0.04,
            metadata={"source": "fixture"},
        )

    result = spsa_gradient_estimate(
        objective,
        [0.1, -0.2],
        perturbation_radius=0.25,
        repetitions=3,
        seed=7,
        shots=200,
        confidence_z=2.0,
    )

    assert result.method == "finite_shot_spsa"
    assert result.total_shots == 1200
    assert result.records[0].plus.shots == 200
    assert result.records[0].plus.metadata == {"source": "fixture"}
    assert np.all(result.standard_error > 0.0)
    _assert_allclose(result.confidence_radius, 2.0 * result.standard_error)


def test_spsa_gradient_estimate_rejects_invalid_contracts() -> None:
    """SPSA should fail closed on invalid controls and objective samples."""

    with pytest.raises(ValueError, match="perturbation_radius"):
        spsa_gradient_estimate(_linear_objective, [0.1], perturbation_radius=0.0)
    with pytest.raises(ValueError, match="repetitions"):
        spsa_gradient_estimate(_linear_objective, [0.1], repetitions=0)
    with pytest.raises(ValueError, match="seed"):
        spsa_gradient_estimate(_linear_objective, [0.1], seed=-1)
    with pytest.raises(ValueError, match="shots"):
        spsa_gradient_estimate(_linear_objective, [0.1], shots=0)
    with pytest.raises(ValueError, match="confidence_z"):
        spsa_gradient_estimate(_linear_objective, [0.1], confidence_z=0.0)
    with pytest.raises(ValueError, match="trainable"):
        spsa_gradient_estimate(
            _linear_objective,
            [0.1],
            parameters=[Parameter("frozen", trainable=False)],
        )
    with pytest.raises(ValueError, match="SPSA finite-shot objective"):
        spsa_gradient_estimate(lambda values, shots: float(values[0]), [0.1], shots=100)
    with pytest.raises(ValueError, match="variance"):
        spsa_gradient_estimate(
            lambda values, shots: SPSAObjectiveSample(value=float(values[0]), shots=shots),
            [0.1],
            shots=100,
        )


def test_score_function_gradient_estimate_uses_materialised_samples() -> None:
    """Score-function gradients should match likelihood-ratio sample moments."""

    rewards = np.array([2.0, 0.0, 4.0], dtype=np.float64)
    scores = np.array([[1.0, 2.0], [-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    result = score_function_gradient_estimate(
        rewards,
        scores,
        baseline=1.0,
        confidence_z=2.0,
        parameters=[Parameter("theta"), Parameter("phi")],
    )

    sample_gradients = (rewards[:, None] - 1.0) * scores
    expected_covariance = np.cov(sample_gradients, rowvar=False, ddof=1) / rewards.size
    assert isinstance(result, ScoreFunctionGradientResult)
    assert result.method == "score_function_likelihood_ratio"
    assert result.sample_count == 3
    assert result.parameter_names == ("theta", "phi")
    _assert_allclose(result.gradient, np.mean(sample_gradients, axis=0))
    _assert_allclose(result.covariance, expected_covariance)
    _assert_allclose(result.confidence_radius, 2.0 * result.standard_error)


def test_score_function_gradient_estimate_honours_frozen_parameters() -> None:
    """Frozen score-function columns should report zero gradient and covariance."""

    result = score_function_gradient_estimate(
        [2.0, 0.0, 4.0],
        [[1.0, 2.0], [-1.0, 0.0], [0.0, 1.0]],
        baseline=1.0,
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
    )

    _assert_allclose(result.gradient[1], 0.0)
    _assert_allclose(result.standard_error[1], 0.0)
    _assert_allclose(result.covariance[1], [0.0, 0.0])
    _assert_allclose(result.covariance[:, 1], [0.0, 0.0])


def test_score_function_gradient_estimate_rejects_invalid_contracts() -> None:
    """Score-function estimation should fail closed on malformed samples."""

    with pytest.raises(ValueError, match="at least two"):
        score_function_gradient_estimate([1.0], [[0.5]])
    with pytest.raises(ValueError, match="two-dimensional"):
        score_function_gradient_estimate([1.0, 2.0], [0.5, 0.25])
    with pytest.raises(ValueError, match="row count"):
        score_function_gradient_estimate([1.0, 2.0, 3.0], [[0.5], [0.25]])
    with pytest.raises(ValueError, match="baseline"):
        score_function_gradient_estimate([1.0, 2.0], [[0.5], [0.25]], baseline=np.inf)
    with pytest.raises(ValueError, match="confidence_z"):
        score_function_gradient_estimate([1.0, 2.0], [[0.5], [0.25]], confidence_z=0.0)
    with pytest.raises(ValueError, match="trainable"):
        score_function_gradient_estimate(
            [1.0, 2.0],
            [[0.5], [0.25]],
            parameters=[Parameter("frozen", trainable=False)],
        )


def test_allocate_parameter_shift_shots_meets_single_and_multi_term_targets() -> None:
    """Shot allocation should plan single-term and multi-frequency budgets."""

    single = allocate_parameter_shift_shots(
        [0.36, 0.25],
        [0.16, 0.09],
        target_standard_error=0.02,
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
    )
    multi_rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    multi = allocate_parameter_shift_shots(
        [[0.20, 0.30], [0.40, 0.50]],
        [[0.10, 0.20], [0.30, 0.40]],
        target_standard_error=0.15,
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
        rule=multi_rule,
        min_shots=7,
    )

    assert isinstance(single, ShotAllocationResult)
    assert single.method == "parameter_shift_target_se"
    assert single.predicted_standard_error[0] <= 0.02
    assert single.predicted_standard_error[1] == pytest.approx(0.0)
    assert multi.method == "multi_frequency_parameter_shift_target_se"
    assert multi.shots.shape == (len(multi_rule.terms), 2, 2)
    assert np.all(multi.shots[:, :, 1] == 7.0)


def test_allocate_parameter_shift_shots_respects_caps_and_zero_noise() -> None:
    """Shot allocation should preserve caps and minimum-shot zero-noise plans."""

    capped = allocate_parameter_shift_shots(
        [1.0],
        [1.0],
        target_standard_error=1.0e-3,
        min_shots=4,
        max_shots_per_evaluation=10,
    )
    zero_noise = allocate_parameter_shift_shots(
        [0.0],
        [0.0],
        target_standard_error=0.1,
        min_shots=3,
        rule=ParameterShiftRule(),
    )

    _assert_allclose(capped.shots, [[10.0], [10.0]])
    assert capped.predicted_standard_error[0] > capped.target_standard_error
    _assert_allclose(zero_noise.shots, [[3.0], [3.0]])
    _assert_allclose(zero_noise.predicted_standard_error, [0.0])


def test_allocate_parameter_shift_shots_rejects_invalid_inputs() -> None:
    """Shot allocation should reject impossible stochastic planning contracts."""

    with pytest.raises(ValueError, match="minus_variances shape"):
        allocate_parameter_shift_shots([0.1], [0.1, 0.2], target_standard_error=0.1)
    with pytest.raises(ValueError, match="shot variances"):
        allocate_parameter_shift_shots([-0.1], [0.1], target_standard_error=0.1)
    with pytest.raises(ValueError, match="target_standard_error"):
        allocate_parameter_shift_shots([0.1], [0.1], target_standard_error=0.0)
    with pytest.raises(ValueError, match="min_shots"):
        allocate_parameter_shift_shots([0.1], [0.1], target_standard_error=0.1, min_shots=0)
    with pytest.raises(ValueError, match="max_shots_per_evaluation"):
        allocate_parameter_shift_shots(
            [0.1],
            [0.1],
            target_standard_error=0.1,
            min_shots=10,
            max_shots_per_evaluation=5,
        )
