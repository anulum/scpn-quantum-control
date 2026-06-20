# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Levenberg-Marquardt differentiable tests
"""Regression tests for robust residual weights and Levenberg-Marquardt paths."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import (
    LevenbergMarquardtDampingUpdate,
    LevenbergMarquardtOptimizer,
    LevenbergMarquardtResult,
    LevenbergMarquardtStep,
    LevenbergMarquardtTrial,
    Parameter,
    ParameterBounds,
    evaluate_levenberg_marquardt_step,
    gauss_newton_gradient,
    huber_residual_weights,
    levenberg_marquardt_step,
    soft_l1_residual_weights,
    update_levenberg_marquardt_damping,
    value_and_finite_difference_jacobian,
)

FloatArray = NDArray[np.float64]


def _assert_allclose(actual: object, expected: object, *, atol: float | None = None) -> None:
    """Assert numerical closeness through NumPy's dynamically typed test helper."""

    assert_allclose = cast(Any, np.testing.assert_allclose)
    if atol is None:
        assert_allclose(actual, expected)
    else:
        assert_allclose(actual, expected, atol=atol)


def test_huber_residual_weights_downweight_outliers_for_residual_maps() -> None:
    """Huber IRLS weights should preserve inliers and bound outlier influence."""

    weights = huber_residual_weights(np.array([0.0, 0.5, -2.0, 10.0]), delta=1.0)

    _assert_allclose(weights, [1.0, 1.0, 0.5, 0.1])


def test_huber_residual_weights_support_floor_for_conditioning() -> None:
    """A positive floor prevents outlier weights from collapsing to zero."""

    weights = huber_residual_weights(np.array([1.0, 100.0]), delta=1.0, min_weight=0.05)

    _assert_allclose(weights, [1.0, 0.05])


def test_huber_residual_weights_reject_invalid_controls() -> None:
    """Robust residual weighting must reject invalid residual and policy inputs."""

    with pytest.raises(ValueError, match="one-dimensional"):
        huber_residual_weights(np.array([[1.0]]))
    with pytest.raises(ValueError, match="Huber delta"):
        huber_residual_weights(np.array([1.0]), delta=0.0)
    with pytest.raises(ValueError, match="Huber min_weight"):
        huber_residual_weights(np.array([1.0]), min_weight=-0.1)
    with pytest.raises(ValueError, match="Huber min_weight"):
        huber_residual_weights(np.array([1.0]), min_weight=1.1)


def test_huber_residual_weights_feed_gauss_newton_metric() -> None:
    """Robust weights should plug directly into Gauss-Newton residual solves."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] - 1.0, 10.0 * (values[1] - 1.0)]),
        [3.0, 2.0],
        parameters=[Parameter("x"), Parameter("y")],
    )
    weights = huber_residual_weights(jacobian_result.value, delta=2.0)
    result = gauss_newton_gradient(jacobian_result, weights=weights, damping=0.5)

    _assert_allclose(weights, [1.0, 0.2], atol=1.0e-6)
    _assert_allclose(result.base_gradient.gradient, [2.0, 20.0], atol=1.0e-6)
    assert result.condition_number > 1.0


def test_soft_l1_residual_weights_smoothly_downweight_outliers() -> None:
    """Soft-L1 weights should provide a smooth influence curve for residuals."""

    weights = soft_l1_residual_weights(np.array([0.0, 1.0, 3.0]), scale=1.0)

    _assert_allclose(weights, [1.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(10.0)])


def test_soft_l1_residual_weights_support_conditioning_floor() -> None:
    """A positive floor keeps Soft-L1 weights usable in ill-scaled residual maps."""

    weights = soft_l1_residual_weights(np.array([0.0, 100.0]), scale=1.0, min_weight=0.05)

    _assert_allclose(weights, [1.0, 0.05])


def test_soft_l1_residual_weights_reject_invalid_controls() -> None:
    """Soft-L1 residual weighting must fail closed on invalid policy inputs."""

    with pytest.raises(ValueError, match="one-dimensional"):
        soft_l1_residual_weights(np.array([[1.0]]))
    with pytest.raises(ValueError, match="Soft-L1 scale"):
        soft_l1_residual_weights(np.array([1.0]), scale=0.0)
    with pytest.raises(ValueError, match="Soft-L1 min_weight"):
        soft_l1_residual_weights(np.array([1.0]), min_weight=-0.1)
    with pytest.raises(ValueError, match="Soft-L1 min_weight"):
        soft_l1_residual_weights(np.array([1.0]), min_weight=1.1)


def test_soft_l1_residual_weights_feed_levenberg_marquardt_trial() -> None:
    """Soft-L1 weights should plug into the weighted LM step and trial path."""

    def objective(values: FloatArray) -> FloatArray:
        return np.array([values[0] - 1.0, 20.0 * (values[1] - 1.0)])

    jacobian_result = value_and_finite_difference_jacobian(
        objective,
        [3.0, 2.0],
        parameters=[Parameter("x"), Parameter("y")],
    )
    weights = soft_l1_residual_weights(jacobian_result.value, scale=2.0, min_weight=0.1)
    step_result = levenberg_marquardt_step(
        jacobian_result,
        [3.0, 2.0],
        weights=weights,
        damping=1.0,
    )
    trial = evaluate_levenberg_marquardt_step(objective, step_result, weights=weights)

    assert weights[0] > weights[1]
    assert trial.candidate_value < step_result.gauss_newton.base_gradient.value


def test_levenberg_marquardt_step_builds_bounded_candidate() -> None:
    """Levenberg-Marquardt should expose a bounded residual descent candidate."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] - 1.0, 2.0 * (values[1] + 0.5)]),
        [3.0, 1.5],
        parameters=[Parameter("x"), Parameter("y")],
    )
    result = levenberg_marquardt_step(
        jacobian_result,
        [3.0, 1.5],
        weights=np.array([1.0, 0.25]),
        damping=0.5,
    )

    assert isinstance(result, LevenbergMarquardtStep)
    _assert_allclose(result.step, [-4.0 / 3.0, -4.0 / 3.0], atol=1.0e-6)
    _assert_allclose(result.candidate_values, [5.0 / 3.0, 1.0 / 6.0], atol=1e-6)
    assert result.predicted_reduction == pytest.approx(8.0 / 3.0, abs=1.0e-6)
    assert result.damping == pytest.approx(0.5)


def test_levenberg_marquardt_step_caps_trainable_norm_and_projects_bounds() -> None:
    """LM candidates must respect update caps, bounds, and frozen parameters."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] - 1.0, values[1] - 2.0]),
        [3.0, 5.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
    )
    result = levenberg_marquardt_step(
        jacobian_result,
        [3.0, 5.0],
        damping=0.0,
        bounds=[ParameterBounds(lower=2.25, upper=3.0), ParameterBounds()],
        max_step_norm=0.5,
    )

    _assert_allclose(result.step, [-0.5, 0.0], atol=1.0e-6)
    _assert_allclose(result.candidate_values, [2.5, 5.0], atol=1.0e-6)
    assert result.predicted_reduction == pytest.approx(0.875)


def test_levenberg_marquardt_step_rejects_invalid_controls() -> None:
    """LM controls must be finite, dimensionally consistent, and physical."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0], values[1]]),
        [1.0, 2.0],
    )
    with pytest.raises(ValueError, match="values length"):
        levenberg_marquardt_step(jacobian_result, [1.0])
    with pytest.raises(ValueError, match="damping"):
        levenberg_marquardt_step(jacobian_result, [1.0, 2.0], damping=-1.0)
    with pytest.raises(ValueError, match="max_step_norm"):
        levenberg_marquardt_step(jacobian_result, [1.0, 2.0], max_step_norm=0.0)


def test_evaluate_levenberg_marquardt_step_accepts_improving_candidate() -> None:
    """LM acceptance should compare actual and predicted residual reduction."""

    def objective(values: FloatArray) -> FloatArray:
        return np.array([values[0] - 1.0, 2.0 * (values[1] + 0.5)])

    jacobian_result = value_and_finite_difference_jacobian(
        objective,
        [3.0, 1.5],
        parameters=[Parameter("x"), Parameter("y")],
    )
    step_result = levenberg_marquardt_step(
        jacobian_result,
        [3.0, 1.5],
        weights=np.array([1.0, 0.25]),
        damping=0.5,
    )
    trial = evaluate_levenberg_marquardt_step(
        objective,
        step_result,
        weights=np.array([1.0, 0.25]),
        acceptance_threshold=0.1,
    )

    assert isinstance(trial, LevenbergMarquardtTrial)
    assert trial.accepted is True
    assert trial.candidate_value == pytest.approx(4.0 / 9.0, abs=1.0e-6)
    assert trial.actual_reduction == pytest.approx(32.0 / 9.0, abs=1.0e-6)
    assert trial.reduction_ratio == pytest.approx(4.0 / 3.0, abs=1.0e-6)


def test_evaluate_levenberg_marquardt_step_rejects_poor_candidate() -> None:
    """A residual-increasing candidate must not be accepted."""

    def objective(values: FloatArray) -> FloatArray:
        return np.array([values[0] - 1.0])

    jacobian_result = value_and_finite_difference_jacobian(
        objective,
        [3.0],
        parameters=[Parameter("x")],
    )
    step_result = LevenbergMarquardtStep(
        gauss_newton=gauss_newton_gradient(jacobian_result, damping=0.5),
        step=np.array([2.0]),
        candidate_values=np.array([5.0]),
        damping=0.5,
        predicted_reduction=0.5,
    )
    trial = evaluate_levenberg_marquardt_step(objective, step_result)

    assert trial.accepted is False
    assert trial.actual_reduction < 0.0
    assert trial.reduction_ratio < 0.0


def test_evaluate_levenberg_marquardt_step_rejects_invalid_controls() -> None:
    """LM acceptance diagnostics must fail closed on invalid policy inputs."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0], values[1]]),
        [1.0, 2.0],
    )
    step_result = levenberg_marquardt_step(jacobian_result, [1.0, 2.0], damping=0.5)

    with pytest.raises(ValueError, match="acceptance_threshold"):
        evaluate_levenberg_marquardt_step(
            lambda values: np.array([values[0], values[1]]),
            step_result,
            acceptance_threshold=-1.0,
        )
    with pytest.raises(ValueError, match="weights"):
        evaluate_levenberg_marquardt_step(
            lambda values: np.array([values[0], values[1]]),
            step_result,
            weights=np.array([1.0]),
        )
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_levenberg_marquardt_step(
            lambda values: np.array([values[0], values[1]]),
            step_result,
            weights=np.array([1.0, -1.0]),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        evaluate_levenberg_marquardt_step(
            lambda _values: np.array([[1.0]]),
            step_result,
        )


def test_update_levenberg_marquardt_damping_decreases_high_quality_acceptance() -> None:
    """High-quality accepted LM trials should reduce damping toward Gauss-Newton."""

    def objective(values: FloatArray) -> FloatArray:
        return np.array([values[0] - 1.0])

    jacobian_result = value_and_finite_difference_jacobian(
        objective,
        [3.0],
        parameters=[Parameter("x")],
    )
    step_result = levenberg_marquardt_step(jacobian_result, [3.0], damping=0.9)
    trial = evaluate_levenberg_marquardt_step(objective, step_result)
    update = update_levenberg_marquardt_damping(
        trial,
        decrease_factor=0.5,
        high_quality_ratio=0.5,
    )

    assert isinstance(update, LevenbergMarquardtDampingUpdate)
    assert update.action == "accept_decrease"
    assert update.next_damping == pytest.approx(0.45)


def test_update_levenberg_marquardt_damping_keeps_marginal_acceptance() -> None:
    """Accepted but marginal LM trials should keep damping unchanged."""

    step_result = LevenbergMarquardtStep(
        gauss_newton=gauss_newton_gradient(
            value_and_finite_difference_jacobian(lambda values: np.array([values[0]]), [1.0]),
            damping=1.0,
        ),
        step=np.array([-0.25]),
        candidate_values=np.array([0.75]),
        damping=1.0,
        predicted_reduction=1.0,
    )
    trial = LevenbergMarquardtTrial(
        step_result=step_result,
        candidate_residual=np.array([0.75]),
        candidate_value=0.25,
        actual_reduction=0.25,
        reduction_ratio=0.25,
        accepted=True,
    )
    update = update_levenberg_marquardt_damping(trial, high_quality_ratio=0.75)

    assert update.action == "accept_keep"
    assert update.next_damping == pytest.approx(1.0)


def test_update_levenberg_marquardt_damping_increases_rejected_trial() -> None:
    """Rejected LM trials should increase damping for a smaller retry step."""

    step_result = LevenbergMarquardtStep(
        gauss_newton=gauss_newton_gradient(
            value_and_finite_difference_jacobian(lambda values: np.array([values[0]]), [1.0]),
            damping=0.5,
        ),
        step=np.array([1.0]),
        candidate_values=np.array([2.0]),
        damping=0.5,
        predicted_reduction=0.25,
    )
    trial = LevenbergMarquardtTrial(
        step_result=step_result,
        candidate_residual=np.array([2.0]),
        candidate_value=2.0,
        actual_reduction=-1.5,
        reduction_ratio=-6.0,
        accepted=False,
    )
    update = update_levenberg_marquardt_damping(
        trial,
        increase_factor=4.0,
        max_damping=1.5,
    )

    assert update.action == "reject_increase"
    assert update.next_damping == pytest.approx(1.5)


def test_update_levenberg_marquardt_damping_rejects_invalid_policy() -> None:
    """Damping policy factors and bounds must fail closed."""

    step_result = LevenbergMarquardtStep(
        gauss_newton=gauss_newton_gradient(
            value_and_finite_difference_jacobian(lambda values: np.array([values[0]]), [1.0]),
            damping=0.5,
        ),
        step=np.array([-0.5]),
        candidate_values=np.array([0.5]),
        damping=0.5,
        predicted_reduction=0.25,
    )
    trial = LevenbergMarquardtTrial(
        step_result=step_result,
        candidate_residual=np.array([0.5]),
        candidate_value=0.125,
        actual_reduction=0.375,
        reduction_ratio=1.5,
        accepted=True,
    )

    invalid_trial = cast(LevenbergMarquardtTrial, step_result)
    with pytest.raises(ValueError, match="LevenbergMarquardtTrial"):
        update_levenberg_marquardt_damping(invalid_trial)
    with pytest.raises(ValueError, match="decrease_factor"):
        update_levenberg_marquardt_damping(trial, decrease_factor=1.0)
    with pytest.raises(ValueError, match="increase_factor"):
        update_levenberg_marquardt_damping(trial, increase_factor=1.0)
    with pytest.raises(ValueError, match="min_damping"):
        update_levenberg_marquardt_damping(trial, min_damping=-1.0)
    with pytest.raises(ValueError, match="max_damping"):
        update_levenberg_marquardt_damping(trial, min_damping=2.0, max_damping=1.0)
    with pytest.raises(ValueError, match="high_quality_ratio"):
        update_levenberg_marquardt_damping(trial, high_quality_ratio=-1.0)


def test_levenberg_marquardt_optimizer_converges_for_residual_map() -> None:
    """The full LM optimizer should solve a trainable residual map with provenance."""

    def objective(values: FloatArray) -> FloatArray:
        return np.array([values[0] - 1.0, 2.0 * (values[1] + 0.5)])

    optimizer = LevenbergMarquardtOptimizer(damping=0.1, max_steps=20, residual_tolerance=1e-7)
    result = optimizer.minimize(
        objective,
        [3.0, 1.5],
        parameters=[Parameter("x"), Parameter("y")],
    )

    assert isinstance(result, LevenbergMarquardtResult)
    assert result.converged
    assert result.reason in {"residual_tolerance", "step_tolerance", "value_tolerance"}
    assert any(result.accepted_history)
    assert len(result.value_history) == result.steps + 1
    assert len(result.damping_history) == result.steps + 1
    assert result.best_value <= result.value_history[0]
    _assert_allclose(result.values, [1.0, -0.5], atol=1.0e-5)


def test_levenberg_marquardt_optimizer_respects_bounds_and_weights() -> None:
    """Bounded weighted LM runs should stay inside the declared parameter domain."""

    def objective(values: FloatArray) -> FloatArray:
        return np.array([values[0] - 2.0, 8.0 * (values[1] - 3.0)])

    optimizer = LevenbergMarquardtOptimizer(
        damping=0.2,
        max_steps=30,
        residual_tolerance=1.0e-7,
        max_step_norm=1.0,
    )
    result = optimizer.minimize(
        objective,
        [0.0, 0.0],
        bounds=[ParameterBounds(lower=-0.5, upper=1.0), ParameterBounds(lower=-1.0, upper=1.5)],
        weight_fn=lambda residuals: soft_l1_residual_weights(
            residuals, scale=2.0, min_weight=0.05
        ),
    )

    assert result.converged or result.reason == "max_steps"
    assert np.all(result.values <= np.array([1.0, 1.5]))
    assert np.all(result.values >= np.array([-0.5, -1.0]))
    _assert_allclose(result.best_values, [1.0, 1.5], atol=1.0e-6)


def test_levenberg_marquardt_optimizer_rejects_invalid_controls() -> None:
    """Full LM optimization controls and IRLS weights must fail closed."""

    with pytest.raises(ValueError, match="max_steps"):
        LevenbergMarquardtOptimizer(max_steps=0)
    with pytest.raises(ValueError, match="tolerances"):
        LevenbergMarquardtOptimizer(residual_tolerance=-1.0)
    with pytest.raises(ValueError, match="finite_difference_step"):
        LevenbergMarquardtOptimizer(finite_difference_step=0.0)

    optimizer = LevenbergMarquardtOptimizer(max_steps=1)
    with pytest.raises(ValueError, match="LM weights"):
        optimizer.minimize(lambda values: np.array([values[0]]), [1.0], weight_fn=lambda _: [-1.0])
    with pytest.raises(ValueError, match="LM weights"):
        optimizer.minimize(
            lambda values: np.array([values[0]]), [1.0], weight_fn=lambda _: [1.0, 1.0]
        )
