# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for differentiable programming primitives
"""Tests for native differentiable-programming primitives."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    ArmijoLineSearchResult,
    CustomDerivativeCheckResult,
    CustomDerivativeRule,
    DifferentiableOptimizer,
    DualNumber,
    FisherConjugateGradientResult,
    FisherVectorProductResult,
    FixedPointSensitivityResult,
    GradientCheckResult,
    GradientResult,
    HessianResult,
    HVPResult,
    ImplicitSensitivityResult,
    JacobianResult,
    JVPResult,
    LeastSquaresCovarianceResult,
    LevenbergMarquardtDampingUpdate,
    LevenbergMarquardtOptimizer,
    LevenbergMarquardtResult,
    LevenbergMarquardtStep,
    LevenbergMarquardtTrial,
    NaturalGradientOptimizationResult,
    NaturalGradientOptimizer,
    NaturalGradientResult,
    OptimizationResult,
    Parameter,
    ParameterBounds,
    ParameterShiftRule,
    ReverseNode,
    ShotAllocationResult,
    SparseMatrixResult,
    StochasticGradientResult,
    VJPResult,
    WeightedGradientResult,
    allocate_parameter_shift_shots,
    armijo_backtracking_line_search,
    batch_complex_step_gradient,
    batch_custom_jacobian,
    batch_custom_jvp,
    batch_custom_vjp,
    batch_finite_difference_hvp,
    batch_finite_difference_jvp,
    batch_finite_difference_vjp,
    batch_value_and_complex_step_grad,
    batch_value_and_custom_jacobian,
    batch_value_and_custom_jvp,
    batch_value_and_custom_vjp,
    batch_value_and_finite_difference_grad,
    batch_value_and_finite_difference_hvp,
    batch_value_and_finite_difference_jvp,
    batch_value_and_finite_difference_vjp,
    batch_value_and_parameter_shift_grad,
    batch_vector_jacobian_product,
    check_custom_derivative_consistency,
    check_parameter_shift_consistency,
    complex_step_gradient,
    custom_gauss_newton_gradient,
    custom_jacobian,
    custom_jvp,
    custom_levenberg_marquardt_step,
    custom_vjp,
    dense_to_sparse_matrix,
    dual_cos,
    dual_exp,
    dual_log,
    dual_sin,
    empirical_fisher_conjugate_gradient,
    empirical_fisher_metric,
    empirical_fisher_vector_product,
    evaluate_levenberg_marquardt_step,
    finite_difference_gradient,
    finite_difference_hessian,
    finite_difference_hvp,
    finite_difference_jacobian,
    finite_difference_jvp,
    finite_difference_vjp,
    forward_mode_gradient,
    gauss_newton_gradient,
    grad,
    hessian,
    huber_residual_weights,
    implicit_fixed_point_sensitivity,
    implicit_stationary_sensitivity,
    jacobian,
    least_squares_covariance,
    levenberg_marquardt_step,
    natural_gradient,
    parameter_shift_gradient,
    parameter_shift_gradient_with_uncertainty,
    reverse_cos,
    reverse_exp,
    reverse_log,
    reverse_mode_gradient,
    reverse_sin,
    soft_l1_residual_weights,
    sparse_empirical_fisher_metric,
    sparse_hessian,
    sparse_jacobian,
    update_levenberg_marquardt_damping,
    value_and_complex_step_grad,
    value_and_custom_jacobian,
    value_and_custom_jvp,
    value_and_custom_vjp,
    value_and_finite_difference_hvp,
    value_and_finite_difference_jvp,
    value_and_forward_mode_grad,
    value_and_grad,
    value_and_hessian,
    value_and_jacobian,
    value_and_reverse_mode_grad,
    vector_jacobian_product,
    weighted_gradient_sum,
)
from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer
from scpn_quantum_control.qsnn.training import QSNNTrainer

try:
    from scpn_quantum_control.hardware.pennylane_adapter import (
        PennyLaneRunner,
        is_pennylane_available,
    )

    _PL_OK = is_pennylane_available()
except (ImportError, AttributeError):
    _PL_OK = False


def test_natural_gradient_damping_repairs_semidefinite_metric() -> None:
    """Damping should make semidefinite trainable metrics solvable."""

    gradient = GradientResult(
        value=1.0,
        gradient=np.array([2.0]),
        method="finite_difference_central",
        shift=1.0e-6,
        coefficient=5.0e5,
        evaluations=3,
        parameter_names=("x",),
        trainable=(True,),
    )
    result = natural_gradient(gradient, np.array([[0.0]]), damping=0.5)

    np.testing.assert_allclose(result.natural_gradient, [4.0])


def test_natural_gradient_rejects_invalid_metric() -> None:
    """Natural-gradient metrics must be symmetric positive definite."""

    gradient = GradientResult(
        value=1.0,
        gradient=np.array([1.0, 2.0]),
        method="finite_difference_central",
        shift=1.0e-6,
        coefficient=5.0e5,
        evaluations=5,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )

    with pytest.raises(ValueError, match="symmetric"):
        natural_gradient(gradient, np.array([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="positive definite"):
        natural_gradient(gradient, np.diag([1.0, 0.0]))
    with pytest.raises(ValueError, match="rcond"):
        natural_gradient(gradient, np.eye(2), rcond=0.0)


def test_parameter_shift_consistency_passes_for_shift_compatible_objective() -> None:
    """Gradient checks should pass for a standard sinusoidal generator rule."""

    result = check_parameter_shift_consistency(
        lambda values: math.sin(values[0]) + math.cos(values[1]),
        [0.3, -0.2],
        tolerance=1.0e-5,
    )

    assert isinstance(result, GradientCheckResult)
    assert result.passed
    assert result.candidate.method == "parameter_shift"
    assert result.reference.method == "finite_difference_central"
    assert result.max_abs_error <= 1.0e-5
    assert result.value_delta == pytest.approx(0.0)


def test_parameter_shift_consistency_detects_wrong_rule_coefficient() -> None:
    """Gradient checks should fail closed when a rule coefficient is invalid."""

    result = check_parameter_shift_consistency(
        lambda values: math.sin(values[0]),
        [0.3],
        rule=ParameterShiftRule(coefficient=0.25),
        tolerance=1.0e-5,
    )

    assert not result.passed
    assert result.max_abs_error > result.tolerance


def test_parameter_shift_consistency_rejects_invalid_tolerance() -> None:
    """Gradient-check tolerances must be explicit non-negative real scalars."""

    with pytest.raises(ValueError, match="gradient check tolerance must be a real numeric scalar"):
        check_parameter_shift_consistency(
            lambda values: math.sin(values[0]), [0.3], tolerance="1e-5"
        )
    with pytest.raises(
        ValueError, match="gradient check tolerance must be finite and non-negative"
    ):
        check_parameter_shift_consistency(
            lambda values: math.sin(values[0]), [0.3], tolerance=-1.0
        )


def test_gradient_descent_step_respects_trainable_mask() -> None:
    """Native optimizer step should update only trainable parameters."""

    result = GradientResult(
        value=1.0,
        gradient=np.array([2.0, 0.0]),
        method="parameter_shift",
        shift=math.pi / 2,
        coefficient=0.5,
        evaluations=5,
        parameter_names=("a", "b"),
        trainable=(True, False),
    )
    optimizer = DifferentiableOptimizer(learning_rate=0.1)

    updated = optimizer.step([1.0, 5.0], result)

    np.testing.assert_allclose(updated, [0.8, 5.0])


def test_gradient_descent_step_projects_box_bounds() -> None:
    """Optimizer updates should respect explicit parameter box constraints."""

    result = GradientResult(
        value=1.0,
        gradient=np.array([10.0, -10.0]),
        method="parameter_shift",
        shift=math.pi / 2,
        coefficient=0.5,
        evaluations=5,
        parameter_names=("a", "b"),
        trainable=(True, True),
    )
    optimizer = DifferentiableOptimizer(learning_rate=0.2)

    updated = optimizer.step(
        [0.0, 0.0],
        result,
        bounds=[ParameterBounds(lower=-0.5, upper=0.5), ParameterBounds(lower=-0.25, upper=0.25)],
    )

    np.testing.assert_allclose(updated, [-0.5, 0.25])


def test_gradient_descent_step_wraps_periodic_bounds() -> None:
    """Periodic quantum-angle domains should wrap instead of clip."""

    result = GradientResult(
        value=1.0,
        gradient=np.array([-4.0 * math.pi]),
        method="parameter_shift",
        shift=math.pi / 2,
        coefficient=0.5,
        evaluations=3,
        parameter_names=("angle",),
        trainable=(True,),
    )
    optimizer = DifferentiableOptimizer(learning_rate=1.0)

    updated = optimizer.step(
        [0.25],
        result,
        bounds=[ParameterBounds(lower=-math.pi, upper=math.pi, periodic=True)],
    )

    np.testing.assert_allclose(updated, [0.25], atol=1.0e-12)


def test_gradient_descent_step_clips_trainable_gradient_norm() -> None:
    """Optimizer steps should optionally clip trainable gradient norm."""

    result = GradientResult(
        value=1.0,
        gradient=np.array([3.0, 4.0, 0.0]),
        method="parameter_shift",
        shift=math.pi / 2,
        coefficient=0.5,
        evaluations=7,
        parameter_names=("a", "b", "frozen"),
        trainable=(True, True, False),
    )
    optimizer = DifferentiableOptimizer(learning_rate=1.0)

    updated = optimizer.step([0.0, 0.0, 0.0], result, max_gradient_norm=1.0)

    np.testing.assert_allclose(updated, [-0.6, -0.8, 0.0], atol=1.0e-12)


def test_parameter_bounds_reject_invalid_intervals() -> None:
    """Box constraints must be explicit finite ordered real intervals."""

    with pytest.raises(ValueError, match="lower bound must be a real numeric scalar"):
        ParameterBounds(lower="0.0")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="less than or equal"):
        ParameterBounds(lower=1.0, upper=0.0)
    with pytest.raises(ValueError, match="periodic flag"):
        ParameterBounds(lower=-math.pi, upper=math.pi, periodic=np.bool_(True))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="periodic bounds require finite"):
        ParameterBounds(lower=-math.pi, periodic=True)
    with pytest.raises(ValueError, match="periodic bounds require lower < upper"):
        ParameterBounds(lower=1.0, upper=1.0, periodic=True)


def test_optimizer_minimize_converges_for_shift_compatible_quadratic() -> None:
    """Bounded optimizer should return convergence metadata and final values."""

    optimizer = DifferentiableOptimizer(learning_rate=0.25)
    result = optimizer.minimize(
        lambda values: 1.0 - math.cos(values[0]),
        [0.4],
        max_steps=80,
        gradient_tolerance=1.0e-8,
    )

    assert isinstance(result, OptimizationResult)
    assert result.converged
    assert result.reason == "gradient_tolerance"
    assert result.steps <= 80
    assert result.value_history[-1] <= result.value_history[0]
    assert result.best_value == pytest.approx(min(result.value_history))
    np.testing.assert_allclose(result.values, [0.0], atol=1.0e-6)
    np.testing.assert_allclose(result.best_values, result.values)


def test_optimizer_minimize_respects_frozen_parameters() -> None:
    """Non-trainable parameters must not move during multi-step optimization."""

    optimizer = DifferentiableOptimizer(learning_rate=0.2)
    result = optimizer.minimize(
        lambda values: 1.0 - math.cos(values[0]) + 1.0 - math.cos(values[1]),
        [0.3, 0.4],
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
        max_steps=5,
    )

    assert result.values[1] == pytest.approx(0.4)
    assert result.final_gradient.trainable == (True, False)


def test_optimizer_minimize_supports_finite_difference_backend() -> None:
    """Optimizer should support smooth non-parameter-shift objectives explicitly."""

    optimizer = DifferentiableOptimizer(learning_rate=0.2)
    result = optimizer.minimize(
        lambda values: values[0] ** 2,
        [0.5],
        gradient_method="finite_difference",
        finite_difference_step=1.0e-6,
        max_steps=80,
        gradient_tolerance=1.0e-7,
    )

    assert result.converged
    assert result.final_gradient.method == "finite_difference_central"
    np.testing.assert_allclose(result.values, [0.0], atol=1.0e-5)


def test_optimizer_minimize_projects_initial_and_updated_bounds() -> None:
    """Bounded minimize should project initial values and subsequent steps."""

    optimizer = DifferentiableOptimizer(learning_rate=0.5)
    result = optimizer.minimize(
        lambda values: values[0] ** 2,
        [2.0],
        gradient_method="finite_difference",
        bounds=[ParameterBounds(lower=-0.25, upper=0.25)],
        max_steps=10,
        gradient_tolerance=1.0e-7,
    )

    assert result.values[0] <= 0.25
    assert result.value_history[0] == pytest.approx(0.25**2)


def test_optimizer_minimize_wraps_initial_periodic_bounds() -> None:
    """Periodic bounds should canonicalize initial angles before evaluation."""

    optimizer = DifferentiableOptimizer(learning_rate=0.1)
    result = optimizer.minimize(
        lambda values: 1.0 - math.cos(values[0]),
        [3.0 * math.pi],
        bounds=[ParameterBounds(lower=-math.pi, upper=math.pi, periodic=True)],
        max_steps=0,
    )

    assert result.values[0] == pytest.approx(-math.pi)
    assert result.value_history[0] == pytest.approx(2.0)


def test_optimizer_minimize_accepts_gradient_clipping() -> None:
    """Bounded optimizer should route clipping through multi-step minimization."""

    optimizer = DifferentiableOptimizer(learning_rate=1.0)
    result = optimizer.minimize(
        lambda values: values[0] ** 2,
        [10.0],
        gradient_method="finite_difference",
        max_gradient_norm=0.5,
        max_steps=1,
    )

    assert not result.converged
    np.testing.assert_allclose(result.values, [9.5], atol=1.0e-5)


def test_optimizer_result_tracks_best_iterate_when_final_worsens() -> None:
    """OptimizationResult should preserve the best observed iterate."""

    gradient = GradientResult(
        value=3.0,
        gradient=np.array([1.0]),
        method="parameter_shift",
        shift=math.pi / 2,
        coefficient=0.5,
        evaluations=3,
        parameter_names=("theta",),
        trainable=(True,),
    )

    result = OptimizationResult(
        values=np.array([2.0]),
        final_gradient=gradient,
        value_history=(1.0, 3.0),
        steps=1,
        converged=False,
        reason="max_steps",
        best_values=np.array([0.0]),
        best_value=1.0,
    )

    assert result.best_value == pytest.approx(1.0)
    np.testing.assert_allclose(result.best_values, [0.0])


def test_optimizer_result_rejects_inconsistent_best_value() -> None:
    """Best-iterate provenance must agree with the reported value history."""

    gradient = GradientResult(
        value=1.0,
        gradient=np.array([1.0]),
        method="parameter_shift",
        shift=math.pi / 2,
        coefficient=0.5,
        evaluations=3,
        parameter_names=("theta",),
        trainable=(True,),
    )

    with pytest.raises(ValueError, match="best_value"):
        OptimizationResult(
            values=np.array([0.0]),
            final_gradient=gradient,
            value_history=(1.0, 0.5),
            steps=1,
            converged=False,
            reason="max_steps",
            best_values=np.array([0.0]),
            best_value=1.0,
        )


def test_optimizer_minimize_rejects_invalid_loop_controls() -> None:
    """Optimizer loop controls must fail closed before objective evaluation."""

    optimizer = DifferentiableOptimizer(learning_rate=0.1)
    with pytest.raises(ValueError, match="gradient_method"):
        optimizer.minimize(lambda values: math.sin(values[0]), [0.1], gradient_method="bogus")
    with pytest.raises(ValueError, match="finite_difference_step"):
        optimizer.minimize(
            lambda values: math.sin(values[0]),
            [0.1],
            gradient_method="finite_difference",
            finite_difference_step=0.0,
        )
    with pytest.raises(ValueError, match="max_gradient_norm"):
        optimizer.minimize(
            lambda values: math.sin(values[0]),
            [0.1],
            max_gradient_norm=0.0,
        )
    with pytest.raises(ValueError, match="max_steps"):
        optimizer.minimize(lambda values: math.sin(values[0]), [0.1], max_steps=True)
    with pytest.raises(ValueError, match="gradient_tolerance"):
        optimizer.minimize(lambda values: math.sin(values[0]), [0.1], gradient_tolerance="1e-3")
    with pytest.raises(ValueError, match="value_tolerance"):
        optimizer.minimize(lambda values: math.sin(values[0]), [0.1], value_tolerance=-1.0)
    with pytest.raises(ValueError, match="bounds length"):
        optimizer.minimize(
            lambda values: math.sin(values[0]),
            [0.1],
            bounds=[ParameterBounds(), ParameterBounds()],
        )


def test_gradient_result_rejects_malformed_metadata() -> None:
    """Gradient provenance should not allow silently inconsistent payloads."""

    with pytest.raises(ValueError, match="parameter_names"):
        GradientResult(
            value=1.0,
            gradient=np.array([1.0, 2.0]),
            method="parameter_shift",
            shift=math.pi / 2,
            coefficient=0.5,
            evaluations=5,
            parameter_names=("a",),
            trainable=(True, True),
        )
    with pytest.raises(ValueError, match="finite"):
        GradientResult(
            value=1.0,
            gradient=np.array([math.nan]),
            method="parameter_shift",
            shift=math.pi / 2,
            coefficient=0.5,
            evaluations=3,
            parameter_names=("a",),
            trainable=(True,),
        )
    with pytest.raises(ValueError, match="gradient must contain real numeric scalars"):
        GradientResult(
            value=1.0,
            gradient=np.array(["1.0"]),
            method="parameter_shift",
            shift=math.pi / 2,
            coefficient=0.5,
            evaluations=3,
            parameter_names=("a",),
            trainable=(True,),
        )
    with pytest.raises(ValueError, match="trainable mask must contain booleans"):
        GradientResult(
            value=1.0,
            gradient=np.array([1.0]),
            method="parameter_shift",
            shift=math.pi / 2,
            coefficient=0.5,
            evaluations=3,
            parameter_names=("a",),
            trainable=(np.bool_(True),),  # type: ignore[arg-type]
        )


def test_qsnn_trainer_uses_native_parameter_shift(monkeypatch: pytest.MonkeyPatch) -> None:
    """QSNN training should delegate gradient evaluation to the core primitive."""

    import scpn_quantum_control.qsnn.training as training

    calls: list[tuple[tuple[float, ...], tuple[str, ...]]] = []

    def fake_value_and_grad(objective, values, *, parameters, rule=None):
        calls.append((tuple(float(value) for value in values), tuple(p.name for p in parameters)))
        value = float(objective(values))
        return GradientResult(
            value=value,
            gradient=np.full(len(values), 0.25),
            method="parameter_shift",
            shift=math.pi / 2,
            coefficient=0.5,
            evaluations=1 + 2 * len(values),
            parameter_names=tuple(p.name for p in parameters),
            trainable=tuple(p.trainable for p in parameters),
        )

    monkeypatch.setattr(training, "value_and_parameter_shift_grad", fake_value_and_grad)

    layer = QuantumDenseLayer(1, 2, seed=0)
    trainer = QSNNTrainer(layer)
    gradient = trainer.parameter_shift_gradient(np.array([0.5, 0.25]), np.array([1.0]))

    assert calls
    assert calls[0][1] == ("synapse_0_0", "synapse_0_1")
    np.testing.assert_allclose(gradient, [[0.25, 0.25]])


def test_differentiable_api_exported_from_package_root() -> None:
    """Stable users should be able to import the native autodiff surface."""

    import scpn_quantum_control as scpn

    assert scpn.ArmijoLineSearchResult is ArmijoLineSearchResult
    assert scpn.armijo_backtracking_line_search is armijo_backtracking_line_search
    assert scpn.batch_custom_jacobian is batch_custom_jacobian
    assert scpn.batch_custom_jvp is batch_custom_jvp
    assert scpn.batch_custom_vjp is batch_custom_vjp
    assert scpn.batch_value_and_custom_jacobian is batch_value_and_custom_jacobian
    assert scpn.batch_value_and_custom_jvp is batch_value_and_custom_jvp
    assert scpn.batch_value_and_custom_vjp is batch_value_and_custom_vjp
    assert scpn.CustomDerivativeCheckResult is CustomDerivativeCheckResult
    assert scpn.CustomDerivativeRule is CustomDerivativeRule
    assert scpn.check_custom_derivative_consistency is check_custom_derivative_consistency
    assert scpn.custom_gauss_newton_gradient is custom_gauss_newton_gradient
    assert scpn.custom_jacobian is custom_jacobian
    assert scpn.custom_jvp is custom_jvp
    assert scpn.custom_levenberg_marquardt_step is custom_levenberg_marquardt_step
    assert scpn.custom_vjp is custom_vjp
    assert scpn.DualNumber is DualNumber
    assert scpn.FixedPointSensitivityResult is FixedPointSensitivityResult
    assert scpn.FisherConjugateGradientResult is FisherConjugateGradientResult
    assert scpn.FisherVectorProductResult is FisherVectorProductResult
    assert scpn.ParameterShiftRule is ParameterShiftRule
    assert scpn.ParameterBounds is ParameterBounds
    assert scpn.ReverseNode is ReverseNode
    assert scpn.ShotAllocationResult is ShotAllocationResult
    assert scpn.SparseMatrixResult is SparseMatrixResult
    assert scpn.StochasticGradientResult is StochasticGradientResult
    assert scpn.WeightedGradientResult is WeightedGradientResult
    assert scpn.allocate_parameter_shift_shots is allocate_parameter_shift_shots
    assert scpn.VJPResult is VJPResult
    assert scpn.batch_complex_step_gradient is batch_complex_step_gradient
    assert scpn.batch_value_and_complex_step_grad is batch_value_and_complex_step_grad
    assert scpn.complex_step_gradient is complex_step_gradient
    assert scpn.dual_cos is dual_cos
    assert scpn.dual_exp is dual_exp
    assert scpn.dual_log is dual_log
    assert scpn.dual_sin is dual_sin
    assert scpn.dense_to_sparse_matrix is dense_to_sparse_matrix
    assert scpn.value_and_complex_step_grad is value_and_complex_step_grad
    assert scpn.value_and_custom_jacobian is value_and_custom_jacobian
    assert scpn.value_and_custom_jvp is value_and_custom_jvp
    assert scpn.value_and_custom_vjp is value_and_custom_vjp
    assert scpn.parameter_shift_gradient is parameter_shift_gradient
    assert scpn.batch_finite_difference_hvp is batch_finite_difference_hvp
    assert scpn.batch_finite_difference_jvp is batch_finite_difference_jvp
    assert scpn.batch_finite_difference_vjp is batch_finite_difference_vjp
    assert scpn.DifferentiableOptimizer is DifferentiableOptimizer
    assert scpn.OptimizationResult is OptimizationResult
    assert scpn.HVPResult is HVPResult
    assert scpn.HessianResult is HessianResult
    assert scpn.ImplicitSensitivityResult is ImplicitSensitivityResult
    assert scpn.JVPResult is JVPResult
    assert scpn.JacobianResult is JacobianResult
    assert scpn.LeastSquaresCovarianceResult is LeastSquaresCovarianceResult
    assert scpn.LevenbergMarquardtDampingUpdate is LevenbergMarquardtDampingUpdate
    assert scpn.LevenbergMarquardtOptimizer is LevenbergMarquardtOptimizer
    assert scpn.LevenbergMarquardtResult is LevenbergMarquardtResult
    assert scpn.LevenbergMarquardtStep is LevenbergMarquardtStep
    assert scpn.LevenbergMarquardtTrial is LevenbergMarquardtTrial
    assert scpn.NaturalGradientOptimizationResult is NaturalGradientOptimizationResult
    assert scpn.NaturalGradientOptimizer is NaturalGradientOptimizer
    assert scpn.NaturalGradientResult is NaturalGradientResult
    assert scpn.finite_difference_gradient is finite_difference_gradient
    assert scpn.empirical_fisher_conjugate_gradient is empirical_fisher_conjugate_gradient
    assert scpn.empirical_fisher_vector_product is empirical_fisher_vector_product
    assert scpn.empirical_fisher_metric is empirical_fisher_metric
    assert scpn.evaluate_levenberg_marquardt_step is evaluate_levenberg_marquardt_step
    assert scpn.finite_difference_hessian is finite_difference_hessian
    assert scpn.finite_difference_hvp is finite_difference_hvp
    assert scpn.finite_difference_jacobian is finite_difference_jacobian
    assert scpn.finite_difference_jvp is finite_difference_jvp
    assert scpn.finite_difference_vjp is finite_difference_vjp
    assert scpn.forward_mode_gradient is forward_mode_gradient
    assert scpn.gauss_newton_gradient is gauss_newton_gradient
    assert scpn.grad is grad
    assert scpn.hessian is hessian
    assert scpn.implicit_fixed_point_sensitivity is implicit_fixed_point_sensitivity
    assert scpn.huber_residual_weights is huber_residual_weights
    assert scpn.implicit_stationary_sensitivity is implicit_stationary_sensitivity
    assert scpn.jacobian is jacobian
    assert scpn.least_squares_covariance is least_squares_covariance
    assert scpn.levenberg_marquardt_step is levenberg_marquardt_step
    assert scpn.natural_gradient is natural_gradient
    assert (
        scpn.parameter_shift_gradient_with_uncertainty is parameter_shift_gradient_with_uncertainty
    )
    assert scpn.reverse_cos is reverse_cos
    assert scpn.reverse_exp is reverse_exp
    assert scpn.reverse_log is reverse_log
    assert scpn.reverse_mode_gradient is reverse_mode_gradient
    assert scpn.reverse_sin is reverse_sin
    assert scpn.soft_l1_residual_weights is soft_l1_residual_weights
    assert scpn.sparse_empirical_fisher_metric is sparse_empirical_fisher_metric
    assert scpn.sparse_hessian is sparse_hessian
    assert scpn.sparse_jacobian is sparse_jacobian
    assert scpn.update_levenberg_marquardt_damping is update_levenberg_marquardt_damping
    assert scpn.weighted_gradient_sum is weighted_gradient_sum
    assert scpn.check_parameter_shift_consistency is check_parameter_shift_consistency
    assert scpn.batch_value_and_parameter_shift_grad is batch_value_and_parameter_shift_grad
    assert scpn.batch_value_and_finite_difference_grad is batch_value_and_finite_difference_grad
    assert scpn.batch_value_and_finite_difference_hvp is batch_value_and_finite_difference_hvp
    assert scpn.batch_value_and_finite_difference_jvp is batch_value_and_finite_difference_jvp
    assert scpn.batch_value_and_finite_difference_vjp is batch_value_and_finite_difference_vjp
    assert scpn.batch_vector_jacobian_product is batch_vector_jacobian_product
    assert scpn.value_and_finite_difference_hvp is value_and_finite_difference_hvp
    assert scpn.value_and_finite_difference_jvp is value_and_finite_difference_jvp
    assert scpn.value_and_forward_mode_grad is value_and_forward_mode_grad
    assert scpn.value_and_grad is value_and_grad
    assert scpn.value_and_hessian is value_and_hessian
    assert scpn.value_and_jacobian is value_and_jacobian
    assert scpn.value_and_reverse_mode_grad is value_and_reverse_mode_grad
    assert scpn.vector_jacobian_product is vector_jacobian_product


@pytest.mark.skipif(not _PL_OK, reason="PennyLane not available or broken")
def test_pennylane_runner_exposes_vqe_value_and_grad() -> None:
    """PennyLane adapter should expose differentiable VQE params, not only optimize."""

    K = np.array([[0.0, 0.25], [0.25, 0.0]])
    omega = np.array([0.1, -0.2])
    runner = PennyLaneRunner(K, omega, device="default.qubit")
    params = np.array([0.01, -0.02, 0.03, 0.04, -0.05, 0.06])

    result = runner.vqe_value_and_grad(params, ansatz_depth=1)

    assert isinstance(result, GradientResult)
    assert result.method == "pennylane_autodiff"
    assert result.gradient.shape == params.shape
    assert result.parameter_names[0] == "vqe_0"
    assert np.isfinite(result.value)
    assert np.all(np.isfinite(result.gradient))
