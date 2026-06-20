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

from scpn_quantum_control import differentiable as differentiable_module
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
    ParameterShiftSampleRecord,
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
    batch_parameter_shift_gradient,
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
    is_jax_autodiff_available,
    jacobian,
    jax_value_and_grad,
    least_squares_covariance,
    levenberg_marquardt_step,
    multi_frequency_parameter_shift_rule,
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
    value_and_finite_difference_grad,
    value_and_finite_difference_hessian,
    value_and_finite_difference_hvp,
    value_and_finite_difference_jacobian,
    value_and_finite_difference_jvp,
    value_and_forward_mode_grad,
    value_and_grad,
    value_and_hessian,
    value_and_jacobian,
    value_and_parameter_shift_grad,
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


def test_parameter_shift_matches_sine_derivative() -> None:
    """Single-parameter shift should recover the exact derivative of sin."""

    theta = 0.37
    rule = ParameterShiftRule()

    gradient = parameter_shift_gradient(
        lambda values: math.sin(values[0]),
        [theta],
        rule=rule,
    )

    np.testing.assert_allclose(gradient, [math.cos(theta)], atol=1e-12)


def test_value_and_parameter_shift_grad_returns_metadata() -> None:
    """The public helper should return value, gradient, and explicit provenance."""

    result = value_and_parameter_shift_grad(
        lambda values: math.sin(values[0]) + math.cos(values[1]),
        [0.1, -0.2],
        parameters=[Parameter("theta"), Parameter("phi", trainable=False)],
    )

    assert isinstance(result, GradientResult)
    assert result.value == pytest.approx(math.sin(0.1) + math.cos(-0.2))
    np.testing.assert_allclose(result.gradient, [math.cos(0.1), 0.0], atol=1e-12)
    assert result.method == "parameter_shift"
    assert result.trainable == (True, False)
    assert result.parameter_names == ("theta", "phi")
    assert result.evaluations == 3


def test_parameter_shift_rejects_non_scalar_objective() -> None:
    """Gradient objectives must be scalar-valued to avoid silent shape bugs."""

    with pytest.raises(ValueError, match="scalar"):
        parameter_shift_gradient(lambda _values: np.array([1.0, 2.0]), [0.1])


def test_parameter_shift_rejects_implicit_parameter_coercion() -> None:
    """Differentiable parameters must be explicit real numeric values."""

    with pytest.raises(ValueError, match="parameters must contain real numeric scalars"):
        parameter_shift_gradient(lambda values: math.sin(values[0]), ["0.1"])

    with pytest.raises(ValueError, match="parameters must contain real numeric scalars"):
        parameter_shift_gradient(lambda values: math.sin(values[0]), [True])


def test_parameter_shift_rejects_non_real_objective_scalar() -> None:
    """Objective return values must be explicit finite real scalars."""

    with pytest.raises(ValueError, match="differentiable objective must return a scalar"):
        parameter_shift_gradient(lambda _values: "1.0", [0.1])

    with pytest.raises(ValueError, match="differentiable objective must return a scalar"):
        parameter_shift_gradient(lambda _values: 1.0 + 0.0j, [0.1])


def test_parameter_metadata_validation_and_custom_rule() -> None:
    """Parameter metadata and custom rules should fail closed."""

    with pytest.raises(ValueError, match="non-empty"):
        Parameter("")
    with pytest.raises(ValueError, match="boolean"):
        Parameter("theta", trainable=np.bool_(True))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        value_and_parameter_shift_grad(
            lambda values: math.sin(values[0]) + math.sin(values[1]),
            [0.1, 0.2],
            parameters=[Parameter("theta"), Parameter("theta")],
        )
    with pytest.raises(ValueError, match="positive"):
        ParameterShiftRule(shift=0.0)
    with pytest.raises(ValueError, match="coefficient must be a real numeric scalar"):
        ParameterShiftRule(coefficient="0.5")  # type: ignore[arg-type]

    result = value_and_parameter_shift_grad(
        lambda values: math.sin(2.0 * values[0]),
        [0.3],
        rule=ParameterShiftRule(shift=math.pi / 4.0, coefficient=1.0),
    )

    np.testing.assert_allclose(result.gradient, [2.0 * math.cos(0.6)], atol=1e-12)


def test_multi_frequency_parameter_shift_rule_matches_analytic_reference() -> None:
    """Multi-frequency rules should differentiate wider generator spectra exactly."""

    theta = 0.23
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0, 3.0])

    def objective(values: np.ndarray) -> float:
        return float(
            math.sin(values[0])
            + 0.2 * math.cos(2.0 * values[0])
            + 0.05 * math.sin(3.0 * values[0])
        )

    result = value_and_parameter_shift_grad(objective, [theta], rule=rule)
    expected = math.cos(theta) - 0.4 * math.sin(2.0 * theta) + 0.15 * math.cos(3.0 * theta)

    assert result.method == "multi_frequency_parameter_shift"
    assert result.shift is None
    assert result.coefficient is None
    assert result.evaluations == 1 + 2 * len(rule.terms)
    assert rule.frequencies == (1.0, 2.0, 3.0)
    np.testing.assert_allclose(result.gradient, [expected], atol=1e-12)


def test_multi_frequency_parameter_shift_preserves_trainable_metadata() -> None:
    """Multi-term rules should skip frozen parameters without spending probes."""

    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    result = value_and_parameter_shift_grad(
        lambda values: math.sin(values[0]) + math.sin(2.0 * values[1]),
        [0.2, -0.3],
        parameters=[Parameter("theta"), Parameter("frozen_phi", trainable=False)],
        rule=rule,
    )

    np.testing.assert_allclose(result.gradient, [math.cos(0.2), 0.0], atol=1e-12)
    assert result.parameter_names == ("theta", "frozen_phi")
    assert result.trainable == (True, False)
    assert result.evaluations == 1 + 2 * len(rule.terms)


def test_multi_frequency_parameter_shift_rule_rejects_invalid_systems() -> None:
    """Multi-frequency rule construction must fail closed for ambiguous systems."""

    with pytest.raises(ValueError, match="positive"):
        multi_frequency_parameter_shift_rule([0.0, 1.0])
    with pytest.raises(ValueError, match="unique"):
        multi_frequency_parameter_shift_rule([1.0, 1.0])
    with pytest.raises(ValueError, match="same length"):
        multi_frequency_parameter_shift_rule([1.0, 2.0], shifts=[0.1])
    with pytest.raises(ValueError, match="ill-conditioned"):
        multi_frequency_parameter_shift_rule([1.0, 2.0], shifts=[math.pi, 2.0 * math.pi])


def test_multi_frequency_parameter_shift_propagates_per_term_shot_noise() -> None:
    """Multi-frequency shot noise should propagate independent per-term records."""

    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    plus = np.array([[1.4, 10.0], [0.9, 20.0]])
    minus = np.array([[0.2, 11.0], [0.1, 19.0]])
    plus_variance = np.array([[0.20, 0.30], [0.40, 0.50]])
    minus_variance = np.array([[0.10, 0.20], [0.30, 0.40]])
    plus_shots = np.array([[100, 101], [200, 201]])
    minus_shots = np.array([[120, 121], [220, 221]])

    result = parameter_shift_gradient_with_uncertainty(
        plus,
        minus,
        plus_variance,
        minus_variance,
        plus_shots,
        minus_shots,
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
        rule=rule,
    )

    expected_gradient = np.zeros(2)
    expected_variance = np.zeros(2)
    for term_index, (_shift, coefficient) in enumerate(rule.terms):
        expected_gradient[0] += coefficient * (plus[term_index, 0] - minus[term_index, 0])
        expected_variance[0] += coefficient**2 * (
            plus_variance[term_index, 0] / plus_shots[term_index, 0]
            + minus_variance[term_index, 0] / minus_shots[term_index, 0]
        )

    assert result.method == "multi_frequency_parameter_shift_shot_noise"
    assert result.shift is None
    assert result.coefficient is None
    assert result.evaluations == 2 * len(rule.terms)
    assert result.shots.shape == (len(rule.terms), 2, 2)
    assert len(result.records) == len(rule.terms) * 2
    assert all(isinstance(record, ParameterShiftSampleRecord) for record in result.records)
    assert all(record.trainable == (record.parameter_name == "theta") for record in result.records)
    assert all(
        record.gradient_contribution == 0.0
        for record in result.records
        if record.parameter_name == "frozen"
    )
    np.testing.assert_allclose(result.gradient, expected_gradient)
    np.testing.assert_allclose(result.standard_error, np.sqrt(expected_variance))
    np.testing.assert_allclose(np.diag(result.covariance), expected_variance)
    np.testing.assert_allclose(result.shots[:, 0, :], plus_shots)
    np.testing.assert_allclose(result.shots[:, 1, :], minus_shots)


def test_multi_frequency_parameter_shift_allocates_per_term_shots() -> None:
    """Shot allocation should support all multi-frequency plus/minus terms."""

    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    allocation = allocate_parameter_shift_shots(
        [[0.20, 0.30], [0.40, 0.50]],
        [[0.10, 0.20], [0.30, 0.40]],
        target_standard_error=0.15,
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
        rule=rule,
        min_shots=7,
    )

    assert allocation.method == "multi_frequency_parameter_shift_target_se"
    assert allocation.shots.shape == (len(rule.terms), 2, 2)
    assert allocation.total_shots == int(np.sum(allocation.shots))
    assert allocation.predicted_standard_error[0] <= 0.15
    assert allocation.predicted_standard_error[1] == 0.0
    assert np.all(allocation.shots[:, :, 1] == 7)


def test_multi_frequency_parameter_shift_rejects_ambiguous_shot_tensors() -> None:
    """Multi-frequency shot-noise inputs must expose the term dimension."""

    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    with pytest.raises(ValueError, match="shape"):
        parameter_shift_gradient_with_uncertainty(
            [1.0, 0.0],
            [0.0, 0.0],
            [0.1, 0.1],
            [0.1, 0.1],
            [100, 100],
            rule=rule,
        )
    with pytest.raises(ValueError, match="first dimension"):
        allocate_parameter_shift_shots(
            [[0.1, 0.1, 0.1]],
            [[0.1, 0.1, 0.1]],
            target_standard_error=0.1,
            rule=rule,
        )


def test_batch_parameter_shift_gradient_stacks_independent_objectives() -> None:
    """Batch helper should produce one gradient row per scalar objective."""

    gradients = batch_parameter_shift_gradient(
        [
            lambda values: math.sin(values[0]),
            lambda values: math.cos(values[0]),
        ],
        [0.25],
    )

    np.testing.assert_allclose(gradients[:, 0], [math.cos(0.25), -math.sin(0.25)])


def test_batch_value_gradient_results_preserve_metadata() -> None:
    """Batch value APIs should preserve objective values and provenance."""

    parameter_shift_results = batch_value_and_parameter_shift_grad(
        [
            lambda values: math.sin(values[0]),
            lambda values: math.cos(values[0]),
        ],
        [0.25],
        parameters=[Parameter("theta")],
    )
    finite_difference_results = batch_value_and_finite_difference_grad(
        [
            lambda values: values[0] ** 2,
            lambda values: 3.0 * values[0],
        ],
        [0.25],
        parameters=[Parameter("theta")],
    )
    complex_step_results = batch_value_and_complex_step_grad(
        [
            lambda values: np.sin(values[0]),
            lambda values: values[0] ** 3,
        ],
        [0.25],
        parameters=[Parameter("theta")],
    )

    assert len(parameter_shift_results) == 2
    assert len(complex_step_results) == 2
    assert parameter_shift_results[0].parameter_names == ("theta",)
    assert parameter_shift_results[0].method == "parameter_shift"
    assert complex_step_results[0].method == "complex_step"
    assert finite_difference_results[0].method == "finite_difference_central"
    np.testing.assert_allclose(
        [result.value for result in parameter_shift_results],
        [math.sin(0.25), math.cos(0.25)],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(finite_difference_results[0].gradient, [0.5], atol=1.0e-6)
    np.testing.assert_allclose(finite_difference_results[1].gradient, [3.0], atol=1.0e-6)
    np.testing.assert_allclose(
        [result.gradient[0] for result in complex_step_results],
        [math.cos(0.25), 3.0 * 0.25**2],
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_batch_value_gradient_results_reject_empty_objectives() -> None:
    """Batch value APIs must fail closed on empty objective lists."""

    with pytest.raises(ValueError, match="objectives"):
        batch_value_and_parameter_shift_grad([], [0.25])
    with pytest.raises(ValueError, match="objectives"):
        batch_value_and_finite_difference_grad([], [0.25])
    with pytest.raises(ValueError, match="objectives"):
        batch_value_and_complex_step_grad([], [0.25])


def test_batch_complex_step_gradient_matches_analytic_derivatives() -> None:
    """Batched complex-step gradients should stack analytic scalar results."""

    gradients = batch_complex_step_gradient(
        [
            lambda values: np.sin(values[0]) + values[1] ** 2,
            lambda values: values[0] * values[1],
        ],
        [0.25, -0.5],
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
    )

    assert gradients.shape == (2, 2)
    np.testing.assert_allclose(
        gradients,
        [[math.cos(0.25), 0.0], [-0.5, 0.0]],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    with pytest.raises(ValueError, match="objectives"):
        batch_complex_step_gradient([], [0.25])


def test_canonical_gradient_transform_dispatches_supported_methods() -> None:
    """Canonical grad transforms should provide a stable method-selected API."""

    parameter_shift = value_and_grad(
        lambda values: np.sin(values[0]),
        [0.25],
        parameters=[Parameter("theta")],
    )
    finite_difference = value_and_grad(
        lambda values: values[0] ** 2,
        [0.25],
        method="finite_difference",
        step=1.0e-6,
    )
    complex_step = value_and_grad(
        lambda values: np.exp(values[0]),
        [0.25],
        method="complex_step",
    )

    assert parameter_shift.method == "parameter_shift"
    assert finite_difference.method == "finite_difference_central"
    assert complex_step.method == "complex_step"
    np.testing.assert_allclose(grad(lambda values: np.sin(values[0]), [0.25]), [math.cos(0.25)])
    np.testing.assert_allclose(finite_difference.gradient, [0.5], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        complex_step.gradient,
        [math.exp(0.25)],
        rtol=1.0e-14,
        atol=1.0e-14,
    )

    with pytest.raises(ValueError, match="gradient method"):
        value_and_grad(lambda values: values[0], [0.25], method="reverse")


def test_canonical_jacobian_and_hessian_transforms() -> None:
    """Canonical second-order transform names should dispatch with provenance."""

    jacobian_result = value_and_jacobian(
        lambda values: np.array([values[0] ** 2, values[0] + 3.0 * values[1]]),
        [2.0, -1.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
        step=1.0e-6,
    )
    hessian_result = value_and_hessian(
        lambda values: values[0] ** 2 + values[0] * values[1],
        [2.0, -1.0],
        step=1.0e-4,
    )

    assert isinstance(jacobian_result, JacobianResult)
    assert isinstance(hessian_result, HessianResult)
    np.testing.assert_allclose(jacobian_result.jacobian, [[4.0, 0.0], [1.0, 0.0]], atol=1.0e-6)
    np.testing.assert_allclose(
        jacobian(lambda values: np.array([values[0] + values[1]]), [2.0, -1.0]),
        [[1.0, 1.0]],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        hessian_result.hessian,
        [[2.0, 1.0], [1.0, 0.0]],
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        hessian(lambda values: values[0] ** 2, [2.0]),
        [[2.0]],
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    with pytest.raises(ValueError, match="Jacobian method"):
        value_and_jacobian(lambda values: np.array([values[0]]), [0.25], method="reverse")
    with pytest.raises(ValueError, match="Hessian method"):
        value_and_hessian(lambda values: values[0] ** 2, [0.25], method="reverse")


def test_forward_mode_dual_gradient_matches_analytic_derivative() -> None:
    """Forward-mode dual numbers should propagate exact first-order tangents."""

    def objective(values: tuple[DualNumber, ...]) -> DualNumber:
        return dual_sin(values[0]) + values[0] * values[1] + values[1] ** 2

    result = value_and_forward_mode_grad(
        objective,
        [0.25, -0.5],
        parameters=[Parameter("theta"), Parameter("bias")],
    )

    assert result.method == "forward_mode_dual"
    assert result.evaluations == 3
    assert result.parameter_names == ("theta", "bias")
    np.testing.assert_allclose(
        result.gradient,
        [math.cos(0.25) - 0.5, 0.25 - 1.0],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        forward_mode_gradient(lambda values: dual_exp(values[0]) + dual_log(values[0]), [2.0]),
        [math.exp(2.0) + 0.5],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        grad(lambda values: dual_cos(values[0]), [0.25], method="forward_mode"),
        [-math.sin(0.25)],
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_forward_mode_dual_gradient_respects_frozen_parameters() -> None:
    """Forward-mode gradients should keep frozen tangent lanes zeroed."""

    result = value_and_forward_mode_grad(
        lambda values: values[0] ** 2 + values[0] * values[1],
        [3.0, 5.0],
        parameters=[Parameter("active"), Parameter("frozen", trainable=False)],
    )

    assert result.trainable == (True, False)
    assert result.evaluations == 2
    np.testing.assert_allclose(result.gradient, [11.0, 0.0])


def test_forward_mode_dual_rejects_invalid_contracts() -> None:
    """Forward-mode AD should fail closed on invalid scalar/domain contracts."""

    with pytest.raises(ValueError, match="forward-mode objective must return a scalar"):
        forward_mode_gradient(lambda _values: np.array([1.0, 2.0]), [1.0])
    with pytest.raises(ValueError, match="dual log input must be positive"):
        forward_mode_gradient(lambda values: dual_log(values[0]), [-1.0])
    with pytest.raises(ValueError, match="dual division denominator"):
        forward_mode_gradient(lambda values: values[0] / (values[0] - 1.0), [1.0])
    with pytest.raises(ValueError, match="dual variable exponent"):
        forward_mode_gradient(lambda values: (-1.0) ** values[0], [2.0])


def test_reverse_mode_tape_gradient_matches_analytic_derivative() -> None:
    """Reverse-mode tape gradients should backpropagate exact adjoints."""

    def objective(values: tuple[ReverseNode, ...]) -> ReverseNode:
        return reverse_sin(values[0]) + values[0] * values[1] + values[1] ** 2

    result = value_and_reverse_mode_grad(
        objective,
        [0.25, -0.5],
        parameters=[Parameter("theta"), Parameter("bias")],
    )

    assert result.method == "reverse_mode_tape"
    assert result.evaluations == 1
    assert result.parameter_names == ("theta", "bias")
    np.testing.assert_allclose(
        result.gradient,
        [math.cos(0.25) - 0.5, 0.25 - 1.0],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        reverse_mode_gradient(
            lambda values: reverse_exp(values[0]) + reverse_log(values[0]), [2.0]
        ),
        [math.exp(2.0) + 0.5],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        grad(lambda values: reverse_cos(values[0]), [0.25], method="reverse_mode"),
        [-math.sin(0.25)],
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_reverse_mode_tape_gradient_respects_frozen_parameters() -> None:
    """Reverse-mode gradients should backpropagate once and mask frozen outputs."""

    result = value_and_reverse_mode_grad(
        lambda values: values[0] ** 2 + values[0] * values[1],
        [3.0, 5.0],
        parameters=[Parameter("active"), Parameter("frozen", trainable=False)],
    )

    assert result.trainable == (True, False)
    assert result.evaluations == 1
    np.testing.assert_allclose(result.gradient, [11.0, 0.0])


def test_reverse_mode_tape_rejects_invalid_contracts() -> None:
    """Reverse-mode AD should fail closed on invalid scalar/domain contracts."""

    with pytest.raises(ValueError, match="reverse-mode objective must return a scalar"):
        reverse_mode_gradient(lambda _values: np.array([1.0, 2.0]), [1.0])
    with pytest.raises(ValueError, match="reverse log input must be positive"):
        reverse_mode_gradient(lambda values: reverse_log(values[0]), [-1.0])
    with pytest.raises(ValueError, match="reverse division denominator"):
        reverse_mode_gradient(lambda values: values[0] / (values[0] - 1.0), [1.0])
    with pytest.raises(ValueError, match="reverse variable exponent"):
        reverse_mode_gradient(lambda values: (-1.0) ** values[0], [2.0])


def test_sparse_matrix_result_round_trips_dense_derivatives() -> None:
    """Sparse coordinate derivatives should preserve dense values and metadata."""

    dense = np.array([[1.0, 0.0, 2.0e-8], [0.0, 0.0, -3.0]])
    sparse = dense_to_sparse_matrix(
        dense,
        parameter_names=("a", "b", "c"),
        trainable=(True, False, True),
        tolerance=1.0e-7,
    )

    assert isinstance(sparse, SparseMatrixResult)
    assert sparse.nnz == 2
    assert sparse.shape == dense.shape
    assert sparse.parameter_names == ("a", "b", "c")
    assert sparse.trainable == (True, False, True)
    np.testing.assert_allclose(sparse.to_dense(), [[1.0, 0.0, 0.0], [0.0, 0.0, -3.0]])


def test_sparse_jacobian_hessian_and_fisher_preserve_provenance() -> None:
    """Sparse helpers should convert derivative result objects without metadata loss."""

    jacobian_result = JacobianResult(
        value=np.array([1.0, -2.0]),
        jacobian=np.array([[1.0, 0.0], [2.0, 0.0]]),
        method="analytic",
        step=1.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, False),
    )
    hessian_result = HessianResult(
        value=1.0,
        hessian=np.array([[2.0, 0.0], [0.0, 0.0]]),
        method="analytic",
        step=1.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, False),
    )

    sparse_j = sparse_jacobian(jacobian_result)
    sparse_h = sparse_hessian(hessian_result)
    sparse_fisher = sparse_empirical_fisher_metric(jacobian_result)

    assert sparse_j.method == "sparse:analytic"
    assert sparse_h.method == "sparse:analytic"
    assert sparse_fisher.method == "sparse:empirical_fisher"
    assert sparse_j.parameter_names == ("x", "y")
    assert sparse_h.trainable == (True, False)
    np.testing.assert_allclose(sparse_j.to_dense(), jacobian_result.jacobian)
    np.testing.assert_allclose(sparse_h.to_dense(), hessian_result.hessian)
    np.testing.assert_allclose(sparse_fisher.to_dense(), [[5.0, 0.0], [0.0, 0.0]])


def test_sparse_matrix_result_rejects_invalid_contracts() -> None:
    """Sparse derivative containers must fail closed on malformed coordinates."""

    with pytest.raises(ValueError, match="duplicate"):
        SparseMatrixResult(
            row_indices=np.array([0, 0]),
            column_indices=np.array([1, 1]),
            values=np.array([1.0, 2.0]),
            shape=(2, 2),
            method="bad",
            parameter_names=("x", "y"),
            trainable=(True, True),
        )
    with pytest.raises(ValueError, match="inside matrix shape"):
        SparseMatrixResult(
            row_indices=np.array([2]),
            column_indices=np.array([0]),
            values=np.array([1.0]),
            shape=(2, 2),
            method="bad",
            parameter_names=("x", "y"),
            trainable=(True, True),
        )
    with pytest.raises(ValueError, match="parameter_names"):
        dense_to_sparse_matrix(np.eye(2), parameter_names=("x",))
    with pytest.raises(ValueError, match="sparse tolerance"):
        dense_to_sparse_matrix(np.eye(2), tolerance=-1.0)


def test_implicit_stationary_sensitivity_solves_trainable_system() -> None:
    """Implicit stationary sensitivities should solve -H^{-1}B on trainable parameters."""

    result = implicit_stationary_sensitivity(
        hessian=np.diag([2.0, 4.0, 9.0]),
        cross_derivative=np.array([[4.0, -2.0], [8.0, 4.0], [9.0, 9.0]]),
        parameters=[Parameter("x"), Parameter("y", trainable=False), Parameter("z")],
        hyperparameter_names=("alpha", "beta"),
    )

    assert isinstance(result, ImplicitSensitivityResult)
    assert result.method == "implicit_stationary_sensitivity"
    assert result.parameter_names == ("x", "y", "z")
    assert result.hyperparameter_names == ("alpha", "beta")
    assert result.trainable == (True, False, True)
    np.testing.assert_allclose(
        result.sensitivity,
        [[-2.0, 1.0], [0.0, 0.0], [-1.0, -1.0]],
    )
    assert result.condition_number == pytest.approx(9.0 / 2.0)


def test_implicit_stationary_sensitivity_applies_damping() -> None:
    """Implicit sensitivity should expose damped positive-definite solves."""

    result = implicit_stationary_sensitivity(
        hessian=np.diag([1.0, 3.0]),
        cross_derivative=[2.0, 6.0],
        damping=1.0,
    )

    np.testing.assert_allclose(result.sensitivity, [[-1.0], [-1.5]])
    assert result.damping == pytest.approx(1.0)
    assert result.hyperparameter_names == ("alpha0",)


def test_implicit_stationary_sensitivity_rejects_invalid_contracts() -> None:
    """Implicit solves must fail closed on invalid stationary systems."""

    with pytest.raises(ValueError, match="square"):
        implicit_stationary_sensitivity([[1.0, 0.0]], [[1.0]])
    with pytest.raises(ValueError, match="row count"):
        implicit_stationary_sensitivity(np.eye(2), [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="symmetric"):
        implicit_stationary_sensitivity([[1.0, 2.0], [0.0, 1.0]], [[1.0], [1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        implicit_stationary_sensitivity([[0.0]], [[1.0]])
    with pytest.raises(ValueError, match="hyperparameter_names"):
        implicit_stationary_sensitivity(np.eye(2), np.ones((2, 2)), hyperparameter_names=("a",))
    with pytest.raises(ValueError, match="implicit rcond"):
        implicit_stationary_sensitivity(np.eye(1), [[1.0]], rcond=0.0)


def test_implicit_fixed_point_sensitivity_solves_trainable_system() -> None:
    """Fixed-point sensitivities should solve (I - dT/dx)^{-1}dT/dalpha."""

    result = implicit_fixed_point_sensitivity(
        state_jacobian=np.diag([0.5, 0.2, 0.0]),
        parameter_jacobian=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        parameters=[Parameter("x"), Parameter("y", trainable=False), Parameter("z")],
        hyperparameter_names=("alpha", "beta"),
    )

    assert isinstance(result, FixedPointSensitivityResult)
    assert result.method == "implicit_fixed_point_sensitivity"
    assert result.parameter_names == ("x", "y", "z")
    assert result.hyperparameter_names == ("alpha", "beta")
    assert result.trainable == (True, False, True)
    np.testing.assert_allclose(
        result.system_matrix,
        np.diag([0.5, 0.8, 1.0]),
    )
    np.testing.assert_allclose(
        result.sensitivity,
        [[2.0, 4.0], [0.0, 0.0], [5.0, 6.0]],
    )
    assert result.condition_number == pytest.approx(2.0)


def test_implicit_fixed_point_sensitivity_applies_damping() -> None:
    """Fixed-point sensitivities should expose damped nonsingular solves."""

    result = implicit_fixed_point_sensitivity(
        state_jacobian=[[0.5]],
        parameter_jacobian=[2.0],
        damping=0.5,
    )

    np.testing.assert_allclose(result.system_matrix, [[1.0]])
    np.testing.assert_allclose(result.sensitivity, [[2.0]])
    assert result.damping == pytest.approx(0.5)
    assert result.hyperparameter_names == ("alpha0",)


def test_implicit_fixed_point_sensitivity_rejects_invalid_contracts() -> None:
    """Fixed-point implicit differentiation must fail closed on bad systems."""

    with pytest.raises(ValueError, match="square"):
        implicit_fixed_point_sensitivity([[0.1, 0.0]], [[1.0]])
    with pytest.raises(ValueError, match="row count"):
        implicit_fixed_point_sensitivity(np.eye(2), [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="ill-conditioned"):
        implicit_fixed_point_sensitivity([[1.0]], [[1.0]])
    with pytest.raises(ValueError, match="hyperparameter_names"):
        implicit_fixed_point_sensitivity(np.eye(2), np.ones((2, 2)), hyperparameter_names=("a",))
    with pytest.raises(ValueError, match="fixed-point rcond"):
        implicit_fixed_point_sensitivity(np.eye(1), [[1.0]], rcond=0.0)
    with pytest.raises(ValueError, match="fixed-point damping"):
        implicit_fixed_point_sensitivity(np.eye(1), [[1.0]], damping=-1.0)


def test_custom_derivative_rule_evaluates_exact_jvp_and_vjp() -> None:
    """Custom rules should expose exact primitive derivatives with provenance."""

    rule = CustomDerivativeRule(
        name="quadratic_coupler",
        value_fn=lambda values: np.array([values[0] * values[1], values[0] ** 2]),
        jvp_rule=lambda values, tangent: np.array(
            [
                tangent[0] * values[1] + values[0] * tangent[1],
                2.0 * values[0] * tangent[0],
            ]
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [
                cotangent[0] * values[1] + 2.0 * cotangent[1] * values[0],
                cotangent[0] * values[0],
            ]
        ),
        parameter_names=("theta", "phi"),
        trainable=(True, False),
    )

    jvp_result = value_and_custom_jvp(rule, [2.0, 3.0], [0.5, 7.0])
    vjp_result = value_and_custom_vjp(rule, [2.0, 3.0], [11.0, 13.0])

    np.testing.assert_allclose(custom_jvp(rule, [2.0, 3.0], [0.5, 7.0]), [1.5, 2.0])
    np.testing.assert_allclose(jvp_result.value, [6.0, 4.0])
    np.testing.assert_allclose(jvp_result.tangent, [0.5, 0.0])
    np.testing.assert_allclose(jvp_result.jvp, [1.5, 2.0])
    assert jvp_result.method == "custom_jvp:quadratic_coupler"
    assert jvp_result.step == pytest.approx(0.0)
    assert jvp_result.parameter_names == ("theta", "phi")
    assert jvp_result.trainable == (True, False)
    np.testing.assert_allclose(custom_vjp(rule, [2.0, 3.0], [11.0, 13.0]).vjp, [85.0, 0.0])
    np.testing.assert_allclose(vjp_result.value, [6.0, 4.0])
    np.testing.assert_allclose(vjp_result.vjp, [85.0, 0.0])
    assert vjp_result.method == "custom_vjp:quadratic_coupler"


def test_custom_derivative_rule_rejects_invalid_contracts() -> None:
    """Custom derivative rules must fail closed on bad exact-rule contracts."""

    with pytest.raises(ValueError, match="requires a JVP or VJP"):
        CustomDerivativeRule(name="bad", value_fn=lambda values: values)
    with pytest.raises(ValueError, match="lengths must match"):
        CustomDerivativeRule(
            name="bad_meta",
            value_fn=lambda values: values,
            jvp_rule=lambda values, tangent: tangent,
            parameter_names=("x", "y"),
            trainable=(True,),
        )

    rule = CustomDerivativeRule(
        name="bad_shapes",
        value_fn=lambda values: np.array([values[0]]),
        jvp_rule=lambda values, tangent: np.array([1.0, 2.0]),
        vjp_rule=lambda values, cotangent: np.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match="JVP output shape"):
        value_and_custom_jvp(rule, [1.0], [1.0])
    with pytest.raises(ValueError, match="cotangent shape"):
        value_and_custom_vjp(rule, [1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="VJP output length"):
        value_and_custom_vjp(rule, [1.0], [1.0])
    with pytest.raises(ValueError, match="does not define a JVP"):
        value_and_custom_jvp(
            CustomDerivativeRule(
                name="vjp_only",
                value_fn=lambda values: values,
                vjp_rule=lambda values, cotangent: cotangent,
            ),
            [1.0],
            [1.0],
        )


def test_check_custom_derivative_consistency_passes_exact_rules() -> None:
    """Custom JVP/VJP rules should satisfy adjoint and finite-difference checks."""

    rule = CustomDerivativeRule(
        name="sin_cos_pair",
        value_fn=lambda values: np.array([np.sin(values[0]), values[0] * values[1]]),
        jvp_rule=lambda values, tangent: np.array(
            [
                np.cos(values[0]) * tangent[0],
                values[1] * tangent[0] + values[0] * tangent[1],
            ]
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [
                np.cos(values[0]) * cotangent[0] + values[1] * cotangent[1],
                values[0] * cotangent[1],
            ]
        ),
        parameter_names=("theta", "phi"),
    )

    result = check_custom_derivative_consistency(
        rule,
        values=[0.4, -0.2],
        tangent=[0.3, 0.7],
        cotangent=[-0.5, 0.25],
        finite_difference_step=1.0e-6,
        tolerance=1.0e-6,
    )

    assert isinstance(result, CustomDerivativeCheckResult)
    assert result.passed is True
    assert result.adjoint_inner_error <= 1.0e-12
    assert result.jvp_l2_error <= 1.0e-6
    assert result.vjp_l2_error <= 1.0e-6
    assert result.reference_jvp.method == "finite_difference_directional"
    assert result.reference_vjp.method == "vjp:finite_difference_central"


def test_check_custom_derivative_consistency_detects_bad_adjoint_rule() -> None:
    """Incorrect exact rules should fail closed without being silently trusted."""

    rule = CustomDerivativeRule(
        name="bad_linear",
        value_fn=lambda values: np.array([2.0 * values[0]]),
        jvp_rule=lambda values, tangent: np.array([2.0 * tangent[0]]),
        vjp_rule=lambda values, cotangent: np.array([3.0 * cotangent[0]]),
    )

    result = check_custom_derivative_consistency(
        rule,
        values=[1.5],
        tangent=[0.25],
        cotangent=[4.0],
        finite_difference_step=1.0e-6,
        tolerance=1.0e-8,
    )

    assert result.passed is False
    assert result.adjoint_inner_error == pytest.approx(1.0)
    assert result.vjp_l2_error == pytest.approx(4.0, rel=1.0e-6)


def test_custom_jacobian_materializes_exact_jvp_columns() -> None:
    """Custom JVP rules should materialise exact dense Jacobian columns."""

    rule = CustomDerivativeRule(
        name="quadratic_vector",
        value_fn=lambda values: np.array([values[0] * values[1], values[0] ** 2]),
        jvp_rule=lambda values, tangent: np.array(
            [
                values[1] * tangent[0] + values[0] * tangent[1],
                2.0 * values[0] * tangent[0],
            ]
        ),
        parameter_names=("theta", "frozen_phi"),
        trainable=(True, False),
    )

    result = value_and_custom_jacobian(rule, [2.0, 3.0])

    assert isinstance(result, JacobianResult)
    assert result.method == "custom_jacobian:quadratic_vector"
    assert result.step == pytest.approx(0.0)
    assert result.evaluations == 1
    assert result.parameter_names == ("theta", "frozen_phi")
    assert result.trainable == (True, False)
    np.testing.assert_allclose(result.value, [6.0, 4.0])
    np.testing.assert_allclose(result.jacobian, [[3.0, 0.0], [4.0, 0.0]])
    np.testing.assert_allclose(custom_jacobian(rule, [2.0, 3.0]), result.jacobian)


def test_custom_jacobian_materializes_exact_vjp_rows() -> None:
    """VJP-only rules should materialise exact dense Jacobian rows."""

    rule = CustomDerivativeRule(
        name="linear_readout",
        value_fn=lambda values: np.array([values[0] + 2.0 * values[1], -values[0]]),
        vjp_rule=lambda values, cotangent: np.array(
            [cotangent[0] - cotangent[1], 2.0 * cotangent[0]]
        ),
    )

    result = value_and_custom_jacobian(rule, [0.25, -0.5])

    np.testing.assert_allclose(result.value, [-0.75, -0.25])
    np.testing.assert_allclose(result.jacobian, [[1.0, 2.0], [-1.0, 0.0]])
    assert result.method == "custom_jacobian:linear_readout"


def test_custom_jacobian_rejects_invalid_exact_rule_shapes() -> None:
    """Custom Jacobian materialisation must reject malformed exact derivatives."""

    with pytest.raises(ValueError, match="CustomDerivativeRule"):
        value_and_custom_jacobian(object(), [1.0])  # type: ignore[arg-type]

    bad_jvp = CustomDerivativeRule(
        name="bad_jvp",
        value_fn=lambda values: np.array([values[0]]),
        jvp_rule=lambda values, tangent: np.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match="JVP output shape"):
        value_and_custom_jacobian(bad_jvp, [1.0])

    bad_vjp = CustomDerivativeRule(
        name="bad_vjp",
        value_fn=lambda values: np.array([values[0]]),
        vjp_rule=lambda values, cotangent: np.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match="VJP output length"):
        value_and_custom_jacobian(bad_vjp, [1.0])


def test_batched_custom_jvp_and_vjp_apply_exact_rules() -> None:
    """Batched custom transforms should preserve exact rule outputs and metadata."""

    rule = CustomDerivativeRule(
        name="affine_pair",
        value_fn=lambda values: np.array([values[0] + values[1], values[0] - values[1]]),
        jvp_rule=lambda values, tangent: np.array(
            [tangent[0] + tangent[1], tangent[0] - tangent[1]]
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [cotangent[0] + cotangent[1], cotangent[0] - cotangent[1]]
        ),
        parameter_names=("theta", "frozen_phi"),
        trainable=(True, False),
    )

    jvp_results = batch_value_and_custom_jvp(
        rule,
        [2.0, 5.0],
        [[1.0, 10.0], [-2.0, 30.0]],
    )
    vjp_results = batch_value_and_custom_vjp(
        rule,
        [2.0, 5.0],
        [[3.0, 4.0], [-1.0, 2.0]],
    )

    assert len(jvp_results) == 2
    assert len(vjp_results) == 2
    np.testing.assert_allclose(
        batch_custom_jvp(rule, [2.0, 5.0], [[1.0, 10.0], [-2.0, 30.0]]), [[1.0, 1.0], [-2.0, -2.0]]
    )
    np.testing.assert_allclose(
        batch_custom_vjp(rule, [2.0, 5.0], [[3.0, 4.0], [-1.0, 2.0]]), [[7.0, 0.0], [1.0, 0.0]]
    )
    assert jvp_results[0].parameter_names == ("theta", "frozen_phi")
    assert vjp_results[0].trainable == (True, False)


def test_batched_custom_jacobian_materializes_parameter_batches() -> None:
    """Custom Jacobians should batch over parameter rows for benchmark workflows."""

    rule = CustomDerivativeRule(
        name="quadratic_vector",
        value_fn=lambda values: np.array([values[0] * values[1], values[0] ** 2]),
        jvp_rule=lambda values, tangent: np.array(
            [
                values[1] * tangent[0] + values[0] * tangent[1],
                2.0 * values[0] * tangent[0],
            ]
        ),
    )

    results = batch_value_and_custom_jacobian(rule, [[2.0, 3.0], [-1.0, 4.0]])

    assert len(results) == 2
    np.testing.assert_allclose(
        batch_custom_jacobian(rule, [[2.0, 3.0], [-1.0, 4.0]]),
        [
            [[3.0, 2.0], [4.0, 0.0]],
            [[4.0, -1.0], [-2.0, 0.0]],
        ],
    )
    np.testing.assert_allclose(results[1].value, [-4.0, 1.0])
    with pytest.raises(ValueError, match="two-dimensional batch"):
        batch_value_and_custom_jacobian(rule, [1.0, 2.0])


def test_custom_gauss_newton_gradient_uses_exact_custom_jacobian() -> None:
    """Exact custom residual Jacobians should feed Gauss-Newton directly."""

    rule = CustomDerivativeRule(
        name="scaled_residual",
        value_fn=lambda values: np.array([2.0 * values[0] - 1.0, values[1] + 3.0]),
        jvp_rule=lambda values, tangent: np.array([2.0 * tangent[0], tangent[1]]),
        parameter_names=("theta", "frozen_phi"),
        trainable=(True, False),
    )

    result = custom_gauss_newton_gradient(
        rule,
        [2.0, -1.0],
        weights=[1.0, 4.0],
        damping=1.0,
    )

    assert isinstance(result, NaturalGradientResult)
    assert result.base_gradient.method == "gauss_newton:custom_jacobian:scaled_residual"
    assert result.base_gradient.parameter_names == ("theta", "frozen_phi")
    assert result.base_gradient.trainable == (True, False)
    assert result.base_gradient.value == pytest.approx(12.5)
    np.testing.assert_allclose(result.base_gradient.gradient, [6.0, 0.0])
    np.testing.assert_allclose(result.metric, [[5.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(result.natural_gradient, [1.2, 0.0])


def test_custom_levenberg_marquardt_step_uses_exact_custom_jacobian() -> None:
    """Exact custom residual Jacobians should feed bounded LM candidates."""

    rule = CustomDerivativeRule(
        name="identity_residual",
        value_fn=lambda values: np.array([values[0], 2.0 * values[1]]),
        jvp_rule=lambda values, tangent: np.array([tangent[0], 2.0 * tangent[1]]),
    )

    result = custom_levenberg_marquardt_step(
        rule,
        [2.0, -1.0],
        damping=1.0,
        max_step_norm=1.0,
    )

    assert isinstance(result, LevenbergMarquardtStep)
    assert (
        result.gauss_newton.base_gradient.method
        == "gauss_newton:custom_jacobian:identity_residual"
    )
    np.testing.assert_allclose(result.gauss_newton.base_gradient.gradient, [2.0, -4.0])
    assert np.linalg.norm(result.step) == pytest.approx(1.0)
    np.testing.assert_allclose(result.candidate_values, np.array([2.0, -1.0]) + result.step)
    assert result.predicted_reduction > 0.0


def test_parameter_shift_gradient_with_uncertainty_propagates_shot_noise() -> None:
    """Independent plus/minus shot noise should propagate into gradient variance."""

    result = parameter_shift_gradient_with_uncertainty(
        plus_values=[0.8, 0.1],
        minus_values=[0.2, -0.3],
        plus_variances=[0.36, 0.25],
        minus_variances=[0.16, 0.09],
        plus_shots=[900, 400],
        minus_shots=[400, 100],
        value=0.5,
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
    )

    assert isinstance(result, StochasticGradientResult)
    assert result.method == "parameter_shift_shot_noise"
    assert result.claim_boundary == differentiable_module.STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY
    assert result.hardware_execution is False
    assert result.evaluations == 2
    assert result.parameter_names == ("theta", "frozen")
    assert result.trainable == (True, False)
    assert len(result.records) == 2
    active_record, frozen_record = result.records
    assert active_record.parameter_name == "theta"
    assert active_record.shift == pytest.approx(math.pi / 2.0)
    assert active_record.coefficient == pytest.approx(0.5)
    assert active_record.plus_shots == 900
    assert active_record.minus_shots == 400
    assert active_record.gradient_contribution == pytest.approx(0.3)
    assert active_record.variance_contribution > 0.0
    assert frozen_record.parameter_name == "frozen"
    assert frozen_record.trainable is False
    assert frozen_record.gradient_contribution == pytest.approx(0.0)
    assert frozen_record.variance_contribution == pytest.approx(0.0)
    np.testing.assert_allclose(result.gradient, [0.3, 0.0])
    expected_variance = 0.5**2 * (0.36 / 900.0 + 0.16 / 400.0)
    np.testing.assert_allclose(result.standard_error, [math.sqrt(expected_variance), 0.0])
    np.testing.assert_allclose(result.covariance, np.diag([expected_variance, 0.0]))
    np.testing.assert_allclose(result.confidence_radius, 1.959963984540054 * result.standard_error)
    np.testing.assert_allclose(result.shots, [[900.0, 400.0], [400.0, 100.0]])
    evidence = result.to_dict()
    assert (
        evidence["claim_boundary"]
        == differentiable_module.STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY
    )
    assert evidence["hardware_execution"] is False
    assert len(evidence["records"]) == 2


def test_parameter_shift_gradient_with_uncertainty_rejects_invalid_inputs() -> None:
    """Shot-noise gradients must fail closed on impossible measurement contracts."""

    with pytest.raises(ValueError, match="variance shapes"):
        parameter_shift_gradient_with_uncertainty([1.0], [0.0], [0.1, 0.2], [0.1], [10])
    with pytest.raises(ValueError, match="shot variances"):
        parameter_shift_gradient_with_uncertainty([1.0], [0.0], [-0.1], [0.1], [10])
    with pytest.raises(ValueError, match="shot counts"):
        parameter_shift_gradient_with_uncertainty([1.0], [0.0], [0.1], [0.1], [0])
    with pytest.raises(ValueError, match="confidence_level"):
        parameter_shift_gradient_with_uncertainty(
            [1.0],
            [0.0],
            [0.1],
            [0.1],
            [10],
            confidence_level=1.0,
        )
    with pytest.raises(ValueError, match="confidence_z"):
        parameter_shift_gradient_with_uncertainty(
            [1.0],
            [0.0],
            [0.1],
            [0.1],
            [10],
            confidence_z=0.0,
        )
    with pytest.raises(ValueError, match="hardware execution"):
        StochasticGradientResult(
            value=0.0,
            gradient=np.array([1.0]),
            standard_error=np.array([0.1]),
            covariance=np.eye(1),
            confidence_radius=np.array([0.2]),
            shots=np.array([[10.0], [10.0]]),
            confidence_level=0.95,
            method="parameter_shift_shot_noise",
            shift=math.pi / 2.0,
            coefficient=0.5,
            evaluations=2,
            parameter_names=("theta",),
            trainable=(True,),
            hardware_execution=True,
        )


def test_allocate_parameter_shift_shots_meets_target_standard_error() -> None:
    """Shot allocation should conservatively meet target gradient uncertainty."""

    allocation = allocate_parameter_shift_shots(
        plus_variances=[0.36, 0.25],
        minus_variances=[0.16, 0.09],
        target_standard_error=0.02,
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
    )

    assert isinstance(allocation, ShotAllocationResult)
    assert allocation.method == "parameter_shift_target_se"
    assert allocation.parameter_names == ("theta", "frozen")
    assert allocation.trainable == (True, False)
    assert allocation.shots.shape == (2, 2)
    assert allocation.shots[0, 0] >= 1.0
    assert allocation.shots[1, 0] >= 1.0
    assert allocation.shots[0, 1] == 1.0
    assert allocation.shots[1, 1] == 1.0
    assert allocation.predicted_standard_error[0] <= 0.02
    assert allocation.predicted_standard_error[1] == pytest.approx(0.0)
    assert allocation.total_shots == int(np.sum(allocation.shots))
    np.testing.assert_allclose(
        allocation.covariance, np.diag(allocation.predicted_standard_error**2)
    )


def test_allocate_parameter_shift_shots_respects_caps() -> None:
    """Shot allocation should report capped uncertainty when budgets are bounded."""

    allocation = allocate_parameter_shift_shots(
        plus_variances=[1.0],
        minus_variances=[1.0],
        target_standard_error=1.0e-3,
        min_shots=4,
        max_shots_per_evaluation=10,
    )

    np.testing.assert_allclose(allocation.shots, [[10.0], [10.0]])
    assert allocation.predicted_standard_error[0] > allocation.target_standard_error


def test_allocate_parameter_shift_shots_rejects_invalid_inputs() -> None:
    """Shot allocation must fail closed on impossible planning contracts."""

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


def test_weighted_gradient_sum_preserves_component_provenance() -> None:
    """Weighted multi-objective aggregation should keep component metadata."""

    components = batch_value_and_parameter_shift_grad(
        [
            lambda values: math.sin(values[0]),
            lambda values: math.cos(values[0]),
        ],
        [0.25],
        parameters=[Parameter("theta")],
    )
    result = weighted_gradient_sum(components, np.array([0.75, 0.25]))

    assert isinstance(result, WeightedGradientResult)
    assert result.components == components
    assert result.evaluations == sum(component.evaluations for component in components)
    assert result.parameter_names == ("theta",)
    assert result.value == pytest.approx(0.75 * math.sin(0.25) + 0.25 * math.cos(0.25))
    np.testing.assert_allclose(
        result.gradient,
        [0.75 * math.cos(0.25) - 0.25 * math.sin(0.25)],
    )


def test_weighted_gradient_sum_rejects_incompatible_components() -> None:
    """Weighted aggregation must fail closed on metadata mismatches."""

    first = value_and_parameter_shift_grad(
        lambda values: math.sin(values[0]),
        [0.1],
        parameters=[Parameter("theta")],
    )
    second = value_and_parameter_shift_grad(
        lambda values: math.sin(values[0]),
        [0.1],
        parameters=[Parameter("phi")],
    )

    with pytest.raises(ValueError, match="weights length"):
        weighted_gradient_sum([first], np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="parameter_names"):
        weighted_gradient_sum([first, second], np.array([0.5, 0.5]))


def test_finite_difference_gradient_matches_quadratic_derivative() -> None:
    """Finite-difference backend should support non-parameter-shift diagnostics."""

    result = value_and_finite_difference_grad(
        lambda values: values[0] ** 2 + 3.0 * values[1],
        [2.0, -1.0],
        parameters=[Parameter("x"), Parameter("bias", trainable=False)],
        step=1.0e-6,
    )

    assert result.method == "finite_difference_central"
    assert (
        result.claim_boundary == differentiable_module.FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
    )
    assert "diagnostic only" in result.claim_boundary
    assert "not analytic" in result.claim_boundary
    assert "whole-program AD" in result.claim_boundary
    assert "production benchmark evidence" in result.claim_boundary
    assert result.evaluations == 3
    np.testing.assert_allclose(result.gradient, [4.0, 0.0], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        finite_difference_gradient(lambda values: values[0] ** 2, [1.5]),
        [3.0],
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_finite_difference_gradient_rejects_invalid_step() -> None:
    """Finite-difference step size must be explicit finite positive real data."""

    with pytest.raises(ValueError, match="finite difference step must be a real numeric scalar"):
        finite_difference_gradient(lambda values: values[0] ** 2, [1.0], step="1e-6")
    with pytest.raises(ValueError, match="finite difference step must be finite and positive"):
        finite_difference_gradient(lambda values: values[0] ** 2, [1.0], step=0.0)


def test_derivative_result_claim_boundary_must_be_explicit() -> None:
    """Derivative result artefacts should reject blank claim boundaries."""

    with pytest.raises(ValueError, match="claim_boundary must be non-empty"):
        GradientResult(
            value=1.0,
            gradient=np.array([1.0]),
            method="finite_difference_central",
            shift=1.0e-6,
            coefficient=5.0e5,
            evaluations=3,
            parameter_names=("x",),
            trainable=(True,),
            claim_boundary=" ",
        )


def test_complex_step_gradient_matches_analytic_derivative() -> None:
    """Complex-step gradients should avoid finite-difference cancellation."""

    result = value_and_complex_step_grad(
        lambda values: np.sin(values[0]) + values[1] ** 3,
        [0.4, -0.2],
        parameters=[Parameter("x"), Parameter("y")],
    )

    assert isinstance(result, GradientResult)
    assert result.method == "complex_step"
    assert result.evaluations == 3
    assert result.parameter_names == ("x", "y")
    np.testing.assert_allclose(
        result.gradient,
        [np.cos(0.4), 3.0 * (-0.2) ** 2],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        complex_step_gradient(lambda values: values[0] ** 2, [3.0]),
        [6.0],
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_complex_step_gradient_respects_frozen_parameters() -> None:
    """Complex-step gradients must preserve trainable masks exactly."""

    result = value_and_complex_step_grad(
        lambda values: values[0] ** 2 + np.exp(values[1]),
        [2.0, 0.5],
        parameters=[Parameter("active"), Parameter("frozen", trainable=False)],
    )

    assert result.evaluations == 2
    assert result.trainable == (True, False)
    np.testing.assert_allclose(result.gradient, [4.0, 0.0], rtol=1.0e-14, atol=1.0e-14)


def test_complex_step_gradient_rejects_invalid_inputs() -> None:
    """Complex-step gradients should fail closed on invalid scalar contracts."""

    with pytest.raises(ValueError, match="complex-step step must be a real numeric scalar"):
        complex_step_gradient(lambda values: values[0] ** 2, [1.0], step="1e-30")
    with pytest.raises(ValueError, match="complex-step step must be finite and positive"):
        complex_step_gradient(lambda values: values[0] ** 2, [1.0], step=0.0)
    with pytest.raises(ValueError, match="complex-step objective must return a scalar"):
        complex_step_gradient(lambda values: np.array([values[0], values[0]]), [1.0])
    with pytest.raises(ValueError, match="complex-step objective must return a scalar"):
        complex_step_gradient(lambda values: "not numeric", [1.0])
    with pytest.raises(ValueError, match="complex-step objective returned a non-finite scalar"):
        complex_step_gradient(lambda values: values[0] * complex(np.nan, 1.0), [1.0])


def test_finite_difference_jacobian_matches_vector_objective() -> None:
    """Vector-valued differentiable diagnostics should expose Jacobians."""

    result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] ** 2, values[0] + 2.0 * values[1]]),
        [3.0, -1.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
        step=1.0e-6,
    )

    assert isinstance(result, JacobianResult)
    assert result.method == "finite_difference_central"
    assert (
        result.claim_boundary == differentiable_module.FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
    )
    assert result.evaluations == 3
    np.testing.assert_allclose(result.value, [9.0, 1.0])
    np.testing.assert_allclose(result.jacobian, [[6.0, 0.0], [1.0, 0.0]], atol=1.0e-6)
    np.testing.assert_allclose(
        finite_difference_jacobian(lambda values: np.array([values[0] ** 2]), [2.0]),
        [[4.0]],
        atol=1.0e-6,
    )


def test_finite_difference_jacobian_rejects_unstable_vector_shape() -> None:
    """Vector objectives must keep output shape stable across perturbations."""

    def unstable(values: np.ndarray) -> np.ndarray:
        if values[0] > 0.0:
            return np.array([values[0], values[0] ** 2])
        return np.array([values[0]])

    with pytest.raises(ValueError, match="shape must remain stable"):
        value_and_finite_difference_jacobian(unstable, [0.0])


def test_finite_difference_jacobian_rejects_non_vector_output() -> None:
    """Jacobian objectives must return explicit finite one-dimensional arrays."""

    with pytest.raises(ValueError, match="one-dimensional"):
        value_and_finite_difference_jacobian(lambda _values: np.array([[1.0]]), [0.0])
    with pytest.raises(ValueError, match="real numeric"):
        value_and_finite_difference_jacobian(lambda _values: np.array(["1.0"]), [0.0])


def test_finite_difference_jvp_matches_jacobian_directional_product() -> None:
    """Directional finite differences should expose native forward-mode JVPs."""

    def objective(values: np.ndarray) -> np.ndarray:
        return np.array([values[0] ** 2 + values[1], values[0] * values[1]])

    result = value_and_finite_difference_jvp(
        objective,
        [2.0, 3.0],
        [0.5, -1.0],
        parameters=[Parameter("x"), Parameter("y")],
    )

    assert isinstance(result, JVPResult)
    assert result.method == "finite_difference_directional"
    assert (
        result.claim_boundary == differentiable_module.FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
    )
    assert result.evaluations == 3
    np.testing.assert_allclose(result.value, [7.0, 6.0])
    np.testing.assert_allclose(result.tangent, [0.5, -1.0])
    np.testing.assert_allclose(result.jvp, [1.0, -0.5], atol=1.0e-6)
    np.testing.assert_allclose(
        finite_difference_jvp(objective, [2.0, 3.0], [0.5, -1.0]), [1.0, -0.5], atol=1.0e-6
    )


def test_finite_difference_jvp_respects_frozen_parameters() -> None:
    """Frozen parameters must be removed from directional tangents."""

    result = value_and_finite_difference_jvp(
        lambda values: np.array([values[0] + 10.0 * values[1]]),
        [1.0, 2.0],
        [0.25, 100.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
    )

    assert result.evaluations == 3
    np.testing.assert_allclose(result.tangent, [0.25, 0.0])
    np.testing.assert_allclose(result.jvp, [0.25], atol=1.0e-6)


def test_finite_difference_jvp_rejects_invalid_inputs() -> None:
    """JVP tangents and vector outputs must be finite and shape-stable."""

    with pytest.raises(ValueError, match="JVP tangent length"):
        value_and_finite_difference_jvp(lambda values: np.array([values[0]]), [1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="real numeric"):
        value_and_finite_difference_jvp(lambda values: np.array([values[0]]), [1.0], ["1.0"])

    def unstable(values: np.ndarray) -> np.ndarray:
        if values[0] > 0.0:
            return np.array([values[0], values[0] ** 2])
        return np.array([values[0]])

    with pytest.raises(ValueError, match="shape must remain stable"):
        value_and_finite_difference_jvp(unstable, [0.0], [1.0])


def test_vector_jacobian_product_contracts_cotangent() -> None:
    """Reverse-mode VJP should contract cotangents against validated Jacobians."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] ** 2, values[0] + 2.0 * values[1]]),
        [3.0, -1.0],
        parameters=[Parameter("x"), Parameter("y")],
    )
    result = vector_jacobian_product(jacobian_result, [0.5, -2.0])

    assert isinstance(result, VJPResult)
    assert result.method == "vjp:finite_difference_central"
    assert (
        result.claim_boundary == differentiable_module.FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
    )
    np.testing.assert_allclose(result.value, [9.0, 1.0])
    np.testing.assert_allclose(result.cotangent, [0.5, -2.0])
    np.testing.assert_allclose(result.vjp, [1.0, -4.0], atol=1.0e-6)
    np.testing.assert_allclose(
        finite_difference_vjp(
            lambda values: np.array([values[0] ** 2, values[0] + 2.0 * values[1]]),
            [3.0, -1.0],
            [0.5, -2.0],
        ).vjp,
        [1.0, -4.0],
        atol=1.0e-6,
    )


def test_vector_jacobian_product_respects_frozen_parameters_and_validation() -> None:
    """VJP products should zero frozen columns and reject malformed cotangents."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] + values[1], values[1] ** 2]),
        [1.0, 2.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
    )
    result = vector_jacobian_product(jacobian_result, [1.0, 10.0])

    np.testing.assert_allclose(result.vjp, [1.0, 0.0], atol=1.0e-6)
    with pytest.raises(ValueError, match="JacobianResult"):
        vector_jacobian_product(np.eye(2), [1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cotangent shape"):
        vector_jacobian_product(jacobian_result, [1.0])


def test_batch_finite_difference_jvp_returns_stacked_products_and_results() -> None:
    """Batched JVP helpers should preserve one result per tangent row."""

    def objective(values: np.ndarray) -> np.ndarray:
        return np.array([values[0] ** 2 + values[1], values[0] * values[1]])

    tangents = np.array([[1.0, 0.0], [0.0, 1.0]])
    results = batch_value_and_finite_difference_jvp(objective, [2.0, 3.0], tangents)
    stacked = batch_finite_difference_jvp(objective, [2.0, 3.0], tangents)

    assert len(results) == 2
    assert all(isinstance(result, JVPResult) for result in results)
    np.testing.assert_allclose(stacked, [[4.0, 3.0], [1.0, 2.0]], atol=1.0e-6)
    np.testing.assert_allclose(np.vstack([result.jvp for result in results]), stacked)


def test_batch_finite_difference_vjp_reuses_single_jacobian() -> None:
    """Batched VJP helpers should contract multiple cotangent rows."""

    def objective(values: np.ndarray) -> np.ndarray:
        return np.array([values[0] ** 2, values[0] + 2.0 * values[1]])

    cotangents = np.array([[1.0, 0.0], [0.0, 1.0]])
    results = batch_value_and_finite_difference_vjp(objective, [3.0, -1.0], cotangents)
    stacked = batch_finite_difference_vjp(objective, [3.0, -1.0], cotangents)

    assert len(results) == 2
    assert all(isinstance(result, VJPResult) for result in results)
    np.testing.assert_allclose(stacked, [[6.0, 0.0], [1.0, 2.0]], atol=1.0e-6)


def test_batch_vector_jacobian_product_contracts_existing_jacobian() -> None:
    """Batched VJP contraction should work from an existing validated Jacobian."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] ** 2, values[0] + 2.0 * values[1]]),
        [3.0, -1.0],
    )
    results = batch_vector_jacobian_product(jacobian_result, np.array([[1.0, 0.0], [0.0, 1.0]]))

    assert len(results) == 2
    np.testing.assert_allclose(
        [result.vjp for result in results], [[6.0, 0.0], [1.0, 2.0]], atol=1.0e-6
    )


def test_batch_finite_difference_hvp_returns_stacked_products_and_results() -> None:
    """Batched HVP helpers should preserve one result per tangent row."""

    def objective(values: np.ndarray) -> float:
        return float(values[0] ** 2 + 3.0 * values[0] * values[1] + 2.0 * values[1] ** 2)

    tangents = np.array([[1.0, 0.0], [0.0, 1.0]])
    results = batch_value_and_finite_difference_hvp(objective, [1.0, -1.0], tangents)
    stacked = batch_finite_difference_hvp(objective, [1.0, -1.0], tangents)

    assert len(results) == 2
    assert all(isinstance(result, HVPResult) for result in results)
    np.testing.assert_allclose(stacked, [[2.0, 3.0], [3.0, 4.0]], atol=1.0e-4)


def test_batch_transform_helpers_reject_malformed_batches() -> None:
    """Batched transform helpers should require explicit two-dimensional batches."""

    with pytest.raises(ValueError, match="two-dimensional batch"):
        batch_finite_difference_jvp(lambda values: np.array([values[0]]), [1.0], [1.0])
    with pytest.raises(ValueError, match="row length"):
        batch_finite_difference_jvp(lambda values: np.array([values[0]]), [1.0], [[1.0, 2.0]])
    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0], values[1]]),
        [1.0, 2.0],
    )
    with pytest.raises(ValueError, match="two-dimensional batch"):
        batch_vector_jacobian_product(jacobian_result, [1.0, 2.0])
    with pytest.raises(ValueError, match="row length"):
        batch_vector_jacobian_product(jacobian_result, [[1.0]])
    with pytest.raises(ValueError, match="two-dimensional batch"):
        batch_finite_difference_hvp(lambda values: float(values[0] ** 2), [1.0], [1.0])


def test_finite_difference_hessian_matches_quadratic_curvature() -> None:
    """Scalar differentiable diagnostics should expose second-order curvature."""

    result = value_and_finite_difference_hessian(
        lambda values: values[0] ** 2 + 3.0 * values[0] * values[1] + 2.0 * values[1] ** 2,
        [1.0, -1.0],
        step=1.0e-4,
    )

    assert isinstance(result, HessianResult)
    assert result.method == "finite_difference_central"
    assert (
        result.claim_boundary == differentiable_module.FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
    )
    np.testing.assert_allclose(result.hessian, [[2.0, 3.0], [3.0, 4.0]], atol=1.0e-5)
    np.testing.assert_allclose(
        finite_difference_hessian(lambda values: values[0] ** 2, [1.0]),
        [[2.0]],
        atol=1.0e-5,
    )


def test_finite_difference_hessian_respects_frozen_parameters() -> None:
    """Frozen parameters should have zero Hessian rows and columns."""

    result = value_and_finite_difference_hessian(
        lambda values: values[0] ** 2 + values[1] ** 2,
        [1.0, 2.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
    )

    np.testing.assert_allclose(result.hessian[:, 1], [0.0, 0.0])
    np.testing.assert_allclose(result.hessian[1, :], [0.0, 0.0])


def test_finite_difference_hessian_rejects_invalid_step() -> None:
    """Hessian step size must be explicit finite positive real data."""

    with pytest.raises(ValueError, match="finite difference step must be a real numeric scalar"):
        finite_difference_hessian(lambda values: values[0] ** 2, [1.0], step="1e-4")
    with pytest.raises(ValueError, match="finite difference step must be finite and positive"):
        finite_difference_hessian(lambda values: values[0] ** 2, [1.0], step=0.0)


def test_finite_difference_hvp_matches_quadratic_curvature_product() -> None:
    """Hessian-vector products should match full Hessian multiplication."""

    def objective(values: np.ndarray) -> float:
        return float(values[0] ** 2 + 3.0 * values[0] * values[1] + 2.0 * values[1] ** 2)

    result = value_and_finite_difference_hvp(objective, [1.0, -1.0], [0.5, -2.0])

    assert isinstance(result, HVPResult)
    assert result.method == "finite_difference_hvp"
    assert (
        result.claim_boundary == differentiable_module.FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
    )
    assert result.value == pytest.approx(0.0)
    np.testing.assert_allclose(result.tangent, [0.5, -2.0])
    np.testing.assert_allclose(result.hvp, [-5.0, -6.5], atol=1.0e-4)
    np.testing.assert_allclose(
        finite_difference_hvp(objective, [1.0, -1.0], [0.5, -2.0]),
        [-5.0, -6.5],
        atol=1.0e-4,
    )


def test_finite_difference_hvp_respects_frozen_parameters() -> None:
    """Frozen parameters should not contribute to HVP tangents or products."""

    result = value_and_finite_difference_hvp(
        lambda values: float(values[0] ** 2 + 100.0 * values[1] ** 2),
        [1.0, 2.0],
        [3.0, 999.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
    )

    np.testing.assert_allclose(result.tangent, [3.0, 0.0])
    np.testing.assert_allclose(result.hvp, [6.0, 0.0], atol=1.0e-4)


def test_finite_difference_hvp_rejects_invalid_inputs() -> None:
    """HVP tangents and controls must be finite, real, and shape-consistent."""

    with pytest.raises(ValueError, match="HVP tangent length"):
        value_and_finite_difference_hvp(lambda values: float(values[0] ** 2), [1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="real numeric"):
        value_and_finite_difference_hvp(lambda values: float(values[0] ** 2), [1.0], ["1.0"])
    with pytest.raises(ValueError, match="finite difference step"):
        value_and_finite_difference_hvp(
            lambda values: float(values[0] ** 2), [1.0], [1.0], step=0.0
        )


def test_natural_gradient_solves_trainable_metric_system() -> None:
    """Natural gradient should precondition only trainable parameters."""

    gradient = GradientResult(
        value=1.0,
        gradient=np.array([2.0, 4.0, 0.0]),
        method="finite_difference_central",
        shift=1.0e-6,
        coefficient=5.0e5,
        evaluations=7,
        parameter_names=("x", "y", "frozen"),
        trainable=(True, True, False),
    )
    result = natural_gradient(gradient, np.diag([2.0, 4.0, 99.0]))

    assert isinstance(result, NaturalGradientResult)
    np.testing.assert_allclose(result.natural_gradient, [1.0, 1.0, 0.0])
    assert result.condition_number == pytest.approx(2.0)


def test_natural_gradient_optimizer_converges_with_explicit_metric() -> None:
    """Natural-gradient optimization should compose gradients, metrics, and bounds."""

    optimizer = NaturalGradientOptimizer(learning_rate=0.5, damping=0.0, max_step_norm=1.0)
    result = optimizer.minimize(
        lambda values: float(0.5 * (4.0 * values[0] ** 2 + values[1] ** 2)),
        [2.0, -2.0],
        lambda _gradient, _values: np.diag([4.0, 1.0]),
        gradient_method="finite_difference",
        bounds=[ParameterBounds(lower=-3.0, upper=3.0), ParameterBounds(lower=-3.0, upper=3.0)],
        max_steps=80,
        gradient_tolerance=1.0e-7,
    )

    assert isinstance(result, NaturalGradientOptimizationResult)
    assert result.converged
    assert result.reason == "gradient_tolerance"
    assert result.best_value <= result.value_history[0]
    assert len(result.natural_step_norm_history) == result.steps
    np.testing.assert_allclose(result.values, [0.0, 0.0], atol=1.0e-5)


def test_natural_gradient_optimizer_respects_frozen_parameters() -> None:
    """Frozen parameters should not move even when the metric includes them."""

    optimizer = NaturalGradientOptimizer(learning_rate=0.5)
    result = optimizer.minimize(
        lambda values: values[0] ** 2 + values[1] ** 2,
        [2.0, 10.0],
        lambda _gradient, _values: np.eye(2),
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
        gradient_method="finite_difference",
        max_steps=40,
        gradient_tolerance=1.0e-7,
    )

    assert result.converged
    np.testing.assert_allclose(result.values, [0.0, 10.0], atol=1.0e-5)
    assert result.final_natural_gradient.natural_gradient[1] == pytest.approx(0.0)


def test_natural_gradient_optimizer_rejects_invalid_controls() -> None:
    """Natural-gradient optimizer controls and metric callback must fail closed."""

    with pytest.raises(ValueError, match="learning_rate"):
        NaturalGradientOptimizer(learning_rate=-1.0)
    with pytest.raises(ValueError, match="damping"):
        NaturalGradientOptimizer(damping=-1.0)
    with pytest.raises(ValueError, match="rcond"):
        NaturalGradientOptimizer(rcond=0.0)
    with pytest.raises(ValueError, match="max_step_norm"):
        NaturalGradientOptimizer(max_step_norm=0.0)

    optimizer = NaturalGradientOptimizer()
    with pytest.raises(ValueError, match="gradient_method"):
        optimizer.minimize(
            lambda values: float(values[0] ** 2),
            [1.0],
            lambda _gradient, _values: np.eye(1),
            gradient_method="unknown",
        )
    with pytest.raises(ValueError, match="positive definite"):
        optimizer.minimize(
            lambda values: float(values[0] ** 2),
            [1.0],
            lambda _gradient, _values: np.array([[0.0]]),
            gradient_method="finite_difference",
            max_steps=1,
        )


def test_armijo_backtracking_line_search_accepts_sufficient_decrease() -> None:
    """Armijo search should accept a descent step with explicit provenance."""

    gradient = value_and_finite_difference_grad(lambda values: float(values[0] ** 2), [2.0])
    result = armijo_backtracking_line_search(
        lambda values: float(values[0] ** 2),
        [2.0],
        gradient,
        -gradient.gradient,
        initial_step=1.0,
        contraction=0.5,
    )

    assert isinstance(result, ArmijoLineSearchResult)
    assert result.accepted
    assert result.reason == "accepted"
    assert result.step_size == pytest.approx(0.5)
    assert result.value < gradient.value
    assert result.value_history[0] == pytest.approx(4.0)
    np.testing.assert_allclose(result.values, [0.0], atol=1.0e-10)


def test_armijo_backtracking_line_search_respects_bounds_and_frozen_parameters() -> None:
    """Line search should project bounds and remove frozen direction components."""

    gradient = value_and_finite_difference_grad(
        lambda values: values[0] ** 2 + values[1] ** 2,
        [2.0, 10.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
    )
    result = armijo_backtracking_line_search(
        lambda values: values[0] ** 2 + values[1] ** 2,
        [2.0, 10.0],
        gradient,
        [-10.0, -999.0],
        bounds=[ParameterBounds(lower=-1.0, upper=1.0), ParameterBounds()],
    )

    assert result.accepted
    np.testing.assert_allclose(result.direction, [-10.0, 0.0])
    np.testing.assert_allclose(result.values, [-1.0, 10.0])


def test_armijo_backtracking_line_search_rejects_invalid_controls() -> None:
    """Line-search controls and directions must fail closed."""

    gradient = value_and_finite_difference_grad(lambda values: float(values[0] ** 2), [2.0])
    with pytest.raises(ValueError, match="GradientResult"):
        armijo_backtracking_line_search(
            lambda values: float(values[0] ** 2),
            [2.0],
            gradient.gradient,  # type: ignore[arg-type]
            [-1.0],
        )
    with pytest.raises(ValueError, match="direction length"):
        armijo_backtracking_line_search(
            lambda values: float(values[0] ** 2), [2.0], gradient, [-1.0, 0.0]
        )
    with pytest.raises(ValueError, match="initial_step"):
        armijo_backtracking_line_search(
            lambda values: float(values[0] ** 2), [2.0], gradient, [-1.0], initial_step=0.0
        )
    with pytest.raises(ValueError, match="contraction"):
        armijo_backtracking_line_search(
            lambda values: float(values[0] ** 2), [2.0], gradient, [-1.0], contraction=1.0
        )
    with pytest.raises(ValueError, match="sufficient_decrease"):
        armijo_backtracking_line_search(
            lambda values: float(values[0] ** 2), [2.0], gradient, [-1.0], sufficient_decrease=0.0
        )
    with pytest.raises(ValueError, match="max_steps"):
        armijo_backtracking_line_search(
            lambda values: float(values[0] ** 2), [2.0], gradient, [-1.0], max_steps=0
        )


def test_armijo_backtracking_line_search_rejects_non_descent_direction() -> None:
    """Non-descent directions should return a rejected result without trials."""

    gradient = value_and_finite_difference_grad(lambda values: float(values[0] ** 2), [2.0])
    result = armijo_backtracking_line_search(
        lambda values: float(values[0] ** 2),
        [2.0],
        gradient,
        gradient.gradient,
    )

    assert result.accepted is False
    assert result.reason == "non_descent_direction"
    assert result.step_size == pytest.approx(0.0)
    assert result.evaluations == 1
    np.testing.assert_allclose(result.values, [2.0])


def test_empirical_fisher_metric_from_jacobian_result() -> None:
    """Fisher/Gauss-Newton metrics should be constructible from Jacobians."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0], 2.0 * values[1]]),
        [1.0, 2.0],
    )
    metric = empirical_fisher_metric(
        jacobian_result,
        weights=np.array([1.0, 0.5]),
        damping=0.25,
    )

    np.testing.assert_allclose(metric, [[1.25, 0.0], [0.0, 2.25]], atol=1.0e-6)


def test_empirical_fisher_metric_rejects_invalid_weights() -> None:
    """Fisher metric weights must match residual rows and stay non-negative."""

    with pytest.raises(ValueError, match="weights"):
        empirical_fisher_metric(np.eye(2), weights=np.array([1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        empirical_fisher_metric(np.eye(2), weights=np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="fisher damping"):
        empirical_fisher_metric(np.eye(2), damping=-1.0)


def test_empirical_fisher_vector_product_matches_materialised_metric() -> None:
    """Matrix-free Fisher products should match explicit metric multiplication."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] + values[1], 2.0 * values[1]]),
        [1.0, -2.0],
        parameters=[Parameter("x"), Parameter("y")],
    )
    weights = np.array([0.5, 2.0])
    tangent = np.array([3.0, -1.0])
    result = empirical_fisher_vector_product(
        jacobian_result,
        tangent,
        weights=weights,
        damping=0.25,
    )
    metric = empirical_fisher_metric(jacobian_result, weights=weights, damping=0.25)

    assert isinstance(result, FisherVectorProductResult)
    assert result.method == "fisher_vector_product:finite_difference_central"
    np.testing.assert_allclose(result.residual_projection, [2.0, -2.0], atol=1.0e-6)
    np.testing.assert_allclose(result.product, metric @ tangent, atol=1.0e-6)
    np.testing.assert_allclose(result.tangent, tangent)


def test_empirical_fisher_vector_product_respects_frozen_parameters() -> None:
    """Frozen parameters should be removed from Fisher-vector products."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] + 10.0 * values[1], values[1] ** 2]),
        [1.0, 2.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
    )
    result = empirical_fisher_vector_product(jacobian_result, [0.5, 100.0], damping=1.0)

    np.testing.assert_allclose(result.tangent, [0.5, 0.0])
    np.testing.assert_allclose(result.residual_projection, [0.5, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(result.product, [1.0, 0.0], atol=1.0e-6)


def test_empirical_fisher_vector_product_rejects_invalid_inputs() -> None:
    """Fisher-vector products should fail closed for malformed tangents and weights."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0], values[1]]),
        [1.0, 2.0],
    )
    with pytest.raises(ValueError, match="JacobianResult"):
        empirical_fisher_vector_product(np.eye(2), [1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tangent length"):
        empirical_fisher_vector_product(jacobian_result, [1.0])
    with pytest.raises(ValueError, match="weights"):
        empirical_fisher_vector_product(jacobian_result, [1.0, 2.0], weights=[1.0])
    with pytest.raises(ValueError, match="non-negative"):
        empirical_fisher_vector_product(jacobian_result, [1.0, 2.0], weights=[1.0, -1.0])
    with pytest.raises(ValueError, match="damping"):
        empirical_fisher_vector_product(jacobian_result, [1.0, 2.0], damping=-1.0)


def test_empirical_fisher_conjugate_gradient_matches_direct_solve() -> None:
    """Matrix-free Fisher CG should solve the same system as a direct metric solve."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] + values[1], 2.0 * values[1]]),
        [1.0, -2.0],
        parameters=[Parameter("x"), Parameter("y")],
    )
    weights = np.array([0.5, 2.0])
    rhs = np.array([1.0, -3.0])
    result = empirical_fisher_conjugate_gradient(
        jacobian_result,
        rhs,
        weights=weights,
        damping=0.25,
        tolerance=1.0e-12,
        max_iterations=10,
    )
    metric = empirical_fisher_metric(jacobian_result, weights=weights, damping=0.25)

    assert isinstance(result, FisherConjugateGradientResult)
    assert result.converged
    assert result.iterations <= 2
    assert result.residual_norm_history[-1] <= 1.0e-10
    np.testing.assert_allclose(result.solution, np.linalg.solve(metric, rhs), atol=1.0e-8)


def test_empirical_fisher_conjugate_gradient_respects_frozen_parameters() -> None:
    """Frozen parameters should receive zero matrix-free Fisher-CG solution components."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] + 10.0 * values[1], values[1] ** 2]),
        [1.0, 2.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
    )
    result = empirical_fisher_conjugate_gradient(
        jacobian_result,
        [2.0, 999.0],
        damping=1.0,
    )

    assert result.converged
    np.testing.assert_allclose(result.solution, [1.0, 0.0], atol=1.0e-8)


def test_empirical_fisher_conjugate_gradient_rejects_invalid_inputs() -> None:
    """Fisher CG must fail closed on malformed controls and indefinite systems."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0], values[1]]),
        [1.0, 2.0],
    )
    with pytest.raises(ValueError, match="JacobianResult"):
        empirical_fisher_conjugate_gradient(np.eye(2), [1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rhs length"):
        empirical_fisher_conjugate_gradient(jacobian_result, [1.0])
    with pytest.raises(ValueError, match="damping"):
        empirical_fisher_conjugate_gradient(jacobian_result, [1.0, 2.0], damping=-1.0)
    with pytest.raises(ValueError, match="tolerance"):
        empirical_fisher_conjugate_gradient(jacobian_result, [1.0, 2.0], tolerance=-1.0)
    with pytest.raises(ValueError, match="max_iterations"):
        empirical_fisher_conjugate_gradient(jacobian_result, [1.0, 2.0], max_iterations=0)
    singular = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] + values[1]]),
        [1.0, 2.0],
        parameters=[Parameter("x"), Parameter("y")],
    )
    with pytest.raises(ValueError, match="positive definite"):
        empirical_fisher_conjugate_gradient(singular, [1.0, -1.0], damping=0.0)


def test_gauss_newton_gradient_solves_weighted_least_squares_residual() -> None:
    """Gauss-Newton should solve the weighted residual normal equations."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] - 1.0, 2.0 * (values[1] + 0.5)]),
        [3.0, 1.5],
        parameters=[Parameter("x"), Parameter("y")],
    )
    result = gauss_newton_gradient(
        jacobian_result,
        weights=np.array([1.0, 0.25]),
        damping=0.5,
    )

    assert isinstance(result, NaturalGradientResult)
    assert result.base_gradient.value == pytest.approx(4.0)
    np.testing.assert_allclose(result.base_gradient.gradient, [2.0, 2.0], atol=1.0e-6)
    np.testing.assert_allclose(
        result.natural_gradient,
        [4.0 / 3.0, 4.0 / 3.0],
        atol=1.0e-6,
    )
    assert result.base_gradient.parameter_names == ("x", "y")


def test_gauss_newton_gradient_respects_frozen_parameters() -> None:
    """Frozen parameters must not receive Gauss-Newton update components."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] - 1.0, values[1] - 2.0]),
        [2.0, 5.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
    )
    result = gauss_newton_gradient(jacobian_result, damping=0.25)

    np.testing.assert_allclose(result.natural_gradient, [0.8, 0.0], atol=1.0e-6)


def test_gauss_newton_gradient_rejects_invalid_inputs() -> None:
    """Gauss-Newton diagnostics require a validated JacobianResult and weights."""

    with pytest.raises(ValueError, match="JacobianResult"):
        gauss_newton_gradient(np.eye(2))  # type: ignore[arg-type]
    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0], values[1]]),
        [1.0, 2.0],
    )
    with pytest.raises(ValueError, match="weights"):
        gauss_newton_gradient(jacobian_result, weights=np.array([1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        gauss_newton_gradient(jacobian_result, weights=np.array([1.0, -1.0]))


def test_least_squares_covariance_estimates_fisher_uncertainty() -> None:
    """Residual-map covariance should invert the trainable empirical Fisher metric."""

    def objective(values: np.ndarray) -> np.ndarray:
        return np.array([values[0], 2.0 * values[1], values[0] + values[1]])

    jacobian_result = value_and_finite_difference_jacobian(
        objective,
        [1.0, 1.0],
        parameters=[Parameter("x"), Parameter("y")],
    )
    result = least_squares_covariance(jacobian_result)

    assert isinstance(result, LeastSquaresCovarianceResult)
    assert result.degrees_of_freedom == 1
    assert result.residual_variance == pytest.approx(9.0)
    assert result.condition_number > 1.0
    np.testing.assert_allclose(result.covariance, [[5.0, -1.0], [-1.0, 2.0]], atol=1.0e-5)
    np.testing.assert_allclose(result.standard_errors, [np.sqrt(5.0), np.sqrt(2.0)], atol=1.0e-5)


def test_least_squares_covariance_respects_trainable_mask_and_variance() -> None:
    """Non-trainable parameters should receive zero covariance and standard error."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0], values[1], values[0] + values[1]]),
        [1.0, 2.0],
        parameters=[Parameter("x"), Parameter("y", trainable=False)],
    )
    result = least_squares_covariance(
        jacobian_result,
        weights=np.array([1.0, 0.25, 1.0]),
        residual_variance=0.5,
    )

    np.testing.assert_allclose(result.covariance, [[0.25, 0.0], [0.0, 0.0]], atol=1.0e-6)
    np.testing.assert_allclose(result.standard_errors, [0.5, 0.0], atol=1.0e-6)
    assert result.parameter_names == ("x", "y")
    assert result.trainable == (True, False)


def test_least_squares_covariance_rejects_invalid_inputs() -> None:
    """Covariance estimation should fail closed for singular or malformed solves."""

    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0], 2.0 * values[0]]),
        [1.0],
        parameters=[Parameter("x")],
    )
    with pytest.raises(ValueError, match="JacobianResult"):
        least_squares_covariance(np.eye(1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="residual_variance"):
        least_squares_covariance(jacobian_result, residual_variance=-1.0)
    with pytest.raises(ValueError, match="weights"):
        least_squares_covariance(jacobian_result, weights=np.array([1.0]))
    with pytest.raises(ValueError, match="rcond"):
        least_squares_covariance(jacobian_result, rcond=1.0)
    singular = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] + values[1]]),
        [1.0, 2.0],
        parameters=[Parameter("x"), Parameter("y")],
    )
    with pytest.raises(ValueError, match="positive definite"):
        least_squares_covariance(singular)


def test_huber_residual_weights_downweight_outliers_for_residual_maps() -> None:
    """Huber IRLS weights should preserve inliers and bound outlier influence."""

    weights = huber_residual_weights(np.array([0.0, 0.5, -2.0, 10.0]), delta=1.0)

    np.testing.assert_allclose(weights, [1.0, 1.0, 0.5, 0.1])


def test_huber_residual_weights_support_floor_for_conditioning() -> None:
    """A positive floor prevents outlier weights from collapsing to zero."""

    weights = huber_residual_weights(np.array([1.0, 100.0]), delta=1.0, min_weight=0.05)

    np.testing.assert_allclose(weights, [1.0, 0.05])


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

    np.testing.assert_allclose(weights, [1.0, 0.2], atol=1.0e-6)
    np.testing.assert_allclose(result.base_gradient.gradient, [2.0, 20.0], atol=1.0e-6)
    assert result.condition_number > 1.0


def test_soft_l1_residual_weights_smoothly_downweight_outliers() -> None:
    """Soft-L1 weights should provide a smooth influence curve for residuals."""

    weights = soft_l1_residual_weights(np.array([0.0, 1.0, 3.0]), scale=1.0)

    np.testing.assert_allclose(weights, [1.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(10.0)])


def test_soft_l1_residual_weights_support_conditioning_floor() -> None:
    """A positive floor keeps Soft-L1 weights usable in ill-scaled residual maps."""

    weights = soft_l1_residual_weights(np.array([0.0, 100.0]), scale=1.0, min_weight=0.05)

    np.testing.assert_allclose(weights, [1.0, 0.05])


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

    def objective(values: np.ndarray) -> np.ndarray:
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
    np.testing.assert_allclose(result.step, [-4.0 / 3.0, -4.0 / 3.0], atol=1.0e-6)
    np.testing.assert_allclose(result.candidate_values, [5.0 / 3.0, 1.0 / 6.0], atol=1e-6)
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

    np.testing.assert_allclose(result.step, [-0.5, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(result.candidate_values, [2.5, 5.0], atol=1.0e-6)
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

    def objective(values: np.ndarray) -> np.ndarray:
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

    def objective(values: np.ndarray) -> np.ndarray:
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

    def objective(values: np.ndarray) -> np.ndarray:
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

    with pytest.raises(ValueError, match="LevenbergMarquardtTrial"):
        update_levenberg_marquardt_damping(step_result)  # type: ignore[arg-type]
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

    def objective(values: np.ndarray) -> np.ndarray:
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
    np.testing.assert_allclose(result.values, [1.0, -0.5], atol=1.0e-5)


def test_levenberg_marquardt_optimizer_respects_bounds_and_weights() -> None:
    """Bounded weighted LM runs should stay inside the declared parameter domain."""

    def objective(values: np.ndarray) -> np.ndarray:
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
    np.testing.assert_allclose(result.best_values, [1.0, 1.5], atol=1.0e-6)


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


def test_jax_value_and_grad_matches_quadratic_when_available() -> None:
    """Optional JAX bridge should expose real autodiff when JAX is installed."""

    if not is_jax_autodiff_available():
        with pytest.raises(ImportError, match="JAX"):
            jax_value_and_grad(lambda values: values[0] ** 2, [2.0])
        return

    value, gradient = jax_value_and_grad(lambda values: values[0] ** 2, [2.0])
    assert value == pytest.approx(4.0)
    np.testing.assert_allclose(gradient, [4.0], rtol=1e-6, atol=1e-6)


def test_jax_value_and_grad_rejects_implicit_parameter_coercion() -> None:
    """JAX bridge input validation should match the native differentiable path."""

    with pytest.raises(ValueError, match="parameters must contain real numeric scalars"):
        jax_value_and_grad(lambda values: values[0] ** 2, ["2.0"])


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
