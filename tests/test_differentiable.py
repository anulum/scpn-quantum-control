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
from collections.abc import Callable

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    ArmijoLineSearchResult,
    CustomDerivativeCheckResult,
    CustomDerivativeRegistry,
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
    PrimitiveBatchingRule,
    PrimitiveContract,
    PrimitiveDTypeRule,
    PrimitiveIdentity,
    PrimitiveLoweringRule,
    PrimitiveShapeRule,
    PrimitiveTransformRule,
    ProgramADAdjointResult,
    ProgramADAliasEdge,
    ProgramADControlRegion,
    ProgramADEffect,
    ProgramADEffectIR,
    ProgramADSSAValue,
    ReverseNode,
    ShotAllocationResult,
    SparseMatrixResult,
    StochasticGradientResult,
    TraceADArray,
    TraceADScalar,
    VJPResult,
    WeightedGradientResult,
    WholeProgramADResult,
    WholeProgramBytecodeInstruction,
    WholeProgramIRNode,
    WholeProgramSemanticsReport,
    WholeProgramSourceIRFeature,
    WholeProgramTraceEvent,
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
    custom_derivative_rule_for,
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
    jacfwd,
    jacobian,
    jacrev,
    jax_value_and_grad,
    jvp,
    least_squares_covariance,
    levenberg_marquardt_step,
    natural_gradient,
    parameter_shift_gradient,
    parameter_shift_gradient_with_uncertainty,
    primitive_complete_contract_for,
    primitive_contract_for,
    primitive_dtype_rule_for,
    primitive_effect_for,
    primitive_nondifferentiable_policy_for,
    primitive_shape_rule_for,
    program_adjoint_gradient,
    program_adjoint_result,
    register_custom_derivative_rule,
    register_primitive_batching_rule,
    register_primitive_lowering_rule,
    register_primitive_transform_rule,
    registered_custom_jacobian,
    registered_custom_jvp,
    registered_custom_vjp,
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
    value_and_jacfwd,
    value_and_jacobian,
    value_and_jacrev,
    value_and_jvp,
    value_and_parameter_shift_grad,
    value_and_reverse_mode_grad,
    value_and_vjp,
    vector_jacobian_product,
    vjp,
    vmap,
    weighted_gradient_sum,
    whole_program_grad,
    whole_program_value_and_grad,
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

    dense = np.array([[1.0, 0.0, 2.0e-8], [0.0, -3.0, 0.0]])
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
    np.testing.assert_allclose(sparse.to_dense(), [[1.0, 0.0, 0.0], [0.0, -3.0, 0.0]])


def test_sparse_jacobian_hessian_and_fisher_preserve_provenance() -> None:
    """Sparse helpers should convert derivative result objects without metadata loss."""

    jacobian_result = JacobianResult(
        value=np.array([1.0, -2.0]),
        jacobian=np.array([[1.0, 0.0], [0.0, 2.0]]),
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
    np.testing.assert_allclose(sparse_fisher.to_dense(), [[1.0, 0.0], [0.0, 4.0]])


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
    assert result.evaluations == 2
    assert result.parameter_names == ("theta", "frozen")
    assert result.trainable == (True, False)
    np.testing.assert_allclose(result.gradient, [0.3, 0.0])
    expected_variance = 0.5**2 * (0.36 / 900.0 + 0.16 / 400.0)
    np.testing.assert_allclose(result.standard_error, [math.sqrt(expected_variance), 0.0])
    np.testing.assert_allclose(result.covariance, np.diag([expected_variance, 0.0]))
    np.testing.assert_allclose(result.confidence_radius, 1.959963984540054 * result.standard_error)
    np.testing.assert_allclose(result.shots, [[900.0, 400.0], [400.0, 100.0]])


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
        gradient=np.array([2.0, 4.0, 9.0]),
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
        gradient=np.array([2.0, 3.0]),
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
        gradient=np.array([3.0, 4.0, 100.0]),
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


def test_vmap_vectorizes_single_argument_scalar_objective() -> None:
    """Canonical vmap should map scalar objectives over a selected input axis."""

    batched = vmap(lambda row: row[0] ** 2 + 3.0 * row[1])
    values = np.array([[2.0, -1.0], [0.5, 4.0], [-2.0, 3.0]], dtype=np.float64)

    np.testing.assert_allclose(batched(values), [1.0, 12.25, 13.0], atol=1.0e-12)


def test_vmap_supports_broadcast_arguments_out_axes_and_nested_outputs() -> None:
    """vmap should preserve nested output structure and broadcast static inputs."""

    def affine(row: np.ndarray, weights: np.ndarray, bias: float) -> dict[str, object]:
        projection = row * weights + bias
        return {
            "projection": projection,
            "summary": (np.array([float(np.sum(projection))], dtype=np.float64), [projection[:1]]),
        }

    batched = vmap(affine, in_axes=(0, None, None), out_axes=1)
    values = np.array([[1.0, 2.0], [3.0, -1.0]], dtype=np.float64)
    weights = np.array([2.0, -0.5], dtype=np.float64)

    result = batched(values, weights, 0.25)

    np.testing.assert_allclose(result["projection"], [[2.25, 6.25], [-0.75, 0.75]])
    np.testing.assert_allclose(result["summary"][0], [[1.5, 7.0]])
    np.testing.assert_allclose(result["summary"][1][0], [[2.25, 6.25]])


def test_vmap_composes_with_grad_transform() -> None:
    """vmap should compose with canonical gradient transforms over batches."""

    batched_grad = vmap(
        lambda row: grad(
            lambda values: values[0] ** 2 + np.sin(values[1]), row, method="finite_difference"
        )
    )
    values = np.array([[2.0, 0.0], [-1.5, 0.25]], dtype=np.float64)

    np.testing.assert_allclose(
        batched_grad(values),
        [[4.0, 1.0], [-3.0, math.cos(0.25)]],
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_vmap_rejects_invalid_axes_and_ragged_outputs() -> None:
    """vmap should fail closed on ambiguous vectorization contracts."""

    with pytest.raises(ValueError, match="same length"):
        vmap(lambda lhs, rhs: lhs + rhs)(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="at least one"):
        vmap(lambda value: value, in_axes=None)(1.0)
    with pytest.raises(ValueError, match="consistent shapes"):
        vmap(lambda value: np.ones(int(value), dtype=np.float64))(
            np.array([1.0, 2.0], dtype=np.float64)
        )


def test_vmap_is_exported_from_package_root() -> None:
    """The vectorization transform should be stable as a root-level API."""

    import scpn_quantum_control as scpn

    assert scpn.vmap is vmap


def test_whole_program_value_and_grad_traces_numpy_control_flow() -> None:
    """Whole-program AD should handle ordinary Python control flow and NumPy calls."""

    def objective(values: np.ndarray) -> object:
        total = values[0] * 0.0
        for index, value in enumerate(values):
            if value > 0.0:
                total = total + np.sin(value) + index * value
            else:
                total = total + value**2
        return total

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, -0.5, 0.75], dtype=np.float64),
        parameters=(Parameter("theta"), Parameter("bias"), Parameter("phase")),
    )

    assert result.method == "whole_program_ad"
    assert result.control_flow_observed is True
    assert result.numpy_observed is True
    assert result.polyglot_targets["python"].startswith("operator-intercepted")
    assert result.polyglot_targets["rust"].startswith("blocked")
    assert result.polyglot_targets["llvm"].startswith("blocked")
    assert len(result.trace_events) >= 4
    np.testing.assert_allclose(
        result.gradient,
        [math.cos(0.25), -1.0, math.cos(0.75) + 2.0],
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_whole_program_ad_captures_bytecode_source_alias_mutation_and_loop_semantics() -> None:
    """Arbitrary whole-program AD should expose frontend IR and semantic analysis."""

    def objective(values: np.ndarray) -> object:
        history = [values[0]]
        alias = history
        total = values[0] * 0.0
        for item in range(3):
            alias.append(values[1] * item)
            total = total + history[item]
        if total > 0.0:
            return np.sin(values[0]) + total
        return np.cos(values[1]) - total

    result = whole_program_value_and_grad(
        objective,
        np.array([0.5, 0.25], dtype=np.float64),
        parameters=(Parameter("theta"), Parameter("phi")),
    )

    assert result.semantics_report is not None
    assert result.semantics_report.bytecode_frontend is True
    assert result.semantics_report.source_frontend is True
    assert result.semantics_report.graph_capture is True
    assert result.semantics_report.aliasing_observed is True
    assert result.semantics_report.mutation_observed is True
    assert result.semantics_report.loop_observed is True
    assert result.semantics_report.control_flow_observed is True
    assert result.semantics_report.numpy_observed is True
    assert result.bytecode_instructions
    assert any(instruction.opname == "FOR_ITER" for instruction in result.bytecode_instructions)
    assert {feature.kind for feature in result.source_ir_features} >= {
        "alias_analysis",
        "control_flow",
        "loop",
        "mutation",
        "numpy",
    }
    assert any(node.op.startswith("branch:") for node in result.ir_nodes)
    np.testing.assert_allclose(result.gradient, [math.cos(0.5) + 1.0, 1.0], atol=1.0e-12)


def test_whole_program_ad_handles_vector_numpy_reductions_dot_and_array_mutation() -> None:
    """Whole-program AD should execute vector NumPy semantics with derivative-carrying arrays."""

    def objective(values: np.ndarray) -> object:
        working = values.copy()
        working[1] = working[1] + values[0] * 2.0
        vector_term = np.sum(np.sin(working) + working**2)
        mean_term = np.mean(working)
        dot_term = np.dot(working, np.array([1.0, -2.0, 0.5], dtype=np.float64))
        return vector_term + mean_term + dot_term

    result = whole_program_value_and_grad(
        objective,
        np.array([0.2, -0.4, 0.7], dtype=np.float64),
        parameters=(Parameter("x"), Parameter("y"), Parameter("z")),
    )

    working = np.array([0.2, 0.0, 0.7], dtype=np.float64)
    base_grad = np.cos(working) + 2.0 * working + np.array([1.0, -2.0, 0.5]) + (1.0 / 3.0)
    expected = np.array(
        [base_grad[0] + 2.0 * base_grad[1], base_grad[1], base_grad[2]],
        dtype=np.float64,
    )
    assert result.method == "whole_program_ad"
    assert any(node.op == "mutation:setitem" for node in result.ir_nodes)
    assert any(node.op == "sin" for node in result.ir_nodes)
    np.testing.assert_allclose(result.gradient, expected, atol=1.0e-12)


def test_whole_program_ad_handles_piecewise_vector_numpy_semantics() -> None:
    """Whole-program AD should differentiate executed vector piecewise NumPy branches."""

    def objective(values: np.ndarray) -> object:
        shifted = values + np.array([2.0, 3.0, 4.0], dtype=np.float64)
        smooth = np.sqrt(shifted) + np.tanh(values) + np.square(values)
        piecewise = np.where(values > 0.0, smooth, np.absolute(values - 1.0))
        clipped = np.maximum(piecewise, values + 0.5)
        return np.sum(np.minimum(clipped, piecewise + 2.0))

    values = np.array([0.25, -0.5, 1.2], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    )

    expected = np.array(
        [
            0.5 / math.sqrt(2.25) + (1.0 - math.tanh(0.25) ** 2) + 0.5,
            -1.0,
            0.5 / math.sqrt(5.2) + (1.0 - math.tanh(1.2) ** 2) + 2.4,
        ],
        dtype=np.float64,
    )
    assert any(node.op == "where" for node in result.ir_nodes)
    assert any(node.op == "maximum" for node in result.ir_nodes)
    assert any(node.op == "minimum" for node in result.ir_nodes)
    np.testing.assert_allclose(result.gradient, expected, atol=1.0e-12)


def test_whole_program_ad_handles_matrix_indexing_reductions_and_products() -> None:
    """Program AD should cover rank-2 array control, mutation, and products."""

    def objective(values: np.ndarray) -> object:
        matrix = values.reshape(2, 2).copy()
        matrix[0, 1] = matrix[0, 1] + matrix[1, 0]
        column_sum = np.sum(matrix, axis=0)
        row_sum = np.sum(matrix, axis=1)
        matrix_vector = matrix @ np.array([2.0, -1.0], dtype=np.float64)
        vector_matrix = np.array([1.5, -0.5], dtype=np.float64) @ matrix
        return (
            np.sum(column_sum)
            + np.sum(row_sum)
            + np.sum(matrix_vector)
            + np.sum(vector_matrix)
            + np.sum(matrix.T)
        )

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c"), Parameter("d")),
    )

    assert any(node.op == "mutation:setitem" for node in result.ir_nodes)
    assert result.semantics_report is not None
    assert result.semantics_report.mutation_observed is True
    np.testing.assert_allclose(result.gradient, [6.5, 3.5, 8.0, 1.5], atol=1.0e-12)


def test_whole_program_ad_handles_numpy_composition_and_norms() -> None:
    """Program AD should cover common NumPy shape composition and norm workflows."""

    def objective(values: np.ndarray) -> object:
        left = values[:2]
        right = values[2:4]
        stacked = np.stack((left, right), axis=0)
        flat = np.concatenate((stacked[0], stacked[1]))
        reshaped = np.reshape(flat, (2, 2))
        transposed = np.transpose(reshaped)
        clipped = np.clip(transposed, -0.25, 1.5)
        return np.linalg.norm(clipped) + np.sum(np.ravel(transposed))

    values = np.array([0.5, -0.1, 2.0, -2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("a"), Parameter("b"), Parameter("c"), Parameter("d")),
    )

    norm = math.sqrt(0.5**2 + (-0.1) ** 2 + 1.5**2 + (-0.25) ** 2)
    expected = np.array([1.0 + 0.5 / norm, 1.0 - 0.1 / norm, 1.0, 1.0], dtype=np.float64)
    assert any(node.op == "clip" for node in result.ir_nodes)
    assert any(node.op == "sqrt" for node in result.ir_nodes)
    np.testing.assert_allclose(result.gradient, expected, atol=1.0e-12)


def test_whole_program_ad_handles_numpy_linear_algebra_primitives() -> None:
    """Program AD should cover bounded NumPy linear algebra forms exactly."""

    def objective(values: np.ndarray) -> object:
        left = values[:2]
        right = values[2:4]
        matrix = np.reshape(values, (2, 2))
        return (
            np.inner(left, right)
            + np.sum(np.outer(left, right))
            + np.trace(matrix)
            + np.sum(np.diag(matrix))
            + np.tensordot(left, right, axes=1)
            + np.sum(np.tensordot(left, right, axes=0))
            + np.einsum("i,i->", left, right)
            + np.sum(np.einsum("i,j->ij", left, right))
            + np.sum(np.einsum("ij,j->i", matrix, left))
            + np.einsum("ii->", matrix)
        )

    values = np.array([0.5, -0.25, 1.5, -2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("a"), Parameter("b"), Parameter("c"), Parameter("d")),
    )
    expected = np.array(
        [
            7.0 * values[2] + 3.0 * values[3] + 2.0 * values[0] + 3.0,
            3.0 * values[2] + 7.0 * values[3] + 2.0 * values[1],
            7.0 * values[0] + 3.0 * values[1],
            3.0 * values[0] + 7.0 * values[1] + 3.0,
        ],
        dtype=np.float64,
    )

    assert result.adjoint_result is not None
    assert result.adjoint_result.supported is True
    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_reciprocal_ufunc_matches_exact_adjoint() -> None:
    """Program AD reciprocal should preserve exact inverse derivatives and adjoints."""

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (2, 2))
        reciprocal = np.reciprocal(values)
        return np.sum(reciprocal * np.array([1.0, -2.0, 3.0, -4.0])) + matrix[0, 1] ** -1

    values = np.array([2.0, -4.0, 0.5, -0.25], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x0"), Parameter("x1"), Parameter("x2"), Parameter("x3")),
    )
    expected = np.array([-0.25, 0.0625, -12.0, 64.0], dtype=np.float64)

    assert result.value == pytest.approx(22.75)
    assert any(node.op == "reciprocal" for node in result.ir_nodes)
    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_reciprocal_ufunc_fails_closed_at_zero() -> None:
    """Program AD reciprocal should reject singular inverse boundaries."""

    with pytest.raises(ValueError, match="reciprocal input must be non-zero"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.reciprocal(values)),
            np.array([1.0, 0.0], dtype=np.float64),
        )


def test_program_ad_log1p_and_expm1_ufuncs_match_exact_adjoint() -> None:
    """Program AD should preserve stable log1p/expm1 derivatives and adjoints."""

    def objective(values: np.ndarray) -> object:
        transformed = np.log1p(values[:2]) + np.expm1(values[1:3])
        return np.sum(transformed * np.array([2.0, -3.0])) + np.log1p(values[2])

    values = np.array([0.5, -0.2, 1.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x0"), Parameter("x1"), Parameter("x2")),
    )
    expected = np.array(
        [
            2.0 / (1.0 + values[0]),
            2.0 * math.exp(values[1]) - 3.0 / (1.0 + values[1]),
            -3.0 * math.exp(values[2]) + 1.0 / (1.0 + values[2]),
        ],
        dtype=np.float64,
    )

    assert any(node.op == "log1p" for node in result.ir_nodes)
    assert any(node.op == "expm1" for node in result.ir_nodes)
    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_log1p_ufunc_fails_closed_at_domain_boundary() -> None:
    """Program AD log1p should reject inputs where the derivative is singular."""

    with pytest.raises(ValueError, match="log1p input must be greater than -1"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.log1p(values)),
            np.array([0.0, -1.0], dtype=np.float64),
        )


def test_program_ad_tan_ufunc_matches_exact_adjoint() -> None:
    """Program AD tangent should preserve exact trigonometric derivatives and adjoints."""

    def objective(values: np.ndarray) -> object:
        angles = values[:2]
        return np.sum(np.tan(angles) * np.array([2.0, -3.0])) + values[2] * np.tan(values[0])

    values = np.array([0.25, -0.4, 1.5], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("theta0"), Parameter("theta1"), Parameter("gain")),
    )
    expected = np.array(
        [
            (2.0 + values[2]) / math.cos(values[0]) ** 2,
            -3.0 / math.cos(values[1]) ** 2,
            math.tan(values[0]),
        ],
        dtype=np.float64,
    )

    assert any(node.op == "tan" for node in result.ir_nodes)
    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_tan_ufunc_fails_closed_at_singular_boundary() -> None:
    """Program AD tangent should reject singular cosine-zero boundaries."""

    with pytest.raises(ValueError, match="tan input must have non-zero cosine"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.tan(values)),
            np.array([math.pi / 2.0], dtype=np.float64),
        )


def test_program_ad_arcsin_arccos_ufuncs_match_exact_adjoint() -> None:
    """Program AD inverse trig ufuncs should preserve exact branch-local adjoints."""

    def objective(values: np.ndarray) -> object:
        return (
            2.0 * np.arcsin(values[0])
            - 3.0 * np.arccos(values[1])
            + values[2] * np.arcsin(values[1])
        )

    values = np.array([0.25, -0.4, 1.5], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x0"), Parameter("x1"), Parameter("gain")),
    )
    expected = np.array(
        [
            2.0 / math.sqrt(1.0 - values[0] ** 2),
            (3.0 + values[2]) / math.sqrt(1.0 - values[1] ** 2),
            math.asin(values[1]),
        ],
        dtype=np.float64,
    )

    assert any(node.op == "arcsin" for node in result.ir_nodes)
    assert any(node.op == "arccos" for node in result.ir_nodes)
    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_arcsin_arccos_fail_closed_at_domain_boundaries() -> None:
    """Program AD inverse trig ufuncs should reject singular or invalid domains."""

    for ufunc in (np.arcsin, np.arccos):
        with pytest.raises(ValueError, match="input must be strictly inside"):
            whole_program_value_and_grad(
                lambda values, fn=ufunc: np.sum(fn(values)),
                np.array([1.0], dtype=np.float64),
            )
        with pytest.raises(ValueError, match="input must be strictly inside"):
            whole_program_value_and_grad(
                lambda values, fn=ufunc: np.sum(fn(values)),
                np.array([1.25], dtype=np.float64),
            )


def test_whole_program_ad_numpy_linear_algebra_fail_closed_paths() -> None:
    """Unsupported NumPy linear algebra modes should fail closed with explicit diagnostics."""

    with pytest.raises(ValueError, match="einsum supports explicit"):
        whole_program_value_and_grad(
            lambda values: np.einsum("i->", values),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="tensordot supports axes"):
        whole_program_value_and_grad(
            lambda values: np.tensordot(values, values, axes=2),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="sort/argsort selection semantics"):
        whole_program_value_and_grad(
            lambda values: np.sort(values)[0],
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_whole_program_ad_handles_numpy_broadcasting_semantics() -> None:
    """Program AD should follow NumPy broadcasting for ufuncs, predicates, and where."""

    def objective(values: np.ndarray) -> object:
        column = np.reshape(values[:2], (2, 1))
        row = values[2:5]
        broadcast_product = column * row
        shifted = broadcast_product + values[5]
        selected = np.where(shifted > 0.0, shifted, -shifted)
        return np.sum(selected) + np.sum(row / (column + 3.0))

    values = np.array([0.5, 1.25, 0.2, -0.4, 0.75, 0.3], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(
            Parameter("c0"),
            Parameter("c1"),
            Parameter("r0"),
            Parameter("r1"),
            Parameter("r2"),
            Parameter("bias"),
        ),
    )

    column = values[:2].reshape(2, 1)
    row = values[2:5]
    shifted = column * row + values[5]
    signs = np.where(shifted > 0.0, 1.0, -1.0)
    expected = np.zeros(6, dtype=np.float64)
    expected[0:2] = np.sum(signs * row, axis=1) - np.sum(row) / (column[:, 0] + 3.0) ** 2
    expected[2:5] = np.sum(signs * column, axis=0) + np.sum(1.0 / (column[:, 0] + 3.0))
    expected[5] = np.sum(signs)

    assert result.adjoint_result is not None
    assert result.adjoint_result.supported is True
    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_whole_program_ad_broadcasting_rejects_incompatible_shapes() -> None:
    """Program AD broadcasting should fail closed on incompatible NumPy shapes."""

    with pytest.raises(ValueError, match="broadcasting rules"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.reshape(values[:4], (2, 2)) + values[4:7]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64),
        )


def test_whole_program_ad_handles_rank_n_reductions_and_transpose() -> None:
    """Program AD should support rank-N reductions and explicit transpose axes."""

    weights_axis0 = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    weights_axis1 = np.array([[2.0, -1.0], [0.25, 1.5]], dtype=np.float64)
    weights_transpose = np.array(
        [[[0.5, -0.25], [1.0, 2.0]], [[-1.5, 0.75], [0.0, 1.25]]],
        dtype=np.float64,
    )

    def objective(values: np.ndarray) -> object:
        tensor = np.reshape(values, (2, 2, 2))
        axis0 = np.sum(tensor, axis=0)
        axis1 = np.mean(tensor, axis=1)
        transposed = np.transpose(tensor, axes=(2, 0, 1))
        reversed_axes = tensor.T
        return (
            np.sum(axis0 * weights_axis0)
            + np.sum(axis1 * weights_axis1)
            + np.sum(transposed * weights_transpose)
            + np.sum(reversed_axes)
        )

    values = np.linspace(-0.4, 0.9, 8, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(8)),
    )
    expected_tensor = np.zeros((2, 2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected_tensor[i, j, k] = (
                    weights_axis0[j, k]
                    + 0.5 * weights_axis1[i, k]
                    + weights_transpose[k, i, j]
                    + 1.0
                )

    assert result.adjoint_result is not None
    assert result.adjoint_result.supported is True
    np.testing.assert_allclose(
        result.gradient, expected_tensor.reshape(-1), rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        program_adjoint_gradient(result),
        expected_tensor.reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_whole_program_ad_rank_n_axis_validation_paths() -> None:
    """Rank-N program AD array operations should reject invalid axes explicitly."""

    with pytest.raises(ValueError, match="axis"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.reshape(values, (2, 2)), axis=3),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="axes must match"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.transpose(np.reshape(values, (2, 2, 1)), axes=(0, 1))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="axes must be a permutation"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.transpose(np.reshape(values, (2, 2)), axes=(0, 0))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )


def test_program_ad_squeeze_expand_dims_preserve_exact_adjoint() -> None:
    """Program AD shape-only transforms should preserve exact element adjoints."""

    def objective(values: np.ndarray) -> object:
        tensor = np.reshape(values, (1, 2, 1, 3))
        squeezed = np.squeeze(tensor, axis=(0, 2))
        method_squeezed = tensor.squeeze()
        expanded = np.expand_dims(squeezed[1], axis=(0, 1))
        first_row = squeezed[0]
        method_expanded = (
            first_row.expand_dims(axis=1)
            if hasattr(first_row, "expand_dims")
            else np.expand_dims(first_row, axis=1)
        )
        return (
            np.sum(squeezed * np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            + np.sum(method_squeezed[0] * np.array([0.5, 1.5, 2.5]))
            + np.sum(expanded * np.array([[[7.0, 11.0, 13.0]]]))
            + np.sum(method_expanded * np.array([[17.0], [19.0], [23.0]]))
        )

    values = np.array([0.2, -0.3, 0.4, 1.1, -1.2, 1.3], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )
    expected = np.array([18.5, 22.5, 28.5, 11.0, 16.0, 19.0], dtype=np.float64)

    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_squeeze_expand_dims_fail_closed_axes() -> None:
    """Program AD shape-only transforms should reject invalid axis semantics."""

    with pytest.raises(ValueError, match="squeeze axis must have length one"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.squeeze(np.reshape(values, (2, 1)), axis=0)),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="expand_dims axes must be unique"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.expand_dims(values, axis=(0, 0))),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="expand_dims axes must be static integers"):
        whole_program_value_and_grad(
            lambda values: np.sum(values.expand_dims(axis=(True,))),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_axis_permutations_preserve_exact_adjoint() -> None:
    """Program AD rank-N axis permutations should preserve exact element adjoints."""

    weights_swap = np.arange(24.0, dtype=np.float64).reshape(4, 3, 2) / 7.0
    weights_method = np.linspace(-1.5, 2.0, 24, dtype=np.float64).reshape(2, 4, 3)
    weights_move = np.linspace(0.25, 3.25, 24, dtype=np.float64).reshape(4, 3, 2)

    def objective(values: np.ndarray) -> object:
        tensor = np.reshape(values, (2, 3, 4))
        swapped = np.swapaxes(tensor, 0, 2)
        method_swapped = tensor.swapaxes(1, 2)
        moved = np.moveaxis(tensor, source=(0, 2), destination=(2, 0))
        return (
            np.sum(swapped * weights_swap)
            + np.sum(method_swapped * weights_method)
            + np.sum(moved * weights_move)
        )

    values = np.linspace(-0.75, 1.5, 24, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )
    expected = (
        np.swapaxes(weights_swap, 0, 2)
        + np.swapaxes(weights_method, 1, 2)
        + np.moveaxis(weights_move, source=(2, 0), destination=(0, 2))
    ).reshape(-1)

    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_roll_preserves_exact_adjoint() -> None:
    """Program AD static roll permutations should preserve exact element adjoints."""

    weights_flat = np.linspace(-2.0, 1.0, 24, dtype=np.float64).reshape(2, 3, 4)
    weights_axes = np.linspace(0.25, 3.25, 24, dtype=np.float64).reshape(2, 3, 4)

    def objective(values: np.ndarray) -> object:
        tensor = np.reshape(values, (2, 3, 4))
        flat_roll = np.roll(tensor, shift=5)
        axis_roll = np.roll(tensor, shift=(1, -2), axis=(0, 2))
        return np.sum(flat_roll * weights_flat) + np.sum(axis_roll * weights_axes)

    values = np.linspace(-1.0, 1.0, 24, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )
    expected = (
        np.roll(weights_flat.reshape(-1), shift=-5).reshape(2, 3, 4)
        + np.roll(weights_axes, shift=(-1, 2), axis=(0, 2))
    ).reshape(-1)

    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_repeat_accumulates_exact_adjoint() -> None:
    """Program AD repeat should accumulate adjoints from repeated source elements."""

    flat_repeats = (1, 2, 0, 3, 1, 2)
    weights_flat = np.linspace(-2.0, 2.0, sum(flat_repeats), dtype=np.float64)
    axis_repeats = (2, 1, 3)
    weights_axis = np.linspace(0.5, 3.5, 12, dtype=np.float64).reshape(2, 6)
    weights_method = np.linspace(-1.25, 1.75, 12, dtype=np.float64).reshape(4, 3)

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (2, 3))
        flat = np.repeat(matrix, flat_repeats)
        axis_repeat = np.repeat(matrix, axis_repeats, axis=1)
        method_repeat = matrix.repeat(2, axis=0)
        return (
            np.sum(flat * weights_flat)
            + np.sum(axis_repeat * weights_axis)
            + np.sum(method_repeat * weights_method)
        )

    values = np.linspace(-0.8, 0.9, 6, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    flat_indices = np.repeat(np.arange(6, dtype=np.int64), flat_repeats)
    expected = np.zeros(6, dtype=np.float64)
    np.add.at(expected, flat_indices, weights_flat)
    axis_indices = np.repeat(np.arange(6, dtype=np.int64).reshape(2, 3), axis_repeats, axis=1)
    np.add.at(expected, axis_indices.reshape(-1), weights_axis.reshape(-1))
    method_indices = np.repeat(np.arange(6, dtype=np.int64).reshape(2, 3), 2, axis=0)
    np.add.at(expected, method_indices.reshape(-1), weights_method.reshape(-1))

    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_repeat_fails_closed_invalid_static_contracts() -> None:
    """Program AD repeat should reject invalid static repeat contracts."""

    with pytest.raises(ValueError, match="repeat counts must be static non-negative integers"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.repeat(values, True)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="repeat counts length must match selected axis"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.repeat(np.reshape(values, (2, 2)), (1, 2, 3), axis=1)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="repeat axis out of bounds"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.repeat(np.reshape(values, (2, 2)), 2, axis=2)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )


def test_program_ad_tile_accumulates_exact_adjoint() -> None:
    """Program AD tile should accumulate adjoints from every tiled source use."""

    weights_matrix = np.linspace(-2.5, 3.5, 24, dtype=np.float64).reshape(4, 6)
    weights_promoted = np.linspace(0.25, 2.25, 36, dtype=np.float64).reshape(3, 2, 6)

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (2, 3))
        tiled = np.tile(matrix, (2, 2))
        promoted = np.tile(matrix, (3, 1, 2))
        return np.sum(tiled * weights_matrix) + np.sum(promoted * weights_promoted)

    values = np.linspace(-0.6, 1.1, 6, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    source = np.arange(6, dtype=np.int64).reshape(2, 3)
    expected = np.zeros(6, dtype=np.float64)
    np.add.at(expected, np.tile(source, (2, 2)).reshape(-1), weights_matrix.reshape(-1))
    np.add.at(
        expected,
        np.tile(source.reshape(1, 2, 3), (3, 1, 2)).reshape(-1),
        weights_promoted.reshape(-1),
    )

    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_tile_fails_closed_invalid_static_contracts() -> None:
    """Program AD tile should reject dynamic or invalid repetition contracts."""

    with pytest.raises(ValueError, match="tile reps must be static non-negative integers"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.tile(values, True)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="tile reps must be static non-negative integers"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.tile(values, (2, -1))),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="tile reps must contain at least one axis"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.tile(values, ())),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )


def test_program_ad_atleast_rank_transforms_preserve_exact_adjoint() -> None:
    """Program AD atleast transforms should preserve derivatives through rank lifts."""

    vector_weights_2d = np.linspace(-1.5, 2.5, 6, dtype=np.float64).reshape(1, 6)
    vector_weights_3d = np.linspace(0.25, 2.75, 6, dtype=np.float64).reshape(1, 6, 1)
    matrix_weights = np.linspace(-2.0, 2.0, 6, dtype=np.float64).reshape(2, 3, 1)
    multi_left_weights = np.linspace(-0.75, 1.25, 6, dtype=np.float64)
    multi_right_weights = np.linspace(1.5, 3.0, 3, dtype=np.float64).reshape(1, 3)

    def objective(values: np.ndarray) -> object:
        vector = values[:6]
        matrix = np.reshape(values[:6], (2, 3))
        left, right = np.atleast_1d(vector, values[1:4])
        return (
            np.sum(np.atleast_2d(vector) * vector_weights_2d)
            + np.sum(np.atleast_3d(vector) * vector_weights_3d)
            + np.sum(np.atleast_3d(matrix) * matrix_weights)
            + np.sum(left * multi_left_weights)
            + np.sum(np.atleast_2d(right) * multi_right_weights)
        )

    values = np.linspace(-1.0, 1.0, 6, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected = np.zeros(6, dtype=np.float64)
    expected += vector_weights_2d.reshape(-1)
    expected += vector_weights_3d.reshape(-1)
    expected += matrix_weights.reshape(-1)
    expected += multi_left_weights
    expected[1:4] += multi_right_weights.reshape(-1)

    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_atleast_rank_transforms_fail_closed_invalid_contracts() -> None:
    """Program AD atleast transforms should reject non-NumPy keyword contracts."""

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.atleast_2d(values, dtype=float)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )


def test_program_ad_reshape_inferred_dimension_preserves_exact_adjoint() -> None:
    """Program AD reshape should support one inferred dimension exactly."""

    matrix_weights = np.linspace(-2.0, 2.0, 6, dtype=np.float64).reshape(2, 3)
    method_weights = np.linspace(0.5, 3.5, 6, dtype=np.float64).reshape(2, 3)
    promoted_weights = np.linspace(-1.25, 1.75, 6, dtype=np.float64).reshape(3, 2)

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (-1, 3))
        method_matrix = values.reshape(2, -1)
        promoted = np.reshape(values, (3, -1))
        return (
            np.sum(matrix * matrix_weights)
            + np.sum(method_matrix * method_weights)
            + np.sum(promoted * promoted_weights)
        )

    values = np.linspace(-1.0, 1.0, 6, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected = (
        matrix_weights.reshape(-1) + method_weights.reshape(-1) + promoted_weights.reshape(-1)
    )
    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_reshape_inferred_dimension_fails_closed_invalid_contracts() -> None:
    """Program AD reshape should reject ambiguous or size-losing inferred shapes."""

    with pytest.raises(ValueError, match="at most one inferred dimension"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.reshape(values, (-1, -1))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="inferred dimension must preserve size"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.reshape(values, (4, -1))),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="dimensions must be non-negative or -1"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.reshape(values, (2, -2))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )


def test_program_ad_vdot_flattens_operands_with_exact_adjoint() -> None:
    """Program AD vdot should apply exact flattened real inner-product semantics."""

    left_weights = np.linspace(-1.5, 2.5, 6, dtype=np.float64).reshape(2, 3)
    right_weights = np.linspace(-2.0, 1.0, 6, dtype=np.float64).reshape(2, 3)

    def objective(values: np.ndarray) -> object:
        left = np.reshape(values[:6], (2, 3))
        right = np.reshape(values[6:], (3, 2))
        return np.vdot(left, right) + np.vdot(left * left_weights, right_weights)

    values = np.linspace(-0.8, 1.2, 12, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected = np.zeros(12, dtype=np.float64)
    expected[:6] = values[6:] + left_weights.reshape(-1) * right_weights.reshape(-1)
    expected[6:] = values[:6]
    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_vdot_fails_closed_size_mismatch() -> None:
    """Program AD vdot should reject flattened size mismatches explicitly."""

    with pytest.raises(ValueError, match="vdot flattened operands must have matching size"):
        whole_program_value_and_grad(
            lambda values: np.vdot(values[:2], values[2:5]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        )


def test_program_ad_linalg_det_matches_cofactor_adjoint() -> None:
    """Program AD determinant should expose exact cofactor derivatives."""

    def objective(values: np.ndarray) -> object:
        two_by_two = np.reshape(values[:4], (2, 2))
        three_by_three = np.reshape(values[4:], (3, 3))
        return np.linalg.det(two_by_two) + 0.5 * np.linalg.det(three_by_three)

    values = np.array(
        [
            1.5,
            -0.25,
            0.75,
            2.0,
            1.0,
            0.5,
            -0.25,
            0.0,
            1.25,
            0.75,
            0.5,
            -0.5,
            1.5,
        ],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    two_by_two = values[:4].reshape(2, 2)
    three_by_three = values[4:].reshape(3, 3)
    expected = np.zeros_like(values)
    expected[:4] = np.array(
        [two_by_two[1, 1], -two_by_two[1, 0], -two_by_two[0, 1], two_by_two[0, 0]],
        dtype=np.float64,
    )
    for row in range(3):
        for col in range(3):
            minor = np.delete(np.delete(three_by_three, row, axis=0), col, axis=1)
            expected[4 + row * 3 + col] = 0.5 * ((-1.0) ** (row + col)) * np.linalg.det(minor)

    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_linalg_det_fails_closed_invalid_matrix_contracts() -> None:
    """Program AD determinant should reject non-rank-2 and non-square inputs."""

    with pytest.raises(ValueError, match="rank-2 matrices only"):
        whole_program_value_and_grad(
            lambda values: np.linalg.det(values),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires a square matrix"):
        whole_program_value_and_grad(
            lambda values: np.linalg.det(np.reshape(values, (2, 3))),
            np.arange(1.0, 7.0, dtype=np.float64),
        )


def test_program_ad_linalg_inv_matches_inverse_differential() -> None:
    """Program AD inverse should match the exact matrix inverse differential."""

    weights_two = np.array([[0.5, -1.25], [2.0, 0.75]], dtype=np.float64)
    weights_three = np.array(
        [[0.25, -0.5, 1.5], [2.0, 0.75, -1.0], [-0.25, 1.25, 0.5]],
        dtype=np.float64,
    )

    def objective(values: np.ndarray) -> object:
        two_by_two = np.reshape(values[:4], (2, 2))
        three_by_three = np.reshape(values[4:], (3, 3))
        return np.sum(np.linalg.inv(two_by_two) * weights_two) + np.sum(
            np.linalg.inv(three_by_three) * weights_three
        )

    values = np.array(
        [
            2.0,
            -0.5,
            0.75,
            1.5,
            1.5,
            0.25,
            -0.5,
            0.0,
            1.25,
            0.5,
            -0.25,
            0.75,
            1.75,
        ],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    two_by_two = values[:4].reshape(2, 2)
    three_by_three = values[4:].reshape(3, 3)
    expected = np.zeros_like(values)
    inv_two = np.linalg.inv(two_by_two)
    inv_three = np.linalg.inv(three_by_three)
    expected[:4] = (-(inv_two.T @ weights_two @ inv_two.T)).reshape(-1)
    expected[4:] = (-(inv_three.T @ weights_three @ inv_three.T)).reshape(-1)

    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_linalg_inv_fails_closed_invalid_matrix_contracts() -> None:
    """Program AD inverse should reject non-rank-2, non-square, and singular inputs."""

    with pytest.raises(ValueError, match="rank-2 matrices only"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.inv(values)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires a square matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.inv(np.reshape(values, (2, 3)))),
            np.arange(1.0, 7.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires a nonsingular matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.inv(np.reshape(values, (2, 2)))),
            np.array([1.0, 2.0, 2.0, 4.0], dtype=np.float64),
        )


def test_program_ad_linalg_solve_matches_implicit_linear_system_differential() -> None:
    """Program AD solve should match exact linear-system differential semantics."""

    vector_weights = np.array([0.5, -1.25], dtype=np.float64)
    matrix_weights = np.array([[1.0, -0.5], [0.25, 1.5], [-1.25, 0.75]], dtype=np.float64)

    def objective(values: np.ndarray) -> object:
        matrix_two = np.reshape(values[:4], (2, 2))
        rhs_two = values[4:6]
        matrix_three = np.reshape(values[6:15], (3, 3))
        rhs_three = np.reshape(values[15:], (3, 2))
        return np.sum(np.linalg.solve(matrix_two, rhs_two) * vector_weights) + np.sum(
            np.linalg.solve(matrix_three, rhs_three) * matrix_weights
        )

    values = np.array(
        [
            2.0,
            -0.5,
            0.75,
            1.5,
            0.25,
            -1.0,
            1.5,
            0.25,
            -0.5,
            0.0,
            1.25,
            0.5,
            -0.25,
            0.75,
            1.75,
            0.5,
            -1.0,
            1.25,
            0.75,
            -0.5,
            1.5,
        ],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    matrix_two = values[:4].reshape(2, 2)
    rhs_two = values[4:6]
    matrix_three = values[6:15].reshape(3, 3)
    rhs_three = values[15:].reshape(3, 2)
    solution_two = np.linalg.solve(matrix_two, rhs_two)
    solution_three = np.linalg.solve(matrix_three, rhs_three)
    expected = np.zeros_like(values)
    adjoint_rhs_two = np.linalg.solve(matrix_two.T, vector_weights)
    expected[:4] = (-np.outer(adjoint_rhs_two, solution_two)).reshape(-1)
    expected[4:6] = adjoint_rhs_two
    adjoint_rhs_three = np.linalg.solve(matrix_three.T, matrix_weights)
    expected[6:15] = (-(adjoint_rhs_three @ solution_three.T)).reshape(-1)
    expected[15:] = adjoint_rhs_three.reshape(-1)

    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_linalg_solve_fails_closed_invalid_contracts() -> None:
    """Program AD solve should reject invalid matrix and right-hand side contracts."""

    with pytest.raises(ValueError, match="matrix must be rank-2"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.solve(values[:2], values[2:4])),
            np.arange(1.0, 5.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="matrix must be square"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.solve(np.reshape(values[:6], (2, 3)), values[6:8])),
            np.arange(1.0, 9.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="vector length must match matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.solve(np.reshape(values[:4], (2, 2)), values[4:7])),
            np.arange(1.0, 8.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires a nonsingular matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.solve(np.reshape(values[:4], (2, 2)), values[4:6])),
            np.array([1.0, 2.0, 2.0, 4.0, 1.0, 0.0], dtype=np.float64),
        )


def test_program_ad_linalg_matrix_power_matches_exact_differential() -> None:
    """Program AD matrix_power should compose exact matrix products and inverses."""

    weights_square = np.array([[0.5, -1.25], [2.0, 0.75]], dtype=np.float64)
    weights_inverse = np.array([[1.0, -0.5], [0.25, 1.5]], dtype=np.float64)
    weights_identity = np.array([[0.75, -0.25], [1.25, 0.5]], dtype=np.float64)

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (2, 2))
        return (
            np.sum(np.linalg.matrix_power(matrix, 2) * weights_square)
            + np.sum(np.linalg.matrix_power(matrix, -1) * weights_inverse)
            + np.sum(np.linalg.matrix_power(matrix, 0) * weights_identity)
        )

    values = np.array([2.0, -0.5, 0.75, 1.5], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    matrix = values.reshape(2, 2)
    inv_matrix = np.linalg.inv(matrix)
    expected_matrix = (
        weights_square @ matrix.T
        + matrix.T @ weights_square
        - inv_matrix.T @ weights_inverse @ inv_matrix.T
    )
    np.testing.assert_allclose(
        result.gradient, expected_matrix.reshape(-1), rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected_matrix.reshape(-1), rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_linalg_matrix_power_fails_closed_invalid_contracts() -> None:
    """Program AD matrix_power should reject invalid matrices and powers."""

    with pytest.raises(ValueError, match="rank-2 matrices only"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.matrix_power(values, 2)),
            np.arange(1.0, 5.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires a square matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.matrix_power(np.reshape(values, (2, 3)), 2)),
            np.arange(1.0, 7.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="exponent must be a static integer"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.matrix_power(np.reshape(values, (2, 2)), 1.5)),
            np.arange(1.0, 5.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires a nonsingular matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.matrix_power(np.reshape(values, (2, 2)), -1)),
            np.array([1.0, 2.0, 2.0, 4.0], dtype=np.float64),
        )


def test_program_ad_linalg_multi_dot_matches_exact_chain_differential() -> None:
    """Program AD multi_dot should compose exact static matrix-chain semantics."""

    matrix_weights = np.array([[0.5, -1.25], [2.0, 0.75]], dtype=np.float64)
    scalar_weight = 1.75

    def objective(values: np.ndarray) -> object:
        left = np.reshape(values[:4], (2, 2))
        middle = np.reshape(values[4:8], (2, 2))
        right = np.reshape(values[8:12], (2, 2))
        vector_left = values[12:14]
        vector_right = values[14:16]
        return np.sum(np.linalg.multi_dot((left, middle, right)) * matrix_weights) + (
            scalar_weight * np.linalg.multi_dot((vector_left, middle, vector_right))
        )

    values = np.array(
        [
            1.0,
            -0.5,
            0.75,
            1.5,
            0.25,
            -1.0,
            1.25,
            0.5,
            -0.75,
            2.0,
            0.5,
            -1.5,
            1.25,
            -0.25,
            0.75,
            -1.0,
        ],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    left = values[:4].reshape(2, 2)
    middle = values[4:8].reshape(2, 2)
    right = values[8:12].reshape(2, 2)
    vector_left = values[12:14]
    vector_right = values[14:16]
    expected = np.zeros_like(values)
    expected[:4] = (matrix_weights @ right.T @ middle.T).reshape(-1)
    expected[4:8] = (
        left.T @ matrix_weights @ right.T + scalar_weight * np.outer(vector_left, vector_right)
    ).reshape(-1)
    expected[8:12] = (middle.T @ left.T @ matrix_weights).reshape(-1)
    expected[12:14] = scalar_weight * (middle @ vector_right)
    expected[14:16] = scalar_weight * (vector_left @ middle)

    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_linalg_multi_dot_fails_closed_invalid_contracts() -> None:
    """Program AD multi_dot should reject dynamic or invalid matrix-chain contracts."""

    with pytest.raises(ValueError, match="requires at least two operands"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.multi_dot((values,))),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="supports rank-1 and rank-2 operands"):
        whole_program_value_and_grad(
            lambda values: np.sum(
                np.linalg.multi_dot((np.reshape(values[:8], (2, 2, 2)), values[8:10]))
            ),
            np.arange(1.0, 11.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="middle operands must be rank-2"):
        whole_program_value_and_grad(
            lambda values: np.sum(
                np.linalg.multi_dot((np.reshape(values[:4], (2, 2)), values[4:6], values[6:8]))
            ),
            np.arange(1.0, 9.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="dimensions must align"):
        whole_program_value_and_grad(
            lambda values: np.sum(
                np.linalg.multi_dot(
                    (np.reshape(values[:4], (2, 2)), np.reshape(values[4:], (3, 2)))
                )
            ),
            np.arange(1.0, 11.0, dtype=np.float64),
        )


def test_program_ad_linalg_spectral_operations_fail_closed_policy_boundary() -> None:
    """Program AD spectral linalg should require explicit primitive policies."""

    values = np.array([2.0, -0.5, -0.5, 1.5], dtype=np.float64)
    spectral_objectives = (
        lambda matrix: np.sum(np.linalg.eig(matrix)[0]),
        lambda matrix: np.sum(np.linalg.eigh(matrix)[0]),
        lambda matrix: np.sum(np.linalg.eigvals(matrix)),
        lambda matrix: np.sum(np.linalg.eigvalsh(matrix)),
        lambda matrix: np.sum(np.linalg.svd(matrix, compute_uv=False)),
        lambda matrix: np.sum(np.linalg.pinv(matrix)),
    )

    for objective in spectral_objectives:
        with pytest.raises(ValueError, match="spectral semantics require an explicit"):
            whole_program_value_and_grad(
                lambda flat_values, objective=objective: objective(
                    np.reshape(flat_values, (2, 2))
                ),
                values,
            )


def test_program_ad_linalg_primitives_are_registry_policy_gated() -> None:
    """Supported program AD linalg primitives should expose registry contracts."""

    for name in ("det", "inv", "solve", "matrix_power", "multi_dot"):
        identity = PrimitiveIdentity("scpn.program_ad.linalg", name, "1")
        contract = primitive_contract_for(identity)
        assert contract.identity == identity
        assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
        assert contract.effect == "pure"
        assert contract.lowering_metadata["program_ad"] == "operator_intercepted_trace"
        assert contract.lowering_metadata["mlir"] == "blocked_until_executable_linalg_lowering"
        assert contract.batching_rule is not None
        assert contract.dtype_rule is not None
        assert contract.shape_rule is not None
        with pytest.raises(ValueError, match="incomplete primitive contract"):
            primitive_complete_contract_for(identity)

    values = np.array([2.0, -0.5, 0.75, 1.5], dtype=np.float64)
    result = whole_program_value_and_grad(
        lambda flat_values: np.linalg.det(np.reshape(flat_values, (2, 2))),
        values,
    )
    np.testing.assert_allclose(
        result.gradient,
        np.array([1.5, -0.75, 0.5, 2.0], dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_program_ad_linalg_primitive_shape_dtype_rules_are_specialized() -> None:
    """Supported linalg primitive contracts should expose concrete shape and dtype rules."""

    matrix = np.array([[2.0, -0.5], [0.75, 1.5]], dtype=np.float64)
    rhs_vector = np.array([1.0, -0.25], dtype=np.float64)
    rhs_matrix = np.array([[1.0, -0.25], [0.5, 1.5]], dtype=np.float64)
    right = np.array([[0.25, 1.0, -0.5], [1.5, -0.25, 0.75]], dtype=np.float64)

    contracts = {
        name: primitive_contract_for(PrimitiveIdentity("scpn.program_ad.linalg", name, "1"))
        for name in ("det", "inv", "solve", "matrix_power", "multi_dot")
    }
    assert contracts["det"].shape_rule is not None
    assert contracts["inv"].shape_rule is not None
    assert contracts["solve"].shape_rule is not None
    assert contracts["matrix_power"].shape_rule is not None
    assert contracts["multi_dot"].shape_rule is not None
    assert contracts["det"].dtype_rule is not None

    assert contracts["det"].shape_rule((matrix,)) == ()
    assert contracts["inv"].shape_rule((matrix,)) == (2, 2)
    assert contracts["solve"].shape_rule((matrix, rhs_vector)) == (2,)
    assert contracts["solve"].shape_rule((matrix, rhs_matrix)) == (2, 2)
    assert contracts["matrix_power"].shape_rule((matrix, 3)) == (2, 2)
    assert contracts["multi_dot"].shape_rule(((rhs_vector, matrix, right),)) == (3,)
    assert contracts["det"].dtype_rule((matrix,)) == "float64"

    with pytest.raises(ValueError, match="requires a square matrix"):
        contracts["det"].shape_rule((np.reshape(np.arange(6.0), (2, 3)),))
    with pytest.raises(ValueError, match="vector length must match matrix"):
        contracts["solve"].shape_rule((matrix, np.arange(3.0)))
    with pytest.raises(ValueError, match="static integer power"):
        contracts["matrix_power"].shape_rule((matrix, 1.5))
    with pytest.raises(ValueError, match="dimensions must align"):
        contracts["multi_dot"].shape_rule(((rhs_vector, np.ones((3, 2), dtype=np.float64)),))


def test_program_ad_linalg_primitive_derivative_rules_are_direct_kernels() -> None:
    """Feasible linalg primitive contracts should expose direct derivative kernels."""

    matrix = np.array([[2.0, -0.5], [0.75, 1.5]], dtype=np.float64)
    tangent_matrix = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float64)
    rhs = np.array([0.25, -1.0], dtype=np.float64)
    tangent_rhs = np.array([0.5, -0.25], dtype=np.float64)

    det_rule = custom_derivative_rule_for(PrimitiveIdentity("scpn.program_ad.linalg", "det", "1"))
    inv_rule = custom_derivative_rule_for(PrimitiveIdentity("scpn.program_ad.linalg", "inv", "1"))
    solve_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "solve", "1")
    )

    assert det_rule.name == "program_ad_linalg_det_direct_rule"
    assert inv_rule.name == "program_ad_linalg_inv_direct_rule"
    assert solve_rule.name == "program_ad_linalg_solve_direct_rule"
    assert det_rule.jvp_rule is not None
    assert inv_rule.jvp_rule is not None
    assert solve_rule.jvp_rule is not None

    np.testing.assert_allclose(det_rule.value_fn(matrix.reshape(-1)), [np.linalg.det(matrix)])
    cofactor = np.array([[matrix[1, 1], -matrix[1, 0]], [-matrix[0, 1], matrix[0, 0]]])
    np.testing.assert_allclose(
        det_rule.jvp_rule(matrix.reshape(-1), tangent_matrix.reshape(-1)),
        [np.sum(cofactor * tangent_matrix)],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    inverse = np.linalg.inv(matrix)
    np.testing.assert_allclose(inv_rule.value_fn(matrix.reshape(-1)), inverse.reshape(-1))
    np.testing.assert_allclose(
        inv_rule.jvp_rule(matrix.reshape(-1), tangent_matrix.reshape(-1)),
        (-(inverse @ tangent_matrix @ inverse)).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    solve_values = np.concatenate((matrix.reshape(-1), rhs))
    solve_tangent = np.concatenate((tangent_matrix.reshape(-1), tangent_rhs))
    solution = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(solve_rule.value_fn(solve_values), solution)
    np.testing.assert_allclose(
        solve_rule.jvp_rule(solve_values, solve_tangent),
        np.linalg.solve(matrix, tangent_rhs - tangent_matrix @ solution),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    matrix_power_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "matrix_power", "1")
    )
    with pytest.raises(ValueError, match="operator-intercepted trace dispatch"):
        matrix_power_rule.value_fn(matrix.reshape(-1))


def test_program_ad_linalg_primitive_batching_rules_vectorize_outputs() -> None:
    """Registered linalg batching rules should vectorize pure NumPy primitive calls."""

    det_contract = primitive_contract_for(PrimitiveIdentity("scpn.program_ad.linalg", "det", "1"))
    solve_contract = primitive_contract_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "solve", "1")
    )
    assert det_contract.batching_rule is not None
    assert solve_contract.batching_rule is not None

    matrices = np.array(
        [
            [[2.0, -0.5], [0.75, 1.5]],
            [[1.25, 0.25], [-0.5, 2.0]],
        ],
        dtype=np.float64,
    )
    rhs = np.array([[1.0, -0.25], [0.5, 1.5]], dtype=np.float64)

    det_values = det_contract.batching_rule(np.linalg.det, (matrices,), (0,), 0)
    solve_values = solve_contract.batching_rule(np.linalg.solve, (matrices, rhs), (0, 0), 0)

    np.testing.assert_allclose(det_values, np.linalg.det(matrices), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        solve_values,
        np.stack([np.linalg.solve(matrices[index], rhs[index]) for index in range(2)]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_program_ad_rot90_preserves_exact_adjoint() -> None:
    """Program AD rot90 permutations should preserve exact element adjoints."""

    weights_default = np.linspace(-2.0, 1.0, 12, dtype=np.float64).reshape(4, 3)
    weights_axes = np.linspace(0.5, 3.5, 24, dtype=np.float64).reshape(2, 3, 4)
    weights_negative = np.linspace(-1.5, 2.5, 24, dtype=np.float64).reshape(2, 4, 3)

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values[:12], (3, 4))
        tensor = np.reshape(values, (2, 3, 4))
        return (
            np.sum(np.rot90(matrix) * weights_default)
            + np.sum(np.rot90(tensor, k=2, axes=(0, 1)) * weights_axes)
            + np.sum(np.rot90(tensor, k=-1, axes=(1, 2)) * weights_negative)
        )

    values = np.linspace(-1.0, 1.0, 24, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )
    expected = np.zeros((2, 3, 4), dtype=np.float64)
    expected.reshape(-1)[:12] += np.rot90(weights_default, k=-1).reshape(-1)
    expected += np.rot90(weights_axes, k=-2, axes=(0, 1))
    expected += np.rot90(weights_negative, k=1, axes=(1, 2))

    np.testing.assert_allclose(result.gradient, expected.reshape(-1), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected.reshape(-1), rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_rot90_fails_closed_invalid_static_contracts() -> None:
    """Program AD rot90 should reject invalid rotation contracts."""

    with pytest.raises(ValueError, match="rot90 k must be a static integer"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.rot90(np.reshape(values, (2, 2)), k=True)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="rot90 axes must contain exactly two axes"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.rot90(np.reshape(values, (2, 2, 1)), axes=(0, 1, 2))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="rot90 axes axes must be unique"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.rot90(np.reshape(values, (2, 2)), axes=(0, 0))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )


def test_program_ad_flip_family_preserves_exact_adjoint() -> None:
    """Program AD flip-family permutations should preserve exact element adjoints."""

    weights_all = np.linspace(-1.0, 2.0, 24, dtype=np.float64).reshape(2, 3, 4)
    weights_axis = np.linspace(0.25, 3.25, 24, dtype=np.float64).reshape(2, 3, 4)
    weights_tuple = np.linspace(-2.5, 1.5, 24, dtype=np.float64).reshape(2, 3, 4)
    weights_ud = np.linspace(1.0, 4.0, 24, dtype=np.float64).reshape(2, 3, 4)
    weights_lr = np.linspace(-3.0, -0.25, 24, dtype=np.float64).reshape(2, 3, 4)

    def objective(values: np.ndarray) -> object:
        tensor = np.reshape(values, (2, 3, 4))
        return (
            np.sum(np.flip(tensor) * weights_all)
            + np.sum(np.flip(tensor, axis=1) * weights_axis)
            + np.sum(np.flip(tensor, axis=(0, 2)) * weights_tuple)
            + np.sum(np.flipud(tensor) * weights_ud)
            + np.sum(np.fliplr(tensor) * weights_lr)
        )

    values = np.linspace(-1.25, 1.25, 24, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )
    expected = (
        np.flip(weights_all)
        + np.flip(weights_axis, axis=1)
        + np.flip(weights_tuple, axis=(0, 2))
        + np.flipud(weights_ud)
        + np.fliplr(weights_lr)
    ).reshape(-1)

    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_flip_family_fails_closed_invalid_axes() -> None:
    """Program AD flip-family permutations should reject invalid axes."""

    with pytest.raises(ValueError, match="flip axes must be static integers"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.flip(values, axis=True)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="flip axis axes must be unique"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.flip(np.reshape(values, (2, 2)), axis=(0, 0))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="fliplr requires at least rank-2 arrays"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.fliplr(values)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )


def test_program_ad_roll_fails_closed_invalid_static_contracts() -> None:
    """Program AD roll should reject dynamic or inconsistent permutation contracts."""

    with pytest.raises(ValueError, match="roll shift must be static integers"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.roll(values, shift=True)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="roll shift and axis lengths must match"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.roll(np.reshape(values, (2, 2)), shift=(1, 2), axis=0)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="roll axis out of bounds"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.roll(np.reshape(values, (2, 2)), shift=1, axis=2)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )


def test_program_ad_axis_permutations_fail_closed_invalid_axes() -> None:
    """Program AD axis permutations should reject invalid static axis contracts."""

    with pytest.raises(ValueError, match="swapaxes axes must be static integers"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.reshape(values, (2, 2)).swapaxes(True, 1)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="moveaxis source and destination lengths must match"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.moveaxis(np.reshape(values, (2, 2, 1)), (0, 1), (2,))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="moveaxis source axes must be unique"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.moveaxis(np.reshape(values, (2, 2, 1)), (0, 0), (1, 2))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )


def test_whole_program_ad_selection_primitives_fail_closed_at_nondifferentiable_boundaries() -> (
    None
):
    """Selection primitives should reject boundary points with ambiguous derivatives."""

    with pytest.raises(ValueError, match="maximum is non-differentiable"):
        whole_program_value_and_grad(
            lambda values: np.maximum(values[0], values[1]),
            np.array([1.0, 1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="minimum is non-differentiable"):
        whole_program_value_and_grad(
            lambda values: np.minimum(values[0], values[1]),
            np.array([1.0, 1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="np.clip is non-differentiable"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.clip(values, -0.5, 0.5)),
            np.array([-0.5, 0.25], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="np.clip is non-differentiable"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.clip(values, -0.5, 0.5)),
            np.array([0.25, 0.5], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="ordering predicate is non-differentiable"):
        whole_program_value_and_grad(
            lambda values: values[0] if values[0] > values[1] else values[1],
            np.array([1.0, 1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="ordering predicate is non-differentiable"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.where(values >= 0.0, values, -values)),
            np.array([0.0, 1.0], dtype=np.float64),
        )


def test_whole_program_grad_respects_trainable_mask() -> None:
    """Whole-program gradients should preserve frozen parameters."""

    gradient = whole_program_grad(
        lambda values: values[0] ** 2 + values[1] ** 2,
        np.array([2.0, 3.0], dtype=np.float64),
        parameters=(Parameter("x"), Parameter("frozen", trainable=False)),
        trace=False,
    )

    np.testing.assert_allclose(gradient, [4.0, 0.0], rtol=1.0e-6, atol=1.0e-6)


def test_whole_program_ad_is_exported_from_package_root() -> None:
    """Whole-program AD should be stable as a package-root API."""

    import scpn_quantum_control as scpn

    assert scpn.TraceADArray is TraceADArray
    assert scpn.ProgramADAdjointResult is ProgramADAdjointResult
    assert scpn.ProgramADAliasEdge is ProgramADAliasEdge
    assert scpn.ProgramADControlRegion is ProgramADControlRegion
    assert scpn.ProgramADEffect is ProgramADEffect
    assert scpn.ProgramADEffectIR is ProgramADEffectIR
    assert scpn.ProgramADSSAValue is ProgramADSSAValue
    assert scpn.WholeProgramADResult is WholeProgramADResult
    assert scpn.WholeProgramBytecodeInstruction is WholeProgramBytecodeInstruction
    assert scpn.WholeProgramTraceEvent is WholeProgramTraceEvent
    assert scpn.WholeProgramSourceIRFeature is WholeProgramSourceIRFeature
    assert scpn.WholeProgramSemanticsReport is WholeProgramSemanticsReport
    assert scpn.program_adjoint_gradient is program_adjoint_gradient
    assert scpn.program_adjoint_result is program_adjoint_result
    assert scpn.whole_program_grad is whole_program_grad
    assert scpn.whole_program_value_and_grad is whole_program_value_and_grad


def test_program_adjoint_replay_matches_forward_program_ad_for_supported_ir() -> None:
    """Reverse-mode program adjoint replay should match forward program AD on supported IR."""

    def objective(values: np.ndarray) -> object:
        x, y, z = values
        branch = x if x > y else y
        return (
            np.sin(x)
            + np.cos(y)
            + np.exp(x - y)
            + np.log(z + 3.0)
            + np.sqrt(z + 4.0)
            + np.tanh(x * z)
            + (x**2.0)
            + (2.0**y)
            + branch
        )

    values = np.array([1.25, -0.4, 0.75], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x"), Parameter("y"), Parameter("z")),
    )
    adjoint = program_adjoint_result(result)

    assert adjoint.supported is True
    assert adjoint.unsupported_ops == ()
    assert adjoint.method == "program_adjoint_replay"
    np.testing.assert_allclose(adjoint.gradient, result.gradient, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_gradient(result), result.gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_adjoint_replay_supports_static_setitem_effects() -> None:
    """Reverse-mode program adjoints should replay static setitem dataflow exactly."""

    def objective(values: np.ndarray) -> object:
        work = values.copy()
        work[0] = values[1] * values[1]
        return work[0] + values[0]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 1.5], dtype=np.float64),
        parameters=(Parameter("x"), Parameter("y")),
    )

    assert result.adjoint_result is not None
    assert result.adjoint_result.supported is True
    assert result.adjoint_result.unsupported_ops == ()
    np.testing.assert_allclose(result.gradient, [1.0, 3.0], atol=1.0e-12)
    np.testing.assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_adjoint_result_validation_paths() -> None:
    """Program adjoint result metadata should reject malformed reverse-mode outputs."""

    result = ProgramADAdjointResult(
        gradient=np.array([1.0], dtype=np.float64),
        supported=True,
        unsupported_ops=(),
        method="program_adjoint_replay",
        claim_boundary="supported scalar replay",
    )

    assert result.supported is True
    with pytest.raises(ValueError, match="one-dimensional"):
        ProgramADAdjointResult(
            gradient=np.array([[1.0]], dtype=np.float64),
            supported=True,
            unsupported_ops=(),
            method="program_adjoint_replay",
            claim_boundary="supported scalar replay",
        )
    with pytest.raises(ValueError, match="cannot be supported"):
        ProgramADAdjointResult(
            gradient=np.array([1.0], dtype=np.float64),
            supported=True,
            unsupported_ops=("mutation:setitem",),
            method="program_adjoint_replay",
            claim_boundary="supported scalar replay",
        )
    with pytest.raises(ValueError, match="method"):
        ProgramADAdjointResult(
            gradient=np.array([1.0], dtype=np.float64),
            supported=True,
            unsupported_ops=(),
            method="",
            claim_boundary="supported scalar replay",
        )


def test_program_ad_extreme_reductions_match_strict_selection_adjoint() -> None:
    """Program AD max/min reductions should route adjoints to unique selected entries."""

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (2, 3))
        column_max = np.max(matrix, axis=0)
        row_min = np.min(matrix, axis=1)
        return column_max[0] + 2.0 * column_max[1] + 3.0 * column_max[2] + row_min[0] - row_min[1]

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 4.0, -2.0, 3.0, 0.5, -1.0], dtype=np.float64),
        parameters=(
            Parameter("x00"),
            Parameter("x01"),
            Parameter("x02"),
            Parameter("x10"),
            Parameter("x11"),
            Parameter("x12"),
        ),
    )

    assert result.value == pytest.approx(7.0)
    np.testing.assert_allclose(
        result.gradient,
        [0.0, 2.0, 1.0, 1.0, 0.0, 2.0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        program_adjoint_gradient(result),
        result.gradient,
        atol=1.0e-12,
    )


def test_program_ad_extreme_reduction_methods_match_numpy_functions() -> None:
    """Trace arrays should support ndarray-style max/min method reductions."""

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (2, 2))
        return matrix.max(axis=0)[1] - matrix.min(axis=1)[0] + matrix.max()

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, -3.0, 4.0, 2.0], dtype=np.float64),
        parameters=(
            Parameter("x00"),
            Parameter("x01"),
            Parameter("x10"),
            Parameter("x11"),
        ),
    )

    assert result.value == pytest.approx(9.0)
    np.testing.assert_allclose(result.gradient, [0.0, -1.0, 1.0, 1.0], atol=1.0e-12)


def test_program_ad_static_take_accumulates_gather_adjoint() -> None:
    """Static NumPy take gathers should preserve exact adjoint accumulation."""

    def objective(values: np.ndarray) -> object:
        vector_gather = np.take(values, [2, 0, 2])
        matrix = np.reshape(values, (2, 3))
        column_gather = matrix.take([1, 0], axis=1)
        return np.sum(vector_gather) + column_gather[0, 0] - 2.0 * column_gather[1, 1]

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(1.0)
    np.testing.assert_allclose(
        result.gradient,
        [1.0, 1.0, 2.0, -2.0, 0.0, 0.0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_static_take_rejects_dynamic_indices_and_modes() -> None:
    """Program AD take should fail closed outside static integer gather semantics."""

    with pytest.raises(ValueError, match="static integer indices"):
        whole_program_value_and_grad(
            lambda values: np.take(values, values[0]),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="mode='raise'"):
        whole_program_value_and_grad(
            lambda values: np.take(values, [0], mode="wrap")[0],
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_product_reductions_match_product_rule_adjoint() -> None:
    """Program AD product reductions should preserve exact product-rule adjoints."""

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (2, 3))
        row_products = np.prod(matrix, axis=1)
        return np.prod(values[:3]) + row_products[0] - 2.0 * row_products[1]

    result = whole_program_value_and_grad(
        objective,
        np.array([2.0, -3.0, 4.0, 5.0, -2.0, 0.5], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(-38.0)
    np.testing.assert_allclose(
        result.gradient,
        [-24.0, 16.0, -12.0, 2.0, -5.0, 20.0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_product_reduction_methods_handle_zero_factor() -> None:
    """Trace-array prod methods should handle single-zero factors without finite differences."""

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (2, 2))
        return matrix.prod(axis=0)[0] + matrix.prod()

    result = whole_program_value_and_grad(
        objective,
        np.array([0.0, 2.0, 3.0, -4.0], dtype=np.float64),
        parameters=(
            Parameter("x00"),
            Parameter("x01"),
            Parameter("x10"),
            Parameter("x11"),
        ),
    )

    assert result.value == pytest.approx(0.0)
    np.testing.assert_allclose(result.gradient, [-21.0, 0.0, 0.0, 0.0], atol=1.0e-12)


def test_program_ad_variance_and_std_reductions_match_analytic_gradients() -> None:
    """Program AD variance and standard deviation should use exact differentiable reductions."""

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (2, 2))
        return np.var(values) + matrix.var(axis=0)[1] + np.std(values[:2])

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
        ),
    )

    assert result.value == pytest.approx(10.0)
    np.testing.assert_allclose(
        result.gradient,
        [-2.0, -2.0, 0.5, 3.5],
        atol=1.0e-12,
    )


def test_program_ad_variance_and_std_reject_invalid_ddof() -> None:
    """Program AD variance/std should fail closed on unsupported or singular ddof."""

    with pytest.raises(ValueError, match="integer ddof"):
        whole_program_value_and_grad(
            lambda values: np.var(values, ddof=0.5),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="ddof must leave"):
        whole_program_value_and_grad(
            lambda values: np.std(values, ddof=2),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_cumulative_sum_matches_prefix_adjoint() -> None:
    """Program AD cumulative sums should accumulate prefix adjoints exactly."""

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (2, 3))
        flat_prefix = np.cumsum(values)
        row_prefix = matrix.cumsum(axis=1)
        return flat_prefix[3] + row_prefix[1, 2] - 2.0 * row_prefix[0, 1]

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(19.0)
    np.testing.assert_allclose(
        result.gradient,
        [-1.0, -1.0, 1.0, 2.0, 1.0, 1.0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_cumulative_product_matches_prefix_product_adjoint() -> None:
    """Program AD cumulative products should preserve exact product-rule adjoints."""

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (2, 3))
        flat_prefix = np.cumprod(values)
        row_prefix = matrix.cumprod(axis=1)
        return flat_prefix[2] + row_prefix[1, 2] - row_prefix[0, 1]

    result = whole_program_value_and_grad(
        objective,
        np.array([2.0, -3.0, 4.0, 5.0, -2.0, 0.5], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(-23.0)
    np.testing.assert_allclose(
        result.gradient,
        [-9.0, 6.0, -6.0, -1.0, 2.5, -10.0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_cumulative_product_method_handles_zero_factor() -> None:
    """Trace-array cumulative product methods should differentiate single-zero prefixes."""

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (2, 2))
        return np.cumprod(values)[3] + matrix.cumprod(axis=1)[0, 1]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.0, 2.0, 3.0, -4.0], dtype=np.float64),
        parameters=(
            Parameter("x00"),
            Parameter("x01"),
            Parameter("x10"),
            Parameter("x11"),
        ),
    )

    assert result.value == pytest.approx(0.0)
    np.testing.assert_allclose(result.gradient, [-22.0, 0.0, 0.0, 0.0], atol=1.0e-12)


def test_program_ad_finite_differences_match_linear_adjoint() -> None:
    """Program AD finite differences should preserve exact linear adjoints."""

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (2, 3))
        first_order = np.diff(values)
        second_order_rows = np.diff(matrix, n=2, axis=1)
        return first_order[2] - 2.0 * first_order[4] + second_order_rows[0, 0]

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 3.0, 6.0, 10.0, 15.0, 21.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(-7.0)
    np.testing.assert_allclose(
        result.gradient,
        [1.0, -2.0, 0.0, 1.0, 2.0, -2.0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_finite_differences_reject_boundary_extensions() -> None:
    """Program AD finite differences should fail closed for boundary-extension modes."""

    with pytest.raises(ValueError, match="non-negative integer n"):
        whole_program_value_and_grad(
            lambda values: np.diff(values, n=-1)[0],
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="prepend/append"):
        whole_program_value_and_grad(
            lambda values: np.diff(values, prepend=0.0)[0],
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_like_constant_constructors_have_zero_derivatives() -> None:
    """Program AD like-constructors should create derivative-zero constant arrays."""

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (2, 2))
        constants = np.zeros_like(values) + np.ones_like(values) + np.full_like(values, 2.0)
        matrix_constants = np.full_like(matrix, -0.5)
        return np.sum(values * constants) + np.sum(matrix_constants * matrix)

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
        ),
    )

    assert result.value == pytest.approx(25.0)
    np.testing.assert_allclose(result.gradient, [2.5, 2.5, 2.5, 2.5], atol=1.0e-12)
    np.testing.assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_like_constant_constructors_reject_shape_overrides() -> None:
    """Program AD like-constructors should fail closed on unsupported shape overrides."""

    with pytest.raises(ValueError, match="shape overrides"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.zeros_like(values, shape=(2, 2))),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_broadcast_to_accumulates_repeated_adjoint_paths() -> None:
    """Program AD broadcast_to should accumulate derivatives from repeated uses."""

    weights = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

    def objective(values: np.ndarray) -> object:
        broadcast = np.broadcast_to(values, (3, 2))
        return np.sum(broadcast * weights)

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0], dtype=np.float64),
        parameters=(Parameter("x0"), Parameter("x1")),
    )

    assert result.value == pytest.approx(33.0)
    np.testing.assert_allclose(result.gradient, [9.0, 12.0], atol=1.0e-12)
    np.testing.assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_broadcast_to_rejects_subclass_propagation() -> None:
    """Program AD broadcast_to should fail closed on subclass propagation."""

    with pytest.raises(ValueError, match="subok"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.broadcast_to(values, (2, 2), subok=True)),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_basic_slicing_preserves_static_adjoint_paths() -> None:
    """Program AD basic slicing should preserve exact static index adjoints."""

    def objective(values: np.ndarray) -> object:
        matrix = np.reshape(values, (2, 3))
        block = matrix[..., 1:]
        expanded = matrix[:, :1][:, None, :]
        return np.sum(block * np.array([[1.0, 2.0], [3.0, 4.0]])) + 2.0 * expanded[1, 0, 0]

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(55.0)
    np.testing.assert_allclose(result.gradient, [0.0, 1.0, 2.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_advanced_indexing_fails_closed() -> None:
    """Program AD array indexing should reject dynamic advanced index selectors."""

    with pytest.raises(ValueError, match="advanced indexing"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.reshape(values, (2, 2))[[0, 1]]),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="advanced indexing"):
        whole_program_value_and_grad(
            lambda values: np.sum(values[np.array([True, False])]),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_index_selection_primitives_fail_closed() -> None:
    """Index-valued selection should require an explicit nondifferentiable policy."""

    with pytest.raises(ValueError, match="argmax/argmin index selection semantics"):
        whole_program_value_and_grad(
            lambda values: np.argmax(values),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="argmax/argmin index selection semantics"):
        whole_program_value_and_grad(
            lambda values: np.reshape(values, (2, 2)).argmin(axis=1)[0],
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )


def test_program_ad_extreme_reductions_fail_closed_on_ties() -> None:
    """Program AD max/min reductions should reject nondifferentiable tied selectors."""

    with pytest.raises(ValueError, match="np.max.*ties"):
        whole_program_value_and_grad(
            lambda values: np.max(values),
            np.array([2.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="np.min.*ties"):
        whole_program_value_and_grad(
            lambda values: np.min(np.reshape(values, (2, 2)), axis=1)[0],
            np.array([1.0, 1.0, 3.0, 4.0], dtype=np.float64),
        )


def test_whole_program_ad_emits_deterministic_ssa_effect_ir() -> None:
    """Program AD should expose deterministic SSA, alias, mutation, and control metadata."""

    def objective(values: np.ndarray) -> object:
        alias = values.copy()
        total = values[0]
        for index in range(1, 3):
            total = total + alias[index] * float(index)
        if total > 0.0:
            alias[0] = total
        return alias[0] + np.sin(values[2])

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, 0.5, 0.75], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    )

    assert result.program_ir is not None
    assert result.program_ir.serialization.startswith('{"alias_edges"')
    assert "program_ad_effect_ir.v1" in result.program_ir.serialization
    assert result.program_ir.ssa_values[0] == ProgramADSSAValue(
        name="%0",
        producer=0,
        version=0,
        shape=(),
        dtype="float64",
        effect=0,
    )
    assert [effect.ordering for effect in result.program_ir.effects] == list(
        range(len(result.program_ir.effects))
    )
    assert any(effect.kind == "mutation" for effect in result.program_ir.effects)
    assert any(effect.kind == "control_branch" for effect in result.program_ir.effects)
    assert any(edge.kind == "mutation_version" for edge in result.program_ir.alias_edges)
    assert result.program_ir.alias_edges
    assert any(region.kind == "runtime_branch" for region in result.program_ir.control_regions)
    assert any(region.kind.startswith("source_") for region in result.program_ir.control_regions)
    np.testing.assert_allclose(
        result.gradient,
        [1.0, 1.0, 2.0 + math.cos(0.75)],
        atol=1.0e-12,
    )


def test_program_ad_effect_ir_validation_paths() -> None:
    """Program AD IR dataclasses should fail closed on malformed compiler metadata."""

    value = ProgramADSSAValue("%0", producer=0, version=0, shape=(), dtype="float64", effect=0)
    effect = ProgramADEffect(
        index=0,
        kind="pure",
        target="%0",
        inputs=("theta",),
        version=0,
        ordering=0,
    )
    edge = ProgramADAliasEdge(source="alias", target="%0", kind="source_alias", version=0)
    region = ProgramADControlRegion(
        index=0,
        kind="runtime_branch",
        predicate="%0:gt:0.0",
        entered=True,
        source_line=None,
    )
    ir = ProgramADEffectIR(
        ssa_values=(value,),
        effects=(effect,),
        alias_edges=(edge,),
        control_regions=(region,),
        serialization="program_ad_effect_ir.v1",
    )

    assert ir.ssa_values == (value,)
    with pytest.raises(ValueError, match="SSA value name"):
        ProgramADSSAValue("", producer=0, version=0, shape=(), dtype="float64")
    with pytest.raises(ValueError, match="effect kind"):
        ProgramADEffect(index=0, kind="", target="%0", inputs=(), version=0, ordering=0)
    with pytest.raises(ValueError, match="alias source"):
        ProgramADAliasEdge(source="", target="%0", kind="source_alias", version=0)
    with pytest.raises(ValueError, match="control region kind"):
        ProgramADControlRegion(index=0, kind="", predicate=None, entered=True, source_line=None)
    with pytest.raises(ValueError, match="serialization"):
        ProgramADEffectIR(
            ssa_values=(value,),
            effects=(effect,),
            alias_edges=(edge,),
            control_regions=(region,),
            serialization="",
        )


def test_custom_derivative_registry_binds_rule_by_primitive_identity() -> None:
    """Registered primitive identities should resolve exact custom rules automatically."""

    identity = PrimitiveIdentity("scpn.quantum", "rx_expectation", "1")
    rule = CustomDerivativeRule(
        name="rx_expectation_rule",
        value_fn=lambda values: np.array([np.cos(values[0]), values[1] ** 2], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [-np.sin(values[0]) * tangent[0], 2.0 * values[1] * tangent[1]],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [-np.sin(values[0]) * cotangent[0], 2.0 * values[1] * cotangent[1]],
            dtype=np.float64,
        ),
        parameter_names=("theta", "gain"),
        trainable=(True, False),
    )
    registry = CustomDerivativeRegistry()

    assert registry.register(identity, rule) is rule
    assert registry.lookup("scpn.quantum:rx_expectation@1") is rule
    assert custom_derivative_rule_for(identity, registry=registry) is rule
    np.testing.assert_allclose(
        registered_custom_jvp(
            "scpn.quantum:rx_expectation@1",
            np.array([0.25, 3.0], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            registry=registry,
        ),
        [-math.sin(0.25), 0.0],
        atol=1.0e-12,
    )
    vjp = registered_custom_vjp(
        identity,
        np.array([0.25, 3.0], dtype=np.float64),
        np.array([2.0, -1.0], dtype=np.float64),
        registry=registry,
    )
    np.testing.assert_allclose(vjp.vjp, [-2.0 * math.sin(0.25), 0.0], atol=1.0e-12)
    jacobian_result = registered_custom_jacobian(
        identity,
        np.array([0.25, 3.0], dtype=np.float64),
        registry=registry,
    )
    np.testing.assert_allclose(
        jacobian_result.jacobian,
        [[-math.sin(0.25), 0.0], [0.0, 0.0]],
        atol=1.0e-12,
    )


def test_custom_derivative_registry_rejects_ambiguous_identity_and_conflicts() -> None:
    """Registry bindings should fail closed on malformed keys and rule conflicts."""

    rule = CustomDerivativeRule(
        name="linear_rule",
        value_fn=lambda values: np.array([values[0]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array([tangent[0]], dtype=np.float64),
    )
    other = CustomDerivativeRule(
        name="other_linear_rule",
        value_fn=lambda values: np.array([2.0 * values[0]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array([2.0 * tangent[0]], dtype=np.float64),
    )
    registry = CustomDerivativeRegistry()
    registry.register("scpn.test:linear@1", rule)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("scpn.test:linear@1", other)
    assert registry.register("scpn.test:linear@1", other, overwrite=True) is other
    with pytest.raises(ValueError, match="namespace:name"):
        PrimitiveIdentity.parse("bad-key")
    with pytest.raises(ValueError, match="no custom derivative rule"):
        custom_derivative_rule_for("scpn.test:missing@1", registry=registry)
    removed = registry.unregister("scpn.test:linear@1")
    assert removed is other
    assert registry.snapshot() == {}


def test_custom_derivative_global_registry_and_root_exports() -> None:
    """The default registry and root package exports should be stable."""

    import scpn_quantum_control as scpn

    assert scpn.PrimitiveIdentity is PrimitiveIdentity
    assert scpn.CustomDerivativeRegistry is CustomDerivativeRegistry
    assert scpn.DEFAULT_CUSTOM_DERIVATIVE_REGISTRY is DEFAULT_CUSTOM_DERIVATIVE_REGISTRY
    assert scpn.register_custom_derivative_rule is register_custom_derivative_rule
    assert scpn.custom_derivative_rule_for is custom_derivative_rule_for
    assert scpn.registered_custom_jvp is registered_custom_jvp
    assert scpn.registered_custom_vjp is registered_custom_vjp
    assert scpn.registered_custom_jacobian is registered_custom_jacobian


def test_whole_program_ad_records_ir_and_executed_branch_semantics() -> None:
    """Whole-program AD should differentiate the executed Python control-flow branch exactly."""

    def objective(values: np.ndarray) -> object:
        total = values[0] * values[0]
        for index, value in enumerate(values):
            if value > 0.0:
                total = total + np.sin(value) + index * value
            else:
                total = total + value**2
        return total

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, -0.5, 0.75], dtype=np.float64),
        parameters=(Parameter("theta"), Parameter("bias"), Parameter("phase")),
    )

    assert result.method == "whole_program_ad"
    assert result.step == 0.0
    assert result.control_flow_observed is True
    assert result.numpy_observed is True
    assert any(node.op.startswith("branch:") for node in result.ir_nodes)
    assert any(node.op == "sin" for node in result.ir_nodes)
    np.testing.assert_allclose(
        result.gradient,
        [2.0 * 0.25 + math.cos(0.25), -1.0, math.cos(0.75) + 2.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_whole_program_grad_respects_trainable_mask_and_rejects_derivative_loss() -> None:
    """Whole-program AD should freeze masked parameters and reject float-cast derivative loss."""

    gradient = whole_program_grad(
        lambda values: values[0] ** 2 + values[1] ** 2,
        np.array([2.0, 3.0], dtype=np.float64),
        parameters=(Parameter("x"), Parameter("frozen", trainable=False)),
        trace=False,
    )
    np.testing.assert_allclose(gradient, [4.0, 0.0], rtol=1.0e-12, atol=1.0e-12)

    with pytest.raises(ValueError, match="converted to float"):
        whole_program_value_and_grad(
            lambda values: float(values[0] ** 2),
            np.array([2.0], dtype=np.float64),
        )


def test_whole_program_ad_operator_surface_and_fail_closed_paths() -> None:
    """Whole-program AD should cover scalar operator interception and reject derivative loss."""

    def objective(values: np.ndarray) -> object:
        x, y = values
        branch = x if x >= y else y
        reverse_branch = y if y <= x else x
        return (
            np.sin(x)
            + np.cos(y)
            + np.exp(x - y)
            + np.log(x + 3.0)
            + (2.0 + x)
            + (5.0 - y)
            + (3.0 * x)
            + (12.0 / (x + 4.0))
            + (x**2.0)
            + (2.0 ** (y + 1.0))
            - branch
            + reverse_branch
            + (-y)
        )

    result = whole_program_value_and_grad(objective, [1.5, 0.25], trace=False)

    assert result.method == "whole_program_ad"
    assert result.evaluations == 1
    assert result.numpy_observed is True
    assert result.control_flow_observed is True
    assert result.gradient.shape == (2,)
    ops = {node.op.split(":", maxsplit=1)[0] for node in result.ir_nodes}
    assert {
        "parameter",
        "branch",
        "sin",
        "cos",
        "exp",
        "log",
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "neg",
    }.issubset(ops)

    with pytest.raises(ValueError, match="converted to float"):
        whole_program_value_and_grad(lambda values: float(values[0]), [1.0])
    with pytest.raises(ValueError, match="denominator"):
        whole_program_value_and_grad(lambda values: values[0] / 0.0, [1.0])
    with pytest.raises(ValueError, match="log input"):
        whole_program_value_and_grad(lambda values: np.log(values[0]), [-1.0])
    with pytest.raises(ValueError, match="unsupported whole-program AD NumPy ufunc"):
        whole_program_value_and_grad(lambda values: np.arctan2(values[0], values[0]), [0.25])
    np.testing.assert_allclose(
        whole_program_grad(lambda values: np.add(values[0], values[0]), [1.0]),
        [2.0],
    )
    with pytest.raises(ValueError, match="direct NumPy scalar ufunc"):
        whole_program_value_and_grad(
            lambda values: values[0].__array_ufunc__(np.sin, "reduce", values[0]), [1.0]
        )
    with pytest.raises(ValueError, match="different traces"):
        left_context = type("_TraceCtx", (), {"parameter_count": 1})()
        right_context = type("_TraceCtx", (), {"parameter_count": 1})()
        left = TraceADScalar(1.0, np.array([1.0], dtype=np.float64), left_context, "%l")  # type: ignore[arg-type]
        right = TraceADScalar(2.0, np.array([1.0], dtype=np.float64), right_context, "%r")  # type: ignore[arg-type]
        _ = left + right


def test_whole_program_result_validation_fail_closed_paths() -> None:
    """Whole-program AD result contracts should reject malformed metadata."""

    valid = {
        "value": 1.0,
        "gradient": np.array([1.0, 2.0], dtype=np.float64),
        "method": "whole_program_ad",
        "step": 0.0,
        "evaluations": 1,
        "parameter_names": ("x", "y"),
        "trainable": (True, False),
        "trace_events": (),
        "ir_nodes": (),
        "source": None,
        "control_flow_observed": False,
        "numpy_observed": False,
        "polyglot_targets": {"python": "available"},
        "claim_boundary": "bounded claim",
    }

    for key, value, message in (
        ("value", float("nan"), "finite"),
        ("gradient", np.array([[1.0]]), "one-dimensional"),
        ("step", -1.0, "step"),
        ("evaluations", 0, "evaluations"),
        ("parameter_names", ("x",), "parameter_names"),
        ("trainable", (True,), "trainable"),
        ("trace_events", (object(),), "trace_events"),
        ("ir_nodes", (object(),), "ir_nodes"),
        ("control_flow_observed", "yes", "control_flow_observed"),
        ("numpy_observed", "yes", "numpy_observed"),
        ("polyglot_targets", {}, "polyglot_targets"),
        ("polyglot_targets", {"": "available"}, "polyglot target"),
        ("claim_boundary", "", "claim_boundary"),
    ):
        payload = dict(valid)
        payload[key] = value
        with pytest.raises(ValueError, match=message):
            WholeProgramADResult(**payload)


def test_whole_program_event_and_ir_node_validation_paths() -> None:
    """Trace event and IR node records should reject malformed trace metadata."""

    event = WholeProgramTraceEvent("objective.py", "loss", 7, "  return x  ")
    assert event.source == "return x"

    valid_tangent = np.array([1.0, 0.0], dtype=np.float64)
    node = WholeProgramIRNode(0, "parameter", ("x",), 1.5, valid_tangent)
    assert node.value == pytest.approx(1.5)
    np.testing.assert_allclose(node.tangent, valid_tangent)

    for kwargs, message in (
        ({"filename": "", "function_name": "loss", "line_number": 1, "source": ""}, "filename"),
        (
            {"filename": "objective.py", "function_name": "", "line_number": 1, "source": ""},
            "function_name",
        ),
        (
            {"filename": "objective.py", "function_name": "loss", "line_number": 0, "source": ""},
            "line_number",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            WholeProgramTraceEvent(**kwargs)

    for args, message in (
        ((-1, "parameter", ("x",), 1.0, valid_tangent), "index"),
        ((0, "", ("x",), 1.0, valid_tangent), "op"),
        ((0, "parameter", ("",), 1.0, valid_tangent), "inputs"),
        ((0, "parameter", ("x",), float("inf"), valid_tangent), "finite"),
        ((0, "parameter", ("x",), 1.0, np.array([[1.0]])), "one-dimensional"),
        ((0, "parameter", ("x",), 1.0, np.array([float("nan")])), "finite"),
    ):
        with pytest.raises(ValueError, match=message):
            WholeProgramIRNode(*args)


def test_whole_program_ad_rejects_unsupported_power_and_return_contracts() -> None:
    """Whole-program AD should fail closed for unsupported powers and non-scalar returns."""

    with pytest.raises(ValueError, match="positive base"):
        whole_program_value_and_grad(lambda values: (-values[0]) ** values[0], [1.0])
    with pytest.raises(ValueError, match="whole-program AD scalar"):
        whole_program_value_and_grad(lambda values: np.array([values[0]], dtype=object), [1.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        TraceADScalar(
            1.0,
            np.array([[1.0]], dtype=np.float64),
            type("_TraceCtx", (), {"parameter_count": 1})(),
            "%bad",
        )


def test_transform_algebra_grad_of_vmap_and_vmap_of_grad_are_consistent() -> None:
    """grad(vmap(f)) and vmap(grad(f)) should agree for separable objectives."""

    def sample_loss(row: np.ndarray) -> float:
        return float(row[0] ** 2 + np.sin(row[1]))

    values = np.array([[1.5, 0.25], [-0.5, -0.75]], dtype=np.float64)
    per_sample_grad = vmap(
        lambda row: grad(sample_loss, row, method="finite_difference"),
    )(values)
    aggregate_grad = grad(
        lambda flat: float(np.sum(vmap(sample_loss)(flat.reshape(values.shape)))),
        values.reshape(-1),
        method="finite_difference",
    ).reshape(values.shape)

    np.testing.assert_allclose(per_sample_grad, aggregate_grad, rtol=1.0e-6, atol=1.0e-6)


def test_transform_algebra_jacfwd_jacrev_jvp_vjp_and_hessian_contracts() -> None:
    """Canonical Jacobian, directional, adjoint, and Hessian transforms should compose."""

    def vector_objective(values: np.ndarray) -> np.ndarray:
        return np.array(
            [values[0] ** 2, np.sin(values[1]), values[0] * values[1]],
            dtype=np.float64,
        )

    values = np.array([1.25, -0.4], dtype=np.float64)
    expected_jacobian = np.array(
        [[2.5, 0.0], [0.0, math.cos(-0.4)], [-0.4, 1.25]],
        dtype=np.float64,
    )
    forward = jacfwd(vector_objective, values)
    reverse = jacrev(vector_objective, values)
    canonical = jacobian(vector_objective, values)

    np.testing.assert_allclose(forward, expected_jacobian, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(reverse, expected_jacobian, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(canonical, expected_jacobian, rtol=1.0e-6, atol=1.0e-6)
    tangent = np.array([0.5, -2.0], dtype=np.float64)
    jvp_result = value_and_finite_difference_jvp(vector_objective, values, tangent)
    np.testing.assert_allclose(
        jvp_result.jvp, expected_jacobian @ tangent, rtol=1.0e-6, atol=1.0e-6
    )
    cotangent = np.array([1.0, -0.5, 2.0], dtype=np.float64)
    vjp_result = finite_difference_vjp(vector_objective, values, cotangent)
    np.testing.assert_allclose(
        vjp_result.vjp, expected_jacobian.T @ cotangent, rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(
        hessian(lambda row: float(row[0] ** 2 + row[0] * row[1] + np.sin(row[1])), values),
        [[2.0, 1.0], [1.0, -math.sin(-0.4)]],
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_transform_algebra_nested_batch_jacobian_and_adjoint_contracts() -> None:
    """Nested grad, vmap, Jacobian, Hessian, JVP, and VJP transforms should agree."""

    def sample_loss(row: np.ndarray) -> float:
        return float(row[0] ** 3 + row[0] * row[1] + np.sin(row[1]))

    def sample_gradient(row: np.ndarray) -> np.ndarray:
        return np.array([3.0 * row[0] ** 2 + row[1], row[0] + math.cos(row[1])])

    def sample_hessian(row: np.ndarray) -> np.ndarray:
        return np.array([[6.0 * row[0], 1.0], [1.0, -math.sin(row[1])]], dtype=np.float64)

    values = np.array([[0.7, -0.2], [-1.1, 0.4]], dtype=np.float64)
    expected_gradients = np.vstack([sample_gradient(row) for row in values])

    per_sample_gradients = vmap(
        lambda row: grad(sample_loss, row, method="finite_difference", step=1.0e-6)
    )(values)
    aggregate_gradient = grad(
        lambda flat: float(np.sum(vmap(sample_loss)(flat.reshape(values.shape)))),
        values.reshape(-1),
        method="finite_difference",
        step=1.0e-6,
    ).reshape(values.shape)
    np.testing.assert_allclose(per_sample_gradients, expected_gradients, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(aggregate_gradient, expected_gradients, rtol=1.0e-5, atol=1.0e-5)

    def batched_vector_objective(flat: np.ndarray) -> np.ndarray:
        return vmap(sample_loss)(flat.reshape(values.shape))

    expected_jacobian = np.zeros((2, 4), dtype=np.float64)
    expected_jacobian[0, 0:2] = expected_gradients[0]
    expected_jacobian[1, 2:4] = expected_gradients[1]
    flat_values = values.reshape(-1)
    forward_jacobian = jacfwd(batched_vector_objective, flat_values, step=1.0e-6)
    reverse_jacobian = jacrev(batched_vector_objective, flat_values, step=1.0e-6)
    np.testing.assert_allclose(forward_jacobian, expected_jacobian, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(reverse_jacobian, expected_jacobian, rtol=1.0e-5, atol=1.0e-5)

    tangent = np.array([0.5, -0.25, 1.25, -0.75], dtype=np.float64)
    jvp_result = value_and_finite_difference_jvp(
        batched_vector_objective,
        flat_values,
        tangent,
        step=1.0e-6,
    )
    np.testing.assert_allclose(
        jvp_result.jvp, expected_jacobian @ tangent, rtol=1.0e-5, atol=1.0e-5
    )

    cotangent = np.array([2.0, -0.5], dtype=np.float64)
    vjp_result = finite_difference_vjp(
        batched_vector_objective,
        flat_values,
        cotangent,
        step=1.0e-6,
    )
    np.testing.assert_allclose(
        vjp_result.vjp, expected_jacobian.T @ cotangent, rtol=1.0e-5, atol=1.0e-5
    )
    np.testing.assert_allclose(
        hessian(sample_loss, values[0], step=1.0e-4),
        sample_hessian(values[0]),
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_canonical_jvp_vjp_transforms_compose_with_vmap_and_jacobians() -> None:
    """Canonical JVP/VJP names should compose with vmap, jacfwd, jacrev, and hessian."""

    def sample_loss(row: np.ndarray) -> float:
        return float(row[0] ** 2 + row[0] * row[1] + np.sin(row[1]))

    values = np.array([[0.4, -0.3], [1.2, 0.5]], dtype=np.float64)
    flat_values = values.reshape(-1)
    tangent = np.array([0.5, -1.0, 1.5, -0.25], dtype=np.float64)
    cotangent = np.array([2.0, -0.75], dtype=np.float64)

    def batched_loss(flat: np.ndarray) -> np.ndarray:
        return vmap(sample_loss)(flat.reshape(values.shape))

    forward_jacobian = jacfwd(batched_loss, flat_values, step=1.0e-6)
    reverse_jacobian = jacrev(batched_loss, flat_values, step=1.0e-6)
    jvp_result = value_and_jvp(batched_loss, flat_values, tangent, step=1.0e-6)
    vjp_result = value_and_vjp(batched_loss, flat_values, cotangent, step=1.0e-6)

    np.testing.assert_allclose(forward_jacobian, reverse_jacobian, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(jvp_result.value, batched_loss(flat_values), atol=1.0e-12)
    np.testing.assert_allclose(vjp_result.value, batched_loss(flat_values), atol=1.0e-12)
    np.testing.assert_allclose(
        jvp(batched_loss, flat_values, tangent, step=1.0e-6),
        forward_jacobian @ tangent,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        vjp(batched_loss, flat_values, cotangent, step=1.0e-6),
        reverse_jacobian.T @ cotangent,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        hessian(sample_loss, values[0], step=1.0e-4),
        jacfwd(lambda row: grad(sample_loss, row, method="finite_difference"), values[0]),
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_transform_algebra_custom_rules_and_whole_program_ad_compose_with_vmap() -> None:
    """Custom rules and whole-program AD should compose under vmap."""

    rule = CustomDerivativeRule(
        name="affine_sine_rule",
        value_fn=lambda values: np.array([values[0] + np.sin(values[1])], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [tangent[0] + math.cos(values[1]) * tangent[1]], dtype=np.float64
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [cotangent[0], math.cos(values[1]) * cotangent[0]], dtype=np.float64
        ),
        parameter_names=("offset", "phase"),
        trainable=(True, True),
    )
    values = np.array([[1.0, 0.25], [-2.0, -0.5]], dtype=np.float64)
    custom_jacobians = vmap(lambda row: custom_jacobian(rule, row))(values)
    trace_gradients = vmap(
        lambda row: whole_program_grad(
            lambda trace_values: trace_values[0] + np.sin(trace_values[1]),
            row,
            trace=False,
        )
    )(values)

    expected = np.array(
        [[[1.0, math.cos(0.25)]], [[1.0, math.cos(-0.5)]]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(custom_jacobians, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(trace_gradients, expected[:, 0, :], rtol=1.0e-12, atol=1.0e-12)


def test_transform_algebra_aliases_are_exported_from_package_root() -> None:
    """Transform algebra aliases should be stable package-root APIs."""

    import scpn_quantum_control as scpn

    assert scpn.jacfwd is jacfwd
    assert scpn.jacrev is jacrev
    assert scpn.jvp is jvp
    assert scpn.vjp is vjp
    assert scpn.value_and_jacfwd is value_and_jacfwd
    assert scpn.value_and_jacrev is value_and_jacrev
    assert scpn.value_and_jvp is value_and_jvp
    assert scpn.value_and_vjp is value_and_vjp


def test_vmap_uses_registered_primitive_batching_rule() -> None:
    """vmap should dispatch to primitive-specific batching rules when requested."""

    identity = PrimitiveIdentity("scpn.quantum", "batched_affine", "1")
    rule = CustomDerivativeRule(
        name="batched_affine_rule",
        value_fn=lambda values: np.array([values[0] + 2.0 * values[1]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [tangent[0] + 2.0 * tangent[1]], dtype=np.float64
        ),
        parameter_names=("offset", "phase"),
        trainable=(True, True),
    )
    registry = CustomDerivativeRegistry()
    registry.register(identity, rule)
    calls: list[str] = []

    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        calls.append("batching_rule")
        assert axes == (0, None)
        batch = np.asarray(args[0], dtype=np.float64)
        scale = float(args[1])
        return np.stack([batch[:, 0] + scale * batch[:, 1]], axis=out_axes)

    registry.register_batching_rule(identity, batching_rule)
    batched = vmap(
        lambda row, scale: row[0] + scale * row[1],
        in_axes=(0, None),
        out_axes=1,
        primitive_identity=identity,
        registry=registry,
    )

    result = batched(np.array([[1.0, 2.0], [3.0, -1.0]], dtype=np.float64), 2.0)

    assert calls == ["batching_rule"]
    np.testing.assert_allclose(result, [[5.0], [1.0]], atol=1.0e-12)


def test_primitive_transform_registry_holds_derivative_batching_and_lowering_metadata() -> None:
    """Registry transform bindings should keep complete primitive metadata together."""

    identity = PrimitiveIdentity("scpn.quantum", "lowered_batch", "1")
    rule = CustomDerivativeRule(
        name="lowered_batch_rule",
        value_fn=lambda values: np.array([values[0]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array([tangent[0]], dtype=np.float64),
    )

    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        del function, axes
        return np.asarray(args[0], dtype=np.float64).sum(axis=1 + out_axes * 0)

    def lowering_rule(lowered_rule: CustomDerivativeRule) -> object:
        return lowered_rule.name

    def shape_rule(args: tuple[object, ...]) -> tuple[int, ...]:
        del args
        return (1,)

    def dtype_rule(args: tuple[object, ...]) -> str:
        del args
        return "float64"

    registry = CustomDerivativeRegistry()
    transform = PrimitiveTransformRule(
        identity=identity,
        derivative_rule=rule,
        batching_rule=batching_rule,
        lowering_rule=lowering_rule,
        lowering_metadata={"mlir_op": "scpn_diff.lowered_batch", "rust": "blocked"},
        shape_rule=shape_rule,
        dtype_rule=dtype_rule,
        nondifferentiable_policy="fail_closed_at_boundaries",
        effect="pure",
    )

    assert registry.register_transform(transform) is transform
    assert registry.require(identity) is rule
    assert registry.require_batching_rule(identity) is batching_rule
    assert registry.require_lowering_rule(identity) is lowering_rule
    assert registry.require_shape_rule(identity) is shape_rule
    assert registry.require_dtype_rule(identity) is dtype_rule
    assert registry.require_nondifferentiable_policy(identity) == "fail_closed_at_boundaries"
    assert registry.require_effect(identity) == "pure"
    assert primitive_shape_rule_for(identity, registry=registry) is shape_rule
    assert primitive_dtype_rule_for(identity, registry=registry) is dtype_rule
    assert (
        primitive_nondifferentiable_policy_for(identity, registry=registry)
        == "fail_closed_at_boundaries"
    )
    assert primitive_effect_for(identity, registry=registry) == "pure"
    snapshot = registry.transform_snapshot()
    assert snapshot[identity].lowering_rule is lowering_rule
    assert snapshot[identity].lowering_metadata["mlir_op"] == "scpn_diff.lowered_batch"
    assert snapshot[identity].shape_rule is shape_rule
    assert snapshot[identity].dtype_rule is dtype_rule
    assert snapshot[identity].shape_rule is not None
    assert snapshot[identity].shape_rule(()) == (1,)
    assert snapshot[identity].dtype_rule is not None
    assert snapshot[identity].dtype_rule(()) == "float64"
    assert snapshot[identity].nondifferentiable_policy == "fail_closed_at_boundaries"
    assert snapshot[identity].effect == "pure"
    with pytest.raises(ValueError, match="batching rule already registered"):
        registry.register_batching_rule(identity, batching_rule)
    with pytest.raises(ValueError, match="lowering rule already registered"):
        registry.register_lowering_rule(identity, lowering_rule)


def test_primitive_transform_registry_preserves_contract_metadata_on_partial_updates() -> None:
    """Partial registry updates should not erase shape, dtype, policy, or effect contracts."""

    identity = PrimitiveIdentity("scpn.quantum", "metadata_preserved", "1")
    rule = CustomDerivativeRule(
        name="metadata_preserved_rule",
        value_fn=lambda values: np.array([values[0]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array([tangent[0]], dtype=np.float64),
    )

    def shape_rule(args: tuple[object, ...]) -> tuple[int, ...]:
        del args
        return (1,)

    def dtype_rule(args: tuple[object, ...]) -> str:
        del args
        return "float64"

    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        del function, axes, out_axes
        return np.asarray(args[0], dtype=np.float64)

    def lowering_rule(lowered_rule: CustomDerivativeRule) -> object:
        return lowered_rule.name

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            shape_rule=shape_rule,
            dtype_rule=dtype_rule,
            nondifferentiable_policy="fail_closed",
            effect="pure",
        )
    )

    registry.register_batching_rule(identity, batching_rule)
    registry.register_lowering_rule(identity, lowering_rule)
    transform = registry.transform_snapshot()[identity]

    assert transform.shape_rule is shape_rule
    assert transform.dtype_rule is dtype_rule
    assert transform.nondifferentiable_policy == "fail_closed"
    assert transform.effect == "pure"
    assert transform.batching_rule is batching_rule
    assert transform.lowering_rule is lowering_rule


def test_primitive_transform_registry_contract_accessors_fail_closed() -> None:
    """Primitive contract accessors should fail closed when metadata is absent."""

    identity = PrimitiveIdentity("scpn.quantum", "contract_missing", "1")
    rule = CustomDerivativeRule(
        name="contract_missing_rule",
        value_fn=lambda values: np.array([values[0]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array([tangent[0]], dtype=np.float64),
    )
    registry = CustomDerivativeRegistry({identity: rule})

    assert registry.shape_rule_for(identity) is None
    assert registry.dtype_rule_for(identity) is None
    assert registry.nondifferentiable_policy_for(identity) == "not_declared"
    assert registry.effect_for(identity) == "pure"
    with pytest.raises(ValueError, match="no shape rule"):
        registry.require_shape_rule(identity)
    with pytest.raises(ValueError, match="no dtype rule"):
        registry.require_dtype_rule(identity)
    with pytest.raises(ValueError, match="no nondifferentiable policy"):
        registry.require_nondifferentiable_policy(identity)


def test_primitive_registry_exposes_unified_contracts() -> None:
    """Primitive registry should expose all transform contracts in one fail-closed lookup."""

    identity = PrimitiveIdentity("scpn.quantum", "unified_contract", "1")
    rule = CustomDerivativeRule(
        name="unified_contract_rule",
        value_fn=lambda values: np.array([values[0]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array([tangent[0]], dtype=np.float64),
    )

    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        del function, axes, out_axes
        return np.asarray(args[0], dtype=np.float64)

    def lowering_rule(lowered_rule: CustomDerivativeRule) -> object:
        return lowered_rule.name

    def shape_rule(args: tuple[object, ...]) -> tuple[int, ...]:
        del args
        return (1,)

    def dtype_rule(args: tuple[object, ...]) -> str:
        del args
        return "float64"

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=batching_rule,
            lowering_rule=lowering_rule,
            lowering_metadata={"mlir_op": "scpn_diff.unified_contract"},
            shape_rule=shape_rule,
            dtype_rule=dtype_rule,
            nondifferentiable_policy="fail_closed_at_boundaries",
            effect="pure",
        )
    )

    contract = registry.require_contract(identity)
    assert isinstance(contract, PrimitiveContract)
    assert contract.identity is identity
    assert contract.derivative_rule is rule
    assert contract.batching_rule is batching_rule
    assert contract.lowering_rule is lowering_rule
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.unified_contract"
    assert contract.shape_rule is shape_rule
    assert contract.dtype_rule is dtype_rule
    assert contract.nondifferentiable_policy == "fail_closed_at_boundaries"
    assert contract.effect == "pure"
    assert primitive_contract_for(identity, registry=registry) == contract

    missing = PrimitiveIdentity("scpn.quantum", "unified_contract_missing", "1")
    with pytest.raises(ValueError, match="no primitive contract"):
        registry.require_contract(missing)


def test_primitive_registry_requires_complete_contract_for_compiler_consumers() -> None:
    """Compiler-facing primitive contracts should fail closed until every facet is declared."""

    identity = PrimitiveIdentity("scpn.quantum", "strict_contract", "1")
    rule = CustomDerivativeRule(
        name="strict_contract_rule",
        value_fn=lambda values: np.array([values[0] * values[1]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [values[1] * tangent[0] + values[0] * tangent[1]], dtype=np.float64
        ),
    )

    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        del function, axes
        return np.moveaxis(np.prod(np.asarray(args[0], dtype=np.float64), axis=1), 0, out_axes)

    def lowering_rule(lowered_rule: CustomDerivativeRule) -> object:
        return lowered_rule.name

    def shape_rule(args: tuple[object, ...]) -> tuple[int, ...]:
        del args
        return (1,)

    def dtype_rule(args: tuple[object, ...]) -> str:
        del args
        return "float64"

    registry = CustomDerivativeRegistry({identity: rule})
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        registry.require_complete_contract(identity)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(identity, registry=registry)

    transform = PrimitiveTransformRule(
        identity=identity,
        derivative_rule=rule,
        batching_rule=batching_rule,
        lowering_rule=lowering_rule,
        lowering_metadata={"mlir_op": "scpn_diff.strict_contract"},
        shape_rule=shape_rule,
        dtype_rule=dtype_rule,
        nondifferentiable_policy="fail_closed_at_boundaries",
        effect="pure",
    )
    registry.register_transform(transform, overwrite=True)

    contract = registry.require_complete_contract(identity)
    assert contract.derivative_rule is rule
    assert contract.batching_rule is batching_rule
    assert contract.lowering_rule is lowering_rule
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.strict_contract"
    assert contract.shape_rule is shape_rule
    assert contract.dtype_rule is dtype_rule
    assert contract.nondifferentiable_policy == "fail_closed_at_boundaries"
    assert contract.effect == "pure"
    assert primitive_complete_contract_for(identity, registry=registry) == contract


def test_primitive_transform_registry_validation_and_overwrite_paths() -> None:
    """Primitive transform registration should fail closed and support explicit overwrite."""

    identity = PrimitiveIdentity("scpn.quantum", "validated_batch", "1")
    first_rule = CustomDerivativeRule(
        name="validated_first",
        value_fn=lambda values: np.array([values[0]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array([tangent[0]], dtype=np.float64),
    )
    second_rule = CustomDerivativeRule(
        name="validated_second",
        value_fn=lambda values: np.array([2.0 * values[0]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array([2.0 * tangent[0]], dtype=np.float64),
    )

    def first_batch(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        del function, axes, out_axes
        return np.asarray(args[0], dtype=np.float64)

    def second_batch(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        del function, axes
        return np.moveaxis(np.asarray(args[0], dtype=np.float64), 0, out_axes)

    with pytest.raises(ValueError, match="PrimitiveIdentity"):
        PrimitiveTransformRule(  # type: ignore[arg-type]
            identity="scpn.quantum.invalid@1",
            derivative_rule=first_rule,
        )
    with pytest.raises(ValueError, match="CustomDerivativeRule"):
        PrimitiveTransformRule(  # type: ignore[arg-type]
            identity=identity,
            derivative_rule=object(),
        )
    with pytest.raises(ValueError, match="batching_rule"):
        PrimitiveTransformRule(  # type: ignore[arg-type]
            identity=identity,
            derivative_rule=first_rule,
            batching_rule=object(),
        )
    with pytest.raises(ValueError, match="lowering_rule"):
        PrimitiveTransformRule(  # type: ignore[arg-type]
            identity=identity,
            derivative_rule=first_rule,
            lowering_rule=object(),
        )
    with pytest.raises(ValueError, match="shape_rule"):
        PrimitiveTransformRule(  # type: ignore[arg-type]
            identity=identity,
            derivative_rule=first_rule,
            shape_rule=object(),
        )
    with pytest.raises(ValueError, match="dtype_rule"):
        PrimitiveTransformRule(  # type: ignore[arg-type]
            identity=identity,
            derivative_rule=first_rule,
            dtype_rule=object(),
        )
    with pytest.raises(ValueError, match="nondifferentiable_policy"):
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=first_rule,
            nondifferentiable_policy="",
        )
    with pytest.raises(ValueError, match="effect"):
        PrimitiveTransformRule(identity=identity, derivative_rule=first_rule, effect="")
    with pytest.raises(ValueError, match="metadata keys"):
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=first_rule,
            lowering_metadata={"": "scpn_diff.validated"},
        )
    with pytest.raises(ValueError, match="metadata values"):
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=first_rule,
            lowering_metadata={"mlir_op": ""},
        )

    registry = CustomDerivativeRegistry()
    transform = PrimitiveTransformRule(
        identity=identity,
        derivative_rule=first_rule,
        batching_rule=first_batch,
        lowering_metadata={"mlir_op": "scpn_diff.validated_batch"},
    )
    assert registry.register_transform(transform) is transform
    with pytest.raises(ValueError, match="already registered"):
        registry.register_transform(
            PrimitiveTransformRule(identity=identity, derivative_rule=second_rule)
        )

    replacement = PrimitiveTransformRule(
        identity=identity,
        derivative_rule=second_rule,
        batching_rule=second_batch,
        lowering_metadata={"mlir_op": "scpn_diff.validated_batch_v2"},
    )
    assert registry.register_transform(replacement, overwrite=True) is replacement
    assert registry.require(identity) is second_rule
    assert registry.batching_rule_for(identity) is second_batch
    assert (
        registry.transform_snapshot()[identity].lowering_metadata["mlir_op"]
        == "scpn_diff.validated_batch_v2"
    )


def test_primitive_batching_registry_helper_and_unregister_paths() -> None:
    """Top-level batching helpers should bind default-registry rules and unregister cleanly."""

    identity = PrimitiveIdentity("scpn.quantum", "default_helper_batch", "1")
    rule = CustomDerivativeRule(
        name="default_helper_rule",
        value_fn=lambda values: np.array([values[0] + values[1]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array([tangent[0] + tangent[1]], dtype=np.float64),
    )

    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        del function
        assert axes == (0,)
        return np.moveaxis(np.asarray(args[0], dtype=np.float64).sum(axis=1), 0, out_axes)

    with pytest.raises(ValueError, match="no custom derivative rule"):
        register_primitive_batching_rule(identity, batching_rule)
    with pytest.raises(ValueError, match="callable"):
        CustomDerivativeRegistry({identity: rule}).register_batching_rule(  # type: ignore[arg-type]
            identity,
            object(),
        )

    register_custom_derivative_rule(identity, rule)
    try:
        assert register_primitive_batching_rule(identity, batching_rule) is batching_rule
        batched = vmap(lambda row: row[0] + row[1], primitive_identity=identity)
        np.testing.assert_allclose(
            batched(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)),
            [3.0, 7.0],
            atol=1.0e-12,
        )
        assert DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.unregister(identity) is rule
        assert DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.batching_rule_for(identity) is None
        with pytest.raises(ValueError, match="no custom derivative rule"):
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.unregister(identity)
    finally:
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.lookup(identity) is not None:
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.unregister(identity)


def test_primitive_lowering_registry_helper_and_unregister_paths() -> None:
    """Top-level lowering helpers should bind default-registry executable compiler rules."""

    identity = PrimitiveIdentity("scpn.quantum", "default_helper_lower", "1")
    rule = CustomDerivativeRule(
        name="default_helper_lower_rule",
        value_fn=lambda values: np.array([values[0]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array([tangent[0]], dtype=np.float64),
    )

    def lowering_rule(lowered_rule: CustomDerivativeRule) -> object:
        return lowered_rule.name

    with pytest.raises(ValueError, match="no custom derivative rule"):
        register_primitive_lowering_rule(identity, lowering_rule)
    with pytest.raises(ValueError, match="callable"):
        CustomDerivativeRegistry({identity: rule}).register_lowering_rule(  # type: ignore[arg-type]
            identity,
            object(),
        )

    register_custom_derivative_rule(identity, rule)
    try:
        assert register_primitive_lowering_rule(identity, lowering_rule) is lowering_rule
        assert DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_lowering_rule(identity) is lowering_rule
        assert DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.unregister(identity) is rule
        assert DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.lowering_rule_for(identity) is None
    finally:
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.lookup(identity) is not None:
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.unregister(identity)


def test_vmap_rejects_missing_requested_primitive_batching_rule() -> None:
    """Explicit primitive batching dispatch should fail closed if the rule is absent."""

    identity = PrimitiveIdentity("scpn.quantum", "missing_batch", "1")
    rule = CustomDerivativeRule(
        name="missing_batch_rule",
        value_fn=lambda values: np.array([values[0]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array([tangent[0]], dtype=np.float64),
    )
    registry = CustomDerivativeRegistry({identity: rule})

    with pytest.raises(ValueError, match="no batching rule"):
        vmap(lambda row: row[0], primitive_identity=identity, registry=registry)


def test_primitive_batching_exports_are_available_from_package_root() -> None:
    """Primitive batching registry APIs should be stable root exports."""

    import scpn_quantum_control as scpn

    assert scpn.PrimitiveBatchingRule is PrimitiveBatchingRule
    assert scpn.PrimitiveContract is PrimitiveContract
    assert scpn.PrimitiveDTypeRule is PrimitiveDTypeRule
    assert scpn.PrimitiveLoweringRule is PrimitiveLoweringRule
    assert scpn.PrimitiveShapeRule is PrimitiveShapeRule
    assert scpn.PrimitiveTransformRule is PrimitiveTransformRule
    assert scpn.primitive_complete_contract_for is primitive_complete_contract_for
    assert scpn.primitive_dtype_rule_for is primitive_dtype_rule_for
    assert scpn.primitive_effect_for is primitive_effect_for
    assert scpn.primitive_contract_for is primitive_contract_for
    assert scpn.primitive_nondifferentiable_policy_for is primitive_nondifferentiable_policy_for
    assert scpn.primitive_shape_rule_for is primitive_shape_rule_for
    assert scpn.register_primitive_batching_rule is register_primitive_batching_rule
    assert scpn.register_primitive_lowering_rule is register_primitive_lowering_rule
    assert scpn.register_primitive_transform_rule is register_primitive_transform_rule
