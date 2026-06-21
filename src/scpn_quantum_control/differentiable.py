# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- native differentiable programming primitives
"""Native differentiable-programming primitives for SCPN quantum objectives.

The base layer is backend-neutral parameter-shift differentiation for scalar
objectives. Optional JAX support is exposed as an adapter without making JAX a
runtime dependency of the core package.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, NoReturn, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_batch_helpers import (
    _as_batch_parameter_array as _as_batch_parameter_array,
)
from .differentiable_batch_helpers import (
    _as_batch_vector_array as _as_batch_vector_array,
)
from .differentiable_batch_helpers import (
    _as_parameter_shift_sample_tensor as _as_parameter_shift_sample_tensor,
)
from .differentiable_consistency import (
    check_custom_derivative_consistency,
    check_parameter_shift_consistency,
)
from .differentiable_custom_derivatives import (
    batch_custom_jacobian,
    batch_custom_jvp,
    batch_custom_vjp,
    batch_value_and_custom_jacobian,
    batch_value_and_custom_jvp,
    batch_value_and_custom_vjp,
    custom_jacobian,
    custom_jvp,
    custom_vjp,
    value_and_custom_jacobian,
    value_and_custom_jvp,
    value_and_custom_vjp,
)
from .differentiable_exact_modes import (
    forward_mode_gradient,
    reverse_mode_gradient,
    value_and_forward_mode_grad,
    value_and_reverse_mode_grad,
)
from .differentiable_finite_difference import (
    batch_complex_step_gradient,
    batch_finite_difference_hvp,
    batch_finite_difference_jvp,
    batch_finite_difference_vjp,
    batch_value_and_complex_step_grad,
    batch_value_and_finite_difference_grad,
    batch_value_and_finite_difference_hvp,
    batch_value_and_finite_difference_jvp,
    batch_value_and_finite_difference_vjp,
    batch_vector_jacobian_product,
    complex_step_gradient,
    finite_difference_gradient,
    finite_difference_hessian,
    finite_difference_hvp,
    finite_difference_jacobian,
    finite_difference_jvp,
    finite_difference_vjp,
    hessian,
    jacfwd,
    jacobian,
    jacrev,
    jvp,
    value_and_complex_step_grad,
    value_and_finite_difference_grad,
    value_and_finite_difference_hessian,
    value_and_finite_difference_hvp,
    value_and_finite_difference_jacobian,
    value_and_finite_difference_jvp,
    value_and_finite_difference_vjp,
    value_and_hessian,
    value_and_jacfwd,
    value_and_jacobian,
    value_and_jacrev,
    value_and_jvp,
    value_and_vjp,
    vector_jacobian_product,
    vjp,
)
from .differentiable_fisher import (
    empirical_fisher_conjugate_gradient,
    empirical_fisher_vector_product,
    least_squares_covariance,
)
from .differentiable_gradient_descent import DifferentiableOptimizer
from .differentiable_implicit_sensitivity import (
    implicit_fixed_point_sensitivity,
    implicit_stationary_sensitivity,
)
from .differentiable_jax_adapter import (
    is_jax_autodiff_available,
    jax_value_and_grad,
)
from .differentiable_levenberg_marquardt import (
    LevenbergMarquardtOptimizer,
    custom_gauss_newton_gradient,
    custom_levenberg_marquardt_step,
    evaluate_levenberg_marquardt_step,
    gauss_newton_gradient,
    levenberg_marquardt_step,
    update_levenberg_marquardt_damping,
)
from .differentiable_natural_gradient import (
    NaturalGradientOptimizer,
    armijo_backtracking_line_search,
    natural_gradient,
    weighted_gradient_sum,
)
from .differentiable_parameter_contracts import (
    Parameter,
    ParameterBounds,
    ParameterShiftRule,
    _as_parameter_array,
    _as_real_numeric_array,
    _as_real_scalar,
    multi_frequency_parameter_shift_rule,
)
from .differentiable_parameter_shift import (
    batch_parameter_shift_gradient,
    batch_value_and_parameter_shift_grad,
    parameter_shift_gradient,
    parameter_shift_gradient_with_uncertainty,
    value_and_parameter_shift_grad,
)
from .differentiable_registered_custom import (
    registered_custom_jacobian,
    registered_custom_jvp,
    registered_custom_vjp,
)
from .differentiable_residual_weights import (
    huber_residual_weights,
    soft_l1_residual_weights,
)
from .differentiable_result_contracts import (
    DIFFERENTIABLE_RESULT_CLAIM_BOUNDARY,
    FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY,
    ArmijoLineSearchResult,
    CustomDerivativeCheckResult,
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
    LevenbergMarquardtResult,
    LevenbergMarquardtStep,
    LevenbergMarquardtTrial,
    NaturalGradientOptimizationResult,
    NaturalGradientResult,
    OptimizationResult,
    ParameterShiftSampleRecord,
    ScoreFunctionGradientResult,
    ScoreFunctionSampleRecord,
    ShotAllocationResult,
    SparseMatrixResult,
    SPSAGradientResult,
    SPSAObjectiveSample,
    SPSAProbeRecord,
    StochasticGradientResult,
    VJPResult,
    WeightedGradientResult,
)
from .differentiable_scalar_kernels import (
    DualNumber,
    ReverseNode,
    dual_cos,
    dual_exp,
    dual_log,
    dual_sin,
    reverse_cos,
    reverse_exp,
    reverse_log,
    reverse_sin,
)
from .differentiable_sparse_derivatives import (
    dense_to_sparse_matrix,
    empirical_fisher_metric,
    sparse_empirical_fisher_metric,
    sparse_hessian,
    sparse_jacobian,
)
from .differentiable_stochastic_estimators import (
    allocate_parameter_shift_shots,
    score_function_gradient_estimate,
    spsa_gradient_estimate,
)
from .differentiable_stochastic_policy import (
    STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY,
    GradientFailurePolicy,
    StochasticGradientConfidenceInterval,
    gradient_confidence_interval,
)
from .differentiable_transform_helpers import (
    _as_scalar as _as_scalar,
)
from .differentiable_transform_helpers import (
    _as_vector_output as _as_vector_output,
)
from .differentiable_transform_helpers import (
    _clip_gradient as _clip_gradient,
)
from .differentiable_transform_helpers import (
    _normalise_parameters,
)
from .differentiable_vmap import vmap
from .program_ad_adjoint import (
    ProgramADAdjointResult,
    ProgramADAdjointStep,
    program_adjoint_gradient,
    program_adjoint_result,
)
from .program_ad_adjoint import (
    _program_adjoint_input_value as _program_adjoint_input_value,
)
from .program_ad_adjoint import (
    _program_adjoint_is_ir_value as _program_adjoint_is_ir_value,
)
from .program_ad_adjoint_generation import (
    _program_adjoint_det_contributions,
    _program_adjoint_diag_contributions,
    _program_adjoint_diagflat_contributions,
    _program_adjoint_eig_eigenvalue_contributions,
    _program_adjoint_eig_eigenvector_contributions,
    _program_adjoint_eigh_eigenvalue_contributions,
    _program_adjoint_eigh_eigenvector_contributions,
    _program_adjoint_eigvals_contributions,
    _program_adjoint_eigvalsh_contributions,
    _program_adjoint_inv_contributions,
    _program_adjoint_matrix_power_contributions,
    _program_adjoint_multi_dot_contributions,
    _program_adjoint_pinv_contributions,
    _program_adjoint_solve_contributions,
    _program_adjoint_svdvals_contributions,
    _program_adjoint_trace_contributions,
)
from .program_ad_alias_analysis import (
    PROGRAM_AD_ALIAS_EFFECT_CLAIM_BOUNDARY as PROGRAM_AD_ALIAS_EFFECT_CLAIM_BOUNDARY,
)
from .program_ad_alias_analysis import (
    PROGRAM_AD_STATIC_ALIAS_LATTICE_CLAIM_BOUNDARY as PROGRAM_AD_STATIC_ALIAS_LATTICE_CLAIM_BOUNDARY,
)
from .program_ad_alias_analysis import (
    ProgramADAliasEffectAnalysis,
    ProgramADAliasSet,
    ProgramADStaticAliasLatticeComponent,
    ProgramADStaticAliasLatticeReport,
    analyze_program_ad_alias_effects,
    program_ad_static_alias_lattice_report,
)
from .program_ad_array_indexing import (
    _program_ad_array_delete_object,
    _program_ad_array_insert_layout,
    _program_ad_array_pad_layout,
    _program_ad_array_pad_mode,
    _program_ad_array_take_indices,
    _program_ad_array_take_mode,
    _register_program_ad_array_primitive_contracts,
    _require_program_ad_array_contract,
    program_ad_array_delete_derivative_rule,
    program_ad_array_getitem_derivative_rule,
    program_ad_array_insert_derivative_rule,
    program_ad_array_pad_derivative_rule,
    program_ad_array_take_along_axis_derivative_rule,
    program_ad_array_take_derivative_rule,
)
from .program_ad_array_indexing import (
    _program_ad_array_shape_of as _program_ad_array_shape_of,
)
from .program_ad_assembly_primitives import (
    _register_program_ad_assembly_primitive_contracts,
    _require_program_ad_assembly_contract,
    program_ad_assembly_append_derivative_rule,
    program_ad_assembly_block_derivative_rule,
    program_ad_assembly_broadcast_arrays_derivative_rule,
    program_ad_assembly_broadcast_to_derivative_rule,
    program_ad_assembly_column_stack_derivative_rule,
    program_ad_assembly_concatenate_derivative_rule,
    program_ad_assembly_diagonal_derivative_rule,
    program_ad_assembly_dstack_derivative_rule,
    program_ad_assembly_hstack_derivative_rule,
    program_ad_assembly_split_derivative_rule,
    program_ad_assembly_stack_derivative_rule,
    program_ad_assembly_tril_derivative_rule,
    program_ad_assembly_triu_derivative_rule,
    program_ad_assembly_vstack_derivative_rule,
)
from .program_ad_cumulative_primitives import (
    _register_program_ad_cumulative_primitive_contracts,
    _require_program_ad_cumulative_contract,
    program_ad_cumulative_cumprod_derivative_rule,
    program_ad_cumulative_cumsum_derivative_rule,
    program_ad_cumulative_diff_derivative_rule,
)
from .program_ad_effect_ir import (
    ProgramADAliasEdge,
    ProgramADControlRegion,
    ProgramADEffect,
    ProgramADEffectIR,
    ProgramADPhiNode,
    ProgramADSSAValue,
    parse_program_ad_effect_ir,
)
from .program_ad_elementwise_primitives import (
    _program_ad_elementwise_name,
    _raise_program_ad_derivative_losing_elementwise,
    _register_program_ad_elementwise_primitive_contracts,
    _require_program_ad_elementwise_contract,
    program_ad_elementwise_binary_derivative_rule,
)
from .program_ad_interpolation_primitives import (
    _normalise_interp_grid,
    _register_program_ad_interpolation_primitive_contracts,
    _require_program_ad_interpolation_contract,
    program_ad_interpolation_interp_derivative_rule,
)
from .program_ad_linalg_primitives import (
    ProgramADLinalgConditioningDiagnostic,
    _program_ad_linalg_det_cofactor_matrix,
    _program_ad_linalg_eig_eigenvector_jvp_matrix,
    _program_ad_linalg_eigh_eigenvector_jvp_matrix,
    _program_ad_linalg_normalise_rcond,
    _program_ad_linalg_pinv_jvp_matrix,
    _program_ad_linalg_pinv_value_matrix,
    _program_ad_linalg_real_simple_eig_decomposition_from_matrix,
    _program_ad_linalg_require_distinct_eigenvalues,
    _program_ad_linalg_require_distinct_positive_singular_values,
    _program_ad_linalg_require_symmetric,
    _program_ad_linalg_uplo,
    _register_program_ad_linalg_primitive_contracts,
    _require_program_ad_linalg_contract,
    diagnose_program_ad_linalg_conditioning,
    program_ad_linalg_diag_derivative_rule,
    program_ad_linalg_diagflat_derivative_rule,
    program_ad_linalg_eig_derivative_rule,
    program_ad_linalg_eigh_derivative_rule,
    program_ad_linalg_eigvals_derivative_rule,
    program_ad_linalg_eigvalsh_derivative_rule,
    program_ad_linalg_matrix_power_derivative_rule,
    program_ad_linalg_multi_dot_derivative_rule,
    program_ad_linalg_pinv_derivative_rule,
    program_ad_linalg_solve_derivative_rule,
    program_ad_linalg_svdvals_derivative_rule,
    program_ad_linalg_trace_derivative_rule,
)
from .program_ad_product_primitives import (
    _normalise_program_ad_product_tensordot_signature,
    _parse_static_einsum_subscripts,
    _register_program_ad_product_primitive_contracts,
    _require_program_ad_product_contract,
    program_ad_product_einsum_derivative_rule,
    program_ad_product_inner_derivative_rule,
    program_ad_product_matmul_derivative_rule,
    program_ad_product_outer_derivative_rule,
    program_ad_product_tensordot_derivative_rule,
)
from .program_ad_reduction_primitives import (
    _normalise_ddof,
    _normalise_order_statistic_axis,
    _normalise_order_statistic_method,
    _normalise_order_statistic_q,
    _register_program_ad_reduction_primitive_contracts,
    _require_program_ad_reduction_contract,
    _require_strict_order_statistic_values,
    program_ad_reduction_max_derivative_rule,
    program_ad_reduction_mean_derivative_rule,
    program_ad_reduction_median_derivative_rule,
    program_ad_reduction_min_derivative_rule,
    program_ad_reduction_percentile_derivative_rule,
    program_ad_reduction_prod_derivative_rule,
    program_ad_reduction_quantile_derivative_rule,
    program_ad_reduction_std_derivative_rule,
    program_ad_reduction_sum_derivative_rule,
    program_ad_reduction_var_derivative_rule,
)
from .program_ad_registry import (
    _PROGRAM_AD_SELECTION_IDENTITIES,
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    PrimitiveBatchingRule,
    PrimitiveContract,
    PrimitiveDTypeRule,
    PrimitiveIdentity,
    PrimitiveLoweringRule,
    PrimitiveShapeRule,
    PrimitiveStaticArgumentRule,
    PrimitiveTransformRule,
    ProgramADRegistryDispatchCoverageReport,
    ProgramADRegistryDispatchCoverageRow,
    custom_derivative_rule_for,
    primitive_complete_contract_for,
    primitive_contract_for,
    primitive_dtype_rule_for,
    primitive_effect_for,
    primitive_nondifferentiable_policy_for,
    primitive_shape_rule_for,
    primitive_static_argument_rule_for,
    program_ad_registry_dispatch_coverage_report,
    register_custom_derivative_rule,
    register_primitive_batching_rule,
    register_primitive_lowering_rule,
    register_primitive_transform_rule,
)
from .program_ad_rust_bridge import (
    RustProgramADInterpreterResult,
    RustProgramADValueAndGradientResult,
    interpret_program_ad_effect_ir_with_rust,
    value_and_grad_program_ad_effect_ir_with_rust,
)
from .program_ad_selection_primitives import (
    _register_program_ad_selection_primitive_contracts,
    _require_program_ad_selection_contract,
    program_ad_selection_clip_derivative_rule,
    program_ad_selection_where_derivative_rule,
)
from .program_ad_shape_transforms import (
    _register_program_ad_shape_primitive_contracts,
    _require_program_ad_shape_contract,
    program_ad_shape_atleast_1d_derivative_rule,
    program_ad_shape_atleast_2d_derivative_rule,
    program_ad_shape_atleast_3d_derivative_rule,
    program_ad_shape_expand_dims_derivative_rule,
    program_ad_shape_flip_derivative_rule,
    program_ad_shape_fliplr_derivative_rule,
    program_ad_shape_flipud_derivative_rule,
    program_ad_shape_moveaxis_derivative_rule,
    program_ad_shape_ravel_derivative_rule,
    program_ad_shape_repeat_derivative_rule,
    program_ad_shape_reshape_derivative_rule,
    program_ad_shape_roll_derivative_rule,
    program_ad_shape_rot90_derivative_rule,
    program_ad_shape_squeeze_derivative_rule,
    program_ad_shape_swapaxes_derivative_rule,
    program_ad_shape_tile_derivative_rule,
    program_ad_shape_transpose_derivative_rule,
)
from .program_ad_signal_primitives import (
    _convolve_output_window,
    _normalise_convolve_mode,
    _normalise_correlate_mode,
    _register_program_ad_signal_primitive_contracts,
    _require_program_ad_signal_contract,
    program_ad_signal_convolve_derivative_rule,
    program_ad_signal_correlate_derivative_rule,
)
from .program_ad_stencil_primitives import (
    _gradient_axis_coefficients,
    _GradientSpacing,
    _normalise_gradient_axes,
    _normalise_gradient_edge_order,
    _normalise_gradient_spacings,
    _register_program_ad_stencil_primitive_contracts,
    _require_program_ad_stencil_contract,
    program_ad_stencil_gradient_derivative_rule,
)
from .program_ad_trapezoid_primitives import (
    program_ad_reduction_trapezoid_derivative_rule,
)
from .whole_program_ad_result import (
    WholeProgramADResult,
    WholeProgramIRNode,
    WholeProgramTraceEvent,
)
from .whole_program_frontend import (
    WholeProgramBytecodeBasicBlock,
    WholeProgramBytecodeInstruction,
    WholeProgramCompilerFrontendReport,
    WholeProgramSemanticsReport,
    WholeProgramSourceBytecodeLineMap,
    WholeProgramSourceIRFeature,
    WholeProgramSourceRegion,
    WholeProgramSymbolScopeEntry,
    WholeProgramUnsupportedSemanticDiagnostic,
    _accepted_python_semantics,
    _objective_bytecode,
    _objective_source,
    _source_ir_features,
    _source_mentions_numpy,
    _unsupported_python_semantics,
    _whole_program_semantics_report,
    compile_whole_program_frontend,
)
from .whole_program_trace_runtime import (
    _trace_whole_program_objective,
    _WholeProgramTraceContext,
)

ScalarObjective = Callable[[NDArray[np.float64]], float | int | np.floating[Any]]
VectorObjective = Callable[[NDArray[np.float64]], ArrayLike]
ComplexStepObjective = Callable[[NDArray[np.complex128]], object]
CustomJVPRule = Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]
CustomVJPRule = Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]
_TraceSortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]


def whole_program_value_and_grad(
    objective: Callable[[Any], object],
    values: ArrayLike,
    parameters: Sequence[Parameter] | None = None,
    *,
    trace: bool = True,
) -> WholeProgramADResult:
    """Differentiate the executed Python/NumPy program by operator-intercepted AD.

    This is the whole-program AD boundary for differentiable Python programs
    that execute through traceable scalar values. It preserves Python execution
    semantics for loops, executed control-flow branches, local aliases, list
    mutation, and supported NumPy scalar ufuncs. Operations that would erase
    derivative information fail closed instead of falling back to finite
    differences or silently returning approximate gradients.
    """

    if not callable(objective):
        raise ValueError("whole-program objective must be callable")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    source = _objective_source(objective)
    bytecode_instructions = _objective_bytecode(objective)
    accepted_python_semantics = _accepted_python_semantics(objective, source)
    unsupported_python_semantics = _unsupported_python_semantics(objective, source)
    source_ir_features = _source_ir_features(
        source,
        accepted_python_semantics=accepted_python_semantics,
        unsupported_python_semantics=unsupported_python_semantics,
    )
    if unsupported_python_semantics:
        unsupported = ", ".join(unsupported_python_semantics)
        raise ValueError(f"unsupported whole-program AD Python semantics: {unsupported}")
    context = _WholeProgramTraceContext(
        parameter_values.size,
        scalar_factory=TraceADScalar,
    )
    traced_values: list[TraceADScalar] = []
    for index, (value, parameter) in enumerate(zip(parameter_values, parameter_meta, strict=True)):
        tangent = np.zeros(parameter_values.size, dtype=np.float64)
        if parameter.trainable:
            tangent[index] = 1.0
        traced_values.append(context.make("parameter", (parameter.name,), float(value), tangent))
    raw = objective(
        TraceADArray(
            tuple(traced_values),
            (len(traced_values),),
            context,
            tuple(range(len(traced_values))),
        )
    )
    if isinstance(raw, TraceADArray):
        if raw.shape != ():
            raise ValueError("whole-program objective must return a whole-program AD scalar")
        raw = raw.item()
    if not isinstance(raw, TraceADScalar):
        raise ValueError("whole-program objective must return a whole-program AD scalar")
    trace_events = (
        _trace_whole_program_objective(cast(ScalarObjective, objective), parameter_values)
        if trace
        else ()
    )
    semantics_report = _whole_program_semantics_report(
        bytecode_instructions=bytecode_instructions,
        source_ir_features=source_ir_features,
        trace_events=trace_events,
        source=source,
        accepted_python_semantics=accepted_python_semantics,
        unsupported_python_semantics=unsupported_python_semantics,
        numpy_observed=_source_mentions_numpy(source)
        or any(node.op in {"sin", "cos", "exp", "log"} for node in context.nodes),
        differentiation_semantics=(
            "operator-intercepted exact forward AD over the executed Python program; "
            "loops, branches, local aliasing, list mutation, closure/default/keyword "
            "calling semantics, and supported NumPy scalar ufuncs execute with "
            "derivative-carrying values, while unsupported derivative-losing or "
            "interpreter-level Python semantics fail closed"
        ),
    )
    program_ir = context.program_ir(
        source_ir_features=source_ir_features,
        bytecode_instructions=bytecode_instructions,
    )
    adjoint_result = _program_adjoint_result_from_nodes(
        nodes=tuple(context.nodes),
        output_name=raw.name,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        program_ir=program_ir,
    )
    return WholeProgramADResult(
        value=raw.primal,
        gradient=raw.tangent.copy(),
        method="whole_program_ad",
        step=0.0,
        evaluations=1 + (1 if trace else 0),
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        trace_events=trace_events,
        ir_nodes=tuple(context.nodes),
        source=source,
        control_flow_observed=semantics_report.control_flow_observed,
        numpy_observed=semantics_report.numpy_observed,
        polyglot_targets={
            "python": "operator-intercepted forward AD and supported scalar adjoint replay available",
            "mlir": "SSA/effect program AD interchange available; executable lowering blocked",
            "rust": "blocked: no Rust whole-program AD interpreter/lowering backend",
            "llvm": "blocked: no LLVM/JIT whole-program AD interpreter/lowering backend",
        },
        claim_boundary=(
            "whole-program operator-intercepted AD for executed Python scalar arithmetic, "
            "loops, local aliasing, list mutation, supported closure/default/keyword calling "
            "semantics, supported NumPy scalar ufuncs, and executed-branch control flow with "
            "deterministic SSA/effect IR evidence; unsupported interpreter-level Python "
            "constructs fail closed before execution; no finite-difference fallback and no "
            "executable Rust, LLVM, or JIT AD lowering claim"
        ),
        bytecode_instructions=bytecode_instructions,
        source_ir_features=source_ir_features,
        semantics_report=semantics_report,
        program_ir=program_ir,
        adjoint_result=adjoint_result,
    )


class _TracePredicate:
    """Primal control-flow predicate recorded by whole-program AD."""

    def __init__(self, value: bool, context: _WholeProgramTraceContext, label: str) -> None:
        self.value = bool(value)
        self.context = context
        self.label = label

    def __bool__(self) -> bool:
        tangent = np.zeros(self.context.parameter_count, dtype=np.float64)
        self.context.make(f"branch:{self.label}:{self.value}", (), float(self.value), tangent)
        return self.value


class TraceADPredicateArray:
    """Derivative-safe vector of primal predicates for piecewise whole-program AD."""

    def __init__(
        self,
        predicates: tuple[_TracePredicate, ...],
        shape: tuple[int, ...],
        context: _WholeProgramTraceContext,
    ) -> None:
        if int(np.prod(shape)) != len(predicates):
            raise ValueError("predicate array shape must match predicate count")
        if any(predicate.context is not context for predicate in predicates):
            raise ValueError("predicate array items must belong to the same trace")
        self.predicates = predicates
        self.shape = shape
        self.context = context

    def __bool__(self) -> bool:
        if self.shape != () or len(self.predicates) != 1:
            raise ValueError("whole-program AD vector predicates cannot be used as scalar bools")
        return bool(self.predicates[0])


class TraceADScalar:
    """Operator-intercepted scalar for exact executed-path whole-program AD."""

    __array_priority__ = 1000.0

    def __init__(
        self,
        primal: float,
        tangent: NDArray[np.float64],
        context: _WholeProgramTraceContext,
        name: str,
    ) -> None:
        self.primal = _as_real_scalar("whole-program AD primal", primal)
        self.tangent = _as_real_numeric_array("whole-program AD tangent", tangent)
        if self.tangent.ndim != 1:
            raise ValueError("whole-program AD tangent must be one-dimensional")
        self.context = context
        self.name = name

    def __float__(self) -> float:
        raise ValueError(
            "whole-program AD scalar cannot be converted to float without losing derivatives"
        )

    def _coerce(self, other: object) -> TraceADScalar:
        if isinstance(other, TraceADScalar):
            if other.context is not self.context:
                raise ValueError("whole-program AD scalars belong to different traces")
            return other
        tangent = np.zeros(self.context.parameter_count, dtype=np.float64)
        return TraceADScalar(
            _as_real_scalar("whole-program AD constant", other), tangent, self.context, repr(other)
        )

    def _binary(self, op: str, other: object) -> TraceADScalar:
        rhs = self._coerce(other)
        if op == "add":
            return self.context.make(
                op, (self.name, rhs.name), self.primal + rhs.primal, self.tangent + rhs.tangent
            )
        if op == "sub":
            return self.context.make(
                op, (self.name, rhs.name), self.primal - rhs.primal, self.tangent - rhs.tangent
            )
        if op == "mul":
            return self.context.make(
                op,
                (self.name, rhs.name),
                self.primal * rhs.primal,
                self.tangent * rhs.primal + self.primal * rhs.tangent,
            )
        if op == "div":
            if rhs.primal == 0.0:
                raise ValueError("whole-program AD division denominator must be non-zero")
            return self.context.make(
                op,
                (self.name, rhs.name),
                self.primal / rhs.primal,
                (self.tangent * rhs.primal - self.primal * rhs.tangent) / rhs.primal**2,
            )
        if op == "pow":
            if self.primal <= 0.0 and np.any(rhs.tangent != 0.0):
                raise ValueError("whole-program AD variable exponent requires positive base")
            primal = self.primal**rhs.primal
            if np.all(rhs.tangent == 0.0):
                tangent = rhs.primal * self.primal ** (rhs.primal - 1.0) * self.tangent
            else:
                tangent = primal * (
                    rhs.tangent * float(np.log(self.primal))
                    + rhs.primal * self.tangent / self.primal
                )
            return self.context.make(op, (self.name, rhs.name), primal, tangent)
        raise ValueError(f"unsupported whole-program AD binary op {op}")

    def __add__(self, other: object) -> TraceADScalar:
        return self._binary("add", other)

    def __radd__(self, other: object) -> TraceADScalar:
        return self.__add__(other)

    def __sub__(self, other: object) -> TraceADScalar:
        return self._binary("sub", other)

    def __rsub__(self, other: object) -> TraceADScalar:
        return self._coerce(other)._binary("sub", self)

    def __mul__(self, other: object) -> TraceADScalar:
        return self._binary("mul", other)

    def __rmul__(self, other: object) -> TraceADScalar:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> TraceADScalar:
        return self._binary("div", other)

    def __rtruediv__(self, other: object) -> TraceADScalar:
        return self._coerce(other)._binary("div", self)

    def __pow__(self, other: object) -> TraceADScalar:
        return self._binary("pow", other)

    def __rpow__(self, other: object) -> TraceADScalar:
        return self._coerce(other)._binary("pow", self)

    def __neg__(self) -> TraceADScalar:
        return self.context.make("neg", (self.name,), -self.primal, -self.tangent)

    def __abs__(self) -> TraceADScalar:
        result = _apply_trace_ufunc(np.absolute, (self,), self.context)
        if not isinstance(result, TraceADScalar):
            raise ValueError("whole-program AD absolute value returned a non-scalar result")
        return result

    def _compare(self, op: str, other: object) -> _TracePredicate:
        rhs = self._coerce(other)
        if op in {"gt", "ge", "lt", "le"} and self.primal == rhs.primal:
            raise ValueError(
                "whole-program AD ordering predicate is non-differentiable at equality"
            )
        comparisons = {
            "gt": self.primal > rhs.primal,
            "ge": self.primal >= rhs.primal,
            "lt": self.primal < rhs.primal,
            "le": self.primal <= rhs.primal,
            "eq": self.primal == rhs.primal,
            "ne": self.primal != rhs.primal,
        }
        return _TracePredicate(comparisons[op], self.context, f"{self.name}:{op}:{rhs.name}")

    def __gt__(self, other: object) -> _TracePredicate:
        return self._compare("gt", other)

    def __ge__(self, other: object) -> _TracePredicate:
        return self._compare("ge", other)

    def __lt__(self, other: object) -> _TracePredicate:
        return self._compare("lt", other)

    def __le__(self, other: object) -> _TracePredicate:
        return self._compare("le", other)

    def __eq__(self, other: object) -> _TracePredicate:  # type: ignore[override]
        rhs = self._coerce(other)
        return _TracePredicate(
            self.primal == rhs.primal, self.context, f"{self.name}:eq:{rhs.name}"
        )

    def __ne__(self, other: object) -> _TracePredicate:  # type: ignore[override]
        rhs = self._coerce(other)
        return _TracePredicate(
            self.primal != rhs.primal, self.context, f"{self.name}:ne:{rhs.name}"
        )

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object
    ) -> TraceADScalar:
        if method != "__call__" or kwargs:
            raise ValueError("whole-program AD supports only direct NumPy scalar ufunc calls")
        result = _apply_trace_ufunc(ufunc, tuple(inputs), self.context)
        if not isinstance(result, TraceADScalar):
            raise ValueError("whole-program AD scalar ufunc returned a non-scalar result")
        return result


class TraceADArray:
    """Derivative-carrying one-dimensional array for whole-program AD."""

    __array_priority__ = 1000.0

    def __init__(
        self,
        items: tuple[TraceADScalar, ...],
        shape: tuple[int, ...],
        context: _WholeProgramTraceContext,
        source_indices: tuple[int | None, ...] | None = None,
    ) -> None:
        if not shape:
            if len(items) != 1:
                raise ValueError("scalar TraceADArray requires exactly one item")
        elif int(np.prod(shape)) != len(items):
            raise ValueError("TraceADArray shape must match item count")
        if any(item.context is not context for item in items):
            raise ValueError("TraceADArray items must belong to the same trace")
        if source_indices is not None and len(source_indices) != len(items):
            raise ValueError("TraceADArray source indices must match item count")
        if source_indices is not None and any(
            source_index is not None and source_index < 0 for source_index in source_indices
        ):
            raise ValueError("TraceADArray source indices must be non-negative or None")
        self._items = list(items)
        self.shape = shape
        self.context = context
        self._source_indices = source_indices

    @property
    def ndim(self) -> int:
        """Return the rank of the derivative-carrying array."""

        return len(self.shape)

    @property
    def size(self) -> int:
        """Return the total number of derivative-carrying elements."""

        return len(self._items)

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("scalar TraceADArray has no len()")
        return self.shape[0]

    def __iter__(self) -> object:
        if self.ndim == 1:
            return iter(self._items)
        if self.ndim == 2:
            rows, cols = self.shape
            return iter(
                TraceADArray(
                    tuple(self._items[row * cols : (row + 1) * cols]),
                    (cols,),
                    self.context,
                    None
                    if self._source_indices is None
                    else tuple(self._source_indices[row * cols : (row + 1) * cols]),
                )
                for row in range(rows)
            )
        raise ValueError("whole-program AD array iteration supports arrays with rank <= 2")

    def __array__(self, dtype: object = None) -> object:
        del dtype
        raise ValueError(
            "whole-program AD array cannot be converted to a NumPy ndarray without losing derivatives"
        )

    def item(self) -> TraceADScalar:
        """Return the only scalar element, failing closed for non-scalar arrays."""

        if self.size != 1:
            raise ValueError("TraceADArray.item requires exactly one element")
        return self._items[0]

    def copy(self) -> TraceADArray:
        """Return a derivative-preserving shallow array copy."""

        return TraceADArray(tuple(self._items), self.shape, self.context, self._source_indices)

    def reshape(self, *shape: int | tuple[int, ...]) -> TraceADArray:
        """Return a derivative-preserving reshaped array view."""

        if len(shape) == 1 and isinstance(shape[0], tuple):
            raw_target: object = shape[0]
        else:
            raw_target = shape
        _require_program_ad_shape_contract("reshape", (self, raw_target))
        target = _normalise_trace_reshape_shape(raw_target, self.size)
        items = tuple(self._items)
        source_indices = _trace_array_source_indices(self)
        self.context.record_array_view_aliases("reshape", source_indices, items)
        return TraceADArray(items, target, self.context, source_indices)

    def ravel(self) -> TraceADArray:
        """Return a flat view-preserving program AD array."""

        _require_program_ad_shape_contract("ravel", (self,))
        items = tuple(self._items)
        source_indices = _trace_array_source_indices(self)
        self.context.record_array_view_aliases("ravel", source_indices, items)
        return TraceADArray(items, (self.size,), self.context, source_indices)

    def flatten(self) -> TraceADArray:
        """Return a flat copy-equivalent program AD array."""

        return self.ravel()

    def repeat(self, repeats: object, axis: int | None = None) -> TraceADArray:
        """Return a derivative-preserving array with repeated elements."""

        return _trace_repeat(self, repeats=repeats, axis=axis)

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> TraceADArray:
        """Return a derivative-preserving array with singleton axes removed."""

        return _trace_squeeze(self, axis=axis)

    def expand_dims(self, axis: int | tuple[int, ...]) -> TraceADArray:
        """Return a derivative-preserving array with singleton axes inserted."""

        return _trace_expand_dims(self, axis=axis)

    def swapaxes(self, axis1: int, axis2: int) -> TraceADArray:
        """Return a derivative-preserving array with two axes exchanged."""

        return _trace_swapaxes(self, axis1=axis1, axis2=axis2)

    @property
    def T(self) -> TraceADArray:
        """Return the NumPy-compatible reversed-axis transpose."""

        if self.ndim < 2:
            return self.copy()
        return _trace_transpose(self, self.context)

    def sum(self, axis: int | None = None) -> TraceADScalar | TraceADArray:
        """Return a derivative-preserving sum over all elements or one axis."""

        return _trace_array_sum(self, axis=axis)

    def cumsum(self, axis: int | None = None) -> TraceADArray:
        """Return a derivative-preserving cumulative sum."""

        return _trace_cumsum(self, axis=axis)

    def prod(self, axis: int | None = None) -> TraceADScalar | TraceADArray:
        """Return a derivative-preserving product over all elements or one axis."""

        return _trace_array_prod(self, axis=axis)

    def cumprod(self, axis: int | None = None) -> TraceADArray:
        """Return a derivative-preserving cumulative product."""

        return _trace_cumprod(self, axis=axis)

    def mean(self, axis: int | None = None) -> TraceADScalar | TraceADArray:
        """Return a derivative-preserving arithmetic mean."""

        _require_program_ad_reduction_contract("mean", (self, axis))
        result = _trace_array_sum(self, axis=axis)
        divisor = (
            self.size if axis is None else self.shape[_normalise_axis("axis", axis, self.ndim)]
        )
        return (
            result / float(divisor)
            if isinstance(result, TraceADScalar)
            else result / float(divisor)
        )

    def var(self, axis: int | None = None, ddof: int = 0) -> TraceADScalar | TraceADArray:
        """Return a derivative-preserving variance with NumPy-compatible ddof."""

        _require_program_ad_reduction_contract("var", (self, axis, ddof))
        return _trace_variance(self, axis=axis, ddof=ddof)

    def std(self, axis: int | None = None, ddof: int = 0) -> TraceADScalar | TraceADArray:
        """Return a derivative-preserving standard deviation."""

        _require_program_ad_reduction_contract("std", (self, axis, ddof))
        return _trace_std(self, axis=axis, ddof=ddof)

    def max(self, axis: int | None = None) -> TraceADScalar | TraceADArray:
        """Return a derivative-preserving maximum with tie-safe semantics."""

        _require_program_ad_reduction_contract("max", (self, axis))
        return _trace_extreme(self, axis=axis, choose_max=True)

    def min(self, axis: int | None = None) -> TraceADScalar | TraceADArray:
        """Return a derivative-preserving minimum with tie-safe semantics."""

        _require_program_ad_reduction_contract("min", (self, axis))
        return _trace_extreme(self, axis=axis, choose_max=False)

    def take(
        self,
        indices: object,
        axis: int | None = None,
        mode: str = "raise",
    ) -> TraceADScalar | TraceADArray:
        """Return derivative-preserving positional elements with fail-closed modes."""

        return _trace_take(self, indices, axis=axis, mode=mode)

    def argmax(self, axis: int | None = None) -> NoReturn:
        """Reject nondifferentiable maximum-index selection."""

        _raise_index_selection_boundary("argmax", (self, axis))

    def argmin(self, axis: int | None = None) -> NoReturn:
        """Reject nondifferentiable minimum-index selection."""

        _raise_index_selection_boundary("argmin", (self, axis))

    def __getitem__(self, index: object) -> TraceADScalar | TraceADArray:
        return _trace_array_getitem(self, index)

    def __setitem__(self, index: object, value: object) -> None:
        if self.ndim > 2:
            raise ValueError("whole-program AD array mutation supports arrays with rank <= 2")
        if isinstance(index, slice):
            if self.ndim != 1:
                raise ValueError("whole-program AD slice mutation supports rank-1 arrays")
            targets = tuple(range(self.size))[index]
            if not targets:
                return
            if isinstance(value, TraceADArray):
                array_value = _coerce_trace_array(value, self.context)
                if array_value.shape == ():
                    scalars = (array_value.item(),) * len(targets)
                elif array_value.size == len(targets):
                    scalars = tuple(array_value._items)
                else:
                    raise ValueError(
                        "whole-program AD slice mutation value length must match target length"
                    )
            elif isinstance(value, TraceADScalar):
                scalars = (_coerce_trace_scalar(value, self.context),) * len(targets)
            else:
                raw_value = np.asarray(value)
                if raw_value.shape == ():
                    scalars = (_coerce_trace_scalar(float(raw_value), self.context),) * len(
                        targets
                    )
                elif raw_value.dtype.kind == "O" and all(
                    isinstance(item, TraceADScalar) for item in raw_value.reshape(-1)
                ):
                    flat_values = tuple(
                        _coerce_trace_scalar(item, self.context) for item in raw_value.reshape(-1)
                    )
                    if len(flat_values) != len(targets):
                        raise ValueError(
                            "whole-program AD slice mutation value length must match target length"
                        )
                    scalars = flat_values
                else:
                    array_value = _coerce_trace_array(value, self.context)
                    if array_value.size != len(targets):
                        raise ValueError(
                            "whole-program AD slice mutation value length must match target length"
                        )
                    scalars = tuple(array_value._items)
            for flat_index, scalar in zip(targets, scalars, strict=True):
                self._set_flat_item(int(flat_index), scalar)
            return
        if isinstance(index, tuple):
            if self.ndim != 2 or len(index) != 2:
                raise ValueError("whole-program AD matrix mutation expects two integer indices")
            row, col = int(index[0]), int(index[1])
            rows, cols = self.shape
            if row < 0:
                row += rows
            if col < 0:
                col += cols
            flat_index = row * cols + col
        elif isinstance(index, (int, np.integer)):
            flat_index = int(index)
        else:
            raise ValueError("whole-program AD array mutation supports integer or slice indices")
        scalar = _coerce_trace_scalar(value, self.context)
        self._set_flat_item(flat_index, scalar)

    def _set_flat_item(self, flat_index: int, scalar: TraceADScalar) -> None:
        """Assign one flattened element and emit deterministic mutation metadata."""

        if flat_index < 0:
            flat_index += self.size
        if flat_index < 0 or flat_index >= self.size:
            raise ValueError("whole-program AD array mutation index out of bounds")
        source_index = None if self._source_indices is None else self._source_indices[flat_index]
        mutation_target = f"%array[{flat_index if source_index is None else source_index}]"
        self.context.make(
            "mutation:setitem",
            (mutation_target, scalar.name),
            scalar.primal,
            scalar.tangent,
        )
        self._items[flat_index] = scalar
        if self._source_indices is not None:
            source_indices = list(self._source_indices)
            source_indices[flat_index] = None
            self._source_indices = tuple(source_indices)

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object
    ) -> TraceADScalar | TraceADArray:
        if method != "__call__" or kwargs:
            raise ValueError("whole-program AD supports only direct NumPy array ufunc calls")
        return _apply_trace_ufunc(ufunc, tuple(inputs), self.context)

    def __array_function__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> TraceADScalar | TraceADArray | list[TraceADArray] | tuple[TraceADArray, TraceADArray]:
        del types
        if func is np.sum:
            if len(args) != 1 or kwargs.keys() - {"axis"}:
                raise ValueError("whole-program AD np.sum supports one array and optional axis")
            return _trace_array_sum(
                _coerce_trace_array(args[0], self.context),
                axis=cast(int | None, kwargs.get("axis")),
            )
        if func is np.cumsum:
            if len(args) != 1 or kwargs.keys() - {"axis"}:
                raise ValueError("program AD np.cumsum supports one array and optional axis")
            return _trace_cumsum(
                _coerce_trace_array(args[0], self.context),
                axis=cast(int | None, kwargs.get("axis")),
            )
        if func is np.prod:
            if len(args) != 1 or kwargs.keys() - {"axis"}:
                raise ValueError("program AD np.prod supports one array and optional axis")
            return _trace_array_prod(
                _coerce_trace_array(args[0], self.context),
                axis=cast(int | None, kwargs.get("axis")),
            )
        if func is np.cumprod:
            if len(args) != 1 or kwargs.keys() - {"axis"}:
                raise ValueError("program AD np.cumprod supports one array and optional axis")
            return _trace_cumprod(
                _coerce_trace_array(args[0], self.context),
                axis=cast(int | None, kwargs.get("axis")),
            )
        if func is np.diff:
            if "prepend" in kwargs or "append" in kwargs:
                raise ValueError("program AD np.diff does not support prepend/append")
            if len(args) < 1 or len(args) > 3 or kwargs.keys() - {"n", "axis"}:
                raise ValueError("program AD np.diff supports array, n, and axis")
            n_value = args[1] if len(args) >= 2 else kwargs.get("n", 1)
            axis_value = args[2] if len(args) >= 3 else kwargs.get("axis", -1)
            return _trace_diff(
                _coerce_trace_array(args[0], self.context),
                n=n_value,
                axis=cast(int, axis_value),
            )
        if func is np.gradient:
            if len(args) < 1 or kwargs.keys() - {"axis", "edge_order"}:
                raise ValueError(
                    "program AD np.gradient supports array, spacing, axis, and edge_order"
                )
            return _trace_gradient(
                _coerce_trace_array(args[0], self.context),
                spacings=args[1:],
                axis=kwargs.get("axis"),
                edge_order=kwargs.get("edge_order", 1),
            )
        if func is np.interp:
            if len(args) < 3 or len(args) > 6 or kwargs.keys() - {"left", "right", "period"}:
                raise ValueError(
                    "program AD np.interp supports x, xp, fp, left, right, and period"
                )
            if len(args) >= 4 and "left" in kwargs:
                raise ValueError("program AD np.interp left must be supplied once")
            if len(args) >= 5 and "right" in kwargs:
                raise ValueError("program AD np.interp right must be supplied once")
            if len(args) >= 6 and "period" in kwargs:
                raise ValueError("program AD np.interp period must be supplied once")
            return _trace_interp(
                args[0],
                args[1],
                args[2],
                left=args[3] if len(args) >= 4 else kwargs.get("left"),
                right=args[4] if len(args) >= 5 else kwargs.get("right"),
                period=args[5] if len(args) >= 6 else kwargs.get("period"),
                context=self.context,
            )
        if func is np.convolve:
            if len(args) < 2 or len(args) > 3 or kwargs.keys() - {"mode"}:
                raise ValueError("program AD np.convolve supports two operands and mode")
            if len(args) == 3 and "mode" in kwargs:
                raise ValueError("program AD np.convolve mode must be supplied once")
            return _trace_convolve(
                args[0],
                args[1],
                context=self.context,
                mode=args[2] if len(args) == 3 else kwargs.get("mode", "full"),
            )
        if func is np.correlate:
            if len(args) < 2 or len(args) > 3 or kwargs.keys() - {"mode"}:
                raise ValueError("program AD np.correlate supports two operands and mode")
            if len(args) == 3 and "mode" in kwargs:
                raise ValueError("program AD np.correlate mode must be supplied once")
            return _trace_correlate(
                args[0],
                args[1],
                context=self.context,
                mode=args[2] if len(args) == 3 else kwargs.get("mode", "valid"),
            )
        if func in {np.zeros_like, np.ones_like}:
            if len(args) != 1:
                raise ValueError("program AD like-constructors require one reference array")
            _validate_trace_like_constructor_kwargs(kwargs)
            if func is np.zeros_like:
                return _trace_like_constant(args[0], 0.0, self.context, name="zeros_like")
            return _trace_like_constant(args[0], 1.0, self.context, name="ones_like")
        if func is np.full_like:
            if len(args) != 2:
                raise ValueError("program AD full_like requires reference array and fill value")
            _validate_trace_like_constructor_kwargs(kwargs)
            return _trace_like_constant(args[0], args[1], self.context, name="full_like")
        if func is np.mean:
            if len(args) != 1 or kwargs.keys() - {"axis"}:
                raise ValueError("whole-program AD np.mean supports one array and optional axis")
            return _coerce_trace_array(args[0], self.context).mean(
                axis=cast(int | None, kwargs.get("axis"))
            )
        if func is np.trapezoid or func is getattr(np, "trapz", None):
            if len(args) < 1 or len(args) > 2 or kwargs.keys() - {"x", "dx", "axis"}:
                raise ValueError("program AD np.trapezoid supports y, x, dx, and axis")
            if len(args) == 2 and "x" in kwargs:
                raise ValueError("program AD np.trapezoid x must be supplied once")
            x_value = args[1] if len(args) == 2 else kwargs.get("x")
            return _trace_trapezoid(
                _coerce_trace_array(args[0], self.context),
                x=x_value,
                dx=kwargs.get("dx", 1.0),
                axis=kwargs.get("axis", -1),
            )
        if func is np.var:
            if len(args) != 1 or kwargs.keys() - {"axis", "ddof"}:
                raise ValueError("program AD np.var supports one array, axis, and ddof")
            var_axis = cast(int | None, kwargs.get("axis"))
            var_ddof = kwargs.get("ddof", 0)
            _require_program_ad_reduction_contract(
                "var",
                (_coerce_trace_array(args[0], self.context), var_axis, var_ddof),
            )
            return _trace_variance(
                _coerce_trace_array(args[0], self.context),
                axis=var_axis,
                ddof=var_ddof,
            )
        if func is np.std:
            if len(args) != 1 or kwargs.keys() - {"axis", "ddof"}:
                raise ValueError("program AD np.std supports one array, axis, and ddof")
            std_axis = cast(int | None, kwargs.get("axis"))
            std_ddof = kwargs.get("ddof", 0)
            _require_program_ad_reduction_contract(
                "std",
                (_coerce_trace_array(args[0], self.context), std_axis, std_ddof),
            )
            return _trace_std(
                _coerce_trace_array(args[0], self.context),
                axis=std_axis,
                ddof=std_ddof,
            )
        if func is np.median:
            if len(args) < 1 or len(args) > 2 or kwargs.keys() - {"axis"}:
                raise ValueError("program AD np.median supports one array and optional axis")
            if len(args) == 2 and "axis" in kwargs:
                raise ValueError("program AD np.median axis must be supplied once")
            median_axis = args[1] if len(args) == 2 else kwargs.get("axis")
            _require_program_ad_reduction_contract(
                "median",
                (_coerce_trace_array(args[0], self.context), median_axis),
            )
            return _trace_order_statistic(
                _coerce_trace_array(args[0], self.context),
                q=0.5,
                axis=median_axis,
                op_name="np.median",
            )
        if func in {np.quantile, np.percentile}:
            if (
                len(args) < 2
                or len(args) > 3
                or kwargs.keys()
                - {
                    "axis",
                    "method",
                    "interpolation",
                }
            ):
                raise ValueError(
                    f"program AD np.{func.__name__} supports array, scalar q, axis, and method"
                )
            if len(args) == 3 and "axis" in kwargs:
                raise ValueError(f"program AD np.{func.__name__} axis must be supplied once")
            if "method" in kwargs and "interpolation" in kwargs:
                raise ValueError(f"program AD np.{func.__name__} method must be supplied once")
            method = kwargs.get("method", kwargs.get("interpolation", "linear"))
            order_statistic_axis = args[2] if len(args) == 3 else kwargs.get("axis")
            order_statistic_q = _normalise_order_statistic_q(
                args[1],
                percentile=func is np.percentile,
            )
            _require_program_ad_reduction_contract(
                func.__name__,
                (
                    _coerce_trace_array(args[0], self.context),
                    args[1],
                    order_statistic_axis,
                    method,
                ),
            )
            return _trace_order_statistic(
                _coerce_trace_array(args[0], self.context),
                q=order_statistic_q,
                axis=order_statistic_axis,
                method=method,
                op_name=f"np.{func.__name__}",
            )
        if func is np.max:
            if len(args) != 1 or kwargs.keys() - {"axis"}:
                raise ValueError("program AD np.max supports one array and optional axis")
            max_axis = cast(int | None, kwargs.get("axis"))
            _require_program_ad_reduction_contract(
                "max",
                (_coerce_trace_array(args[0], self.context), max_axis),
            )
            return _trace_extreme(
                _coerce_trace_array(args[0], self.context),
                axis=max_axis,
                choose_max=True,
            )
        if func is np.min:
            if len(args) != 1 or kwargs.keys() - {"axis"}:
                raise ValueError("program AD np.min supports one array and optional axis")
            min_axis = cast(int | None, kwargs.get("axis"))
            _require_program_ad_reduction_contract(
                "min",
                (_coerce_trace_array(args[0], self.context), min_axis),
            )
            return _trace_extreme(
                _coerce_trace_array(args[0], self.context),
                axis=min_axis,
                choose_max=False,
            )
        if func is np.dot:
            if len(args) != 2 or kwargs:
                raise ValueError("whole-program AD np.dot supports two operands")
            return _trace_dot(args[0], args[1], self.context)
        if func is np.vdot:
            if len(args) != 2 or kwargs:
                raise ValueError("program AD np.vdot supports two operands")
            return _trace_vdot(args[0], args[1], self.context)
        if func is np.inner:
            if len(args) != 2 or kwargs:
                raise ValueError("whole-program AD np.inner supports two operands")
            return _trace_inner(args[0], args[1], self.context)
        if func is np.outer:
            if len(args) != 2 or kwargs:
                raise ValueError("whole-program AD np.outer supports two operands")
            return _trace_outer(args[0], args[1], self.context)
        if func is np.tensordot:
            if len(args) < 2 or len(args) > 3 or kwargs.keys() - {"axes"}:
                raise ValueError("whole-program AD np.tensordot supports two operands and axes")
            axes = args[2] if len(args) == 3 else kwargs.get("axes", 2)
            return _trace_tensordot(args[0], args[1], self.context, axes=axes)
        if func is np.einsum:
            if len(args) < 2 or kwargs:
                raise ValueError("whole-program AD np.einsum supports explicit operands only")
            if not isinstance(args[0], str):
                raise ValueError("whole-program AD np.einsum requires a string subscript")
            return _trace_einsum(args[0], args[1:], self.context)
        if func is np.matmul:
            if len(args) != 2 or kwargs:
                raise ValueError("whole-program AD np.matmul supports two operands")
            return _trace_matmul(args[0], args[1], self.context)
        if func is np.where:
            if len(args) != 3 or kwargs:
                raise ValueError("whole-program AD np.where supports condition, x, and y")
            return _trace_where(args[0], args[1], args[2], self.context)
        if func is np.select:
            if len(args) < 2 or len(args) > 3 or kwargs.keys() - {"default"}:
                raise ValueError("program AD np.select supports condlist, choicelist, and default")
            if len(args) == 3 and "default" in kwargs:
                raise ValueError("program AD np.select default must be supplied once")
            default = args[2] if len(args) == 3 else kwargs.get("default", 0.0)
            return _trace_select(args[0], args[1], default, self.context)
        if func is np.piecewise:
            if len(args) != 3 or kwargs:
                raise ValueError("program AD np.piecewise supports array, condlist, and funclist")
            return _trace_piecewise(args[0], args[1], args[2], self.context)
        if func is np.choose:
            if len(args) != 2 or kwargs.keys() - {"mode"}:
                raise ValueError("program AD np.choose supports selector, choices, and mode")
            return _trace_choose(
                args[0],
                args[1],
                self.context,
                mode=cast(str, kwargs.get("mode", "raise")),
            )
        if func is np.compress:
            if len(args) < 2 or len(args) > 3 or kwargs.keys() - {"axis"}:
                raise ValueError("program AD np.compress supports condition, array, and axis")
            if len(args) == 3 and "axis" in kwargs:
                raise ValueError("program AD np.compress axis must be supplied once")
            axis = args[2] if len(args) == 3 else kwargs.get("axis")
            return _trace_compress(
                args[0],
                _coerce_trace_array(args[1], self.context),
                axis=axis,
            )
        if func is np.extract:
            if len(args) != 2 or kwargs:
                raise ValueError("program AD np.extract supports condition and array")
            return _trace_extract(
                args[0],
                _coerce_trace_array(args[1], self.context),
            )
        if func is np.reshape:
            if len(args) != 2 or kwargs:
                raise ValueError("whole-program AD np.reshape supports array and shape")
            shape = args[1]
            if isinstance(shape, int):
                return _coerce_trace_array(args[0], self.context).reshape(shape)
            return _coerce_trace_array(args[0], self.context).reshape(cast(tuple[int, ...], shape))
        if func is np.broadcast_to:
            if len(args) != 2 or kwargs.keys() - {"subok"}:
                raise ValueError("program AD np.broadcast_to supports array, shape, and subok")
            if kwargs.get("subok", False):
                raise ValueError("program AD np.broadcast_to does not support subok")
            trace_array = _coerce_trace_array(args[0], self.context)
            output_shape = _normalise_trace_broadcast_shape(args[1])
            _require_program_ad_assembly_contract("broadcast_to", (trace_array, output_shape))
            return _broadcast_trace_array(trace_array, output_shape, self.context)
        if func is np.broadcast_arrays:
            if not args or kwargs.keys() - {"subok"}:
                raise ValueError("program AD np.broadcast_arrays supports operands and subok")
            if kwargs.get("subok", False):
                raise ValueError("program AD np.broadcast_arrays does not support subok")
            return _trace_broadcast_arrays(args, self.context)
        if func is np.ravel:
            if len(args) != 1 or kwargs:
                raise ValueError("whole-program AD np.ravel supports one array")
            return _coerce_trace_array(args[0], self.context).ravel()
        if func in {np.atleast_1d, np.atleast_2d, np.atleast_3d}:
            if not args or kwargs:
                raise ValueError("program AD atleast transforms support positional arrays only")
            target_rank = 1 if func is np.atleast_1d else 2 if func is np.atleast_2d else 3
            transformed = tuple(
                _trace_atleast_nd(_coerce_trace_array(item, self.context), rank=target_rank)
                for item in args
            )
            return transformed[0] if len(transformed) == 1 else list(transformed)
        if func is np.squeeze:
            if len(args) != 1 or kwargs.keys() - {"axis"}:
                raise ValueError("program AD np.squeeze supports one array and optional axis")
            return _trace_squeeze(
                _coerce_trace_array(args[0], self.context),
                axis=cast(int | tuple[int, ...] | None, kwargs.get("axis")),
            )
        if func is np.expand_dims:
            if len(args) == 2 and not kwargs:
                axis = args[1]
            elif len(args) == 1 and set(kwargs) == {"axis"}:
                axis = kwargs["axis"]
            else:
                raise ValueError("program AD np.expand_dims supports one array and axis")
            return _trace_expand_dims(
                _coerce_trace_array(args[0], self.context),
                axis=cast(int | tuple[int, ...], axis),
            )
        if func is np.swapaxes:
            if len(args) == 3 and not kwargs:
                axis1 = args[1]
                axis2 = args[2]
            elif len(args) == 1 and set(kwargs) == {"axis1", "axis2"}:
                axis1 = kwargs["axis1"]
                axis2 = kwargs["axis2"]
            else:
                raise ValueError("program AD np.swapaxes supports array, axis1, and axis2")
            return _trace_swapaxes(
                _coerce_trace_array(args[0], self.context),
                axis1=cast(int, axis1),
                axis2=cast(int, axis2),
            )
        if func is np.moveaxis:
            if len(args) == 3 and not kwargs:
                source = args[1]
                destination = args[2]
            elif len(args) == 1 and set(kwargs) == {"source", "destination"}:
                source = kwargs["source"]
                destination = kwargs["destination"]
            else:
                raise ValueError("program AD np.moveaxis supports array, source, and destination")
            return _trace_moveaxis(
                _coerce_trace_array(args[0], self.context),
                source=cast(int | tuple[int, ...], source),
                destination=cast(int | tuple[int, ...], destination),
            )
        if func is np.repeat:
            if len(args) == 2 and kwargs.keys() <= {"axis"}:
                repeats = args[1]
                axis = kwargs.get("axis")
            elif len(args) == 3 and not kwargs:
                repeats = args[1]
                axis = args[2]
            elif len(args) == 1 and "repeats" in kwargs and kwargs.keys() <= {"repeats", "axis"}:
                repeats = kwargs["repeats"]
                axis = kwargs.get("axis")
            else:
                raise ValueError("program AD np.repeat supports array, repeats, and optional axis")
            return _trace_repeat(
                _coerce_trace_array(args[0], self.context),
                repeats=repeats,
                axis=cast(int | None, axis),
            )
        if func is np.tile:
            if len(args) == 2 and not kwargs:
                reps = args[1]
            elif len(args) == 1 and set(kwargs) == {"reps"}:
                reps = kwargs["reps"]
            else:
                raise ValueError("program AD np.tile supports array and reps")
            return _trace_tile(_coerce_trace_array(args[0], self.context), reps=reps)
        if func is np.roll:
            if len(args) == 2 and kwargs.keys() <= {"axis"}:
                shift = args[1]
                axis = kwargs.get("axis")
            elif len(args) == 3 and not kwargs:
                shift = args[1]
                axis = args[2]
            elif len(args) == 1 and "shift" in kwargs and kwargs.keys() <= {"shift", "axis"}:
                shift = kwargs["shift"]
                axis = kwargs.get("axis")
            else:
                raise ValueError("program AD np.roll supports array, shift, and optional axis")
            return _trace_roll(
                _coerce_trace_array(args[0], self.context),
                shift=shift,
                axis=axis,
            )
        if func is np.rot90:
            if len(args) == 1 and kwargs.keys() <= {"k", "axes"}:
                k_value = kwargs.get("k", 1)
                axes_value = kwargs.get("axes", (0, 1))
            elif len(args) == 2 and kwargs.keys() <= {"axes"}:
                k_value = args[1]
                axes_value = kwargs.get("axes", (0, 1))
            elif len(args) == 3 and not kwargs:
                k_value = args[1]
                axes_value = args[2]
            else:
                raise ValueError("program AD np.rot90 supports array, k, and axes")
            return _trace_rot90(
                _coerce_trace_array(args[0], self.context),
                k=k_value,
                axes=axes_value,
            )
        if func is np.flip:
            if len(args) != 1 or kwargs.keys() - {"axis"}:
                raise ValueError("program AD np.flip supports one array and optional axis")
            return _trace_flip(
                _coerce_trace_array(args[0], self.context),
                axis=kwargs.get("axis"),
            )
        if func is np.flipud:
            if len(args) != 1 or kwargs:
                raise ValueError("program AD np.flipud supports one array")
            return _trace_flipud(_coerce_trace_array(args[0], self.context))
        if func is np.fliplr:
            if len(args) != 1 or kwargs:
                raise ValueError("program AD np.fliplr supports one array")
            return _trace_fliplr(_coerce_trace_array(args[0], self.context))
        if func is np.take:
            if len(args) < 2 or len(args) > 3 or kwargs.keys() - {"axis", "mode"}:
                raise ValueError("program AD np.take supports array, indices, axis, and mode")
            axis = args[2] if len(args) == 3 else kwargs.get("axis")
            return _trace_take(
                _coerce_trace_array(args[0], self.context),
                args[1],
                axis=cast(int | None, axis),
                mode=cast(str, kwargs.get("mode", "raise")),
            )
        if func is np.take_along_axis:
            if len(args) < 2 or len(args) > 3 or kwargs.keys() - {"axis"}:
                raise ValueError("program AD np.take_along_axis supports array, indices, and axis")
            axis = args[2] if len(args) == 3 else kwargs.get("axis", -1)
            return _trace_take_along_axis(
                _coerce_trace_array(args[0], self.context), args[1], axis=axis
            )
        if func is np.delete:
            if len(args) < 2 or len(args) > 3 or kwargs.keys() - {"axis"}:
                raise ValueError("program AD np.delete supports array, object, and axis")
            axis = args[2] if len(args) == 3 else kwargs.get("axis")
            return _trace_delete(
                _coerce_trace_array(args[0], self.context),
                args[1],
                axis=axis,
            )
        if func is np.pad:
            if len(args) < 2 or len(args) > 3 or kwargs.keys() - {"mode", "constant_values"}:
                raise ValueError(
                    "program AD np.pad supports array, pad_width, constant mode, "
                    "and constant_values"
                )
            mode = args[2] if len(args) == 3 else kwargs.get("mode", "constant")
            return _trace_pad(
                _coerce_trace_array(args[0], self.context),
                args[1],
                mode=mode,
                constant_values=kwargs.get("constant_values", 0.0),
            )
        if func is np.insert:
            if len(args) < 3 or len(args) > 4 or kwargs.keys() - {"axis"}:
                raise ValueError("program AD np.insert supports array, object, values, and axis")
            axis = args[3] if len(args) == 4 else kwargs.get("axis")
            return _trace_insert(
                _coerce_trace_array(args[0], self.context),
                args[1],
                args[2],
                axis=axis,
            )
        if func is np.append:
            if len(args) != 2 or kwargs.keys() - {"axis"}:
                raise ValueError("program AD np.append supports array, values, and axis")
            return _trace_append(
                _coerce_trace_array(args[0], self.context),
                args[1],
                self.context,
                axis=kwargs.get("axis"),
            )
        if func is np.transpose:
            if len(args) != 1 or kwargs.keys() - {"axes"}:
                raise ValueError("whole-program AD np.transpose supports one array and axes")
            return _trace_transpose(
                args[0],
                self.context,
                axes=cast(tuple[int, ...] | None, kwargs.get("axes")),
            )
        if func is np.trace:
            if len(args) != 1 or kwargs.keys() - {"offset", "axis1", "axis2"}:
                raise ValueError("whole-program AD np.trace supports one matrix")
            return _trace_trace(
                args[0],
                self.context,
                offset=cast(int, kwargs.get("offset", 0)),
                axis1=cast(int, kwargs.get("axis1", 0)),
                axis2=cast(int, kwargs.get("axis2", 1)),
            )
        if func is np.diag:
            if len(args) != 1 or kwargs.keys() - {"k"}:
                raise ValueError("whole-program AD np.diag supports one vector or matrix")
            return _trace_diag(args[0], self.context, k=cast(int, kwargs.get("k", 0)))
        if func is np.diagflat:
            if len(args) != 1 or kwargs.keys() - {"k"}:
                raise ValueError("program AD np.diagflat supports one array and k")
            return _trace_diagflat(args[0], self.context, k=cast(int, kwargs.get("k", 0)))
        if func is np.diagonal:
            if len(args) < 1 or len(args) > 4 or kwargs.keys() - {"offset", "axis1", "axis2"}:
                raise ValueError("program AD np.diagonal supports array, offset, axis1, and axis2")
            if len(args) >= 2 and "offset" in kwargs:
                raise ValueError("program AD np.diagonal offset must be supplied once")
            if len(args) >= 3 and "axis1" in kwargs:
                raise ValueError("program AD np.diagonal axis1 must be supplied once")
            if len(args) >= 4 and "axis2" in kwargs:
                raise ValueError("program AD np.diagonal axis2 must be supplied once")
            return _trace_diagonal(
                _coerce_trace_array(args[0], self.context),
                offset=args[1] if len(args) >= 2 else kwargs.get("offset", 0),
                axis1=args[2] if len(args) >= 3 else kwargs.get("axis1", 0),
                axis2=args[3] if len(args) >= 4 else kwargs.get("axis2", 1),
            )
        if func is np.concatenate:
            if len(args) != 1 or kwargs.keys() - {"axis"}:
                raise ValueError(
                    "whole-program AD np.concatenate supports arrays and optional axis"
                )
            return _trace_concatenate(
                cast(Sequence[object], args[0]),
                self.context,
                axis=cast(int, kwargs.get("axis", 0)),
            )
        if func is np.stack:
            if len(args) != 1 or kwargs.keys() - {"axis"}:
                raise ValueError("whole-program AD np.stack supports arrays and optional axis")
            return _trace_stack(
                cast(Sequence[object], args[0]),
                self.context,
                axis=cast(int, kwargs.get("axis", 0)),
            )
        if func in {np.hstack, np.vstack, np.column_stack, np.dstack}:
            if len(args) != 1 or kwargs:
                raise ValueError(f"program AD np.{func.__name__} supports one array sequence")
            return _trace_stack_convenience(
                func.__name__,
                cast(Sequence[object], args[0]),
                self.context,
            )
        if func is np.block:
            if len(args) != 1 or kwargs:
                raise ValueError("program AD np.block supports one nested block sequence")
            return _trace_block(args[0], self.context)
        if func in {np.split, np.array_split}:
            if len(args) < 2 or len(args) > 3 or kwargs.keys() - {"axis"}:
                raise ValueError(
                    f"program AD np.{func.__name__} supports array, sections, and axis"
                )
            axis = args[2] if len(args) == 3 else kwargs.get("axis", 0)
            return _trace_split(
                func.__name__,
                _coerce_trace_array(args[0], self.context),
                args[1],
                self.context,
                axis=axis,
            )
        if func in {np.hsplit, np.vsplit, np.dsplit}:
            if len(args) != 2 or kwargs:
                raise ValueError(f"program AD np.{func.__name__} supports array and sections")
            return _trace_split(
                func.__name__,
                _coerce_trace_array(args[0], self.context),
                args[1],
                self.context,
                axis=None,
            )
        if func in {np.tril, np.triu}:
            if len(args) == 1 and kwargs.keys() <= {"k"}:
                k_value = kwargs.get("k", 0)
            elif len(args) == 2 and not kwargs:
                k_value = args[1]
            else:
                raise ValueError(f"program AD np.{func.__name__} supports array and k")
            return _trace_triangular_mask(
                _coerce_trace_array(args[0], self.context),
                k=k_value,
                lower=func is np.tril,
            )
        if func is np.clip:
            if len(args) < 3 or len(args) > 4 or kwargs:
                raise ValueError("whole-program AD np.clip supports array, lower, and upper")
            return _trace_clip(args[0], args[1], args[2], self.context)
        if func is np.linalg.norm:
            if len(args) < 1 or len(args) > 3 or kwargs.keys() - {"ord", "axis"}:
                raise ValueError(
                    "whole-program AD np.linalg.norm supports array, optional ord, and optional axis"
                )
            if len(args) >= 2 and "ord" in kwargs:
                raise ValueError("whole-program AD np.linalg.norm ord must be supplied once")
            if len(args) >= 3 and "axis" in kwargs:
                raise ValueError("whole-program AD np.linalg.norm axis must be supplied once")
            return _trace_norm(
                args[0],
                self.context,
                ord_value=args[1] if len(args) >= 2 else kwargs.get("ord"),
                axis=args[2] if len(args) >= 3 else kwargs.get("axis"),
            )
        if func is np.linalg.det:
            if len(args) != 1 or kwargs:
                raise ValueError("program AD np.linalg.det supports one matrix")
            _require_program_ad_linalg_contract("det", args)
            return _trace_det(args[0], self.context)
        if func is np.linalg.inv:
            if len(args) != 1 or kwargs:
                raise ValueError("program AD np.linalg.inv supports one matrix")
            _require_program_ad_linalg_contract("inv", args)
            return _trace_inv(args[0], self.context)
        if func is np.linalg.solve:
            if len(args) != 2 or kwargs:
                raise ValueError("program AD np.linalg.solve supports matrix and right-hand side")
            _require_program_ad_linalg_contract("solve", args)
            return _trace_solve(args[0], args[1], self.context)
        if func is np.linalg.matrix_power:
            if len(args) != 2 or kwargs:
                raise ValueError("program AD np.linalg.matrix_power supports matrix and power")
            _require_program_ad_linalg_contract("matrix_power", args)
            return _trace_matrix_power(args[0], args[1], self.context)
        if func is np.linalg.multi_dot:
            if len(args) != 1 or kwargs:
                raise ValueError("program AD np.linalg.multi_dot supports one operand sequence")
            _require_program_ad_linalg_contract("multi_dot", args)
            return _trace_multi_dot(args[0], self.context)
        if func is np.linalg.eig:
            if len(args) != 1 or kwargs:
                raise ValueError("program AD np.linalg.eig supports one matrix")
            _require_program_ad_linalg_contract("eig", (args[0],))
            return _trace_eig(args[0], self.context)
        if func is np.linalg.eigh:
            if len(args) == 2 and not kwargs:
                matrix, uplo = args
            elif len(args) == 1 and set(kwargs) <= {"UPLO"}:
                matrix = args[0]
                uplo = kwargs.get("UPLO", "L")
            else:
                raise ValueError("program AD np.linalg.eigh supports one matrix and optional UPLO")
            _require_program_ad_linalg_contract("eigh", (matrix,))
            return _trace_eigh(matrix, self.context, uplo=str(uplo))
        if func is np.linalg.eigvalsh:
            if len(args) == 2 and not kwargs:
                matrix, uplo = args
            elif len(args) == 1 and set(kwargs) <= {"UPLO"}:
                matrix = args[0]
                uplo = kwargs.get("UPLO", "L")
            else:
                raise ValueError(
                    "program AD np.linalg.eigvalsh supports one matrix and optional UPLO"
                )
            _require_program_ad_linalg_contract("eigvalsh", (matrix,))
            return _trace_eigvalsh(matrix, self.context, uplo=str(uplo))
        if func is np.linalg.eigvals:
            if len(args) != 1 or kwargs:
                raise ValueError("program AD np.linalg.eigvals supports one matrix")
            _require_program_ad_linalg_contract("eigvals", (args[0],))
            return _trace_eigvals(args[0], self.context)
        if func is np.linalg.svd:
            if not 1 <= len(args) <= 4:
                raise ValueError(
                    "program AD np.linalg.svd supports one matrix and static SVD options"
                )
            matrix = args[0]
            full_matrices = args[1] if len(args) >= 2 else kwargs.pop("full_matrices", True)
            compute_uv = args[2] if len(args) >= 3 else kwargs.pop("compute_uv", True)
            hermitian = args[3] if len(args) >= 4 else kwargs.pop("hermitian", False)
            if kwargs:
                raise ValueError(
                    "program AD np.linalg.svd supports full_matrices, compute_uv, and hermitian"
                )
            if not isinstance(full_matrices, (bool, np.bool_)):
                raise ValueError("program AD np.linalg.svd full_matrices must be static boolean")
            if not isinstance(compute_uv, (bool, np.bool_)):
                raise ValueError("program AD np.linalg.svd compute_uv must be static boolean")
            if not isinstance(hermitian, (bool, np.bool_)):
                raise ValueError("program AD np.linalg.svd hermitian must be static boolean")
            if bool(compute_uv):
                raise ValueError("program AD np.linalg.svd supports compute_uv=False only")
            if bool(hermitian):
                raise ValueError("program AD np.linalg.svd supports hermitian=False only")
            _require_program_ad_linalg_contract("svd", (matrix,))
            return _trace_svdvals(matrix, self.context)
        if func is np.linalg.pinv:
            if not 1 <= len(args) <= 3:
                raise ValueError(
                    "program AD np.linalg.pinv supports one matrix and static cutoff options"
                )
            matrix = args[0]
            rcond = args[1] if len(args) >= 2 else kwargs.pop("rcond", None)
            hermitian = args[2] if len(args) >= 3 else kwargs.pop("hermitian", False)
            rtol = kwargs.pop("rtol", None)
            if kwargs:
                raise ValueError("program AD np.linalg.pinv supports rcond, rtol, and hermitian")
            if rcond is not None and rtol is not None:
                raise ValueError("program AD np.linalg.pinv accepts only one of rcond or rtol")
            if not isinstance(hermitian, (bool, np.bool_)):
                raise ValueError("program AD np.linalg.pinv hermitian must be static boolean")
            if bool(hermitian):
                raise ValueError("program AD np.linalg.pinv supports hermitian=False only")
            cutoff = _program_ad_linalg_normalise_rcond(rtol if rtol is not None else rcond)
            _require_program_ad_linalg_contract("pinv", (matrix,))
            return _trace_pinv(matrix, self.context, rcond=cutoff)
        if func in {np.argmax, np.argmin}:
            if len(args) not in {1, 2}:
                raise ValueError(f"program AD np.{func.__name__} supports array and optional axis")
            unsupported_index_kwargs = set(kwargs) - {"axis", "out", "keepdims"}
            if unsupported_index_kwargs:
                raise ValueError(
                    f"program AD np.{func.__name__} only supports axis, out, and keepdims"
                )
            if kwargs.get("out") is not None:
                raise ValueError(f"program AD np.{func.__name__} does not support out")
            keepdims = kwargs.get("keepdims", False)
            if not isinstance(keepdims, (bool, np.bool_)) or bool(keepdims):
                raise ValueError(f"program AD np.{func.__name__} supports keepdims=False only")
            if len(args) == 2 and "axis" in kwargs:
                raise ValueError(f"program AD np.{func.__name__} received duplicate axis")
            axis = args[1] if len(args) == 2 else kwargs.get("axis")
            _raise_index_selection_boundary(func.__name__, (args[0], axis))
        if func is np.sort:
            if len(args) != 1:
                raise ValueError("program AD np.sort expects exactly one differentiable array")
            unsupported_sort_kwargs = set(kwargs) - {"axis", "kind", "order"}
            if unsupported_sort_kwargs:
                raise ValueError(
                    "program AD np.sort only supports axis, kind, and order keyword arguments"
                )
            if kwargs.get("order") is not None:
                raise ValueError("program AD np.sort does not support structured-array order")
            kind = kwargs.get("kind")
            if kind not in {None, "quicksort", "mergesort", "heapsort", "stable"}:
                raise ValueError("program AD np.sort kind must be a NumPy sort kind")
            sort_axis = kwargs.get("axis", -1)
            _require_program_ad_selection_contract("sort", (args[0], sort_axis, kind))
            return _trace_sort(
                _coerce_trace_array(args[0], self.context),
                axis=sort_axis,
                kind=cast(_TraceSortKind | None, kind),
            )
        if func is np.argsort:
            if len(args) != 1:
                raise ValueError("program AD np.argsort expects exactly one differentiable array")
            unsupported_argsort_kwargs = set(kwargs) - {"axis", "kind", "order", "stable"}
            if unsupported_argsort_kwargs:
                raise ValueError(
                    "program AD np.argsort only supports axis, kind, order, and stable"
                )
            if kwargs.get("order") is not None:
                raise ValueError("program AD np.argsort does not support structured-array order")
            if kwargs.get("stable") is not None:
                raise ValueError("program AD np.argsort does not support stable keyword")
            kind = kwargs.get("kind")
            if kind not in {None, "quicksort", "mergesort", "heapsort", "stable"}:
                raise ValueError("program AD np.argsort kind must be a NumPy sort kind")
            axis = kwargs.get("axis", -1)
            _raise_index_selection_boundary("argsort", (args[0], axis, kind))
        raise ValueError(f"unsupported whole-program AD NumPy function {func.__name__}")

    def _binary(self, other: object, op: np.ufunc) -> TraceADScalar | TraceADArray:
        return _apply_trace_ufunc(op, (self, other), self.context)

    def __add__(self, other: object) -> TraceADScalar | TraceADArray:
        return self._binary(other, np.add)

    def __radd__(self, other: object) -> TraceADScalar | TraceADArray:
        return self._binary(other, np.add)

    def __sub__(self, other: object) -> TraceADScalar | TraceADArray:
        return self._binary(other, np.subtract)

    def __rsub__(self, other: object) -> TraceADScalar | TraceADArray:
        return _apply_trace_ufunc(np.subtract, (other, self), self.context)

    def __mul__(self, other: object) -> TraceADScalar | TraceADArray:
        return self._binary(other, np.multiply)

    def __rmul__(self, other: object) -> TraceADScalar | TraceADArray:
        return self._binary(other, np.multiply)

    def __truediv__(self, other: object) -> TraceADScalar | TraceADArray:
        return self._binary(other, np.divide)

    def __rtruediv__(self, other: object) -> TraceADScalar | TraceADArray:
        return _apply_trace_ufunc(np.divide, (other, self), self.context)

    def __pow__(self, other: object) -> TraceADScalar | TraceADArray:
        return self._binary(other, np.power)

    def __rpow__(self, other: object) -> TraceADScalar | TraceADArray:
        return _apply_trace_ufunc(np.power, (other, self), self.context)

    def __neg__(self) -> TraceADScalar | TraceADArray:
        return _apply_trace_ufunc(np.negative, (self,), self.context)

    def __matmul__(self, other: object) -> TraceADScalar | TraceADArray:
        return _trace_matmul(self, other, self.context)

    def __rmatmul__(self, other: object) -> TraceADScalar | TraceADArray:
        return _trace_matmul(other, self, self.context)

    def _compare(self, op: str, other: object) -> _TracePredicate | TraceADPredicateArray:
        right = _coerce_trace_array(other, self.context)
        shape = _broadcast_shape(self.shape, right.shape)
        left = _broadcast_trace_array(self, shape, self.context)
        right = _broadcast_trace_array(right, shape, self.context)
        predicates = tuple(
            left_item._compare(op, right_item)
            for left_item, right_item in zip(left._items, right._items, strict=True)
        )
        return (
            predicates[0]
            if shape == ()
            else TraceADPredicateArray(predicates, shape, self.context)
        )

    def __gt__(self, other: object) -> _TracePredicate | TraceADPredicateArray:
        return self._compare("gt", other)

    def __ge__(self, other: object) -> _TracePredicate | TraceADPredicateArray:
        return self._compare("ge", other)

    def __lt__(self, other: object) -> _TracePredicate | TraceADPredicateArray:
        return self._compare("lt", other)

    def __le__(self, other: object) -> _TracePredicate | TraceADPredicateArray:
        return self._compare("le", other)

    def __eq__(self, other: object) -> _TracePredicate | TraceADPredicateArray:  # type: ignore[override]
        return self._compare("eq", other)

    def __ne__(self, other: object) -> _TracePredicate | TraceADPredicateArray:  # type: ignore[override]
        return self._compare("ne", other)


def _coerce_trace_scalar(value: object, context: _WholeProgramTraceContext) -> TraceADScalar:
    if isinstance(value, TraceADScalar):
        if value.context is not context:
            raise ValueError("whole-program AD scalars belong to different traces")
        return value
    if isinstance(value, TraceADArray):
        if value.context is not context:
            raise ValueError("whole-program AD arrays belong to different traces")
        return value.item()
    tangent = np.zeros(context.parameter_count, dtype=np.float64)
    return TraceADScalar(
        _as_real_scalar("whole-program AD constant", value), tangent, context, repr(value)
    )


def _coerce_trace_array(value: object, context: _WholeProgramTraceContext) -> TraceADArray:
    if isinstance(value, TraceADArray):
        if value.context is not context:
            raise ValueError("whole-program AD arrays belong to different traces")
        return value
    if isinstance(value, TraceADScalar):
        if value.context is not context:
            raise ValueError("whole-program AD scalars belong to different traces")
        return TraceADArray((value,), (), context)
    raw = np.asarray(value)
    if raw.dtype.kind in {"O", "S", "U", "c"}:
        raise ValueError("whole-program AD array operands must be real numeric")
    array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("whole-program AD array operands must be finite")
    tangent = np.zeros(context.parameter_count, dtype=np.float64)
    items = tuple(
        TraceADScalar(float(item), tangent.copy(), context, repr(float(item)))
        for item in array.reshape(-1)
    )
    return TraceADArray(items, tuple(array.shape), context)


def _validate_trace_like_constructor_kwargs(kwargs: Mapping[str, object]) -> None:
    if "shape" in kwargs:
        raise ValueError("program AD like-constructors do not support shape overrides")
    unsupported = kwargs.keys() - {"dtype", "order", "subok"}
    if unsupported:
        raise ValueError("program AD like-constructors support dtype, order, and subok only")
    if "dtype" in kwargs and kwargs["dtype"] is not None:
        dtype = np.dtype(cast(Any, kwargs["dtype"]))
        if dtype.kind in {"O", "S", "U", "c"}:
            raise ValueError("program AD like-constructors require real numeric dtype")


def _trace_like_constant(
    reference: object,
    fill_value: object,
    context: _WholeProgramTraceContext,
    *,
    name: Literal["zeros_like", "ones_like", "full_like"],
) -> TraceADArray:
    array = _coerce_trace_array(reference, context)
    _require_program_ad_assembly_contract(
        name, (array,) if name != "full_like" else (array, fill_value)
    )
    scalar = _coerce_trace_scalar(fill_value, context)
    return TraceADArray(tuple(scalar for _ in range(array.size)), array.shape, context)


def _normalise_trace_reshape_shape(shape: object, size: int) -> tuple[int, ...]:
    dimensions: tuple[int, ...]
    if isinstance(shape, (int, np.integer)) and not isinstance(shape, bool):
        dimensions = (int(shape),)
    elif isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
        raw_dimensions = tuple(cast(Any, shape))
        if any(
            isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer))
            for dimension in raw_dimensions
        ):
            raise ValueError("program AD reshape shape dimensions must be static integers")
        dimensions = tuple(int(dimension) for dimension in raw_dimensions)
    else:
        raise ValueError("program AD reshape shape must be a static integer or shape tuple")
    inferred_axes = tuple(index for index, dimension in enumerate(dimensions) if dimension == -1)
    if len(inferred_axes) > 1:
        raise ValueError("program AD reshape supports at most one inferred dimension")
    if any(dimension < -1 for dimension in dimensions):
        raise ValueError("program AD reshape dimensions must be non-negative or -1")
    known_product = int(np.prod(tuple(dimension for dimension in dimensions if dimension != -1)))
    if inferred_axes:
        if known_product == 0:
            raise ValueError("program AD reshape cannot infer dimension from zero product")
        if size % known_product != 0:
            raise ValueError("program AD reshape inferred dimension must preserve size")
        inferred = size // known_product
        dimensions = tuple(inferred if dimension == -1 else dimension for dimension in dimensions)
    if int(np.prod(dimensions)) != size:
        raise ValueError("program AD reshape must preserve size")
    return dimensions


def _normalise_trace_broadcast_shape(shape: object) -> tuple[int, ...]:
    dimensions: tuple[int, ...]
    if isinstance(shape, (int, np.integer)) and not isinstance(shape, bool):
        dimensions = (int(shape),)
    elif isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
        dimensions = tuple(int(dimension) for dimension in shape)
    else:
        raise ValueError("program AD np.broadcast_to requires an integer shape")
    if any(dimension < 0 for dimension in dimensions):
        raise ValueError("program AD np.broadcast_to shape dimensions must be non-negative")
    return dimensions


def _trace_array_source_indices(array: TraceADArray) -> tuple[int | None, ...]:
    """Return original parameter-array slots carried by a trace array, if known."""

    if array._source_indices is None:
        return tuple(None for _ in range(array.size))
    return array._source_indices


def _trace_array_view_from_local_indices(
    array: TraceADArray,
    op: str,
    local_indices: Sequence[int],
    shape: tuple[int, ...],
) -> TraceADArray:
    """Return a derivative-preserving view and record source-index alias metadata."""

    source_indices = tuple(
        _trace_array_source_indices(array)[int(local_index)] for local_index in local_indices
    )
    items = tuple(array._items[int(local_index)] for local_index in local_indices)
    array.context.record_array_view_aliases(op, source_indices, items)
    return TraceADArray(items, shape, array.context, source_indices)


def _trace_array_getitem(array: TraceADArray, index: object) -> TraceADScalar | TraceADArray:
    _require_program_ad_array_contract("getitem", (array, index))
    _validate_trace_basic_index(index)
    source = np.arange(array.size, dtype=np.int64).reshape(array.shape)
    try:
        selected = source[cast(Any, index)]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "program AD basic indexing requires static in-bounds integer, slice, "
            "ellipsis, or newaxis selectors"
        ) from exc
    selected_array = np.asarray(selected)
    if selected_array.shape == ():
        return array._items[int(selected_array)]
    local_indices = tuple(int(item) for item in selected_array.reshape(-1))
    return _trace_array_view_from_local_indices(
        array,
        "getitem",
        local_indices,
        tuple(int(dimension) for dimension in selected_array.shape),
    )


def _normalise_shape_transform_axes(
    name: str, axis: int | tuple[int, ...], *, output_rank: int
) -> tuple[int, ...]:
    axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    normalised: list[int] = []
    for item in axes:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"program AD {name} axes must be static integers")
        value = int(item)
        if value < 0:
            value += output_rank
        if value < 0 or value >= output_rank:
            raise ValueError(f"program AD {name} axis out of bounds")
        if value in normalised:
            raise ValueError(f"program AD {name} axes must be unique")
        normalised.append(value)
    return tuple(sorted(normalised))


def _trace_squeeze(
    array: TraceADArray, *, axis: int | tuple[int, ...] | None = None
) -> TraceADArray:
    _require_program_ad_shape_contract("squeeze", (array,) if axis is None else (array, axis))
    local_indices = tuple(range(array.size))
    if axis is None:
        target_shape = tuple(dimension for dimension in array.shape if dimension != 1)
        return _trace_array_view_from_local_indices(array, "squeeze", local_indices, target_shape)
    axes = _normalise_shape_transform_axes("squeeze", axis, output_rank=array.ndim)
    for item in axes:
        if array.shape[item] != 1:
            raise ValueError("program AD squeeze axis must have length one")
    target_shape = tuple(
        dimension for index, dimension in enumerate(array.shape) if index not in axes
    )
    return _trace_array_view_from_local_indices(array, "squeeze", local_indices, target_shape)


def _trace_expand_dims(array: TraceADArray, *, axis: int | tuple[int, ...]) -> TraceADArray:
    _require_program_ad_shape_contract("expand_dims", (array, axis))
    axis_tuple = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    output_rank = array.ndim + len(axis_tuple)
    axes = _normalise_shape_transform_axes("expand_dims", axis_tuple, output_rank=output_rank)
    shape = list(array.shape)
    for item in axes:
        shape.insert(item, 1)
    return _trace_array_view_from_local_indices(
        array,
        "expand_dims",
        tuple(range(array.size)),
        tuple(shape),
    )


def _trace_atleast_nd(array: TraceADArray, *, rank: int) -> TraceADArray:
    if rank not in {1, 2, 3}:
        raise ValueError("program AD atleast rank must be 1, 2, or 3")
    _require_program_ad_shape_contract(f"atleast_{rank}d", (array,))
    if rank == 1:
        shape = array.shape if array.ndim >= 1 else (1,)
    elif rank == 2:
        if array.ndim == 0:
            shape = (1, 1)
        elif array.ndim == 1:
            shape = (1, array.shape[0])
        else:
            shape = array.shape
    elif rank == 3:
        if array.ndim == 0:
            shape = (1, 1, 1)
        elif array.ndim == 1:
            shape = (1, array.shape[0], 1)
        elif array.ndim == 2:
            shape = (array.shape[0], array.shape[1], 1)
        else:
            shape = array.shape
    return _trace_array_view_from_local_indices(
        array,
        f"atleast_{rank}d",
        tuple(range(array.size)),
        shape,
    )


def _normalise_axis_permutation_axis(name: str, axis: object, *, rank: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError(f"program AD {name} axes must be static integers")
    value = int(axis)
    if value < 0:
        value += rank
    if value < 0 or value >= rank:
        raise ValueError(f"program AD {name} axis out of bounds")
    return value


def _normalise_axis_permutation_axes(
    name: str, axes: object, *, rank: int, role: str
) -> tuple[int, ...]:
    if isinstance(axes, (int, np.integer)):
        raw_axes = (axes,)
    else:
        try:
            raw_axes = tuple(cast(Any, axes))
        except TypeError as exc:
            raise ValueError(f"program AD {name} {role} axes must be static integers") from exc
    normalised = tuple(
        _normalise_axis_permutation_axis(name, axis, rank=rank) for axis in raw_axes
    )
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"program AD {name} {role} axes must be unique")
    return normalised


def _trace_swapaxes(array: TraceADArray, *, axis1: int, axis2: int) -> TraceADArray:
    _require_program_ad_shape_contract("swapaxes", (array, axis1, axis2))
    first = _normalise_axis_permutation_axis("swapaxes", axis1, rank=array.ndim)
    second = _normalise_axis_permutation_axis("swapaxes", axis2, rank=array.ndim)
    source = np.arange(array.size, dtype=np.int64).reshape(array.shape)
    moved = np.swapaxes(source, first, second)
    return _trace_array_view_from_local_indices(
        array,
        "swapaxes",
        tuple(int(index) for index in moved.reshape(-1)),
        tuple(map(int, moved.shape)),
    )


def _trace_moveaxis(
    array: TraceADArray,
    *,
    source: int | tuple[int, ...],
    destination: int | tuple[int, ...],
) -> TraceADArray:
    _require_program_ad_shape_contract("moveaxis", (array, source, destination))
    source_axes = _normalise_axis_permutation_axes(
        "moveaxis", source, rank=array.ndim, role="source"
    )
    destination_axes = _normalise_axis_permutation_axes(
        "moveaxis", destination, rank=array.ndim, role="destination"
    )
    if len(source_axes) != len(destination_axes):
        raise ValueError("program AD moveaxis source and destination lengths must match")
    moved_indices = np.moveaxis(
        np.arange(array.size, dtype=np.int64).reshape(array.shape),
        source_axes,
        destination_axes,
    )
    return _trace_array_view_from_local_indices(
        array,
        "moveaxis",
        tuple(int(index) for index in moved_indices.reshape(-1)),
        tuple(map(int, moved_indices.shape)),
    )


def _normalise_repeat_count(count: object) -> int:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise ValueError("program AD repeat counts must be static non-negative integers")
    value = int(count)
    if value < 0:
        raise ValueError("program AD repeat counts must be static non-negative integers")
    return value


def _normalise_repeat_counts(repeats: object, selected_size: int) -> int | tuple[int, ...]:
    if isinstance(repeats, (int, np.integer)) and not isinstance(repeats, bool):
        return _normalise_repeat_count(repeats)
    try:
        raw_repeats = tuple(cast(Any, repeats))
    except TypeError as exc:
        raise ValueError("program AD repeat counts must be static non-negative integers") from exc
    if len(raw_repeats) != selected_size:
        raise ValueError("program AD repeat counts length must match selected axis")
    return tuple(_normalise_repeat_count(item) for item in raw_repeats)


def _trace_repeat(
    array: TraceADArray, *, repeats: object, axis: int | None = None
) -> TraceADArray:
    if axis is None:
        _require_program_ad_shape_contract("repeat", (array, repeats))
    else:
        _require_program_ad_shape_contract("repeat", (array, repeats, axis))
    source = np.arange(array.size, dtype=np.int64).reshape(array.shape)
    if axis is None:
        repeat_counts = _normalise_repeat_counts(repeats, array.size)
        repeated = np.repeat(source.reshape(-1), repeat_counts)
    else:
        axis_index = _normalise_axis_permutation_axis("repeat", axis, rank=array.ndim)
        repeat_counts = _normalise_repeat_counts(repeats, array.shape[axis_index])
        repeated = np.repeat(source, repeat_counts, axis=axis_index)
    return _trace_array_view_from_local_indices(
        array,
        "repeat",
        tuple(int(index) for index in repeated.reshape(-1)),
        tuple(map(int, repeated.shape)),
    )


def _normalise_tile_reps(reps: object) -> tuple[int, ...]:
    values: tuple[int, ...]
    if isinstance(reps, (int, np.integer)) and not isinstance(reps, bool):
        values = (int(reps),)
    else:
        try:
            values = tuple(cast(Any, reps))
        except TypeError as exc:
            raise ValueError("program AD tile reps must be static non-negative integers") from exc
        if not values:
            raise ValueError("program AD tile reps must contain at least one axis")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in values
        ):
            raise ValueError("program AD tile reps must be static non-negative integers")
        values = tuple(int(item) for item in values)
    if any(value < 0 for value in values):
        raise ValueError("program AD tile reps must be static non-negative integers")
    return values


def _trace_tile(array: TraceADArray, *, reps: object) -> TraceADArray:
    _require_program_ad_shape_contract("tile", (array, reps))
    reps_tuple = _normalise_tile_reps(reps)
    rank = max(array.ndim, len(reps_tuple))
    source_shape = (1,) * (rank - array.ndim) + array.shape
    reps_aligned = (1,) * (rank - len(reps_tuple)) + reps_tuple
    source = np.arange(array.size, dtype=np.int64).reshape(source_shape)
    tiled = np.tile(source, reps_aligned)
    return _trace_array_view_from_local_indices(
        array,
        "tile",
        tuple(int(index) for index in tiled.reshape(-1)),
        tuple(map(int, tiled.shape)),
    )


def _normalise_roll_shift_scalar(shift: object) -> int:
    if isinstance(shift, bool) or not isinstance(shift, (int, np.integer)):
        raise ValueError("program AD roll shift must be static integers")
    return int(shift)


def _normalise_roll_shift_tuple(shift: object, axis_count: int) -> tuple[int, ...]:
    if isinstance(shift, (int, np.integer)) and not isinstance(shift, bool):
        return tuple(int(shift) for _ in range(axis_count))
    try:
        raw_shifts = tuple(cast(Any, shift))
    except TypeError as exc:
        raise ValueError("program AD roll shift must be static integers") from exc
    if len(raw_shifts) != axis_count:
        raise ValueError("program AD roll shift and axis lengths must match")
    return tuple(_normalise_roll_shift_scalar(item) for item in raw_shifts)


def _trace_roll(array: TraceADArray, *, shift: object, axis: object = None) -> TraceADArray:
    _require_program_ad_shape_contract(
        "roll", (array, shift) if axis is None else (array, shift, axis)
    )
    if axis is None:
        flat_shift = _normalise_roll_shift_scalar(shift)
        rolled = np.roll(np.arange(array.size, dtype=np.int64), flat_shift).reshape(array.shape)
    else:
        axes = _normalise_axis_permutation_axes("roll", axis, rank=array.ndim, role="axis")
        shifts = _normalise_roll_shift_tuple(shift, len(axes))
        rolled = np.roll(
            np.arange(array.size, dtype=np.int64).reshape(array.shape),
            shifts,
            axis=axes,
        )
    return _trace_array_view_from_local_indices(
        array,
        "roll",
        tuple(int(index) for index in rolled.reshape(-1)),
        tuple(map(int, rolled.shape)),
    )


def _normalise_rot90_k(k: object) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("program AD rot90 k must be a static integer")
    return int(k)


def _normalise_rot90_axes(axes: object, *, rank: int) -> tuple[int, int]:
    normalised = _normalise_axis_permutation_axes("rot90", axes, rank=rank, role="axes")
    if len(normalised) != 2:
        raise ValueError("program AD rot90 axes must contain exactly two axes")
    return (normalised[0], normalised[1])


def _trace_rot90(array: TraceADArray, *, k: object = 1, axes: object = (0, 1)) -> TraceADArray:
    _require_program_ad_shape_contract("rot90", (array, k, axes))
    k_value = _normalise_rot90_k(k)
    axes_value = _normalise_rot90_axes(axes, rank=array.ndim)
    rotated = np.rot90(
        np.arange(array.size, dtype=np.int64).reshape(array.shape),
        k=k_value,
        axes=axes_value,
    )
    return _trace_array_view_from_local_indices(
        array,
        "rot90",
        tuple(int(index) for index in rotated.reshape(-1)),
        tuple(map(int, rotated.shape)),
    )


def _trace_flip(array: TraceADArray, *, axis: object = None) -> TraceADArray:
    _require_program_ad_shape_contract("flip", (array,) if axis is None else (array, axis))
    source = np.arange(array.size, dtype=np.int64).reshape(array.shape)
    if axis is None:
        flipped = np.flip(source)
    else:
        axes = _normalise_axis_permutation_axes("flip", axis, rank=array.ndim, role="axis")
        flipped = np.flip(source, axis=axes)
    return _trace_array_view_from_local_indices(
        array,
        "flip",
        tuple(int(index) for index in flipped.reshape(-1)),
        tuple(map(int, flipped.shape)),
    )


def _normalise_sort_axis(axis: object, rank: int) -> int:
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD np.sort axis must be a static integer or None")
    axis_index = int(axis)
    if axis_index < 0:
        axis_index += rank
    if axis_index < 0 or axis_index >= rank:
        raise ValueError("program AD np.sort axis out of bounds")
    return axis_index


def _require_strict_sort_values(values: NDArray[np.float64]) -> None:
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("program AD np.sort requires finite values")
    if values.size <= 1:
        return
    sorted_values = np.sort(values.reshape(-1))
    if bool(np.any(np.diff(sorted_values) == 0.0)):
        raise ValueError(
            "program AD np.sort requires strictly ordered values; equal values form "
            "a nondifferentiable selection boundary"
        )


def _require_strict_sort_axis(values: NDArray[np.float64], *, axis: int) -> None:
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("program AD np.sort requires finite values")
    if values.shape[axis] <= 1:
        return
    sorted_values = np.sort(values, axis=axis)
    if bool(np.any(np.diff(sorted_values, axis=axis) == 0.0)):
        raise ValueError(
            "program AD np.sort requires strictly ordered values; equal values form "
            "a nondifferentiable selection boundary"
        )


def _trace_sort(
    array: TraceADArray,
    *,
    axis: object = -1,
    kind: _TraceSortKind | None = None,
) -> TraceADArray:
    values = np.array([item.primal for item in array._items], dtype=np.float64)
    source = np.arange(array.size, dtype=np.int64)
    sort_kind: _TraceSortKind = "quicksort" if kind is None else kind
    if axis is None:
        _require_strict_sort_values(values)
        order = np.argsort(values, kind=sort_kind)
        sorted_indices = source[order].reshape(array.shape)
    else:
        axis_index = _normalise_sort_axis(axis, array.ndim)
        shaped_values = values.reshape(array.shape)
        _require_strict_sort_axis(shaped_values, axis=axis_index)
        shaped_source = source.reshape(array.shape)
        order = np.argsort(shaped_values, axis=axis_index, kind=sort_kind)
        sorted_indices = np.take_along_axis(shaped_source, order, axis=axis_index)
    return TraceADArray(
        tuple(array._items[int(index)] for index in sorted_indices.reshape(-1)),
        tuple(map(int, sorted_indices.shape)),
        array.context,
    )


def _trace_order_statistic_items(
    items: tuple[TraceADScalar, ...],
    *,
    q: float,
    op_name: str,
) -> TraceADScalar:
    values = np.array([item.primal for item in items], dtype=np.float64)
    _require_strict_order_statistic_values(values, op_name)
    order = np.argsort(values, kind="stable")
    position = q * float(len(items) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    upper_weight = position - float(lower)
    lower_item = items[int(order[lower])]
    if lower == upper:
        return lower_item
    upper_item = items[int(order[upper])]
    return lower_item * (1.0 - upper_weight) + upper_item * upper_weight


def _trace_order_statistic(
    array: TraceADArray,
    *,
    q: float,
    axis: object = None,
    method: object = "linear",
    op_name: str,
) -> TraceADScalar | TraceADArray:
    _normalise_order_statistic_method(method)
    axis_index = _normalise_order_statistic_axis(axis, array.ndim)
    if axis_index is None:
        return _trace_order_statistic_items(tuple(array._items), q=q, op_name=op_name)
    reduced_shape = array.shape[:axis_index] + array.shape[axis_index + 1 :]
    if reduced_shape == ():
        return _trace_order_statistic_items(tuple(array._items), q=q, op_name=op_name)
    items: list[TraceADScalar] = []
    for reduced_flat in range(int(np.prod(reduced_shape))):
        reduced_index = np.unravel_index(reduced_flat, reduced_shape)
        source_items = tuple(
            array._items[
                int(
                    np.ravel_multi_index(
                        reduced_index[:axis_index] + (axis_position,) + reduced_index[axis_index:],
                        array.shape,
                    )
                )
            ]
            for axis_position in range(array.shape[axis_index])
        )
        items.append(_trace_order_statistic_items(source_items, q=q, op_name=op_name))
    return TraceADArray(tuple(items), reduced_shape, array.context)


def _trace_flipud(array: TraceADArray) -> TraceADArray:
    _require_program_ad_shape_contract("flipud", (array,))
    if array.ndim < 1:
        raise ValueError("program AD flipud requires at least rank-1 arrays")
    source = np.arange(array.size, dtype=np.int64).reshape(array.shape)
    flipped = np.flipud(source)
    return _trace_array_view_from_local_indices(
        array,
        "flipud",
        tuple(int(index) for index in flipped.reshape(-1)),
        tuple(map(int, flipped.shape)),
    )


def _trace_fliplr(array: TraceADArray) -> TraceADArray:
    _require_program_ad_shape_contract("fliplr", (array,))
    if array.ndim < 2:
        raise ValueError("program AD fliplr requires at least rank-2 arrays")
    source = np.arange(array.size, dtype=np.int64).reshape(array.shape)
    flipped = np.fliplr(source)
    return _trace_array_view_from_local_indices(
        array,
        "fliplr",
        tuple(int(index) for index in flipped.reshape(-1)),
        tuple(map(int, flipped.shape)),
    )


def _validate_trace_basic_index(index: object) -> None:
    if isinstance(index, tuple):
        for selector in index:
            _validate_trace_basic_index_selector(selector)
        return
    _validate_trace_basic_index_selector(index)


_PROGRAM_AD_STATIC_INDEX_ERROR = (
    "program AD array getitem requires static integer or boolean index arrays, "
    "integer/slice/ellipsis/newaxis selectors, and static integer slice bounds"
)


def _validate_trace_basic_index_selector(selector: object) -> None:
    if isinstance(selector, (TraceADScalar, TraceADArray, _TracePredicate, TraceADPredicateArray)):
        raise ValueError(_PROGRAM_AD_STATIC_INDEX_ERROR)
    if isinstance(selector, (bool, np.bool_)):
        raise ValueError(_PROGRAM_AD_STATIC_INDEX_ERROR)
    if selector is Ellipsis or selector is None:
        return
    if isinstance(selector, (int, np.integer)):
        return
    if isinstance(selector, slice):
        for item in (selector.start, selector.stop, selector.step):
            if item is not None and (
                isinstance(
                    item,
                    (
                        bool,
                        np.bool_,
                        TraceADScalar,
                        TraceADArray,
                        _TracePredicate,
                        TraceADPredicateArray,
                    ),
                )
                or not isinstance(item, (int, np.integer))
            ):
                raise ValueError("program AD basic indexing requires static integer slice bounds")
        return
    if isinstance(selector, (np.ndarray, list)):
        _trace_static_index_array(selector)
        return
    raise ValueError(_PROGRAM_AD_STATIC_INDEX_ERROR)


def _trace_static_index_array(selector: object) -> NDArray[Any]:
    array = np.asarray(selector)
    if array.dtype == object and any(
        isinstance(
            item,
            (TraceADScalar, TraceADArray, _TracePredicate, TraceADPredicateArray),
        )
        for item in array.reshape(-1)
    ):
        raise ValueError(_PROGRAM_AD_STATIC_INDEX_ERROR)
    if array.dtype.kind not in {"i", "u", "b"}:
        raise ValueError(_PROGRAM_AD_STATIC_INDEX_ERROR)
    if array.shape == () and array.dtype.kind == "b":
        raise ValueError(_PROGRAM_AD_STATIC_INDEX_ERROR)
    return array


def _broadcast_trace_array(
    value: object, shape: tuple[int, ...], context: _WholeProgramTraceContext
) -> TraceADArray:
    array = _coerce_trace_array(value, context)
    if array.shape == shape:
        return array
    if array.shape == ():
        return TraceADArray(
            tuple(array.item() for _ in range(int(np.prod(shape)))), shape, context
        )
    try:
        source_indices = np.arange(array.size, dtype=np.int64).reshape(array.shape)
        broadcast_indices = np.broadcast_to(source_indices, shape).reshape(-1)
    except ValueError as exc:
        raise ValueError(
            "whole-program AD array operands must follow NumPy broadcasting rules"
        ) from exc
    return TraceADArray(
        tuple(array._items[int(index)] for index in broadcast_indices), shape, context
    )


def _trace_broadcast_arrays(
    values: Sequence[object], context: _WholeProgramTraceContext
) -> list[TraceADArray]:
    arrays = tuple(_coerce_trace_array(value, context) for value in values)
    shape = _broadcast_shape(*(array.shape for array in arrays))
    _require_program_ad_assembly_contract("broadcast_arrays", arrays)
    return [_broadcast_trace_array(array, shape, context) for array in arrays]


def _broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Return a NumPy-compatible broadcast shape or fail closed."""

    try:
        shape: tuple[int, ...] = np.broadcast_shapes(*shapes)
        return shape
    except ValueError as exc:
        raise ValueError(
            "whole-program AD array operands must follow NumPy broadcasting rules"
        ) from exc


def _apply_trace_ufunc(
    ufunc: np.ufunc,
    inputs: tuple[object, ...],
    context: _WholeProgramTraceContext,
) -> TraceADScalar | TraceADArray:
    if ufunc is np.sign and len(inputs) == 1:
        _require_program_ad_elementwise_contract("sign", inputs)
        _raise_program_ad_derivative_losing_elementwise("sign")
    if ufunc is np.heaviside and len(inputs) == 2:
        _require_program_ad_elementwise_contract("heaviside", inputs)
        _raise_program_ad_derivative_losing_elementwise("heaviside")
    if ufunc is np.negative and len(inputs) == 1:
        operand = _coerce_trace_array(inputs[0], context)
        _require_program_ad_elementwise_contract("negative", (operand,))
        items = tuple(-item for item in operand._items)
        return items[0] if operand.shape == () else TraceADArray(items, operand.shape, context)
    if (
        ufunc
        in {
            np.sin,
            np.cos,
            np.exp,
            np.expm1,
            np.log,
            np.log1p,
            np.sqrt,
            np.tan,
            np.tanh,
            np.arcsin,
            np.arccos,
            np.reciprocal,
            np.square,
            np.absolute,
        }
        and len(inputs) == 1
    ):
        operand = _coerce_trace_array(inputs[0], context)
        _require_program_ad_elementwise_contract(_program_ad_elementwise_name(ufunc), (operand,))
        items = tuple(_apply_unary_trace_ufunc(ufunc, item) for item in operand._items)
        return items[0] if operand.shape == () else TraceADArray(items, operand.shape, context)
    if (
        ufunc
        in {
            np.add,
            np.subtract,
            np.multiply,
            np.divide,
            np.power,
            np.maximum,
            np.minimum,
        }
        and len(inputs) == 2
    ):
        left = _coerce_trace_array(inputs[0], context)
        right = _coerce_trace_array(inputs[1], context)
        _require_program_ad_elementwise_contract(
            _program_ad_elementwise_name(ufunc), (left, right)
        )
        shape = _broadcast_shape(left.shape, right.shape)
        left = _broadcast_trace_array(left, shape, context)
        right = _broadcast_trace_array(right, shape, context)
        items = tuple(
            _apply_binary_trace_ufunc(ufunc, lhs, rhs)
            for lhs, rhs in zip(left._items, right._items, strict=True)
        )
        return items[0] if shape == () else TraceADArray(items, shape, context)
    if ufunc is np.matmul and len(inputs) == 2:
        return _trace_matmul(inputs[0], inputs[1], context)
    raise ValueError(f"unsupported whole-program AD NumPy ufunc {ufunc.__name__}")


def _apply_unary_trace_ufunc(ufunc: np.ufunc, arg: TraceADScalar) -> TraceADScalar:
    if ufunc is np.sin:
        return arg.context.make(
            "sin", (arg.name,), float(np.sin(arg.primal)), float(np.cos(arg.primal)) * arg.tangent
        )
    if ufunc is np.cos:
        return arg.context.make(
            "cos", (arg.name,), float(np.cos(arg.primal)), -float(np.sin(arg.primal)) * arg.tangent
        )
    if ufunc is np.exp:
        primal = float(np.exp(arg.primal))
        return arg.context.make("exp", (arg.name,), primal, primal * arg.tangent)
    if ufunc is np.expm1:
        primal = float(np.expm1(arg.primal))
        return arg.context.make("expm1", (arg.name,), primal, np.exp(arg.primal) * arg.tangent)
    if ufunc is np.log:
        if arg.primal <= 0.0:
            raise ValueError("whole-program AD log input must be positive")
        return arg.context.make(
            "log", (arg.name,), float(np.log(arg.primal)), arg.tangent / arg.primal
        )
    if ufunc is np.log1p:
        if arg.primal <= -1.0:
            raise ValueError("whole-program AD log1p input must be greater than -1")
        return arg.context.make(
            "log1p",
            (arg.name,),
            float(np.log1p(arg.primal)),
            arg.tangent / (1.0 + arg.primal),
        )
    if ufunc is np.sqrt:
        if arg.primal <= 0.0:
            raise ValueError("whole-program AD sqrt input must be positive")
        primal = float(np.sqrt(arg.primal))
        return arg.context.make("sqrt", (arg.name,), primal, arg.tangent / (2.0 * primal))
    if ufunc is np.tan:
        cosine = float(np.cos(arg.primal))
        if abs(cosine) <= 1.0e-15:
            raise ValueError("whole-program AD tan input must have non-zero cosine")
        primal = float(np.tan(arg.primal))
        return arg.context.make("tan", (arg.name,), primal, arg.tangent / cosine**2)
    if ufunc is np.tanh:
        primal = float(np.tanh(arg.primal))
        return arg.context.make("tanh", (arg.name,), primal, (1.0 - primal**2) * arg.tangent)
    if ufunc in {np.arcsin, np.arccos}:
        if abs(arg.primal) >= 1.0:
            raise ValueError(
                f"whole-program AD {ufunc.__name__} input must be strictly inside (-1, 1)"
            )
        scale = 1.0 / float(np.sqrt(1.0 - arg.primal**2))
        if ufunc is np.arccos:
            scale = -scale
        return arg.context.make(
            ufunc.__name__, (arg.name,), float(ufunc(arg.primal)), scale * arg.tangent
        )
    if ufunc is np.reciprocal:
        if arg.primal == 0.0:
            raise ValueError("whole-program AD reciprocal input must be non-zero")
        primal = 1.0 / arg.primal
        return arg.context.make(
            "reciprocal",
            (arg.name,),
            primal,
            -arg.tangent / arg.primal**2,
        )
    if ufunc is np.square:
        return arg.context.make(
            "square", (arg.name,), arg.primal**2, 2.0 * arg.primal * arg.tangent
        )
    if ufunc is np.absolute:
        if arg.primal == 0.0:
            raise ValueError("whole-program AD absolute value is non-differentiable at zero")
        sign = 1.0 if arg.primal > 0.0 else -1.0
        return arg.context.make("abs", (arg.name,), abs(arg.primal), sign * arg.tangent)
    raise ValueError(f"unsupported whole-program AD NumPy ufunc {ufunc.__name__}")


def _apply_binary_trace_ufunc(
    ufunc: np.ufunc,
    left: TraceADScalar,
    right: TraceADScalar,
) -> TraceADScalar:
    if ufunc is np.add:
        return left + right
    if ufunc is np.subtract:
        return left - right
    if ufunc is np.multiply:
        return left * right
    if ufunc is np.divide:
        return left / right
    if ufunc is np.power:
        return left**right
    if ufunc is np.maximum:
        if left.primal == right.primal:
            raise ValueError("whole-program AD maximum is non-differentiable at equal inputs")
        chosen = left if left.primal >= right.primal else right
        return left.context.make("maximum", (left.name, right.name), chosen.primal, chosen.tangent)
    if ufunc is np.minimum:
        if left.primal == right.primal:
            raise ValueError("whole-program AD minimum is non-differentiable at equal inputs")
        chosen = left if left.primal <= right.primal else right
        return left.context.make("minimum", (left.name, right.name), chosen.primal, chosen.tangent)
    raise ValueError(f"unsupported whole-program AD NumPy ufunc {ufunc.__name__}")


def _trace_array_sum(array: TraceADArray, axis: int | None = None) -> TraceADScalar | TraceADArray:
    _require_program_ad_reduction_contract("sum", (array, axis))
    if not array._items:
        raise ValueError("whole-program AD array reductions require at least one element")
    if axis is None:
        total = array._items[0]
        for item in array._items[1:]:
            total = total + item
        return total
    axis = _normalise_axis("axis", axis, array.ndim)
    reduced_shape = array.shape[:axis] + array.shape[axis + 1 :]
    if reduced_shape == ():
        total = array._items[0]
        for item in array._items[1:]:
            total = total + item
        return total
    items: list[TraceADScalar] = []
    for reduced_flat in range(int(np.prod(reduced_shape))):
        reduced_index = np.unravel_index(reduced_flat, reduced_shape)
        source_index = reduced_index[:axis] + (0,) + reduced_index[axis:]
        total = array._items[int(np.ravel_multi_index(source_index, array.shape))]
        for axis_index in range(1, array.shape[axis]):
            source_index = reduced_index[:axis] + (axis_index,) + reduced_index[axis:]
            total = total + array._items[int(np.ravel_multi_index(source_index, array.shape))]
        items.append(total)
    return TraceADArray(tuple(items), reduced_shape, array.context)


def _normalise_trapezoid_axis(axis: object, ndim: int) -> int:
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD np.trapezoid axis must be a static integer")
    try:
        return _normalise_axis("axis", int(axis), ndim)
    except ValueError as exc:
        if "out of bounds" in str(exc):
            raise ValueError("program AD np.trapezoid axis out of bounds") from exc
        raise


def _trace_trapezoid_widths(
    array: TraceADArray,
    *,
    x: object,
    dx: object,
    axis: int,
) -> NDArray[np.float64]:
    axis_size = array.shape[axis]
    if axis_size < 2:
        raise ValueError("program AD np.trapezoid requires at least two samples along axis")
    width_shape = array.shape[:axis] + (axis_size - 1,) + array.shape[axis + 1 :]
    if isinstance(x, (TraceADArray, TraceADScalar)):
        raise ValueError("program AD np.trapezoid grid x must be static real numeric")
    if x is None:
        dx_value = _as_real_scalar("program AD np.trapezoid dx", dx)
        if not np.isfinite(dx_value):
            raise ValueError("program AD np.trapezoid dx must be finite")
        return np.full(width_shape, dx_value, dtype=np.float64)
    dx_value = _as_real_scalar("program AD np.trapezoid dx", dx)
    if dx_value != 1.0:
        raise ValueError("program AD np.trapezoid accepts either x or dx, not both")
    x_array = _as_real_numeric_array("program AD np.trapezoid x", x)
    if not bool(np.all(np.isfinite(x_array))):
        raise ValueError("program AD np.trapezoid x must contain only finite values")
    if x_array.ndim == 1:
        if x_array.shape[0] != axis_size:
            raise ValueError("program AD np.trapezoid x must match the integration axis")
        reshape = [1 for _ in array.shape]
        reshape[axis] = axis_size - 1
        return cast(
            NDArray[np.float64],
            np.broadcast_to(np.diff(x_array).reshape(tuple(reshape)), width_shape).copy(),
        )
    if tuple(x_array.shape) != array.shape:
        raise ValueError(
            "program AD np.trapezoid x must match the integration axis or full array shape"
        )
    return cast(NDArray[np.float64], np.diff(x_array, axis=axis))


def _trace_trapezoid(
    array: TraceADArray,
    *,
    x: object = None,
    dx: object = 1.0,
    axis: object = -1,
) -> TraceADScalar | TraceADArray:
    _require_program_ad_reduction_contract("trapezoid", (array, x, dx, axis))
    axis_index = _normalise_trapezoid_axis(axis, array.ndim)
    widths = _trace_trapezoid_widths(array, x=x, dx=dx, axis=axis_index)
    reduced_shape = array.shape[:axis_index] + array.shape[axis_index + 1 :]

    def integrate_at(reduced_index: tuple[int, ...]) -> TraceADScalar:
        total = _coerce_trace_scalar(0.0, array.context)
        for segment_index in range(array.shape[axis_index] - 1):
            left_index = reduced_index[:axis_index] + (segment_index,) + reduced_index[axis_index:]
            right_index = (
                reduced_index[:axis_index] + (segment_index + 1,) + reduced_index[axis_index:]
            )
            width_index = (
                reduced_index[:axis_index] + (segment_index,) + reduced_index[axis_index:]
            )
            left = array._items[int(np.ravel_multi_index(left_index, array.shape))]
            right = array._items[int(np.ravel_multi_index(right_index, array.shape))]
            total = total + (left + right) * (0.5 * float(widths[width_index]))
        return total

    if reduced_shape == ():
        return integrate_at(())
    items = tuple(integrate_at(tuple(index)) for index in np.ndindex(reduced_shape))
    return TraceADArray(items, reduced_shape, array.context)


def _trace_gradient_axis(
    array: TraceADArray,
    *,
    axis: int,
    spacing: _GradientSpacing,
    edge_order: int,
) -> TraceADArray:
    items: list[TraceADScalar] = []
    for flat_index in range(array.size):
        target_index = np.unravel_index(flat_index, array.shape)
        total = _coerce_trace_scalar(0.0, array.context)
        for source_axis_index, coefficient in _gradient_axis_coefficients(
            int(target_index[axis]),
            array.shape[axis],
            spacing,
            edge_order,
        ):
            source_index = target_index[:axis] + (source_axis_index,) + target_index[axis + 1 :]
            source = array._items[int(np.ravel_multi_index(source_index, array.shape))]
            total = total + source * coefficient
        items.append(total)
    return TraceADArray(tuple(items), array.shape, array.context)


def _trace_gradient(
    array: TraceADArray,
    *,
    spacings: tuple[object, ...],
    axis: object = None,
    edge_order: object = 1,
) -> TraceADArray | list[TraceADArray]:
    edge = _normalise_gradient_edge_order(edge_order)
    axes = _normalise_gradient_axes(axis, array.ndim)
    spacing_specs = _normalise_gradient_spacings(spacings, axes, array.shape)
    _require_program_ad_stencil_contract("gradient", (array, spacings, axis, edge))
    gradients = [
        _trace_gradient_axis(array, axis=axis_index, spacing=spacing, edge_order=edge)
        for axis_index, spacing in zip(axes, spacing_specs, strict=True)
    ]
    return gradients[0] if len(gradients) == 1 else gradients


def _normalise_interp_trace_values(
    fp: object, *, grid_size: int, context: _WholeProgramTraceContext
) -> tuple[TraceADScalar, ...]:
    if isinstance(fp, TraceADArray):
        if fp.ndim != 1 or fp.size != grid_size:
            raise ValueError("program AD np.interp fp values must match xp grid length")
        return tuple(fp._items)
    if isinstance(fp, TraceADScalar):
        raise ValueError("program AD np.interp fp values must be one-dimensional")
    values = _as_real_numeric_array("program AD np.interp fp values", fp)
    if values.ndim != 1 or values.size != grid_size:
        raise ValueError("program AD np.interp fp values must match xp grid length")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("program AD np.interp fp values must contain only finite values")
    return tuple(_coerce_trace_scalar(float(value), context) for value in values)


def _normalise_interp_boundary(
    name: str, value: object, context: _WholeProgramTraceContext
) -> TraceADScalar | None:
    if value is None:
        return None
    if isinstance(value, (TraceADArray, TraceADScalar)):
        raise ValueError(f"program AD np.interp {name} boundary must be static real numeric")
    return _coerce_trace_scalar(_as_real_scalar(f"program AD np.interp {name}", value), context)


def _normalise_interp_samples(
    x: object, *, context: _WholeProgramTraceContext
) -> tuple[tuple[TraceADScalar, ...], tuple[int, ...]]:
    if isinstance(x, TraceADArray):
        return tuple(x._items), x.shape
    if isinstance(x, TraceADScalar):
        return (x,), ()
    samples = _as_real_numeric_array("program AD np.interp x samples", x)
    if not bool(np.all(np.isfinite(samples))):
        raise ValueError("program AD np.interp x samples must contain only finite values")
    return tuple(
        _coerce_trace_scalar(float(value), context) for value in samples.reshape(-1)
    ), tuple(samples.shape)


def _trace_interp_scalar(
    sample: TraceADScalar,
    *,
    grid: NDArray[np.float64],
    values: tuple[TraceADScalar, ...],
    left: TraceADScalar | None,
    right: TraceADScalar | None,
) -> TraceADScalar:
    primal = sample.primal
    if not math.isfinite(primal):
        raise ValueError("program AD np.interp x samples must contain only finite values")
    if bool(np.any(grid == primal)):
        raise ValueError("program AD np.interp differentiable samples must avoid grid knots")
    if primal < float(grid[0]):
        return values[0] if left is None else left
    if primal > float(grid[-1]):
        return values[-1] if right is None else right
    segment = int(np.searchsorted(grid, primal, side="right") - 1)
    lower = float(grid[segment])
    upper = float(grid[segment + 1])
    weight = (sample - lower) / (upper - lower)
    return values[segment] + (values[segment + 1] - values[segment]) * weight


def _trace_interp(
    x: object,
    xp: object,
    fp: object,
    *,
    left: object = None,
    right: object = None,
    period: object = None,
    context: _WholeProgramTraceContext,
) -> TraceADScalar | TraceADArray:
    if period is not None:
        raise ValueError("program AD np.interp period is not supported")
    _require_program_ad_interpolation_contract("interp", (x, xp, fp, left, right, period))
    grid = _normalise_interp_grid(xp)
    values = _normalise_interp_trace_values(fp, grid_size=grid.size, context=context)
    left_value = _normalise_interp_boundary("left", left, context)
    right_value = _normalise_interp_boundary("right", right, context)
    samples, shape = _normalise_interp_samples(x, context=context)
    outputs = tuple(
        _trace_interp_scalar(
            sample,
            grid=grid,
            values=values,
            left=left_value,
            right=right_value,
        )
        for sample in samples
    )
    if shape == ():
        return outputs[0]
    return TraceADArray(outputs, shape, context)


def _normalise_convolve_operand(
    name: str, operand: object, context: _WholeProgramTraceContext
) -> tuple[TraceADScalar, ...]:
    if isinstance(operand, TraceADArray):
        if operand.ndim != 1:
            raise ValueError(f"program AD np.convolve {name} operand must be one-dimensional")
        if operand.size == 0:
            raise ValueError(f"program AD np.convolve {name} operand must be non-empty")
        return tuple(operand._items)
    if isinstance(operand, TraceADScalar):
        raise ValueError(f"program AD np.convolve {name} operand must be one-dimensional")
    values = _as_real_numeric_array(f"program AD np.convolve {name} operand", operand)
    if values.ndim != 1:
        raise ValueError(f"program AD np.convolve {name} operand must be one-dimensional")
    if values.size == 0:
        raise ValueError(f"program AD np.convolve {name} operand must be non-empty")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"program AD np.convolve {name} operand must contain only finite values")
    return tuple(_coerce_trace_scalar(float(value), context) for value in values)


def _trace_convolve(
    left: object,
    right: object,
    *,
    context: _WholeProgramTraceContext,
    mode: object = "full",
) -> TraceADArray:
    mode_value = _normalise_convolve_mode(mode)
    _require_program_ad_signal_contract("convolve", (left, right, mode_value))
    left_values = _normalise_convolve_operand("left", left, context)
    right_values = _normalise_convolve_operand("right", right, context)
    full_items: list[TraceADScalar] = []
    for output_index in range(len(left_values) + len(right_values) - 1):
        total = _coerce_trace_scalar(0.0, context)
        left_start = max(0, output_index - len(right_values) + 1)
        left_stop = min(len(left_values), output_index + 1)
        for left_index in range(left_start, left_stop):
            right_index = output_index - left_index
            total = total + left_values[left_index] * right_values[right_index]
        full_items.append(total)
    start, stop = _convolve_output_window(len(left_values), len(right_values), mode_value)
    return TraceADArray(tuple(full_items[start:stop]), (stop - start,), context)


def _normalise_correlate_operand(
    name: str, operand: object, context: _WholeProgramTraceContext
) -> tuple[TraceADScalar, ...]:
    if isinstance(operand, TraceADArray):
        if operand.ndim != 1:
            raise ValueError(f"program AD np.correlate {name} operand must be one-dimensional")
        if operand.size == 0:
            raise ValueError(f"program AD np.correlate {name} operand must be non-empty")
        return tuple(operand._items)
    if isinstance(operand, TraceADScalar):
        raise ValueError(f"program AD np.correlate {name} operand must be one-dimensional")
    values = _as_real_numeric_array(f"program AD np.correlate {name} operand", operand)
    if values.ndim != 1:
        raise ValueError(f"program AD np.correlate {name} operand must be one-dimensional")
    if values.size == 0:
        raise ValueError(f"program AD np.correlate {name} operand must be non-empty")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"program AD np.correlate {name} operand must contain only finite values")
    return tuple(_coerce_trace_scalar(float(value), context) for value in values)


def _trace_correlate(
    left: object,
    right: object,
    *,
    context: _WholeProgramTraceContext,
    mode: object = "valid",
) -> TraceADArray:
    mode_value = _normalise_correlate_mode(mode)
    _require_program_ad_signal_contract("correlate", (left, right, mode_value))
    left_values = _normalise_correlate_operand("left", left, context)
    right_values = tuple(reversed(_normalise_correlate_operand("right", right, context)))
    full_items: list[TraceADScalar] = []
    for output_index in range(len(left_values) + len(right_values) - 1):
        total = _coerce_trace_scalar(0.0, context)
        left_start = max(0, output_index - len(right_values) + 1)
        left_stop = min(len(left_values), output_index + 1)
        for left_index in range(left_start, left_stop):
            right_index = output_index - left_index
            total = total + left_values[left_index] * right_values[right_index]
        full_items.append(total)
    start, stop = _convolve_output_window(len(left_values), len(right_values), mode_value)
    return TraceADArray(tuple(full_items[start:stop]), (stop - start,), context)


def _trace_cumsum(array: TraceADArray, axis: int | None = None) -> TraceADArray:
    _require_program_ad_cumulative_contract("cumsum", (array, axis))
    if not array._items:
        raise ValueError("program AD cumulative sum requires at least one element")
    if axis is None:
        items: list[TraceADScalar] = []
        total = array._items[0]
        items.append(total)
        for item in array._items[1:]:
            total = total + item
            items.append(total)
        return TraceADArray(tuple(items), (array.size,), array.context)
    axis = _normalise_axis("axis", axis, array.ndim)
    axis_items: list[TraceADScalar] = []
    for flat_index in range(array.size):
        target_index = np.unravel_index(flat_index, array.shape)
        source_index = target_index[:axis] + (0,) + target_index[axis + 1 :]
        total = array._items[int(np.ravel_multi_index(source_index, array.shape))]
        for axis_index in range(1, target_index[axis] + 1):
            source_index = target_index[:axis] + (axis_index,) + target_index[axis + 1 :]
            total = total + array._items[int(np.ravel_multi_index(source_index, array.shape))]
        axis_items.append(total)
    return TraceADArray(tuple(axis_items), array.shape, array.context)


def _trace_array_prod(
    array: TraceADArray, axis: int | None = None
) -> TraceADScalar | TraceADArray:
    _require_program_ad_reduction_contract("prod", (array, axis))
    if not array._items:
        raise ValueError("program AD array product reductions require at least one element")
    if axis is None:
        total = array._items[0]
        for item in array._items[1:]:
            total = total * item
        return total
    axis = _normalise_axis("axis", axis, array.ndim)
    reduced_shape = array.shape[:axis] + array.shape[axis + 1 :]
    if reduced_shape == ():
        total = array._items[0]
        for item in array._items[1:]:
            total = total * item
        return total
    items: list[TraceADScalar] = []
    for reduced_flat in range(int(np.prod(reduced_shape))):
        reduced_index = np.unravel_index(reduced_flat, reduced_shape)
        source_index = reduced_index[:axis] + (0,) + reduced_index[axis:]
        total = array._items[int(np.ravel_multi_index(source_index, array.shape))]
        for axis_index in range(1, array.shape[axis]):
            source_index = reduced_index[:axis] + (axis_index,) + reduced_index[axis:]
            total = total * array._items[int(np.ravel_multi_index(source_index, array.shape))]
        items.append(total)
    return TraceADArray(tuple(items), reduced_shape, array.context)


def _trace_cumprod(array: TraceADArray, axis: int | None = None) -> TraceADArray:
    _require_program_ad_cumulative_contract("cumprod", (array, axis))
    if not array._items:
        raise ValueError("program AD cumulative product requires at least one element")
    if axis is None:
        prod_items: list[TraceADScalar] = []
        total = array._items[0]
        prod_items.append(total)
        for item in array._items[1:]:
            total = total * item
            prod_items.append(total)
        return TraceADArray(tuple(prod_items), (array.size,), array.context)
    axis = _normalise_axis("axis", axis, array.ndim)
    axis_prod_items: list[TraceADScalar] = []
    for flat_index in range(array.size):
        target_index = np.unravel_index(flat_index, array.shape)
        source_index = target_index[:axis] + (0,) + target_index[axis + 1 :]
        total = array._items[int(np.ravel_multi_index(source_index, array.shape))]
        for axis_index in range(1, target_index[axis] + 1):
            source_index = target_index[:axis] + (axis_index,) + target_index[axis + 1 :]
            total = total * array._items[int(np.ravel_multi_index(source_index, array.shape))]
        axis_prod_items.append(total)
    return TraceADArray(tuple(axis_prod_items), array.shape, array.context)


def _trace_diff(array: TraceADArray, *, n: object, axis: int) -> TraceADArray:
    _require_program_ad_cumulative_contract("diff", (array, n, axis))
    if not isinstance(n, (int, np.integer)):
        raise ValueError("program AD np.diff requires non-negative integer n")
    order = int(n)
    if order < 0:
        raise ValueError("program AD np.diff requires non-negative integer n")
    result = array.copy()
    for _ in range(order):
        result = _trace_first_diff(result, axis=axis)
    return result


def _trace_first_diff(array: TraceADArray, *, axis: int) -> TraceADArray:
    axis = _normalise_axis("axis", axis, array.ndim)
    target_axis_size = max(array.shape[axis] - 1, 0)
    target_shape = array.shape[:axis] + (target_axis_size,) + array.shape[axis + 1 :]
    if target_axis_size == 0:
        return TraceADArray((), target_shape, array.context)
    items: list[TraceADScalar] = []
    for target_flat in range(int(np.prod(target_shape))):
        target_index = np.unravel_index(target_flat, target_shape)
        left_index = target_index[:axis] + (target_index[axis],) + target_index[axis + 1 :]
        right_index = target_index[:axis] + (target_index[axis] + 1,) + target_index[axis + 1 :]
        items.append(
            array._items[int(np.ravel_multi_index(right_index, array.shape))]
            - array._items[int(np.ravel_multi_index(left_index, array.shape))]
        )
    return TraceADArray(tuple(items), target_shape, array.context)


def _trace_variance(
    array: TraceADArray,
    *,
    axis: int | None,
    ddof: object,
) -> TraceADScalar | TraceADArray:
    if not array._items:
        raise ValueError("program AD variance reductions require at least one element")
    count = array.size if axis is None else array.shape[_normalise_axis("axis", axis, array.ndim)]
    ddof_int = _normalise_ddof(ddof, count)
    mean = array.mean(axis=axis)
    if axis is None:
        if not isinstance(mean, TraceADScalar):
            raise ValueError("program AD variance scalar mean expected")
        squared = tuple((item - mean) * (item - mean) for item in array._items)
        total = squared[0]
        for item in squared[1:]:
            total = total + item
        return total / float(count - ddof_int)
    axis = _normalise_axis("axis", axis, array.ndim)
    if not isinstance(mean, TraceADArray):
        raise ValueError("program AD variance axis mean expected an array")
    reduced_shape = array.shape[:axis] + array.shape[axis + 1 :]
    if reduced_shape == ():
        return _trace_variance(array, axis=None, ddof=ddof_int)
    items: list[TraceADScalar] = []
    for reduced_flat in range(int(np.prod(reduced_shape))):
        reduced_index = np.unravel_index(reduced_flat, reduced_shape)
        centre = mean._items[reduced_flat]
        source_index = reduced_index[:axis] + (0,) + reduced_index[axis:]
        delta = array._items[int(np.ravel_multi_index(source_index, array.shape))] - centre
        total = delta * delta
        for axis_index in range(1, array.shape[axis]):
            source_index = reduced_index[:axis] + (axis_index,) + reduced_index[axis:]
            delta = array._items[int(np.ravel_multi_index(source_index, array.shape))] - centre
            total = total + delta * delta
        items.append(total / float(count - ddof_int))
    return TraceADArray(tuple(items), reduced_shape, array.context)


def _trace_std(
    array: TraceADArray,
    *,
    axis: int | None,
    ddof: object,
) -> TraceADScalar | TraceADArray:
    variance = _trace_variance(array, axis=axis, ddof=ddof)
    if isinstance(variance, TraceADScalar):
        return _apply_trace_ufunc(np.sqrt, (variance,), array.context)
    return _apply_trace_ufunc(np.sqrt, (variance,), array.context)


def _trace_extreme(
    array: TraceADArray,
    *,
    axis: int | None,
    choose_max: bool,
) -> TraceADScalar | TraceADArray:
    op_name = "np.max" if choose_max else "np.min"
    if array.size == 0:
        raise ValueError(f"program AD {op_name} requires at least one element")
    if axis is None:
        return _trace_strict_extreme(array._items, op_name=op_name, choose_max=choose_max)
    axis = _normalise_axis("axis", axis, array.ndim)
    if array.shape[axis] == 0:
        raise ValueError(f"program AD {op_name} requires at least one element")
    reduced_shape = array.shape[:axis] + array.shape[axis + 1 :]
    if reduced_shape == ():
        candidates = tuple(
            array._items[int(np.ravel_multi_index((axis_index,), array.shape))]
            for axis_index in range(array.shape[axis])
        )
        return _trace_strict_extreme(candidates, op_name=op_name, choose_max=choose_max)
    items: list[TraceADScalar] = []
    for reduced_flat in range(int(np.prod(reduced_shape))):
        reduced_index = np.unravel_index(reduced_flat, reduced_shape)
        candidates = tuple(
            array._items[
                int(
                    np.ravel_multi_index(
                        reduced_index[:axis] + (axis_index,) + reduced_index[axis:],
                        array.shape,
                    )
                )
            ]
            for axis_index in range(array.shape[axis])
        )
        items.append(_trace_strict_extreme(candidates, op_name=op_name, choose_max=choose_max))
    return TraceADArray(tuple(items), reduced_shape, array.context)


def _trace_strict_extreme(
    items: Sequence[TraceADScalar],
    *,
    op_name: str,
    choose_max: bool,
) -> TraceADScalar:
    if not items:
        raise ValueError(f"program AD {op_name} requires at least one element")
    selected = items[0]
    for item in items[1:]:
        if item.primal == selected.primal:
            raise ValueError(f"program AD {op_name} is non-differentiable at ties")
        if (item.primal > selected.primal) if choose_max else (item.primal < selected.primal):
            selected = item
    return selected


def _trace_take(
    array: TraceADArray,
    indices: object,
    *,
    axis: int | None,
    mode: str,
) -> TraceADScalar | TraceADArray:
    _require_program_ad_array_contract("take", (array, indices, axis, mode))
    mode_name = _program_ad_array_take_mode(mode, context="trace")
    if isinstance(indices, (TraceADScalar, TraceADArray)):
        raise ValueError("program AD np.take requires static integer indices")
    raw_indices = np.asarray(indices)
    if raw_indices.dtype.kind not in {"i", "u"}:
        raise ValueError("program AD np.take requires static integer indices")
    source = np.arange(array.size, dtype=np.int64).reshape(array.shape)
    try:
        selected = np.take(source, raw_indices, axis=axis, mode=mode_name)
    except (IndexError, ValueError) as exc:
        if mode_name == "raise":
            raise ValueError("program AD np.take indices must be in bounds") from exc
        raise ValueError("program AD np.take requires axis-compatible static indices") from exc
    selected_array = np.asarray(selected)
    if selected_array.shape == ():
        return array._items[int(selected_array)]
    local_indices = tuple(int(index) for index in selected_array.reshape(-1))
    source_indices = tuple(_trace_array_source_indices(array)[index] for index in local_indices)
    items = tuple(array._items[index] for index in local_indices)
    array.context.record_array_view_aliases("take", source_indices, items)
    return TraceADArray(
        items, tuple(int(dim) for dim in selected_array.shape), array.context, source_indices
    )


def _trace_take_along_axis(
    array: TraceADArray,
    indices: object,
    *,
    axis: object,
) -> TraceADArray:
    _require_program_ad_array_contract("take_along_axis", (array, indices, axis))
    if isinstance(indices, (TraceADScalar, TraceADArray)):
        raise ValueError("program AD np.take_along_axis requires static integer indices")
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD np.take_along_axis requires a static integer axis")
    raw_indices = _program_ad_array_take_indices(indices)
    normalised_axis = _normalise_axis("axis", int(axis), array.ndim)
    source = np.arange(array.size, dtype=np.int64).reshape(array.shape)
    try:
        selected = np.take_along_axis(source, raw_indices, axis=normalised_axis)
    except (IndexError, ValueError) as exc:
        raise ValueError(
            "program AD np.take_along_axis requires static in-bounds indices "
            "with shape compatible with the source"
        ) from exc
    selected_array = np.asarray(selected)
    items = tuple(array._items[int(index)] for index in selected_array.reshape(-1))
    return TraceADArray(items, tuple(int(dim) for dim in selected_array.shape), array.context)


def _trace_delete(
    array: TraceADArray,
    obj: object,
    *,
    axis: object,
) -> TraceADArray:
    _require_program_ad_array_contract("delete", (array, obj, axis))
    delete_obj = _program_ad_array_delete_object(obj, context="trace")
    source: NDArray[np.int64]
    if axis is None:
        source = np.arange(array.size, dtype=np.int64).reshape(-1)
        normalised_axis = None
    else:
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
            raise ValueError("program AD np.delete requires a static integer axis or None")
        normalised_axis = _normalise_axis("axis", int(axis), array.ndim)
        source = np.arange(array.size, dtype=np.int64).reshape(array.shape)
    try:
        selected = np.delete(source, cast(Any, delete_obj), axis=normalised_axis)
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "program AD np.delete requires static in-bounds deletion selectors "
            "and a compatible axis"
        ) from exc
    selected_array = np.asarray(selected, dtype=np.int64)
    items = tuple(array._items[int(index)] for index in selected_array.reshape(-1))
    return TraceADArray(items, tuple(int(dim) for dim in selected_array.shape), array.context)


def _program_ad_contains_trace_value(value: object) -> bool:
    if isinstance(value, (TraceADScalar, TraceADArray)):
        return True
    if isinstance(value, Mapping):
        return any(
            _program_ad_contains_trace_value(key) or _program_ad_contains_trace_value(item)
            for key, item in value.items()
        )
    if isinstance(value, np.ndarray) and value.dtype == object:
        return any(_program_ad_contains_trace_value(item) for item in value.reshape(-1))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_program_ad_contains_trace_value(item) for item in value)
    return False


def _trace_pad(
    array: TraceADArray,
    pad_width: object,
    *,
    mode: object,
    constant_values: object,
) -> TraceADArray:
    _require_program_ad_array_contract("pad", (array, pad_width, mode, constant_values))
    _program_ad_array_pad_mode(mode, context="trace")
    flat_indices, flat_constants, output_shape = _program_ad_array_pad_layout(
        array.shape,
        pad_width,
        constant_values,
        context="trace",
    )
    items = tuple(
        array._items[int(index)]
        if int(index) >= 0
        else _trace_constant(float(flat_constants[position]), array.context)
        for position, index in enumerate(flat_indices)
    )
    return TraceADArray(items, output_shape, array.context)


def _trace_insert(
    array: TraceADArray,
    obj: object,
    values: object,
    *,
    axis: object,
) -> TraceADArray:
    _require_program_ad_array_contract("insert", (array, obj, values, axis))
    flat_indices, flat_constants, output_shape = _program_ad_array_insert_layout(
        array.shape,
        obj,
        values,
        axis,
        context="trace",
    )
    items = tuple(
        array._items[int(index)]
        if int(index) >= 0
        else _trace_constant(float(flat_constants[position]), array.context)
        for position, index in enumerate(flat_indices)
    )
    return TraceADArray(items, output_shape, array.context)


def _raise_index_selection_boundary(
    name: str = "argmax",
    args: tuple[object, ...] = (),
) -> NoReturn:
    if name in _PROGRAM_AD_SELECTION_IDENTITIES and args:
        _require_program_ad_selection_contract(name, args)
    raise ValueError(
        "program AD argmax/argmin/argsort index selection semantics are registered "
        "nondifferentiable integer selection primitives and fail closed"
    )


def _raise_spectral_linalg_boundary(function_name: str) -> NoReturn:
    raise ValueError(
        f"program AD np.linalg.{function_name} spectral semantics require an explicit "
        "differentiable primitive rule for eigenvalue degeneracy, singular-value "
        "multiplicity, and nondifferentiable selection policy"
    )


def _trace_transpose(
    values: object,
    context: _WholeProgramTraceContext,
    *,
    axes: tuple[int, ...] | None = None,
) -> TraceADArray:
    array = _coerce_trace_array(values, context)
    _require_program_ad_shape_contract("transpose", (array, axes))
    if array.ndim < 2:
        return array.copy()
    if axes is None:
        axes = tuple(reversed(range(array.ndim)))
    if len(axes) != array.ndim:
        raise ValueError("whole-program AD np.transpose axes must match array rank")
    normalised_axes = tuple(_normalise_axis("axis", axis, array.ndim) for axis in axes)
    if sorted(normalised_axes) != list(range(array.ndim)):
        raise ValueError("whole-program AD np.transpose axes must be a permutation")
    target_shape = tuple(array.shape[axis] for axis in normalised_axes)
    inverse_axes = tuple(normalised_axes.index(axis) for axis in range(array.ndim))
    items: list[TraceADScalar] = []
    for target_flat in range(int(np.prod(target_shape))):
        target_index = np.unravel_index(target_flat, target_shape)
        source_index = tuple(target_index[inverse_axes[axis]] for axis in range(array.ndim))
        items.append(array._items[int(np.ravel_multi_index(source_index, array.shape))])
    local_indices = tuple(
        int(np.ravel_multi_index(source_index, array.shape))
        for source_index in (
            tuple(target_index[inverse_axes[axis]] for axis in range(array.ndim))
            for target_index in np.ndindex(target_shape)
        )
    )
    source_indices = tuple(
        _trace_array_source_indices(array)[local_index] for local_index in local_indices
    )
    context.record_array_view_aliases("transpose", source_indices, items)
    return TraceADArray(tuple(items), target_shape, context, source_indices)


def _trace_dot(
    left: object,
    right: object,
    context: _WholeProgramTraceContext,
) -> TraceADScalar:
    lhs = _coerce_trace_array(left, context)
    rhs = _coerce_trace_array(right, context)
    _require_program_ad_product_contract("dot", (lhs, rhs))
    if lhs.ndim == 1 and rhs.ndim == 1 and lhs.shape == rhs.shape:
        total = lhs._items[0] * rhs._items[0]
        for left_item, right_item in zip(lhs._items[1:], rhs._items[1:], strict=True):
            total = total + left_item * right_item
        return total
    result = _trace_matmul(lhs, rhs, context)
    if isinstance(result, TraceADArray) and result.shape == ():
        return result.item()
    if isinstance(result, TraceADScalar):
        return result
    raise ValueError("whole-program AD np.dot result must be scalar for this operand pair")


def _trace_vdot(
    left: object,
    right: object,
    context: _WholeProgramTraceContext,
) -> TraceADScalar:
    lhs = _coerce_trace_array(left, context)
    rhs = _coerce_trace_array(right, context)
    _require_program_ad_product_contract("vdot", (lhs, rhs))
    if lhs.size != rhs.size:
        raise ValueError("program AD np.vdot flattened operands must have matching size")
    if lhs.size == 0:
        return _coerce_trace_scalar(0.0, context)
    total = lhs._items[0] * rhs._items[0]
    for left_item, right_item in zip(lhs._items[1:], rhs._items[1:], strict=True):
        total = total + left_item * right_item
    return total


def _trace_inner(
    left: object,
    right: object,
    context: _WholeProgramTraceContext,
) -> TraceADScalar | TraceADArray:
    lhs = _coerce_trace_array(left, context)
    rhs = _coerce_trace_array(right, context)
    if lhs.ndim == 0 or rhs.ndim == 0:
        return _trace_multiply_arrays(lhs, rhs, context)
    _require_program_ad_product_contract("inner", (lhs, rhs))
    if lhs.ndim > 2 or rhs.ndim > 2:
        raise ValueError("whole-program AD np.inner supports operands with rank <= 2")
    lhs_outer = lhs.shape[:-1]
    rhs_outer = rhs.shape[:-1]
    shared = lhs.shape[-1]
    if rhs.shape[-1] != shared:
        raise ValueError("whole-program AD np.inner last dimensions must align")
    result_items: list[TraceADScalar] = []
    lhs_rows = int(np.prod(lhs_outer)) if lhs_outer else 1
    rhs_rows = int(np.prod(rhs_outer)) if rhs_outer else 1
    for lhs_row in range(lhs_rows):
        for rhs_row in range(rhs_rows):
            total = lhs._items[lhs_row * shared] * rhs._items[rhs_row * shared]
            for index in range(1, shared):
                total = (
                    total
                    + lhs._items[lhs_row * shared + index] * rhs._items[rhs_row * shared + index]
                )
            result_items.append(total)
    shape = lhs_outer + rhs_outer
    return result_items[0] if shape == () else TraceADArray(tuple(result_items), shape, context)


def _trace_outer(
    left: object,
    right: object,
    context: _WholeProgramTraceContext,
) -> TraceADArray:
    lhs = _coerce_trace_array(left, context)
    rhs = _coerce_trace_array(right, context)
    _require_program_ad_product_contract("outer", (lhs, rhs))
    left_items = tuple(lhs._items)
    right_items = tuple(rhs._items)
    items = tuple(left_item * right_item for left_item in left_items for right_item in right_items)
    return TraceADArray(items, (len(left_items), len(right_items)), context)


def _trace_tensordot(
    left: object,
    right: object,
    context: _WholeProgramTraceContext,
    *,
    axes: object,
) -> TraceADScalar | TraceADArray:
    lhs = _coerce_trace_array(left, context)
    rhs = _coerce_trace_array(right, context)
    _require_program_ad_product_contract("tensordot", (lhs, rhs, axes))
    _left_shape, _right_shape, left_axes, right_axes, output_shape = (
        _normalise_program_ad_product_tensordot_signature(lhs.shape, rhs.shape, axes)
    )
    left_free_axes = tuple(axis for axis in range(lhs.ndim) if axis not in left_axes)
    right_free_axes = tuple(axis for axis in range(rhs.ndim) if axis not in right_axes)
    contraction_shape = tuple(lhs.shape[axis] for axis in left_axes)
    output_items: list[TraceADScalar] = []
    output_indices = np.ndindex(output_shape) if output_shape else iter(((),))
    contraction_indices = tuple(np.ndindex(contraction_shape)) if contraction_shape else ((),)
    for output_index in output_indices:
        output_tuple = tuple(int(index) for index in output_index)
        left_free_index = output_tuple[: len(left_free_axes)]
        right_free_index = output_tuple[len(left_free_axes) :]
        total = _coerce_trace_scalar(0.0, context)
        for contraction_index in contraction_indices:
            left_index = [0 for _ in range(lhs.ndim)]
            right_index = [0 for _ in range(rhs.ndim)]
            for axis, index in zip(left_free_axes, left_free_index, strict=True):
                left_index[axis] = index
            for axis, index in zip(right_free_axes, right_free_index, strict=True):
                right_index[axis] = index
            for left_axis, right_axis, index in zip(
                left_axes, right_axes, contraction_index, strict=True
            ):
                left_index[left_axis] = int(index)
                right_index[right_axis] = int(index)
            lhs_item = lhs._items[int(np.ravel_multi_index(tuple(left_index), lhs.shape))]
            rhs_item = rhs._items[int(np.ravel_multi_index(tuple(right_index), rhs.shape))]
            total = total + lhs_item * rhs_item
        output_items.append(total)
    if output_shape == ():
        return output_items[0]
    return TraceADArray(tuple(output_items), output_shape, context)


def _parse_trace_einsum_subscripts(
    subscripts: str,
    operands: Sequence[TraceADArray],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], dict[str, int]]:
    return _parse_static_einsum_subscripts(
        subscripts,
        tuple(operand.shape for operand in operands),
    )


def _trace_einsum_scalar_at(
    operands: Sequence[TraceADArray],
    input_labels: Sequence[tuple[str, ...]],
    dimensions: Mapping[str, int],
    assignment: Mapping[str, int],
    contraction_labels: tuple[str, ...],
    context: _WholeProgramTraceContext,
) -> TraceADScalar:
    total = _coerce_trace_scalar(0.0, context)
    contraction_shape = tuple(dimensions[label] for label in contraction_labels)
    contraction_indices = np.ndindex(contraction_shape) if contraction_shape else iter(((),))
    for contraction_index in contraction_indices:
        label_indices = dict(assignment)
        label_indices.update(
            {
                label: int(index)
                for label, index in zip(contraction_labels, contraction_index, strict=True)
            }
        )
        term: TraceADScalar | None = None
        for operand, labels in zip(operands, input_labels, strict=True):
            item_index = tuple(label_indices[label] for label in labels)
            item = operand._items[int(np.ravel_multi_index(item_index, operand.shape))]
            term = item if term is None else term * item
        if term is None:
            raise ValueError("whole-program AD np.einsum requires at least one operand")
        total = total + term
    return total


def _trace_einsum(
    subscripts: str,
    operands: Sequence[object],
    context: _WholeProgramTraceContext,
) -> TraceADScalar | TraceADArray:
    normalised = subscripts.replace(" ", "")
    if normalised == "i,i->" and len(operands) == 2:
        return _trace_dot(operands[0], operands[1], context)
    if normalised == "i,j->ij" and len(operands) == 2:
        return _trace_outer(operands[0], operands[1], context)
    if normalised == "ij,j->i" and len(operands) == 2:
        return _trace_matmul(operands[0], operands[1], context)
    if normalised == "i,ij->j" and len(operands) == 2:
        return _trace_matmul(operands[0], operands[1], context)
    if normalised == "ij,jk->ik" and len(operands) == 2:
        return _trace_matmul(operands[0], operands[1], context)
    if normalised == "ii->" and len(operands) == 1:
        return _trace_trace(operands[0], context)
    if normalised == "ii->i" and len(operands) == 1:
        return _trace_diag(operands[0], context)
    trace_operands = tuple(_coerce_trace_array(operand, context) for operand in operands)
    _require_program_ad_product_contract("einsum", (normalised, *trace_operands))
    output_labels, input_labels, dimensions = _parse_trace_einsum_subscripts(
        normalised,
        trace_operands,
    )
    contraction_labels = tuple(
        label
        for label in dict.fromkeys(label for labels in input_labels for label in labels)
        if label not in output_labels
    )
    output_shape = tuple(dimensions[label] for label in output_labels)
    output_items: list[TraceADScalar] = []
    output_indices = np.ndindex(output_shape) if output_shape else iter(((),))
    for output_index in output_indices:
        assignment = {
            label: int(index) for label, index in zip(output_labels, output_index, strict=True)
        }
        output_items.append(
            _trace_einsum_scalar_at(
                trace_operands,
                input_labels,
                dimensions,
                assignment,
                contraction_labels,
                context,
            )
        )
    if output_shape == ():
        return output_items[0]
    return TraceADArray(tuple(output_items), output_shape, context)


def _trace_matmul(
    left: object,
    right: object,
    context: _WholeProgramTraceContext,
) -> TraceADScalar | TraceADArray:
    lhs = _coerce_trace_array(left, context)
    rhs = _coerce_trace_array(right, context)
    _require_program_ad_product_contract("matmul", (lhs, rhs))
    if lhs.ndim == 2 and rhs.ndim == 1:
        rows, cols = lhs.shape
        if rhs.shape != (cols,):
            raise ValueError("whole-program AD matrix-vector dimensions must align")
        items = []
        for row in range(rows):
            total = lhs._items[row * cols] * rhs._items[0]
            for col in range(1, cols):
                total = total + lhs._items[row * cols + col] * rhs._items[col]
            items.append(total)
        return TraceADArray(tuple(items), (rows,), context)
    if lhs.ndim == 1 and rhs.ndim == 2:
        rows, cols = rhs.shape
        if lhs.shape != (rows,):
            raise ValueError("whole-program AD vector-matrix dimensions must align")
        items = []
        for col in range(cols):
            total = lhs._items[0] * rhs._items[col]
            for row in range(1, rows):
                total = total + lhs._items[row] * rhs._items[row * cols + col]
            items.append(total)
        return TraceADArray(tuple(items), (cols,), context)
    if lhs.ndim == 2 and rhs.ndim == 2:
        lhs_rows, lhs_cols = lhs.shape
        rhs_rows, rhs_cols = rhs.shape
        if lhs_cols != rhs_rows:
            raise ValueError("whole-program AD matrix-matrix dimensions must align")
        items = []
        for row in range(lhs_rows):
            for col in range(rhs_cols):
                total = lhs._items[row * lhs_cols] * rhs._items[col]
                for inner in range(1, lhs_cols):
                    total = (
                        total
                        + lhs._items[row * lhs_cols + inner] * rhs._items[inner * rhs_cols + col]
                    )
                items.append(total)
        return TraceADArray(tuple(items), (lhs_rows, rhs_cols), context)
    if lhs.ndim == 1 and rhs.ndim == 1:
        return _trace_dot(lhs, rhs, context)
    raise ValueError("whole-program AD matmul supports rank-1 and rank-2 operands")


def _trace_det(matrix: object, context: _WholeProgramTraceContext) -> TraceADScalar:
    array = _coerce_trace_array(matrix, context)
    if array.ndim != 2:
        raise ValueError("program AD np.linalg.det supports rank-2 matrices only")
    rows, cols = array.shape
    if rows != cols:
        raise ValueError("program AD np.linalg.det requires a square matrix")
    if rows == 0:
        return _coerce_trace_scalar(1.0, context)
    primal = np.array([item.primal for item in array._items], dtype=np.float64).reshape(rows, cols)
    determinant = float(np.linalg.det(primal))
    if not np.isfinite(determinant):
        raise ValueError("program AD np.linalg.det requires a finite determinant")
    tangent_tensor = np.stack([item.tangent for item in array._items], axis=0).reshape(
        rows, cols, context.parameter_count
    )
    cofactors = _program_ad_linalg_det_cofactor_matrix(primal)
    tangent = np.einsum("ij,ijp->p", cofactors, tangent_tensor)
    return context.make(
        f"linalg:det:{rows}x{cols}",
        tuple(item.name for item in array._items),
        determinant,
        np.asarray(tangent, dtype=np.float64),
    )


def _trace_det_items(
    items: tuple[TraceADScalar, ...],
    size: int,
    context: _WholeProgramTraceContext,
) -> TraceADScalar:
    if size == 0:
        return _coerce_trace_scalar(1.0, context)
    if size == 1:
        return items[0]
    if size == 2:
        return items[0] * items[3] - items[1] * items[2]
    total: TraceADScalar | None = None
    for col in range(size):
        minor_items = tuple(
            items[row * size + minor_col]
            for row in range(1, size)
            for minor_col in range(size)
            if minor_col != col
        )
        term = items[col] * _trace_det_items(minor_items, size - 1, context)
        if total is None:
            total = term
        elif col % 2 == 0:
            total = total + term
        else:
            total = total - term
    if total is None:
        return _coerce_trace_scalar(1.0, context)
    return total


def _trace_inv(matrix: object, context: _WholeProgramTraceContext) -> TraceADArray:
    array = _coerce_trace_array(matrix, context)
    if array.ndim != 2:
        raise ValueError("program AD np.linalg.inv supports rank-2 matrices only")
    rows, cols = array.shape
    if rows != cols:
        raise ValueError("program AD np.linalg.inv requires a square matrix")
    if rows == 0:
        return TraceADArray((), (0, 0), context)
    primal = np.array([item.primal for item in array._items], dtype=np.float64).reshape(rows, cols)
    try:
        inverse = np.linalg.inv(primal)
    except np.linalg.LinAlgError as exc:
        raise ValueError("program AD np.linalg.inv requires a nonsingular matrix") from exc
    if not np.all(np.isfinite(inverse)):
        raise ValueError("program AD np.linalg.inv requires a nonsingular matrix")
    tangent_tensor = np.stack([item.tangent for item in array._items], axis=0).reshape(
        rows, cols, context.parameter_count
    )
    input_names = tuple(item.name for item in array._items)
    inverse_items: list[TraceADScalar] = []
    for row in range(rows):
        for col in range(cols):
            tangent = np.array(
                [
                    -(inverse @ tangent_tensor[:, :, parameter_index] @ inverse)[row, col]
                    for parameter_index in range(context.parameter_count)
                ],
                dtype=np.float64,
            )
            inverse_items.append(
                context.make(
                    f"linalg:inv:{rows}x{cols}:{row}:{col}",
                    input_names,
                    float(inverse[row, col]),
                    tangent,
                )
            )
    return TraceADArray(tuple(inverse_items), (rows, cols), context)


def _trace_solve(
    matrix: object,
    rhs: object,
    context: _WholeProgramTraceContext,
) -> TraceADScalar | TraceADArray:
    lhs = _coerce_trace_array(matrix, context)
    right = _coerce_trace_array(rhs, context)
    if lhs.ndim != 2:
        raise ValueError("program AD np.linalg.solve matrix must be rank-2")
    rows, cols = lhs.shape
    if rows != cols:
        raise ValueError("program AD np.linalg.solve matrix must be square")
    if right.ndim == 1:
        if right.shape[0] != rows:
            raise ValueError("program AD np.linalg.solve vector length must match matrix")
    elif right.ndim == 2:
        if right.shape[0] != rows:
            raise ValueError("program AD np.linalg.solve right-hand matrix rows must match matrix")
    else:
        raise ValueError("program AD np.linalg.solve right-hand side must be rank-1 or rank-2")
    matrix_primal = np.array([item.primal for item in lhs._items], dtype=np.float64).reshape(
        rows, cols
    )
    rhs_primal = np.array([item.primal for item in right._items], dtype=np.float64).reshape(
        right.shape
    )
    try:
        solution = np.linalg.solve(matrix_primal, rhs_primal)
    except np.linalg.LinAlgError as exc:
        raise ValueError("program AD np.linalg.solve requires a nonsingular matrix") from exc
    if not np.all(np.isfinite(solution)):
        raise ValueError("program AD np.linalg.solve requires a finite solution")
    matrix_tangent = np.stack([item.tangent for item in lhs._items], axis=0).reshape(
        rows, cols, context.parameter_count
    )
    rhs_tangent = np.stack([item.tangent for item in right._items], axis=0).reshape(
        (*right.shape, context.parameter_count)
    )
    input_names = tuple(item.name for item in lhs._items) + tuple(
        item.name for item in right._items
    )
    solution_array = np.asarray(solution, dtype=np.float64)
    items: list[TraceADScalar] = []
    if right.ndim == 1:
        if context.parameter_count:
            tangent_solution = np.array(
                [
                    np.linalg.solve(
                        matrix_primal,
                        rhs_tangent[:, parameter_index]
                        - matrix_tangent[:, :, parameter_index] @ solution_array,
                    )
                    for parameter_index in range(context.parameter_count)
                ],
                dtype=np.float64,
            ).T
        else:
            tangent_solution = np.zeros((rows, 0), dtype=np.float64)
        for row in range(rows):
            items.append(
                context.make(
                    f"linalg:solve:{rows}x{cols}:rhs:{right.shape[0]}:{row}",
                    input_names,
                    float(solution_array[row]),
                    tangent_solution[row, :],
                )
            )
        return TraceADArray(tuple(items), right.shape, context)
    rhs_cols = right.shape[1]
    if context.parameter_count:
        tangent_solution_matrix = np.array(
            [
                np.linalg.solve(
                    matrix_primal,
                    rhs_tangent[:, :, parameter_index]
                    - matrix_tangent[:, :, parameter_index] @ solution_array,
                )
                for parameter_index in range(context.parameter_count)
            ],
            dtype=np.float64,
        ).transpose(1, 2, 0)
    else:
        tangent_solution_matrix = np.zeros((rows, rhs_cols, 0), dtype=np.float64)
    for row in range(rows):
        for col in range(rhs_cols):
            items.append(
                context.make(
                    f"linalg:solve:{rows}x{cols}:rhs:{right.shape[0]}x{rhs_cols}:{row}:{col}",
                    input_names,
                    float(solution_array[row, col]),
                    tangent_solution_matrix[row, col, :],
                )
            )
    return TraceADArray(tuple(items), solution_array.shape, context)


def _trace_identity_matrix(size: int, context: _WholeProgramTraceContext) -> TraceADArray:
    zero = _coerce_trace_scalar(0.0, context)
    one = _coerce_trace_scalar(1.0, context)
    return TraceADArray(
        tuple(one if row == col else zero for row in range(size) for col in range(size)),
        (size, size),
        context,
    )


def _trace_matrix_power(
    matrix: object,
    power: object,
    context: _WholeProgramTraceContext,
) -> TraceADArray:
    array = _coerce_trace_array(matrix, context)
    if array.ndim != 2:
        raise ValueError("program AD np.linalg.matrix_power supports rank-2 matrices only")
    rows, cols = array.shape
    if rows != cols:
        raise ValueError("program AD np.linalg.matrix_power requires a square matrix")
    if isinstance(power, bool) or not isinstance(power, (int, np.integer)):
        raise ValueError("program AD np.linalg.matrix_power exponent must be a static integer")
    exponent = int(power)
    rule = program_ad_linalg_matrix_power_derivative_rule(exponent)
    if rule.jvp_rule is None:
        raise ValueError("program AD np.linalg.matrix_power requires a JVP rule")
    flat_values = np.array([item.primal for item in array._items], dtype=np.float64)
    try:
        output_flat = np.asarray(rule.value_fn(flat_values), dtype=np.float64).reshape(-1)
        flat_tangent = np.stack([item.tangent for item in array._items], axis=0)
        if context.parameter_count:
            tangent_outputs = np.array(
                [
                    rule.jvp_rule(flat_values, flat_tangent[:, parameter_index])
                    for parameter_index in range(context.parameter_count)
                ],
                dtype=np.float64,
            ).T
        else:
            tangent_outputs = np.zeros((rows * cols, 0), dtype=np.float64)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "program AD np.linalg.matrix_power requires a nonsingular matrix"
        ) from exc
    if not np.all(np.isfinite(output_flat)):
        raise ValueError("program AD np.linalg.matrix_power requires finite outputs")
    input_names = tuple(item.name for item in array._items)
    items: list[TraceADScalar] = []
    for row in range(rows):
        for col in range(cols):
            flat_index = row * cols + col
            items.append(
                context.make(
                    f"linalg:matrix_power:{_trace_shape_label(array.shape)}:"
                    f"power:{exponent}:{row}:{col}",
                    input_names,
                    float(output_flat[flat_index]),
                    tangent_outputs[flat_index, :],
                )
            )
    return TraceADArray(tuple(items), array.shape, context)


def _trace_multi_dot(
    operands: object,
    context: _WholeProgramTraceContext,
) -> TraceADScalar | TraceADArray:
    if isinstance(operands, (TraceADArray, np.ndarray)):
        raise ValueError("program AD np.linalg.multi_dot requires a static operand sequence")
    if not isinstance(operands, Sequence):
        raise ValueError("program AD np.linalg.multi_dot requires a static operand sequence")
    arrays = tuple(_coerce_trace_array(operand, context) for operand in operands)
    if len(arrays) < 2:
        raise ValueError("program AD np.linalg.multi_dot requires at least two operands")
    for index, array in enumerate(arrays):
        if array.ndim not in {1, 2}:
            raise ValueError("program AD np.linalg.multi_dot supports rank-1 and rank-2 operands")
        if 0 < index < len(arrays) - 1 and array.ndim != 2:
            raise ValueError("program AD np.linalg.multi_dot middle operands must be rank-2")
    operand_shapes = tuple(array.shape for array in arrays)
    rule = program_ad_linalg_multi_dot_derivative_rule(operand_shapes)
    if rule.jvp_rule is None:
        raise ValueError("program AD np.linalg.multi_dot requires a JVP rule")
    primal_operands = tuple(
        np.array([item.primal for item in array._items], dtype=np.float64).reshape(array.shape)
        for array in arrays
    )
    try:
        output = np.asarray(np.linalg.multi_dot(primal_operands), dtype=np.float64)
    except ValueError as exc:
        raise ValueError("program AD np.linalg.multi_dot dimensions must align") from exc
    output_shape = tuple(int(dimension) for dimension in output.shape)
    output_flat = output.reshape(-1)
    flat_values = np.concatenate(
        [operand.reshape(-1) for operand in primal_operands], dtype=np.float64
    )
    flat_tangent = np.concatenate(
        [np.stack([item.tangent for item in array._items], axis=0) for array in arrays],
        axis=0,
        dtype=np.float64,
    )
    if context.parameter_count:
        tangent_outputs = np.array(
            [
                rule.jvp_rule(flat_values, flat_tangent[:, parameter_index])
                for parameter_index in range(context.parameter_count)
            ],
            dtype=np.float64,
        ).T
    else:
        tangent_outputs = np.zeros((output_flat.size, 0), dtype=np.float64)
    if not np.all(np.isfinite(output_flat)):
        raise ValueError("program AD np.linalg.multi_dot requires finite outputs")
    input_names = tuple(item.name for array in arrays for item in array._items)
    shape_signature = "__".join(_trace_shape_label(shape) for shape in operand_shapes)
    if output_shape == ():
        return context.make(
            f"linalg:multi_dot:{shape_signature}:out:scalar",
            input_names,
            float(output_flat[0]),
            tangent_outputs[0, :],
        )
    output_label = _trace_shape_label(output_shape)
    items = tuple(
        context.make(
            f"linalg:multi_dot:{shape_signature}:out:{output_label}:{flat_index}",
            input_names,
            float(output_flat[flat_index]),
            tangent_outputs[flat_index, :],
        )
        for flat_index in range(output_flat.size)
    )
    return TraceADArray(items, output_shape, context)


def _trace_eigvalsh(
    matrix: object, context: _WholeProgramTraceContext, *, uplo: str = "L"
) -> TraceADArray:
    array = _coerce_trace_array(matrix, context)
    if array.ndim != 2:
        raise ValueError("program AD np.linalg.eigvalsh requires a rank-2 matrix")
    rows, cols = array.shape
    if rows != cols:
        raise ValueError("program AD np.linalg.eigvalsh requires a square matrix")
    uplo_value = _program_ad_linalg_uplo(uplo, "np.linalg.eigvalsh")
    primal = np.array([item.primal for item in array._items], dtype=np.float64).reshape(rows, cols)
    if not np.allclose(primal, primal.T, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("program AD np.linalg.eigvalsh requires a symmetric matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(primal, UPLO=uplo_value)
    _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, "eigvalsh")
    tangent_tensor = np.stack([item.tangent for item in array._items], axis=0).reshape(
        rows, cols, context.parameter_count
    )
    items: list[TraceADScalar] = []
    input_names = tuple(item.name for item in array._items)
    for index, eigenvalue in enumerate(eigenvalues):
        eigenvector = eigenvectors[:, index]
        tangent = np.einsum("i,j,ijp->p", eigenvector, eigenvector, tangent_tensor)
        items.append(
            context.make(
                f"linalg:eigvalsh:{index}",
                input_names,
                float(eigenvalue),
                np.asarray(tangent, dtype=np.float64),
            )
        )
    return TraceADArray(tuple(items), (rows,), context)


def _trace_eigvals(matrix: object, context: _WholeProgramTraceContext) -> TraceADArray:
    array = _coerce_trace_array(matrix, context)
    if array.ndim != 2:
        raise ValueError("program AD np.linalg.eigvals requires a rank-2 matrix")
    rows, cols = array.shape
    if rows != cols:
        raise ValueError("program AD np.linalg.eigvals requires a square matrix")
    primal = np.array([item.primal for item in array._items], dtype=np.float64).reshape(rows, cols)
    eigenvalues, right_eigenvectors, left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition_from_matrix("eigvals", primal)
    )
    tangent_tensor = np.stack([item.tangent for item in array._items], axis=0).reshape(
        rows, cols, context.parameter_count
    )
    items: list[TraceADScalar] = []
    input_names = tuple(item.name for item in array._items)
    for index, eigenvalue in enumerate(eigenvalues):
        tangent = np.einsum(
            "i,j,ijp->p",
            left_eigenvector_rows[index, :],
            right_eigenvectors[:, index],
            tangent_tensor,
        )
        items.append(
            context.make(
                f"linalg:eigvals:{rows}x{cols}:{index}",
                input_names,
                float(eigenvalue),
                np.asarray(tangent, dtype=np.float64),
            )
        )
    return TraceADArray(tuple(items), (rows,), context)


def _trace_eig(
    matrix: object, context: _WholeProgramTraceContext
) -> tuple[TraceADArray, TraceADArray]:
    array = _coerce_trace_array(matrix, context)
    if array.ndim != 2:
        raise ValueError("program AD np.linalg.eig requires a rank-2 matrix")
    rows, cols = array.shape
    if rows != cols:
        raise ValueError("program AD np.linalg.eig requires a square matrix")
    primal = np.array([item.primal for item in array._items], dtype=np.float64).reshape(rows, cols)
    eigenvalues, right_eigenvectors, left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition_from_matrix("eig", primal)
    )
    tangent_tensor = np.stack([item.tangent for item in array._items], axis=0).reshape(
        rows, cols, context.parameter_count
    )
    input_names = tuple(item.name for item in array._items)
    eigenvalue_items: list[TraceADScalar] = []
    eigenvector_tangents = tuple(
        _program_ad_linalg_eig_eigenvector_jvp_matrix(
            eigenvalues, right_eigenvectors, left_eigenvector_rows, tangent_tensor[:, :, index]
        )
        for index in range(context.parameter_count)
    )
    for index, eigenvalue in enumerate(eigenvalues):
        tangent = np.einsum(
            "i,j,ijp->p",
            left_eigenvector_rows[index, :],
            right_eigenvectors[:, index],
            tangent_tensor,
        )
        eigenvalue_items.append(
            context.make(
                f"linalg:eig:eigenvalue:{rows}x{cols}:{index}",
                input_names,
                float(eigenvalue),
                np.asarray(tangent, dtype=np.float64),
            )
        )
    eigenvector_items: list[TraceADScalar] = []
    for row in range(rows):
        for col in range(cols):
            tangent = np.array(
                [eigenvector_tangent[row, col] for eigenvector_tangent in eigenvector_tangents],
                dtype=np.float64,
            )
            eigenvector_items.append(
                context.make(
                    f"linalg:eig:eigenvector:{rows}x{cols}:{col}:{row}",
                    input_names,
                    float(right_eigenvectors[row, col]),
                    tangent,
                )
            )
    return (
        TraceADArray(tuple(eigenvalue_items), (rows,), context),
        TraceADArray(tuple(eigenvector_items), (rows, cols), context),
    )


def _trace_eigh(
    matrix: object, context: _WholeProgramTraceContext, *, uplo: str = "L"
) -> tuple[TraceADArray, TraceADArray]:
    array = _coerce_trace_array(matrix, context)
    if array.ndim != 2:
        raise ValueError("program AD np.linalg.eigh requires a rank-2 matrix")
    rows, cols = array.shape
    if rows != cols:
        raise ValueError("program AD np.linalg.eigh requires a square matrix")
    uplo_value = _program_ad_linalg_uplo(uplo, "np.linalg.eigh")
    primal = np.array([item.primal for item in array._items], dtype=np.float64).reshape(rows, cols)
    _program_ad_linalg_require_symmetric("np.linalg.eigh", primal)
    eigenvalues, eigenvectors = np.linalg.eigh(primal, UPLO=uplo_value)
    _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, "eigh")
    tangent_tensor = np.stack([item.tangent for item in array._items], axis=0).reshape(
        rows, cols, context.parameter_count
    )
    for index in range(context.parameter_count):
        _program_ad_linalg_require_symmetric("eigh tangent", tangent_tensor[:, :, index])
    input_names = tuple(item.name for item in array._items)
    eigenvalue_items: list[TraceADScalar] = []
    eigenvector_tangents = tuple(
        _program_ad_linalg_eigh_eigenvector_jvp_matrix(
            eigenvalues, eigenvectors, tangent_tensor[:, :, index]
        )
        for index in range(context.parameter_count)
    )
    for index, eigenvalue in enumerate(eigenvalues):
        eigenvector = eigenvectors[:, index]
        tangent = np.einsum("i,j,ijp->p", eigenvector, eigenvector, tangent_tensor)
        eigenvalue_items.append(
            context.make(
                f"linalg:eigh:eigenvalue:{rows}x{cols}:{uplo_value}:{index}",
                input_names,
                float(eigenvalue),
                np.asarray(tangent, dtype=np.float64),
            )
        )
    eigenvector_items: list[TraceADScalar] = []
    for row in range(rows):
        for col in range(cols):
            tangent = np.array(
                [eigenvector_tangent[row, col] for eigenvector_tangent in eigenvector_tangents],
                dtype=np.float64,
            )
            eigenvector_items.append(
                context.make(
                    f"linalg:eigh:eigenvector:{rows}x{cols}:{uplo_value}:{col}:{row}",
                    input_names,
                    float(eigenvectors[row, col]),
                    tangent,
                )
            )
    return (
        TraceADArray(tuple(eigenvalue_items), (rows,), context),
        TraceADArray(tuple(eigenvector_items), (rows, cols), context),
    )


def _trace_svdvals(matrix: object, context: _WholeProgramTraceContext) -> TraceADArray:
    array = _coerce_trace_array(matrix, context)
    if array.ndim != 2:
        raise ValueError("program AD np.linalg.svd requires a rank-2 matrix")
    rows, cols = array.shape
    if rows <= 0 or cols <= 0:
        raise ValueError("program AD np.linalg.svd requires non-empty matrix dimensions")
    primal = np.array([item.primal for item in array._items], dtype=np.float64).reshape(rows, cols)
    left, singular_values, right_h = np.linalg.svd(primal, full_matrices=False)
    _program_ad_linalg_require_distinct_positive_singular_values(singular_values, "svd")
    tangent_tensor = np.stack([item.tangent for item in array._items], axis=0).reshape(
        rows, cols, context.parameter_count
    )
    items: list[TraceADScalar] = []
    input_names = tuple(item.name for item in array._items)
    for index, singular_value in enumerate(singular_values):
        tangent = np.einsum(
            "i,j,ijp->p",
            left[:, index],
            right_h[index, :],
            tangent_tensor,
        )
        items.append(
            context.make(
                f"linalg:svdvals:{rows}x{cols}:{index}",
                input_names,
                float(singular_value),
                np.asarray(tangent, dtype=np.float64),
            )
        )
    return TraceADArray(tuple(items), (singular_values.size,), context)


def _trace_pinv(
    matrix: object,
    context: _WholeProgramTraceContext,
    *,
    rcond: float = 1.0e-15,
) -> TraceADArray:
    array = _coerce_trace_array(matrix, context)
    if array.ndim != 2:
        raise ValueError("program AD np.linalg.pinv requires a rank-2 matrix")
    rows, cols = array.shape
    if rows <= 0 or cols <= 0:
        raise ValueError("program AD np.linalg.pinv requires non-empty matrix dimensions")
    primal = np.array([item.primal for item in array._items], dtype=np.float64).reshape(rows, cols)
    pinv = _program_ad_linalg_pinv_value_matrix(primal, rcond=rcond)
    tangent_tensor = np.stack([item.tangent for item in array._items], axis=0).reshape(
        rows, cols, context.parameter_count
    )
    items: list[TraceADScalar] = []
    input_names = tuple(item.name for item in array._items)
    for row in range(cols):
        for col in range(rows):
            tangent = np.array(
                [
                    _program_ad_linalg_pinv_jvp_matrix(primal, pinv, tangent_tensor[:, :, index])[
                        row, col
                    ]
                    for index in range(context.parameter_count)
                ],
                dtype=np.float64,
            )
            items.append(
                context.make(
                    f"linalg:pinv:{rows}x{cols}:{rcond:.17g}:{row}:{col}",
                    input_names,
                    float(pinv[row, col]),
                    tangent,
                )
            )
    return TraceADArray(tuple(items), (cols, rows), context)


def _trace_trace(
    values: object,
    context: _WholeProgramTraceContext,
    *,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
) -> TraceADScalar:
    array = _coerce_trace_array(values, context)
    if array.ndim != 2:
        raise ValueError("whole-program AD np.trace supports matrices only")
    if (axis1, axis2) != (0, 1):
        raise ValueError("whole-program AD np.trace supports axis1=0 and axis2=1")
    _require_program_ad_linalg_contract("trace", (array, offset, axis1, axis2))
    offset_value = int(offset)
    rows, cols = array.shape
    selected_items = tuple(
        array._items[row * cols + row + offset_value]
        for row in range(rows)
        if 0 <= row + offset_value < cols
    )
    if not selected_items:
        raise ValueError("whole-program AD np.trace offset selects an empty diagonal")
    tangent = sum(
        (item.tangent for item in selected_items),
        np.zeros(context.parameter_count, dtype=np.float64),
    )
    return context.make(
        f"linalg:trace:{_trace_shape_label(array.shape)}:offset:{offset_value}",
        tuple(item.name for item in selected_items),
        float(sum(item.primal for item in selected_items)),
        tangent,
    )


def _trace_diag(
    values: object,
    context: _WholeProgramTraceContext,
    *,
    k: int = 0,
) -> TraceADArray:
    array = _coerce_trace_array(values, context)
    _require_program_ad_linalg_contract("diag", (array, k))
    offset = int(k)
    source_shape = _trace_shape_label(array.shape)
    if array.ndim == 1:
        size = array.shape[0] + abs(offset)
        zero = _trace_constant(0.0, context)
        items: list[TraceADScalar] = []
        for row in range(size):
            for col in range(size):
                source_index = row if offset >= 0 else col
                on_diag = (col - row) == offset
                if on_diag:
                    source = array._items[source_index]
                    items.append(
                        context.make(
                            f"linalg:diag:{source_shape}:offset:{offset}:construct:{source_index}",
                            (source.name,),
                            source.primal,
                            source.tangent,
                        )
                    )
                else:
                    items.append(zero)
        return TraceADArray(tuple(items), (size, size), context)
    if array.ndim == 2:
        rows, cols = array.shape
        items = []
        for row in range(rows):
            col = row + offset
            if 0 <= col < cols:
                source = array._items[row * cols + col]
                items.append(
                    context.make(
                        f"linalg:diag:{source_shape}:offset:{offset}:extract:{len(items)}",
                        (source.name,),
                        source.primal,
                        source.tangent,
                    )
                )
        if not items:
            raise ValueError("whole-program AD np.diag offset selects an empty diagonal")
        return TraceADArray(tuple(items), (len(items),), context)
    raise ValueError("whole-program AD np.diag supports vectors and matrices only")


def _trace_diagflat(
    values: object,
    context: _WholeProgramTraceContext,
    *,
    k: int = 0,
) -> TraceADArray:
    array = _coerce_trace_array(values, context)
    _require_program_ad_linalg_contract("diagflat", (array, k))
    offset = int(k)
    flattened = array.ravel()
    size = flattened.shape[0] + abs(offset)
    zero = _trace_constant(0.0, context)
    source_shape = _trace_shape_label(array.shape)
    items: list[TraceADScalar] = []
    for row in range(size):
        for col in range(size):
            source_index = row if offset >= 0 else col
            on_diag = (col - row) == offset
            if on_diag:
                source = flattened._items[source_index]
                items.append(
                    context.make(
                        f"linalg:diagflat:{source_shape}:offset:{offset}:construct:{source_index}",
                        (source.name,),
                        source.primal,
                        source.tangent,
                    )
                )
            else:
                items.append(zero)
    return TraceADArray(tuple(items), (size, size), context)


def _trace_shape_label(shape: tuple[int, ...]) -> str:
    """Return a compact static shape label for primitive IR metadata."""

    return "x".join(str(int(dimension)) for dimension in shape)


def _trace_multiply_arrays(
    left: TraceADArray,
    right: TraceADArray,
    context: _WholeProgramTraceContext,
) -> TraceADScalar | TraceADArray:
    shape = _broadcast_shape(left.shape, right.shape)
    left = _broadcast_trace_array(left, shape, context)
    right = _broadcast_trace_array(right, shape, context)
    items = tuple(lhs * rhs for lhs, rhs in zip(left._items, right._items, strict=True))
    return items[0] if shape == () else TraceADArray(items, shape, context)


def _trace_constant(value: float, context: _WholeProgramTraceContext) -> TraceADScalar:
    tangent = np.zeros(context.parameter_count, dtype=np.float64)
    return TraceADScalar(value, tangent, context, repr(float(value)))


def _coerce_trace_predicate_array(
    condition: object,
    shape: tuple[int, ...],
    context: _WholeProgramTraceContext,
) -> TraceADPredicateArray:
    if isinstance(condition, _TracePredicate):
        if condition.context is not context:
            raise ValueError("whole-program AD predicate belongs to a different trace")
        return TraceADPredicateArray(
            tuple(condition for _ in range(int(np.prod(shape)))), shape, context
        )
    if isinstance(condition, TraceADPredicateArray):
        if condition.context is not context:
            raise ValueError("whole-program AD predicate array belongs to a different trace")
        if condition.shape == shape:
            return condition
        if condition.shape == ():
            return TraceADPredicateArray(
                tuple(condition.predicates[0] for _ in range(int(np.prod(shape)))),
                shape,
                context,
            )
        raise ValueError("whole-program AD np.where predicate shape must match operands")
    if isinstance(condition, (bool, np.bool_)):
        predicate = _TracePredicate(bool(condition), context, f"constant:{bool(condition)}")
        return TraceADPredicateArray(
            tuple(predicate for _ in range(int(np.prod(shape)))), shape, context
        )
    raw = np.asarray(condition)
    if raw.dtype.kind != "b":
        raise ValueError("whole-program AD np.where condition must be boolean or AD predicate")
    if tuple(raw.shape) not in {shape, ()}:
        raise ValueError("whole-program AD np.where condition shape must match operands")
    flat = np.broadcast_to(raw, shape).reshape(-1)
    predicates = tuple(
        _TracePredicate(bool(item), context, f"constant:{bool(item)}") for item in flat
    )
    return TraceADPredicateArray(predicates, shape, context)


def _trace_choose(
    selector: object,
    choices: object,
    context: _WholeProgramTraceContext,
    *,
    mode: str,
) -> TraceADScalar | TraceADArray:
    choice_arrays = _trace_choose_choice_arrays(choices, context)
    _require_program_ad_selection_contract("choose", (selector, choice_arrays, mode))
    selector_indices = _trace_choose_selector_indices(
        selector,
        choice_count=len(choice_arrays),
        mode=mode,
    )
    shape = _broadcast_shape(
        tuple(int(dimension) for dimension in selector_indices.shape),
        *(choice.shape for choice in choice_arrays),
    )
    broadcast_selector = np.broadcast_to(selector_indices, shape).reshape(-1)
    broadcast_choices = tuple(
        _broadcast_trace_array(choice, shape, context) for choice in choice_arrays
    )
    items: list[TraceADScalar] = []
    for flat_index, choice_index in enumerate(broadcast_selector):
        chosen = broadcast_choices[int(choice_index)]._items[flat_index]
        items.append(
            context.make(
                "choose",
                (f"static_selector:{int(choice_index)}", chosen.name),
                chosen.primal,
                chosen.tangent,
            )
        )
    result = tuple(items)
    return result[0] if shape == () else TraceADArray(result, shape, context)


def _trace_choose_choice_arrays(
    choices: object,
    context: _WholeProgramTraceContext,
) -> tuple[TraceADArray, ...]:
    if isinstance(choices, TraceADArray):
        raise ValueError("program AD np.choose requires a static choice sequence")
    if isinstance(choices, (np.ndarray, Sequence)):
        choice_sequence = tuple(choices)
    else:
        raise ValueError("program AD np.choose requires a static choice sequence")
    if not choice_sequence:
        raise ValueError("program AD np.choose requires at least one choice")
    return tuple(_coerce_trace_array(choice, context) for choice in choice_sequence)


def _trace_choose_selector_indices(
    selector: object,
    *,
    choice_count: int,
    mode: str,
) -> NDArray[np.int64]:
    if isinstance(selector, (TraceADScalar, TraceADArray, _TracePredicate, TraceADPredicateArray)):
        raise ValueError("program AD np.choose requires a static integer selector")
    raw = np.asarray(selector)
    if raw.dtype == object and any(
        isinstance(
            item,
            (TraceADScalar, TraceADArray, _TracePredicate, TraceADPredicateArray),
        )
        for item in raw.reshape(-1)
    ):
        raise ValueError("program AD np.choose requires a static integer selector")
    if raw.dtype.kind not in {"i", "u", "b"}:
        raise ValueError("program AD np.choose requires a static integer selector")
    indices = raw.astype(np.int64, copy=False)
    if mode == "raise":
        if bool(np.any(indices < 0)) or bool(np.any(indices >= choice_count)):
            raise ValueError("program AD np.choose selector indices out of bounds")
        return indices
    if mode == "wrap":
        return cast(NDArray[np.int64], np.mod(indices, choice_count).astype(np.int64))
    if mode == "clip":
        return cast(
            NDArray[np.int64],
            np.clip(indices, 0, choice_count - 1).astype(np.int64),
        )
    raise ValueError("program AD np.choose mode must be raise, wrap, or clip")


def _trace_compress(
    condition: object,
    array: TraceADArray,
    *,
    axis: object,
) -> TraceADScalar | TraceADArray:
    _require_program_ad_selection_contract("compress", (condition, array, axis))
    indices = _trace_compress_condition_indices(condition)
    if axis is None:
        return _trace_take(array.ravel(), indices, axis=0, mode="raise")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD np.compress requires a static integer axis or None")
    normalised_axis = _normalise_axis("axis", int(axis), array.ndim)
    return _trace_take(array, indices, axis=normalised_axis, mode="raise")


def _trace_compress_condition_indices(condition: object) -> NDArray[np.int64]:
    if isinstance(
        condition, (TraceADScalar, TraceADArray, _TracePredicate, TraceADPredicateArray)
    ):
        raise ValueError("program AD np.compress requires a static boolean condition")
    raw = np.asarray(condition)
    if raw.dtype == object and any(
        isinstance(
            item,
            (TraceADScalar, TraceADArray, _TracePredicate, TraceADPredicateArray),
        )
        for item in raw.reshape(-1)
    ):
        raise ValueError("program AD np.compress requires a static boolean condition")
    if raw.ndim != 1:
        raise ValueError("program AD np.compress requires a one-dimensional condition")
    if raw.dtype.kind != "b":
        raise ValueError("program AD np.compress requires a static boolean condition")
    return cast(NDArray[np.int64], np.flatnonzero(raw).astype(np.int64))


def _trace_extract(
    condition: object,
    array: TraceADArray,
) -> TraceADScalar | TraceADArray:
    _require_program_ad_selection_contract("extract", (condition, array))
    indices = _trace_extract_condition_indices(condition, array.size)
    return _trace_take(array.ravel(), indices, axis=0, mode="raise")


def _trace_extract_condition_indices(condition: object, array_size: int) -> NDArray[np.int64]:
    if isinstance(
        condition, (TraceADScalar, TraceADArray, _TracePredicate, TraceADPredicateArray)
    ):
        raise ValueError("program AD np.extract requires a static boolean condition")
    raw = np.asarray(condition)
    if raw.dtype == object and any(
        isinstance(
            item,
            (TraceADScalar, TraceADArray, _TracePredicate, TraceADPredicateArray),
        )
        for item in raw.reshape(-1)
    ):
        raise ValueError("program AD np.extract requires a static boolean condition")
    if raw.dtype.kind != "b":
        raise ValueError("program AD np.extract requires a static boolean condition")
    if raw.size != array_size:
        raise ValueError("program AD np.extract condition size must match array size")
    return cast(NDArray[np.int64], np.flatnonzero(raw.reshape(-1)).astype(np.int64))


def _trace_select(
    condlist: object,
    choicelist: object,
    default: object,
    context: _WholeProgramTraceContext,
) -> TraceADScalar | TraceADArray:
    if isinstance(condlist, (TraceADArray, np.ndarray)) or not isinstance(condlist, Sequence):
        raise ValueError("program AD np.select requires a static condition sequence")
    if isinstance(choicelist, (TraceADArray, np.ndarray)) or not isinstance(choicelist, Sequence):
        raise ValueError("program AD np.select requires a static choice sequence")
    conditions = tuple(condlist)
    choices = tuple(choicelist)
    if len(conditions) != len(choices):
        raise ValueError("program AD np.select requires matching condition and choice counts")
    _require_program_ad_selection_contract("select", (conditions, choices, default))
    if not conditions:
        default_array = _coerce_trace_array(default, context)
        return default_array._items[0] if default_array.shape == () else default_array
    result: object = default
    for condition, choice in reversed(tuple(zip(conditions, choices, strict=True))):
        result = _trace_where(condition, choice, result, context)
    return cast(TraceADScalar | TraceADArray, result)


def _trace_piecewise(
    values: object,
    condlist: object,
    funclist: object,
    context: _WholeProgramTraceContext,
) -> TraceADScalar | TraceADArray:
    if isinstance(condlist, (TraceADArray, np.ndarray)) or not isinstance(condlist, Sequence):
        raise ValueError("program AD np.piecewise requires a static condition sequence")
    if isinstance(funclist, (TraceADArray, np.ndarray)) or not isinstance(funclist, Sequence):
        raise ValueError("program AD np.piecewise requires a static function sequence")
    conditions = tuple(condlist)
    functions = tuple(funclist)
    if len(functions) not in {len(conditions), len(conditions) + 1}:
        raise ValueError(
            "program AD np.piecewise requires one function per condition and optional default"
        )
    array = _coerce_trace_array(values, context)
    _require_program_ad_selection_contract("piecewise", (array, conditions, functions))
    if len(functions) == len(conditions) + 1:
        default_function = functions[-1]
        result: object = (
            default_function(array) if callable(default_function) else default_function
        )
        branch_functions = functions[:-1]
    else:
        result = 0.0
        branch_functions = functions
    for condition, function in zip(conditions, branch_functions, strict=True):
        choice = function(array) if callable(function) else function
        result = _trace_where(condition, choice, result, context)
    return cast(TraceADScalar | TraceADArray, result)


def _trace_where(
    condition: object,
    x_value: object,
    y_value: object,
    context: _WholeProgramTraceContext,
) -> TraceADScalar | TraceADArray:
    x_array = _coerce_trace_array(x_value, context)
    y_array = _coerce_trace_array(y_value, context)
    shape = _broadcast_shape(x_array.shape, y_array.shape)
    x_array = _broadcast_trace_array(x_array, shape, context)
    y_array = _broadcast_trace_array(y_array, shape, context)
    _require_program_ad_selection_contract("where", (condition, x_array, y_array))
    predicates = _coerce_trace_predicate_array(condition, shape, context)
    items = []
    for predicate, x_item, y_item in zip(
        predicates.predicates, x_array._items, y_array._items, strict=True
    ):
        chosen = x_item if bool(predicate) else y_item
        items.append(
            context.make(
                "where",
                (_trace_predicate_ir_label(predicate), x_item.name, y_item.name),
                chosen.primal,
                chosen.tangent,
            )
        )
    result = tuple(items)
    return result[0] if shape == () else TraceADArray(result, shape, context)


def _trace_predicate_ir_label(predicate: _TracePredicate) -> str:
    return f"{predicate.label}:truth:{int(bool(predicate))}"


def _trace_stack_convenience(
    name: str,
    arrays: Sequence[object],
    context: _WholeProgramTraceContext,
) -> TraceADArray:
    if not arrays:
        raise ValueError(f"program AD np.{name} requires at least one array")
    trace_arrays = tuple(_coerce_trace_array(array, context) for array in arrays)
    _require_program_ad_assembly_contract(name, (trace_arrays,))
    try:
        if name == "hstack":
            operands = tuple(_trace_atleast_nd(array, rank=1) for array in trace_arrays)
            axis = 0 if operands[0].ndim == 1 else 1
            return _trace_concatenate(operands, context, axis=axis)
        if name == "vstack":
            operands = tuple(_trace_atleast_nd(array, rank=2) for array in trace_arrays)
            return _trace_concatenate(operands, context, axis=0)
        if name == "column_stack":
            operands = tuple(
                array.reshape((array.size, 1)) if array.ndim < 2 else array
                for array in trace_arrays
            )
            return _trace_concatenate(operands, context, axis=1)
        if name == "dstack":
            operands = tuple(_trace_atleast_nd(array, rank=3) for array in trace_arrays)
            return _trace_concatenate(operands, context, axis=2)
    except ValueError as exc:
        raise ValueError(f"program AD np.{name} requires shape-compatible arrays") from exc
    raise ValueError(f"unsupported program AD stack convenience {name}")


def _trace_block(
    blocks: object,
    context: _WholeProgramTraceContext,
) -> TraceADArray:
    flat_items: list[TraceADScalar] = []

    def index_layout(node: object) -> object:
        if isinstance(node, (list, tuple)):
            if not node:
                raise ValueError("program AD np.block requires non-empty nested sequences")
            return [index_layout(item) for item in node]
        array = _coerce_trace_array(node, context)
        offset = len(flat_items)
        flat_items.extend(array._items)
        return np.arange(offset, offset + array.size, dtype=np.int64).reshape(array.shape)

    try:
        selected = np.block(cast(Any, index_layout(blocks)))
    except (TypeError, ValueError, np.exceptions.AxisError) as exc:
        raise ValueError(
            "program AD np.block requires a non-empty nested sequence of shape-compatible arrays"
        ) from exc
    _require_program_ad_assembly_contract("block", (blocks,))
    selected_array = np.asarray(selected, dtype=np.int64)
    items = tuple(flat_items[int(index)] for index in selected_array.reshape(-1))
    return TraceADArray(
        items, tuple(int(dimension) for dimension in selected_array.shape), context
    )


def _trace_split(
    name: str,
    array: TraceADArray,
    indices_or_sections: object,
    context: _WholeProgramTraceContext,
    *,
    axis: object,
) -> list[TraceADArray]:
    if array.ndim == 0:
        raise ValueError(
            f"program AD np.{name} requires static split sections compatible with array shape"
        )
    if name == "hsplit":
        axis_value = 0 if array.ndim == 1 else 1
    elif name == "vsplit":
        if array.ndim < 2:
            raise ValueError(
                "program AD np.vsplit requires static split sections compatible with array shape"
            )
        axis_value = 0
    elif name == "dsplit":
        if array.ndim < 3:
            raise ValueError(
                "program AD np.dsplit requires static split sections compatible with array shape"
            )
        axis_value = 2
    else:
        if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
            raise ValueError(
                f"program AD np.{name} requires static split sections compatible with array shape"
            )
        axis_value = _normalise_axis("axis", int(axis), array.ndim)

    index_array = np.arange(array.size, dtype=np.int64).reshape(array.shape)
    try:
        if name == "array_split":
            selected = np.array_split(index_array, cast(Any, indices_or_sections), axis=axis_value)
        else:
            selected = np.split(index_array, cast(Any, indices_or_sections), axis=axis_value)
    except (TypeError, ValueError, np.exceptions.AxisError) as exc:
        raise ValueError(
            f"program AD np.{name} requires static split sections compatible with array shape"
        ) from exc
    _require_program_ad_assembly_contract(name, (array, indices_or_sections, axis_value))
    result: list[TraceADArray] = []
    for part in selected:
        part_array = np.asarray(part, dtype=np.int64)
        items = tuple(array._items[int(index)] for index in part_array.reshape(-1))
        result.append(
            TraceADArray(
                items,
                tuple(int(dimension) for dimension in part_array.shape),
                context,
            )
        )
    return result


def _trace_triangular_mask(
    array: TraceADArray,
    *,
    k: object,
    lower: bool,
) -> TraceADArray:
    name = "tril" if lower else "triu"
    if array.ndim < 2:
        raise ValueError(f"program AD np.{name} requires rank >= 2")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"program AD np.{name} requires static integer k")
    _require_program_ad_assembly_contract(name, (array, int(k)))
    rows, cols = array.shape[-2:]
    row_index, col_index = np.ogrid[:rows, :cols]
    if lower:
        base_mask = row_index + int(k) >= col_index
    else:
        base_mask = row_index + int(k) <= col_index
    mask = np.broadcast_to(base_mask, array.shape).reshape(-1)
    zero = _trace_constant(0.0, array.context)
    items = tuple(
        item if bool(mask_value) else zero
        for item, mask_value in zip(array._items, mask, strict=True)
    )
    return TraceADArray(items, array.shape, array.context)


def _trace_diagonal(
    array: TraceADArray,
    *,
    offset: object,
    axis1: object,
    axis2: object,
) -> TraceADArray:
    if array.ndim < 2:
        raise ValueError("program AD np.diagonal requires rank >= 2")
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)):
        raise ValueError("program AD np.diagonal requires static integer offset")
    if isinstance(axis1, bool) or not isinstance(axis1, (int, np.integer)):
        raise ValueError("program AD np.diagonal requires static integer axes")
    if isinstance(axis2, bool) or not isinstance(axis2, (int, np.integer)):
        raise ValueError("program AD np.diagonal requires static integer axes")
    axis1_value = _normalise_axis("axis1", int(axis1), array.ndim)
    axis2_value = _normalise_axis("axis2", int(axis2), array.ndim)
    if axis1_value == axis2_value:
        raise ValueError("program AD np.diagonal requires distinct axes")
    _require_program_ad_assembly_contract(
        "diagonal", (array, int(offset), axis1_value, axis2_value)
    )
    index_array = np.arange(array.size, dtype=np.int64).reshape(array.shape)
    try:
        selected = np.diagonal(
            index_array,
            offset=int(offset),
            axis1=axis1_value,
            axis2=axis2_value,
        )
    except (TypeError, ValueError, np.exceptions.AxisError) as exc:
        raise ValueError(
            "program AD np.diagonal requires static offset and distinct axes "
            "compatible with array shape"
        ) from exc
    selected_array = np.asarray(selected, dtype=np.int64)
    items = tuple(array._items[int(index)] for index in selected_array.reshape(-1))
    return TraceADArray(
        items, tuple(int(dimension) for dimension in selected_array.shape), array.context
    )


def _trace_concatenate(
    arrays: Sequence[object],
    context: _WholeProgramTraceContext,
    *,
    axis: int | None = 0,
) -> TraceADArray:
    if not arrays:
        raise ValueError("whole-program AD np.concatenate requires at least one array")
    trace_arrays = tuple(_coerce_trace_array(array, context) for array in arrays)
    _require_program_ad_assembly_contract("concatenate", (trace_arrays, axis))
    flat_items = tuple(item for array in trace_arrays for item in array._items)
    index_arrays: list[NDArray[np.int64]] = []
    offset = 0
    for array in trace_arrays:
        next_offset = offset + array.size
        index_array = np.arange(offset, next_offset, dtype=np.int64).reshape(array.shape)
        index_arrays.append(index_array if axis is not None else index_array.reshape(-1))
        offset = next_offset
    try:
        if axis is None:
            selected = np.concatenate(index_arrays, axis=0)
        else:
            if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
                raise TypeError("axis must be an integer or None")
            selected = np.concatenate(index_arrays, axis=int(axis))
    except (TypeError, ValueError, np.exceptions.AxisError) as exc:
        raise ValueError(
            "whole-program AD np.concatenate requires shape-compatible arrays "
            "and a static integer axis or None"
        ) from exc
    selected_array = np.asarray(selected, dtype=np.int64)
    items = tuple(flat_items[int(index)] for index in selected_array.reshape(-1))
    return TraceADArray(
        items, tuple(int(dimension) for dimension in selected_array.shape), context
    )


def _trace_append(
    array: TraceADArray,
    values: object,
    context: _WholeProgramTraceContext,
    *,
    axis: object,
) -> TraceADArray:
    if axis is not None and (
        isinstance(axis, (bool, np.bool_, TraceADScalar, TraceADArray))
        or not isinstance(axis, (int, np.integer))
    ):
        raise ValueError("program AD np.append requires a static integer axis or None")
    trace_values = _coerce_trace_array(values, context)
    _require_program_ad_assembly_contract("append", (array, trace_values, axis))
    operands = (array.ravel(), trace_values.ravel()) if axis is None else (array, trace_values)
    try:
        return _trace_concatenate(
            operands,
            context,
            axis=None if axis is None else int(axis),
        )
    except ValueError as exc:
        raise ValueError(
            "program AD np.append requires axis-compatible arrays and a static integer axis or None"
        ) from exc


def _trace_stack(
    arrays: Sequence[object],
    context: _WholeProgramTraceContext,
    *,
    axis: int = 0,
) -> TraceADArray:
    if not arrays:
        raise ValueError("whole-program AD np.stack requires at least one array")
    trace_arrays = tuple(_coerce_trace_array(array, context) for array in arrays)
    _require_program_ad_assembly_contract("stack", (trace_arrays, axis))
    shape = trace_arrays[0].shape
    if any(array.shape != shape for array in trace_arrays):
        raise ValueError("whole-program AD np.stack operands must have matching shapes")
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("whole-program AD np.stack requires a static integer axis")
    flat_items = tuple(item for array in trace_arrays for item in array._items)
    index_arrays: list[NDArray[np.int64]] = []
    offset = 0
    for array in trace_arrays:
        next_offset = offset + array.size
        index_arrays.append(np.arange(offset, next_offset, dtype=np.int64).reshape(shape))
        offset = next_offset
    try:
        selected = np.stack(index_arrays, axis=int(axis))
    except (ValueError, np.exceptions.AxisError) as exc:
        raise ValueError(
            "whole-program AD np.stack requires a valid static axis for matching-shape arrays"
        ) from exc
    selected_array = np.asarray(selected, dtype=np.int64)
    items = tuple(flat_items[int(index)] for index in selected_array.reshape(-1))
    return TraceADArray(
        items, tuple(int(dimension) for dimension in selected_array.shape), context
    )


def _trace_clip(
    values: object,
    lower: object,
    upper: object,
    context: _WholeProgramTraceContext,
) -> TraceADScalar | TraceADArray:
    value_array = _coerce_trace_array(values, context)
    lower_array = _broadcast_trace_array(lower, value_array.shape, context)
    upper_array = _broadcast_trace_array(upper, value_array.shape, context)
    _require_program_ad_selection_contract("clip", (value_array, lower_array, upper_array))
    items = []
    for value, lower_item, upper_item in zip(
        value_array._items, lower_array._items, upper_array._items, strict=True
    ):
        if lower_item.primal > upper_item.primal:
            raise ValueError("whole-program AD np.clip lower bound must not exceed upper bound")
        if value.primal == lower_item.primal or value.primal == upper_item.primal:
            raise ValueError("whole-program AD np.clip is non-differentiable at clipping boundary")
        if value.primal < lower_item.primal:
            chosen = lower_item
        elif value.primal > upper_item.primal:
            chosen = upper_item
        else:
            chosen = value
        items.append(
            context.make(
                "clip",
                (value.name, lower_item.name, upper_item.name),
                chosen.primal,
                chosen.tangent,
            )
        )
    result = tuple(items)
    return (
        result[0] if value_array.shape == () else TraceADArray(result, value_array.shape, context)
    )


def _trace_norm(
    values: object,
    context: _WholeProgramTraceContext,
    *,
    ord_value: object = None,
    axis: object | None = None,
) -> TraceADScalar | TraceADArray:
    array = _coerce_trace_array(values, context)

    def norm_from_items(
        items: tuple[TraceADScalar, ...],
        *,
        zero_boundary_message: str,
    ) -> TraceADScalar:
        if not items:
            raise ValueError("whole-program AD np.linalg.norm requires at least one element")
        squared = items[0] * items[0]
        for item in items[1:]:
            squared = squared + item * item
        if squared.primal <= 0.0:
            raise ValueError(zero_boundary_message)
        norm = np.sqrt(squared)
        if not isinstance(norm, TraceADScalar):
            raise ValueError("whole-program AD norm must return a scalar")
        return norm

    if axis is None:
        if ord_value not in {None, 2, 2.0, "fro"}:
            raise ValueError("whole-program AD np.linalg.norm supports only Euclidean norm")
        if ord_value == "fro" and array.ndim < 2:
            raise ValueError("whole-program AD np.linalg.norm matrix norms require rank >= 2")
        return norm_from_items(
            tuple(array._items),
            zero_boundary_message=(
                "whole-program AD np.linalg.norm requires non-zero Frobenius norms"
                if ord_value == "fro"
                else "whole-program AD np.linalg.norm requires non-zero Euclidean norms"
            ),
        )
    if isinstance(axis, tuple):
        if len(axis) != 2:
            raise ValueError("whole-program AD np.linalg.norm matrix axes must have length two")
        if ord_value not in {None, "fro"}:
            raise ValueError("whole-program AD np.linalg.norm matrix norms support only Frobenius")
        if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in axis):
            raise ValueError("whole-program AD np.linalg.norm axes must be static integers")
        axes = tuple(
            _normalise_axis("whole-program AD np.linalg.norm matrix axis", int(item), array.ndim)
            for item in axis
        )
        if axes[0] == axes[1]:
            raise ValueError("whole-program AD np.linalg.norm axes must be distinct")
        reduced_axes = tuple(index for index in range(array.ndim) if index not in axes)
        reduced_shape = tuple(array.shape[index] for index in reduced_axes)
        if reduced_shape == ():
            return norm_from_items(
                tuple(array._items),
                zero_boundary_message=(
                    "whole-program AD np.linalg.norm requires non-zero Frobenius norms"
                ),
            )
        frobenius_items: list[TraceADScalar] = []
        for reduced_flat in range(int(np.prod(reduced_shape))):
            reduced_index = np.unravel_index(reduced_flat, reduced_shape)
            source_items: list[TraceADScalar] = []
            for first in range(array.shape[axes[0]]):
                for second in range(array.shape[axes[1]]):
                    frobenius_source_index = [0] * array.ndim
                    for position, dimension in enumerate(reduced_axes):
                        frobenius_source_index[dimension] = int(reduced_index[position])
                    frobenius_source_index[axes[0]] = first
                    frobenius_source_index[axes[1]] = second
                    source_items.append(
                        array._items[
                            int(np.ravel_multi_index(tuple(frobenius_source_index), array.shape))
                        ]
                    )
            frobenius_items.append(
                norm_from_items(
                    tuple(source_items),
                    zero_boundary_message=(
                        "whole-program AD np.linalg.norm requires non-zero Frobenius norms"
                    ),
                )
            )
        return TraceADArray(tuple(frobenius_items), reduced_shape, context)
    if ord_value not in {None, 2, 2.0}:
        raise ValueError("whole-program AD np.linalg.norm supports only Euclidean norm")
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("whole-program AD np.linalg.norm axis must be a static integer")
    axis_index = _normalise_axis("whole-program AD np.linalg.norm axis", int(axis), array.ndim)
    reduced_shape = array.shape[:axis_index] + array.shape[axis_index + 1 :]
    if reduced_shape == ():
        return norm_from_items(
            tuple(array._items),
            zero_boundary_message=(
                "whole-program AD np.linalg.norm requires non-zero Euclidean norms"
            ),
        )
    euclidean_items: list[TraceADScalar] = []
    for reduced_flat in range(int(np.prod(reduced_shape))):
        reduced_index = np.unravel_index(reduced_flat, reduced_shape)
        axis_items: list[TraceADScalar] = []
        for axis_position in range(array.shape[axis_index]):
            euclidean_source_index = (
                reduced_index[:axis_index] + (axis_position,) + reduced_index[axis_index:]
            )
            axis_items.append(
                array._items[int(np.ravel_multi_index(euclidean_source_index, array.shape))]
            )
        euclidean_items.append(
            norm_from_items(
                tuple(axis_items),
                zero_boundary_message=(
                    "whole-program AD np.linalg.norm requires non-zero Euclidean norms"
                ),
            )
        )
    return TraceADArray(tuple(euclidean_items), reduced_shape, context)


def whole_program_grad(
    objective: Callable[[Any], object],
    values: ArrayLike,
    parameters: Sequence[Parameter] | None = None,
    *,
    trace: bool = True,
) -> NDArray[np.float64]:
    """Return only the exact whole-program AD gradient."""

    return whole_program_value_and_grad(
        objective, values, parameters=parameters, trace=trace
    ).gradient


def program_adjoint_grad(
    objective: Callable[[Any], object],
    values: ArrayLike,
    parameters: Sequence[Parameter] | None = None,
    *,
    trace: bool = True,
) -> NDArray[np.float64]:
    """Return the reverse-mode program AD gradient for supported captured IR.

    The execution path first captures the operator-intercepted Program AD trace,
    then returns the reverse adjoint generation gradient. If generation does
    not support every captured IR node, this function fails closed instead of
    substituting a forward-mode tangent or finite-difference result.
    """

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=parameters,
        trace=trace,
    )
    return program_adjoint_gradient(result)


def program_adjoint_value_and_grad(
    objective: Callable[[Any], object],
    values: ArrayLike,
    parameters: Sequence[Parameter] | None = None,
    *,
    trace: bool = True,
) -> tuple[float, NDArray[np.float64]]:
    """Return the program value and reverse-mode adjoint generation gradient.

    This is the first-class reverse-mode program AD API. It keeps the same
    fail-closed generation boundary as :func:`program_adjoint_grad` and does not
    claim executable compiler lowering or arbitrary Python differentiation.
    """

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=parameters,
        trace=trace,
    )
    return result.value, program_adjoint_gradient(result)


def _program_adjoint_result_from_nodes(
    *,
    nodes: tuple[WholeProgramIRNode, ...],
    output_name: str,
    parameter_names: tuple[str, ...],
    trainable: tuple[bool, ...],
    program_ir: ProgramADEffectIR | None = None,
) -> ProgramADAdjointResult:
    """Generate reverse-mode adjoints over supported scalar Program AD IR nodes."""

    parameter_count = len(parameter_names)
    unsupported_ops: set[str] = {
        node.op
        for node in nodes
        if node.op.startswith("mutation:") and node.op != "mutation:setitem"
    }
    node_by_name = {f"%{node.index}": node for node in nodes}
    adjoints = {name: 0.0 for name in node_by_name}
    if output_name not in adjoints:
        unsupported_ops.add("output:not_in_ir")
    else:
        adjoints[output_name] = 1.0
    for node in reversed(nodes):
        name = f"%{node.index}"
        cotangent = adjoints.get(name, 0.0)
        if cotangent == 0.0:
            continue
        try:
            contributions = _program_adjoint_node_contributions(node, node_by_name)
        except ValueError:
            unsupported_ops.add(node.op)
            continue
        for input_name, contribution in contributions:
            if input_name in adjoints:
                adjoints[input_name] += cotangent * contribution
    gradient = np.zeros(parameter_count, dtype=np.float64)
    for index, (name, trainable_flag) in enumerate(zip(parameter_names, trainable, strict=True)):
        if not trainable_flag:
            continue
        for node in nodes:
            if node.op == "parameter" and node.inputs == (name,):
                gradient[index] = adjoints.get(f"%{node.index}", 0.0)
                break
    supported = not unsupported_ops
    if not supported:
        gradient = np.zeros(parameter_count, dtype=np.float64)
    replay_effect_count = len(program_ir.effects) if program_ir is not None else 0
    replay_control_region_count = len(program_ir.control_regions) if program_ir is not None else 0
    replay_phi_node_count = len(program_ir.phi_nodes) if program_ir is not None else 0
    adjoint_steps = (
        _program_adjoint_steps_from_ir(
            nodes=nodes,
            node_by_name=node_by_name,
            program_ir=program_ir,
            cotangents=adjoints,
        )
        if program_ir is not None
        else ()
    )
    return ProgramADAdjointResult(
        gradient=gradient,
        supported=supported,
        unsupported_ops=tuple(sorted(unsupported_ops)),
        method="program_adjoint_ir_generation",
        claim_boundary=(
            "reverse-mode adjoint generation over stabilized program_ad_effect_ir.v1 "
            "for supported executed scalar Program AD operations; unsupported operations "
            "fail closed without substituting finite differences or forward tangents; "
            "no non-executed branch adjoints or executable Rust/LLVM/JIT lowering claim"
        ),
        replay_node_count=len(nodes),
        replay_effect_count=replay_effect_count,
        replay_control_region_count=replay_control_region_count,
        replay_phi_node_count=replay_phi_node_count,
        replay_ir_format="program_ad_effect_ir.v1",
        adjoint_steps=adjoint_steps,
    )


def _program_adjoint_steps_from_ir(
    *,
    nodes: tuple[WholeProgramIRNode, ...],
    node_by_name: Mapping[str, WholeProgramIRNode],
    program_ir: ProgramADEffectIR,
    cotangents: Mapping[str, float],
) -> tuple[ProgramADAdjointStep, ...]:
    """Generate reverse-adjoint steps from stabilized Program AD IR metadata."""

    ssa_by_name = {value.name: value for value in program_ir.ssa_values}
    effect_by_index = {effect.index: effect for effect in program_ir.effects}
    runtime_regions_by_predicate: dict[str, list[ProgramADControlRegion]] = {}
    for region in program_ir.control_regions:
        if region.source_line is None and region.predicate is not None:
            runtime_regions_by_predicate.setdefault(region.predicate, []).append(region)
    runtime_phi_by_region: dict[int, ProgramADPhiNode] = {}
    ambiguous_phi_regions: set[int] = set()
    for phi_node in program_ir.phi_nodes:
        if phi_node.source_line is not None or phi_node.control_region is None:
            continue
        if phi_node.control_region in runtime_phi_by_region:
            ambiguous_phi_regions.add(phi_node.control_region)
            runtime_phi_by_region.pop(phi_node.control_region, None)
        elif phi_node.control_region not in ambiguous_phi_regions:
            runtime_phi_by_region[phi_node.control_region] = phi_node
    steps: list[ProgramADAdjointStep] = []
    for node in reversed(nodes):
        primal_value = f"%{node.index}"
        ssa_value = ssa_by_name.get(primal_value)
        primal_effect = None if ssa_value is None else ssa_value.effect
        effect = None if primal_effect is None else effect_by_index.get(primal_effect)
        effect_kind = None if effect is None else effect.kind
        effect_version = None if effect is None else effect.version
        effect_ordering = None if effect is None else effect.ordering
        unsupported_reason: str | None = None
        supported = True
        contribution_inputs: tuple[str, ...] = ()
        incoming_cotangent = float(cotangents.get(primal_value, 0.0))
        contribution_scales: tuple[float, ...] = ()
        contribution_cotangents: tuple[float, ...] = ()
        control_region: int | None = None
        control_region_kind: str | None = None
        control_region_entered: bool | None = None
        phi_node_index: int | None = None
        phi_selected: str | None = None
        if node.op.startswith("branch:"):
            runtime_regions = tuple(runtime_regions_by_predicate.get(node.op, ()))
            if len(runtime_regions) == 1:
                runtime_region = runtime_regions[0]
                control_region = runtime_region.index
                control_region_kind = runtime_region.kind
                control_region_entered = runtime_region.entered
                runtime_phi = runtime_phi_by_region.get(runtime_region.index)
                if runtime_phi is not None:
                    phi_node_index = runtime_phi.index
                    phi_selected = runtime_phi.selected
        if ssa_value is None:
            supported = False
            unsupported_reason = "missing_ssa_value"
        elif primal_effect is not None and effect is None:
            supported = False
            unsupported_reason = "missing_effect"
        else:
            try:
                contributions = _program_adjoint_node_contributions(node, node_by_name)
            except ValueError as exc:
                supported = False
                unsupported_reason = str(exc)
            else:
                scale_by_input: dict[str, float] = {}
                for input_name, scale in contributions:
                    scale_by_input[input_name] = scale_by_input.get(input_name, 0.0) + scale
                contribution_inputs = tuple(sorted(scale_by_input))
                contribution_scales = tuple(
                    scale_by_input[input_name] for input_name in contribution_inputs
                )
                contribution_cotangents = tuple(
                    incoming_cotangent * scale for scale in contribution_scales
                )
        steps.append(
            ProgramADAdjointStep(
                index=len(steps),
                primal_value=primal_value,
                primal_effect=primal_effect,
                effect_kind=effect_kind,
                effect_version=effect_version,
                effect_ordering=effect_ordering,
                control_region=control_region,
                control_region_kind=control_region_kind,
                control_region_entered=control_region_entered,
                phi_node=phi_node_index,
                phi_selected=phi_selected,
                operation=node.op,
                input_values=node.inputs,
                contribution_inputs=contribution_inputs,
                incoming_cotangent=incoming_cotangent,
                contribution_scales=contribution_scales,
                contribution_cotangents=contribution_cotangents,
                supported=supported,
                unsupported_reason=unsupported_reason,
            )
        )
    return tuple(steps)


def _program_adjoint_node_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse-mode contributions for one captured IR node."""

    if node.op == "parameter":
        return ()
    if node.op.startswith("branch:"):
        return ()
    if node.op == "mutation:setitem":
        return ()
    if node.op.startswith("mutation:"):
        raise ValueError("mutation adjoints require alias/effect semantics")
    if node.op == "neg":
        return ((node.inputs[0], -1.0),)
    if node.op in {
        "sin",
        "cos",
        "exp",
        "expm1",
        "log",
        "log1p",
        "sqrt",
        "tan",
        "tanh",
        "arcsin",
        "arccos",
        "reciprocal",
        "square",
        "abs",
    }:
        arg_name = node.inputs[0]
        arg_value = _program_adjoint_input_value(arg_name, node_by_name)
        if node.op == "sin":
            return ((arg_name, float(np.cos(arg_value))),)
        if node.op == "cos":
            return ((arg_name, -float(np.sin(arg_value))),)
        if node.op == "exp":
            return ((arg_name, node.value),)
        if node.op == "expm1":
            return ((arg_name, float(np.exp(arg_value))),)
        if node.op == "log":
            return ((arg_name, 1.0 / arg_value),)
        if node.op == "log1p":
            if arg_value <= -1.0:
                raise ValueError("log1p adjoint requires input greater than -1")
            return ((arg_name, 1.0 / (1.0 + arg_value)),)
        if node.op == "sqrt":
            return ((arg_name, 1.0 / (2.0 * node.value)),)
        if node.op == "tan":
            cosine = float(np.cos(arg_value))
            if abs(cosine) <= 1.0e-15:
                raise ValueError("tan adjoint requires non-zero cosine")
            return ((arg_name, 1.0 / cosine**2),)
        if node.op == "tanh":
            return ((arg_name, 1.0 - node.value**2),)
        if node.op in {"arcsin", "arccos"}:
            if abs(arg_value) >= 1.0:
                raise ValueError(f"{node.op} adjoint requires input strictly inside (-1, 1)")
            scale = 1.0 / float(np.sqrt(1.0 - arg_value**2))
            if node.op == "arccos":
                scale = -scale
            return ((arg_name, scale),)
        if node.op == "reciprocal":
            if arg_value == 0.0:
                raise ValueError("reciprocal adjoint requires non-zero input")
            return ((arg_name, -1.0 / arg_value**2),)
        if node.op == "square":
            return ((arg_name, 2.0 * arg_value),)
        if node.op == "abs":
            if arg_value == 0.0:
                raise ValueError("abs adjoint is undefined at zero")
            return ((arg_name, 1.0 if arg_value > 0.0 else -1.0),)
    if node.op in {
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "maximum",
        "minimum",
        "where",
        "clip",
        "choose",
    }:
        return _program_adjoint_binary_or_selection_contributions(node, node_by_name)
    if node.op.startswith("linalg:det:"):
        return _program_adjoint_det_contributions(node, node_by_name)
    if node.op.startswith("linalg:inv:"):
        return _program_adjoint_inv_contributions(node, node_by_name)
    if node.op.startswith("linalg:solve:"):
        return _program_adjoint_solve_contributions(node, node_by_name)
    if node.op.startswith("linalg:trace:"):
        return _program_adjoint_trace_contributions(node)
    if node.op.startswith("linalg:diag:"):
        return _program_adjoint_diag_contributions(node)
    if node.op.startswith("linalg:diagflat:"):
        return _program_adjoint_diagflat_contributions(node)
    if node.op.startswith("linalg:matrix_power:"):
        return _program_adjoint_matrix_power_contributions(node, node_by_name)
    if node.op.startswith("linalg:multi_dot:"):
        return _program_adjoint_multi_dot_contributions(node, node_by_name)
    if node.op.startswith("linalg:eigh:eigenvalue:"):
        return _program_adjoint_eigh_eigenvalue_contributions(node, node_by_name)
    if node.op.startswith("linalg:eigh:eigenvector:"):
        return _program_adjoint_eigh_eigenvector_contributions(node, node_by_name)
    if node.op.startswith("linalg:eig:eigenvalue:"):
        return _program_adjoint_eig_eigenvalue_contributions(node, node_by_name)
    if node.op.startswith("linalg:eig:eigenvector:"):
        return _program_adjoint_eig_eigenvector_contributions(node, node_by_name)
    if node.op.startswith("linalg:eigvalsh:"):
        return _program_adjoint_eigvalsh_contributions(node, node_by_name)
    if node.op.startswith("linalg:eigvals:"):
        return _program_adjoint_eigvals_contributions(node, node_by_name)
    if node.op.startswith("linalg:svdvals:"):
        return _program_adjoint_svdvals_contributions(node, node_by_name)
    if node.op.startswith("linalg:pinv:"):
        return _program_adjoint_pinv_contributions(node, node_by_name)
    raise ValueError(f"unsupported program AD adjoint op {node.op}")


def _program_adjoint_binary_or_selection_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    if node.op == "where":
        if len(node.inputs) != 3:
            raise ValueError("where adjoint requires predicate, true value, and false value")
        predicate_truth = _program_adjoint_where_predicate_truth(node.inputs[0])
        left_name = node.inputs[1]
        right_name = node.inputs[2]
        return ((left_name, 1.0),) if predicate_truth else ((right_name, 1.0),)
    if node.op == "clip":
        if len(node.inputs) != 3:
            raise ValueError("clip adjoint requires value, lower, and upper inputs")
        value_name, lower_name, upper_name = node.inputs
        value = _program_adjoint_input_value(value_name, node_by_name)
        lower = _program_adjoint_input_value(lower_name, node_by_name)
        upper = _program_adjoint_input_value(upper_name, node_by_name)
        if value < lower:
            return ((lower_name, 1.0),)
        if value > upper:
            return ((upper_name, 1.0),)
        if value in (lower, upper):
            raise ValueError("clip adjoint is undefined at clipping boundary")
        return ((value_name, 1.0),)
    if node.op == "choose":
        if len(node.inputs) != 2 or not node.inputs[0].startswith("static_selector:"):
            raise ValueError("choose adjoint requires static selector and selected value")
        return ((node.inputs[1], 1.0),)
    left_name = node.inputs[0]
    right_name = node.inputs[1] if len(node.inputs) > 1 else ""
    left = _program_adjoint_input_value(left_name, node_by_name)
    right = _program_adjoint_input_value(right_name, node_by_name) if right_name else 0.0
    if node.op == "add":
        return ((left_name, 1.0), (right_name, 1.0))
    if node.op == "sub":
        return ((left_name, 1.0), (right_name, -1.0))
    if node.op == "mul":
        return ((left_name, right), (right_name, left))
    if node.op == "div":
        return ((left_name, 1.0 / right), (right_name, -left / right**2))
    if node.op == "pow":
        if left <= 0.0 and _program_adjoint_is_ir_value(right_name):
            raise ValueError("variable exponent adjoint requires positive base")
        primal = node.value
        contributions = [(left_name, right * left ** (right - 1.0))]
        if _program_adjoint_is_ir_value(right_name):
            contributions.append((right_name, primal * float(np.log(left))))
        return tuple(contributions)
    if node.op == "maximum":
        if left == right:
            raise ValueError("maximum adjoint is undefined at ties")
        return ((left_name, 1.0),) if node.value == left else ((right_name, 1.0),)
    if node.op == "minimum":
        if left == right:
            raise ValueError("minimum adjoint is undefined at ties")
        return ((left_name, 1.0),) if node.value == left else ((right_name, 1.0),)
    raise ValueError(f"unsupported program AD adjoint op {node.op}")


def _program_adjoint_where_predicate_truth(predicate_name: str) -> bool:
    if predicate_name.endswith(":truth:1"):
        return True
    if predicate_name.endswith(":truth:0"):
        return False
    if predicate_name == "constant:True":
        return True
    if predicate_name == "constant:False":
        return False
    raise ValueError("where adjoint requires recorded predicate branch")


def _normalise_axis(name: str, axis: int, ndim: int) -> int:
    """Return a non-negative axis for an array with ``ndim`` dimensions."""

    if ndim == 0:
        raise ValueError(f"{name} cannot map over a scalar")
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"{name} is out of bounds for argument rank {ndim}")
    return axis


def _program_ad_float64_vector_result(values: object) -> NDArray[np.float64]:
    return cast(NDArray[np.float64], np.asarray(values, dtype=np.float64).reshape(-1))


def _program_ad_elementwise_unbroadcast(
    values: NDArray[np.float64],
    *,
    target_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if target_shape == ():
        return np.array([float(np.sum(result))], dtype=np.float64)
    while result.ndim > len(target_shape):
        result = np.sum(result, axis=0)
    for axis, dimension in enumerate(target_shape):
        if dimension == 1 and result.shape[axis] != 1:
            result = np.sum(result, axis=axis, keepdims=True)
    return _program_ad_float64_vector_result(result.reshape(target_shape))


_register_program_ad_array_primitive_contracts()
_register_program_ad_interpolation_primitive_contracts()
_register_program_ad_assembly_primitive_contracts()
_register_program_ad_signal_primitive_contracts()
_register_program_ad_shape_primitive_contracts()
_register_program_ad_reduction_primitive_contracts()
_register_program_ad_stencil_primitive_contracts()
_register_program_ad_elementwise_primitive_contracts()
_register_program_ad_selection_primitive_contracts()
_register_program_ad_product_primitive_contracts()
_register_program_ad_cumulative_primitive_contracts()
_register_program_ad_linalg_primitive_contracts()


def value_and_grad(
    objective: Callable[[Any], Any],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "parameter_shift",
    rule: ParameterShiftRule | None = None,
    step: float | None = None,
) -> GradientResult | WholeProgramADResult:
    """Evaluate a scalar objective and gradient through a canonical transform API."""

    if method == "parameter_shift":
        return value_and_parameter_shift_grad(
            cast(ScalarObjective, objective),
            values,
            parameters=parameters,
            rule=rule,
        )
    if method == "finite_difference":
        return value_and_finite_difference_grad(
            cast(ScalarObjective, objective),
            values,
            parameters=parameters,
            step=1.0e-6 if step is None else step,
        )
    if method == "complex_step":
        return value_and_complex_step_grad(
            cast(ComplexStepObjective, objective),
            values,
            parameters=parameters,
            step=1.0e-30 if step is None else step,
        )
    if method == "forward_mode":
        return value_and_forward_mode_grad(
            cast(Callable[[tuple[DualNumber, ...]], object], objective),
            values,
            parameters=parameters,
        )
    if method == "reverse_mode":
        return value_and_reverse_mode_grad(
            cast(Callable[[tuple[ReverseNode, ...]], object], objective),
            values,
            parameters=parameters,
        )
    if method == "whole_program":
        return whole_program_value_and_grad(
            objective,
            values,
            parameters=parameters,
            trace=True,
        )
    raise ValueError(
        "gradient method must be one of: parameter_shift, finite_difference, complex_step, "
        "forward_mode, reverse_mode, whole_program"
    )


def grad(
    objective: Callable[[Any], Any],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "parameter_shift",
    rule: ParameterShiftRule | None = None,
    step: float | None = None,
) -> NDArray[np.float64]:
    """Return a scalar-objective gradient through the canonical transform API."""

    result = value_and_grad(
        objective,
        values,
        parameters=parameters,
        method=method,
        rule=rule,
        step=step,
    )
    return result.gradient


__all__ = [
    "ArmijoLineSearchResult",
    "CustomDerivativeCheckResult",
    "CustomDerivativeRule",
    "CustomDerivativeRegistry",
    "DEFAULT_CUSTOM_DERIVATIVE_REGISTRY",
    "DifferentiableOptimizer",
    "DualNumber",
    "DIFFERENTIABLE_RESULT_CLAIM_BOUNDARY",
    "FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY",
    "FixedPointSensitivityResult",
    "FisherConjugateGradientResult",
    "FisherVectorProductResult",
    "GradientCheckResult",
    "GradientFailurePolicy",
    "GradientResult",
    "HVPResult",
    "HessianResult",
    "ImplicitSensitivityResult",
    "JVPResult",
    "JacobianResult",
    "LeastSquaresCovarianceResult",
    "LevenbergMarquardtDampingUpdate",
    "LevenbergMarquardtOptimizer",
    "LevenbergMarquardtResult",
    "LevenbergMarquardtStep",
    "LevenbergMarquardtTrial",
    "NaturalGradientResult",
    "NaturalGradientOptimizationResult",
    "NaturalGradientOptimizer",
    "OptimizationResult",
    "Parameter",
    "ParameterBounds",
    "ParameterShiftRule",
    "ParameterShiftSampleRecord",
    "ProgramADAdjointResult",
    "ProgramADAdjointStep",
    "ProgramADAliasEdge",
    "ProgramADAliasEffectAnalysis",
    "ProgramADAliasSet",
    "ProgramADStaticAliasLatticeComponent",
    "ProgramADStaticAliasLatticeReport",
    "ProgramADControlRegion",
    "ProgramADEffect",
    "ProgramADEffectIR",
    "ProgramADLinalgConditioningDiagnostic",
    "ProgramADPhiNode",
    "ProgramADRegistryDispatchCoverageReport",
    "ProgramADRegistryDispatchCoverageRow",
    "RustProgramADInterpreterResult",
    "RustProgramADValueAndGradientResult",
    "ProgramADSSAValue",
    "PrimitiveBatchingRule",
    "PrimitiveContract",
    "PrimitiveDTypeRule",
    "PrimitiveIdentity",
    "PrimitiveLoweringRule",
    "PrimitiveShapeRule",
    "PrimitiveStaticArgumentRule",
    "PrimitiveTransformRule",
    "ReverseNode",
    "ShotAllocationResult",
    "ScoreFunctionGradientResult",
    "ScoreFunctionSampleRecord",
    "SPSAGradientResult",
    "SPSAObjectiveSample",
    "SPSAProbeRecord",
    "SparseMatrixResult",
    "StochasticGradientConfidenceInterval",
    "StochasticGradientResult",
    "STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY",
    "VJPResult",
    "WeightedGradientResult",
    "analyze_program_ad_alias_effects",
    "program_ad_static_alias_lattice_report",
    "armijo_backtracking_line_search",
    "allocate_parameter_shift_shots",
    "batch_custom_jacobian",
    "batch_custom_jvp",
    "batch_custom_vjp",
    "batch_finite_difference_hvp",
    "batch_finite_difference_jvp",
    "batch_finite_difference_vjp",
    "batch_complex_step_gradient",
    "batch_parameter_shift_gradient",
    "batch_value_and_complex_step_grad",
    "batch_value_and_custom_jacobian",
    "batch_value_and_custom_jvp",
    "batch_value_and_custom_vjp",
    "batch_value_and_finite_difference_grad",
    "batch_value_and_finite_difference_hvp",
    "batch_value_and_finite_difference_jvp",
    "batch_value_and_finite_difference_vjp",
    "batch_value_and_parameter_shift_grad",
    "batch_vector_jacobian_product",
    "check_parameter_shift_consistency",
    "check_custom_derivative_consistency",
    "complex_step_gradient",
    "custom_jacobian",
    "custom_gauss_newton_gradient",
    "custom_derivative_rule_for",
    "custom_jvp",
    "custom_levenberg_marquardt_step",
    "custom_vjp",
    "dual_cos",
    "dual_exp",
    "dual_log",
    "dual_sin",
    "dense_to_sparse_matrix",
    "diagnose_program_ad_linalg_conditioning",
    "empirical_fisher_conjugate_gradient",
    "empirical_fisher_metric",
    "empirical_fisher_vector_product",
    "evaluate_levenberg_marquardt_step",
    "finite_difference_gradient",
    "finite_difference_hessian",
    "finite_difference_hvp",
    "finite_difference_jacobian",
    "finite_difference_jvp",
    "finite_difference_vjp",
    "forward_mode_gradient",
    "gauss_newton_gradient",
    "grad",
    "gradient_confidence_interval",
    "huber_residual_weights",
    "hessian",
    "implicit_fixed_point_sensitivity",
    "implicit_stationary_sensitivity",
    "is_jax_autodiff_available",
    "jacfwd",
    "jacobian",
    "jacrev",
    "jax_value_and_grad",
    "jvp",
    "least_squares_covariance",
    "levenberg_marquardt_step",
    "natural_gradient",
    "interpret_program_ad_effect_ir_with_rust",
    "value_and_grad_program_ad_effect_ir_with_rust",
    "parse_program_ad_effect_ir",
    "primitive_complete_contract_for",
    "primitive_contract_for",
    "primitive_dtype_rule_for",
    "primitive_effect_for",
    "primitive_nondifferentiable_policy_for",
    "primitive_shape_rule_for",
    "primitive_static_argument_rule_for",
    "program_ad_registry_dispatch_coverage_report",
    "program_ad_shape_atleast_1d_derivative_rule",
    "program_ad_shape_atleast_2d_derivative_rule",
    "program_ad_shape_atleast_3d_derivative_rule",
    "program_ad_assembly_append_derivative_rule",
    "program_ad_assembly_block_derivative_rule",
    "program_ad_assembly_broadcast_arrays_derivative_rule",
    "program_ad_assembly_broadcast_to_derivative_rule",
    "program_ad_assembly_column_stack_derivative_rule",
    "program_ad_assembly_concatenate_derivative_rule",
    "program_ad_assembly_dstack_derivative_rule",
    "program_ad_assembly_hstack_derivative_rule",
    "program_ad_assembly_split_derivative_rule",
    "program_ad_assembly_stack_derivative_rule",
    "program_ad_assembly_vstack_derivative_rule",
    "program_ad_array_delete_derivative_rule",
    "program_ad_array_getitem_derivative_rule",
    "program_ad_array_insert_derivative_rule",
    "program_ad_array_pad_derivative_rule",
    "program_ad_array_take_along_axis_derivative_rule",
    "program_ad_array_take_derivative_rule",
    "program_ad_cumulative_cumprod_derivative_rule",
    "program_ad_cumulative_cumsum_derivative_rule",
    "program_ad_cumulative_diff_derivative_rule",
    "program_ad_elementwise_binary_derivative_rule",
    "program_ad_interpolation_interp_derivative_rule",
    "program_ad_assembly_diagonal_derivative_rule",
    "program_ad_assembly_tril_derivative_rule",
    "program_ad_assembly_triu_derivative_rule",
    "program_ad_signal_convolve_derivative_rule",
    "program_ad_signal_correlate_derivative_rule",
    "program_ad_linalg_diag_derivative_rule",
    "program_ad_linalg_diagflat_derivative_rule",
    "program_ad_linalg_eig_derivative_rule",
    "program_ad_linalg_eigh_derivative_rule",
    "program_ad_linalg_eigvals_derivative_rule",
    "program_ad_linalg_eigvalsh_derivative_rule",
    "program_ad_linalg_matrix_power_derivative_rule",
    "program_ad_linalg_multi_dot_derivative_rule",
    "program_ad_linalg_pinv_derivative_rule",
    "program_ad_linalg_solve_derivative_rule",
    "program_ad_linalg_svdvals_derivative_rule",
    "program_ad_linalg_trace_derivative_rule",
    "program_ad_product_einsum_derivative_rule",
    "program_ad_product_inner_derivative_rule",
    "program_ad_product_matmul_derivative_rule",
    "program_ad_product_outer_derivative_rule",
    "program_ad_product_tensordot_derivative_rule",
    "program_ad_reduction_max_derivative_rule",
    "program_ad_reduction_mean_derivative_rule",
    "program_ad_reduction_median_derivative_rule",
    "program_ad_reduction_min_derivative_rule",
    "program_ad_reduction_percentile_derivative_rule",
    "program_ad_reduction_prod_derivative_rule",
    "program_ad_reduction_quantile_derivative_rule",
    "program_ad_reduction_sum_derivative_rule",
    "program_ad_reduction_std_derivative_rule",
    "program_ad_reduction_trapezoid_derivative_rule",
    "program_ad_reduction_var_derivative_rule",
    "program_ad_selection_clip_derivative_rule",
    "program_ad_selection_where_derivative_rule",
    "program_ad_shape_expand_dims_derivative_rule",
    "program_ad_shape_flip_derivative_rule",
    "program_ad_shape_fliplr_derivative_rule",
    "program_ad_shape_flipud_derivative_rule",
    "program_ad_shape_moveaxis_derivative_rule",
    "program_ad_shape_ravel_derivative_rule",
    "program_ad_shape_repeat_derivative_rule",
    "program_ad_shape_reshape_derivative_rule",
    "program_ad_shape_roll_derivative_rule",
    "program_ad_shape_rot90_derivative_rule",
    "program_ad_shape_squeeze_derivative_rule",
    "program_ad_shape_swapaxes_derivative_rule",
    "program_ad_shape_tile_derivative_rule",
    "program_ad_shape_transpose_derivative_rule",
    "program_ad_stencil_gradient_derivative_rule",
    "program_adjoint_grad",
    "program_adjoint_gradient",
    "program_adjoint_result",
    "program_adjoint_value_and_grad",
    "registered_custom_jacobian",
    "register_primitive_batching_rule",
    "register_primitive_lowering_rule",
    "register_primitive_transform_rule",
    "registered_custom_jvp",
    "registered_custom_vjp",
    "register_custom_derivative_rule",
    "reverse_cos",
    "reverse_exp",
    "reverse_log",
    "reverse_mode_gradient",
    "reverse_sin",
    "soft_l1_residual_weights",
    "sparse_empirical_fisher_metric",
    "sparse_hessian",
    "sparse_jacobian",
    "score_function_gradient_estimate",
    "spsa_gradient_estimate",
    "multi_frequency_parameter_shift_rule",
    "parameter_shift_gradient_with_uncertainty",
    "update_levenberg_marquardt_damping",
    "weighted_gradient_sum",
    "value_and_grad",
    "whole_program_grad",
    "whole_program_value_and_grad",
    "TraceADArray",
    "TraceADScalar",
    "WholeProgramADResult",
    "WholeProgramBytecodeBasicBlock",
    "WholeProgramBytecodeInstruction",
    "WholeProgramCompilerFrontendReport",
    "WholeProgramIRNode",
    "WholeProgramSemanticsReport",
    "WholeProgramSourceBytecodeLineMap",
    "WholeProgramSourceIRFeature",
    "WholeProgramSourceRegion",
    "WholeProgramSymbolScopeEntry",
    "WholeProgramTraceEvent",
    "WholeProgramUnsupportedSemanticDiagnostic",
    "compile_whole_program_frontend",
    "vmap",
    "parameter_shift_gradient",
    "value_and_complex_step_grad",
    "value_and_custom_jacobian",
    "value_and_custom_jvp",
    "value_and_custom_vjp",
    "value_and_finite_difference_grad",
    "value_and_finite_difference_hessian",
    "value_and_finite_difference_hvp",
    "value_and_finite_difference_jacobian",
    "value_and_finite_difference_jvp",
    "value_and_finite_difference_vjp",
    "value_and_forward_mode_grad",
    "value_and_hessian",
    "value_and_jacfwd",
    "value_and_jacobian",
    "value_and_jacrev",
    "value_and_jvp",
    "value_and_parameter_shift_grad",
    "value_and_reverse_mode_grad",
    "value_and_vjp",
    "vector_jacobian_product",
    "vjp",
]
