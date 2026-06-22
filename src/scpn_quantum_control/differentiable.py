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

from collections.abc import Callable, Sequence
from typing import Any, cast

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
    multi_frequency_parameter_shift_rule,
)
from .differentiable_parameter_contracts import (
    _as_real_numeric_array as _as_real_numeric_array,
)
from .differentiable_parameter_contracts import (
    _as_real_scalar as _as_real_scalar,
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
    _program_adjoint_result_from_nodes,
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
    _program_ad_array_delete_object as _program_ad_array_delete_object,
)
from .program_ad_array_indexing import (
    _program_ad_array_insert_layout as _program_ad_array_insert_layout,
)
from .program_ad_array_indexing import (
    _program_ad_array_pad_layout as _program_ad_array_pad_layout,
)
from .program_ad_array_indexing import (
    _program_ad_array_pad_mode as _program_ad_array_pad_mode,
)
from .program_ad_array_indexing import (
    _program_ad_array_shape_of as _program_ad_array_shape_of,
)
from .program_ad_array_indexing import (
    _program_ad_array_take_indices as _program_ad_array_take_indices,
)
from .program_ad_array_indexing import (
    _program_ad_array_take_mode as _program_ad_array_take_mode,
)
from .program_ad_array_indexing import (
    _register_program_ad_array_primitive_contracts,
    program_ad_array_delete_derivative_rule,
    program_ad_array_getitem_derivative_rule,
    program_ad_array_insert_derivative_rule,
    program_ad_array_pad_derivative_rule,
    program_ad_array_take_along_axis_derivative_rule,
    program_ad_array_take_derivative_rule,
)
from .program_ad_array_indexing import (
    _require_program_ad_array_contract as _require_program_ad_array_contract,
)
from .program_ad_assembly_primitives import (
    _register_program_ad_assembly_primitive_contracts,
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
from .program_ad_assembly_primitives import (
    _require_program_ad_assembly_contract as _require_program_ad_assembly_contract,
)
from .program_ad_cumulative_primitives import (
    _register_program_ad_cumulative_primitive_contracts,
    program_ad_cumulative_cumprod_derivative_rule,
    program_ad_cumulative_cumsum_derivative_rule,
    program_ad_cumulative_diff_derivative_rule,
)
from .program_ad_cumulative_primitives import (
    _require_program_ad_cumulative_contract as _require_program_ad_cumulative_contract,
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
    _program_ad_elementwise_name as _program_ad_elementwise_name,
)
from .program_ad_elementwise_primitives import (
    _raise_program_ad_derivative_losing_elementwise as _raise_program_ad_derivative_losing_elementwise,
)
from .program_ad_elementwise_primitives import (
    _register_program_ad_elementwise_primitive_contracts,
    program_ad_elementwise_binary_derivative_rule,
)
from .program_ad_elementwise_primitives import (
    _require_program_ad_elementwise_contract as _require_program_ad_elementwise_contract,
)
from .program_ad_interpolation_primitives import (
    _normalise_interp_grid as _normalise_interp_grid,
)
from .program_ad_interpolation_primitives import (
    _register_program_ad_interpolation_primitive_contracts,
    program_ad_interpolation_interp_derivative_rule,
)
from .program_ad_interpolation_primitives import (
    _require_program_ad_interpolation_contract as _require_program_ad_interpolation_contract,
)
from .program_ad_linalg_primitives import (
    ProgramADLinalgConditioningDiagnostic,
    _register_program_ad_linalg_primitive_contracts,
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
from .program_ad_linalg_primitives import (
    _program_ad_linalg_det_cofactor_matrix as _program_ad_linalg_det_cofactor_matrix,
)
from .program_ad_linalg_primitives import (
    _program_ad_linalg_eig_eigenvector_jvp_matrix as _program_ad_linalg_eig_eigenvector_jvp_matrix,
)
from .program_ad_linalg_primitives import (
    _program_ad_linalg_eigh_eigenvector_jvp_matrix as _program_ad_linalg_eigh_eigenvector_jvp_matrix,
)
from .program_ad_linalg_primitives import (
    _program_ad_linalg_normalise_rcond as _program_ad_linalg_normalise_rcond,
)
from .program_ad_linalg_primitives import (
    _program_ad_linalg_pinv_jvp_matrix as _program_ad_linalg_pinv_jvp_matrix,
)
from .program_ad_linalg_primitives import (
    _program_ad_linalg_pinv_value_matrix as _program_ad_linalg_pinv_value_matrix,
)
from .program_ad_linalg_primitives import (
    _program_ad_linalg_real_simple_eig_decomposition_from_matrix as _program_ad_linalg_real_simple_eig_decomposition_from_matrix,
)
from .program_ad_linalg_primitives import (
    _program_ad_linalg_require_distinct_eigenvalues as _program_ad_linalg_require_distinct_eigenvalues,
)
from .program_ad_linalg_primitives import (
    _program_ad_linalg_require_distinct_positive_singular_values as _program_ad_linalg_require_distinct_positive_singular_values,
)
from .program_ad_linalg_primitives import (
    _program_ad_linalg_require_symmetric as _program_ad_linalg_require_symmetric,
)
from .program_ad_linalg_primitives import (
    _program_ad_linalg_uplo as _program_ad_linalg_uplo,
)
from .program_ad_linalg_primitives import (
    _require_program_ad_linalg_contract as _require_program_ad_linalg_contract,
)
from .program_ad_product_primitives import (
    _normalise_program_ad_product_tensordot_signature as _normalise_program_ad_product_tensordot_signature,
)
from .program_ad_product_primitives import (
    _parse_static_einsum_subscripts as _parse_static_einsum_subscripts,
)
from .program_ad_product_primitives import (
    _register_program_ad_product_primitive_contracts,
    program_ad_product_einsum_derivative_rule,
    program_ad_product_inner_derivative_rule,
    program_ad_product_matmul_derivative_rule,
    program_ad_product_outer_derivative_rule,
    program_ad_product_tensordot_derivative_rule,
)
from .program_ad_product_primitives import (
    _require_program_ad_product_contract as _require_program_ad_product_contract,
)
from .program_ad_reduction_primitives import (
    _normalise_ddof as _normalise_ddof,
)
from .program_ad_reduction_primitives import (
    _normalise_order_statistic_axis as _normalise_order_statistic_axis,
)
from .program_ad_reduction_primitives import (
    _normalise_order_statistic_method as _normalise_order_statistic_method,
)
from .program_ad_reduction_primitives import (
    _normalise_order_statistic_q as _normalise_order_statistic_q,
)
from .program_ad_reduction_primitives import (
    _register_program_ad_reduction_primitive_contracts,
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
from .program_ad_reduction_primitives import (
    _require_program_ad_reduction_contract as _require_program_ad_reduction_contract,
)
from .program_ad_reduction_primitives import (
    _require_strict_order_statistic_values as _require_strict_order_statistic_values,
)
from .program_ad_registry import (
    _PROGRAM_AD_SELECTION_IDENTITIES as _PROGRAM_AD_SELECTION_IDENTITIES,
)
from .program_ad_registry import (
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
    program_ad_selection_clip_derivative_rule,
    program_ad_selection_where_derivative_rule,
)
from .program_ad_selection_primitives import (
    _require_program_ad_selection_contract as _require_program_ad_selection_contract,
)
from .program_ad_shape_transforms import (
    _register_program_ad_shape_primitive_contracts,
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
from .program_ad_shape_transforms import (
    _require_program_ad_shape_contract as _require_program_ad_shape_contract,
)
from .program_ad_signal_primitives import (
    _convolve_output_window as _convolve_output_window,
)
from .program_ad_signal_primitives import (
    _normalise_convolve_mode as _normalise_convolve_mode,
)
from .program_ad_signal_primitives import (
    _normalise_correlate_mode as _normalise_correlate_mode,
)
from .program_ad_signal_primitives import (
    _register_program_ad_signal_primitive_contracts,
    program_ad_signal_convolve_derivative_rule,
    program_ad_signal_correlate_derivative_rule,
)
from .program_ad_signal_primitives import (
    _require_program_ad_signal_contract as _require_program_ad_signal_contract,
)
from .program_ad_stencil_primitives import (
    _gradient_axis_coefficients as _gradient_axis_coefficients,
)
from .program_ad_stencil_primitives import (
    _GradientSpacing as _GradientSpacing,
)
from .program_ad_stencil_primitives import (
    _normalise_gradient_axes as _normalise_gradient_axes,
)
from .program_ad_stencil_primitives import (
    _normalise_gradient_edge_order as _normalise_gradient_edge_order,
)
from .program_ad_stencil_primitives import (
    _normalise_gradient_spacings as _normalise_gradient_spacings,
)
from .program_ad_stencil_primitives import (
    _register_program_ad_stencil_primitive_contracts,
    program_ad_stencil_gradient_derivative_rule,
)
from .program_ad_stencil_primitives import (
    _require_program_ad_stencil_contract as _require_program_ad_stencil_contract,
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
from .whole_program_trace_values import (
    ScalarObjective as ScalarObjective,
)
from .whole_program_trace_values import (
    TraceADArray as TraceADArray,
)
from .whole_program_trace_values import (
    TraceADScalar as TraceADScalar,
)

VectorObjective = Callable[[NDArray[np.float64]], ArrayLike]
ComplexStepObjective = Callable[[NDArray[np.complex128]], object]
CustomJVPRule = Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]
CustomVJPRule = Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]


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
