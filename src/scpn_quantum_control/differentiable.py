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
from dataclasses import dataclass
from typing import Any, Literal, NoReturn, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_parameter_contracts import (
    Parameter,
    ParameterBounds,
    ParameterShiftRule,
    _as_parameter_array,
    _as_real_numeric_array,
    _as_real_scalar,
    multi_frequency_parameter_shift_rule,
)
from .differentiable_result_contracts import (
    DIFFERENTIABLE_RESULT_CLAIM_BOUNDARY,
    FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY,
    ArmijoLineSearchResult,
    CustomDerivativeCheckResult,
    FixedPointSensitivityResult,
    GradientCheckResult,
    GradientResult,
    HessianResult,
    HVPResult,
    ImplicitSensitivityResult,
    JacobianResult,
    JVPResult,
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
    _normalise_claim_boundary,
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
from .differentiable_stochastic_policy import (
    STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY,
    GradientFailurePolicy,
    StochasticGradientConfidenceInterval,
    gradient_confidence_interval,
)
from .program_ad_adjoint import (
    ProgramADAdjointResult,
    ProgramADAdjointStep,
    _program_adjoint_input_value,
    _program_adjoint_is_ir_value,
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
    _program_ad_array_derivative_rule,
    _program_ad_array_direct_jvp,
    _program_ad_array_direct_value,
    _program_ad_array_insert_axis,
    _program_ad_array_insert_layout,
    _program_ad_array_insert_object,
    _program_ad_array_insert_values,
    _program_ad_array_normalise_static_shape,
    _program_ad_array_pad_constant_values,
    _program_ad_array_pad_layout,
    _program_ad_array_pad_mode,
    _program_ad_array_pad_width,
    _program_ad_array_signature,
    _program_ad_array_static_size,
    _program_ad_array_take_indices,
    _program_ad_array_take_mode,
    _program_ad_array_vector,
    program_ad_array_delete_derivative_rule,
    program_ad_array_getitem_derivative_rule,
    program_ad_array_insert_derivative_rule,
    program_ad_array_pad_derivative_rule,
    program_ad_array_take_along_axis_derivative_rule,
    program_ad_array_take_derivative_rule,
)
from .program_ad_cumulative_primitives import (
    _program_ad_cumulative_derivative_rule,
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
from .program_ad_interpolation_primitives import (
    _normalise_interp_grid,
    _program_ad_interp_static_boundary,
    program_ad_interpolation_interp_derivative_rule,
)
from .program_ad_registry import (
    _PROGRAM_AD_ARRAY_IDENTITIES,
    _PROGRAM_AD_ARRAY_POLICY,
    _PROGRAM_AD_ASSEMBLY_IDENTITIES,
    _PROGRAM_AD_ASSEMBLY_POLICY,
    _PROGRAM_AD_ASSEMBLY_SPLIT_NAMES,
    _PROGRAM_AD_CUMULATIVE_IDENTITIES,
    _PROGRAM_AD_CUMULATIVE_POLICY,
    _PROGRAM_AD_ELEMENTWISE_BINARY_NAMES,
    _PROGRAM_AD_ELEMENTWISE_DISCONTINUOUS_NAMES,
    _PROGRAM_AD_ELEMENTWISE_IDENTITIES,
    _PROGRAM_AD_ELEMENTWISE_NAMES,
    _PROGRAM_AD_ELEMENTWISE_POLICY,
    _PROGRAM_AD_ELEMENTWISE_UNARY_NAMES,
    _PROGRAM_AD_INTERPOLATION_IDENTITIES,
    _PROGRAM_AD_INTERPOLATION_POLICY,
    _PROGRAM_AD_LINALG_IDENTITIES,
    _PROGRAM_AD_LINALG_POLICY,
    _PROGRAM_AD_PRODUCT_IDENTITIES,
    _PROGRAM_AD_PRODUCT_POLICY,
    _PROGRAM_AD_REDUCTION_IDENTITIES,
    _PROGRAM_AD_REDUCTION_POLICY,
    _PROGRAM_AD_SELECTION_IDENTITIES,
    _PROGRAM_AD_SELECTION_POLICY,
    _PROGRAM_AD_SHAPE_IDENTITIES,
    _PROGRAM_AD_SHAPE_POLICY,
    _PROGRAM_AD_SIGNAL_IDENTITIES,
    _PROGRAM_AD_SIGNAL_POLICY,
    _PROGRAM_AD_STENCIL_IDENTITIES,
    _PROGRAM_AD_STENCIL_POLICY,
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
from .program_ad_shape_transforms import (
    _program_ad_shape_atleast_target_shape,
    _program_ad_shape_derivative_rule,
    _program_ad_shape_insert_singleton_axes,
    _program_ad_shape_normalise_expand_dims_axes,
    _program_ad_shape_normalise_flip_axis,
    _program_ad_shape_normalise_moveaxis_axes,
    _program_ad_shape_normalise_repeat_signature,
    _program_ad_shape_normalise_roll_signature,
    _program_ad_shape_normalise_squeeze_axes,
    _program_ad_shape_normalise_tile_signature,
    _program_ad_shape_remove_axes,
    _program_ad_shape_rot90_target_shape,
    _program_ad_shape_signature,
    _program_ad_shape_static_size,
    _program_ad_shape_vector,
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
    _program_ad_signal_convolve_output_size,
    _program_ad_signal_correlate_output_size,
    program_ad_signal_convolve_derivative_rule,
    program_ad_signal_correlate_derivative_rule,
)
from .program_ad_stack_block_assembly import (
    _program_ad_assembly_append_output_shape,
    _program_ad_assembly_block_output_shape,
    _program_ad_assembly_concatenate_axis,
    _program_ad_assembly_concatenate_output_shape,
    _program_ad_assembly_stack_axis,
    _program_ad_assembly_stack_convenience_selected_indices,
    _program_ad_assembly_stack_output_shape,
    program_ad_assembly_append_derivative_rule,
    program_ad_assembly_block_derivative_rule,
    program_ad_assembly_column_stack_derivative_rule,
    program_ad_assembly_concatenate_derivative_rule,
    program_ad_assembly_dstack_derivative_rule,
    program_ad_assembly_hstack_derivative_rule,
    program_ad_assembly_stack_derivative_rule,
    program_ad_assembly_vstack_derivative_rule,
)
from .program_ad_stencil_primitives import (
    _gradient_axis_coefficients,
    _GradientSpacing,
    _normalise_gradient_axes,
    _normalise_gradient_edge_order,
    _normalise_gradient_spacings,
    _program_ad_gradient_spacing_signature,
    program_ad_stencil_gradient_derivative_rule,
)
from .program_ad_trapezoid_primitives import (
    _program_ad_reduction_trapezoid_jvp,
    _program_ad_reduction_trapezoid_static_widths,
    _program_ad_reduction_trapezoid_value,
    _program_ad_reduction_trapezoid_vjp,
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
VMapInAxes = int | None | Sequence[int | None]
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


def _normalise_order_statistic_axis(axis: object, rank: int) -> int | None:
    if axis is None:
        return None
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD order-statistic axis must be a static integer or None")
    try:
        return _normalise_axis("axis", int(axis), rank)
    except ValueError as exc:
        if "out of bounds" in str(exc):
            raise ValueError("program AD order-statistic axis out of bounds") from exc
        raise


def _normalise_order_statistic_method(method: object) -> None:
    if not isinstance(method, str):
        raise ValueError("program AD order-statistic method must be static string")
    if method != "linear":
        raise ValueError("program AD order-statistic reductions only supports method='linear'")


def _normalise_order_statistic_q(q: object, *, percentile: bool) -> float:
    if isinstance(q, (TraceADArray, TraceADScalar)):
        raise ValueError("program AD order-statistic q must be static")
    raw = np.asarray(q)
    if raw.shape != ():
        raise ValueError("program AD order-statistic reductions require scalar q")
    if raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError("program AD order-statistic q must be static real numeric")
    q_value = _as_real_scalar("program AD order-statistic q", raw.item())
    if not math.isfinite(q_value):
        raise ValueError("program AD order-statistic q must be finite")
    if percentile:
        if q_value < 0.0 or q_value > 100.0:
            raise ValueError("program AD np.percentile q must be in [0, 100]")
        return q_value / 100.0
    if q_value < 0.0 or q_value > 1.0:
        raise ValueError("program AD np.quantile q must be in [0, 1]")
    return q_value


def _require_strict_order_statistic_values(values: NDArray[np.float64], op_name: str) -> None:
    if values.size == 0:
        raise ValueError(f"program AD {op_name} requires at least one element")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"program AD {op_name} requires finite values")
    if values.size <= 1:
        return
    sorted_values = np.sort(values.reshape(-1))
    if bool(np.any(np.diff(sorted_values) == 0.0)):
        raise ValueError(
            "program AD order-statistic reductions require strictly ordered values; "
            "equal values form a nondifferentiable selection boundary"
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


def _normalise_ddof(ddof: object, count: int) -> int:
    if not isinstance(ddof, (int, np.integer)):
        raise ValueError("program AD variance/std reductions require integer ddof")
    ddof_int = int(ddof)
    if ddof_int < 0:
        raise ValueError("program AD variance/std reductions require non-negative ddof")
    if count - ddof_int <= 0:
        raise ValueError("program AD variance/std ddof must leave a positive denominator")
    return ddof_int


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


def _parse_static_einsum_subscripts(
    subscripts: str,
    operand_shapes: Sequence[Sequence[int]],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], dict[str, int]]:
    normalised = subscripts.replace(" ", "")
    if "..." in normalised:
        raise ValueError("whole-program AD np.einsum ellipsis forms require explicit expansion")
    if "->" not in normalised:
        raise ValueError("whole-program AD np.einsum requires an explicit output subscript")
    if normalised.count("->") != 1:
        raise ValueError("whole-program AD np.einsum requires one explicit output separator")
    input_spec, output_spec = normalised.split("->", 1)
    input_labels = tuple(tuple(part) for part in input_spec.split(","))
    output_labels = tuple(output_spec)
    if len(input_labels) != len(operand_shapes):
        raise ValueError("whole-program AD np.einsum operand count must match subscripts")
    if len(set(output_labels)) != len(output_labels):
        raise ValueError("whole-program AD np.einsum output labels must be unique")
    if any(not label.isalpha() for labels in input_labels for label in labels) or any(
        not label.isalpha() for label in output_labels
    ):
        raise ValueError("whole-program AD np.einsum supports alphabetic labels only")

    dimensions: dict[str, int] = {}
    seen_labels: set[str] = set()
    for labels, raw_shape in zip(input_labels, operand_shapes, strict=True):
        shape = tuple(int(dimension) for dimension in raw_shape)
        if len(labels) != len(shape):
            raise ValueError("whole-program AD np.einsum labels must match operand rank")
        local_dimensions: dict[str, int] = {}
        for label, dimension in zip(labels, shape, strict=True):
            if dimension <= 0:
                raise ValueError("whole-program AD np.einsum operand dimensions must be positive")
            seen_labels.add(label)
            previous = dimensions.get(label)
            if previous is not None and previous != dimension:
                raise ValueError("whole-program AD np.einsum label dimensions must agree")
            local_previous = local_dimensions.get(label)
            if local_previous is not None and local_previous != dimension:
                raise ValueError("whole-program AD np.einsum repeated-label dimensions must agree")
            dimensions[label] = dimension
            local_dimensions[label] = dimension
    missing_output = set(output_labels) - seen_labels
    if missing_output:
        raise ValueError("whole-program AD np.einsum output labels must appear in operands")
    return output_labels, input_labels, dimensions


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


def _program_ad_linalg_eig_eigenvector_jvp_matrix(
    eigenvalues: NDArray[np.float64],
    right_eigenvectors: NDArray[np.float64],
    left_eigenvector_rows: NDArray[np.float64],
    tangent_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    size = eigenvalues.size
    tangent = np.zeros_like(right_eigenvectors, dtype=np.float64)
    for column in range(size):
        source: NDArray[np.float64] = np.asarray(right_eigenvectors[:, column], dtype=np.float64)
        raw_column: NDArray[np.float64] = np.zeros(size, dtype=np.float64)
        for other in range(size):
            if other == column:
                continue
            scale = float(left_eigenvector_rows[other, :] @ tangent_matrix @ source) / float(
                eigenvalues[column] - eigenvalues[other]
            )
            raw_column = raw_column + scale * np.asarray(
                right_eigenvectors[:, other], dtype=np.float64
            )
        tangent[:, column] = raw_column - source * float(source.T @ raw_column)
    return tangent


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


def program_adjoint_result(result: WholeProgramADResult) -> ProgramADAdjointResult:
    """Return the reverse-mode adjoint generation result attached to Program AD."""

    if not isinstance(result, WholeProgramADResult):
        raise ValueError("program adjoint input must be a WholeProgramADResult")
    if result.adjoint_result is None:
        raise ValueError("program AD result does not contain adjoint generation metadata")
    return result.adjoint_result


def program_adjoint_gradient(result: WholeProgramADResult) -> NDArray[np.float64]:
    """Return a supported reverse-mode adjoint gradient or fail closed."""

    adjoint = program_adjoint_result(result)
    if not adjoint.supported:
        unsupported = ", ".join(adjoint.unsupported_ops)
        raise ValueError(f"program AD adjoint generation unsupported for ops: {unsupported}")
    gradient: NDArray[np.float64] = adjoint.gradient.copy()
    return gradient


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


def _program_adjoint_det_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for a determinant primitive node."""

    parts = node.op.split(":")
    if len(parts) != 3:
        raise ValueError("det adjoint requires shape-qualified determinant metadata")
    try:
        rows, cols = (int(part) for part in parts[2].split("x", maxsplit=1))
    except ValueError as exc:
        raise ValueError("det adjoint metadata is malformed") from exc
    if rows != cols or rows < 0 or rows * cols != len(node.inputs):
        raise ValueError("det adjoint requires flattened square matrix inputs")
    if rows == 0:
        return ()
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(rows, cols)
    cofactors = _program_ad_linalg_det_cofactor_matrix(matrix)
    return tuple(
        (
            node.inputs[row * cols + col],
            float(cofactors[row, col]),
        )
        for row in range(rows)
        for col in range(cols)
    )


def _program_adjoint_inv_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one inverse-output primitive node."""

    parts = node.op.split(":")
    if len(parts) != 5:
        raise ValueError("inverse adjoint requires shape and output-index metadata")
    try:
        rows, cols = (int(part) for part in parts[2].split("x", maxsplit=1))
        output_row = int(parts[3])
        output_col = int(parts[4])
    except ValueError as exc:
        raise ValueError("inverse adjoint metadata is malformed") from exc
    if rows != cols or rows <= 0 or rows * cols != len(node.inputs):
        raise ValueError("inverse adjoint requires flattened square matrix inputs")
    if output_row < 0 or output_row >= rows or output_col < 0 or output_col >= cols:
        raise ValueError("inverse adjoint output index is outside inverse shape")
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(rows, cols)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("inverse adjoint requires a nonsingular matrix") from exc
    cotangent = np.zeros((rows, cols), dtype=np.float64)
    cotangent[output_row, output_col] = 1.0
    local_adjoint = -(inverse.T @ cotangent @ inverse.T)
    return tuple(
        (
            node.inputs[row * cols + col],
            float(local_adjoint[row, col]),
        )
        for row in range(rows)
        for col in range(cols)
    )


def _program_adjoint_solve_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one linear-solve output node."""

    parts = node.op.split(":")
    if len(parts) not in {6, 7} or parts[3] != "rhs":
        raise ValueError("solve adjoint requires shape, rhs, and output-index metadata")
    try:
        rows, cols = (int(part) for part in parts[2].split("x", maxsplit=1))
        rhs_shape = tuple(int(part) for part in parts[4].split("x"))
        output_row = int(parts[5])
        output_col = int(parts[6]) if len(parts) == 7 else -1
    except ValueError as exc:
        raise ValueError("solve adjoint metadata is malformed") from exc
    if rows != cols or rows <= 0:
        raise ValueError("solve adjoint requires a non-empty square matrix")
    if len(rhs_shape) == 1:
        if len(parts) != 6 or rhs_shape[0] != rows:
            raise ValueError("solve vector adjoint rhs shape is incompatible with matrix")
        if output_row < 0 or output_row >= rows:
            raise ValueError("solve adjoint output row is outside solution shape")
    elif len(rhs_shape) == 2:
        if len(parts) != 7 or rhs_shape[0] != rows or rhs_shape[1] <= 0:
            raise ValueError("solve matrix adjoint rhs shape is incompatible with matrix")
        if output_row < 0 or output_row >= rows or output_col < 0 or output_col >= rhs_shape[1]:
            raise ValueError("solve adjoint output index is outside solution shape")
    else:
        raise ValueError("solve adjoint rhs shape must be rank-1 or rank-2")
    rhs_size = int(np.prod(rhs_shape, dtype=np.int64))
    matrix_size = rows * cols
    if len(node.inputs) != matrix_size + rhs_size:
        raise ValueError("solve adjoint inputs must contain matrix followed by rhs")
    matrix_input_names = node.inputs[:matrix_size]
    rhs_input_names = node.inputs[matrix_size:]
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in matrix_input_names],
        dtype=np.float64,
    ).reshape(rows, cols)
    rhs = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in rhs_input_names],
        dtype=np.float64,
    ).reshape(rhs_shape)
    try:
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("solve adjoint requires a nonsingular matrix") from exc
    cotangent = np.zeros_like(solution, dtype=np.float64)
    if len(rhs_shape) == 1:
        cotangent[output_row] = 1.0
        rhs_adjoint = np.linalg.solve(matrix.T, cotangent)
        matrix_adjoint = -np.outer(rhs_adjoint, solution)
    else:
        cotangent[output_row, output_col] = 1.0
        rhs_adjoint = np.linalg.solve(matrix.T, cotangent)
        matrix_adjoint = -(rhs_adjoint @ solution.T)
    flat_rhs_adjoint = np.asarray(rhs_adjoint, dtype=np.float64).reshape(-1)
    return tuple(
        (
            matrix_input_names[row * cols + col],
            float(matrix_adjoint[row, col]),
        )
        for row in range(rows)
        for col in range(cols)
    ) + tuple(
        (
            rhs_input_names[index],
            float(flat_rhs_adjoint[index]),
        )
        for index in range(rhs_size)
    )


def _program_adjoint_trace_contributions(
    node: WholeProgramIRNode,
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for a trace primitive node."""

    parts = node.op.split(":")
    if len(parts) != 5 or parts[3] != "offset":
        raise ValueError("trace adjoint requires shape and offset metadata")
    try:
        shape = _program_adjoint_parse_shape_label(parts[2])
        offset = int(parts[4])
    except ValueError as exc:
        raise ValueError("trace adjoint metadata is malformed") from exc
    if len(shape) != 2:
        raise ValueError("trace adjoint requires rank-2 matrix metadata")
    rows, cols = shape
    diagonal_length = sum(1 for row in range(rows) if 0 <= row + offset < cols)
    if diagonal_length <= 0 or len(node.inputs) != diagonal_length:
        raise ValueError("trace adjoint inputs must match the selected diagonal")
    return tuple((name, 1.0) for name in node.inputs)


def _program_adjoint_diag_contributions(
    node: WholeProgramIRNode,
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one diag primitive output node."""

    parts = node.op.split(":")
    if len(parts) != 7 or parts[3] != "offset" or parts[5] not in {"construct", "extract"}:
        raise ValueError("diag adjoint requires shape, offset, mode, and index metadata")
    if len(node.inputs) != 1:
        raise ValueError("diag adjoint primitive outputs must have one source input")
    try:
        shape = _program_adjoint_parse_shape_label(parts[2])
        offset = int(parts[4])
        source_index = int(parts[6])
    except ValueError as exc:
        raise ValueError("diag adjoint metadata is malformed") from exc
    mode = parts[5]
    if mode == "construct":
        if len(shape) != 1 or source_index < 0 or source_index >= shape[0]:
            raise ValueError("diag construct adjoint source index is outside vector shape")
    else:
        if len(shape) != 2:
            raise ValueError("diag extract adjoint requires rank-2 source metadata")
        rows, cols = shape
        diagonal_length = sum(1 for row in range(rows) if 0 <= row + offset < cols)
        if source_index < 0 or source_index >= diagonal_length:
            raise ValueError("diag extract adjoint output index is outside diagonal shape")
    return ((node.inputs[0], 1.0),)


def _program_adjoint_diagflat_contributions(
    node: WholeProgramIRNode,
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one diagflat primitive output node."""

    parts = node.op.split(":")
    if len(parts) != 7 or parts[3] != "offset" or parts[5] != "construct":
        raise ValueError("diagflat adjoint requires shape, offset, construct, and index metadata")
    if len(node.inputs) != 1:
        raise ValueError("diagflat adjoint primitive outputs must have one source input")
    try:
        shape = _program_adjoint_parse_shape_label(parts[2])
        int(parts[4])
        source_index = int(parts[6])
    except ValueError as exc:
        raise ValueError("diagflat adjoint metadata is malformed") from exc
    source_size = int(np.prod(shape, dtype=np.int64))
    if source_index < 0 or source_index >= source_size:
        raise ValueError("diagflat adjoint source index is outside flattened source shape")
    return ((node.inputs[0], 1.0),)


def _program_adjoint_parse_shape_label(label: str) -> tuple[int, ...]:
    """Parse static primitive shape metadata from compact IR labels."""

    if not label:
        raise ValueError("shape label must not be empty")
    shape = tuple(int(part) for part in label.split("x"))
    if any(dimension < 0 for dimension in shape):
        raise ValueError("shape dimensions must be non-negative")
    return shape


def _program_adjoint_matrix_power_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one matrix-power output node."""

    parts = node.op.split(":")
    if len(parts) != 7 or parts[3] != "power":
        raise ValueError("matrix_power adjoint requires shape, power, and output-index metadata")
    try:
        shape = _program_adjoint_parse_shape_label(parts[2])
        exponent = int(parts[4])
        output_row = int(parts[5])
        output_col = int(parts[6])
    except ValueError as exc:
        raise ValueError("matrix_power adjoint metadata is malformed") from exc
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] <= 0:
        raise ValueError("matrix_power adjoint requires non-empty square matrix metadata")
    rows, cols = shape
    if len(node.inputs) != rows * cols:
        raise ValueError("matrix_power adjoint inputs must contain one flattened square matrix")
    if output_row < 0 or output_row >= rows or output_col < 0 or output_col >= cols:
        raise ValueError("matrix_power adjoint output index is outside matrix shape")
    flat_values = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    )
    cotangent = np.zeros((rows, cols), dtype=np.float64)
    cotangent[output_row, output_col] = 1.0
    rule = program_ad_linalg_matrix_power_derivative_rule(exponent)
    if rule.vjp_rule is None:
        raise ValueError("matrix_power adjoint requires a VJP rule")
    try:
        local_adjoint = np.asarray(
            rule.vjp_rule(flat_values, cotangent.reshape(-1)), dtype=np.float64
        ).reshape(-1)
    except np.linalg.LinAlgError as exc:
        raise ValueError("matrix_power adjoint requires a nonsingular matrix") from exc
    return tuple(
        (name, float(value)) for name, value in zip(node.inputs, local_adjoint, strict=True)
    )


def _program_adjoint_multi_dot_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one multi-dot output node."""

    parts = node.op.split(":")
    if len(parts) not in {5, 6} or parts[3] != "out":
        raise ValueError("multi_dot adjoint requires operand-shape and output metadata")
    try:
        operand_shapes = tuple(
            _program_adjoint_parse_shape_label(label) for label in parts[2].split("__")
        )
        output_shape, output_index = _program_adjoint_multi_dot_output_metadata(parts[4:])
    except ValueError as exc:
        raise ValueError("multi_dot adjoint metadata is malformed") from exc
    expected_inputs = sum(int(np.prod(shape, dtype=np.int64)) for shape in operand_shapes)
    if len(node.inputs) != expected_inputs:
        raise ValueError("multi_dot adjoint inputs must match flattened operand shapes")
    output_size = int(np.prod(output_shape, dtype=np.int64)) if output_shape else 1
    if output_index < 0 or output_index >= output_size:
        raise ValueError("multi_dot adjoint output index is outside result shape")
    flat_values = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    )
    cotangent = np.zeros(output_size, dtype=np.float64)
    cotangent[output_index] = 1.0
    rule = program_ad_linalg_multi_dot_derivative_rule(operand_shapes)
    if rule.vjp_rule is None:
        raise ValueError("multi_dot adjoint requires a VJP rule")
    local_adjoint = np.asarray(rule.vjp_rule(flat_values, cotangent), dtype=np.float64).reshape(-1)
    return tuple(
        (name, float(value)) for name, value in zip(node.inputs, local_adjoint, strict=True)
    )


def _program_adjoint_multi_dot_output_metadata(parts: list[str]) -> tuple[tuple[int, ...], int]:
    """Parse multi-dot output shape and flat index metadata."""

    if len(parts) == 1 and parts[0] == "scalar":
        return (), 0
    if len(parts) != 2:
        raise ValueError("multi_dot output metadata must be scalar or shape plus index")
    shape = _program_adjoint_parse_shape_label(parts[0])
    if not shape:
        raise ValueError("multi_dot non-scalar output shape must not be empty")
    return shape, int(parts[1])


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


def _program_adjoint_eigvalsh_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for a distinct symmetric eigvalsh node."""

    try:
        eigenvalue_index = int(node.op.rsplit(":", 1)[1])
    except ValueError as exc:
        raise ValueError("eigvalsh adjoint requires an eigenvalue index") from exc
    matrix_size = int(math.isqrt(len(node.inputs)))
    if matrix_size * matrix_size != len(node.inputs):
        raise ValueError("eigvalsh adjoint requires flattened square matrix inputs")
    if eigenvalue_index < 0 or eigenvalue_index >= matrix_size:
        raise ValueError("eigvalsh adjoint eigenvalue index is outside the spectrum")
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(matrix_size, matrix_size)
    _program_ad_linalg_require_symmetric("eigvalsh adjoint replay", matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, "eigvalsh adjoint replay")
    eigenvector = eigenvectors[:, eigenvalue_index]
    return tuple(
        (
            node.inputs[row * matrix_size + col],
            float(eigenvector[row] * eigenvector[col]),
        )
        for row in range(matrix_size)
        for col in range(matrix_size)
    )


def _program_adjoint_eigvals_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one real-simple eigenvalue."""

    parts = node.op.split(":")
    if len(parts) != 4:
        raise ValueError("eigvals adjoint requires shape-qualified spectral metadata")
    try:
        rows, cols = (int(part) for part in parts[2].split("x", maxsplit=1))
        eigenvalue_index = int(parts[3])
    except ValueError as exc:
        raise ValueError("eigvals adjoint metadata is malformed") from exc
    if rows != cols or rows <= 0 or rows * cols != len(node.inputs):
        raise ValueError("eigvals adjoint requires flattened square matrix inputs")
    if eigenvalue_index < 0 or eigenvalue_index >= rows:
        raise ValueError("eigvals adjoint eigenvalue index is outside the spectrum")
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(rows, cols)
    _eigenvalues, right_eigenvectors, left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition_from_matrix(
            "eigvals adjoint replay", matrix
        )
    )
    local_adjoint = np.outer(
        left_eigenvector_rows[eigenvalue_index, :], right_eigenvectors[:, eigenvalue_index]
    )
    return tuple(
        (
            node.inputs[row * rows + col],
            float(local_adjoint[row, col]),
        )
        for row in range(rows)
        for col in range(rows)
    )


def _program_adjoint_eig_eigenvalue_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one real-simple eig eigenvalue."""

    matrix, _eigenvalues, right_eigenvectors, left_eigenvector_rows, eigenvalue_index = cast(
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            int,
        ],
        _program_adjoint_eig_metadata(node, node_by_name, expect_eigenvector=False),
    )
    del matrix, _eigenvalues
    size = right_eigenvectors.shape[0]
    local_adjoint = np.outer(
        left_eigenvector_rows[eigenvalue_index, :], right_eigenvectors[:, eigenvalue_index]
    )
    return tuple(
        (
            node.inputs[row * size + col],
            float(local_adjoint[row, col]),
        )
        for row in range(size)
        for col in range(size)
    )


def _program_adjoint_eig_eigenvector_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one real-simple eig eigenvector element."""

    matrix, eigenvalues, right_eigenvectors, left_eigenvector_rows, eigenvalue_index, row_index = (
        cast(
            tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                int,
                int,
            ],
            _program_adjoint_eig_metadata(node, node_by_name, expect_eigenvector=True),
        )
    )
    del matrix
    size = right_eigenvectors.shape[0]
    local_adjoint = np.zeros((size, size), dtype=np.float64)
    for tangent_row in range(size):
        for tangent_col in range(size):
            tangent_matrix = np.zeros((size, size), dtype=np.float64)
            tangent_matrix[tangent_row, tangent_col] = 1.0
            eigenvector_tangent = _program_ad_linalg_eig_eigenvector_jvp_matrix(
                eigenvalues, right_eigenvectors, left_eigenvector_rows, tangent_matrix
            )
            local_adjoint[tangent_row, tangent_col] = eigenvector_tangent[
                row_index, eigenvalue_index
            ]
    return tuple(
        (
            node.inputs[row * size + col],
            float(local_adjoint[row, col]),
        )
        for row in range(size)
        for col in range(size)
    )


def _program_adjoint_eig_metadata(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
    *,
    expect_eigenvector: bool,
) -> (
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        int,
    ]
    | tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        int,
        int,
    ]
):
    parts = node.op.split(":")
    expected_length = 6 if expect_eigenvector else 5
    if len(parts) != expected_length:
        raise ValueError("eig adjoint requires shape-qualified spectral metadata")
    try:
        rows, cols = (int(part) for part in parts[3].split("x", maxsplit=1))
        eigenvalue_index = int(parts[4])
        row_index = int(parts[5]) if expect_eigenvector else -1
    except ValueError as exc:
        raise ValueError("eig adjoint metadata is malformed") from exc
    if rows != cols or rows <= 0 or rows * cols != len(node.inputs):
        raise ValueError("eig adjoint requires flattened square matrix inputs")
    if eigenvalue_index < 0 or eigenvalue_index >= rows:
        raise ValueError("eig adjoint eigenvalue index is outside the spectrum")
    if expect_eigenvector and (row_index < 0 or row_index >= rows):
        raise ValueError("eig adjoint eigenvector row index is outside the matrix")
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(rows, cols)
    eigenvalues, right_eigenvectors, left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition_from_matrix("eig adjoint replay", matrix)
    )
    if expect_eigenvector:
        return (
            matrix,
            eigenvalues,
            right_eigenvectors,
            left_eigenvector_rows,
            eigenvalue_index,
            row_index,
        )
    return matrix, eigenvalues, right_eigenvectors, left_eigenvector_rows, eigenvalue_index


def _program_adjoint_eigh_eigenvalue_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one symmetric eigh eigenvalue."""

    matrix, eigenvalues, eigenvectors, eigenvalue_index = cast(
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int],
        _program_adjoint_eigh_metadata(node, node_by_name, expect_eigenvector=False),
    )
    del matrix, eigenvalues
    eigenvector = eigenvectors[:, eigenvalue_index]
    size = eigenvectors.shape[0]
    return tuple(
        (
            node.inputs[row * size + col],
            float(eigenvector[row] * eigenvector[col]),
        )
        for row in range(size)
        for col in range(size)
    )


def _program_adjoint_eigh_eigenvector_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for one symmetric eigh eigenvector element."""

    matrix, eigenvalues, eigenvectors, eigenvalue_index, row_index = cast(
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int, int],
        _program_adjoint_eigh_metadata(node, node_by_name, expect_eigenvector=True),
    )
    del matrix
    size = eigenvectors.shape[0]
    cotangent = np.zeros_like(eigenvectors, dtype=np.float64)
    cotangent[row_index, eigenvalue_index] = 1.0
    local_adjoint = _program_ad_linalg_eigh_vjp_matrix(
        eigenvalues, eigenvectors, np.zeros(size, dtype=np.float64), cotangent
    )
    return tuple(
        (
            node.inputs[row * size + col],
            float(local_adjoint[row, col]),
        )
        for row in range(size)
        for col in range(size)
    )


def _program_adjoint_eigh_metadata(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
    *,
    expect_eigenvector: bool,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]
    | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int, int]
):
    parts = node.op.split(":")
    expected_length = 7 if expect_eigenvector else 6
    if len(parts) != expected_length:
        raise ValueError("eigh adjoint requires shape-qualified spectral metadata")
    try:
        rows, cols = (int(part) for part in parts[3].split("x", maxsplit=1))
        uplo = _program_ad_linalg_uplo(parts[4], "eigh adjoint replay")
        eigenvalue_index = int(parts[5])
        row_index = int(parts[6]) if expect_eigenvector else -1
    except ValueError as exc:
        raise ValueError("eigh adjoint metadata is malformed") from exc
    if rows != cols or rows <= 0 or rows * cols != len(node.inputs):
        raise ValueError("eigh adjoint requires flattened square matrix inputs")
    if eigenvalue_index < 0 or eigenvalue_index >= rows:
        raise ValueError("eigh adjoint eigenvalue index is outside the spectrum")
    if expect_eigenvector and (row_index < 0 or row_index >= rows):
        raise ValueError("eigh adjoint eigenvector row index is outside the matrix")
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(rows, cols)
    _program_ad_linalg_require_symmetric("eigh adjoint replay", matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix, UPLO=uplo)
    _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, "eigh adjoint replay")
    if expect_eigenvector:
        return matrix, eigenvalues, eigenvectors, eigenvalue_index, row_index
    return matrix, eigenvalues, eigenvectors, eigenvalue_index


def _program_adjoint_svdvals_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for a distinct positive SVD singular value."""

    parts = node.op.split(":")
    if len(parts) != 4:
        raise ValueError("svd adjoint requires shape-qualified singular-value metadata")
    try:
        rows, cols = (int(part) for part in parts[2].split("x", maxsplit=1))
        singular_value_index = int(parts[3])
    except ValueError as exc:
        raise ValueError("svd adjoint metadata is malformed") from exc
    if rows <= 0 or cols <= 0 or rows * cols != len(node.inputs):
        raise ValueError("svd adjoint requires flattened matrix inputs matching metadata")
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(rows, cols)
    left, singular_values, right_h = np.linalg.svd(matrix, full_matrices=False)
    _program_ad_linalg_require_distinct_positive_singular_values(
        singular_values, "svd adjoint replay"
    )
    if singular_value_index < 0 or singular_value_index >= singular_values.size:
        raise ValueError("svd adjoint singular-value index is outside the spectrum")
    return tuple(
        (
            node.inputs[row * cols + col],
            float(left[row, singular_value_index] * right_h[singular_value_index, col]),
        )
        for row in range(rows)
        for col in range(cols)
    )


def _program_adjoint_pinv_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    """Return local reverse contributions for a constant-full-rank pinv element."""

    parts = node.op.split(":")
    if len(parts) != 6:
        raise ValueError("pinv adjoint requires shape, cutoff, and output-index metadata")
    try:
        rows, cols = (int(part) for part in parts[2].split("x", maxsplit=1))
        rcond = float(parts[3])
        output_row = int(parts[4])
        output_col = int(parts[5])
    except ValueError as exc:
        raise ValueError("pinv adjoint metadata is malformed") from exc
    if rows <= 0 or cols <= 0 or rows * cols != len(node.inputs):
        raise ValueError("pinv adjoint requires flattened matrix inputs matching metadata")
    if output_row < 0 or output_row >= cols or output_col < 0 or output_col >= rows:
        raise ValueError("pinv adjoint output index is outside pseudoinverse shape")
    matrix = np.array(
        [_program_adjoint_input_value(name, node_by_name) for name in node.inputs],
        dtype=np.float64,
    ).reshape(rows, cols)
    pinv = _program_ad_linalg_pinv_value_matrix(matrix, rcond=rcond)
    cotangent = np.zeros_like(pinv, dtype=np.float64)
    cotangent[output_row, output_col] = 1.0
    local_adjoint = _program_ad_linalg_pinv_vjp_matrix(matrix, pinv, cotangent)
    return tuple(
        (
            node.inputs[row * cols + col],
            float(local_adjoint[row, col]),
        )
        for row in range(rows)
        for col in range(cols)
    )


def vmap(
    function: Callable[..., object],
    in_axes: VMapInAxes = 0,
    out_axes: int = 0,
    *,
    primitive_identity: PrimitiveIdentity | str | None = None,
    registry: CustomDerivativeRegistry | None = None,
) -> Callable[..., object]:
    """Return a composable vectorizing transform over leading or selected axes.

    The transform mirrors the practical contract of a JAX-style ``vmap`` for the
    native NumPy differentiable layer: mapped arguments are sliced along their
    declared axes, ``None`` axes are broadcast unchanged, and stackable scalar,
    array, tuple, list, or dict outputs are reassembled with the mapped axis at
    ``out_axes``. It is an eager deterministic transform, not a JIT compiler.
    """

    if not callable(function):
        raise ValueError("vmap function must be callable")
    if not isinstance(out_axes, int):
        raise ValueError("out_axes must be an integer")
    batching_rule: PrimitiveBatchingRule | None = None
    if primitive_identity is not None:
        target_registry = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
        batching_rule = target_registry.require_batching_rule(primitive_identity)

    def vectorized(*args: object) -> object:
        if not args:
            raise ValueError("vmap requires at least one argument")
        axes = _normalise_vmap_in_axes(in_axes, len(args))
        mapped: list[tuple[NDArray[np.float64] | TraceADArray, int] | None] = []
        batch_size: int | None = None
        for index, (arg, axis) in enumerate(zip(args, axes, strict=True)):
            if axis is None:
                mapped.append(None)
                continue
            array = (
                arg
                if isinstance(arg, TraceADArray)
                else _as_real_numeric_array(f"vmap argument {index}", arg)
            )
            axis_index = _normalise_axis(f"in_axes[{index}]", axis, array.ndim)
            size = int(array.shape[axis_index])
            if size <= 0:
                raise ValueError("mapped axes must be non-empty")
            if batch_size is None:
                batch_size = size
            elif size != batch_size:
                raise ValueError("all mapped axes must have the same length")
            mapped.append((array, axis_index))
        if batch_size is None:
            raise ValueError("at least one in_axes entry must be mapped")
        if batching_rule is not None:
            return batching_rule(function, args, axes, out_axes)

        outputs = []
        for item in range(batch_size):
            call_args = []
            for arg, mapping in zip(args, mapped, strict=True):
                if mapping is None:
                    call_args.append(arg)
                else:
                    array, axis_index = mapping
                    if isinstance(array, TraceADArray):
                        call_args.append(_trace_take(array, item, axis=axis_index, mode="raise"))
                    else:
                        call_args.append(np.take(array, item, axis=axis_index))
            outputs.append(function(*call_args))
        return _stack_vmap_outputs(outputs, out_axes)

    return vectorized


def _normalise_vmap_in_axes(in_axes: VMapInAxes, arity: int) -> tuple[int | None, ...]:
    """Return one input-axis declaration per positional argument."""

    if isinstance(in_axes, int) or in_axes is None:
        return tuple(in_axes for _ in range(arity))
    axes = tuple(in_axes)
    if len(axes) != arity:
        raise ValueError("in_axes length must match positional argument count")
    if any(axis is not None and not isinstance(axis, int) for axis in axes):
        raise ValueError("in_axes entries must be integers or None")
    return axes


def _normalise_axis(name: str, axis: int, ndim: int) -> int:
    """Return a non-negative axis for an array with ``ndim`` dimensions."""

    if ndim == 0:
        raise ValueError(f"{name} cannot map over a scalar")
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"{name} is out of bounds for argument rank {ndim}")
    return axis


def _stack_vmap_outputs(outputs: Sequence[object], out_axes: int) -> object:
    """Stack per-example outputs while preserving simple pytree structure."""

    if not outputs:
        raise ValueError("vmap outputs must be non-empty")
    first = outputs[0]
    if isinstance(first, (TraceADScalar, TraceADArray)):
        context = first.context
        trace_arrays = [_coerce_trace_array(output, context) for output in outputs]
        shape = trace_arrays[0].shape
        if any(array.shape != shape for array in trace_arrays):
            raise ValueError("vmap output leaves must have consistent shapes")
        return _trace_stack(tuple(trace_arrays), context, axis=out_axes)
    if isinstance(first, np.ndarray) or np.isscalar(first):
        numeric_arrays = [np.asarray(output) for output in outputs]
        shape = numeric_arrays[0].shape
        if any(array.shape != shape for array in numeric_arrays):
            raise ValueError("vmap output leaves must have consistent shapes")
        axis = out_axes
        result_rank = numeric_arrays[0].ndim + 1
        if axis < 0:
            axis += result_rank
        if axis < 0 or axis >= result_rank:
            raise ValueError("out_axes is out of bounds for stacked output rank")
        stacked: NDArray[Any] = np.stack(numeric_arrays, axis=axis)
        if stacked.dtype.kind in {"b", "O", "S", "U"}:
            raise ValueError("vmap output leaves must be numeric")
        return stacked
    if isinstance(first, tuple):
        if any(not isinstance(output, tuple) or len(output) != len(first) for output in outputs):
            raise ValueError("vmap tuple outputs must have consistent structure")
        return tuple(
            _stack_vmap_outputs(
                [cast(tuple[object, ...], output)[index] for output in outputs], out_axes
            )
            for index in range(len(first))
        )
    if isinstance(first, list):
        if any(not isinstance(output, list) or len(output) != len(first) for output in outputs):
            raise ValueError("vmap list outputs must have consistent structure")
        return [
            _stack_vmap_outputs(
                [cast(list[object], output)[index] for output in outputs], out_axes
            )
            for index in range(len(first))
        ]
    if isinstance(first, dict):
        keys = tuple(first.keys())
        if any(not isinstance(output, dict) or tuple(output.keys()) != keys for output in outputs):
            raise ValueError("vmap dict outputs must have consistent keys")
        return {
            key: _stack_vmap_outputs(
                [cast(dict[object, object], output)[key] for output in outputs], out_axes
            )
            for key in keys
        }
    raise ValueError("vmap output leaves must be numeric arrays, scalars, tuples, lists, or dicts")


def _program_ad_reduction_vector(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _as_real_numeric_array(f"program AD reduction {name} values", values).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"program AD reduction {name} direct rule requires at least one value")
    return vector


def _program_ad_reduction_tangent_pair(
    name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _program_ad_reduction_vector(name, values)
    tangent_vector = _as_real_numeric_array(
        f"program AD reduction {name} tangent", tangent
    ).reshape(-1)
    if tangent_vector.shape != vector.shape:
        raise ValueError(f"program AD reduction {name} tangent shape must match values shape")
    return vector, tangent_vector


def _program_ad_reduction_sum_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("sum", values)
    return np.array([float(np.sum(vector))], dtype=np.float64)


def _program_ad_reduction_sum_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    _vector, tangent_vector = _program_ad_reduction_tangent_pair("sum", values, tangent)
    return np.array([float(np.sum(tangent_vector))], dtype=np.float64)


def _program_ad_reduction_scalar_cotangent(
    name: str,
    cotangent: NDArray[np.float64],
) -> float:
    cotangent_vector = _as_real_numeric_array(
        f"program AD reduction {name} cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.shape != (1,):
        raise ValueError(f"program AD reduction {name} VJP requires one scalar cotangent")
    return float(cotangent_vector[0])


def _program_ad_reduction_sum_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("sum", values)
    scalar_cotangent = _program_ad_reduction_scalar_cotangent("sum", cotangent)
    return np.full(vector.shape, scalar_cotangent, dtype=np.float64)


def _program_ad_reduction_prod_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("prod", values)
    return np.array([float(np.prod(vector))], dtype=np.float64)


def _program_ad_reduction_prod_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector, tangent_vector = _program_ad_reduction_tangent_pair("prod", values, tangent)
    total = 0.0
    for tangent_index in range(vector.size):
        product = 1.0
        for factor_index in range(vector.size):
            product *= (
                tangent_vector[factor_index]
                if factor_index == tangent_index
                else vector[factor_index]
            )
        total += product
    return np.array([float(total)], dtype=np.float64)


def _program_ad_reduction_prod_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("prod", values)
    scalar_cotangent = _program_ad_reduction_scalar_cotangent("prod", cotangent)
    result = np.empty_like(vector, dtype=np.float64)
    for tangent_index in range(vector.size):
        product = 1.0
        for factor_index in range(vector.size):
            if factor_index != tangent_index:
                product *= vector[factor_index]
        result[tangent_index] = scalar_cotangent * product
    return result


def _program_ad_reduction_mean_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("mean", values)
    return np.array([float(np.mean(vector))], dtype=np.float64)


def _program_ad_reduction_mean_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    _vector, tangent_vector = _program_ad_reduction_tangent_pair("mean", values, tangent)
    return np.array([float(np.mean(tangent_vector))], dtype=np.float64)


def _program_ad_reduction_mean_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("mean", values)
    scalar_cotangent = _program_ad_reduction_scalar_cotangent("mean", cotangent)
    return np.full(vector.shape, scalar_cotangent / float(vector.size), dtype=np.float64)


def _program_ad_reduction_variance_gradient(
    name: str,
    values: NDArray[np.float64],
    *,
    ddof: int,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector(name, values)
    ddof_int = _normalise_ddof(ddof, vector.size)
    mean = float(np.mean(vector))
    return (2.0 / float(vector.size - ddof_int)) * (vector - mean)


def _program_ad_reduction_var_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("var", values)
    _normalise_ddof(0, vector.size)
    return np.array([float(np.var(vector))], dtype=np.float64)


def _program_ad_reduction_var_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector, tangent_vector = _program_ad_reduction_tangent_pair("var", values, tangent)
    gradient = _program_ad_reduction_variance_gradient("var", vector, ddof=0)
    return np.array([float(np.dot(gradient, tangent_vector))], dtype=np.float64)


def _program_ad_reduction_var_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("var", values)
    scalar_cotangent = _program_ad_reduction_scalar_cotangent("var", cotangent)
    return scalar_cotangent * _program_ad_reduction_variance_gradient("var", vector, ddof=0)


def _program_ad_reduction_std_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("std", values)
    _normalise_ddof(0, vector.size)
    return np.array([float(np.std(vector))], dtype=np.float64)


def _program_ad_reduction_std_gradient(
    values: NDArray[np.float64],
    *,
    ddof: int,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("std", values)
    standard_deviation = float(np.std(vector, ddof=ddof))
    if standard_deviation == 0.0:
        raise ValueError("program AD reduction std direct rule is undefined at zero variance")
    return _program_ad_reduction_variance_gradient("std", vector, ddof=ddof) / (
        2.0 * standard_deviation
    )


def _program_ad_reduction_std_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector, tangent_vector = _program_ad_reduction_tangent_pair("std", values, tangent)
    gradient = _program_ad_reduction_std_gradient(vector, ddof=0)
    return np.array([float(np.dot(gradient, tangent_vector))], dtype=np.float64)


def _program_ad_reduction_std_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("std", values)
    scalar_cotangent = _program_ad_reduction_scalar_cotangent("std", cotangent)
    return scalar_cotangent * _program_ad_reduction_std_gradient(vector, ddof=0)


def _program_ad_reduction_order_statistic_value(
    name: str,
    values: NDArray[np.float64],
    *,
    q: float,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector(name, values)
    _require_strict_order_statistic_values(vector, f"np.{name}")
    order = np.argsort(vector, kind="stable")
    position = q * float(vector.size - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    upper_weight = position - float(lower)
    value = vector[int(order[lower])]
    if lower != upper:
        value = value * (1.0 - upper_weight) + vector[int(order[upper])] * upper_weight
    return np.array([float(value)], dtype=np.float64)


def _program_ad_reduction_order_statistic_jvp(
    name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    q: float,
) -> NDArray[np.float64]:
    vector, tangent_vector = _program_ad_reduction_tangent_pair(name, values, tangent)
    _require_strict_order_statistic_values(vector, f"np.{name}")
    order = np.argsort(vector, kind="stable")
    position = q * float(vector.size - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    upper_weight = position - float(lower)
    value = tangent_vector[int(order[lower])]
    if lower != upper:
        value = value * (1.0 - upper_weight) + tangent_vector[int(order[upper])] * upper_weight
    return np.array([float(value)], dtype=np.float64)


def _program_ad_reduction_order_statistic_vjp(
    name: str,
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    q: float,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector(name, values)
    scalar_cotangent = _program_ad_reduction_scalar_cotangent(name, cotangent)
    _require_strict_order_statistic_values(vector, f"np.{name}")
    order = np.argsort(vector, kind="stable")
    position = q * float(vector.size - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    upper_weight = position - float(lower)
    result = np.zeros_like(vector, dtype=np.float64)
    result[int(order[lower])] += scalar_cotangent * (1.0 - upper_weight)
    if lower != upper:
        result[int(order[upper])] += scalar_cotangent * upper_weight
    return result


def _program_ad_reduction_derivative_rule(name: str) -> CustomDerivativeRule:
    if name == "sum":
        return CustomDerivativeRule(
            name="program_ad_reduction_sum_direct_rule",
            value_fn=_program_ad_reduction_sum_value,
            jvp_rule=_program_ad_reduction_sum_jvp,
            vjp_rule=_program_ad_reduction_sum_vjp,
        )
    if name == "prod":
        return CustomDerivativeRule(
            name="program_ad_reduction_prod_direct_rule",
            value_fn=_program_ad_reduction_prod_value,
            jvp_rule=_program_ad_reduction_prod_jvp,
            vjp_rule=_program_ad_reduction_prod_vjp,
        )
    if name == "mean":
        return CustomDerivativeRule(
            name="program_ad_reduction_mean_direct_rule",
            value_fn=_program_ad_reduction_mean_value,
            jvp_rule=_program_ad_reduction_mean_jvp,
            vjp_rule=_program_ad_reduction_mean_vjp,
        )
    if name == "var":
        return CustomDerivativeRule(
            name="program_ad_reduction_var_direct_rule",
            value_fn=_program_ad_reduction_var_value,
            jvp_rule=_program_ad_reduction_var_jvp,
            vjp_rule=_program_ad_reduction_var_vjp,
        )
    if name == "std":
        return CustomDerivativeRule(
            name="program_ad_reduction_std_direct_rule",
            value_fn=_program_ad_reduction_std_value,
            jvp_rule=_program_ad_reduction_std_jvp,
            vjp_rule=_program_ad_reduction_std_vjp,
        )
    if name == "max":
        return _program_ad_reduction_order_statistic_rule(name, q=1.0)
    if name == "min":
        return _program_ad_reduction_order_statistic_rule(name, q=0.0)
    if name == "median":
        return _program_ad_reduction_order_statistic_rule(name, q=0.5)
    if name == "quantile":
        return _program_ad_reduction_order_statistic_rule(name, q=0.5)
    if name == "percentile":
        return _program_ad_reduction_order_statistic_rule(name, q=0.5)
    if name == "trapezoid":
        return CustomDerivativeRule(
            name="program_ad_reduction_trapezoid_direct_rule",
            value_fn=_program_ad_reduction_trapezoid_value,
            jvp_rule=_program_ad_reduction_trapezoid_jvp,
            vjp_rule=_program_ad_reduction_trapezoid_vjp,
        )
    raise ValueError(f"unsupported program AD reduction primitive {name}")


def _program_ad_reduction_normalise_static_shape(
    name: str,
    source_shape: Sequence[int],
) -> tuple[int, ...]:
    shape = tuple(int(dimension) for dimension in source_shape)
    if any(dimension < 0 for dimension in shape):
        raise ValueError(
            f"program AD reduction {name} direct rule requires non-negative dimensions"
        )
    if _program_ad_shape_static_size(shape) == 0:
        raise ValueError(f"program AD reduction {name} direct rule requires at least one value")
    return shape


def _program_ad_reduction_axis_signature(axis: int | None) -> str:
    return "flat" if axis is None else str(axis)


def _program_ad_reduction_output_shape(
    source_shape: tuple[int, ...],
    axis: int | None,
) -> tuple[int, ...]:
    if axis is None:
        return ()
    normalised_axis = _normalise_axis("axis", axis, len(source_shape))
    return source_shape[:normalised_axis] + source_shape[normalised_axis + 1 :]


def _program_ad_reduction_q_signature(q: float) -> str:
    return str(float(q)).replace("-", "neg_").replace(".", "_")


def _program_ad_reduction_order_statistic_static_value(
    name: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    q: float,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_source_vector(name, "values", values, source_shape=source_shape)
    value_array = vector.reshape(source_shape)
    if axis is None:
        return _program_ad_reduction_order_statistic_value(name, vector, q=q)
    output = np.empty(_program_ad_reduction_output_shape(source_shape, axis), dtype=np.float64)
    for reduced_index in np.ndindex(output.shape):
        source_values = value_array[
            reduced_index[:axis] + (slice(None),) + reduced_index[axis:]
        ].reshape(-1)
        output[reduced_index] = _program_ad_reduction_order_statistic_value(
            name, source_values, q=q
        )[0]
    return _program_ad_float64_vector_result(output)


def _program_ad_reduction_var_static_value(
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    ddof: int,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_source_vector(
        "var", "values", values, source_shape=source_shape
    )
    count = vector.size if axis is None else source_shape[axis]
    _normalise_ddof(ddof, count)
    return _program_ad_float64_vector_result(
        np.var(vector.reshape(source_shape), axis=axis, ddof=ddof)
    )


def _program_ad_reduction_var_static_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    ddof: int,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        "var", "values", values, source_shape=source_shape
    ).reshape(source_shape)
    tangent_array = _program_ad_reduction_source_vector(
        "var", "tangent", tangent, source_shape=source_shape
    ).reshape(source_shape)
    count = value_array.size if axis is None else source_shape[axis]
    ddof_int = _normalise_ddof(ddof, count)
    mean = np.mean(value_array, axis=axis, keepdims=True)
    gradient = (2.0 / float(count - ddof_int)) * (value_array - mean)
    return _program_ad_float64_vector_result(np.sum(gradient * tangent_array, axis=axis))


def _program_ad_reduction_var_static_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    ddof: int,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        "var", "values", values, source_shape=source_shape
    ).reshape(source_shape)
    output_shape = _program_ad_reduction_output_shape(source_shape, axis)
    cotangent_array = _program_ad_reduction_cotangent_array(
        "var", cotangent, output_shape=output_shape
    )
    count = value_array.size if axis is None else source_shape[axis]
    ddof_int = _normalise_ddof(ddof, count)
    mean = np.mean(value_array, axis=axis, keepdims=True)
    gradient = (2.0 / float(count - ddof_int)) * (value_array - mean)
    if axis is None:
        return _program_ad_float64_vector_result(gradient * float(cotangent_array))
    expanded_cotangent = np.expand_dims(cotangent_array, axis=axis)
    return _program_ad_float64_vector_result(gradient * expanded_cotangent)


def _program_ad_reduction_std_static_value(
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    ddof: int,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_source_vector(
        "std", "values", values, source_shape=source_shape
    )
    count = vector.size if axis is None else source_shape[axis]
    _normalise_ddof(ddof, count)
    result = np.std(vector.reshape(source_shape), axis=axis, ddof=ddof)
    if bool(np.any(np.asarray(result) == 0.0)):
        raise ValueError("program AD reduction std direct rule is undefined at zero variance")
    return _program_ad_float64_vector_result(result)


def _program_ad_reduction_std_static_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    ddof: int,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        "std", "values", values, source_shape=source_shape
    ).reshape(source_shape)
    tangent_array = _program_ad_reduction_source_vector(
        "std", "tangent", tangent, source_shape=source_shape
    ).reshape(source_shape)
    count = value_array.size if axis is None else source_shape[axis]
    ddof_int = _normalise_ddof(ddof, count)
    standard_deviation = np.std(value_array, axis=axis, ddof=ddof_int, keepdims=True)
    if bool(np.any(standard_deviation == 0.0)):
        raise ValueError("program AD reduction std direct rule is undefined at zero variance")
    mean = np.mean(value_array, axis=axis, keepdims=True)
    gradient = (value_array - mean) / (float(count - ddof_int) * standard_deviation)
    return _program_ad_float64_vector_result(np.sum(gradient * tangent_array, axis=axis))


def _program_ad_reduction_std_static_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    ddof: int,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        "std", "values", values, source_shape=source_shape
    ).reshape(source_shape)
    output_shape = _program_ad_reduction_output_shape(source_shape, axis)
    cotangent_array = _program_ad_reduction_cotangent_array(
        "std", cotangent, output_shape=output_shape
    )
    count = value_array.size if axis is None else source_shape[axis]
    ddof_int = _normalise_ddof(ddof, count)
    standard_deviation = np.std(value_array, axis=axis, ddof=ddof_int, keepdims=True)
    if bool(np.any(standard_deviation == 0.0)):
        raise ValueError("program AD reduction std direct rule is undefined at zero variance")
    mean = np.mean(value_array, axis=axis, keepdims=True)
    gradient = (value_array - mean) / (float(count - ddof_int) * standard_deviation)
    if axis is None:
        return _program_ad_float64_vector_result(gradient * float(cotangent_array))
    expanded_cotangent = np.expand_dims(cotangent_array, axis=axis)
    return _program_ad_float64_vector_result(gradient * expanded_cotangent)


def _program_ad_reduction_order_statistic_static_jvp(
    name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    q: float,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        name, "values", values, source_shape=source_shape
    ).reshape(source_shape)
    tangent_array = _program_ad_reduction_source_vector(
        name, "tangent", tangent, source_shape=source_shape
    ).reshape(source_shape)
    if axis is None:
        return _program_ad_reduction_order_statistic_jvp(
            name, value_array.reshape(-1), tangent_array.reshape(-1), q=q
        )
    output = np.empty(_program_ad_reduction_output_shape(source_shape, axis), dtype=np.float64)
    for reduced_index in np.ndindex(output.shape):
        selector = reduced_index[:axis] + (slice(None),) + reduced_index[axis:]
        output[reduced_index] = _program_ad_reduction_order_statistic_jvp(
            name,
            value_array[selector].reshape(-1),
            tangent_array[selector].reshape(-1),
            q=q,
        )[0]
    return _program_ad_float64_vector_result(output)


def _program_ad_reduction_order_statistic_static_vjp(
    name: str,
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    q: float,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        name, "values", values, source_shape=source_shape
    ).reshape(source_shape)
    output_shape = _program_ad_reduction_output_shape(source_shape, axis)
    cotangent_array = _program_ad_reduction_cotangent_array(
        name, cotangent, output_shape=output_shape
    )
    if axis is None:
        return _program_ad_reduction_order_statistic_vjp(
            name, value_array.reshape(-1), cotangent_array.reshape(-1), q=q
        )
    result = np.zeros_like(value_array, dtype=np.float64)
    for reduced_index in np.ndindex(output_shape):
        selector = reduced_index[:axis] + (slice(None),) + reduced_index[axis:]
        result[selector] += _program_ad_reduction_order_statistic_vjp(
            name,
            value_array[selector].reshape(-1),
            np.array([float(cotangent_array[reduced_index])], dtype=np.float64),
            q=q,
        ).reshape(result[selector].shape)
    return _program_ad_float64_vector_result(result)


def _program_ad_reduction_order_statistic_rule(
    name: str,
    *,
    q: float,
    source_shape: Sequence[int] | None = None,
    axis: int | None = None,
) -> CustomDerivativeRule:
    if source_shape is None:

        def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return _program_ad_reduction_order_statistic_value(name, values, q=q)

        def jvp_rule(
            values: NDArray[np.float64],
            tangent: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return _program_ad_reduction_order_statistic_jvp(name, values, tangent, q=q)

        def vjp_rule(
            values: NDArray[np.float64],
            cotangent: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return _program_ad_reduction_order_statistic_vjp(name, values, cotangent, q=q)

        return CustomDerivativeRule(
            name=f"program_ad_reduction_{name}_q_{_program_ad_reduction_q_signature(q)}_direct_rule",
            value_fn=value_fn,
            jvp_rule=jvp_rule,
            vjp_rule=vjp_rule,
        )

    source = _program_ad_reduction_normalise_static_shape(name, source_shape)
    normalised_axis = None if axis is None else _normalise_axis("axis", axis, len(source))

    def static_value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_reduction_order_statistic_static_value(
            name, values, source_shape=source, axis=normalised_axis, q=q
        )

    def static_jvp_rule(
        values: NDArray[np.float64],
        tangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return _program_ad_reduction_order_statistic_static_jvp(
            name, values, tangent, source_shape=source, axis=normalised_axis, q=q
        )

    def static_vjp_rule(
        values: NDArray[np.float64],
        cotangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return _program_ad_reduction_order_statistic_static_vjp(
            name, values, cotangent, source_shape=source, axis=normalised_axis, q=q
        )

    return CustomDerivativeRule(
        name=(
            f"program_ad_reduction_{name}_{_program_ad_shape_signature(source)}_axis_"
            f"{_program_ad_reduction_axis_signature(normalised_axis)}_q_"
            f"{_program_ad_reduction_q_signature(q)}_direct_rule"
        ),
        value_fn=static_value_fn,
        jvp_rule=static_jvp_rule,
        vjp_rule=static_vjp_rule,
    )


def _program_ad_reduction_source_vector(
    name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    return _program_ad_shape_vector(
        f"reduction {name}",
        role,
        values,
        expected_size=_program_ad_shape_static_size(source_shape),
    )


def _program_ad_reduction_cotangent_array(
    name: str,
    cotangent: NDArray[np.float64],
    *,
    output_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    cotangent_vector = _as_real_numeric_array(
        f"program AD reduction {name} cotangent", cotangent
    ).reshape(-1)
    expected_size = _program_ad_shape_static_size(output_shape)
    if cotangent_vector.size != expected_size:
        raise ValueError(
            f"program AD reduction {name} VJP requires cotangent with {expected_size} values"
        )
    return cotangent_vector.reshape(output_shape)


def _program_ad_reduction_sum_static_value(
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_source_vector(
        "sum", "values", values, source_shape=source_shape
    )
    return _program_ad_float64_vector_result(np.sum(vector.reshape(source_shape), axis=axis))


def _program_ad_reduction_sum_static_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    _program_ad_reduction_source_vector("sum", "values", values, source_shape=source_shape)
    tangent_vector = _program_ad_reduction_source_vector(
        "sum", "tangent", tangent, source_shape=source_shape
    )
    return _program_ad_float64_vector_result(
        np.sum(tangent_vector.reshape(source_shape), axis=axis)
    )


def _program_ad_reduction_sum_static_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    _program_ad_reduction_source_vector("sum", "values", values, source_shape=source_shape)
    output_shape = _program_ad_reduction_output_shape(source_shape, axis)
    cotangent_array = _program_ad_reduction_cotangent_array(
        "sum", cotangent, output_shape=output_shape
    )
    if axis is None:
        return np.full(_program_ad_shape_static_size(source_shape), float(cotangent_array))
    expanded = np.expand_dims(cotangent_array, axis=axis)
    return _program_ad_float64_vector_result(np.broadcast_to(expanded, source_shape))


def _program_ad_reduction_mean_static_value(
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_source_vector(
        "mean", "values", values, source_shape=source_shape
    )
    return _program_ad_float64_vector_result(np.mean(vector.reshape(source_shape), axis=axis))


def _program_ad_reduction_mean_static_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    _program_ad_reduction_source_vector("mean", "values", values, source_shape=source_shape)
    tangent_vector = _program_ad_reduction_source_vector(
        "mean", "tangent", tangent, source_shape=source_shape
    )
    return _program_ad_float64_vector_result(
        np.mean(tangent_vector.reshape(source_shape), axis=axis)
    )


def _program_ad_reduction_mean_static_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    scale = float(
        _program_ad_shape_static_size(source_shape) if axis is None else source_shape[axis]
    )
    return (
        _program_ad_reduction_sum_static_vjp(
            values, cotangent, source_shape=source_shape, axis=axis
        )
        / scale
    )


def _program_ad_reduction_prod_static_value(
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_source_vector(
        "prod", "values", values, source_shape=source_shape
    )
    return _program_ad_float64_vector_result(np.prod(vector.reshape(source_shape), axis=axis))


def _program_ad_reduction_prod_static_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        "prod", "values", values, source_shape=source_shape
    ).reshape(source_shape)
    tangent_array = _program_ad_reduction_source_vector(
        "prod", "tangent", tangent, source_shape=source_shape
    ).reshape(source_shape)
    derivative = np.zeros_like(value_array, dtype=np.float64)
    for flat_index in range(value_array.size):
        multi_index = np.unravel_index(flat_index, source_shape)
        basis = np.zeros_like(value_array, dtype=np.float64)
        basis[multi_index] = tangent_array[multi_index]
        derivative[multi_index] = np.sum(
            np.prod(np.where(basis != 0.0, basis, value_array), axis=axis)
        )
    if axis is None:
        total = 0.0
        flat_values = value_array.reshape(-1)
        flat_tangent = tangent_array.reshape(-1)
        for tangent_index in range(flat_values.size):
            product = 1.0
            for factor_index in range(flat_values.size):
                product *= (
                    flat_tangent[factor_index]
                    if factor_index == tangent_index
                    else flat_values[factor_index]
                )
            total += product
        return np.array([float(total)], dtype=np.float64)
    output = np.zeros(_program_ad_reduction_output_shape(source_shape, axis), dtype=np.float64)
    for flat_index in range(value_array.size):
        multi_index = np.unravel_index(flat_index, source_shape)
        output_index = multi_index[:axis] + multi_index[axis + 1 :]
        product = 1.0
        for factor_index in range(source_shape[axis]):
            candidate_index = multi_index[:axis] + (factor_index,) + multi_index[axis + 1 :]
            product *= (
                tangent_array[candidate_index]
                if factor_index == multi_index[axis]
                else value_array[candidate_index]
            )
        output[output_index] += product
    return _program_ad_float64_vector_result(output)


def _program_ad_reduction_prod_static_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        "prod", "values", values, source_shape=source_shape
    ).reshape(source_shape)
    output_shape = _program_ad_reduction_output_shape(source_shape, axis)
    cotangent_array = _program_ad_reduction_cotangent_array(
        "prod", cotangent, output_shape=output_shape
    )
    result = np.zeros_like(value_array, dtype=np.float64)
    if axis is None:
        scalar_cotangent = float(cotangent_array)
        flat_values = value_array.reshape(-1)
        flat_result = result.reshape(-1)
        for tangent_index in range(flat_values.size):
            product = 1.0
            for factor_index in range(flat_values.size):
                if factor_index != tangent_index:
                    product *= flat_values[factor_index]
            flat_result[tangent_index] = scalar_cotangent * product
        return flat_result
    for flat_index in range(value_array.size):
        multi_index = np.unravel_index(flat_index, source_shape)
        output_index = multi_index[:axis] + multi_index[axis + 1 :]
        product = 1.0
        for factor_index in range(source_shape[axis]):
            candidate_index = multi_index[:axis] + (factor_index,) + multi_index[axis + 1 :]
            if factor_index != multi_index[axis]:
                product *= value_array[candidate_index]
        result[multi_index] = cotangent_array[output_index] * product
    return _program_ad_float64_vector_result(result)


def _program_ad_reduction_static_rule(
    name: str,
    source_shape: Sequence[int],
    axis: int | None,
) -> CustomDerivativeRule:
    source = _program_ad_reduction_normalise_static_shape(name, source_shape)
    normalised_axis = None if axis is None else _normalise_axis("axis", axis, len(source))

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        if name == "sum":
            return _program_ad_reduction_sum_static_value(
                values, source_shape=source, axis=normalised_axis
            )
        if name == "mean":
            return _program_ad_reduction_mean_static_value(
                values, source_shape=source, axis=normalised_axis
            )
        if name == "prod":
            return _program_ad_reduction_prod_static_value(
                values, source_shape=source, axis=normalised_axis
            )
        raise ValueError(f"unsupported program AD reduction primitive {name}")

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        if name == "sum":
            return _program_ad_reduction_sum_static_jvp(
                values, tangent, source_shape=source, axis=normalised_axis
            )
        if name == "mean":
            return _program_ad_reduction_mean_static_jvp(
                values, tangent, source_shape=source, axis=normalised_axis
            )
        if name == "prod":
            return _program_ad_reduction_prod_static_jvp(
                values, tangent, source_shape=source, axis=normalised_axis
            )
        raise ValueError(f"unsupported program AD reduction primitive {name}")

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if name == "sum":
            return _program_ad_reduction_sum_static_vjp(
                values, cotangent, source_shape=source, axis=normalised_axis
            )
        if name == "mean":
            return _program_ad_reduction_mean_static_vjp(
                values, cotangent, source_shape=source, axis=normalised_axis
            )
        if name == "prod":
            return _program_ad_reduction_prod_static_vjp(
                values, cotangent, source_shape=source, axis=normalised_axis
            )
        raise ValueError(f"unsupported program AD reduction primitive {name}")

    return CustomDerivativeRule(
        name=(
            f"program_ad_reduction_{name}_{_program_ad_shape_signature(source)}_axis_"
            f"{_program_ad_reduction_axis_signature(normalised_axis)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_reduction_sum_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed sum reduction signature."""

    return _program_ad_reduction_static_rule("sum", source_shape, axis)


def program_ad_reduction_mean_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed mean reduction signature."""

    return _program_ad_reduction_static_rule("mean", source_shape, axis)


def _program_ad_reduction_var_std_static_rule(
    name: Literal["var", "std"],
    source_shape: Sequence[int],
    *,
    axis: int | None,
    ddof: int,
) -> CustomDerivativeRule:
    source = _program_ad_reduction_normalise_static_shape(name, source_shape)
    normalised_axis = None if axis is None else _normalise_axis("axis", axis, len(source))
    count = (
        _program_ad_shape_static_size(source)
        if normalised_axis is None
        else source[normalised_axis]
    )
    ddof_int = _normalise_ddof(ddof, count)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        if name == "var":
            return _program_ad_reduction_var_static_value(
                values, source_shape=source, axis=normalised_axis, ddof=ddof_int
            )
        return _program_ad_reduction_std_static_value(
            values, source_shape=source, axis=normalised_axis, ddof=ddof_int
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        if name == "var":
            return _program_ad_reduction_var_static_jvp(
                values, tangent, source_shape=source, axis=normalised_axis, ddof=ddof_int
            )
        return _program_ad_reduction_std_static_jvp(
            values, tangent, source_shape=source, axis=normalised_axis, ddof=ddof_int
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if name == "var":
            return _program_ad_reduction_var_static_vjp(
                values, cotangent, source_shape=source, axis=normalised_axis, ddof=ddof_int
            )
        return _program_ad_reduction_std_static_vjp(
            values, cotangent, source_shape=source, axis=normalised_axis, ddof=ddof_int
        )

    return CustomDerivativeRule(
        name=(
            f"program_ad_reduction_{name}_{_program_ad_shape_signature(source)}_axis_"
            f"{_program_ad_reduction_axis_signature(normalised_axis)}_ddof_"
            f"{ddof_int}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_reduction_var_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
    *,
    ddof: int = 0,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed variance signature."""

    return _program_ad_reduction_var_std_static_rule("var", source_shape, axis=axis, ddof=ddof)


def program_ad_reduction_std_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
    *,
    ddof: int = 0,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed standard-deviation signature."""

    return _program_ad_reduction_var_std_static_rule("std", source_shape, axis=axis, ddof=ddof)


def program_ad_reduction_max_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed maximum reduction signature."""

    return _program_ad_reduction_order_statistic_rule(
        "max", q=1.0, source_shape=source_shape, axis=axis
    )


def program_ad_reduction_min_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed minimum reduction signature."""

    return _program_ad_reduction_order_statistic_rule(
        "min", q=0.0, source_shape=source_shape, axis=axis
    )


def program_ad_reduction_median_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed median reduction signature."""

    return _program_ad_reduction_order_statistic_rule(
        "median", q=0.5, source_shape=source_shape, axis=axis
    )


def program_ad_reduction_quantile_derivative_rule(
    source_shape: Sequence[int],
    *,
    q: object = 0.5,
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed scalar-quantile signature."""

    return _program_ad_reduction_order_statistic_rule(
        "quantile",
        q=_normalise_order_statistic_q(q, percentile=False),
        source_shape=source_shape,
        axis=axis,
    )


def program_ad_reduction_percentile_derivative_rule(
    source_shape: Sequence[int],
    *,
    q: object = 50.0,
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed scalar-percentile signature."""

    return _program_ad_reduction_order_statistic_rule(
        "percentile",
        q=_normalise_order_statistic_q(q, percentile=True),
        source_shape=source_shape,
        axis=axis,
    )


def program_ad_reduction_prod_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed product reduction signature."""

    return _program_ad_reduction_static_rule("prod", source_shape, axis)


def _program_ad_elementwise_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    raise ValueError(
        "program AD elementwise primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


def _program_ad_elementwise_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    raise ValueError(
        "program AD elementwise primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


def _raise_program_ad_derivative_losing_elementwise(name: str) -> NoReturn:
    raise ValueError(
        f"program AD {name} is derivative-losing and fails closed under the registered "
        "nondifferentiability policy"
    )


def _program_ad_elementwise_derivative_losing_value_for(name: str) -> VectorObjective:
    def value_fn(_values: NDArray[np.float64]) -> NDArray[np.float64]:
        _raise_program_ad_derivative_losing_elementwise(name)

    return value_fn


def _program_ad_elementwise_derivative_losing_jvp_for(name: str) -> CustomJVPRule:
    def jvp_rule(
        _values: NDArray[np.float64],
        _tangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _raise_program_ad_derivative_losing_elementwise(name)

    return jvp_rule


def _program_ad_elementwise_unary_vector(
    name: str, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    return _as_real_numeric_array(f"program AD elementwise {name} values", values).reshape(-1)


def _program_ad_elementwise_unary_tangent_pair(
    name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _program_ad_elementwise_unary_vector(name, values)
    tangent_vector = _as_real_numeric_array(
        f"program AD elementwise {name} tangent", tangent
    ).reshape(-1)
    if tangent_vector.shape != vector.shape:
        raise ValueError(f"program AD elementwise {name} tangent shape must match values shape")
    return vector, tangent_vector


def _program_ad_float64_vector_result(values: object) -> NDArray[np.float64]:
    return cast(NDArray[np.float64], np.asarray(values, dtype=np.float64).reshape(-1))


def _program_ad_elementwise_require_domain(
    name: str,
    vector: NDArray[np.float64],
    *,
    derivative: bool,
) -> None:
    if name == "log" and np.any(vector <= 0.0):
        raise ValueError(
            "program AD elementwise log direct rule requires values greater than zero"
        )
    if name == "log1p" and np.any(vector <= -1.0):
        raise ValueError(
            "program AD elementwise log1p direct rule requires values greater than -1"
        )
    if name == "sqrt" and np.any(vector < 0.0):
        raise ValueError("program AD elementwise sqrt direct rule requires non-negative values")
    if name == "sqrt" and derivative and np.any(vector <= 0.0):
        raise ValueError(
            "program AD elementwise sqrt derivative is singular at non-positive values"
        )
    if name in {"arcsin", "arccos"} and np.any(np.abs(vector) > 1.0):
        raise ValueError(f"program AD elementwise {name} direct rule requires values in [-1, 1]")
    if name in {"arcsin", "arccos"} and derivative and np.any(np.abs(vector) >= 1.0):
        raise ValueError(
            f"program AD elementwise {name} derivative is singular at boundary values"
        )
    if name == "reciprocal" and np.any(vector == 0.0):
        raise ValueError("program AD elementwise reciprocal direct rule requires non-zero values")
    if name == "tan" and derivative and np.any(np.cos(vector) == 0.0):
        raise ValueError("program AD elementwise tan derivative is singular at odd pi/2 values")
    if name == "abs" and derivative and np.any(vector == 0.0):
        raise ValueError("program AD elementwise abs derivative is undefined at zero")


def _program_ad_elementwise_unary_value(
    name: str, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    vector = _program_ad_elementwise_unary_vector(name, values)
    _program_ad_elementwise_require_domain(name, vector, derivative=False)
    if name == "sin":
        return _program_ad_float64_vector_result(np.sin(vector))
    if name == "cos":
        return _program_ad_float64_vector_result(np.cos(vector))
    if name == "exp":
        return _program_ad_float64_vector_result(np.exp(vector))
    if name == "expm1":
        return _program_ad_float64_vector_result(np.expm1(vector))
    if name == "log":
        return _program_ad_float64_vector_result(np.log(vector))
    if name == "log1p":
        return _program_ad_float64_vector_result(np.log1p(vector))
    if name == "sqrt":
        return _program_ad_float64_vector_result(np.sqrt(vector))
    if name == "tan":
        return _program_ad_float64_vector_result(np.tan(vector))
    if name == "tanh":
        return _program_ad_float64_vector_result(np.tanh(vector))
    if name == "arcsin":
        return _program_ad_float64_vector_result(np.arcsin(vector))
    if name == "arccos":
        return _program_ad_float64_vector_result(np.arccos(vector))
    if name == "reciprocal":
        return _program_ad_float64_vector_result(np.reciprocal(vector))
    if name == "square":
        return _program_ad_float64_vector_result(np.square(vector))
    if name == "abs":
        return _program_ad_float64_vector_result(np.abs(vector))
    if name == "negative":
        return _program_ad_float64_vector_result(np.negative(vector))
    raise ValueError(f"unsupported program AD elementwise primitive {name}")


def _program_ad_elementwise_unary_jvp(
    name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector, tangent_vector = _program_ad_elementwise_unary_tangent_pair(name, values, tangent)
    _program_ad_elementwise_require_domain(name, vector, derivative=True)
    if name == "sin":
        return _program_ad_float64_vector_result(np.cos(vector) * tangent_vector)
    if name == "cos":
        return _program_ad_float64_vector_result(-np.sin(vector) * tangent_vector)
    if name == "exp":
        return _program_ad_float64_vector_result(np.exp(vector) * tangent_vector)
    if name == "expm1":
        return _program_ad_float64_vector_result(np.exp(vector) * tangent_vector)
    if name == "log":
        return _program_ad_float64_vector_result(tangent_vector / vector)
    if name == "log1p":
        return _program_ad_float64_vector_result(tangent_vector / (1.0 + vector))
    if name == "sqrt":
        return _program_ad_float64_vector_result(tangent_vector / (2.0 * np.sqrt(vector)))
    if name == "tan":
        return _program_ad_float64_vector_result(tangent_vector / np.cos(vector) ** 2)
    if name == "tanh":
        return _program_ad_float64_vector_result(tangent_vector * (1.0 - np.tanh(vector) ** 2))
    if name == "arcsin":
        return _program_ad_float64_vector_result(tangent_vector / np.sqrt(1.0 - vector**2))
    if name == "arccos":
        return _program_ad_float64_vector_result(-tangent_vector / np.sqrt(1.0 - vector**2))
    if name == "reciprocal":
        return _program_ad_float64_vector_result(-tangent_vector / vector**2)
    if name == "square":
        return _program_ad_float64_vector_result(2.0 * vector * tangent_vector)
    if name == "abs":
        return _program_ad_float64_vector_result(np.sign(vector) * tangent_vector)
    if name == "negative":
        return _program_ad_float64_vector_result(np.negative(tangent_vector))
    raise ValueError(f"unsupported program AD elementwise primitive {name}")


def _program_ad_elementwise_direct_value_for(name: str) -> VectorObjective:
    return lambda values: _program_ad_elementwise_unary_value(name, values)


def _program_ad_elementwise_direct_jvp_for(name: str) -> CustomJVPRule:
    return lambda values, tangent: _program_ad_elementwise_unary_jvp(name, values, tangent)


def _program_ad_elementwise_direct_vjp_for(name: str) -> CustomVJPRule:
    return lambda values, cotangent: _program_ad_elementwise_unary_jvp(name, values, cotangent)


def _program_ad_elementwise_binary_pair(
    name: str,
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD elementwise {name} values", values).reshape(-1)
    if vector.size == 0 or vector.size % 2 != 0:
        raise ValueError(
            f"program AD elementwise {name} direct rule requires two equal flat operands"
        )
    midpoint = vector.size // 2
    return vector[:midpoint], vector[midpoint:]


def _program_ad_elementwise_binary_tangent_pair(
    name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    left, right = _program_ad_elementwise_binary_pair(name, values)
    tangent_left, tangent_right = _program_ad_elementwise_binary_pair(f"{name} tangent", tangent)
    if tangent_left.shape != left.shape or tangent_right.shape != right.shape:
        raise ValueError(f"program AD elementwise {name} tangent shape must match values shape")
    return left, right, tangent_left, tangent_right


def _program_ad_elementwise_require_binary_domain(
    name: str,
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    *,
    derivative: bool,
) -> None:
    if name == "divide" and np.any(right == 0.0):
        raise ValueError(
            "program AD elementwise divide direct rule requires non-zero right operand"
        )
    if name == "power" and np.any(left <= 0.0):
        raise ValueError("program AD elementwise power direct rule requires positive left operand")
    if name in {"maximum", "minimum"} and derivative and np.any(left == right):
        raise ValueError(
            f"program AD elementwise {name} derivative is undefined at equal operands"
        )


def _program_ad_elementwise_binary_value(
    name: str,
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_elementwise_binary_pair(name, values)
    _program_ad_elementwise_require_binary_domain(name, left, right, derivative=False)
    if name == "add":
        return _program_ad_float64_vector_result(left + right)
    if name == "subtract":
        return _program_ad_float64_vector_result(left - right)
    if name == "multiply":
        return _program_ad_float64_vector_result(left * right)
    if name == "divide":
        return _program_ad_float64_vector_result(left / right)
    if name == "power":
        return _program_ad_float64_vector_result(left**right)
    if name == "maximum":
        return _program_ad_float64_vector_result(np.maximum(left, right))
    if name == "minimum":
        return _program_ad_float64_vector_result(np.minimum(left, right))
    raise ValueError(f"unsupported program AD elementwise primitive {name}")


def _program_ad_elementwise_binary_jvp(
    name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right, tangent_left, tangent_right = _program_ad_elementwise_binary_tangent_pair(
        name, values, tangent
    )
    _program_ad_elementwise_require_binary_domain(name, left, right, derivative=True)
    if name == "add":
        return _program_ad_float64_vector_result(tangent_left + tangent_right)
    if name == "subtract":
        return _program_ad_float64_vector_result(tangent_left - tangent_right)
    if name == "multiply":
        return _program_ad_float64_vector_result(tangent_left * right + left * tangent_right)
    if name == "divide":
        return _program_ad_float64_vector_result(
            (tangent_left * right - left * tangent_right) / right**2
        )
    if name == "power":
        return _program_ad_float64_vector_result(
            left**right * (tangent_right * np.log(left) + right * tangent_left / left)
        )
    if name == "maximum":
        return _program_ad_float64_vector_result(
            np.where(left > right, tangent_left, tangent_right)
        )
    if name == "minimum":
        return _program_ad_float64_vector_result(
            np.where(left < right, tangent_left, tangent_right)
        )
    raise ValueError(f"unsupported program AD elementwise primitive {name}")


def _program_ad_elementwise_binary_vjp(
    name: str,
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_elementwise_binary_pair(name, values)
    cotangent_vector = _as_real_numeric_array(
        f"program AD elementwise {name} cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.shape != left.shape:
        raise ValueError(
            f"program AD elementwise {name} VJP cotangent shape must match output shape"
        )
    _program_ad_elementwise_require_binary_domain(name, left, right, derivative=True)
    left_vjp: NDArray[np.float64]
    right_vjp: NDArray[np.float64]
    if name == "add":
        left_vjp = cotangent_vector
        right_vjp = cotangent_vector
    elif name == "subtract":
        left_vjp = cotangent_vector
        right_vjp = -cotangent_vector
    elif name == "multiply":
        left_vjp = cotangent_vector * right
        right_vjp = cotangent_vector * left
    elif name == "divide":
        left_vjp = cotangent_vector / right
        right_vjp = -cotangent_vector * left / right**2
    elif name == "power":
        left_vjp = cotangent_vector * right * left ** (right - 1.0)
        right_vjp = cotangent_vector * left**right * np.log(left)
    elif name == "maximum":
        left_vjp = _program_ad_float64_vector_result(np.where(left > right, cotangent_vector, 0.0))
        right_vjp = _program_ad_float64_vector_result(
            np.where(left > right, 0.0, cotangent_vector)
        )
    elif name == "minimum":
        left_vjp = _program_ad_float64_vector_result(np.where(left < right, cotangent_vector, 0.0))
        right_vjp = _program_ad_float64_vector_result(
            np.where(left < right, 0.0, cotangent_vector)
        )
    else:
        raise ValueError(f"unsupported program AD elementwise primitive {name}")
    return _program_ad_float64_vector_result(np.concatenate([left_vjp, right_vjp]))


def _program_ad_elementwise_binary_value_for(name: str) -> VectorObjective:
    return lambda values: _program_ad_elementwise_binary_value(name, values)


def _program_ad_elementwise_binary_jvp_for(name: str) -> CustomJVPRule:
    return lambda values, tangent: _program_ad_elementwise_binary_jvp(name, values, tangent)


def _program_ad_elementwise_binary_vjp_for(name: str) -> CustomVJPRule:
    return lambda values, cotangent: _program_ad_elementwise_binary_vjp(name, values, cotangent)


def _program_ad_elementwise_normalise_binary_static_shapes(
    name: str,
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if name not in _PROGRAM_AD_ELEMENTWISE_BINARY_NAMES:
        raise ValueError(f"unsupported program AD elementwise binary primitive {name}")
    left = tuple(int(dimension) for dimension in left_shape)
    right = tuple(int(dimension) for dimension in right_shape)
    if any(dimension < 0 for dimension in (*left, *right)):
        raise ValueError(
            f"program AD elementwise {name} direct rule requires non-negative dimensions"
        )
    try:
        output = np.broadcast_shapes(left, right)
    except ValueError as exc:
        raise ValueError(
            f"program AD elementwise {name} direct rule requires broadcast-compatible shapes"
        ) from exc
    return left, right, tuple(int(dimension) for dimension in output)


def _program_ad_elementwise_binary_static_split(
    name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD elementwise {name} {role}", values).reshape(-1)
    left_size = _program_ad_shape_static_size(left_shape)
    right_size = _program_ad_shape_static_size(right_shape)
    if vector.size != left_size + right_size:
        raise ValueError(
            f"program AD elementwise {name} direct rule requires flattened left operand "
            "followed by right operand"
        )
    return (
        vector[:left_size].reshape(left_shape),
        vector[left_size:].reshape(right_shape),
    )


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


def _program_ad_elementwise_binary_static_value_array(
    name: str,
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    _program_ad_elementwise_require_binary_domain(name, left, right, derivative=False)
    if name == "add":
        return _program_ad_float64_vector_result(left + right)
    if name == "subtract":
        return _program_ad_float64_vector_result(left - right)
    if name == "multiply":
        return _program_ad_float64_vector_result(left * right)
    if name == "divide":
        return _program_ad_float64_vector_result(left / right)
    if name == "power":
        return _program_ad_float64_vector_result(left**right)
    if name == "maximum":
        return _program_ad_float64_vector_result(np.maximum(left, right))
    if name == "minimum":
        return _program_ad_float64_vector_result(np.minimum(left, right))
    raise ValueError(f"unsupported program AD elementwise primitive {name}")


def _program_ad_elementwise_binary_static_jvp_array(
    name: str,
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    tangent_left: NDArray[np.float64],
    tangent_right: NDArray[np.float64],
) -> NDArray[np.float64]:
    _program_ad_elementwise_require_binary_domain(name, left, right, derivative=True)
    if name == "add":
        return _program_ad_float64_vector_result(tangent_left + tangent_right)
    if name == "subtract":
        return _program_ad_float64_vector_result(tangent_left - tangent_right)
    if name == "multiply":
        return _program_ad_float64_vector_result(tangent_left * right + left * tangent_right)
    if name == "divide":
        return _program_ad_float64_vector_result(
            (tangent_left * right - left * tangent_right) / right**2
        )
    if name == "power":
        return _program_ad_float64_vector_result(
            left**right * (tangent_right * np.log(left) + right * tangent_left / left)
        )
    if name == "maximum":
        return _program_ad_float64_vector_result(
            np.where(left > right, tangent_left, tangent_right)
        )
    if name == "minimum":
        return _program_ad_float64_vector_result(
            np.where(left < right, tangent_left, tangent_right)
        )
    raise ValueError(f"unsupported program AD elementwise primitive {name}")


def _program_ad_elementwise_binary_static_adjoint_arrays(
    name: str,
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    _program_ad_elementwise_require_binary_domain(name, left, right, derivative=True)
    if name == "add":
        return cotangent, cotangent
    if name == "subtract":
        return cotangent, -cotangent
    if name == "multiply":
        return cotangent * right, cotangent * left
    if name == "divide":
        return cotangent / right, -cotangent * left / right**2
    if name == "power":
        return (
            cotangent * right * left ** (right - 1.0),
            cotangent * left**right * np.log(left),
        )
    if name == "maximum":
        return (
            _program_ad_float64_vector_result(np.where(left > right, cotangent, 0.0)).reshape(
                cotangent.shape
            ),
            _program_ad_float64_vector_result(np.where(left > right, 0.0, cotangent)).reshape(
                cotangent.shape
            ),
        )
    if name == "minimum":
        return (
            _program_ad_float64_vector_result(np.where(left < right, cotangent, 0.0)).reshape(
                cotangent.shape
            ),
            _program_ad_float64_vector_result(np.where(left < right, 0.0, cotangent)).reshape(
                cotangent.shape
            ),
        )
    raise ValueError(f"unsupported program AD elementwise primitive {name}")


def program_ad_elementwise_binary_derivative_rule(
    name: str,
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed broadcasted binary primitive."""

    left_static_shape, right_static_shape, output_shape = (
        _program_ad_elementwise_normalise_binary_static_shapes(name, left_shape, right_shape)
    )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_elementwise_binary_static_split(
            name, "values", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        return _program_ad_elementwise_binary_static_value_array(name, left, right)

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_elementwise_binary_static_split(
            name, "values", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        tangent_left, tangent_right = _program_ad_elementwise_binary_static_split(
            name,
            "tangent",
            tangent,
            left_shape=left_static_shape,
            right_shape=right_static_shape,
        )
        return _program_ad_elementwise_binary_static_jvp_array(
            name, left, right, tangent_left, tangent_right
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        left, right = _program_ad_elementwise_binary_static_split(
            name, "values", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        cotangent_vector = _as_real_numeric_array(
            f"program AD elementwise {name} cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != _program_ad_shape_static_size(output_shape):
            raise ValueError(
                f"program AD elementwise {name} VJP cotangent shape must match output shape"
            )
        cotangent_array = cotangent_vector.reshape(output_shape)
        left_adjoint, right_adjoint = _program_ad_elementwise_binary_static_adjoint_arrays(
            name, left, right, cotangent_array
        )
        return _program_ad_float64_vector_result(
            np.concatenate(
                (
                    _program_ad_elementwise_unbroadcast(
                        left_adjoint, target_shape=left_static_shape
                    ),
                    _program_ad_elementwise_unbroadcast(
                        right_adjoint, target_shape=right_static_shape
                    ),
                )
            )
        )

    return CustomDerivativeRule(
        name=(
            f"program_ad_elementwise_{name}_{_program_ad_shape_signature(left_static_shape)}_by_"
            f"{_program_ad_shape_signature(right_static_shape)}_broadcast_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_elementwise_derivative_rule(name: str) -> CustomDerivativeRule:
    if name in _PROGRAM_AD_ELEMENTWISE_DISCONTINUOUS_NAMES:
        return CustomDerivativeRule(
            name=f"program_ad_elementwise_{name}_fail_closed_rule",
            value_fn=_program_ad_elementwise_derivative_losing_value_for(name),
            jvp_rule=_program_ad_elementwise_derivative_losing_jvp_for(name),
            vjp_rule=_program_ad_elementwise_derivative_losing_jvp_for(name),
        )
    if name in _PROGRAM_AD_ELEMENTWISE_UNARY_NAMES:
        return CustomDerivativeRule(
            name=f"program_ad_elementwise_{name}_direct_rule",
            value_fn=_program_ad_elementwise_direct_value_for(name),
            jvp_rule=_program_ad_elementwise_direct_jvp_for(name),
            vjp_rule=_program_ad_elementwise_direct_vjp_for(name),
        )
    if name in _PROGRAM_AD_ELEMENTWISE_BINARY_NAMES:
        return CustomDerivativeRule(
            name=f"program_ad_elementwise_{name}_direct_rule",
            value_fn=_program_ad_elementwise_binary_value_for(name),
            jvp_rule=_program_ad_elementwise_binary_jvp_for(name),
            vjp_rule=_program_ad_elementwise_binary_vjp_for(name),
        )
    return CustomDerivativeRule(
        name=f"program_ad_elementwise_{name}_trace_contract",
        value_fn=_program_ad_elementwise_direct_value,
        jvp_rule=_program_ad_elementwise_direct_jvp,
    )


def _program_ad_selection_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    raise ValueError(
        "program AD selection primitive contracts require static derivative factories "
        "or operator-intercepted trace dispatch"
    )


def _program_ad_selection_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    raise ValueError(
        "program AD selection primitive contracts require static derivative factories "
        "or operator-intercepted trace dispatch"
    )


def _program_ad_selection_derivative_rule(name: str) -> CustomDerivativeRule:
    return CustomDerivativeRule(
        name=f"program_ad_selection_{name}_trace_contract",
        value_fn=_program_ad_selection_direct_value,
        jvp_rule=_program_ad_selection_direct_jvp,
    )


def _program_ad_selection_normalise_shapes(
    primitive_name: str,
    true_shape: Sequence[int],
    false_shape: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    true_static_shape = tuple(int(dimension) for dimension in true_shape)
    false_static_shape = tuple(int(dimension) for dimension in false_shape)
    if any(dimension < 0 for dimension in (*true_static_shape, *false_static_shape)):
        raise ValueError(
            f"program AD selection {primitive_name} direct rule requires non-negative dimensions"
        )
    try:
        output_shape = np.broadcast_shapes(true_static_shape, false_static_shape)
    except ValueError as exc:
        raise ValueError(
            f"program AD selection {primitive_name} direct rule requires "
            "broadcast-compatible branch shapes"
        ) from exc
    return true_static_shape, false_static_shape, tuple(int(dim) for dim in output_shape)


def _program_ad_selection_condition_mask(
    condition: object,
    output_shape: tuple[int, ...],
) -> NDArray[np.bool_]:
    raw_condition = np.asarray(condition)
    if raw_condition.dtype.kind != "b":
        raise ValueError("program AD selection where direct rule requires a boolean condition")
    if tuple(raw_condition.shape) not in {(), output_shape}:
        raise ValueError(
            "program AD selection where direct rule requires scalar or output-shaped condition"
        )
    return np.broadcast_to(raw_condition, output_shape).astype(np.bool_, copy=False)


def _program_ad_selection_split_pair(
    primitive_name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    true_shape: tuple[int, ...],
    false_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(
        f"program AD selection {primitive_name} {role}", values
    ).reshape(-1)
    true_size = _program_ad_shape_static_size(true_shape)
    false_size = _program_ad_shape_static_size(false_shape)
    if vector.size != true_size + false_size:
        raise ValueError(
            f"program AD selection {primitive_name} direct rule requires flattened true branch "
            "followed by false branch"
        )
    return (
        vector[:true_size].reshape(true_shape),
        vector[true_size:].reshape(false_shape),
    )


def program_ad_selection_where_derivative_rule(
    condition: object,
    true_shape: Sequence[int],
    false_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed NumPy where signature."""

    true_static_shape, false_static_shape, output_shape = _program_ad_selection_normalise_shapes(
        "where", true_shape, false_shape
    )
    condition_mask = _program_ad_selection_condition_mask(condition, output_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        true_values, false_values = _program_ad_selection_split_pair(
            "where",
            "values",
            values,
            true_shape=true_static_shape,
            false_shape=false_static_shape,
        )
        return _program_ad_float64_vector_result(
            np.where(
                condition_mask,
                np.broadcast_to(true_values, output_shape),
                np.broadcast_to(false_values, output_shape),
            )
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_selection_split_pair(
            "where",
            "values",
            values,
            true_shape=true_static_shape,
            false_shape=false_static_shape,
        )
        true_tangent, false_tangent = _program_ad_selection_split_pair(
            "where",
            "tangent",
            tangent,
            true_shape=true_static_shape,
            false_shape=false_static_shape,
        )
        return _program_ad_float64_vector_result(
            np.where(
                condition_mask,
                np.broadcast_to(true_tangent, output_shape),
                np.broadcast_to(false_tangent, output_shape),
            )
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_selection_split_pair(
            "where",
            "values",
            values,
            true_shape=true_static_shape,
            false_shape=false_static_shape,
        )
        cotangent_array = _as_real_numeric_array(
            "program AD selection where cotangent", cotangent
        ).reshape(-1)
        if cotangent_array.size != _program_ad_shape_static_size(output_shape):
            raise ValueError(
                "program AD selection where VJP cotangent shape must match output shape"
            )
        cotangent_output = cotangent_array.reshape(output_shape)
        true_adjoint = _program_ad_elementwise_unbroadcast(
            np.where(condition_mask, cotangent_output, 0.0), target_shape=true_static_shape
        )
        false_adjoint = _program_ad_elementwise_unbroadcast(
            np.where(condition_mask, 0.0, cotangent_output), target_shape=false_static_shape
        )
        return _program_ad_float64_vector_result(np.concatenate((true_adjoint, false_adjoint)))

    return CustomDerivativeRule(
        name=(
            f"program_ad_selection_where_{_program_ad_shape_signature(true_static_shape)}_by_"
            f"{_program_ad_shape_signature(false_static_shape)}_static_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_selection_normalise_clip_shapes(
    source_shape: Sequence[int],
    lower_shape: Sequence[int],
    upper_shape: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    source_static_shape = tuple(int(dimension) for dimension in source_shape)
    lower_static_shape = tuple(int(dimension) for dimension in lower_shape)
    upper_static_shape = tuple(int(dimension) for dimension in upper_shape)
    if any(
        dimension < 0
        for dimension in (*source_static_shape, *lower_static_shape, *upper_static_shape)
    ):
        raise ValueError("program AD selection clip direct rule requires non-negative dimensions")
    try:
        lower_output = np.broadcast_shapes(source_static_shape, lower_static_shape)
        upper_output = np.broadcast_shapes(source_static_shape, upper_static_shape)
    except ValueError as exc:
        raise ValueError(
            "program AD selection clip direct rule requires bounds broadcastable to source shape"
        ) from exc
    if tuple(lower_output) != source_static_shape or tuple(upper_output) != source_static_shape:
        raise ValueError(
            "program AD selection clip direct rule requires bounds broadcastable to source shape"
        )
    return source_static_shape, lower_static_shape, upper_static_shape


def _program_ad_selection_split_clip(
    role: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    lower_shape: tuple[int, ...],
    upper_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD selection clip {role}", values).reshape(-1)
    source_size = _program_ad_shape_static_size(source_shape)
    lower_size = _program_ad_shape_static_size(lower_shape)
    upper_size = _program_ad_shape_static_size(upper_shape)
    if vector.size != source_size + lower_size + upper_size:
        raise ValueError(
            "program AD selection clip direct rule requires flattened source, lower, and upper"
        )
    lower_start = source_size
    upper_start = source_size + lower_size
    return (
        vector[:source_size].reshape(source_shape),
        vector[lower_start:upper_start].reshape(lower_shape),
        vector[upper_start:].reshape(upper_shape),
    )


def _program_ad_selection_clip_domain(
    values: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    *,
    derivative: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower_broadcast = np.broadcast_to(lower, values.shape)
    upper_broadcast = np.broadcast_to(upper, values.shape)
    if np.any(lower_broadcast > upper_broadcast):
        raise ValueError("program AD selection clip lower bound must not exceed upper bound")
    if derivative and np.any((values == lower_broadcast) | (values == upper_broadcast)):
        raise ValueError("program AD selection clip derivative is undefined at clipping boundary")
    return lower_broadcast, upper_broadcast


def program_ad_selection_clip_derivative_rule(
    source_shape: Sequence[int],
    *,
    lower_shape: Sequence[int] = (),
    upper_shape: Sequence[int] = (),
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed NumPy clip signature."""

    source_static_shape, lower_static_shape, upper_static_shape = (
        _program_ad_selection_normalise_clip_shapes(source_shape, lower_shape, upper_shape)
    )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        source, lower, upper = _program_ad_selection_split_clip(
            "values",
            values,
            source_shape=source_static_shape,
            lower_shape=lower_static_shape,
            upper_shape=upper_static_shape,
        )
        lower_broadcast, upper_broadcast = _program_ad_selection_clip_domain(
            source, lower, upper, derivative=False
        )
        return _program_ad_float64_vector_result(np.clip(source, lower_broadcast, upper_broadcast))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        source, lower, upper = _program_ad_selection_split_clip(
            "values",
            values,
            source_shape=source_static_shape,
            lower_shape=lower_static_shape,
            upper_shape=upper_static_shape,
        )
        source_tangent, lower_tangent, upper_tangent = _program_ad_selection_split_clip(
            "tangent",
            tangent,
            source_shape=source_static_shape,
            lower_shape=lower_static_shape,
            upper_shape=upper_static_shape,
        )
        lower_broadcast, upper_broadcast = _program_ad_selection_clip_domain(
            source, lower, upper, derivative=True
        )
        lower_tangent = np.broadcast_to(lower_tangent, source_static_shape)
        upper_tangent = np.broadcast_to(upper_tangent, source_static_shape)
        return _program_ad_float64_vector_result(
            np.where(
                source < lower_broadcast,
                lower_tangent,
                np.where(source > upper_broadcast, upper_tangent, source_tangent),
            )
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        source, lower, upper = _program_ad_selection_split_clip(
            "values",
            values,
            source_shape=source_static_shape,
            lower_shape=lower_static_shape,
            upper_shape=upper_static_shape,
        )
        cotangent_array = _as_real_numeric_array(
            "program AD selection clip cotangent", cotangent
        ).reshape(-1)
        if cotangent_array.size != _program_ad_shape_static_size(source_static_shape):
            raise ValueError(
                "program AD selection clip VJP cotangent shape must match output shape"
            )
        cotangent_output = cotangent_array.reshape(source_static_shape)
        lower_broadcast, upper_broadcast = _program_ad_selection_clip_domain(
            source, lower, upper, derivative=True
        )
        below = source < lower_broadcast
        above = source > upper_broadcast
        source_adjoint = np.where(~below & ~above, cotangent_output, 0.0)
        lower_adjoint = _program_ad_elementwise_unbroadcast(
            np.where(below, cotangent_output, 0.0), target_shape=lower_static_shape
        )
        upper_adjoint = _program_ad_elementwise_unbroadcast(
            np.where(above, cotangent_output, 0.0), target_shape=upper_static_shape
        )
        return _program_ad_float64_vector_result(
            np.concatenate((source_adjoint.reshape(-1), lower_adjoint, upper_adjoint))
        )

    return CustomDerivativeRule(
        name=(
            f"program_ad_selection_clip_{_program_ad_shape_signature(source_static_shape)}_bounds_"
            f"{_program_ad_shape_signature(lower_static_shape)}_by_"
            f"{_program_ad_shape_signature(upper_static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_elementwise_name(ufunc: np.ufunc) -> str:
    if ufunc is np.absolute:
        return "abs"
    return str(ufunc.__name__)


def _program_ad_product_split_pair(
    primitive_name: str,
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD product {primitive_name} values", values).reshape(
        -1
    )
    if vector.size == 0 or vector.size % 2 != 0:
        raise ValueError(
            f"program AD product {primitive_name} direct rule requires two equal flat operands"
        )
    midpoint = vector.size // 2
    return vector[:midpoint], vector[midpoint:]


def _program_ad_product_dot_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("dot", values)
    return np.array([float(np.dot(left, right))], dtype=np.float64)


def _program_ad_product_dot_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("dot", values)
    tangent_left, tangent_right = _program_ad_product_split_pair("dot tangent", tangent)
    if tangent_left.shape != left.shape or tangent_right.shape != right.shape:
        raise ValueError("program AD product dot tangent shape must match values shape")
    return np.array(
        [float(np.dot(tangent_left, right) + np.dot(left, tangent_right))], dtype=np.float64
    )


def _program_ad_product_scalar_cotangent(
    primitive_name: str,
    cotangent: NDArray[np.float64],
) -> float:
    cotangent_vector = _as_real_numeric_array(
        f"program AD product {primitive_name} cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.shape != (1,):
        raise ValueError(f"program AD product {primitive_name} VJP requires one scalar cotangent")
    return float(cotangent_vector[0])


def _program_ad_product_dot_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("dot", values)
    scalar_cotangent = _program_ad_product_scalar_cotangent("dot", cotangent)
    return _program_ad_float64_vector_result(
        np.concatenate((scalar_cotangent * right, scalar_cotangent * left))
    )


def _program_ad_product_vdot_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("vdot", values)
    return np.array([float(np.vdot(left, right))], dtype=np.float64)


def _program_ad_product_vdot_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("vdot", values)
    tangent_left, tangent_right = _program_ad_product_split_pair("vdot tangent", tangent)
    if tangent_left.shape != left.shape or tangent_right.shape != right.shape:
        raise ValueError("program AD product vdot tangent shape must match values shape")
    return np.array(
        [float(np.vdot(tangent_left, right) + np.vdot(left, tangent_right))],
        dtype=np.float64,
    )


def _program_ad_product_vdot_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("vdot", values)
    scalar_cotangent = _program_ad_product_scalar_cotangent("vdot", cotangent)
    return _program_ad_float64_vector_result(
        np.concatenate((scalar_cotangent * right, scalar_cotangent * left))
    )


def _program_ad_product_inner_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("inner", values)
    return _program_ad_float64_vector_result([float(np.inner(left, right))])


def _program_ad_product_inner_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("inner", values)
    tangent_left, tangent_right = _program_ad_product_split_pair("inner tangent", tangent)
    if tangent_left.shape != left.shape or tangent_right.shape != right.shape:
        raise ValueError("program AD product inner tangent shape must match values shape")
    return _program_ad_float64_vector_result(
        [float(np.inner(tangent_left, right) + np.inner(left, tangent_right))]
    )


def _program_ad_product_inner_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("inner", values)
    scalar_cotangent = _program_ad_product_scalar_cotangent("inner", cotangent)
    return _program_ad_float64_vector_result(
        np.concatenate((scalar_cotangent * right, scalar_cotangent * left))
    )


def _program_ad_product_outer_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("outer", values)
    return _program_ad_float64_vector_result(np.outer(left, right))


def _program_ad_product_outer_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("outer", values)
    tangent_left, tangent_right = _program_ad_product_split_pair("outer tangent", tangent)
    if tangent_left.shape != left.shape or tangent_right.shape != right.shape:
        raise ValueError("program AD product outer tangent shape must match values shape")
    return _program_ad_float64_vector_result(
        np.outer(tangent_left, right) + np.outer(left, tangent_right)
    )


def _program_ad_product_outer_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("outer", values)
    cotangent_vector = _as_real_numeric_array(
        "program AD product outer cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.size != left.size * right.size:
        raise ValueError("program AD product outer VJP cotangent shape must match output shape")
    cotangent_matrix = cotangent_vector.reshape(left.size, right.size)
    return _program_ad_float64_vector_result(
        np.concatenate((cotangent_matrix @ right, cotangent_matrix.T @ left))
    )


def _program_ad_product_square_matrix_pair(
    primitive_name: str,
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    left, right = _program_ad_product_split_pair(primitive_name, values)
    rows = int(math.isqrt(left.size))
    if rows * rows != left.size:
        raise ValueError(
            f"program AD product {primitive_name} direct rule requires two square matrices"
        )
    return left.reshape(rows, rows), right.reshape(rows, rows)


def _program_ad_product_matmul_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    left, right = _program_ad_product_square_matrix_pair("matmul", values)
    return (left @ right).reshape(-1).astype(np.float64)


def _program_ad_product_matmul_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_square_matrix_pair("matmul", values)
    tangent_left, tangent_right = _program_ad_product_square_matrix_pair("matmul tangent", tangent)
    if tangent_left.shape != left.shape or tangent_right.shape != right.shape:
        raise ValueError("program AD product matmul tangent shape must match values shape")
    return (tangent_left @ right + left @ tangent_right).reshape(-1).astype(np.float64)


def _program_ad_product_matmul_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_square_matrix_pair("matmul", values)
    cotangent_matrix = _as_real_numeric_array(
        "program AD product matmul cotangent", cotangent
    ).reshape(-1)
    if cotangent_matrix.shape != (left.size,):
        raise ValueError("program AD product matmul VJP cotangent shape must match output shape")
    cotangent_square = cotangent_matrix.reshape(left.shape)
    return _program_ad_float64_vector_result(
        np.concatenate(
            (
                (cotangent_square @ right.T).reshape(-1),
                (left.T @ cotangent_square).reshape(-1),
            )
        )
    )


def _program_ad_product_normalise_matmul_shapes(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    left = tuple(int(dimension) for dimension in left_shape)
    right = tuple(int(dimension) for dimension in right_shape)
    if any(dimension < 0 for dimension in (*left, *right)):
        raise ValueError("program AD product matmul direct rule requires non-negative dimensions")
    if len(left) not in {1, 2} or len(right) not in {1, 2}:
        raise ValueError(
            "program AD product matmul direct rule supports rank-1 or rank-2 operands"
        )
    if len(left) == 1 and len(right) == 1:
        if left[0] != right[0]:
            raise ValueError("program AD product matmul direct rule dimensions must align")
        return left, right, ()
    if len(left) == 2 and len(right) == 1:
        if left[1] != right[0]:
            raise ValueError("program AD product matmul direct rule dimensions must align")
        return left, right, (left[0],)
    if len(left) == 1 and len(right) == 2:
        if left[0] != right[0]:
            raise ValueError("program AD product matmul direct rule dimensions must align")
        return left, right, (right[1],)
    if left[1] != right[0]:
        raise ValueError("program AD product matmul direct rule dimensions must align")
    return left, right, (left[0], right[1])


def _program_ad_product_matmul_static_split(
    role: str,
    values: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD product matmul {role}", values).reshape(-1)
    left_size = _program_ad_shape_static_size(left_shape)
    right_size = _program_ad_shape_static_size(right_shape)
    if vector.size != left_size + right_size:
        raise ValueError(
            "program AD product matmul direct rule requires flattened left operand "
            "followed by right operand"
        )
    return (
        vector[:left_size].reshape(left_shape),
        vector[left_size:].reshape(right_shape),
    )


def program_ad_product_matmul_derivative_rule(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed matmul primitive signature."""

    left_static_shape, right_static_shape, output_shape = (
        _program_ad_product_normalise_matmul_shapes(left_shape, right_shape)
    )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_matmul_static_split(
            "values", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        return _program_ad_float64_vector_result(left @ right)

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_matmul_static_split(
            "values", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        tangent_left, tangent_right = _program_ad_product_matmul_static_split(
            "tangent", tangent, left_shape=left_static_shape, right_shape=right_static_shape
        )
        return _program_ad_float64_vector_result(tangent_left @ right + left @ tangent_right)

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        left, right = _program_ad_product_matmul_static_split(
            "values", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        cotangent_vector = _as_real_numeric_array(
            "program AD product matmul cotangent", cotangent
        ).reshape(-1)
        expected_size = _program_ad_shape_static_size(output_shape)
        if cotangent_vector.size != expected_size:
            raise ValueError(
                "program AD product matmul VJP cotangent shape must match output shape"
            )
        cotangent_value = (
            float(cotangent_vector[0])
            if output_shape == ()
            else cotangent_vector.reshape(output_shape)
        )
        left_adjoint: NDArray[np.float64]
        right_adjoint: NDArray[np.float64]
        if left.ndim == 1 and right.ndim == 1:
            scalar = float(cotangent_value)
            left_adjoint = scalar * right
            right_adjoint = scalar * left
        elif left.ndim == 2 and right.ndim == 1:
            cotangent_array = cast(NDArray[np.float64], cotangent_value)
            left_adjoint = _program_ad_float64_vector_result(
                np.outer(cotangent_array, right)
            ).reshape(left.shape)
            right_adjoint = left.T @ cotangent_array
        elif left.ndim == 1 and right.ndim == 2:
            cotangent_array = cast(NDArray[np.float64], cotangent_value)
            left_adjoint = right @ cotangent_array
            right_adjoint = _program_ad_float64_vector_result(
                np.outer(left, cotangent_array)
            ).reshape(right.shape)
        else:
            cotangent_array = cast(NDArray[np.float64], cotangent_value)
            left_adjoint = cotangent_array @ right.T
            right_adjoint = left.T @ cotangent_array
        return _program_ad_float64_vector_result(
            np.concatenate(
                (np.asarray(left_adjoint).reshape(-1), np.asarray(right_adjoint).reshape(-1))
            )
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_product_matmul_"
            f"{_program_ad_shape_signature(left_static_shape)}_by_"
            f"{_program_ad_shape_signature(right_static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_product_normalise_inner_shapes(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    left = tuple(int(dimension) for dimension in left_shape)
    right = tuple(int(dimension) for dimension in right_shape)
    if not left or not right:
        raise ValueError("program AD product inner direct rule requires non-scalar operands")
    if any(dimension <= 0 for dimension in (*left, *right)):
        raise ValueError("program AD product inner direct rule dimensions must be positive")
    if left[-1] != right[-1]:
        raise ValueError("program AD product inner direct rule last dimensions must align")
    return left, right, left[:-1] + right[:-1]


def _program_ad_product_static_split_pair(
    primitive_name: str,
    values: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD product {primitive_name} values", values).reshape(
        -1
    )
    left_size = _program_ad_shape_static_size(left_shape)
    right_size = _program_ad_shape_static_size(right_shape)
    if vector.size != left_size + right_size:
        raise ValueError(
            f"program AD product {primitive_name} direct rule requires flattened left "
            "operand followed by right operand"
        )
    return (
        vector[:left_size].reshape(left_shape),
        vector[left_size:].reshape(right_shape),
    )


def _normalise_program_ad_product_tensordot_axis_sequence(
    label: str,
    axes: object,
    rank: int,
) -> tuple[int, ...]:
    if isinstance(axes, (bool, np.bool_)):
        raise ValueError(f"program AD product tensordot {label} axes must be static integers")
    if isinstance(axes, (int, np.integer)):
        return (_normalise_axis(label, int(axes), rank),)
    if isinstance(axes, np.ndarray):
        raw_axes = tuple(axes.reshape(-1).tolist())
    elif isinstance(axes, Sequence) and not isinstance(axes, (str, bytes)):
        raw_axes = tuple(axes)
    else:
        raise ValueError(f"program AD product tensordot {label} axes must be static integers")
    normalised: list[int] = []
    for raw_axis in raw_axes:
        if isinstance(raw_axis, (bool, np.bool_)) or not isinstance(raw_axis, (int, np.integer)):
            raise ValueError(f"program AD product tensordot {label} axes must be static integers")
        normalised.append(_normalise_axis(label, int(raw_axis), rank))
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"program AD product tensordot {label} axes must be unique")
    return tuple(normalised)


def _normalise_program_ad_product_tensordot_shape(shape: Sequence[int]) -> tuple[int, ...]:
    static_shape = tuple(int(dimension) for dimension in shape)
    if any(dimension <= 0 for dimension in static_shape):
        raise ValueError("program AD product tensordot dimensions must be positive")
    return static_shape


def _normalise_program_ad_product_tensordot_signature(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
    axes: object,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    left = _normalise_program_ad_product_tensordot_shape(left_shape)
    right = _normalise_program_ad_product_tensordot_shape(right_shape)
    if isinstance(axes, (bool, np.bool_)):
        raise ValueError("program AD product tensordot axes must be a static integer or pair")
    if isinstance(axes, (int, np.integer)):
        axis_count = int(axes)
        if axis_count < 0:
            raise ValueError("program AD product tensordot axis count must be non-negative")
        if axis_count > min(len(left), len(right)):
            raise ValueError("program AD product tensordot axis count exceeds operand rank")
        left_axes = tuple(range(len(left) - axis_count, len(left)))
        right_axes = tuple(range(axis_count))
    elif isinstance(axes, Sequence) and not isinstance(axes, (str, bytes)) and len(axes) == 2:
        left_axes = _normalise_program_ad_product_tensordot_axis_sequence(
            "left",
            axes[0],
            len(left),
        )
        right_axes = _normalise_program_ad_product_tensordot_axis_sequence(
            "right",
            axes[1],
            len(right),
        )
    else:
        raise ValueError("program AD product tensordot axes must be a static integer or pair")
    if len(left_axes) != len(right_axes):
        raise ValueError("program AD product tensordot axis lists must have equal length")
    for left_axis, right_axis in zip(left_axes, right_axes, strict=True):
        if left[left_axis] != right[right_axis]:
            raise ValueError("program AD product tensordot contracted dimensions must align")
    left_free = tuple(axis for axis in range(len(left)) if axis not in left_axes)
    right_free = tuple(axis for axis in range(len(right)) if axis not in right_axes)
    output_shape = tuple(left[axis] for axis in left_free) + tuple(
        right[axis] for axis in right_free
    )
    return left, right, left_axes, right_axes, output_shape


def _program_ad_product_tensordot_axes_signature(
    left_axes: tuple[int, ...],
    right_axes: tuple[int, ...],
) -> str:
    left = "_".join(str(axis) for axis in left_axes) if left_axes else "none"
    right = "_".join(str(axis) for axis in right_axes) if right_axes else "none"
    return f"{left}_by_{right}"


def program_ad_product_tensordot_derivative_rule(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
    *,
    axes: object = 2,
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed ``np.tensordot`` signature."""

    left_static_shape, right_static_shape, left_axes, right_axes, output_shape = (
        _normalise_program_ad_product_tensordot_signature(left_shape, right_shape, axes)
    )
    output_size = _program_ad_shape_static_size(output_shape)
    normalised_axes = (left_axes, right_axes)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "tensordot", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        return _program_ad_float64_vector_result(np.tensordot(left, right, axes=normalised_axes))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "tensordot", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        tangent_left, tangent_right = _program_ad_product_static_split_pair(
            "tensordot tangent",
            tangent,
            left_shape=left_static_shape,
            right_shape=right_static_shape,
        )
        return _program_ad_float64_vector_result(
            np.tensordot(tangent_left, right, axes=normalised_axes)
            + np.tensordot(left, tangent_right, axes=normalised_axes)
        )

    def vjp_rule(
        values: NDArray[np.float64],
        cotangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "tensordot", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        cotangent_vector = _as_real_numeric_array(
            "program AD product tensordot cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != output_size:
            raise ValueError(
                "program AD product tensordot VJP cotangent size must match output shape"
            )
        left_adjoint = np.zeros_like(left, dtype=np.float64)
        right_adjoint = np.zeros_like(right, dtype=np.float64)
        for element_index in np.ndindex(left.shape):
            basis = np.zeros_like(left, dtype=np.float64)
            basis[element_index] = 1.0
            left_adjoint[element_index] = float(
                np.dot(
                    cotangent_vector,
                    _program_ad_float64_vector_result(
                        np.tensordot(basis, right, axes=normalised_axes)
                    ),
                )
            )
        for element_index in np.ndindex(right.shape):
            basis = np.zeros_like(right, dtype=np.float64)
            basis[element_index] = 1.0
            right_adjoint[element_index] = float(
                np.dot(
                    cotangent_vector,
                    _program_ad_float64_vector_result(
                        np.tensordot(left, basis, axes=normalised_axes)
                    ),
                )
            )
        return _program_ad_float64_vector_result(
            np.concatenate((left_adjoint.reshape(-1), right_adjoint.reshape(-1)))
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_product_tensordot_"
            f"{_program_ad_shape_signature(left_static_shape)}_by_"
            f"{_program_ad_shape_signature(right_static_shape)}_axes_"
            f"{_program_ad_product_tensordot_axes_signature(left_axes, right_axes)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_product_inner_derivative_rule(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed inner-product signature."""

    left_static_shape, right_static_shape, output_shape = (
        _program_ad_product_normalise_inner_shapes(left_shape, right_shape)
    )
    expected_output_size = _program_ad_shape_static_size(output_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "inner", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        return _program_ad_float64_vector_result(np.inner(left, right))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "inner", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        tangent_left, tangent_right = _program_ad_product_static_split_pair(
            "inner tangent",
            tangent,
            left_shape=left_static_shape,
            right_shape=right_static_shape,
        )
        return _program_ad_float64_vector_result(
            np.inner(tangent_left, right) + np.inner(left, tangent_right)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "inner", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        cotangent_vector = _as_real_numeric_array(
            "program AD product inner cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != expected_output_size:
            raise ValueError(
                "program AD product inner VJP cotangent shape must match output shape"
            )
        cotangent_array = (
            cotangent_vector.reshape(output_shape)
            if output_shape
            else np.asarray(cotangent_vector[0], dtype=np.float64)
        )
        left_outer_rank = left.ndim - 1
        right_outer_rank = right.ndim - 1
        left_adjoint = np.tensordot(
            cotangent_array,
            right,
            axes=(
                tuple(range(left_outer_rank, left_outer_rank + right_outer_rank)),
                tuple(range(right_outer_rank)),
            ),
        )
        right_adjoint = np.tensordot(
            cotangent_array,
            left,
            axes=(tuple(range(left_outer_rank)), tuple(range(left_outer_rank))),
        )
        return _program_ad_float64_vector_result(
            np.concatenate(
                (np.asarray(left_adjoint).reshape(-1), np.asarray(right_adjoint).reshape(-1))
            )
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_product_inner_"
            f"{_program_ad_shape_signature(left_static_shape)}_by_"
            f"{_program_ad_shape_signature(right_static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_product_normalise_outer_shapes(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    left = tuple(int(dimension) for dimension in left_shape)
    right = tuple(int(dimension) for dimension in right_shape)
    if any(dimension <= 0 for dimension in (*left, *right)):
        raise ValueError("program AD product outer direct rule dimensions must be positive")
    return left, right


def program_ad_product_outer_derivative_rule(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed outer-product signature."""

    left_static_shape, right_static_shape = _program_ad_product_normalise_outer_shapes(
        left_shape, right_shape
    )
    left_size = _program_ad_shape_static_size(left_static_shape)
    right_size = _program_ad_shape_static_size(right_static_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "outer", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        return _program_ad_float64_vector_result(np.outer(left.reshape(-1), right.reshape(-1)))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "outer", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        tangent_left, tangent_right = _program_ad_product_static_split_pair(
            "outer tangent",
            tangent,
            left_shape=left_static_shape,
            right_shape=right_static_shape,
        )
        return _program_ad_float64_vector_result(
            np.outer(tangent_left.reshape(-1), right.reshape(-1))
            + np.outer(left.reshape(-1), tangent_right.reshape(-1))
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "outer", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        cotangent_vector = _as_real_numeric_array(
            "program AD product outer cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != left_size * right_size:
            raise ValueError(
                "program AD product outer VJP cotangent shape must match output shape"
            )
        cotangent_matrix = cotangent_vector.reshape(left_size, right_size)
        left_adjoint = (cotangent_matrix @ right.reshape(-1)).reshape(left_static_shape)
        right_adjoint = (cotangent_matrix.T @ left.reshape(-1)).reshape(right_static_shape)
        return _program_ad_float64_vector_result(
            np.concatenate((left_adjoint.reshape(-1), right_adjoint.reshape(-1)))
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_product_outer_"
            f"{_program_ad_shape_signature(left_static_shape)}_by_"
            f"{_program_ad_shape_signature(right_static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _normalise_program_ad_product_einsum_shapes(
    operand_shapes: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    shapes = tuple(tuple(int(dimension) for dimension in shape) for shape in operand_shapes)
    if not shapes:
        raise ValueError("program AD product einsum derivative rule requires operands")
    if any(any(dimension <= 0 for dimension in shape) for shape in shapes):
        raise ValueError("program AD product einsum derivative rule dimensions must be positive")
    return shapes


def _split_program_ad_product_einsum_operands(
    name: str,
    values: NDArray[np.float64],
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[NDArray[np.float64], ...]:
    vector = _as_real_numeric_array(f"program AD product einsum {name}", values).reshape(-1)
    expected = sum(_program_ad_shape_static_size(shape) for shape in shapes)
    if vector.size != expected:
        raise ValueError("program AD product einsum direct rule values size must match shapes")
    operands: list[NDArray[np.float64]] = []
    offset = 0
    for shape in shapes:
        size = _program_ad_shape_static_size(shape)
        operands.append(vector[offset : offset + size].reshape(shape))
        offset += size
    return tuple(operands)


def program_ad_product_einsum_derivative_rule(
    subscripts: str,
    operand_shapes: Sequence[Sequence[int]],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for fixed explicit ``np.einsum`` signatures."""

    shapes = _normalise_program_ad_product_einsum_shapes(operand_shapes)
    normalised = subscripts.replace(" ", "")
    output_labels, input_labels, dimensions = _parse_static_einsum_subscripts(
        normalised,
        shapes,
    )
    output_shape = tuple(dimensions[label] for label in output_labels)
    output_size = _program_ad_shape_static_size(output_shape)

    def flat_einsum(operands: tuple[NDArray[np.float64], ...]) -> NDArray[np.float64]:
        result = np.einsum(normalised, *operands)
        return _program_ad_float64_vector_result(result)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        operands = _split_program_ad_product_einsum_operands("values", values, shapes)
        return flat_einsum(operands)

    def jvp_rule(
        values: NDArray[np.float64],
        tangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        operands = _split_program_ad_product_einsum_operands("values", values, shapes)
        tangent_operands = _split_program_ad_product_einsum_operands("tangent", tangent, shapes)
        total = np.zeros(output_size, dtype=np.float64)
        for operand_index, tangent_operand in enumerate(tangent_operands):
            varied = operands[:operand_index] + (tangent_operand,) + operands[operand_index + 1 :]
            total += flat_einsum(varied)
        return _program_ad_float64_vector_result(total)

    def vjp_rule(
        values: NDArray[np.float64],
        cotangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        operands = _split_program_ad_product_einsum_operands("values", values, shapes)
        cotangent_vector = _as_real_numeric_array(
            "program AD product einsum cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != output_size:
            raise ValueError(
                "program AD product einsum VJP cotangent size must match output shape"
            )
        adjoints: list[NDArray[np.float64]] = []
        for operand_index, operand in enumerate(operands):
            operand_adjoint = np.zeros_like(operand, dtype=np.float64)
            for element_index in np.ndindex(operand.shape):
                basis = np.zeros_like(operand, dtype=np.float64)
                basis[element_index] = 1.0
                varied = operands[:operand_index] + (basis,) + operands[operand_index + 1 :]
                operand_adjoint[element_index] = float(
                    np.dot(cotangent_vector, flat_einsum(varied))
                )
            adjoints.append(operand_adjoint.reshape(-1))
        return _program_ad_float64_vector_result(np.concatenate(adjoints))

    label_signature = "_".join(
        "".join(labels) for labels in (*input_labels, ("".join(output_labels),))
    )
    shape_signature = "_by_".join(_program_ad_shape_signature(shape) for shape in shapes)
    return CustomDerivativeRule(
        name=f"program_ad_product_einsum_{label_signature}_{shape_signature}_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_product_einsum_unconfigured_value(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    del values
    raise ValueError(
        "program AD product einsum direct rule requires fixed subscripts and operand shapes"
    )


def _program_ad_product_einsum_unconfigured_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    del values, tangent
    raise ValueError(
        "program AD product einsum JVP rule requires fixed subscripts and operand shapes"
    )


def _program_ad_product_einsum_unconfigured_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    del values, cotangent
    raise ValueError(
        "program AD product einsum VJP rule requires fixed subscripts and operand shapes"
    )


def _program_ad_product_tensordot_unconfigured_value(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    del values
    raise ValueError("program AD product tensordot direct rule requires fixed shapes and axes")


def _program_ad_product_tensordot_unconfigured_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    del values, tangent
    raise ValueError("program AD product tensordot JVP rule requires fixed shapes and axes")


def _program_ad_product_tensordot_unconfigured_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    del values, cotangent
    raise ValueError("program AD product tensordot VJP rule requires fixed shapes and axes")


def _program_ad_product_derivative_rule(name: str) -> CustomDerivativeRule:
    if name == "dot":
        return CustomDerivativeRule(
            name="program_ad_product_dot_direct_rule",
            value_fn=_program_ad_product_dot_value,
            jvp_rule=_program_ad_product_dot_jvp,
            vjp_rule=_program_ad_product_dot_vjp,
        )
    if name == "vdot":
        return CustomDerivativeRule(
            name="program_ad_product_vdot_direct_rule",
            value_fn=_program_ad_product_vdot_value,
            jvp_rule=_program_ad_product_vdot_jvp,
            vjp_rule=_program_ad_product_vdot_vjp,
        )
    if name == "inner":
        return CustomDerivativeRule(
            name="program_ad_product_inner_direct_rule",
            value_fn=_program_ad_product_inner_value,
            jvp_rule=_program_ad_product_inner_jvp,
            vjp_rule=_program_ad_product_inner_vjp,
        )
    if name == "outer":
        return CustomDerivativeRule(
            name="program_ad_product_outer_direct_rule",
            value_fn=_program_ad_product_outer_value,
            jvp_rule=_program_ad_product_outer_jvp,
            vjp_rule=_program_ad_product_outer_vjp,
        )
    if name == "matmul":
        return CustomDerivativeRule(
            name="program_ad_product_matmul_direct_rule",
            value_fn=_program_ad_product_matmul_value,
            jvp_rule=_program_ad_product_matmul_jvp,
            vjp_rule=_program_ad_product_matmul_vjp,
        )
    if name == "tensordot":
        return CustomDerivativeRule(
            name="program_ad_product_tensordot_static_signature_required_rule",
            value_fn=_program_ad_product_tensordot_unconfigured_value,
            jvp_rule=_program_ad_product_tensordot_unconfigured_jvp,
            vjp_rule=_program_ad_product_tensordot_unconfigured_vjp,
        )
    if name == "einsum":
        return CustomDerivativeRule(
            name="program_ad_product_einsum_static_signature_required_rule",
            value_fn=_program_ad_product_einsum_unconfigured_value,
            jvp_rule=_program_ad_product_einsum_unconfigured_jvp,
            vjp_rule=_program_ad_product_einsum_unconfigured_vjp,
        )
    raise ValueError(f"unsupported program AD product primitive {name}")


def _program_ad_array_shape_of(value: object) -> tuple[int, ...]:
    if isinstance(value, TraceADArray):
        return value.shape
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _program_ad_array_dtype_of(value: object) -> str:
    if isinstance(value, TraceADArray):
        return "float64"
    array = np.asarray(value)
    if array.dtype.kind in {"O", "S", "U", "c"}:
        raise ValueError("program AD array primitive dtype rule requires real numeric arrays")
    return str(array.dtype)


def _program_ad_array_getitem_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD array getitem shape rule requires array and index")
    _validate_trace_basic_index(args[1])
    source = np.arange(int(np.prod(_program_ad_array_shape_of(args[0]))), dtype=np.int64).reshape(
        _program_ad_array_shape_of(args[0])
    )
    try:
        selected = source[cast(Any, args[1])]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("program AD array getitem shape rule requires in-bounds indices") from exc
    return tuple(int(dimension) for dimension in np.asarray(selected).shape)


def _program_ad_array_take_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {2, 3, 4}:
        raise ValueError(
            "program AD array take shape rule requires array, indices, axis, and mode"
        )
    indices = args[1]
    axis = cast(int | None, args[2]) if len(args) >= 3 else None
    mode = cast(str, args[3]) if len(args) == 4 else "raise"
    mode_name = _program_ad_array_take_mode(mode, context="shape rule")
    raw_indices = np.asarray(indices)
    if raw_indices.dtype.kind not in {"i", "u"}:
        raise ValueError("program AD array take shape rule requires static integer indices")
    source = np.arange(int(np.prod(_program_ad_array_shape_of(args[0]))), dtype=np.int64).reshape(
        _program_ad_array_shape_of(args[0])
    )
    try:
        selected = np.take(source, raw_indices, axis=axis, mode=mode_name)
    except (IndexError, ValueError) as exc:
        if mode_name == "raise":
            raise ValueError("program AD array take shape rule indices must be in bounds") from exc
        raise ValueError(
            "program AD array take shape rule requires axis-compatible indices"
        ) from exc
    return tuple(int(dimension) for dimension in np.asarray(selected).shape)


def _program_ad_array_take_along_axis_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 3:
        raise ValueError(
            "program AD array take_along_axis shape rule requires array, indices, and axis"
        )
    axis = args[2]
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError(
            "program AD array take_along_axis shape rule requires static integer axis"
        )
    raw_indices = np.asarray(args[1])
    if raw_indices.dtype.kind not in {"i", "u"}:
        raise ValueError(
            "program AD array take_along_axis shape rule requires static integer indices"
        )
    source = np.arange(int(np.prod(_program_ad_array_shape_of(args[0]))), dtype=np.int64).reshape(
        _program_ad_array_shape_of(args[0])
    )
    try:
        selected = np.take_along_axis(source, raw_indices, axis=int(axis))
    except (IndexError, ValueError) as exc:
        raise ValueError(
            "program AD array take_along_axis shape rule indices must be in bounds "
            "and shape-compatible"
        ) from exc
    return tuple(int(dimension) for dimension in np.asarray(selected).shape)


def _program_ad_array_delete_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {2, 3}:
        raise ValueError("program AD array delete shape rule requires array, object, and axis")
    source_shape = _program_ad_array_shape_of(args[0])
    delete_obj = _program_ad_array_delete_object(args[1], context="shape rule")
    axis = args[2] if len(args) == 3 else None
    source: NDArray[np.int64]
    if axis is None:
        source = np.arange(int(np.prod(source_shape)), dtype=np.int64).reshape(-1)
        normalised_axis = None
    else:
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
            raise ValueError("program AD array delete shape rule requires static integer axis")
        normalised_axis = _normalise_axis("axis", int(axis), len(source_shape))
        source = np.arange(int(np.prod(source_shape)), dtype=np.int64).reshape(source_shape)
    try:
        selected = np.delete(source, cast(Any, delete_obj), axis=normalised_axis)
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "program AD array delete shape rule requires static in-bounds deletion selectors"
        ) from exc
    return tuple(int(dimension) for dimension in np.asarray(selected).shape)


def _program_ad_array_pad_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {2, 3, 4}:
        raise ValueError(
            "program AD array pad shape rule requires array, pad_width, mode, and constants"
        )
    mode = args[2] if len(args) >= 3 else "constant"
    _program_ad_array_pad_mode(mode, context="shape rule")
    _, _, output_shape = _program_ad_array_pad_layout(
        _program_ad_array_shape_of(args[0]),
        args[1],
        args[3] if len(args) == 4 else 0.0,
        context="shape rule",
    )
    return output_shape


def _program_ad_array_insert_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {3, 4}:
        raise ValueError(
            "program AD array insert shape rule requires array, object, values, and axis"
        )
    _, _, output_shape = _program_ad_array_insert_layout(
        _program_ad_array_shape_of(args[0]),
        args[1],
        args[2],
        args[3] if len(args) == 4 else None,
        context="shape rule",
    )
    return output_shape


def _program_ad_array_dtype_rule(args: tuple[object, ...]) -> str:
    if not args:
        raise ValueError("program AD array dtype rule requires an array operand")
    return _program_ad_array_dtype_of(args[0])


def _program_ad_shape_reshape_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD shape reshape rule requires array and target shape")
    source_shape = _program_ad_array_shape_of(args[0])
    return _normalise_trace_reshape_shape(args[1], int(np.prod(source_shape)))


def _program_ad_shape_expand_dims_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD shape expand_dims rule requires array and axis")
    source_shape = _program_ad_array_shape_of(args[0])
    axes = _program_ad_shape_normalise_expand_dims_axes(source_shape, cast(Any, args[1]))
    return _program_ad_shape_insert_singleton_axes(source_shape, axes)


def _program_ad_shape_ravel_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD shape ravel rule requires one array")
    return (int(np.prod(_program_ad_array_shape_of(args[0]))),)


def _program_ad_shape_normalised_transpose_axes(
    array_shape: tuple[int, ...],
    axes: object,
) -> tuple[int, ...]:
    if len(array_shape) < 2:
        return ()
    if axes is None:
        return tuple(reversed(range(len(array_shape))))
    if not isinstance(axes, Sequence) or isinstance(axes, (str, bytes)):
        raise ValueError("program AD shape transpose axes must be a static axis sequence")
    raw_axes = tuple(cast(Any, axes))
    if len(raw_axes) != len(array_shape):
        raise ValueError("program AD shape transpose axes must match array rank")
    normalised_axes = tuple(
        _normalise_axis("axis", cast(int, axis), len(array_shape)) for axis in raw_axes
    )
    if sorted(normalised_axes) != list(range(len(array_shape))):
        raise ValueError("program AD shape transpose axes must be a permutation")
    return normalised_axes


def _program_ad_shape_transpose_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD shape transpose rule requires array and optional axes")
    source_shape = _program_ad_array_shape_of(args[0])
    axes = _program_ad_shape_normalised_transpose_axes(
        source_shape, args[1] if len(args) == 2 else None
    )
    if not axes:
        return source_shape
    return tuple(source_shape[axis] for axis in axes)


def _program_ad_shape_atleast_rank_shape(
    args: tuple[object, ...], *, rank: Literal[1, 2, 3]
) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError(f"program AD shape atleast_{rank}d rule requires one array")
    return _program_ad_shape_atleast_target_shape(_program_ad_array_shape_of(args[0]), rank)


def _program_ad_shape_atleast_1d_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    return _program_ad_shape_atleast_rank_shape(args, rank=1)


def _program_ad_shape_atleast_2d_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    return _program_ad_shape_atleast_rank_shape(args, rank=2)


def _program_ad_shape_atleast_3d_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    return _program_ad_shape_atleast_rank_shape(args, rank=3)


def _program_ad_shape_swapaxes_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 3:
        raise ValueError("program AD shape swapaxes rule requires array, axis1, and axis2")
    source_shape = _program_ad_array_shape_of(args[0])
    first = _normalise_axis_permutation_axis(
        "swapaxes", cast(int, args[1]), rank=len(source_shape)
    )
    second = _normalise_axis_permutation_axis(
        "swapaxes", cast(int, args[2]), rank=len(source_shape)
    )
    target = list(source_shape)
    target[first], target[second] = target[second], target[first]
    return tuple(target)


def _program_ad_shape_moveaxis_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 3:
        raise ValueError("program AD shape moveaxis rule requires array, source, and destination")
    source_shape = _program_ad_array_shape_of(args[0])
    _, _, order = _program_ad_shape_normalise_moveaxis_axes(
        source_shape, cast(Any, args[1]), cast(Any, args[2])
    )
    return tuple(source_shape[axis] for axis in order)


def _program_ad_shape_roll_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {2, 3}:
        raise ValueError("program AD shape roll rule requires array, shift, and optional axis")
    source_shape = _program_ad_array_shape_of(args[0])
    _program_ad_shape_normalise_roll_signature(
        source_shape, args[1], args[2] if len(args) == 3 else None
    )
    return source_shape


def _program_ad_shape_flip_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD shape flip rule requires array and optional axis")
    source_shape = _program_ad_array_shape_of(args[0])
    _program_ad_shape_normalise_flip_axis(source_shape, args[1] if len(args) == 2 else None)
    return source_shape


def _program_ad_shape_flipud_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD shape flipud rule requires one array")
    source_shape = _program_ad_array_shape_of(args[0])
    if len(source_shape) < 1:
        raise ValueError("program AD flipud requires at least rank-1 arrays")
    return source_shape


def _program_ad_shape_fliplr_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD shape fliplr rule requires one array")
    source_shape = _program_ad_array_shape_of(args[0])
    if len(source_shape) < 2:
        raise ValueError("program AD fliplr requires at least rank-2 arrays")
    return source_shape


def _program_ad_shape_rot90_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD shape rot90 rule requires array, k, and axes")
    source_shape = _program_ad_array_shape_of(args[0])
    k_value = _normalise_rot90_k(args[1] if len(args) >= 2 else 1)
    axes_value = _normalise_rot90_axes(
        args[2] if len(args) == 3 else (0, 1), rank=len(source_shape)
    )
    return _program_ad_shape_rot90_target_shape(source_shape, k_value, axes_value)


def _program_ad_shape_repeat_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {2, 3}:
        raise ValueError("program AD shape repeat rule requires array, repeats, and optional axis")
    source_shape = _program_ad_array_shape_of(args[0])
    _, _, target_shape = _program_ad_shape_normalise_repeat_signature(
        source_shape, args[1], args[2] if len(args) == 3 else None
    )
    return target_shape


def _program_ad_shape_tile_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD shape tile rule requires array and reps")
    source_shape = _program_ad_array_shape_of(args[0])
    _, _, target_shape = _program_ad_shape_normalise_tile_signature(source_shape, args[1])
    return target_shape


def _program_ad_shape_squeeze_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD shape squeeze rule requires array and optional axis")
    source_shape = _program_ad_array_shape_of(args[0])
    axes = _program_ad_shape_normalise_squeeze_axes(
        source_shape, cast(Any, args[1]) if len(args) == 2 else None
    )
    return _program_ad_shape_remove_axes(source_shape, axes)


def _program_ad_shape_dtype_rule(args: tuple[object, ...]) -> str:
    if not args:
        raise ValueError("program AD shape dtype rule requires an array operand")
    return _program_ad_array_dtype_of(args[0])


def _program_ad_reduction_axis(args: tuple[object, ...]) -> int | None:
    if len(args) not in {1, 2}:
        raise ValueError("program AD reduction rule requires array and optional axis")
    if len(args) == 1 or args[1] is None:
        return None
    axis = args[1]
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD reduction axis must be a static integer or None")
    return int(axis)


def _program_ad_reduction_axis_ddof(args: tuple[object, ...]) -> tuple[int | None, int]:
    if len(args) != 3:
        raise ValueError("program AD variance/std rule requires array, axis, and ddof")
    source_shape = _program_ad_array_shape_of(args[0])
    raw_axis = args[1]
    if raw_axis is None:
        axis = None
    elif isinstance(raw_axis, bool) or not isinstance(raw_axis, (int, np.integer)):
        raise ValueError("program AD variance/std axis must be a static integer or None")
    else:
        axis = _normalise_axis("axis", int(raw_axis), len(source_shape))
    count = int(np.prod(source_shape)) if axis is None else source_shape[axis]
    return axis, _normalise_ddof(args[2], count)


def _program_ad_order_statistic_reduction_axis(args: tuple[object, ...]) -> int | None:
    if len(args) == 2:
        axis = args[1]
    elif len(args) == 4:
        _normalise_order_statistic_method(args[3])
        axis = args[2]
    else:
        raise ValueError("program AD order-statistic reduction rule requires static arguments")
    return _normalise_order_statistic_axis(axis, len(_program_ad_array_shape_of(args[0])))


def _program_ad_reduction_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape = _program_ad_array_shape_of(args[0])
    if int(np.prod(source_shape)) == 0:
        raise ValueError("program AD reduction shape rule requires at least one element")
    axis = _program_ad_reduction_axis(args)
    if axis is None:
        return ()
    normalised_axis = _normalise_axis("axis", axis, len(source_shape))
    return source_shape[:normalised_axis] + source_shape[normalised_axis + 1 :]


def _program_ad_reduction_var_std_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape = _program_ad_array_shape_of(args[0])
    if int(np.prod(source_shape)) == 0:
        raise ValueError("program AD variance/std shape rule requires at least one element")
    axis, _ddof = _program_ad_reduction_axis_ddof(args)
    if axis is None:
        return ()
    return source_shape[:axis] + source_shape[axis + 1 :]


def _program_ad_order_statistic_reduction_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape = _program_ad_array_shape_of(args[0])
    if int(np.prod(source_shape)) == 0:
        raise ValueError(
            "program AD order-statistic reduction shape rule requires at least one element"
        )
    axis = _program_ad_order_statistic_reduction_axis(args)
    if axis is None:
        return ()
    return source_shape[:axis] + source_shape[axis + 1 :]


def _program_ad_reduction_trapezoid_axis(args: tuple[object, ...]) -> int:
    if len(args) != 4:
        raise ValueError("program AD trapezoid rule requires y, x, dx, and axis")
    return _normalise_trapezoid_axis(args[3], len(_program_ad_array_shape_of(args[0])))


def _program_ad_reduction_trapezoid_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape = _program_ad_array_shape_of(args[0])
    axis = _program_ad_reduction_trapezoid_axis(args)
    _program_ad_reduction_trapezoid_static_widths(source_shape, x=args[1], dx=args[2], axis=axis)
    return source_shape[:axis] + source_shape[axis + 1 :]


def _program_ad_reduction_dtype_rule(args: tuple[object, ...]) -> str:
    if not args:
        raise ValueError("program AD reduction dtype rule requires an array operand")
    return _program_ad_array_dtype_of(args[0])


def _program_ad_elementwise_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) == 1:
        return _program_ad_array_shape_of(args[0])
    if len(args) == 2:
        return _broadcast_shape(
            _program_ad_array_shape_of(args[0]), _program_ad_array_shape_of(args[1])
        )
    raise ValueError("program AD elementwise shape rule requires one or two operands")


def _program_ad_elementwise_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) not in {1, 2}:
        raise ValueError("program AD elementwise dtype rule requires one or two operands")
    dtypes = tuple(np.dtype(_program_ad_array_dtype_of(arg)) for arg in args)
    return str(np.result_type(*dtypes))


def _program_ad_selection_condition_shape(condition: object) -> tuple[int, ...]:
    if isinstance(condition, _TracePredicate):
        return ()
    if isinstance(condition, TraceADPredicateArray):
        return condition.shape
    raw = np.asarray(condition)
    if raw.dtype.kind != "b":
        raise ValueError("program AD selection condition must be boolean")
    return tuple(int(dimension) for dimension in raw.shape)


def _program_ad_selection_where_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 3:
        raise ValueError("program AD selection where shape rule requires condition, true, false")
    output_shape = _broadcast_shape(
        _program_ad_array_shape_of(args[1]),
        _program_ad_array_shape_of(args[2]),
    )
    condition_shape = _program_ad_selection_condition_shape(args[0])
    if condition_shape not in {(), output_shape}:
        raise ValueError(
            "program AD selection where condition shape must be scalar or output-shaped"
        )
    return output_shape


def _program_ad_selection_clip_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 3:
        raise ValueError("program AD selection clip shape rule requires source, lower, and upper")
    source_shape = _program_ad_array_shape_of(args[0])
    try:
        lower_shape = np.broadcast_shapes(source_shape, _program_ad_array_shape_of(args[1]))
        upper_shape = np.broadcast_shapes(source_shape, _program_ad_array_shape_of(args[2]))
    except ValueError as exc:
        raise ValueError("program AD selection clip bounds must broadcast to source") from exc
    if tuple(lower_shape) != source_shape or tuple(upper_shape) != source_shape:
        raise ValueError("program AD selection clip bounds must broadcast to source")
    return source_shape


def _program_ad_selection_sort_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD selection sort shape rule requires source, axis, and kind")
    source_shape = _program_ad_array_shape_of(args[0])
    axis = args[1] if len(args) >= 2 else -1
    if axis is None:
        return (int(np.prod(source_shape)),)
    _normalise_sort_axis(axis, len(source_shape))
    return source_shape


def _program_ad_selection_index_reduce_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD index selection shape rule requires source and optional axis")
    source_shape = _program_ad_array_shape_of(args[0])
    axis = args[1] if len(args) == 2 else None
    if axis is None:
        return ()
    axis_index = _normalise_sort_axis(axis, len(source_shape))
    return tuple(dimension for index, dimension in enumerate(source_shape) if index != axis_index)


def _program_ad_selection_argsort_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD argsort shape rule requires source, axis, and kind")
    source_shape = _program_ad_array_shape_of(args[0])
    axis = args[1] if len(args) >= 2 else -1
    kind = args[2] if len(args) == 3 else None
    if kind not in {None, "quicksort", "mergesort", "heapsort", "stable"}:
        raise ValueError("program AD argsort shape rule requires a NumPy sort kind")
    if axis is None:
        return (int(np.prod(source_shape)),)
    _normalise_sort_axis(axis, len(source_shape))
    return source_shape


def _program_ad_selection_sequence(name: str, value: object, role: str) -> tuple[object, ...]:
    if isinstance(value, (TraceADArray, np.ndarray)) or not isinstance(value, Sequence):
        raise ValueError(f"program AD {name} requires a static {role} sequence")
    return tuple(value)


def _program_ad_selection_select_parts(
    args: tuple[object, ...],
) -> tuple[tuple[object, ...], tuple[object, ...], object]:
    if len(args) != 3:
        raise ValueError("program AD select contract requires conditions, choices, and default")
    conditions = _program_ad_selection_sequence("select", args[0], "condition")
    choices = _program_ad_selection_sequence("select", args[1], "choice")
    if len(conditions) != len(choices):
        raise ValueError("program AD select requires matching condition and choice counts")
    return conditions, choices, args[2]


def _program_ad_selection_select_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    conditions, choices, default = _program_ad_selection_select_parts(args)
    output_shape = _program_ad_array_shape_of(default)
    for condition, choice in reversed(tuple(zip(conditions, choices, strict=True))):
        choice_shape = _program_ad_array_shape_of(choice)
        try:
            output_shape = tuple(
                int(dim) for dim in np.broadcast_shapes(choice_shape, output_shape)
            )
        except ValueError as exc:
            raise ValueError("program AD select choices must broadcast with default") from exc
        condition_shape = _program_ad_selection_condition_shape(condition)
        if condition_shape not in {(), output_shape}:
            raise ValueError("program AD select condition shape must be scalar or output-shaped")
    return output_shape


def _program_ad_selection_piecewise_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], tuple[object, ...], tuple[object, ...]]:
    if len(args) != 3:
        raise ValueError("program AD piecewise contract requires source, conditions, functions")
    source_shape = _program_ad_array_shape_of(args[0])
    conditions = _program_ad_selection_sequence("piecewise", args[1], "condition")
    functions = _program_ad_selection_sequence("piecewise", args[2], "function")
    if len(functions) not in {len(conditions), len(conditions) + 1}:
        raise ValueError(
            "program AD piecewise requires one function per condition and optional default"
        )
    for condition in conditions:
        condition_shape = _program_ad_selection_condition_shape(condition)
        if condition_shape not in {(), source_shape}:
            raise ValueError(
                "program AD piecewise condition shape must be scalar or source-shaped"
            )
    return source_shape, conditions, functions


def _program_ad_selection_piecewise_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape, _conditions, _functions = _program_ad_selection_piecewise_parts(args)
    return source_shape


def _program_ad_selection_choose_parts(
    args: tuple[object, ...],
) -> tuple[NDArray[np.int64], tuple[tuple[int, ...], ...], str]:
    if len(args) != 3:
        raise ValueError("program AD choose contract requires selector, choices, and mode")
    raw_choices = args[1]
    if isinstance(raw_choices, TraceADArray):
        raise ValueError("program AD choose requires a static choice sequence")
    if isinstance(raw_choices, (np.ndarray, Sequence)):
        choices = tuple(raw_choices)
    else:
        raise ValueError("program AD choose requires a static choice sequence")
    if not choices:
        raise ValueError("program AD choose requires at least one choice")
    mode = args[2]
    if not isinstance(mode, str):
        raise ValueError("program AD choose mode must be raise, wrap, or clip")
    selector = _trace_choose_selector_indices(args[0], choice_count=len(choices), mode=mode)
    choice_shapes = tuple(_program_ad_array_shape_of(choice) for choice in choices)
    return selector, choice_shapes, mode


def _program_ad_selection_choose_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    selector, choice_shapes, _mode = _program_ad_selection_choose_parts(args)
    try:
        return tuple(
            int(dim) for dim in np.broadcast_shapes(tuple(selector.shape), *choice_shapes)
        )
    except ValueError as exc:
        raise ValueError(
            "program AD choose selector and choices must be broadcast-compatible"
        ) from exc


def _program_ad_selection_compress_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], NDArray[np.int64], int | None]:
    if len(args) != 3:
        raise ValueError("program AD compress contract requires condition, array, and axis")
    source_shape = _program_ad_array_shape_of(args[1])
    indices = _trace_compress_condition_indices(args[0])
    axis_arg = args[2]
    if axis_arg is None:
        source_size = _program_ad_array_static_size(source_shape)
        if bool(np.any(indices >= source_size)):
            raise ValueError("program AD compress condition length exceeds flattened array")
        return source_shape, indices, None
    if isinstance(axis_arg, (bool, np.bool_)) or not isinstance(axis_arg, (int, np.integer)):
        raise ValueError("program AD compress requires a static integer axis or None")
    axis = _normalise_axis("axis", int(axis_arg), len(source_shape))
    if bool(np.any(indices >= source_shape[axis])):
        raise ValueError("program AD compress condition length exceeds selected axis")
    return source_shape, indices, axis


def _program_ad_selection_compress_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape, indices, axis = _program_ad_selection_compress_parts(args)
    if axis is None:
        return (int(indices.size),)
    result_shape = list(source_shape)
    result_shape[axis] = int(indices.size)
    return tuple(result_shape)


def _program_ad_selection_extract_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD extract contract requires condition and array")
    source_shape = _program_ad_array_shape_of(args[1])
    indices = _trace_extract_condition_indices(
        args[0], _program_ad_array_static_size(source_shape)
    )
    return (int(indices.size),)


def _program_ad_selection_where_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) != 3:
        raise ValueError("program AD selection where dtype rule requires condition, true, false")
    dtypes = tuple(np.dtype(_program_ad_array_dtype_of(arg)) for arg in args[1:])
    return str(np.result_type(*dtypes))


def _program_ad_selection_clip_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) != 3:
        raise ValueError("program AD selection clip dtype rule requires source, lower, and upper")
    dtypes = tuple(np.dtype(_program_ad_array_dtype_of(arg)) for arg in args)
    return str(np.result_type(*dtypes))


def _program_ad_selection_sort_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD selection sort dtype rule requires source, axis, and kind")
    return _program_ad_array_dtype_of(args[0])


def _program_ad_selection_index_dtype_rule(args: tuple[object, ...]) -> str:
    if not args:
        raise ValueError("program AD index selection dtype rule requires a source operand")
    return "int64"


def _program_ad_selection_select_dtype_rule(args: tuple[object, ...]) -> str:
    _conditions, choices, default = _program_ad_selection_select_parts(args)
    dtypes = [np.dtype(_program_ad_array_dtype_of(choice)) for choice in choices]
    dtypes.append(np.dtype(_program_ad_array_dtype_of(default)))
    return str(np.result_type(*dtypes))


def _program_ad_selection_piecewise_dtype_rule(args: tuple[object, ...]) -> str:
    _source_shape, _conditions, _functions = _program_ad_selection_piecewise_parts(args)
    return str(np.dtype(_program_ad_array_dtype_of(args[0])))


def _program_ad_selection_choose_dtype_rule(args: tuple[object, ...]) -> str:
    _selector, choice_shapes, _mode = _program_ad_selection_choose_parts(args)
    if not choice_shapes:
        raise ValueError("program AD choose requires at least one choice")
    raw_choices = tuple(cast(Sequence[object], args[1]))
    return str(
        np.result_type(*(np.dtype(_program_ad_array_dtype_of(choice)) for choice in raw_choices))
    )


def _program_ad_selection_compress_dtype_rule(args: tuple[object, ...]) -> str:
    _program_ad_selection_compress_parts(args)
    return str(np.dtype(_program_ad_array_dtype_of(args[1])))


def _program_ad_selection_extract_dtype_rule(args: tuple[object, ...]) -> str:
    _program_ad_selection_extract_shape(args)
    return str(np.dtype(_program_ad_array_dtype_of(args[1])))


def _program_ad_product_matmul_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD product matmul shape rule requires two operands")
    lhs_shape = _program_ad_array_shape_of(args[0])
    rhs_shape = _program_ad_array_shape_of(args[1])
    if len(lhs_shape) == 1 and len(rhs_shape) == 1:
        if lhs_shape != rhs_shape:
            raise ValueError("program AD product vector dimensions must align")
        return ()
    if len(lhs_shape) == 2 and len(rhs_shape) == 1:
        if lhs_shape[1] != rhs_shape[0]:
            raise ValueError("program AD product matrix-vector dimensions must align")
        return (lhs_shape[0],)
    if len(lhs_shape) == 1 and len(rhs_shape) == 2:
        if lhs_shape[0] != rhs_shape[0]:
            raise ValueError("program AD product vector-matrix dimensions must align")
        return (rhs_shape[1],)
    if len(lhs_shape) == 2 and len(rhs_shape) == 2:
        if lhs_shape[1] != rhs_shape[0]:
            raise ValueError("program AD product matrix-matrix dimensions must align")
        return (lhs_shape[0], rhs_shape[1])
    raise ValueError("program AD product matmul supports rank-1 and rank-2 operands")


def _program_ad_product_dot_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    shape = _program_ad_product_matmul_shape(args)
    if shape != ():
        raise ValueError("program AD product dot contract supports scalar dot results only")
    return shape


def _program_ad_product_vdot_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD product vdot shape rule requires two operands")
    lhs_size = int(np.prod(_program_ad_array_shape_of(args[0])))
    rhs_size = int(np.prod(_program_ad_array_shape_of(args[1])))
    if lhs_size != rhs_size:
        raise ValueError("program AD np.vdot flattened operands must have matching size")
    return ()


def _program_ad_product_inner_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD product inner shape rule requires two operands")
    lhs_shape = _program_ad_array_shape_of(args[0])
    rhs_shape = _program_ad_array_shape_of(args[1])
    if not lhs_shape or not rhs_shape:
        raise ValueError("program AD product inner shape rule requires non-scalar operands")
    if lhs_shape[-1] != rhs_shape[-1]:
        raise ValueError("program AD product inner last dimensions must align")
    return lhs_shape[:-1] + rhs_shape[:-1]


def _program_ad_product_outer_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD product outer shape rule requires two operands")
    lhs_size = int(np.prod(_program_ad_array_shape_of(args[0])))
    rhs_size = int(np.prod(_program_ad_array_shape_of(args[1])))
    if lhs_size <= 0 or rhs_size <= 0:
        raise ValueError("program AD product outer shape rule requires non-empty operands")
    return (lhs_size, rhs_size)


def _program_ad_product_einsum_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) < 2:
        raise ValueError("program AD product einsum shape rule requires subscripts and operands")
    if not isinstance(args[0], str):
        raise ValueError("program AD product einsum shape rule requires static subscripts")
    operand_shapes = tuple(_program_ad_array_shape_of(arg) for arg in args[1:])
    output_labels, _input_labels, dimensions = _parse_static_einsum_subscripts(
        args[0],
        operand_shapes,
    )
    return tuple(dimensions[label] for label in output_labels)


def _program_ad_product_tensordot_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 3:
        raise ValueError("program AD product tensordot shape rule requires two operands and axes")
    _left, _right, _left_axes, _right_axes, output_shape = (
        _normalise_program_ad_product_tensordot_signature(
            _program_ad_array_shape_of(args[0]),
            _program_ad_array_shape_of(args[1]),
            args[2],
        )
    )
    return output_shape


def _program_ad_product_dtype_rule(args: tuple[object, ...]) -> str:
    if args and isinstance(args[0], str):
        product_args = args[1:]
    elif len(args) == 3:
        product_args = args[:2]
    else:
        product_args = args
    if not product_args:
        raise ValueError("program AD product dtype rule requires operands")
    dtypes = tuple(np.dtype(_program_ad_array_dtype_of(arg)) for arg in product_args)
    return str(np.result_type(*dtypes))


def _program_ad_cumulative_axis(args: tuple[object, ...]) -> int | None:
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD cumulative rule requires array and static parameters")
    if len(args) == 1 or args[1] is None:
        return None
    axis = args[1]
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD cumulative axis must be a static integer or None")
    return int(axis)


def _program_ad_cumulative_diff_order(args: tuple[object, ...]) -> int:
    if len(args) < 2:
        return 1
    order = args[1]
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise ValueError("program AD np.diff requires non-negative integer n")
    order = int(order)
    if order < 0:
        raise ValueError("program AD np.diff requires non-negative integer n")
    return order


def _program_ad_cumulative_scan_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD cumulative scan shape rule requires array and axis")
    source_shape = _program_ad_array_shape_of(args[0])
    if int(np.prod(source_shape)) == 0:
        raise ValueError("program AD cumulative scan requires at least one element")
    axis = _program_ad_cumulative_axis(args)
    return (int(np.prod(source_shape)),) if axis is None else source_shape


def _program_ad_cumulative_diff_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD diff shape rule requires array, order, and axis")
    source_shape = _program_ad_array_shape_of(args[0])
    order = _program_ad_cumulative_diff_order(args)
    axis = args[2] if len(args) == 3 else -1
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD diff axis must be a static integer")
    axis_index = _normalise_axis("axis", int(axis), len(source_shape))
    axis_size = max(source_shape[axis_index] - order, 0)
    return source_shape[:axis_index] + (axis_size,) + source_shape[axis_index + 1 :]


def _program_ad_cumulative_dtype_rule(args: tuple[object, ...]) -> str:
    if not args:
        raise ValueError("program AD cumulative dtype rule requires an array operand")
    return _program_ad_array_dtype_of(args[0])


def _program_ad_array_getitem_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 2:
        raise ValueError("program AD array getitem static rule requires array and index")
    _validate_trace_basic_index(args[1])
    index = args[1]
    if isinstance(index, tuple):
        return (tuple(_program_ad_array_static_index_component(item) for item in index),)
    return (_program_ad_array_static_index_component(index),)


def _program_ad_array_static_index_component(selector: object) -> object:
    if selector is Ellipsis or selector is None:
        return selector
    if isinstance(selector, (int, np.integer)) and not isinstance(selector, (bool, np.bool_)):
        return int(selector)
    if isinstance(selector, slice):
        return slice(
            None if selector.start is None else int(selector.start),
            None if selector.stop is None else int(selector.stop),
            None if selector.step is None else int(selector.step),
        )
    if isinstance(selector, (np.ndarray, list)):
        array = _trace_static_index_array(selector)
        dtype_name = "bool" if array.dtype.kind == "b" else "int64"
        values = tuple(
            bool(item) if array.dtype.kind == "b" else int(item) for item in array.reshape(-1)
        )
        return (
            "static_index_array",
            dtype_name,
            tuple(int(dimension) for dimension in array.shape),
            values,
        )
    raise ValueError(_PROGRAM_AD_STATIC_INDEX_ERROR)


def _program_ad_array_take_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {2, 3, 4}:
        raise ValueError(
            "program AD array take static rule requires array, indices, axis, and mode"
        )
    raw_indices = np.asarray(args[1])
    if raw_indices.dtype.kind not in {"i", "u"}:
        raise ValueError("program AD array take static rule requires static integer indices")
    axis = cast(int | None, args[2]) if len(args) >= 3 else None
    if axis is not None and (isinstance(axis, bool) or not isinstance(axis, (int, np.integer))):
        raise ValueError("program AD array take static rule requires static integer axis")
    mode = cast(str, args[3]) if len(args) == 4 else "raise"
    mode_name = _program_ad_array_take_mode(mode, context="static rule")
    return (
        tuple(int(index) for index in raw_indices.reshape(-1)),
        None if axis is None else int(axis),
        mode_name,
    )


def _program_ad_array_take_along_axis_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    if len(args) != 3:
        raise ValueError(
            "program AD array take_along_axis static rule requires array, indices, and axis"
        )
    raw_indices = np.asarray(args[1])
    if raw_indices.dtype.kind not in {"i", "u"}:
        raise ValueError(
            "program AD array take_along_axis static rule requires static integer indices"
        )
    axis = args[2]
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError(
            "program AD array take_along_axis static rule requires static integer axis"
        )
    normalised_axis = _normalise_axis("axis", int(axis), len(_program_ad_array_shape_of(args[0])))
    return (
        tuple(int(index) for index in raw_indices.reshape(-1)),
        tuple(int(dimension) for dimension in raw_indices.shape),
        normalised_axis,
    )


def _program_ad_array_delete_static_object(obj: object) -> object:
    delete_obj = _program_ad_array_delete_object(obj, context="static rule")
    if isinstance(delete_obj, int):
        return delete_obj
    if isinstance(delete_obj, slice):
        return delete_obj
    delete_array = np.asarray(delete_obj)
    if delete_array.dtype.kind == "b":
        return (
            "static_delete_mask",
            tuple(int(dimension) for dimension in delete_array.shape),
            tuple(bool(item) for item in delete_array.reshape(-1)),
        )
    return tuple(int(index) for index in delete_array.reshape(-1))


def _program_ad_array_delete_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {2, 3}:
        raise ValueError("program AD array delete static rule requires array, object, and axis")
    axis = args[2] if len(args) == 3 else None
    if axis is None:
        normalised_axis = None
    else:
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
            raise ValueError("program AD array delete static rule requires static integer axis")
        normalised_axis = _normalise_axis(
            "axis", int(axis), len(_program_ad_array_shape_of(args[0]))
        )
    return (_program_ad_array_delete_static_object(args[1]), normalised_axis)


def _program_ad_array_pad_static_constants(value: object) -> object:
    constants = _program_ad_array_pad_constant_values(value, context="static rule")
    constant_array = np.asarray(constants, dtype=np.float64)
    if constant_array.shape == ():
        return float(constant_array)
    return (
        "static_pad_constants",
        tuple(int(dimension) for dimension in constant_array.shape),
        tuple(float(item) for item in constant_array.reshape(-1)),
    )


def _program_ad_array_pad_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {2, 3, 4}:
        raise ValueError(
            "program AD array pad static rule requires array, pad_width, mode, and constants"
        )
    mode = args[2] if len(args) >= 3 else "constant"
    mode_name = _program_ad_array_pad_mode(mode, context="static rule")
    pad_width = _program_ad_array_pad_width(
        args[1],
        len(_program_ad_array_shape_of(args[0])),
        context="static rule",
    )
    return (
        pad_width,
        mode_name,
        _program_ad_array_pad_static_constants(args[3] if len(args) == 4 else 0.0),
    )


def _program_ad_array_insert_static_object(obj: object) -> object:
    insert_obj = _program_ad_array_insert_object(obj, context="static rule")
    if isinstance(insert_obj, int):
        return insert_obj
    if isinstance(insert_obj, slice):
        return insert_obj
    return tuple(int(index) for index in np.asarray(insert_obj).reshape(-1))


def _program_ad_array_insert_static_values(values: object) -> object:
    insert_values = _program_ad_array_insert_values(values, context="static rule")
    if insert_values.shape == ():
        return float(insert_values)
    return (
        "static_insert_values",
        tuple(int(dimension) for dimension in insert_values.shape),
        tuple(float(item) for item in insert_values.reshape(-1)),
    )


def _program_ad_array_insert_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {3, 4}:
        raise ValueError(
            "program AD array insert static rule requires array, object, values, and axis"
        )
    normalised_axis = _program_ad_array_insert_axis(
        args[3] if len(args) == 4 else None,
        len(_program_ad_array_shape_of(args[0])),
        context="static rule",
    )
    return (
        _program_ad_array_insert_static_object(args[1]),
        _program_ad_array_insert_static_values(args[2]),
        normalised_axis,
    )


def _program_ad_shape_reshape_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 2:
        raise ValueError("program AD shape reshape static rule requires array and target shape")
    source_shape = _program_ad_array_shape_of(args[0])
    return (_normalise_trace_reshape_shape(args[1], int(np.prod(source_shape))),)


def _program_ad_shape_expand_dims_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    if len(args) != 2:
        raise ValueError("program AD shape expand_dims static rule requires array and axis")
    source_shape = _program_ad_array_shape_of(args[0])
    return (_program_ad_shape_normalise_expand_dims_axes(source_shape, cast(Any, args[1])),)


def _program_ad_shape_no_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 1:
        raise ValueError("program AD shape static rule requires one array")
    return ()


def _program_ad_shape_atleast_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 1:
        raise ValueError("program AD shape atleast static rule requires one array")
    return ()


def _program_ad_shape_transpose_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD shape transpose static rule requires array and optional axes")
    source_shape = _program_ad_array_shape_of(args[0])
    axes = _program_ad_shape_normalised_transpose_axes(
        source_shape, args[1] if len(args) == 2 else None
    )
    return () if not axes else (axes,)


def _program_ad_shape_swapaxes_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 3:
        raise ValueError("program AD shape swapaxes static rule requires array, axis1, and axis2")
    source_shape = _program_ad_array_shape_of(args[0])
    return (
        _normalise_axis_permutation_axis("swapaxes", cast(int, args[1]), rank=len(source_shape)),
        _normalise_axis_permutation_axis("swapaxes", cast(int, args[2]), rank=len(source_shape)),
    )


def _program_ad_shape_moveaxis_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 3:
        raise ValueError(
            "program AD shape moveaxis static rule requires array, source, and destination"
        )
    source_shape = _program_ad_array_shape_of(args[0])
    source_axes, destination_axes, _ = _program_ad_shape_normalise_moveaxis_axes(
        source_shape, cast(Any, args[1]), cast(Any, args[2])
    )
    return source_axes, destination_axes


def _program_ad_shape_roll_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {2, 3}:
        raise ValueError(
            "program AD shape roll static rule requires array, shift, and optional axis"
        )
    source_shape = _program_ad_array_shape_of(args[0])
    return _program_ad_shape_normalise_roll_signature(
        source_shape, args[1], args[2] if len(args) == 3 else None
    )


def _program_ad_shape_flip_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD shape flip static rule requires array and optional axis")
    source_shape = _program_ad_array_shape_of(args[0])
    return (
        _program_ad_shape_normalise_flip_axis(source_shape, args[1] if len(args) == 2 else None),
    )


def _program_ad_shape_rot90_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD shape rot90 static rule requires array, k, and axes")
    source_shape = _program_ad_array_shape_of(args[0])
    return (
        _normalise_rot90_k(args[1] if len(args) >= 2 else 1),
        _normalise_rot90_axes(args[2] if len(args) == 3 else (0, 1), rank=len(source_shape)),
    )


def _program_ad_shape_repeat_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {2, 3}:
        raise ValueError(
            "program AD shape repeat static rule requires array, repeats, and optional axis"
        )
    source_shape = _program_ad_array_shape_of(args[0])
    repeat_counts, axis_index, _ = _program_ad_shape_normalise_repeat_signature(
        source_shape, args[1], args[2] if len(args) == 3 else None
    )
    return repeat_counts, axis_index


def _program_ad_shape_tile_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 2:
        raise ValueError("program AD shape tile static rule requires array and reps")
    source_shape = _program_ad_array_shape_of(args[0])
    reps_tuple, _, _ = _program_ad_shape_normalise_tile_signature(source_shape, args[1])
    return (reps_tuple,)


def _program_ad_shape_squeeze_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD shape squeeze static rule requires array and optional axis")
    source_shape = _program_ad_array_shape_of(args[0])
    return (
        _program_ad_shape_normalise_squeeze_axes(
            source_shape, cast(Any, args[1]) if len(args) == 2 else None
        ),
    )


def _program_ad_reduction_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) == 4:
        source_shape = _program_ad_array_shape_of(args[0])
        axis = _program_ad_reduction_trapezoid_axis(args)
        _program_ad_reduction_trapezoid_static_widths(
            source_shape, x=args[1], dx=args[2], axis=axis
        )
        x = args[1]
        if x is None:
            return (None, _as_real_scalar("program AD trapezoid dx", args[2]), axis)
        x_array = _as_real_numeric_array("program AD trapezoid x", x)
        return (
            (
                "x",
                tuple(int(dimension) for dimension in x_array.shape),
                tuple(float(item) for item in x_array.reshape(-1)),
            ),
            1.0,
            axis,
        )
    return (_program_ad_reduction_axis(args),)


def _program_ad_reduction_order_statistic_axis_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return (_program_ad_order_statistic_reduction_axis(args),)


def _program_ad_reduction_var_std_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return _program_ad_reduction_axis_ddof(args)


def _program_ad_reduction_quantile_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    if len(args) != 4:
        raise ValueError("program AD quantile rule requires array, q, axis, and method")
    _normalise_order_statistic_method(args[3])
    return (
        _normalise_order_statistic_q(args[1], percentile=False),
        _program_ad_order_statistic_reduction_axis(args),
        "linear",
    )


def _program_ad_reduction_percentile_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    if len(args) != 4:
        raise ValueError("program AD percentile rule requires array, q, axis, and method")
    _normalise_order_statistic_method(args[3])
    return (
        _normalise_order_statistic_q(args[1], percentile=True),
        _program_ad_order_statistic_reduction_axis(args),
        "linear",
    )


def _program_ad_elementwise_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD elementwise static rule requires one or two operands")
    return ()


def _program_ad_selection_where_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 3:
        raise ValueError("program AD selection where static rule requires condition, true, false")
    condition = args[0]
    output_shape = _program_ad_selection_where_shape(args)
    if isinstance(condition, _TracePredicate):
        return ("runtime_predicate", (), output_shape)
    if isinstance(condition, TraceADPredicateArray):
        return ("runtime_predicate", condition.shape, output_shape)
    raw = np.asarray(condition)
    if raw.dtype.kind != "b":
        raise ValueError("program AD selection where static rule requires boolean condition")
    if tuple(raw.shape) not in {(), output_shape}:
        raise ValueError(
            "program AD selection where condition shape must be scalar or output-shaped"
        )
    mask = np.broadcast_to(raw, output_shape).reshape(-1)
    return ("static_condition", tuple(bool(item) for item in mask), output_shape)


def _program_ad_selection_clip_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 3:
        raise ValueError("program AD selection clip static rule requires source, lower, and upper")
    _program_ad_selection_clip_shape(args)
    return ()


def _program_ad_selection_sort_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD selection sort static rule requires source, axis, and kind")
    source_shape = _program_ad_array_shape_of(args[0])
    axis = args[1] if len(args) >= 2 else -1
    kind = args[2] if len(args) == 3 else None
    if axis is not None:
        axis = _normalise_sort_axis(axis, len(source_shape))
    if kind not in {None, "quicksort", "mergesort", "heapsort", "stable"}:
        raise ValueError("program AD selection sort static rule requires a NumPy sort kind")
    return ("axis", axis, "kind", "quicksort" if kind is None else kind)


def _program_ad_selection_index_reduce_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    if len(args) not in {1, 2}:
        raise ValueError(
            "program AD index selection static rule requires source and optional axis"
        )
    source_shape = _program_ad_array_shape_of(args[0])
    axis = args[1] if len(args) == 2 else None
    if axis is None:
        return ("axis", None)
    return ("axis", _normalise_sort_axis(axis, len(source_shape)))


def _program_ad_selection_argsort_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    if len(args) not in {1, 2, 3}:
        raise ValueError("program AD argsort static rule requires source, axis, and kind")
    source_shape = _program_ad_array_shape_of(args[0])
    axis = args[1] if len(args) >= 2 else -1
    kind = args[2] if len(args) == 3 else None
    if axis is not None:
        axis = _normalise_sort_axis(axis, len(source_shape))
    if kind not in {None, "quicksort", "mergesort", "heapsort", "stable"}:
        raise ValueError("program AD argsort static rule requires a NumPy sort kind")
    return ("axis", axis, "kind", "quicksort" if kind is None else kind)


def _program_ad_selection_select_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    conditions, choices, default = _program_ad_selection_select_parts(args)
    output_shape = _program_ad_selection_select_shape(args)
    condition_signatures: list[tuple[str, object, tuple[int, ...]]] = []
    for condition in conditions:
        condition_shape = _program_ad_selection_condition_shape(condition)
        if isinstance(condition, _TracePredicate):
            condition_signatures.append(("runtime_predicate", (), output_shape))
        elif isinstance(condition, TraceADPredicateArray):
            condition_signatures.append(("runtime_predicate", condition.shape, output_shape))
        else:
            raw = np.asarray(condition)
            condition_signatures.append(
                (
                    "static_condition",
                    tuple(bool(item) for item in np.broadcast_to(raw, output_shape).reshape(-1)),
                    output_shape,
                )
            )
        if condition_shape not in {(), output_shape}:
            raise ValueError("program AD select condition shape must be scalar or output-shaped")
    return (
        "branch_count",
        len(conditions),
        "condition_signatures",
        tuple(condition_signatures),
        "choice_shapes",
        tuple(_program_ad_array_shape_of(choice) for choice in choices),
        "default_shape",
        _program_ad_array_shape_of(default),
        "output_shape",
        output_shape,
    )


def _program_ad_selection_piecewise_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    source_shape, conditions, functions = _program_ad_selection_piecewise_parts(args)
    return (
        "source_shape",
        source_shape,
        "condition_shapes",
        tuple(_program_ad_selection_condition_shape(condition) for condition in conditions),
        "function_count",
        len(functions),
        "has_default",
        len(functions) == len(conditions) + 1,
    )


def _program_ad_selection_choose_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    selector, choice_shapes, mode = _program_ad_selection_choose_parts(args)
    return (
        "selector",
        tuple(int(item) for item in selector.reshape(-1)),
        "selector_shape",
        tuple(int(dimension) for dimension in selector.shape),
        "choice_shapes",
        choice_shapes,
        "mode",
        mode,
    )


def _program_ad_selection_compress_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    source_shape, indices, axis = _program_ad_selection_compress_parts(args)
    return (
        "source_shape",
        source_shape,
        "indices",
        tuple(int(item) for item in indices.reshape(-1)),
        "axis",
        axis,
    )


def _program_ad_selection_extract_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 2:
        raise ValueError("program AD extract static rule requires condition and array")
    source_shape = _program_ad_array_shape_of(args[1])
    indices = _trace_extract_condition_indices(
        args[0], _program_ad_array_static_size(source_shape)
    )
    return (
        "source_shape",
        source_shape,
        "indices",
        tuple(int(item) for item in indices.reshape(-1)),
    )


def _program_ad_product_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if args and isinstance(args[0], str):
        operand_shapes = tuple(_program_ad_array_shape_of(arg) for arg in args[1:])
        output_labels, input_labels, _dimensions = _parse_static_einsum_subscripts(
            args[0],
            operand_shapes,
        )
        return (
            args[0].replace(" ", ""),
            operand_shapes,
            tuple("".join(labels) for labels in input_labels),
            "".join(output_labels),
        )
    if len(args) == 3:
        left_shape = _program_ad_array_shape_of(args[0])
        right_shape = _program_ad_array_shape_of(args[1])
        _left, _right, left_axes, right_axes, _output_shape = (
            _normalise_program_ad_product_tensordot_signature(
                left_shape,
                right_shape,
                args[2],
            )
        )
        return (left_shape, right_shape, (left_axes, right_axes))
    if len(args) != 2:
        raise ValueError("program AD product static rule requires two operands")
    return ()


def _program_ad_cumulative_scan_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return (_program_ad_cumulative_axis(args),)


def _program_ad_cumulative_diff_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    order = _program_ad_cumulative_diff_order(args)
    axis = args[2] if len(args) == 3 else -1
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD diff axis must be a static integer")
    axis_index = _normalise_axis("axis", int(axis), len(_program_ad_array_shape_of(args[0])))
    return (order, axis_index)


_PROGRAM_AD_ARRAY_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "getitem": _program_ad_array_getitem_shape,
    "take": _program_ad_array_take_shape,
    "take_along_axis": _program_ad_array_take_along_axis_shape,
    "delete": _program_ad_array_delete_shape,
    "pad": _program_ad_array_pad_shape,
    "insert": _program_ad_array_insert_shape,
}

_PROGRAM_AD_ARRAY_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "getitem": _program_ad_array_getitem_static_arguments,
    "take": _program_ad_array_take_static_arguments,
    "take_along_axis": _program_ad_array_take_along_axis_static_arguments,
    "delete": _program_ad_array_delete_static_arguments,
    "pad": _program_ad_array_pad_static_arguments,
    "insert": _program_ad_array_insert_static_arguments,
}


def _program_ad_interpolation_sample_shape(value: object) -> tuple[int, ...]:
    if isinstance(value, TraceADArray):
        return value.shape
    if isinstance(value, TraceADScalar):
        return ()
    return tuple(int(dimension) for dimension in np.asarray(value).shape)


def _program_ad_interpolation_fp_shape(value: object) -> tuple[int, ...]:
    if isinstance(value, TraceADArray):
        return value.shape
    if isinstance(value, TraceADScalar):
        raise ValueError("program AD interpolation interp fp must be one-dimensional")
    return tuple(int(dimension) for dimension in np.asarray(value).shape)


def _program_ad_interpolation_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], NDArray[np.float64], tuple[int, ...], float | None, float | None]:
    if len(args) != 6:
        raise ValueError(
            "program AD interpolation interp rule requires x, xp, fp, left, right, and period"
        )
    if args[5] is not None:
        raise ValueError("program AD interpolation interp period is not supported")
    sample_shape = _program_ad_interpolation_sample_shape(args[0])
    grid = _normalise_interp_grid(args[1])
    fp_shape = _program_ad_interpolation_fp_shape(args[2])
    if fp_shape != (grid.size,):
        raise ValueError("program AD np.interp fp values must match xp grid")
    left = _program_ad_interp_static_boundary("left", args[3])
    right = _program_ad_interp_static_boundary("right", args[4])
    return sample_shape, grid, fp_shape, left, right


def _program_ad_interpolation_interp_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    sample_shape, _grid, _fp_shape, _left, _right = _program_ad_interpolation_static_parts(args)
    return sample_shape


def _program_ad_interpolation_interp_dtype_rule(_args: tuple[object, ...]) -> str:
    return "float64"


def _program_ad_interpolation_interp_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    sample_shape, grid, fp_shape, left, right = _program_ad_interpolation_static_parts(args)
    return (
        sample_shape,
        ("xp", tuple(int(dimension) for dimension in grid.shape), tuple(float(x) for x in grid)),
        fp_shape,
        left,
        right,
        None,
    )


def _program_ad_interpolation_derivative_rule(name: str) -> CustomDerivativeRule:
    if name == "interp":
        return CustomDerivativeRule(
            name="program_ad_interpolation_interp_trace_contract",
            value_fn=_program_ad_array_direct_value,
            jvp_rule=_program_ad_array_direct_jvp,
        )
    raise ValueError(f"unsupported program AD interpolation primitive {name}")


def _program_ad_interpolation_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 6 or len(axes) != 6:
        raise ValueError(
            "program AD interpolation interp batching requires x, xp, fp, left, right, and period"
        )
    if any(axis is not None for axis in axes[1:]):
        raise ValueError(
            "program AD interpolation interp batching keeps xp, fp, left, right, and period static"
        )
    if axes[0] is None:
        return _as_real_numeric_array(
            "program AD interpolation interp batched output", function(*args)
        )
    samples = _as_real_numeric_array("program AD interpolation interp batched samples", args[0])
    batch_axis = _normalise_axis("axes[0]", axes[0], samples.ndim)
    outputs = [
        _as_real_numeric_array(
            "program AD interpolation interp batched output",
            function(np.take(samples, batch_index, axis=batch_axis), *args[1:]),
        )
        for batch_index in range(samples.shape[batch_axis])
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_interpolation_lowering_metadata(name: str) -> Mapping[str, str]:
    if name != "interp":
        raise ValueError(f"unsupported program AD interpolation primitive {name}")
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff interpolation dialect interchange; executable lowering blocked",
        "mlir_op": "scpn_diff.interpolation.interp",
        "llvm": "blocked_until_executable_interpolation_lowering",
        "rust": "blocked_until_polyglot_interpolation_ad",
        "static_argument_rule": "required",
        "static_derivative_factory": "program_ad_interpolation_interp_derivative_rule",
        "static_signature": "sample_shape:ranked_tensor_shape;xp_grid;fp_shape;left_right_period",
        "nondifferentiable_boundary": "static_grid_knot_and_period_boundary",
        "nondifferentiable_boundary_policy": "fail_closed",
    }


def _program_ad_assembly_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    raise ValueError(
        "program AD assembly primitive contracts are executable only through "
        "operator-intercepted trace dispatch or fixed-shape derivative factories"
    )


def _program_ad_assembly_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    raise ValueError(
        "program AD assembly primitive contracts are executable only through "
        "operator-intercepted trace dispatch or fixed-shape derivative factories"
    )


def _program_ad_assembly_derivative_rule(name: str) -> CustomDerivativeRule:
    if name in _PROGRAM_AD_ASSEMBLY_IDENTITIES:
        return CustomDerivativeRule(
            name=f"program_ad_assembly_{name}_trace_contract",
            value_fn=_program_ad_assembly_direct_value,
            jvp_rule=_program_ad_assembly_direct_jvp,
        )
    raise ValueError(f"unsupported program AD assembly primitive {name}")


def _program_ad_assembly_like_reference_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if not args:
        raise ValueError("program AD like-constructor requires a reference operand")
    source_shape = _program_ad_array_shape_of(args[0])
    if int(np.prod(source_shape)) == 0:
        raise ValueError("program AD like-constructor requires at least one element")
    return source_shape


def _program_ad_assembly_zeros_like_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD zeros_like requires one reference operand")
    return _program_ad_assembly_like_reference_shape(args)


def _program_ad_assembly_ones_like_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD ones_like requires one reference operand")
    return _program_ad_assembly_like_reference_shape(args)


def _program_ad_assembly_full_like_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD full_like requires reference and scalar fill operands")
    _program_ad_assembly_static_scalar_fill(args[1])
    return _program_ad_assembly_like_reference_shape(args)


def _program_ad_assembly_like_dtype_rule(_args: tuple[object, ...]) -> str:
    return "float64"


def _program_ad_assembly_static_scalar_fill(value: object) -> str:
    if isinstance(value, TraceADScalar):
        return "trace_scalar"
    if isinstance(value, TraceADArray):
        raise ValueError("program AD full_like fill value must be scalar")
    array = np.asarray(value)
    if array.shape not in {(), (1,)}:
        raise ValueError("program AD full_like fill value must be scalar")
    if array.dtype.kind in {"O", "S", "U", "c"}:
        raise ValueError("program AD full_like fill value must be real numeric")
    scalar = float(array.reshape(-1)[0])
    if not math.isfinite(scalar):
        raise ValueError("program AD full_like fill value must be finite")
    return "static_scalar"


def _program_ad_assembly_zeros_like_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return (_program_ad_assembly_zeros_like_shape(args),)


def _program_ad_assembly_ones_like_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return (_program_ad_assembly_ones_like_shape(args),)


def _program_ad_assembly_full_like_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return (
        _program_ad_assembly_full_like_shape(args),
        _program_ad_assembly_static_scalar_fill(args[1]),
    )


def _program_ad_assembly_concatenate_operands(
    args: tuple[object, ...],
) -> tuple[tuple[object, ...], object]:
    if len(args) != 2:
        raise ValueError("program AD assembly concatenate requires operands and axis")
    operands = args[0]
    if not isinstance(operands, (tuple, list)):
        raise ValueError("program AD assembly concatenate requires a static operand sequence")
    operand_tuple = tuple(operands)
    if not operand_tuple:
        raise ValueError("program AD assembly concatenate requires operands")
    return operand_tuple, args[1]


def _program_ad_assembly_concatenate_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[tuple[int, ...], ...], int | None]:
    operands, axis = _program_ad_assembly_concatenate_operands(args)
    operand_shapes = tuple(_program_ad_array_shape_of(operand) for operand in operands)
    if axis is None:
        _program_ad_assembly_concatenate_output_shape(operand_shapes, None)
        return operand_shapes, None
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD assembly concatenate requires a static integer axis or None")
    normalised_axis = _program_ad_assembly_concatenate_axis(axis, rank=len(operand_shapes[0]))
    _program_ad_assembly_concatenate_output_shape(operand_shapes, normalised_axis)
    return operand_shapes, normalised_axis


def _program_ad_assembly_concatenate_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    operand_shapes, axis = _program_ad_assembly_concatenate_static_parts(args)
    return _program_ad_assembly_concatenate_output_shape(operand_shapes, axis)


def _program_ad_assembly_concatenate_dtype_rule(args: tuple[object, ...]) -> str:
    operands, _axis = _program_ad_assembly_concatenate_operands(args)
    operand_dtypes = [np.dtype(_program_ad_array_dtype_of(operand)) for operand in operands]
    return str(np.result_type(*operand_dtypes))


def _program_ad_assembly_concatenate_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    operand_shapes, axis = _program_ad_assembly_concatenate_static_parts(args)
    return operand_shapes, axis


def _program_ad_assembly_stack_operands(
    args: tuple[object, ...],
) -> tuple[tuple[object, ...], object]:
    if len(args) != 2:
        raise ValueError("program AD assembly stack requires operands and axis")
    operands = args[0]
    if not isinstance(operands, (tuple, list)):
        raise ValueError("program AD assembly stack requires a static operand sequence")
    operand_tuple = tuple(operands)
    if not operand_tuple:
        raise ValueError("program AD assembly stack requires operands")
    return operand_tuple, args[1]


def _program_ad_assembly_stack_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[tuple[int, ...], ...], int]:
    operands, axis = _program_ad_assembly_stack_operands(args)
    operand_shapes = tuple(_program_ad_array_shape_of(operand) for operand in operands)
    normalised_axis = _program_ad_assembly_stack_axis(axis, rank=len(operand_shapes[0]))
    _program_ad_assembly_stack_output_shape(operand_shapes, normalised_axis)
    return operand_shapes, normalised_axis


def _program_ad_assembly_stack_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    operand_shapes, axis = _program_ad_assembly_stack_static_parts(args)
    return _program_ad_assembly_stack_output_shape(operand_shapes, axis)


def _program_ad_assembly_stack_dtype_rule(args: tuple[object, ...]) -> str:
    operands, _axis = _program_ad_assembly_stack_operands(args)
    operand_dtypes = [np.dtype(_program_ad_array_dtype_of(operand)) for operand in operands]
    return str(np.result_type(*operand_dtypes))


def _program_ad_assembly_stack_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    operand_shapes, axis = _program_ad_assembly_stack_static_parts(args)
    return operand_shapes, axis


def _program_ad_assembly_append_operands(
    args: tuple[object, ...],
) -> tuple[object, object, object]:
    if len(args) != 3:
        raise ValueError("program AD assembly append requires source, values, and axis")
    return args[0], args[1], args[2]


def _program_ad_assembly_append_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], int | None]:
    source, values, axis = _program_ad_assembly_append_operands(args)
    source_shape = _program_ad_array_shape_of(source)
    values_shape = _program_ad_array_shape_of(values)
    if axis is None:
        _program_ad_assembly_append_output_shape(source_shape, values_shape, axis=None)
        return source_shape, values_shape, None
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD assembly append requires a static integer axis or None")
    normalised_axis = _program_ad_assembly_concatenate_axis(axis, rank=len(source_shape))
    _program_ad_assembly_append_output_shape(source_shape, values_shape, axis=normalised_axis)
    return source_shape, values_shape, normalised_axis


def _program_ad_assembly_append_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape, values_shape, axis = _program_ad_assembly_append_static_parts(args)
    return _program_ad_assembly_append_output_shape(source_shape, values_shape, axis=axis)


def _program_ad_assembly_append_dtype_rule(args: tuple[object, ...]) -> str:
    source, values, _axis = _program_ad_assembly_append_operands(args)
    return str(
        np.result_type(
            np.dtype(_program_ad_array_dtype_of(source)),
            np.dtype(_program_ad_array_dtype_of(values)),
        )
    )


def _program_ad_assembly_append_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    source_shape, values_shape, axis = _program_ad_assembly_append_static_parts(args)
    return source_shape, values_shape, axis


def _program_ad_assembly_block_layout(args: tuple[object, ...]) -> object:
    if len(args) != 1:
        raise ValueError("program AD assembly block requires one nested layout argument")
    layout = args[0]
    if not isinstance(layout, (tuple, list)):
        raise ValueError("program AD assembly block requires a nested layout")
    if not layout:
        raise ValueError("program AD assembly block requires a non-empty nested layout")
    return layout


def _program_ad_assembly_block_shape_layout_from_operands(layout: object) -> tuple[object, ...]:
    if isinstance(layout, (tuple, list)):
        if not layout:
            raise ValueError("program AD assembly block requires a non-empty nested layout")
        return tuple(
            _program_ad_assembly_block_shape_layout_from_operands(item) for item in layout
        )
    return _program_ad_array_shape_of(layout)


def _program_ad_assembly_block_dtype_leaves(layout: object) -> tuple[np.dtype[Any], ...]:
    if isinstance(layout, (tuple, list)):
        if not layout:
            raise ValueError("program AD assembly block requires a non-empty nested layout")
        dtypes: list[np.dtype[Any]] = []
        for item in layout:
            dtypes.extend(_program_ad_assembly_block_dtype_leaves(item))
        return tuple(dtypes)
    return (np.dtype(_program_ad_array_dtype_of(layout)),)


def _program_ad_assembly_block_numpy_layout(layout: object) -> object:
    if isinstance(layout, (tuple, list)):
        if not layout:
            raise ValueError("program AD assembly block requires a non-empty nested layout")
        return [_program_ad_assembly_block_numpy_layout(item) for item in layout]
    return layout


def _program_ad_assembly_block_static_parts(args: tuple[object, ...]) -> tuple[object, ...]:
    layout = _program_ad_assembly_block_layout(args)
    layout_shapes = _program_ad_assembly_block_shape_layout_from_operands(layout)
    _program_ad_assembly_block_output_shape(layout_shapes)
    return layout_shapes


def _program_ad_assembly_block_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    return _program_ad_assembly_block_output_shape(_program_ad_assembly_block_static_parts(args))


def _program_ad_assembly_block_dtype_rule(args: tuple[object, ...]) -> str:
    layout = _program_ad_assembly_block_layout(args)
    return str(np.result_type(*_program_ad_assembly_block_dtype_leaves(layout)))


def _program_ad_assembly_block_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return _program_ad_assembly_block_static_parts(args)


def _program_ad_assembly_stack_convenience_operands(
    name: str,
    args: tuple[object, ...],
) -> tuple[object, ...]:
    if len(args) != 1:
        raise ValueError(f"program AD assembly {name} requires one operand sequence")
    operands = args[0]
    if not isinstance(operands, (tuple, list)):
        raise ValueError(f"program AD assembly {name} requires a static operand sequence")
    operand_tuple = tuple(operands)
    if not operand_tuple:
        raise ValueError(f"program AD assembly {name} requires operands")
    return operand_tuple


def _program_ad_assembly_stack_convenience_static_parts(
    name: str,
    args: tuple[object, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    operands = _program_ad_assembly_stack_convenience_operands(name, args)
    operand_shapes = tuple(_program_ad_array_shape_of(operand) for operand in operands)
    _shapes, selected = _program_ad_assembly_stack_convenience_selected_indices(
        name, operand_shapes
    )
    return operand_shapes, tuple(int(dimension) for dimension in selected.shape)


def _program_ad_assembly_stack_convenience_shape_rule_for(name: str) -> PrimitiveShapeRule:
    def shape_rule(args: tuple[object, ...]) -> tuple[int, ...]:
        _operand_shapes, output_shape = _program_ad_assembly_stack_convenience_static_parts(
            name, args
        )
        return output_shape

    return shape_rule


def _program_ad_assembly_stack_convenience_dtype_rule_for(name: str) -> PrimitiveDTypeRule:
    def dtype_rule(args: tuple[object, ...]) -> str:
        operands = _program_ad_assembly_stack_convenience_operands(name, args)
        operand_dtypes = [np.dtype(_program_ad_array_dtype_of(operand)) for operand in operands]
        return str(np.result_type(*operand_dtypes))

    return dtype_rule


def _program_ad_assembly_stack_convenience_static_arguments_rule_for(
    name: str,
) -> PrimitiveStaticArgumentRule:
    def static_argument_rule(args: tuple[object, ...]) -> tuple[object, ...]:
        operand_shapes, output_shape = _program_ad_assembly_stack_convenience_static_parts(
            name, args
        )
        return operand_shapes, output_shape

    return static_argument_rule


def _program_ad_assembly_broadcast_to_shapes(
    source_shape: Sequence[int],
    output_shape: object,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    source = _program_ad_array_normalise_static_shape("assembly broadcast_to source", source_shape)
    output = _normalise_trace_broadcast_shape(output_shape)
    try:
        np.broadcast_to(np.empty(source, dtype=np.float64), output)
    except ValueError as exc:
        raise ValueError(
            "program AD assembly broadcast_to requires output shape compatible "
            "with source broadcasting rules"
        ) from exc
    return source, output


def _program_ad_assembly_broadcast_to_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(args) != 2:
        raise ValueError("program AD assembly broadcast_to requires source and output shape")
    return _program_ad_assembly_broadcast_to_shapes(_program_ad_array_shape_of(args[0]), args[1])


def _program_ad_assembly_broadcast_to_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    _source_shape, output_shape = _program_ad_assembly_broadcast_to_static_parts(args)
    return output_shape


def _program_ad_assembly_broadcast_to_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) != 2:
        raise ValueError("program AD assembly broadcast_to requires source and output shape")
    return str(np.dtype(_program_ad_array_dtype_of(args[0])))


def _program_ad_assembly_broadcast_to_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return _program_ad_assembly_broadcast_to_static_parts(args)


def _program_ad_assembly_broadcast_arrays_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    if not args:
        raise ValueError("program AD assembly broadcast_arrays requires at least one operand")
    operand_shapes = tuple(_program_ad_array_shape_of(arg) for arg in args)
    try:
        output_shape = tuple(int(dimension) for dimension in np.broadcast_shapes(*operand_shapes))
    except ValueError as exc:
        raise ValueError(
            "program AD assembly broadcast_arrays requires operands compatible "
            "with broadcasting rules"
        ) from exc
    return operand_shapes, output_shape


def _program_ad_assembly_broadcast_arrays_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    operand_shapes, output_shape = _program_ad_assembly_broadcast_arrays_static_parts(args)
    return (len(operand_shapes) * _program_ad_array_static_size(output_shape),)


def _program_ad_assembly_broadcast_arrays_dtype_rule(args: tuple[object, ...]) -> str:
    if not args:
        raise ValueError("program AD assembly broadcast_arrays requires at least one operand")
    operand_dtypes = [np.dtype(_program_ad_array_dtype_of(arg)) for arg in args]
    return str(np.result_type(*operand_dtypes))


def _program_ad_assembly_broadcast_arrays_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return _program_ad_assembly_broadcast_arrays_static_parts(args)


@dataclass(frozen=True)
class _ProgramADAssemblyBlockMappedLeaf:
    array: NDArray[np.float64]
    axis: int


def _program_ad_assembly_split_sections(indices_or_sections: object) -> int | tuple[int, ...]:
    if isinstance(indices_or_sections, bool):
        raise ValueError("program AD assembly split requires static integer sections")
    if isinstance(indices_or_sections, (int, np.integer)):
        sections = int(indices_or_sections)
        if sections <= 0:
            raise ValueError("program AD assembly split requires positive static sections")
        return sections
    array = np.asarray(indices_or_sections)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("program AD assembly split requires static integer split indices")
    return tuple(int(index) for index in array.tolist())


def _program_ad_assembly_split_axis(axis: object, *, rank: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD assembly split requires a static integer axis")
    if rank <= 0:
        raise ValueError("program AD assembly split requires ranked source arrays")
    return _normalise_axis("axis", int(axis), rank)


def _program_ad_assembly_split_selected_indices(
    split_name: str,
    source_shape: Sequence[int],
    indices_or_sections: object,
    *,
    axis: object,
) -> tuple[NDArray[np.int64], ...]:
    if split_name not in _PROGRAM_AD_ASSEMBLY_SPLIT_NAMES:
        raise ValueError(f"unsupported program AD assembly split primitive {split_name}")
    shape = _program_ad_array_normalise_static_shape("assembly split source", source_shape)
    axis_index = _program_ad_assembly_split_axis(axis, rank=len(shape))
    sections = _program_ad_assembly_split_sections(indices_or_sections)
    index_array = np.arange(_program_ad_array_static_size(shape), dtype=np.int64).reshape(shape)
    try:
        if split_name == "array_split":
            selected = np.array_split(index_array, cast(Any, sections), axis=axis_index)
        else:
            selected = np.split(index_array, cast(Any, sections), axis=axis_index)
    except (TypeError, ValueError, np.exceptions.AxisError) as exc:
        raise ValueError(
            "program AD assembly split requires static split sections compatible with source shape"
        ) from exc
    return tuple(np.asarray(part, dtype=np.int64) for part in selected)


def _program_ad_assembly_split_static_parts(
    split_name: str,
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], int | tuple[int, ...], int, tuple[tuple[int, ...], ...]]:
    if len(args) != 3:
        raise ValueError("program AD assembly split requires source, sections, and axis")
    source_shape = _program_ad_array_shape_of(args[0])
    axis = _program_ad_assembly_split_axis(args[2], rank=len(source_shape))
    sections = _program_ad_assembly_split_sections(args[1])
    selected = _program_ad_assembly_split_selected_indices(
        split_name,
        source_shape,
        sections,
        axis=axis,
    )
    part_shapes = tuple(tuple(int(dimension) for dimension in part.shape) for part in selected)
    return source_shape, sections, axis, part_shapes


def _program_ad_assembly_split_shape_rule_for(split_name: str) -> PrimitiveShapeRule:
    def shape_rule(args: tuple[object, ...]) -> tuple[int, ...]:
        source_shape, _sections, _axis, _part_shapes = _program_ad_assembly_split_static_parts(
            split_name, args
        )
        return (_program_ad_array_static_size(source_shape),)

    return shape_rule


def _program_ad_assembly_split_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) != 3:
        raise ValueError("program AD assembly split requires source, sections, and axis")
    return str(np.dtype(_program_ad_array_dtype_of(args[0])))


def _program_ad_assembly_split_static_arguments_rule_for(
    split_name: str,
) -> PrimitiveStaticArgumentRule:
    def static_argument_rule(args: tuple[object, ...]) -> tuple[object, ...]:
        return _program_ad_assembly_split_static_parts(split_name, args)

    return static_argument_rule


def _program_ad_assembly_triangular_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], int]:
    if len(args) != 2:
        raise ValueError("program AD assembly triangular mask requires source and k")
    source_shape = _program_ad_array_shape_of(args[0])
    if len(source_shape) < 2:
        raise ValueError("program AD assembly triangular mask requires rank >= 2")
    k = args[1]
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("program AD assembly triangular mask requires static integer k")
    return source_shape, int(k)


def _program_ad_assembly_triangular_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape, _k = _program_ad_assembly_triangular_static_parts(args)
    return source_shape


def _program_ad_assembly_triangular_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) != 2:
        raise ValueError("program AD assembly triangular mask requires source and k")
    return str(np.dtype(_program_ad_array_dtype_of(args[0])))


def _program_ad_assembly_triangular_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return _program_ad_assembly_triangular_static_parts(args)


def _program_ad_assembly_diagonal_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], int, int, int, tuple[int, ...]]:
    if len(args) != 4:
        raise ValueError("program AD assembly diagonal requires source, offset, axis1, and axis2")
    source_shape = _program_ad_array_shape_of(args[0])
    if len(source_shape) < 2:
        raise ValueError("program AD assembly diagonal requires rank >= 2")
    offset, axis1, axis2 = args[1], args[2], args[3]
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)):
        raise ValueError("program AD assembly diagonal requires static integer offset")
    if isinstance(axis1, bool) or not isinstance(axis1, (int, np.integer)):
        raise ValueError("program AD assembly diagonal requires static integer axes")
    if isinstance(axis2, bool) or not isinstance(axis2, (int, np.integer)):
        raise ValueError("program AD assembly diagonal requires static integer axes")
    axis1_value = _normalise_axis("axis1", int(axis1), len(source_shape))
    axis2_value = _normalise_axis("axis2", int(axis2), len(source_shape))
    if axis1_value == axis2_value:
        raise ValueError("program AD assembly diagonal requires distinct axes")
    try:
        output = np.diagonal(
            np.empty(source_shape, dtype=np.float64),
            offset=int(offset),
            axis1=axis1_value,
            axis2=axis2_value,
        )
    except (TypeError, ValueError, np.exceptions.AxisError) as exc:
        raise ValueError(
            "program AD assembly diagonal requires static offset and axes "
            "compatible with source shape"
        ) from exc
    return (
        source_shape,
        int(offset),
        axis1_value,
        axis2_value,
        tuple(int(dimension) for dimension in output.shape),
    )


def _program_ad_assembly_diagonal_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    _source_shape, _offset, _axis1, _axis2, output_shape = (
        _program_ad_assembly_diagonal_static_parts(args)
    )
    return output_shape


def _program_ad_assembly_diagonal_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) != 4:
        raise ValueError("program AD assembly diagonal requires source, offset, axis1, and axis2")
    return str(np.dtype(_program_ad_array_dtype_of(args[0])))


def _program_ad_assembly_diagonal_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return _program_ad_assembly_diagonal_static_parts(args)


def _program_ad_assembly_split_move_output_batch_axis(output: object, out_axes: int) -> object:
    if isinstance(output, tuple):
        return tuple(
            _program_ad_assembly_split_move_output_batch_axis(item, out_axes) for item in output
        )
    if isinstance(output, list):
        return [
            _program_ad_assembly_split_move_output_batch_axis(item, out_axes) for item in output
        ]
    array = _as_real_numeric_array("program AD assembly split batched output", output)
    return np.moveaxis(array, 0, _normalise_axis("out_axes", out_axes, array.ndim))


def _program_ad_assembly_split_stack_outputs(outputs: Sequence[object], out_axes: int) -> object:
    if not outputs:
        raise ValueError("program AD assembly split batching requires non-empty outputs")
    first = outputs[0]
    if isinstance(first, tuple):
        if any(not isinstance(output, tuple) or len(output) != len(first) for output in outputs):
            raise ValueError(
                "program AD assembly split batching requires stable output partitions"
            )
        return tuple(
            _program_ad_assembly_split_stack_outputs(
                [cast(tuple[object, ...], output)[index] for output in outputs],
                out_axes,
            )
            for index in range(len(first))
        )
    if isinstance(first, list):
        if any(not isinstance(output, list) or len(output) != len(first) for output in outputs):
            raise ValueError(
                "program AD assembly split batching requires stable output partitions"
            )
        return [
            _program_ad_assembly_split_stack_outputs(
                [cast(list[object], output)[index] for output in outputs],
                out_axes,
            )
            for index in range(len(first))
        ]
    arrays = [
        _as_real_numeric_array("program AD assembly split batched output", output)
        for output in outputs
    ]
    stacked = np.stack(arrays, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_split_batching_rule_for(split_name: str) -> PrimitiveBatchingRule:
    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        if len(args) != 3 or len(axes) != 3:
            raise ValueError(
                "program AD assembly split batching requires source, sections, and axis"
            )
        if axes[1] is not None or axes[2] is not None:
            raise ValueError("program AD assembly split batching keeps split metadata static")
        source = _as_real_numeric_array("program AD assembly split batched source", args[0])
        sections = _program_ad_assembly_split_sections(args[1])
        split_axis = _program_ad_assembly_split_axis(args[2], rank=source.ndim)
        source_axis = axes[0]
        if source_axis is None:
            return function(source, sections, split_axis)
        source_axis_index = _normalise_axis("source axis", source_axis, source.ndim)
        if source_axis_index == split_axis:
            raise ValueError("program AD assembly split batching cannot map the split axis")
        adjusted_split_axis = split_axis - 1 if source_axis_index < split_axis else split_axis
        outputs = [
            function(
                np.take(source, batch_index, axis=source_axis_index),
                sections,
                adjusted_split_axis,
            )
            for batch_index in range(int(source.shape[source_axis_index]))
        ]
        return _program_ad_assembly_split_stack_outputs(outputs, out_axes)

    return batching_rule


def _program_ad_assembly_broadcast_to_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 2 or len(axes) != 2:
        raise ValueError(
            "program AD assembly broadcast_to batching requires source and output shape"
        )
    source, output_shape_arg = args
    source_axis, output_shape_axis = axes
    if output_shape_axis is not None:
        raise ValueError("program AD assembly broadcast_to batching keeps output shape static")
    output_shape = _normalise_trace_broadcast_shape(output_shape_arg)
    if source_axis is None:
        return function(source, output_shape)
    source_array = _as_real_numeric_array(
        "program AD assembly broadcast_to batched source", source
    )
    axis_index = _normalise_axis("source axis", source_axis, source_array.ndim)
    moved = np.moveaxis(source_array, axis_index, 0)
    outputs = [
        _as_real_numeric_array(
            "program AD assembly broadcast_to batched output",
            function(moved[index], output_shape),
        )
        for index in range(moved.shape[0])
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_broadcast_arrays_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if not args:
        raise ValueError("program AD assembly broadcast_arrays batching requires operands")
    if len(args) != len(axes):
        raise ValueError(
            "program AD assembly broadcast_arrays batching requires one axis per operand"
        )
    moved_args: list[object] = []
    batch_size: int | None = None
    for operand_index, (arg, axis) in enumerate(zip(args, axes, strict=True)):
        if axis is None:
            moved_args.append(arg)
            continue
        operand = _as_real_numeric_array(
            f"program AD assembly broadcast_arrays operand {operand_index}", arg
        )
        axis_index = _normalise_axis("operand axis", axis, operand.ndim)
        moved = np.moveaxis(operand, axis_index, 0)
        if batch_size is None:
            batch_size = int(moved.shape[0])
        elif int(moved.shape[0]) != batch_size:
            raise ValueError(
                "program AD assembly broadcast_arrays batching requires same batch size"
            )
        moved_args.append(moved)
    if batch_size is None:
        return function(*args)
    outputs: list[tuple[NDArray[np.float64], ...]] = []
    for batch_index in range(batch_size):
        sliced_args = tuple(
            cast(NDArray[np.float64], arg)[batch_index] if axis is not None else arg
            for arg, axis in zip(moved_args, axes, strict=True)
        )
        result = function(*sliced_args)
        if not isinstance(result, (tuple, list)):
            raise ValueError(
                "program AD assembly broadcast_arrays batching requires tuple/list outputs"
            )
        outputs.append(
            tuple(
                _as_real_numeric_array("program AD assembly broadcast_arrays batched output", item)
                for item in result
            )
        )
    return _program_ad_assembly_split_stack_outputs(outputs, out_axes)


def _program_ad_assembly_triangular_batching_rule_for(name: str) -> PrimitiveBatchingRule:
    if name not in {"tril", "triu"}:
        raise ValueError(f"unsupported program AD assembly triangular primitive {name}")

    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        if len(args) != 2 or len(axes) != 2:
            raise ValueError(f"program AD assembly {name} batching requires source and k")
        source, k = args
        source_axis, k_axis = axes
        if k_axis is not None:
            raise ValueError(f"program AD assembly {name} batching keeps k static")
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise ValueError(f"program AD assembly {name} batching requires static integer k")
        if source_axis is None:
            return function(source, int(k))
        source_array = _as_real_numeric_array(f"program AD assembly {name} batched source", source)
        if source_array.ndim < 3:
            raise ValueError(
                f"program AD assembly {name} batching requires an outer batch axis "
                "separate from matrix axes"
            )
        axis_index = _normalise_axis("source axis", source_axis, source_array.ndim)
        if axis_index >= source_array.ndim - 2:
            raise ValueError(f"program AD assembly {name} batching cannot map matrix axes")
        moved = np.moveaxis(source_array, axis_index, 0)
        outputs = [
            _as_real_numeric_array(
                f"program AD assembly {name} batched output",
                function(moved[index], int(k)),
            )
            for index in range(moved.shape[0])
        ]
        stacked = np.stack(outputs, axis=0)
        return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))

    return batching_rule


def _program_ad_assembly_diagonal_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 4 or len(axes) != 4:
        raise ValueError(
            "program AD assembly diagonal batching requires source, offset, axis1, and axis2"
        )
    source, offset, axis1, axis2 = args
    source_axis, offset_axis, axis1_axis, axis2_axis = axes
    if offset_axis is not None or axis1_axis is not None or axis2_axis is not None:
        raise ValueError("program AD assembly diagonal batching keeps offset and axes static")
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)):
        raise ValueError("program AD assembly diagonal batching requires static integer offset")
    if isinstance(axis1, bool) or not isinstance(axis1, (int, np.integer)):
        raise ValueError("program AD assembly diagonal batching requires static integer axes")
    if isinstance(axis2, bool) or not isinstance(axis2, (int, np.integer)):
        raise ValueError("program AD assembly diagonal batching requires static integer axes")
    if source_axis is None:
        return function(source, int(offset), int(axis1), int(axis2))
    source_array = _as_real_numeric_array("program AD assembly diagonal batched source", source)
    if source_array.ndim < 3:
        raise ValueError(
            "program AD assembly diagonal batching requires an outer batch axis "
            "separate from diagonal axes"
        )
    batch_axis = _normalise_axis("source axis", source_axis, source_array.ndim)
    axis1_value = _normalise_axis("axis1", int(axis1), source_array.ndim)
    axis2_value = _normalise_axis("axis2", int(axis2), source_array.ndim)
    if axis1_value == axis2_value:
        raise ValueError("program AD assembly diagonal batching requires distinct diagonal axes")
    if batch_axis in {axis1_value, axis2_value}:
        raise ValueError("program AD assembly diagonal batching cannot map diagonal axes")

    def adjusted_axis(axis: int) -> int:
        if batch_axis < axis:
            return axis - 1
        return axis

    moved = np.moveaxis(source_array, batch_axis, 0)
    outputs = [
        _as_real_numeric_array(
            "program AD assembly diagonal batched output",
            function(
                moved[index],
                int(offset),
                adjusted_axis(axis1_value),
                adjusted_axis(axis2_value),
            ),
        )
        for index in range(moved.shape[0])
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 2 or len(axes) != 2:
        raise ValueError("program AD assembly concatenate batching requires operands and axis")
    if axes[1] is not None:
        raise ValueError("program AD assembly concatenate batching keeps axis static")
    operands, axis = _program_ad_assembly_concatenate_operands(args)
    operand_axes = cast(Any, axes[0])
    if operand_axes is None:
        return _as_real_numeric_array(
            "program AD assembly concatenate batched output",
            function(operands, axis),
        )
    if not isinstance(operand_axes, (tuple, list)) or len(operand_axes) != len(operands):
        raise ValueError(
            "program AD assembly concatenate batching requires one operand axis per operand"
        )

    mapped: list[tuple[NDArray[np.float64], int] | None] = []
    batch_size: int | None = None
    adjusted_axis: int | None
    if axis is None:
        adjusted_axis = None
    elif isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD assembly concatenate batching keeps axis static")
    else:
        adjusted_axis = None

    for operand_index, (operand, operand_axis) in enumerate(
        zip(operands, operand_axes, strict=True)
    ):
        if operand_axis is None:
            if axis is not None:
                raise ValueError(
                    "program AD assembly concatenate batching maps every operand for ranked axes"
                )
            mapped.append(None)
            continue
        array = _as_real_numeric_array(
            f"program AD assembly concatenate batched operand {operand_index}",
            operand,
        )
        batch_axis = _normalise_axis(f"axes[0][{operand_index}]", operand_axis, array.ndim)
        size = int(array.shape[batch_axis])
        if size <= 0:
            raise ValueError("program AD assembly concatenate batching axes must be non-empty")
        if batch_size is None:
            batch_size = size
        elif size != batch_size:
            raise ValueError(
                "program AD assembly concatenate batching axes must share one batch size"
            )
        if axis is not None:
            axis_index = _normalise_axis("axis", int(axis), array.ndim)
            if axis_index == batch_axis:
                raise ValueError(
                    "program AD assembly concatenate batching cannot map the concatenate axis"
                )
            operand_adjusted_axis = axis_index - 1 if axis_index > batch_axis else axis_index
            if adjusted_axis is None:
                adjusted_axis = operand_adjusted_axis
            elif adjusted_axis != operand_adjusted_axis:
                raise ValueError(
                    "program AD assembly concatenate batching requires one adjusted axis"
                )
        mapped.append((array, batch_axis))
    if batch_size is None:
        return _as_real_numeric_array(
            "program AD assembly concatenate batched output",
            function(operands, axis),
        )

    outputs: list[NDArray[np.float64]] = []
    for batch_index in range(batch_size):
        sliced_operands: list[object] = []
        for operand, mapped_operand in zip(operands, mapped, strict=True):
            if mapped_operand is None:
                sliced_operands.append(operand)
                continue
            array, batch_axis = mapped_operand
            sliced_operands.append(np.take(array, batch_index, axis=batch_axis))
        outputs.append(
            _as_real_numeric_array(
                "program AD assembly concatenate batched output",
                function(tuple(sliced_operands), adjusted_axis),
            )
        )
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_append_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 3 or len(axes) != 3:
        raise ValueError("program AD assembly append batching requires source, values, and axis")
    if axes[2] is not None:
        raise ValueError("program AD assembly append batching keeps axis static")
    source, values, axis = _program_ad_assembly_append_operands(args)
    if axis is not None and (isinstance(axis, bool) or not isinstance(axis, (int, np.integer))):
        raise ValueError("program AD assembly append batching keeps axis static")
    arrays = (
        _as_real_numeric_array("program AD assembly append batched source", source),
        _as_real_numeric_array("program AD assembly append batched values", values),
    )
    mapped: list[tuple[NDArray[np.float64], int] | None] = []
    batch_size: int | None = None
    adjusted_axis: int | None = None
    for operand_index, (array, operand_axis) in enumerate(zip(arrays, axes[:2], strict=True)):
        if operand_axis is None:
            if axis is not None:
                raise ValueError(
                    "program AD assembly append batching maps source and values for ranked axes"
                )
            mapped.append(None)
            continue
        batch_axis = _normalise_axis(f"axes[{operand_index}]", operand_axis, array.ndim)
        size = int(array.shape[batch_axis])
        if size <= 0:
            raise ValueError("program AD assembly append batching axes must be non-empty")
        if batch_size is None:
            batch_size = size
        elif size != batch_size:
            raise ValueError("program AD assembly append batching axes must share one batch size")
        if axis is not None:
            axis_index = _normalise_axis("axis", int(axis), array.ndim)
            if axis_index == batch_axis:
                raise ValueError("program AD assembly append batching cannot map the append axis")
            operand_adjusted_axis = axis_index - 1 if axis_index > batch_axis else axis_index
            if adjusted_axis is None:
                adjusted_axis = operand_adjusted_axis
            elif adjusted_axis != operand_adjusted_axis:
                raise ValueError("program AD assembly append batching requires one adjusted axis")
        mapped.append((array, batch_axis))
    if batch_size is None:
        return _as_real_numeric_array(
            "program AD assembly append batched output",
            function(source, values, axis),
        )

    outputs: list[NDArray[np.float64]] = []
    for batch_index in range(batch_size):
        sliced_args: list[object] = []
        for original, mapped_operand in zip((source, values), mapped, strict=True):
            if mapped_operand is None:
                sliced_args.append(original)
                continue
            array, batch_axis = mapped_operand
            sliced_args.append(np.take(array, batch_index, axis=batch_axis))
        outputs.append(
            _as_real_numeric_array(
                "program AD assembly append batched output",
                function(sliced_args[0], sliced_args[1], adjusted_axis),
            )
        )
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_block_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[object, ...],
    out_axes: int,
) -> object:
    if len(args) != 1 or len(axes) != 1:
        raise ValueError("program AD assembly block batching requires one nested layout argument")
    layout = _program_ad_assembly_block_layout(args)
    layout_axes = axes[0]
    if layout_axes is None:
        return _as_real_numeric_array(
            "program AD assembly block batched output",
            function(_program_ad_assembly_block_numpy_layout(layout)),
        )

    batch_size: int | None = None

    def map_layout(
        node: object,
        axis_node: object,
        path: str,
    ) -> object:
        nonlocal batch_size
        if isinstance(node, (tuple, list)):
            if not isinstance(axis_node, (tuple, list)) or len(axis_node) != len(node):
                raise ValueError(
                    "program AD assembly block batching requires axes matching layout"
                )
            if not node:
                raise ValueError("program AD assembly block requires a non-empty nested layout")
            return tuple(
                map_layout(child, child_axis, f"{path}.{index}")
                for index, (child, child_axis) in enumerate(zip(node, axis_node, strict=True))
            )
        if axis_node is None:
            return node
        if isinstance(axis_node, bool) or not isinstance(axis_node, (int, np.integer)):
            raise ValueError("program AD assembly block batching requires axes matching layout")
        array = _as_real_numeric_array(
            "program AD assembly block batched leaf",
            node,
        )
        axis_index = _normalise_axis(f"{path} axis", int(axis_node), array.ndim)
        current_batch_size = int(array.shape[axis_index])
        if batch_size is None:
            batch_size = current_batch_size
        elif batch_size != current_batch_size:
            raise ValueError("program AD assembly block batching requires equal batch sizes")
        return _ProgramADAssemblyBlockMappedLeaf(array=array, axis=axis_index)

    mapped_layout = map_layout(layout, layout_axes, "layout")
    if batch_size is None:
        return _as_real_numeric_array(
            "program AD assembly block batched output",
            function(_program_ad_assembly_block_numpy_layout(layout)),
        )

    def slice_layout(node: object, index: int) -> object:
        if isinstance(node, _ProgramADAssemblyBlockMappedLeaf):
            return np.take(node.array, index, axis=node.axis)
        if isinstance(node, tuple):
            return [slice_layout(child, index) for child in node]
        return node

    outputs = [
        _as_real_numeric_array(
            "program AD assembly block batched output",
            function(slice_layout(mapped_layout, batch_index)),
        )
        for batch_index in range(batch_size)
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_stack_convenience_batching_rule_for(
    name: str,
) -> PrimitiveBatchingRule:
    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        if len(args) != 1 or len(axes) != 1:
            raise ValueError(f"program AD assembly {name} batching requires operands")
        operands = _program_ad_assembly_stack_convenience_operands(name, args)
        operand_axes = cast(Any, axes[0])
        if operand_axes is None:
            return _as_real_numeric_array(
                f"program AD assembly {name} batched output",
                function(operands),
            )
        if not isinstance(operand_axes, (tuple, list)) or len(operand_axes) != len(operands):
            raise ValueError(
                f"program AD assembly {name} batching requires one operand axis per operand"
            )

        mapped: list[tuple[NDArray[np.float64], int] | None] = []
        batch_size: int | None = None
        for operand_index, (operand, operand_axis) in enumerate(
            zip(operands, operand_axes, strict=True)
        ):
            if operand_axis is None:
                mapped.append(None)
                continue
            array = _as_real_numeric_array(
                f"program AD assembly {name} batched operand {operand_index}",
                operand,
            )
            batch_axis = _normalise_axis(f"axes[0][{operand_index}]", operand_axis, array.ndim)
            current_batch_size = int(array.shape[batch_axis])
            if current_batch_size <= 0:
                raise ValueError(f"program AD assembly {name} batching axes must be non-empty")
            if batch_size is None:
                batch_size = current_batch_size
            elif batch_size != current_batch_size:
                raise ValueError(
                    f"program AD assembly {name} batching axes must share one batch size"
                )
            mapped.append((array, batch_axis))
        if batch_size is None:
            return _as_real_numeric_array(
                f"program AD assembly {name} batched output",
                function(operands),
            )

        outputs: list[NDArray[np.float64]] = []
        for batch_index in range(batch_size):
            sliced_operands: list[object] = []
            for operand, mapped_operand in zip(operands, mapped, strict=True):
                if mapped_operand is None:
                    sliced_operands.append(operand)
                    continue
                array, batch_axis = mapped_operand
                sliced_operands.append(np.take(array, batch_index, axis=batch_axis))
            outputs.append(
                _as_real_numeric_array(
                    f"program AD assembly {name} batched output",
                    function(tuple(sliced_operands)),
                )
            )
        stacked = np.stack(outputs, axis=0)
        return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))

    return batching_rule


def _program_ad_assembly_stack_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 2 or len(axes) != 2:
        raise ValueError("program AD assembly stack batching requires operands and axis")
    if axes[1] is not None:
        raise ValueError("program AD assembly stack batching keeps axis static")
    operands, axis = _program_ad_assembly_stack_operands(args)
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD assembly stack batching keeps axis static")
    operand_axes = cast(Any, axes[0])
    if operand_axes is None:
        return _as_real_numeric_array(
            "program AD assembly stack batched output",
            function(operands, axis),
        )
    if not isinstance(operand_axes, (tuple, list)) or len(operand_axes) != len(operands):
        raise ValueError(
            "program AD assembly stack batching requires one operand axis per operand"
        )

    mapped: list[tuple[NDArray[np.float64], int] | None] = []
    batch_size: int | None = None
    adjusted_axis: int | None = None
    for operand_index, (operand, operand_axis) in enumerate(
        zip(operands, operand_axes, strict=True)
    ):
        if operand_axis is None:
            mapped.append(None)
            continue
        array = _as_real_numeric_array(
            f"program AD assembly stack batched operand {operand_index}",
            operand,
        )
        batch_axis = _normalise_axis(f"axes[0][{operand_index}]", operand_axis, array.ndim)
        size = int(array.shape[batch_axis])
        if size <= 0:
            raise ValueError("program AD assembly stack batching axes must be non-empty")
        if batch_size is None:
            batch_size = size
        elif size != batch_size:
            raise ValueError("program AD assembly stack batching axes must share one batch size")
        axis_index = _normalise_axis("axis", int(axis), array.ndim + 1)
        if axis_index == batch_axis:
            raise ValueError("program AD assembly stack batching cannot map the stack axis")
        operand_adjusted_axis = axis_index - 1 if axis_index > batch_axis else axis_index
        if adjusted_axis is None:
            adjusted_axis = operand_adjusted_axis
        elif adjusted_axis != operand_adjusted_axis:
            raise ValueError("program AD assembly stack batching requires one adjusted axis")
        mapped.append((array, batch_axis))
    if batch_size is None:
        return _as_real_numeric_array(
            "program AD assembly stack batched output",
            function(operands, axis),
        )
    if adjusted_axis is None:
        raise ValueError("program AD assembly stack batching requires a mapped operand axis")

    outputs: list[NDArray[np.float64]] = []
    for batch_index in range(batch_size):
        sliced_operands: list[object] = []
        for operand, mapped_operand in zip(operands, mapped, strict=True):
            if mapped_operand is None:
                sliced_operands.append(operand)
                continue
            array, batch_axis = mapped_operand
            sliced_operands.append(np.take(array, batch_index, axis=batch_axis))
        outputs.append(
            _as_real_numeric_array(
                "program AD assembly stack batched output",
                function(tuple(sliced_operands), adjusted_axis),
            )
        )
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_like_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != len(axes):
        raise ValueError("program AD like-constructor batching axes must match arguments")
    if not args:
        raise ValueError("program AD like-constructor batching requires a reference operand")
    if axes[0] is None:
        return _as_real_numeric_array(
            "program AD like-constructor batched output", function(*args)
        )
    if any(axis is not None for axis in axes[1:]):
        raise ValueError("program AD like-constructor batching keeps fill values static")
    reference = _as_real_numeric_array("program AD like-constructor batched reference", args[0])
    batch_axis = _normalise_axis("axes[0]", axes[0], reference.ndim)
    outputs = [
        _as_real_numeric_array(
            "program AD like-constructor batched output",
            function(np.take(reference, batch_index, axis=batch_axis), *args[1:]),
        )
        for batch_index in range(int(reference.shape[batch_axis]))
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_assembly_lowering_metadata(name: str) -> Mapping[str, str]:
    if name not in _PROGRAM_AD_ASSEMBLY_IDENTITIES:
        raise ValueError(f"unsupported program AD assembly primitive {name}")
    factory_names = {
        "append": "program_ad_assembly_append_derivative_rule",
        "block": "program_ad_assembly_block_derivative_rule",
        "broadcast_arrays": "program_ad_assembly_broadcast_arrays_derivative_rule",
        "broadcast_to": "program_ad_assembly_broadcast_to_derivative_rule",
        "concatenate": "program_ad_assembly_concatenate_derivative_rule",
        "diagonal": "program_ad_assembly_diagonal_derivative_rule",
        "dstack": "program_ad_assembly_dstack_derivative_rule",
        "full_like": "program_ad_assembly_full_like_derivative_rule",
        "hstack": "program_ad_assembly_hstack_derivative_rule",
        "column_stack": "program_ad_assembly_column_stack_derivative_rule",
        "vstack": "program_ad_assembly_vstack_derivative_rule",
        "ones_like": "program_ad_assembly_ones_like_derivative_rule",
        "split": "program_ad_assembly_split_derivative_rule",
        "array_split": "program_ad_assembly_split_derivative_rule",
        "hsplit": "program_ad_assembly_split_derivative_rule",
        "vsplit": "program_ad_assembly_split_derivative_rule",
        "dsplit": "program_ad_assembly_split_derivative_rule",
        "stack": "program_ad_assembly_stack_derivative_rule",
        "tril": "program_ad_assembly_tril_derivative_rule",
        "triu": "program_ad_assembly_triu_derivative_rule",
        "zeros_like": "program_ad_assembly_zeros_like_derivative_rule",
    }
    boundaries = {
        "append": "static_source_values_shape_axis_append",
        "block": "static_nested_block_shape_layout",
        "broadcast_arrays": "static_operand_shape_broadcast_arrays",
        "broadcast_to": "static_source_shape_broadcast_to",
        "concatenate": "static_operand_shape_axis_concatenate",
        "diagonal": "static_diagonal_offset_axis_gather_scatter",
        "dstack": "static_operand_shape_dstack",
        "full_like": "static_reference_shape_scalar_fill",
        "hstack": "static_operand_shape_hstack",
        "column_stack": "static_operand_shape_column_stack",
        "vstack": "static_operand_shape_vstack",
        "ones_like": "static_reference_shape_unit_fill",
        "split": "static_split_sections_gather_scatter",
        "array_split": "static_array_split_sections_gather_scatter",
        "hsplit": "static_hsplit_sections_gather_scatter",
        "vsplit": "static_vsplit_sections_gather_scatter",
        "dsplit": "static_dsplit_sections_gather_scatter",
        "stack": "static_operand_shape_axis_stack",
        "tril": "static_lower_triangular_mask",
        "triu": "static_upper_triangular_mask",
        "zeros_like": "static_reference_shape_zero_fill",
    }
    static_signatures = {
        "append": "source_shape:ranked_tensor_shape;values_shape:ranked_tensor_shape;axis",
        "block": "layout_shapes:nested_ranked_tensor_shapes",
        "broadcast_arrays": "operand_shapes:ranked_tensor_shapes;output_shape",
        "broadcast_to": "source_shape:ranked_tensor_shape;output_shape",
        "concatenate": "operand_shapes:ranked_tensor_shapes;axis",
        "diagonal": "source_shape:rank_ge_2;offset_axis_pair;output_shape",
        "full_like": "source_shape:ranked_tensor_shape;scalar_fill",
        "hstack": "operand_shapes:ranked_tensor_shapes;output_shape",
        "vstack": "operand_shapes:ranked_tensor_shapes;output_shape",
        "column_stack": "operand_shapes:ranked_tensor_shapes;output_shape",
        "dstack": "operand_shapes:ranked_tensor_shapes;output_shape",
        "ones_like": "source_shape:ranked_tensor_shape",
        "split": "source_shape:ranked_tensor_shape;indices_or_sections;axis;part_shapes",
        "array_split": "source_shape:ranked_tensor_shape;indices_or_sections;axis;part_shapes",
        "hsplit": "source_shape:ranked_tensor_shape;indices_or_sections;axis;part_shapes",
        "vsplit": "source_shape:ranked_tensor_shape;indices_or_sections;axis;part_shapes",
        "dsplit": "source_shape:ranked_tensor_shape;indices_or_sections;axis;part_shapes",
        "stack": "operand_shapes:ranked_tensor_shapes;axis",
        "tril": "source_shape:rank_ge_2;k",
        "triu": "source_shape:rank_ge_2;k",
        "zeros_like": "source_shape:ranked_tensor_shape",
    }
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff assembly dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.assembly.{name}",
        "llvm": "blocked_until_executable_assembly_lowering",
        "rust": "blocked_until_polyglot_assembly_ad",
        "static_argument_rule": "required",
        "static_derivative_factory": factory_names[name],
        "static_signature": static_signatures[name],
        "nondifferentiable_boundary": boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
    }


def _program_ad_signal_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    raise ValueError(
        "program AD signal primitive contracts are executable only through "
        "operator-intercepted trace dispatch or fixed-shape derivative factories"
    )


def _program_ad_signal_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    raise ValueError(
        "program AD signal primitive contracts are executable only through "
        "operator-intercepted trace dispatch or fixed-shape derivative factories"
    )


def _program_ad_signal_derivative_rule(name: str) -> CustomDerivativeRule:
    if name in {"convolve", "correlate"}:
        return CustomDerivativeRule(
            name=f"program_ad_signal_{name}_trace_contract",
            value_fn=_program_ad_signal_direct_value,
            jvp_rule=_program_ad_signal_direct_jvp,
        )
    raise ValueError(f"unsupported program AD signal primitive {name}")


def _program_ad_signal_shape_of(name: str, value: object) -> tuple[int, ...]:
    shape = _program_ad_array_shape_of(value)
    if len(shape) != 1:
        raise ValueError(f"program AD signal {name} shape rule requires rank-1 operands")
    if shape[0] <= 0:
        raise ValueError(f"program AD signal {name} shape rule requires non-empty operands")
    return shape


def _program_ad_signal_convolve_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], Literal["full", "same", "valid"]]:
    if len(args) != 3:
        raise ValueError("program AD signal convolve requires left, right, and mode")
    left_shape = _program_ad_signal_shape_of("convolve", args[0])
    right_shape = _program_ad_signal_shape_of("convolve", args[1])
    mode = _normalise_convolve_mode(args[2])
    return left_shape, right_shape, mode


def _program_ad_signal_convolve_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    left_shape, right_shape, mode = _program_ad_signal_convolve_static_parts(args)
    return (_program_ad_signal_convolve_output_size(left_shape, right_shape, mode),)


def _program_ad_signal_convolve_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) != 3:
        raise ValueError("program AD signal convolve dtype rule requires left, right, and mode")
    return str(
        np.result_type(
            np.dtype(_program_ad_array_dtype_of(args[0])),
            np.dtype(_program_ad_array_dtype_of(args[1])),
        )
    )


def _program_ad_signal_convolve_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    left_shape, right_shape, mode = _program_ad_signal_convolve_static_parts(args)
    return left_shape, right_shape, mode


def _program_ad_signal_correlate_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], Literal["full", "same", "valid"]]:
    if len(args) != 3:
        raise ValueError("program AD signal correlate requires left, right, and mode")
    left_shape = _program_ad_signal_shape_of("correlate", args[0])
    right_shape = _program_ad_signal_shape_of("correlate", args[1])
    mode = _normalise_correlate_mode(args[2])
    return left_shape, right_shape, mode


def _program_ad_signal_correlate_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    left_shape, right_shape, mode = _program_ad_signal_correlate_static_parts(args)
    return (_program_ad_signal_correlate_output_size(left_shape, right_shape, mode),)


def _program_ad_signal_correlate_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    left_shape, right_shape, mode = _program_ad_signal_correlate_static_parts(args)
    return left_shape, right_shape, mode


def _program_ad_signal_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 3 or len(axes) != 3:
        raise ValueError("program AD signal convolve batching requires left, right, and mode")
    if any(axis is not None for axis in axes[1:]):
        raise ValueError("program AD signal convolve batching keeps right operand and mode static")
    if axes[0] is None:
        return _as_real_numeric_array("program AD signal convolve batched output", function(*args))

    left_batch = _as_real_numeric_array("program AD signal convolve batched left operand", args[0])
    batch_axis = _normalise_axis("axes[0]", axes[0], left_batch.ndim)
    outputs = [
        _as_real_numeric_array(
            "program AD signal convolve batched output",
            function(np.take(left_batch, batch_index, axis=batch_axis), args[1], args[2]),
        )
        for batch_index in range(left_batch.shape[batch_axis])
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_signal_lowering_metadata(name: str) -> Mapping[str, str]:
    factory_names = {
        "convolve": "program_ad_signal_convolve_derivative_rule",
        "correlate": "program_ad_signal_correlate_derivative_rule",
    }
    if name not in factory_names:
        raise ValueError(f"unsupported program AD signal primitive {name}")
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff signal dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.signal.{name}",
        "llvm": "blocked_until_executable_signal_lowering",
        "rust": "blocked_until_polyglot_signal_ad",
        "static_argument_rule": "required",
        "static_derivative_factory": factory_names[name],
        "static_signature": "left_shape:rank1;right_shape:rank1;mode",
        "nondifferentiable_boundary": "rank1_nonempty_static_mode_window",
        "nondifferentiable_boundary_policy": "fail_closed",
    }


_PROGRAM_AD_SIGNAL_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "convolve": _program_ad_signal_convolve_shape,
    "correlate": _program_ad_signal_correlate_shape,
}

_PROGRAM_AD_SIGNAL_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "convolve": _program_ad_signal_convolve_static_arguments,
    "correlate": _program_ad_signal_correlate_static_arguments,
}


_PROGRAM_AD_SHAPE_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "atleast_1d": _program_ad_shape_atleast_1d_shape,
    "atleast_2d": _program_ad_shape_atleast_2d_shape,
    "atleast_3d": _program_ad_shape_atleast_3d_shape,
    "expand_dims": _program_ad_shape_expand_dims_shape,
    "flip": _program_ad_shape_flip_shape,
    "fliplr": _program_ad_shape_fliplr_shape,
    "flipud": _program_ad_shape_flipud_shape,
    "moveaxis": _program_ad_shape_moveaxis_shape,
    "reshape": _program_ad_shape_reshape_shape,
    "ravel": _program_ad_shape_ravel_shape,
    "repeat": _program_ad_shape_repeat_shape,
    "roll": _program_ad_shape_roll_shape,
    "rot90": _program_ad_shape_rot90_shape,
    "squeeze": _program_ad_shape_squeeze_shape,
    "swapaxes": _program_ad_shape_swapaxes_shape,
    "tile": _program_ad_shape_tile_shape,
    "transpose": _program_ad_shape_transpose_shape,
}

_PROGRAM_AD_SHAPE_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "atleast_1d": _program_ad_shape_atleast_static_arguments,
    "atleast_2d": _program_ad_shape_atleast_static_arguments,
    "atleast_3d": _program_ad_shape_atleast_static_arguments,
    "expand_dims": _program_ad_shape_expand_dims_static_arguments,
    "flip": _program_ad_shape_flip_static_arguments,
    "fliplr": _program_ad_shape_no_static_arguments,
    "flipud": _program_ad_shape_no_static_arguments,
    "moveaxis": _program_ad_shape_moveaxis_static_arguments,
    "reshape": _program_ad_shape_reshape_static_arguments,
    "ravel": _program_ad_shape_no_static_arguments,
    "repeat": _program_ad_shape_repeat_static_arguments,
    "roll": _program_ad_shape_roll_static_arguments,
    "rot90": _program_ad_shape_rot90_static_arguments,
    "squeeze": _program_ad_shape_squeeze_static_arguments,
    "swapaxes": _program_ad_shape_swapaxes_static_arguments,
    "tile": _program_ad_shape_tile_static_arguments,
    "transpose": _program_ad_shape_transpose_static_arguments,
}

_PROGRAM_AD_REDUCTION_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "sum": _program_ad_reduction_shape,
    "prod": _program_ad_reduction_shape,
    "mean": _program_ad_reduction_shape,
    "var": _program_ad_reduction_var_std_shape,
    "std": _program_ad_reduction_var_std_shape,
    "max": _program_ad_order_statistic_reduction_shape,
    "min": _program_ad_order_statistic_reduction_shape,
    "median": _program_ad_order_statistic_reduction_shape,
    "quantile": _program_ad_order_statistic_reduction_shape,
    "percentile": _program_ad_order_statistic_reduction_shape,
    "trapezoid": _program_ad_reduction_trapezoid_shape,
}

_PROGRAM_AD_REDUCTION_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "sum": _program_ad_reduction_static_arguments,
    "prod": _program_ad_reduction_static_arguments,
    "mean": _program_ad_reduction_static_arguments,
    "var": _program_ad_reduction_var_std_static_arguments,
    "std": _program_ad_reduction_var_std_static_arguments,
    "max": _program_ad_reduction_order_statistic_axis_static_arguments,
    "min": _program_ad_reduction_order_statistic_axis_static_arguments,
    "median": _program_ad_reduction_order_statistic_axis_static_arguments,
    "quantile": _program_ad_reduction_quantile_static_arguments,
    "percentile": _program_ad_reduction_percentile_static_arguments,
    "trapezoid": _program_ad_reduction_static_arguments,
}

_PROGRAM_AD_ELEMENTWISE_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    name: _program_ad_elementwise_shape for name in _PROGRAM_AD_ELEMENTWISE_NAMES
}

_PROGRAM_AD_SELECTION_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "where": _program_ad_selection_where_shape,
    "clip": _program_ad_selection_clip_shape,
    "sort": _program_ad_selection_sort_shape,
    "select": _program_ad_selection_select_shape,
    "piecewise": _program_ad_selection_piecewise_shape,
    "choose": _program_ad_selection_choose_shape,
    "compress": _program_ad_selection_compress_shape,
    "extract": _program_ad_selection_extract_shape,
    "argmax": _program_ad_selection_index_reduce_shape,
    "argmin": _program_ad_selection_index_reduce_shape,
    "argsort": _program_ad_selection_argsort_shape,
}

_PROGRAM_AD_SELECTION_DTYPE_RULES: Mapping[str, PrimitiveDTypeRule] = {
    "where": _program_ad_selection_where_dtype_rule,
    "clip": _program_ad_selection_clip_dtype_rule,
    "sort": _program_ad_selection_sort_dtype_rule,
    "select": _program_ad_selection_select_dtype_rule,
    "piecewise": _program_ad_selection_piecewise_dtype_rule,
    "choose": _program_ad_selection_choose_dtype_rule,
    "compress": _program_ad_selection_compress_dtype_rule,
    "extract": _program_ad_selection_extract_dtype_rule,
    "argmax": _program_ad_selection_index_dtype_rule,
    "argmin": _program_ad_selection_index_dtype_rule,
    "argsort": _program_ad_selection_index_dtype_rule,
}

_PROGRAM_AD_SELECTION_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "where": _program_ad_selection_where_static_arguments,
    "clip": _program_ad_selection_clip_static_arguments,
    "sort": _program_ad_selection_sort_static_arguments,
    "select": _program_ad_selection_select_static_arguments,
    "piecewise": _program_ad_selection_piecewise_static_arguments,
    "choose": _program_ad_selection_choose_static_arguments,
    "compress": _program_ad_selection_compress_static_arguments,
    "extract": _program_ad_selection_extract_static_arguments,
    "argmax": _program_ad_selection_index_reduce_static_arguments,
    "argmin": _program_ad_selection_index_reduce_static_arguments,
    "argsort": _program_ad_selection_argsort_static_arguments,
}

_PROGRAM_AD_PRODUCT_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "dot": _program_ad_product_dot_shape,
    "vdot": _program_ad_product_vdot_shape,
    "inner": _program_ad_product_inner_shape,
    "outer": _program_ad_product_outer_shape,
    "matmul": _program_ad_product_matmul_shape,
    "tensordot": _program_ad_product_tensordot_shape,
    "einsum": _program_ad_product_einsum_shape,
}

_PROGRAM_AD_CUMULATIVE_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "cumsum": _program_ad_cumulative_scan_shape,
    "cumprod": _program_ad_cumulative_scan_shape,
    "diff": _program_ad_cumulative_diff_shape,
}

_PROGRAM_AD_CUMULATIVE_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "cumsum": _program_ad_cumulative_scan_static_arguments,
    "cumprod": _program_ad_cumulative_scan_static_arguments,
    "diff": _program_ad_cumulative_diff_static_arguments,
}


def _program_ad_array_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != len(axes):
        raise ValueError("program AD array batching axes must match argument count")
    if not args:
        raise ValueError("program AD array batching requires an array operand")
    array = _as_real_numeric_array("program AD array batched operand", args[0])
    axis = axes[0]
    if axis is None:
        raise ValueError("program AD array batching requires the array operand to be mapped")
    axis_index = _normalise_axis("axes[0]", axis, array.ndim)
    batch_size = int(array.shape[axis_index])
    if any(item is not None for item in axes[1:]):
        raise ValueError("program AD array batching supports static non-array arguments only")
    outputs = [
        _as_real_numeric_array(
            "program AD array batched output",
            function(np.take(array, batch_index, axis=axis_index), *args[1:]),
        )
        for batch_index in range(batch_size)
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_array_lowering_metadata(name: str) -> Mapping[str, str]:
    static_signature = {
        "getitem": "source_shape:ranked_tensor_shape;index:static_gather_index",
        "take": "source_shape:ranked_tensor_shape;indices_axis_mode",
        "take_along_axis": "source_shape:ranked_tensor_shape;indices_shape_axis",
        "delete": "source_shape:ranked_tensor_shape;object_axis",
        "pad": "source_shape:ranked_tensor_shape;pad_width_constant_values",
        "insert": "source_shape:ranked_tensor_shape;object_values_axis",
    }[name]
    static_factory = {
        "getitem": "program_ad_array_getitem_derivative_rule",
        "take": "program_ad_array_take_derivative_rule",
        "take_along_axis": "program_ad_array_take_along_axis_derivative_rule",
        "delete": "program_ad_array_delete_derivative_rule",
        "pad": "program_ad_array_pad_derivative_rule",
        "insert": "program_ad_array_insert_derivative_rule",
    }[name]
    nondifferentiable_boundaries = {
        "getitem": "static_gather_index_scatter_add",
        "take": "static_integer_gather_scatter_add",
        "take_along_axis": "static_along_axis_gather_scatter_add",
        "delete": "static_delete_gather_scatter_add",
        "pad": "static_constant_pad_scatter_add",
        "insert": "static_constant_insert_scatter_add",
    }
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff array dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.array.{name}",
        "llvm": "blocked_until_executable_array_lowering",
        "rust": "blocked_until_polyglot_array_ad",
        "static_argument_rule": "required",
        "static_derivative_factory": static_factory,
        "static_signature": static_signature,
        "nondifferentiable_boundary": nondifferentiable_boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
    }


def _program_ad_shape_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != len(axes):
        raise ValueError("program AD shape batching axes must match argument count")
    if not args:
        raise ValueError("program AD shape batching requires an array operand")
    array = _as_real_numeric_array("program AD shape batched operand", args[0])
    axis = axes[0]
    if axis is None:
        return function(*args)
    if any(item is not None for item in axes[1:]):
        raise ValueError("program AD shape batching supports static non-array arguments only")
    axis_index = _normalise_axis("axes[0]", axis, array.ndim)
    outputs = [
        _as_real_numeric_array(
            "program AD shape batched output",
            function(np.take(array, batch_index, axis=axis_index), *args[1:]),
        )
        for batch_index in range(int(array.shape[axis_index]))
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_shape_lowering_metadata(name: str) -> Mapping[str, str]:
    static_factory = {
        "atleast_1d": "program_ad_shape_atleast_1d_derivative_rule",
        "atleast_2d": "program_ad_shape_atleast_2d_derivative_rule",
        "atleast_3d": "program_ad_shape_atleast_3d_derivative_rule",
        "expand_dims": "program_ad_shape_expand_dims_derivative_rule",
        "flip": "program_ad_shape_flip_derivative_rule",
        "fliplr": "program_ad_shape_fliplr_derivative_rule",
        "flipud": "program_ad_shape_flipud_derivative_rule",
        "moveaxis": "program_ad_shape_moveaxis_derivative_rule",
        "repeat": "program_ad_shape_repeat_derivative_rule",
        "reshape": "program_ad_shape_reshape_derivative_rule",
        "ravel": "program_ad_shape_ravel_derivative_rule",
        "roll": "program_ad_shape_roll_derivative_rule",
        "rot90": "program_ad_shape_rot90_derivative_rule",
        "squeeze": "program_ad_shape_squeeze_derivative_rule",
        "swapaxes": "program_ad_shape_swapaxes_derivative_rule",
        "tile": "program_ad_shape_tile_derivative_rule",
        "transpose": "program_ad_shape_transpose_derivative_rule",
    }[name]
    static_signature = {
        "atleast_1d": "source_shape:ranked_tensor_shape",
        "atleast_2d": "source_shape:ranked_tensor_shape",
        "atleast_3d": "source_shape:ranked_tensor_shape",
        "expand_dims": "source_shape:ranked_tensor_shape;axis",
        "flip": "source_shape:ranked_tensor_shape;axis",
        "fliplr": "source_shape:rank_ge_2",
        "flipud": "source_shape:rank_ge_1",
        "moveaxis": "source_shape:ranked_tensor_shape;source_destination",
        "repeat": "source_shape:ranked_tensor_shape;repeats_axis",
        "reshape": "source_shape:ranked_tensor_shape;target_shape",
        "ravel": "source_shape:ranked_tensor_shape",
        "roll": "source_shape:ranked_tensor_shape;shift_axis",
        "rot90": "source_shape:ranked_tensor_shape;k_axes",
        "squeeze": "source_shape:ranked_tensor_shape;axis",
        "swapaxes": "source_shape:ranked_tensor_shape;axis1_axis2",
        "tile": "source_shape:ranked_tensor_shape;reps",
        "transpose": "source_shape:ranked_tensor_shape;axes",
    }[name]
    nondifferentiable_boundaries = {
        "atleast_1d": "static_rank_promotion",
        "atleast_2d": "static_rank_promotion",
        "atleast_3d": "static_rank_promotion",
        "expand_dims": "static_singleton_axis_insertion",
        "flip": "static_axis_flip_permutation",
        "fliplr": "static_second_axis_flip_permutation",
        "flipud": "static_first_axis_flip_permutation",
        "moveaxis": "static_axis_move_permutation",
        "repeat": "static_repeat_scatter_add",
        "reshape": "element_count_preserving_static_shape",
        "ravel": "contiguous_flat_view_shape",
        "roll": "static_integer_roll_permutation",
        "rot90": "static_quarter_turn_axis_permutation",
        "squeeze": "static_singleton_axis_removal",
        "swapaxes": "static_axis_swap_permutation",
        "tile": "static_tile_scatter_add",
        "transpose": "static_axis_permutation",
    }
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff shape dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.shape.{name}",
        "llvm": "blocked_until_executable_shape_lowering",
        "rust": "blocked_until_polyglot_shape_ad",
        "static_argument_rule": "required",
        "static_derivative_factory": static_factory,
        "static_signature": static_signature,
        "nondifferentiable_boundary": nondifferentiable_boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
    }


def _program_ad_reduction_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != len(axes):
        raise ValueError("program AD reduction batching axes must match argument count")
    if not args:
        raise ValueError("program AD reduction batching requires an array operand")
    array = _as_real_numeric_array("program AD reduction batched operand", args[0])
    batch_axis = axes[0]
    if batch_axis is None:
        return function(*args)
    if any(item is not None for item in axes[1:]):
        raise ValueError("program AD reduction batching supports static axes only")
    batch_axis = _normalise_axis("axes[0]", batch_axis, array.ndim)
    order_statistic = len(args) == 4 and isinstance(args[3], str)
    if order_statistic:
        reduction_axis = _program_ad_order_statistic_reduction_axis(args)
    elif len(args) == 3:
        reduction_axis, ddof = _program_ad_reduction_axis_ddof(args)
    elif len(args) == 4:
        reduction_axis = _program_ad_reduction_trapezoid_axis(args)
    else:
        reduction_axis = _program_ad_reduction_axis(args)
    if reduction_axis is not None:
        reduction_axis = _normalise_axis("reduction axis", reduction_axis, array.ndim)
        if reduction_axis == batch_axis:
            raise ValueError("program AD reduction batching cannot reduce the mapped batch axis")
        if reduction_axis > batch_axis:
            reduction_axis -= 1
    static_tail: tuple[object, ...] = (reduction_axis,)
    if order_statistic:
        q = args[1]
        method = args[3]
        outputs = [
            _as_real_numeric_array(
                "program AD reduction batched output",
                function(
                    np.take(array, batch_index, axis=batch_axis),
                    q,
                    axis=reduction_axis,
                    method=method,
                ),
            )
            for batch_index in range(int(array.shape[batch_axis]))
        ]
        stacked = np.stack(outputs, axis=0)
        return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))
    if len(args) == 3:
        outputs = [
            _as_real_numeric_array(
                "program AD reduction batched output",
                function(
                    np.take(array, batch_index, axis=batch_axis),
                    axis=reduction_axis,
                    ddof=ddof,
                ),
            )
            for batch_index in range(int(array.shape[batch_axis]))
        ]
        stacked = np.stack(outputs, axis=0)
        return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))
    if len(args) == 4 and args[1] is not None:
        x_array = _as_real_numeric_array("program AD trapezoid batched x", args[1])
        if tuple(x_array.shape) == tuple(array.shape):
            raise ValueError(
                "program AD trapezoid batching requires scalar dx or one-dimensional static x"
            )
        static_tail = (args[1], args[2], reduction_axis)
    outputs = [
        _as_real_numeric_array(
            "program AD reduction batched output",
            function(np.take(array, batch_index, axis=batch_axis), *static_tail),
        )
        for batch_index in range(int(array.shape[batch_axis]))
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_reduction_lowering_metadata(name: str) -> Mapping[str, str]:
    static_factory = {
        "sum": "program_ad_reduction_sum_derivative_rule",
        "prod": "program_ad_reduction_prod_derivative_rule",
        "mean": "program_ad_reduction_mean_derivative_rule",
        "var": "program_ad_reduction_var_derivative_rule",
        "std": "program_ad_reduction_std_derivative_rule",
        "max": "program_ad_reduction_max_derivative_rule",
        "min": "program_ad_reduction_min_derivative_rule",
        "median": "program_ad_reduction_median_derivative_rule",
        "quantile": "program_ad_reduction_quantile_derivative_rule",
        "percentile": "program_ad_reduction_percentile_derivative_rule",
        "trapezoid": "program_ad_reduction_trapezoid_derivative_rule",
    }[name]
    nondifferentiable_boundaries = {
        "sum": "static_axis_and_stable_output_shape",
        "prod": "static_axis_zero_factor_sensitive",
        "mean": "static_axis_nonempty_reduction",
        "var": "static_axis_ddof_positive_denominator",
        "std": "static_axis_ddof_positive_denominator_nonzero_variance",
        "max": "static_axis_unique_max_selector",
        "min": "static_axis_unique_min_selector",
        "median": "static_axis_strict_order_selection",
        "quantile": "static_scalar_q_axis_method_strict_order_selection",
        "percentile": "static_scalar_q_axis_method_strict_order_selection",
        "trapezoid": "static_axis_and_static_grid_spacing",
    }
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff reduction dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.reduction.{name}",
        "llvm": "blocked_until_executable_reduction_lowering",
        "rust": "blocked_until_polyglot_reduction_ad",
        "static_argument_rule": "required",
        "static_derivative_factory": static_factory,
        "static_signature": (
            "source_shape:ranked_tensor_shape;x_or_dx;axis"
            if name == "trapezoid"
            else (
                "source_shape:ranked_tensor_shape;q;axis;method"
                if name in {"quantile", "percentile"}
                else (
                    "source_shape:ranked_tensor_shape;axis;ddof"
                    if name in {"var", "std"}
                    else "source_shape:ranked_tensor_shape;axis"
                )
            )
        ),
        "nondifferentiable_boundary": nondifferentiable_boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
    }


def _program_ad_elementwise_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != len(axes):
        raise ValueError("program AD elementwise batching axes must match argument count")
    if len(args) not in {1, 2}:
        raise ValueError("program AD elementwise batching requires one or two operands")
    arrays = tuple(
        _as_real_numeric_array(f"program AD elementwise batched operand {index}", arg)
        for index, arg in enumerate(args)
    )
    if all(axis is None for axis in axes):
        result = _as_real_numeric_array("program AD elementwise batched output", function(*arrays))
        return result
    mapped_axes: list[int | None] = [
        None if axis is None else _normalise_axis(f"axes[{index}]", axis, array.ndim)
        for index, (axis, array) in enumerate(zip(axes, arrays, strict=True))
    ]
    batch_sizes = {
        int(array.shape[axis])
        for array, axis in zip(arrays, mapped_axes, strict=True)
        if axis is not None
    }
    if len(batch_sizes) != 1:
        raise ValueError("program AD elementwise batching axes must share one batch size")
    batch_size = batch_sizes.pop()
    outputs = []
    for batch_index in range(batch_size):
        sliced_args = tuple(
            array if axis is None else np.take(array, batch_index, axis=axis)
            for array, axis in zip(arrays, mapped_axes, strict=True)
        )
        outputs.append(
            _as_real_numeric_array("program AD elementwise batched output", function(*sliced_args))
        )
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_elementwise_lowering_metadata(name: str) -> Mapping[str, str]:
    is_binary = name in _PROGRAM_AD_ELEMENTWISE_BINARY_NAMES
    is_derivative_losing = name in _PROGRAM_AD_ELEMENTWISE_DISCONTINUOUS_NAMES
    metadata = {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff elementwise dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.elementwise.{name}",
        "llvm": "blocked_until_executable_elementwise_lowering",
        "rust": "blocked_until_polyglot_elementwise_ad",
        "static_argument_rule": "none",
        "static_derivative_factory": (
            "blocked_derivative_losing"
            if is_derivative_losing
            else "program_ad_elementwise_binary_derivative_rule"
            if is_binary
            else "not_required"
        ),
        "static_signature": (
            "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape"
            if is_binary
            else "source_shape:ranked_tensor_shape;step_value"
            if name == "heaviside"
            else "source_shape:ranked_tensor_shape"
            if is_derivative_losing
            else "none"
        ),
    }
    nondifferentiable_boundaries = {
        "log": "positive_domain",
        "log1p": "greater_than_minus_one_domain",
        "sqrt": "nonnegative_domain_with_singular_zero_derivative",
        "arcsin": "closed_unit_interval_with_singular_endpoints",
        "arccos": "closed_unit_interval_with_singular_endpoints",
        "reciprocal": "nonzero_domain",
        "abs": "zero_cusp",
        "divide": "nonzero_denominator",
        "power": "positive_base_for_variable_exponent",
        "maximum": "equal_operand_tie",
        "minimum": "equal_operand_tie",
        "sign": "sign_step_derivative_losing_boundary",
        "heaviside": "heaviside_step_derivative_losing_boundary",
    }
    boundary = nondifferentiable_boundaries.get(name)
    if boundary is not None:
        metadata["nondifferentiable_boundary"] = boundary
    else:
        metadata["nondifferentiable_boundary"] = "none"
    metadata["nondifferentiable_boundary_policy"] = "fail_closed"
    return metadata


def _program_ad_selection_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 3 or len(axes) != 3:
        raise ValueError("program AD selection batching requires three operands and axes")
    arrays = tuple(np.asarray(arg) for arg in args)
    if all(axis is None for axis in axes):
        return function(*args)
    mapped_axes: list[int | None] = [
        None if axis is None else _normalise_axis(f"axes[{index}]", axis, array.ndim)
        for index, (axis, array) in enumerate(zip(axes, arrays, strict=True))
    ]
    batch_sizes = {
        int(array.shape[axis])
        for array, axis in zip(arrays, mapped_axes, strict=True)
        if axis is not None
    }
    if len(batch_sizes) != 1:
        raise ValueError("program AD selection batching axes must share one batch size")
    batch_size = batch_sizes.pop()
    outputs = []
    for batch_index in range(batch_size):
        sliced_args = tuple(
            array if axis is None else np.take(array, batch_index, axis=axis)
            for array, axis in zip(arrays, mapped_axes, strict=True)
        )
        outputs.append(np.asarray(function(*sliced_args)))
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_selection_sort_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) not in {1, 2, 3} or len(args) != len(axes):
        raise ValueError("program AD selection sort batching requires source, axis, and kind")
    if len(axes) >= 2 and any(axis is not None for axis in axes[1:]):
        raise ValueError("program AD selection sort batching requires static axis and kind")
    source = _as_real_numeric_array("program AD selection sort batched source", args[0])
    batch_axis = axes[0]
    if batch_axis is None:
        return function(*args)
    batch_axis_index = _normalise_axis("axes[0]", batch_axis, source.ndim)
    outputs = [
        _as_real_numeric_array(
            "program AD selection sort batched output",
            function(np.take(source, batch_index, axis=batch_axis_index), *args[1:]),
        )
        for batch_index in range(int(source.shape[batch_axis_index]))
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_selection_index_batching_rule(
    _function: Callable[..., object],
    _args: tuple[object, ...],
    _axes: tuple[int | None, ...],
    _out_axes: int,
) -> NoReturn:
    raise ValueError(
        "program AD argmax/argmin/argsort batching is unsupported because integer "
        "index selection is nondifferentiable"
    )


def _program_ad_selection_batching_rule_for(name: str) -> PrimitiveBatchingRule:
    if name == "sort":
        return _program_ad_selection_sort_batching_rule
    if name in {"argmax", "argmin", "argsort"}:
        return _program_ad_selection_index_batching_rule
    return _program_ad_selection_batching_rule


def _program_ad_selection_lowering_metadata(name: str) -> Mapping[str, str]:
    static_factories = {
        "where": "program_ad_selection_where_derivative_rule",
        "clip": "program_ad_selection_clip_derivative_rule",
        "sort": "operator_intercepted_sort_permutation_trace",
        "select": "operator_intercepted_static_select_fold_trace",
        "piecewise": "operator_intercepted_static_piecewise_fold_trace",
        "choose": "operator_intercepted_static_choose_gather_trace",
        "compress": "operator_intercepted_static_compress_gather_trace",
        "extract": "operator_intercepted_static_extract_gather_trace",
        "argmax": "unsupported_nondifferentiable_index_selection",
        "argmin": "unsupported_nondifferentiable_index_selection",
        "argsort": "unsupported_nondifferentiable_index_selection",
    }
    static_signatures = {
        "where": (
            "condition:static_bool_mask;true_shape:ranked_tensor_shape;"
            "false_shape:ranked_tensor_shape"
        ),
        "clip": (
            "source_shape:ranked_tensor_shape;lower_shape:ranked_tensor_shape;"
            "upper_shape:ranked_tensor_shape"
        ),
        "sort": "source_shape:ranked_tensor_shape;axis_kind",
        "select": "condition_sequence;choice_shapes;default_shape",
        "piecewise": "source_shape;condition_sequence;function_count",
        "choose": "selector_shape;choice_shapes;mode",
        "compress": "source_shape;condition_indices;axis",
        "extract": "source_shape;condition_indices",
        "argmax": "source_shape:ranked_tensor_shape;axis",
        "argmin": "source_shape:ranked_tensor_shape;axis",
        "argsort": "source_shape:ranked_tensor_shape;axis_kind",
    }
    nondifferentiable_boundaries = {
        "where": "predicate_branch_boundary",
        "clip": "clipping_boundary_and_bound_order",
        "sort": "strict_total_order_required",
        "select": "static_condition_sequence_branch_fold",
        "piecewise": "static_condition_sequence_callable_fold",
        "choose": "static_integer_selector_gather",
        "compress": "static_boolean_mask_gather",
        "extract": "static_boolean_mask_flat_gather",
        "argmax": "integer_index_selection_nondifferentiable",
        "argmin": "integer_index_selection_nondifferentiable",
        "argsort": "integer_index_permutation_nondifferentiable",
    }
    return {
        "program_ad": (
            "unsupported_index_selection_fail_closed"
            if name in {"argmax", "argmin", "argsort"}
            else "operator_intercepted_trace"
        ),
        "mlir": "available: scpn_diff selection dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.selection.{name}",
        "llvm": "blocked_until_executable_selection_lowering",
        "rust": "blocked_until_polyglot_selection_ad",
        "static_argument_rule": (
            "required"
            if name
            in {
                "where",
                "select",
                "piecewise",
                "choose",
                "compress",
                "extract",
                "argmax",
                "argmin",
                "argsort",
            }
            else "none"
        ),
        "static_derivative_factory": static_factories[name],
        "static_signature": static_signatures[name],
        "nondifferentiable_boundary": nondifferentiable_boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
    }


def _program_ad_product_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if args and isinstance(args[0], str):
        if len(args) != len(axes):
            raise ValueError("program AD product einsum batching axes must match arguments")
        if axes[0] is not None:
            raise ValueError("program AD product einsum batching requires static subscripts")
        arrays = tuple(
            _as_real_numeric_array(f"program AD product einsum batched operand {index}", arg)
            for index, arg in enumerate(args[1:])
        )
        if not arrays:
            raise ValueError("program AD product einsum batching requires operands")
        if all(axis is None for axis in axes[1:]):
            return _as_real_numeric_array(
                "program AD product einsum batched output", function(*args)
            )
        einsum_mapped_axes: list[int | None] = [
            None if axis is None else _normalise_axis(f"axes[{index + 1}]", axis, array.ndim)
            for index, (axis, array) in enumerate(zip(axes[1:], arrays, strict=True))
        ]
        batch_sizes = {
            int(array.shape[axis])
            for array, axis in zip(arrays, einsum_mapped_axes, strict=True)
            if axis is not None
        }
        if len(batch_sizes) != 1:
            raise ValueError("program AD product einsum batching axes must share one batch size")
        batch_size = batch_sizes.pop()
        outputs = []
        for batch_index in range(batch_size):
            sliced_args = (
                args[0],
                *(
                    array if axis is None else np.take(array, batch_index, axis=axis)
                    for array, axis in zip(arrays, einsum_mapped_axes, strict=True)
                ),
            )
            outputs.append(
                _as_real_numeric_array(
                    "program AD product einsum batched output",
                    function(*sliced_args),
                )
            )
        stacked = np.stack(outputs, axis=0)
        return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))
    if len(args) == 3:
        if len(axes) != 3:
            raise ValueError("program AD product tensordot batching axes must match arguments")
        if axes[2] is not None:
            raise ValueError("program AD product tensordot batching requires static axes")
        arrays = tuple(
            _as_real_numeric_array(f"program AD product tensordot batched operand {index}", arg)
            for index, arg in enumerate(args[:2])
        )
        if all(axis is None for axis in axes[:2]):
            return _as_real_numeric_array(
                "program AD product tensordot batched output",
                function(*arrays, args[2]),
            )
        tensordot_mapped_axes: list[int | None] = [
            None if axis is None else _normalise_axis(f"axes[{index}]", axis, array.ndim)
            for index, (axis, array) in enumerate(zip(axes[:2], arrays, strict=True))
        ]
        batch_sizes = {
            int(array.shape[axis])
            for array, axis in zip(arrays, tensordot_mapped_axes, strict=True)
            if axis is not None
        }
        if len(batch_sizes) != 1:
            raise ValueError(
                "program AD product tensordot batching axes must share one batch size"
            )
        batch_size = batch_sizes.pop()
        outputs = []
        for batch_index in range(batch_size):
            sliced_args = (
                *(
                    array if axis is None else np.take(array, batch_index, axis=axis)
                    for array, axis in zip(arrays, tensordot_mapped_axes, strict=True)
                ),
                args[2],
            )
            outputs.append(
                _as_real_numeric_array(
                    "program AD product tensordot batched output",
                    function(*sliced_args),
                )
            )
        stacked = np.stack(outputs, axis=0)
        return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))
    if len(args) != 2 or len(axes) != 2:
        raise ValueError("program AD product batching requires two operands and two axes")
    arrays = tuple(
        _as_real_numeric_array(f"program AD product batched operand {index}", arg)
        for index, arg in enumerate(args)
    )
    if all(axis is None for axis in axes):
        return _as_real_numeric_array("program AD product batched output", function(*arrays))
    mapped_axes: list[int | None] = [
        None if axis is None else _normalise_axis(f"axes[{index}]", axis, array.ndim)
        for index, (axis, array) in enumerate(zip(axes, arrays, strict=True))
    ]
    batch_sizes = {
        int(array.shape[axis])
        for array, axis in zip(arrays, mapped_axes, strict=True)
        if axis is not None
    }
    if len(batch_sizes) != 1:
        raise ValueError("program AD product batching axes must share one batch size")
    batch_size = batch_sizes.pop()
    outputs = []
    for batch_index in range(batch_size):
        sliced_args = tuple(
            array if axis is None else np.take(array, batch_index, axis=axis)
            for array, axis in zip(arrays, mapped_axes, strict=True)
        )
        outputs.append(
            _as_real_numeric_array("program AD product batched output", function(*sliced_args))
        )
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_product_lowering_metadata(name: str) -> Mapping[str, str]:
    static_factories = {
        "dot": "not_required",
        "vdot": "not_required",
        "inner": "program_ad_product_inner_derivative_rule",
        "outer": "program_ad_product_outer_derivative_rule",
        "matmul": "program_ad_product_matmul_derivative_rule",
        "tensordot": "program_ad_product_tensordot_derivative_rule",
        "einsum": "program_ad_product_einsum_derivative_rule",
    }
    static_signatures = {
        "dot": "none",
        "vdot": "none",
        "inner": "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape",
        "outer": "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape",
        "matmul": "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape",
        "tensordot": "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape;axes",
        "einsum": "subscripts:explicit_static;operand_shapes:ranked_tensor_shapes",
    }
    nondifferentiable_boundaries = {
        "dot": "inner_dimension_alignment",
        "vdot": "flattened_size_alignment",
        "inner": "last_dimension_alignment",
        "outer": "flattened_outer_product",
        "matmul": "core_dimension_alignment",
        "tensordot": "static_axes_tensor_contraction",
        "einsum": "explicit_static_tensor_contraction",
    }
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff product dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.product.{name}",
        "llvm": "blocked_until_executable_product_lowering",
        "rust": "blocked_until_polyglot_product_ad",
        "static_argument_rule": "required" if name in {"einsum", "tensordot"} else "none",
        "static_derivative_factory": static_factories[name],
        "static_signature": static_signatures[name],
        "nondifferentiable_boundary": nondifferentiable_boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
    }


def _program_ad_cumulative_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != len(axes):
        raise ValueError("program AD cumulative batching axes must match argument count")
    if not args:
        raise ValueError("program AD cumulative batching requires an array operand")
    array = _as_real_numeric_array("program AD cumulative batched operand", args[0])
    batch_axis = axes[0]
    if batch_axis is None:
        return _as_real_numeric_array("program AD cumulative batched output", function(*args))
    if any(item is not None for item in axes[1:]):
        raise ValueError("program AD cumulative batching supports static parameters only")
    batch_axis = _normalise_axis("axes[0]", batch_axis, array.ndim)
    outputs = [
        _as_real_numeric_array(
            "program AD cumulative batched output",
            function(np.take(array, batch_index, axis=batch_axis), *args[1:]),
        )
        for batch_index in range(int(array.shape[batch_axis]))
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_cumulative_lowering_metadata(name: str) -> Mapping[str, str]:
    static_factory = {
        "cumsum": "program_ad_cumulative_cumsum_derivative_rule",
        "cumprod": "program_ad_cumulative_cumprod_derivative_rule",
        "diff": "program_ad_cumulative_diff_derivative_rule",
    }[name]
    nondifferentiable_boundaries = {
        "cumsum": "ordered_axis_sequence",
        "cumprod": "ordered_axis_zero_factor_sensitive",
        "diff": "finite_difference_order_and_spacing",
    }
    static_signature = (
        "source_shape:ranked_tensor_shape;order_axis"
        if name == "diff"
        else "source_shape:ranked_tensor_shape;axis"
    )
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff cumulative dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.cumulative.{name}",
        "llvm": "blocked_until_executable_cumulative_lowering",
        "rust": "blocked_until_polyglot_cumulative_ad",
        "static_argument_rule": "required",
        "static_derivative_factory": static_factory,
        "static_signature": static_signature,
        "nondifferentiable_boundary": nondifferentiable_boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
    }


def _register_program_ad_array_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_ARRAY_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_array_derivative_rule(name),
                batching_rule=_program_ad_array_batching_rule,
                lowering_metadata=_program_ad_array_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_ARRAY_SHAPE_RULES[name],
                dtype_rule=_program_ad_array_dtype_rule,
                static_argument_rule=_PROGRAM_AD_ARRAY_STATIC_ARGUMENT_RULES[name],
                nondifferentiable_policy=_PROGRAM_AD_ARRAY_POLICY,
                effect="pure",
            )
        )


def _register_program_ad_interpolation_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_INTERPOLATION_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_interpolation_derivative_rule(name),
                batching_rule=_program_ad_interpolation_batching_rule,
                lowering_metadata=_program_ad_interpolation_lowering_metadata(name),
                shape_rule=_program_ad_interpolation_interp_shape,
                dtype_rule=_program_ad_interpolation_interp_dtype_rule,
                static_argument_rule=_program_ad_interpolation_interp_static_arguments,
                nondifferentiable_policy=_PROGRAM_AD_INTERPOLATION_POLICY,
                effect="pure",
            )
        )


def _register_program_ad_assembly_primitive_contracts() -> None:
    batching_rules: Mapping[str, PrimitiveBatchingRule] = {
        "append": _program_ad_assembly_append_batching_rule,
        "block": _program_ad_assembly_block_batching_rule,
        "broadcast_arrays": _program_ad_assembly_broadcast_arrays_batching_rule,
        "broadcast_to": _program_ad_assembly_broadcast_to_batching_rule,
        "concatenate": _program_ad_assembly_batching_rule,
        "diagonal": _program_ad_assembly_diagonal_batching_rule,
        "dstack": _program_ad_assembly_stack_convenience_batching_rule_for("dstack"),
        "full_like": _program_ad_assembly_like_batching_rule,
        "hstack": _program_ad_assembly_stack_convenience_batching_rule_for("hstack"),
        "column_stack": _program_ad_assembly_stack_convenience_batching_rule_for("column_stack"),
        "ones_like": _program_ad_assembly_like_batching_rule,
        "split": _program_ad_assembly_split_batching_rule_for("split"),
        "array_split": _program_ad_assembly_split_batching_rule_for("array_split"),
        "hsplit": _program_ad_assembly_split_batching_rule_for("hsplit"),
        "vsplit": _program_ad_assembly_split_batching_rule_for("vsplit"),
        "dsplit": _program_ad_assembly_split_batching_rule_for("dsplit"),
        "stack": _program_ad_assembly_stack_batching_rule,
        "tril": _program_ad_assembly_triangular_batching_rule_for("tril"),
        "triu": _program_ad_assembly_triangular_batching_rule_for("triu"),
        "vstack": _program_ad_assembly_stack_convenience_batching_rule_for("vstack"),
        "zeros_like": _program_ad_assembly_like_batching_rule,
    }
    shape_rules: Mapping[str, PrimitiveShapeRule] = {
        "append": _program_ad_assembly_append_shape,
        "block": _program_ad_assembly_block_shape,
        "broadcast_arrays": _program_ad_assembly_broadcast_arrays_shape,
        "broadcast_to": _program_ad_assembly_broadcast_to_shape,
        "concatenate": _program_ad_assembly_concatenate_shape,
        "diagonal": _program_ad_assembly_diagonal_shape,
        "dstack": _program_ad_assembly_stack_convenience_shape_rule_for("dstack"),
        "full_like": _program_ad_assembly_full_like_shape,
        "hstack": _program_ad_assembly_stack_convenience_shape_rule_for("hstack"),
        "column_stack": _program_ad_assembly_stack_convenience_shape_rule_for("column_stack"),
        "ones_like": _program_ad_assembly_ones_like_shape,
        "split": _program_ad_assembly_split_shape_rule_for("split"),
        "array_split": _program_ad_assembly_split_shape_rule_for("array_split"),
        "hsplit": _program_ad_assembly_split_shape_rule_for("hsplit"),
        "vsplit": _program_ad_assembly_split_shape_rule_for("vsplit"),
        "dsplit": _program_ad_assembly_split_shape_rule_for("dsplit"),
        "stack": _program_ad_assembly_stack_shape,
        "tril": _program_ad_assembly_triangular_shape,
        "triu": _program_ad_assembly_triangular_shape,
        "vstack": _program_ad_assembly_stack_convenience_shape_rule_for("vstack"),
        "zeros_like": _program_ad_assembly_zeros_like_shape,
    }
    dtype_rules: Mapping[str, PrimitiveDTypeRule] = {
        "append": _program_ad_assembly_append_dtype_rule,
        "block": _program_ad_assembly_block_dtype_rule,
        "broadcast_arrays": _program_ad_assembly_broadcast_arrays_dtype_rule,
        "broadcast_to": _program_ad_assembly_broadcast_to_dtype_rule,
        "concatenate": _program_ad_assembly_concatenate_dtype_rule,
        "diagonal": _program_ad_assembly_diagonal_dtype_rule,
        "dstack": _program_ad_assembly_stack_convenience_dtype_rule_for("dstack"),
        "full_like": _program_ad_assembly_like_dtype_rule,
        "hstack": _program_ad_assembly_stack_convenience_dtype_rule_for("hstack"),
        "column_stack": _program_ad_assembly_stack_convenience_dtype_rule_for("column_stack"),
        "ones_like": _program_ad_assembly_like_dtype_rule,
        "split": _program_ad_assembly_split_dtype_rule,
        "array_split": _program_ad_assembly_split_dtype_rule,
        "hsplit": _program_ad_assembly_split_dtype_rule,
        "vsplit": _program_ad_assembly_split_dtype_rule,
        "dsplit": _program_ad_assembly_split_dtype_rule,
        "stack": _program_ad_assembly_stack_dtype_rule,
        "tril": _program_ad_assembly_triangular_dtype_rule,
        "triu": _program_ad_assembly_triangular_dtype_rule,
        "vstack": _program_ad_assembly_stack_convenience_dtype_rule_for("vstack"),
        "zeros_like": _program_ad_assembly_like_dtype_rule,
    }
    static_argument_rules: Mapping[str, PrimitiveStaticArgumentRule] = {
        "append": _program_ad_assembly_append_static_arguments,
        "block": _program_ad_assembly_block_static_arguments,
        "broadcast_arrays": _program_ad_assembly_broadcast_arrays_static_arguments,
        "broadcast_to": _program_ad_assembly_broadcast_to_static_arguments,
        "concatenate": _program_ad_assembly_concatenate_static_arguments,
        "diagonal": _program_ad_assembly_diagonal_static_arguments,
        "dstack": _program_ad_assembly_stack_convenience_static_arguments_rule_for("dstack"),
        "full_like": _program_ad_assembly_full_like_static_arguments,
        "hstack": _program_ad_assembly_stack_convenience_static_arguments_rule_for("hstack"),
        "column_stack": _program_ad_assembly_stack_convenience_static_arguments_rule_for(
            "column_stack"
        ),
        "ones_like": _program_ad_assembly_ones_like_static_arguments,
        "split": _program_ad_assembly_split_static_arguments_rule_for("split"),
        "array_split": _program_ad_assembly_split_static_arguments_rule_for("array_split"),
        "hsplit": _program_ad_assembly_split_static_arguments_rule_for("hsplit"),
        "vsplit": _program_ad_assembly_split_static_arguments_rule_for("vsplit"),
        "dsplit": _program_ad_assembly_split_static_arguments_rule_for("dsplit"),
        "stack": _program_ad_assembly_stack_static_arguments,
        "tril": _program_ad_assembly_triangular_static_arguments,
        "triu": _program_ad_assembly_triangular_static_arguments,
        "vstack": _program_ad_assembly_stack_convenience_static_arguments_rule_for("vstack"),
        "zeros_like": _program_ad_assembly_zeros_like_static_arguments,
    }
    for name, identity in _PROGRAM_AD_ASSEMBLY_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_assembly_derivative_rule(name),
                batching_rule=batching_rules[name],
                lowering_metadata=_program_ad_assembly_lowering_metadata(name),
                shape_rule=shape_rules[name],
                dtype_rule=dtype_rules[name],
                static_argument_rule=static_argument_rules[name],
                nondifferentiable_policy=_PROGRAM_AD_ASSEMBLY_POLICY,
                effect="pure",
            )
        )


def _register_program_ad_signal_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_SIGNAL_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_signal_derivative_rule(name),
                batching_rule=_program_ad_signal_batching_rule,
                lowering_metadata=_program_ad_signal_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_SIGNAL_SHAPE_RULES[name],
                dtype_rule=_program_ad_signal_convolve_dtype_rule,
                static_argument_rule=_PROGRAM_AD_SIGNAL_STATIC_ARGUMENT_RULES[name],
                nondifferentiable_policy=_PROGRAM_AD_SIGNAL_POLICY,
                effect="pure",
            )
        )


def _register_program_ad_shape_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_SHAPE_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_shape_derivative_rule(name),
                batching_rule=_program_ad_shape_batching_rule,
                lowering_metadata=_program_ad_shape_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_SHAPE_SHAPE_RULES[name],
                dtype_rule=_program_ad_shape_dtype_rule,
                static_argument_rule=_PROGRAM_AD_SHAPE_STATIC_ARGUMENT_RULES[name],
                nondifferentiable_policy=_PROGRAM_AD_SHAPE_POLICY,
                effect="pure",
            )
        )


def _register_program_ad_reduction_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_REDUCTION_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_reduction_derivative_rule(name),
                batching_rule=_program_ad_reduction_batching_rule,
                lowering_metadata=_program_ad_reduction_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_REDUCTION_SHAPE_RULES[name],
                dtype_rule=_program_ad_reduction_dtype_rule,
                static_argument_rule=_PROGRAM_AD_REDUCTION_STATIC_ARGUMENT_RULES[name],
                nondifferentiable_policy=_PROGRAM_AD_REDUCTION_POLICY,
                effect="pure",
            )
        )


def _register_program_ad_elementwise_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_ELEMENTWISE_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_elementwise_derivative_rule(name),
                batching_rule=_program_ad_elementwise_batching_rule,
                lowering_metadata=_program_ad_elementwise_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_ELEMENTWISE_SHAPE_RULES[name],
                dtype_rule=_program_ad_elementwise_dtype_rule,
                static_argument_rule=_program_ad_elementwise_static_arguments,
                nondifferentiable_policy=_PROGRAM_AD_ELEMENTWISE_POLICY,
                effect="pure",
            )
        )


def _register_program_ad_selection_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_SELECTION_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_selection_derivative_rule(name),
                batching_rule=_program_ad_selection_batching_rule_for(name),
                lowering_metadata=_program_ad_selection_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_SELECTION_SHAPE_RULES[name],
                dtype_rule=_PROGRAM_AD_SELECTION_DTYPE_RULES[name],
                static_argument_rule=_PROGRAM_AD_SELECTION_STATIC_ARGUMENT_RULES[name],
                nondifferentiable_policy=_PROGRAM_AD_SELECTION_POLICY,
                effect="pure",
            )
        )


def _register_program_ad_product_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_PRODUCT_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_product_derivative_rule(name),
                batching_rule=_program_ad_product_batching_rule,
                lowering_metadata=_program_ad_product_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_PRODUCT_SHAPE_RULES[name],
                dtype_rule=_program_ad_product_dtype_rule,
                static_argument_rule=_program_ad_product_static_arguments,
                nondifferentiable_policy=_PROGRAM_AD_PRODUCT_POLICY,
                effect="pure",
            )
        )


def _register_program_ad_cumulative_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_CUMULATIVE_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_cumulative_derivative_rule(name),
                batching_rule=_program_ad_cumulative_batching_rule,
                lowering_metadata=_program_ad_cumulative_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_CUMULATIVE_SHAPE_RULES[name],
                dtype_rule=_program_ad_cumulative_dtype_rule,
                static_argument_rule=_PROGRAM_AD_CUMULATIVE_STATIC_ARGUMENT_RULES[name],
                nondifferentiable_policy=_PROGRAM_AD_CUMULATIVE_POLICY,
                effect="pure",
            )
        )


def _validate_program_ad_primitive_contract_dispatch(
    contract: PrimitiveContract,
    args: tuple[object, ...],
) -> None:
    if contract.static_argument_rule is None:
        raise ValueError(
            f"program AD primitive {contract.identity.key} missing static argument rule"
        )
    if contract.shape_rule is None:
        raise ValueError(f"program AD primitive {contract.identity.key} missing shape rule")
    if contract.dtype_rule is None:
        raise ValueError(f"program AD primitive {contract.identity.key} missing dtype rule")
    static_arguments = contract.static_argument_rule(args)
    if not isinstance(static_arguments, tuple):
        raise ValueError(
            f"program AD primitive {contract.identity.key} static rule must return a tuple"
        )
    shape = contract.shape_rule(args)
    if not isinstance(shape, tuple) or any(
        not isinstance(dimension, int) or dimension < 0 for dimension in shape
    ):
        raise ValueError(
            f"program AD primitive {contract.identity.key} shape rule must return "
            "non-negative integer dimensions"
        )
    dtype = contract.dtype_rule(args)
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(
            f"program AD primitive {contract.identity.key} dtype rule must return a dtype name"
        )


def _require_program_ad_runtime_contract(
    name: str,
    *,
    family: str,
    identities: Mapping[str, PrimitiveIdentity],
    expected_policy: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    identity = identities.get(name)
    if identity is None:
        raise ValueError(f"no program AD {family} primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != expected_policy:
        raise ValueError(f"invalid program AD {family} primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD {family} primitive effect for {identity.key}")

    missing: list[str] = []
    if contract.batching_rule is None:
        missing.append("batching_rule")
    if not contract.lowering_metadata:
        missing.append("lowering_metadata")
    if not contract.lowering_metadata.get("mlir_op"):
        missing.append("mlir_op")
    if not contract.lowering_metadata.get("nondifferentiable_boundary"):
        missing.append("nondifferentiable_boundary")
    if contract.lowering_metadata.get("nondifferentiable_boundary_policy") != "fail_closed":
        missing.append("nondifferentiable_boundary_policy")
    if contract.shape_rule is None:
        missing.append("shape_rule")
    if contract.dtype_rule is None:
        missing.append("dtype_rule")
    if contract.static_argument_rule is None:
        missing.append("static_argument_rule")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"incomplete program AD {family} primitive runtime contract for "
            f"{identity.key}: missing {joined}"
        )
    if args is not None:
        _validate_program_ad_primitive_contract_dispatch(contract, args)
    return contract


def _require_program_ad_array_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    return _require_program_ad_runtime_contract(
        name,
        family="array",
        identities=_PROGRAM_AD_ARRAY_IDENTITIES,
        expected_policy=_PROGRAM_AD_ARRAY_POLICY,
        args=args,
    )


def _require_program_ad_interpolation_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    return _require_program_ad_runtime_contract(
        name,
        family="interpolation",
        identities=_PROGRAM_AD_INTERPOLATION_IDENTITIES,
        expected_policy=_PROGRAM_AD_INTERPOLATION_POLICY,
        args=args,
    )


def _require_program_ad_assembly_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    return _require_program_ad_runtime_contract(
        name,
        family="assembly",
        identities=_PROGRAM_AD_ASSEMBLY_IDENTITIES,
        expected_policy=_PROGRAM_AD_ASSEMBLY_POLICY,
        args=args,
    )


def _require_program_ad_signal_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    return _require_program_ad_runtime_contract(
        name,
        family="signal",
        identities=_PROGRAM_AD_SIGNAL_IDENTITIES,
        expected_policy=_PROGRAM_AD_SIGNAL_POLICY,
        args=args,
    )


def _require_program_ad_shape_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    return _require_program_ad_runtime_contract(
        name,
        family="shape",
        identities=_PROGRAM_AD_SHAPE_IDENTITIES,
        expected_policy=_PROGRAM_AD_SHAPE_POLICY,
        args=args,
    )


def _require_program_ad_reduction_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    return _require_program_ad_runtime_contract(
        name,
        family="reduction",
        identities=_PROGRAM_AD_REDUCTION_IDENTITIES,
        expected_policy=_PROGRAM_AD_REDUCTION_POLICY,
        args=args,
    )


def _require_program_ad_elementwise_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    return _require_program_ad_runtime_contract(
        name,
        family="elementwise",
        identities=_PROGRAM_AD_ELEMENTWISE_IDENTITIES,
        expected_policy=_PROGRAM_AD_ELEMENTWISE_POLICY,
        args=args,
    )


def _require_program_ad_selection_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    return _require_program_ad_runtime_contract(
        name,
        family="selection",
        identities=_PROGRAM_AD_SELECTION_IDENTITIES,
        expected_policy=_PROGRAM_AD_SELECTION_POLICY,
        args=args,
    )


def _require_program_ad_product_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    return _require_program_ad_runtime_contract(
        name,
        family="product",
        identities=_PROGRAM_AD_PRODUCT_IDENTITIES,
        expected_policy=_PROGRAM_AD_PRODUCT_POLICY,
        args=args,
    )


def _require_program_ad_cumulative_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    return _require_program_ad_runtime_contract(
        name,
        family="cumulative",
        identities=_PROGRAM_AD_CUMULATIVE_IDENTITIES,
        expected_policy=_PROGRAM_AD_CUMULATIVE_POLICY,
        args=args,
    )


def _program_ad_linalg_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    raise ValueError(
        "program AD linalg primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


def _program_ad_linalg_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    raise ValueError(
        "program AD linalg primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


@dataclass(frozen=True)
class ProgramADLinalgConditioningDiagnostic:
    """Conditioning report for numerically sensitive program-AD linalg primitives."""

    primitive: str
    shape: tuple[int, ...]
    status: str
    differentiability_ready: bool
    condition_number: float
    rank: int
    smallest_scale: float
    largest_scale: float
    minimum_gap: float | None
    threshold: float
    required_boundary: str
    message: str
    claim_boundary: str = (
        "program-AD linalg conditioning diagnostic only; no provider, hardware, "
        "native-framework, or production benchmark evidence is implied"
    )

    def __post_init__(self) -> None:
        if not self.primitive:
            raise ValueError("conditioning diagnostic primitive must be non-empty")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("conditioning diagnostic shape dimensions must be non-negative")
        if self.status not in {
            "well_conditioned",
            "ill_conditioned",
            "rank_deficient",
            "zero_norm_boundary",
        }:
            raise ValueError("conditioning diagnostic status is unsupported")
        for field_name, value in (
            ("condition_number", self.condition_number),
            ("smallest_scale", self.smallest_scale),
            ("largest_scale", self.largest_scale),
            ("threshold", self.threshold),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"conditioning diagnostic {field_name} must be finite non-negative"
                )
        if self.minimum_gap is not None and (
            not math.isfinite(self.minimum_gap) or self.minimum_gap < 0.0
        ):
            raise ValueError("conditioning diagnostic minimum_gap must be finite non-negative")
        if self.rank < 0:
            raise ValueError("conditioning diagnostic rank must be non-negative")
        if not self.required_boundary:
            raise ValueError("conditioning diagnostic required_boundary must be non-empty")
        if not self.message:
            raise ValueError("conditioning diagnostic message must be non-empty")
        object.__setattr__(
            self,
            "claim_boundary",
            _normalise_claim_boundary("conditioning diagnostic", self.claim_boundary),
        )

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready representation."""

        return {
            "primitive": self.primitive,
            "shape": list(self.shape),
            "status": self.status,
            "differentiability_ready": self.differentiability_ready,
            "condition_number": self.condition_number,
            "rank": self.rank,
            "smallest_scale": self.smallest_scale,
            "largest_scale": self.largest_scale,
            "minimum_gap": self.minimum_gap,
            "threshold": self.threshold,
            "required_boundary": self.required_boundary,
            "message": self.message,
            "claim_boundary": self.claim_boundary,
        }


_PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES: Mapping[str, str] = {
    "norm": "non-zero norm for norm-gradient division",
    "det": "non-singular matrix away from determinant rank drop",
    "inv": "non-singular matrix inverse",
    "solve": "non-singular linear system with compatible right-hand side",
    "matrix_power": "non-singular matrix for negative powers",
    "eig": "real simple diagonalizable eigensystem",
    "eigh": "symmetric matrix with distinct eigenvalues",
    "eigvals": "real simple diagonalizable spectrum",
    "eigvalsh": "symmetric matrix with distinct eigenvalues",
    "svd": "distinct positive singular values",
    "pinv": "constant rank away from rank threshold crossing",
}


def _program_ad_linalg_conditioning_matrix(
    primitive: str,
    values: ArrayLike,
) -> NDArray[np.float64]:
    matrix = _as_real_numeric_array(f"program AD linalg {primitive} conditioning values", values)
    if matrix.ndim != 2:
        raise ValueError(f"program AD linalg {primitive} conditioning requires a rank-2 matrix")
    if 0 in matrix.shape:
        raise ValueError(f"program AD linalg {primitive} conditioning requires non-empty axes")
    return matrix


def _program_ad_linalg_condition_number(
    singular_values: NDArray[np.float64],
) -> tuple[float, float, float]:
    if singular_values.size == 0:
        return 0.0, 0.0, 0.0
    largest = float(np.max(singular_values))
    smallest = float(np.min(singular_values))
    if smallest == 0.0:
        return math.inf, smallest, largest
    return float(largest / smallest), smallest, largest


def _program_ad_linalg_minimum_gap(values: NDArray[np.float64]) -> float | None:
    if values.size < 2:
        return None
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    return float(np.min(np.diff(ordered)))


def _program_ad_linalg_diagnostic_from_singular_values(
    primitive: str,
    shape: tuple[int, ...],
    singular_values: NDArray[np.float64],
    *,
    condition_threshold: float,
    rank_tolerance: float,
    minimum_gap: float | None,
) -> ProgramADLinalgConditioningDiagnostic:
    condition_number, smallest, largest = _program_ad_linalg_condition_number(singular_values)
    rank = int(np.sum(singular_values > rank_tolerance))
    full_rank = rank == int(singular_values.size)
    if not full_rank:
        return ProgramADLinalgConditioningDiagnostic(
            primitive=primitive,
            shape=shape,
            status="rank_deficient",
            differentiability_ready=False,
            condition_number=0.0 if math.isinf(condition_number) else condition_number,
            rank=rank,
            smallest_scale=smallest,
            largest_scale=largest,
            minimum_gap=minimum_gap,
            threshold=condition_threshold,
            required_boundary=_PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES[primitive],
            message=(
                f"program AD linalg {primitive} is at a rank threshold boundary; "
                "the derivative contract remains fail-closed"
            ),
        )
    status = "ill_conditioned" if condition_number > condition_threshold else "well_conditioned"
    message = (
        f"program AD linalg {primitive} is ill-conditioned but remains differentiable "
        "inside the declared rank/spectrum boundary"
        if status == "ill_conditioned"
        else f"program AD linalg {primitive} conditioning is inside the declared boundary"
    )
    return ProgramADLinalgConditioningDiagnostic(
        primitive=primitive,
        shape=shape,
        status=status,
        differentiability_ready=True,
        condition_number=condition_number,
        rank=rank,
        smallest_scale=smallest,
        largest_scale=largest,
        minimum_gap=minimum_gap,
        threshold=condition_threshold,
        required_boundary=_PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES[primitive],
        message=message,
    )


def diagnose_program_ad_linalg_conditioning(
    primitive: str,
    values: ArrayLike,
    *,
    condition_threshold: float = 1.0e12,
    rank_tolerance: float = 1.0e-12,
) -> ProgramADLinalgConditioningDiagnostic:
    """Diagnose conditioning for supported norm and program-AD linalg primitives."""

    name = str(primitive).strip().lower()
    if name not in _PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES:
        raise ValueError(f"unsupported program AD linalg conditioning primitive {primitive!r}")
    if not math.isfinite(condition_threshold) or condition_threshold <= 0.0:
        raise ValueError("program AD linalg conditioning threshold must be positive and finite")
    if not math.isfinite(rank_tolerance) or rank_tolerance < 0.0:
        raise ValueError("program AD linalg rank tolerance must be finite non-negative")

    if name == "norm":
        array = _as_real_numeric_array("program AD linalg norm conditioning values", values)
        norm_value = float(np.linalg.norm(array.reshape(-1), ord=2))
        if norm_value <= rank_tolerance:
            return ProgramADLinalgConditioningDiagnostic(
                primitive=name,
                shape=tuple(int(dimension) for dimension in array.shape),
                status="zero_norm_boundary",
                differentiability_ready=False,
                condition_number=0.0,
                rank=0,
                smallest_scale=0.0,
                largest_scale=norm_value,
                minimum_gap=None,
                threshold=condition_threshold,
                required_boundary=_PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES[name],
                message="program AD linalg norm is at the zero norm nondifferentiable boundary",
            )
        return ProgramADLinalgConditioningDiagnostic(
            primitive=name,
            shape=tuple(int(dimension) for dimension in array.shape),
            status="well_conditioned",
            differentiability_ready=True,
            condition_number=1.0,
            rank=1,
            smallest_scale=norm_value,
            largest_scale=norm_value,
            minimum_gap=None,
            threshold=condition_threshold,
            required_boundary=_PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES[name],
            message="program AD linalg norm is away from the zero norm boundary",
        )

    matrix = _program_ad_linalg_conditioning_matrix(name, values)
    singular_values = np.linalg.svd(matrix, compute_uv=False).astype(np.float64)
    minimum_gap: float | None = None
    if name in {"eig", "eigvals"}:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        if np.max(np.abs(eigenvalues.imag)) > 1.0e-10:
            return ProgramADLinalgConditioningDiagnostic(
                primitive=name,
                shape=tuple(int(dimension) for dimension in matrix.shape),
                status="rank_deficient",
                differentiability_ready=False,
                condition_number=0.0,
                rank=int(np.linalg.matrix_rank(matrix, tol=rank_tolerance)),
                smallest_scale=0.0,
                largest_scale=float(np.max(np.abs(eigenvalues))),
                minimum_gap=None,
                threshold=condition_threshold,
                required_boundary=_PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES[name],
                message=f"program AD linalg {name} conditioning requires real eigenvalues",
            )
        minimum_gap = _program_ad_linalg_minimum_gap(eigenvalues.real.astype(np.float64))
        singular_values = np.linalg.svd(eigenvectors.real, compute_uv=False).astype(np.float64)
    elif name in {"eigh", "eigvalsh"}:
        _program_ad_linalg_require_symmetric(name, matrix)
        eigenvalues = np.asarray(np.linalg.eigvalsh(matrix), dtype=np.float64)
        minimum_gap = _program_ad_linalg_minimum_gap(eigenvalues)
        singular_values = np.abs(eigenvalues).astype(np.float64)
    elif name == "svd":
        minimum_gap = _program_ad_linalg_minimum_gap(singular_values)

    if minimum_gap is not None and minimum_gap <= rank_tolerance:
        return ProgramADLinalgConditioningDiagnostic(
            primitive=name,
            shape=tuple(int(dimension) for dimension in matrix.shape),
            status="rank_deficient",
            differentiability_ready=False,
            condition_number=0.0,
            rank=int(np.sum(singular_values > rank_tolerance)),
            smallest_scale=float(np.min(singular_values)) if singular_values.size else 0.0,
            largest_scale=float(np.max(singular_values)) if singular_values.size else 0.0,
            minimum_gap=minimum_gap,
            threshold=condition_threshold,
            required_boundary=_PROGRAM_AD_LINALG_CONDITIONING_BOUNDARIES[name],
            message=(
                f"program AD linalg {name} is at a repeated spectrum boundary; "
                "the derivative contract remains fail-closed"
            ),
        )

    return _program_ad_linalg_diagnostic_from_singular_values(
        name,
        tuple(int(dimension) for dimension in matrix.shape),
        singular_values,
        condition_threshold=condition_threshold,
        rank_tolerance=rank_tolerance,
        minimum_gap=minimum_gap,
    )


def _program_ad_linalg_square_matrix(
    primitive_name: str,
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array(f"program AD linalg {primitive_name} values", values).reshape(
        -1
    )
    size = int(vector.size)
    rows = int(math.isqrt(size))
    if rows * rows != size:
        raise ValueError(
            f"program AD linalg {primitive_name} direct rule requires a flattened square matrix"
        )
    return vector.reshape(rows, rows)


def _program_ad_linalg_det_cofactor_matrix(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("program AD linalg det direct rule requires a square matrix")
    if rows == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if rows == 1:
        return np.ones((1, 1), dtype=np.float64)
    cofactors = np.zeros_like(matrix, dtype=np.float64)
    for row in range(rows):
        for col in range(cols):
            minor = np.delete(np.delete(matrix, row, axis=0), col, axis=1)
            cofactors[row, col] = ((-1.0) ** (row + col)) * float(np.linalg.det(minor))
    return cofactors


def _program_ad_linalg_det_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("det", values)
    return np.array([float(np.linalg.det(matrix))], dtype=np.float64)


def _program_ad_linalg_det_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("det", values)
    tangent_matrix = _program_ad_linalg_square_matrix("det", tangent)
    if tangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg det tangent shape must match matrix shape")
    cofactors = _program_ad_linalg_det_cofactor_matrix(matrix)
    return np.array([float(np.sum(cofactors * tangent_matrix))], dtype=np.float64)


def _program_ad_linalg_scalar_cotangent(
    primitive_name: str,
    cotangent: NDArray[np.float64],
) -> float:
    cotangent_vector = _as_real_numeric_array(
        f"program AD linalg {primitive_name} cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.shape != (1,):
        raise ValueError(f"program AD linalg {primitive_name} VJP requires one scalar cotangent")
    return float(cotangent_vector[0])


def _program_ad_linalg_det_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("det", values)
    scalar_cotangent = _program_ad_linalg_scalar_cotangent("det", cotangent)
    cofactors = _program_ad_linalg_det_cofactor_matrix(matrix)
    return _program_ad_float64_vector_result(scalar_cotangent * cofactors)


def _program_ad_linalg_inv_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("inv", values)
    return np.linalg.inv(matrix).reshape(-1).astype(np.float64)


def _program_ad_linalg_inv_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("inv", values)
    tangent_matrix = _program_ad_linalg_square_matrix("inv", tangent)
    if tangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg inv tangent shape must match matrix shape")
    inverse = np.linalg.inv(matrix)
    return (-(inverse @ tangent_matrix @ inverse)).reshape(-1).astype(np.float64)


def _program_ad_linalg_inv_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("inv", values)
    cotangent_matrix = _program_ad_linalg_square_matrix("inv cotangent", cotangent)
    if cotangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg inv VJP cotangent shape must match output shape")
    inverse = np.linalg.inv(matrix)
    return _program_ad_float64_vector_result(-(inverse.T @ cotangent_matrix @ inverse.T))


def _program_ad_linalg_solve_split(
    primitive_name: str,
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD linalg {primitive_name} values", values).reshape(
        -1
    )
    total = int(vector.size)
    rows = int((math.isqrt(1 + 4 * total) - 1) // 2)
    if rows * rows + rows != total:
        raise ValueError(
            "program AD linalg solve direct rule requires flattened square matrix "
            "followed by vector right-hand side"
        )
    matrix = vector[: rows * rows].reshape(rows, rows)
    rhs = vector[rows * rows :]
    return matrix, rhs


def _program_ad_linalg_solve_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix, rhs = _program_ad_linalg_solve_split("solve", values)
    return np.linalg.solve(matrix, rhs).astype(np.float64)


def _program_ad_linalg_solve_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, rhs = _program_ad_linalg_solve_split("solve", values)
    tangent_matrix, tangent_rhs = _program_ad_linalg_solve_split("solve", tangent)
    if tangent_matrix.shape != matrix.shape or tangent_rhs.shape != rhs.shape:
        raise ValueError("program AD linalg solve tangent shape must match primal shape")
    solution = np.linalg.solve(matrix, rhs)
    return np.linalg.solve(matrix, tangent_rhs - tangent_matrix @ solution).astype(np.float64)


def _program_ad_linalg_solve_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, rhs = _program_ad_linalg_solve_split("solve", values)
    cotangent_vector = _as_real_numeric_array(
        "program AD linalg solve cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.shape != rhs.shape:
        raise ValueError("program AD linalg solve VJP cotangent shape must match solution shape")
    solution = np.linalg.solve(matrix, rhs)
    rhs_adjoint = np.linalg.solve(matrix.T, cotangent_vector)
    matrix_adjoint = -np.outer(rhs_adjoint, solution)
    return _program_ad_float64_vector_result(
        np.concatenate((matrix_adjoint.reshape(-1), rhs_adjoint))
    )


def _program_ad_linalg_normalise_solve_shapes(
    matrix_shape: Sequence[int],
    rhs_shape: Sequence[int],
) -> tuple[tuple[int, int], tuple[int, ...]]:
    matrix = tuple(int(dimension) for dimension in matrix_shape)
    rhs = tuple(int(dimension) for dimension in rhs_shape)
    if len(matrix) != 2 or matrix[0] != matrix[1]:
        raise ValueError("program AD linalg solve direct rule requires a square matrix")
    if any(dimension < 0 for dimension in (*matrix, *rhs)):
        raise ValueError("program AD linalg solve direct rule requires non-negative dimensions")
    if len(rhs) not in {1, 2}:
        raise ValueError("program AD linalg solve direct rule requires rank-1 or rank-2 rhs")
    if rhs[0] != matrix[0]:
        raise ValueError(
            "program AD linalg solve direct rule right-hand side rows must match matrix"
        )
    return matrix, rhs


def _program_ad_linalg_solve_static_split(
    name: str,
    values: NDArray[np.float64],
    *,
    matrix_shape: tuple[int, int],
    rhs_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD linalg solve {name}", values).reshape(-1)
    matrix_size = _program_ad_shape_static_size(matrix_shape)
    rhs_size = _program_ad_shape_static_size(rhs_shape)
    if vector.size != matrix_size + rhs_size:
        raise ValueError(
            "program AD linalg solve direct rule requires flattened matrix followed by rhs"
        )
    return (
        vector[:matrix_size].reshape(matrix_shape),
        vector[matrix_size:].reshape(rhs_shape),
    )


def program_ad_linalg_solve_derivative_rule(
    matrix_shape: Sequence[int],
    rhs_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed solve primitive signature."""

    matrix_static_shape, rhs_static_shape = _program_ad_linalg_normalise_solve_shapes(
        matrix_shape, rhs_shape
    )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix, rhs = _program_ad_linalg_solve_static_split(
            "values", values, matrix_shape=matrix_static_shape, rhs_shape=rhs_static_shape
        )
        return _program_ad_float64_vector_result(np.linalg.solve(matrix, rhs))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix, rhs = _program_ad_linalg_solve_static_split(
            "values", values, matrix_shape=matrix_static_shape, rhs_shape=rhs_static_shape
        )
        tangent_matrix, tangent_rhs = _program_ad_linalg_solve_static_split(
            "tangent", tangent, matrix_shape=matrix_static_shape, rhs_shape=rhs_static_shape
        )
        solution = np.linalg.solve(matrix, rhs)
        return _program_ad_float64_vector_result(
            np.linalg.solve(matrix, tangent_rhs - tangent_matrix @ solution)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        matrix, rhs = _program_ad_linalg_solve_static_split(
            "values", values, matrix_shape=matrix_static_shape, rhs_shape=rhs_static_shape
        )
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg solve cotangent", cotangent
        ).reshape(-1)
        rhs_size = _program_ad_shape_static_size(rhs_static_shape)
        if cotangent_vector.size != rhs_size:
            raise ValueError(
                "program AD linalg solve VJP cotangent shape must match solution shape"
            )
        cotangent_rhs = cotangent_vector.reshape(rhs_static_shape)
        solution = np.linalg.solve(matrix, rhs)
        rhs_adjoint = np.linalg.solve(matrix.T, cotangent_rhs)
        if rhs_adjoint.ndim == 1:
            matrix_adjoint = -np.outer(rhs_adjoint, solution)
        else:
            matrix_adjoint = -(rhs_adjoint @ solution.T)
        return _program_ad_float64_vector_result(
            np.concatenate((matrix_adjoint.reshape(-1), rhs_adjoint.reshape(-1)))
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_linalg_solve_"
            f"{_program_ad_shape_signature(matrix_static_shape)}_rhs_"
            f"{_program_ad_shape_signature(rhs_static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_linalg_matrix_power_derivative_rule(
    power: int | np.integer,
) -> CustomDerivativeRule:
    """Build a direct value/JVP rule for a fixed matrix-power primitive."""

    if isinstance(power, bool) or not isinstance(power, (int, np.integer)):
        raise ValueError("program AD linalg matrix_power derivative rule requires integer power")
    exponent = int(power)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = _program_ad_linalg_square_matrix("matrix_power", values)
        return np.linalg.matrix_power(matrix, exponent).reshape(-1).astype(np.float64)

    def jvp_rule(
        values: NDArray[np.float64],
        tangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        matrix = _program_ad_linalg_square_matrix("matrix_power", values)
        tangent_matrix = _program_ad_linalg_square_matrix("matrix_power", tangent)
        if tangent_matrix.shape != matrix.shape:
            raise ValueError(
                "program AD linalg matrix_power tangent shape must match matrix shape"
            )
        if exponent == 0:
            return np.zeros_like(matrix, dtype=np.float64).reshape(-1)
        if exponent > 0:
            total = np.zeros_like(matrix, dtype=np.float64)
            powers = [np.linalg.matrix_power(matrix, index) for index in range(exponent)]
            for index in range(exponent):
                total = total + powers[index] @ tangent_matrix @ powers[exponent - 1 - index]
            return total.reshape(-1).astype(np.float64)
        inverse = np.linalg.inv(matrix)
        inverse_tangent = -(inverse @ tangent_matrix @ inverse)
        positive_exponent = -exponent
        total = np.zeros_like(matrix, dtype=np.float64)
        powers = [np.linalg.matrix_power(inverse, index) for index in range(positive_exponent)]
        for index in range(positive_exponent):
            total = total + powers[index] @ inverse_tangent @ powers[positive_exponent - 1 - index]
        return total.reshape(-1).astype(np.float64)

    def vjp_rule(
        values: NDArray[np.float64],
        cotangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        matrix = _program_ad_linalg_square_matrix("matrix_power", values)
        cotangent_matrix = _program_ad_linalg_square_matrix("matrix_power cotangent", cotangent)
        if cotangent_matrix.shape != matrix.shape:
            raise ValueError(
                "program AD linalg matrix_power VJP cotangent shape must match output shape"
            )
        if exponent == 0:
            return np.zeros_like(matrix, dtype=np.float64).reshape(-1)
        if exponent > 0:
            total = np.zeros_like(matrix, dtype=np.float64)
            powers = [np.linalg.matrix_power(matrix, index) for index in range(exponent)]
            for index in range(exponent):
                total = total + powers[index].T @ cotangent_matrix @ powers[exponent - 1 - index].T
            return total.reshape(-1).astype(np.float64)
        inverse = np.linalg.inv(matrix)
        positive_exponent = -exponent
        inverse_adjoint = np.zeros_like(matrix, dtype=np.float64)
        powers = [np.linalg.matrix_power(inverse, index) for index in range(positive_exponent)]
        for index in range(positive_exponent):
            inverse_adjoint = (
                inverse_adjoint
                + powers[index].T @ cotangent_matrix @ powers[positive_exponent - 1 - index].T
            )
        return _program_ad_float64_vector_result(-(inverse.T @ inverse_adjoint @ inverse.T))

    return CustomDerivativeRule(
        name=f"program_ad_linalg_matrix_power_{exponent}_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _normalise_program_ad_linalg_multi_dot_shapes(
    operand_shapes: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    shapes = tuple(tuple(int(dim) for dim in shape) for shape in operand_shapes)
    if len(shapes) < 2:
        raise ValueError(
            "program AD linalg multi_dot derivative rule requires at least two shapes"
        )
    for index, shape in enumerate(shapes):
        if len(shape) not in {1, 2}:
            raise ValueError("program AD linalg multi_dot derivative rule supports rank-1/rank-2")
        if any(dim <= 0 for dim in shape):
            raise ValueError(
                "program AD linalg multi_dot derivative rule dimensions must be positive"
            )
        if 0 < index < len(shapes) - 1 and len(shape) != 2:
            raise ValueError(
                "program AD linalg multi_dot derivative rule middle operands must be rank-2"
            )
    _program_ad_linalg_multi_dot_shape((tuple(np.zeros(shape) for shape in shapes),))
    return shapes


def _split_program_ad_linalg_multi_dot_operands(
    name: str,
    values: NDArray[np.float64],
    operand_shapes: tuple[tuple[int, ...], ...],
) -> tuple[NDArray[np.float64], ...]:
    vector = _as_real_numeric_array(f"program AD linalg multi_dot {name}", values).reshape(-1)
    expected_size = sum(int(np.prod(shape)) for shape in operand_shapes)
    if vector.size != expected_size:
        raise ValueError("program AD linalg multi_dot direct rule values size must match shapes")
    operands: list[NDArray[np.float64]] = []
    cursor = 0
    for shape in operand_shapes:
        size = int(np.prod(shape))
        operands.append(vector[cursor : cursor + size].reshape(shape))
        cursor += size
    return tuple(operands)


def _as_flat_multi_dot_result(value: object) -> NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def program_ad_linalg_multi_dot_derivative_rule(
    operand_shapes: Sequence[Sequence[int]],
) -> CustomDerivativeRule:
    """Build a direct value/JVP rule for a fixed multi-dot operand signature."""

    shapes = _normalise_program_ad_linalg_multi_dot_shapes(operand_shapes)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        operands = _split_program_ad_linalg_multi_dot_operands("values", values, shapes)
        return _as_flat_multi_dot_result(np.linalg.multi_dot(operands))

    def jvp_rule(
        values: NDArray[np.float64],
        tangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        operands = _split_program_ad_linalg_multi_dot_operands("values", values, shapes)
        tangent_operands = _split_program_ad_linalg_multi_dot_operands("tangent", tangent, shapes)
        total: NDArray[np.float64] | None = None
        for index, tangent_operand in enumerate(tangent_operands):
            varied = operands[:index] + (tangent_operand,) + operands[index + 1 :]
            contribution = _as_flat_multi_dot_result(np.linalg.multi_dot(varied))
            total = contribution if total is None else total + contribution
        if total is None:
            raise ValueError("program AD linalg multi_dot direct rule requires operands")
        return total.astype(np.float64)

    def vjp_rule(
        values: NDArray[np.float64],
        cotangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        operands = _split_program_ad_linalg_multi_dot_operands("values", values, shapes)
        output = _as_flat_multi_dot_result(np.linalg.multi_dot(operands))
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg multi_dot cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.shape != output.shape:
            raise ValueError(
                "program AD linalg multi_dot VJP cotangent shape must match output shape"
            )
        adjoints: list[NDArray[np.float64]] = []
        for operand_index, operand in enumerate(operands):
            operand_adjoint = np.zeros_like(operand, dtype=np.float64)
            for element_index in np.ndindex(operand.shape):
                basis = np.zeros_like(operand, dtype=np.float64)
                basis[element_index] = 1.0
                varied = operands[:operand_index] + (basis,) + operands[operand_index + 1 :]
                contribution = _as_flat_multi_dot_result(np.linalg.multi_dot(varied))
                operand_adjoint[element_index] = float(np.dot(cotangent_vector, contribution))
            adjoints.append(operand_adjoint.reshape(-1))
        return _program_ad_float64_vector_result(np.concatenate(adjoints))

    signature = "x".join("_".join(str(dim) for dim in shape) for shape in shapes)
    return CustomDerivativeRule(
        name=f"program_ad_linalg_multi_dot_{signature}_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_linalg_offset(name: str, offset: int | np.integer) -> int:
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)):
        raise ValueError(f"program AD linalg {name} derivative rule requires integer offset")
    return int(offset)


def _program_ad_assembly_split_source_shape(source_shape: Sequence[int]) -> tuple[int, ...]:
    shape = _program_ad_array_normalise_static_shape("assembly split source", source_shape)
    if not shape:
        raise ValueError("program AD assembly split direct rule requires ranked source arrays")
    return shape


def program_ad_assembly_split_derivative_rule(
    source_shape: Sequence[int],
    indices_or_sections: object,
    *,
    axis: object = 0,
    split_name: str = "split",
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static split-family layouts."""

    if split_name not in _PROGRAM_AD_ASSEMBLY_SPLIT_NAMES:
        raise ValueError(f"unsupported program AD assembly split primitive {split_name}")
    shape = _program_ad_assembly_split_source_shape(source_shape)
    axis_index = _program_ad_assembly_split_axis(axis, rank=len(shape))
    sections = _program_ad_assembly_split_sections(indices_or_sections)
    selected_indices = _program_ad_assembly_split_selected_indices(
        split_name,
        shape,
        sections,
        axis=axis_index,
    )
    source_size = _program_ad_array_static_size(shape)
    part_count = len(selected_indices)

    def split_array(source: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
        if split_name == "array_split":
            parts = np.array_split(source, cast(Any, sections), axis=axis_index)
        else:
            parts = np.split(source, cast(Any, sections), axis=axis_index)
        return tuple(np.asarray(part, dtype=np.float64) for part in parts)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        source = _program_ad_array_vector("split", "values", values, expected_size=source_size)
        parts = split_array(source.reshape(shape))
        return _program_ad_float64_vector_result(
            np.concatenate([part.reshape(-1) for part in parts])
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_array_vector("split", "values", values, expected_size=source_size)
        tangent_source = _program_ad_array_vector(
            "split", "tangent", tangent, expected_size=source_size
        )
        tangent_parts = split_array(tangent_source.reshape(shape))
        return _program_ad_float64_vector_result(
            np.concatenate([part.reshape(-1) for part in tangent_parts])
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_array_vector("split", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_array_vector(
            "split", "cotangent", cotangent, expected_size=source_size
        )
        adjoint = np.zeros(source_size, dtype=np.float64)
        offset = 0
        for index_part in selected_indices:
            part_size = int(index_part.size)
            np.add.at(
                adjoint,
                index_part.reshape(-1),
                cotangent_vector[offset : offset + part_size],
            )
            offset += part_size
        return _program_ad_float64_vector_result(adjoint)

    return CustomDerivativeRule(
        name=f"program_ad_assembly_{split_name}_axis{axis_index}_{part_count}_parts_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_assembly_broadcast_adjoint(
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    adjoint = np.asarray(cotangent, dtype=np.float64)
    if not source_shape:
        return np.asarray(float(np.sum(adjoint)), dtype=np.float64).reshape(())
    if adjoint.ndim > len(source_shape):
        leading_axes = tuple(range(adjoint.ndim - len(source_shape)))
        adjoint = np.sum(adjoint, axis=leading_axes)
    if adjoint.ndim != len(source_shape):
        raise ValueError("program AD assembly broadcast adjoint rank mismatch")
    for axis, dimension in enumerate(source_shape):
        if dimension == 1 and adjoint.shape[axis] != 1:
            adjoint = np.sum(adjoint, axis=axis, keepdims=True)
        elif dimension != adjoint.shape[axis]:
            raise ValueError("program AD assembly broadcast adjoint shape mismatch")
    return np.asarray(adjoint, dtype=np.float64).reshape(source_shape)


def program_ad_assembly_broadcast_to_derivative_rule(
    source_shape: Sequence[int],
    output_shape: object,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.broadcast_to``."""

    source_static_shape, output_static_shape = _program_ad_assembly_broadcast_to_shapes(
        source_shape, output_shape
    )
    source_size = _program_ad_array_static_size(source_static_shape)
    output_size = _program_ad_array_static_size(output_static_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        source = _program_ad_array_vector(
            "broadcast_to", "values", values, expected_size=source_size
        ).reshape(source_static_shape)
        return _program_ad_float64_vector_result(np.broadcast_to(source, output_static_shape))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_array_vector("broadcast_to", "values", values, expected_size=source_size)
        tangent_source = _program_ad_array_vector(
            "broadcast_to", "tangent", tangent, expected_size=source_size
        ).reshape(source_static_shape)
        return _program_ad_float64_vector_result(
            np.broadcast_to(tangent_source, output_static_shape)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_array_vector("broadcast_to", "values", values, expected_size=source_size)
        cotangent_array = _program_ad_array_vector(
            "broadcast_to", "cotangent", cotangent, expected_size=output_size
        ).reshape(output_static_shape)
        return _program_ad_float64_vector_result(
            _program_ad_assembly_broadcast_adjoint(
                cotangent_array,
                source_shape=source_static_shape,
            )
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_assembly_broadcast_to_"
            f"{_program_ad_array_signature(source_static_shape)}_to_"
            f"{_program_ad_array_signature(output_static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_assembly_broadcast_arrays_shapes(
    operand_shapes: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    shapes = tuple(
        _program_ad_array_normalise_static_shape("assembly broadcast_arrays operand", shape)
        for shape in operand_shapes
    )
    if not shapes:
        raise ValueError("program AD assembly broadcast_arrays direct rule requires operands")
    try:
        output_shape = tuple(int(dimension) for dimension in np.broadcast_shapes(*shapes))
    except ValueError as exc:
        raise ValueError(
            "program AD assembly broadcast_arrays direct rule requires broadcast-compatible operands"
        ) from exc
    return shapes, output_shape


def program_ad_assembly_broadcast_arrays_derivative_rule(
    operand_shapes: Sequence[Sequence[int]],
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.broadcast_arrays``."""

    shapes, output_shape = _program_ad_assembly_broadcast_arrays_shapes(operand_shapes)
    source_sizes = tuple(_program_ad_array_static_size(shape) for shape in shapes)
    source_size = sum(source_sizes)
    output_size = _program_ad_array_static_size(output_shape)
    flat_output_size = len(shapes) * output_size

    def split_sources(
        role: str,
        values: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], ...]:
        vector = _program_ad_array_vector(
            "broadcast_arrays", role, values, expected_size=source_size
        )
        offset = 0
        operands: list[NDArray[np.float64]] = []
        for shape, size in zip(shapes, source_sizes, strict=True):
            operands.append(vector[offset : offset + size].reshape(shape))
            offset += size
        return tuple(operands)

    def broadcast_flat(operands: tuple[NDArray[np.float64], ...]) -> NDArray[np.float64]:
        return _program_ad_float64_vector_result(
            np.concatenate(
                [
                    np.asarray(item, dtype=np.float64).reshape(-1)
                    for item in np.broadcast_arrays(*operands)
                ]
            )
        )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return broadcast_flat(split_sources("values", values))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_array_vector("broadcast_arrays", "values", values, expected_size=source_size)
        return broadcast_flat(split_sources("tangent", tangent))

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_array_vector("broadcast_arrays", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_array_vector(
            "broadcast_arrays", "cotangent", cotangent, expected_size=flat_output_size
        )
        adjoints: list[NDArray[np.float64]] = []
        offset = 0
        for shape in shapes:
            cotangent_array = cotangent_vector[offset : offset + output_size].reshape(output_shape)
            adjoints.append(
                _program_ad_assembly_broadcast_adjoint(cotangent_array, source_shape=shape)
            )
            offset += output_size
        return _program_ad_float64_vector_result(
            np.concatenate([item.reshape(-1) for item in adjoints])
        )

    return CustomDerivativeRule(
        name=f"program_ad_assembly_broadcast_arrays_{len(shapes)}_operands_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_assembly_triangular_source_shape(
    source_shape: Sequence[int],
) -> tuple[int, ...]:
    shape = _program_ad_array_normalise_static_shape(
        "assembly triangular mask source", source_shape
    )
    if len(shape) < 2:
        raise ValueError("program AD assembly triangular mask direct rule requires rank >= 2")
    return shape


def _program_ad_assembly_triangular_k(name: str, k: object) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"program AD assembly {name} direct rule requires static integer k")
    return int(k)


def _program_ad_assembly_triangular_derivative_rule(
    name: str,
    source_shape: Sequence[int],
    *,
    k: object = 0,
) -> CustomDerivativeRule:
    shape = _program_ad_assembly_triangular_source_shape(source_shape)
    k_value = _program_ad_assembly_triangular_k(name, k)
    source_size = _program_ad_array_static_size(shape)
    numpy_fn = np.tril if name == "tril" else np.triu

    def masked_array(values: NDArray[np.float64], role: str) -> NDArray[np.float64]:
        vector = _program_ad_array_vector(name, role, values, expected_size=source_size)
        return cast(NDArray[np.float64], numpy_fn(vector.reshape(shape), k=k_value))

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_float64_vector_result(masked_array(values, "values"))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_array_vector(name, "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(masked_array(tangent, "tangent"))

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_array_vector(name, "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(masked_array(cotangent, "cotangent"))

    return CustomDerivativeRule(
        name=f"program_ad_assembly_{name}_k{k_value}_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_assembly_tril_derivative_rule(
    source_shape: Sequence[int],
    *,
    k: object = 0,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.tril`` masks."""

    return _program_ad_assembly_triangular_derivative_rule("tril", source_shape, k=k)


def program_ad_assembly_triu_derivative_rule(
    source_shape: Sequence[int],
    *,
    k: object = 0,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.triu`` masks."""

    return _program_ad_assembly_triangular_derivative_rule("triu", source_shape, k=k)


def program_ad_assembly_diagonal_derivative_rule(
    source_shape: Sequence[int],
    *,
    offset: object = 0,
    axis1: object = 0,
    axis2: object = 1,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.diagonal`` gathers."""

    source = np.empty(
        _program_ad_array_normalise_static_shape("assembly diagonal source", source_shape),
        dtype=np.float64,
    )
    shape, offset_value, axis1_value, axis2_value, output_shape = (
        _program_ad_assembly_diagonal_static_parts((source, offset, axis1, axis2))
    )
    source_size = _program_ad_array_static_size(shape)
    output_size = _program_ad_array_static_size(output_shape)
    source_indices = np.arange(source_size, dtype=np.int64).reshape(shape)
    selected_indices = np.asarray(
        np.diagonal(source_indices, offset=offset_value, axis1=axis1_value, axis2=axis2_value),
        dtype=np.int64,
    ).reshape(-1)

    def gather_array(values: NDArray[np.float64], role: str) -> NDArray[np.float64]:
        vector = _program_ad_array_vector("diagonal", role, values, expected_size=source_size)
        return np.asarray(
            np.diagonal(
                vector.reshape(shape),
                offset=offset_value,
                axis1=axis1_value,
                axis2=axis2_value,
            ),
            dtype=np.float64,
        )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_float64_vector_result(gather_array(values, "values"))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_array_vector("diagonal", "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(gather_array(tangent, "tangent"))

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_array_vector("diagonal", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_array_vector(
            "diagonal", "cotangent", cotangent, expected_size=output_size
        )
        adjoint = np.zeros(source_size, dtype=np.float64)
        np.add.at(adjoint, selected_indices, cotangent_vector)
        return _program_ad_float64_vector_result(adjoint)

    return CustomDerivativeRule(
        name=(
            "program_ad_assembly_diagonal_"
            f"offset{offset_value}_axis{axis1_value}_{axis2_value}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_linalg_rank2_shape(
    name: str,
    source_shape: Sequence[int],
) -> tuple[int, int]:
    shape = tuple(int(dimension) for dimension in source_shape)
    if len(shape) != 2:
        raise ValueError(f"program AD linalg {name} derivative rule requires a rank-2 matrix")
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"program AD linalg {name} derivative rule dimensions must be positive")
    return shape


def _program_ad_linalg_trace_positions(
    matrix_shape: tuple[int, int],
    offset: int,
) -> tuple[tuple[int, int], ...]:
    rows, cols = matrix_shape
    positions = tuple((row, row + offset) for row in range(rows) if 0 <= row + offset < cols)
    if not positions:
        raise ValueError("program AD linalg trace offset selects an empty diagonal")
    return positions


def _program_ad_linalg_trace_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("trace", values)
    return _program_ad_float64_vector_result([float(np.trace(matrix))])


def _program_ad_linalg_trace_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("trace", values)
    tangent_matrix = _program_ad_linalg_square_matrix("trace", tangent)
    if tangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg trace tangent shape must match matrix shape")
    return _program_ad_float64_vector_result([float(np.trace(tangent_matrix))])


def _program_ad_linalg_trace_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("trace", values)
    cotangent_vector = _as_real_numeric_array(
        "program AD linalg trace cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.size != 1:
        raise ValueError("program AD linalg trace VJP cotangent must be scalar")
    return _program_ad_float64_vector_result(cotangent_vector[0] * np.eye(matrix.shape[0]))


def program_ad_linalg_trace_derivative_rule(
    matrix_shape: Sequence[int],
    *,
    offset: int | np.integer = 0,
    axis1: int | np.integer = 0,
    axis2: int | np.integer = 1,
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed trace primitive signature."""

    trace_axis1 = _program_ad_linalg_offset("trace", axis1)
    trace_axis2 = _program_ad_linalg_offset("trace", axis2)
    if (trace_axis1, trace_axis2) != (0, 1):
        raise ValueError("program AD linalg trace derivative rule supports axis1=0 and axis2=1")
    trace_offset = _program_ad_linalg_offset("trace", offset)
    static_shape = _program_ad_linalg_rank2_shape("trace", matrix_shape)
    positions = _program_ad_linalg_trace_positions(static_shape, trace_offset)
    size = _program_ad_shape_static_size(static_shape)

    def split(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg trace {name}", values).reshape(-1)
        if vector.size != size:
            raise ValueError("program AD linalg trace direct rule values size must match shape")
        return vector.reshape(static_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = split("values", values)
        return _program_ad_float64_vector_result(
            [sum(float(matrix[row, col]) for row, col in positions)]
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        del values
        tangent_matrix = split("tangent", tangent)
        return _program_ad_float64_vector_result(
            [sum(float(tangent_matrix[row, col]) for row, col in positions)]
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        split("values", values)
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg trace cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != 1:
            raise ValueError("program AD linalg trace VJP cotangent must be scalar")
        adjoint = np.zeros(static_shape, dtype=np.float64)
        for row, col in positions:
            adjoint[row, col] += cotangent_vector[0]
        return _program_ad_float64_vector_result(adjoint)

    return CustomDerivativeRule(
        name=(
            "program_ad_linalg_trace_"
            f"{_program_ad_shape_signature(static_shape)}_offset_{trace_offset}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_linalg_diag_positions(
    source_shape: tuple[int, ...],
    offset: int,
) -> tuple[tuple[int, int], ...]:
    if len(source_shape) == 1:
        size = source_shape[0] + abs(offset)
        positions = tuple(
            (index, index + offset) if offset >= 0 else (index - offset, index)
            for index in range(source_shape[0])
        )
        if any(row < 0 or row >= size or col < 0 or col >= size for row, col in positions):
            raise ValueError("program AD linalg diag offset is inconsistent with vector shape")
        return positions
    if len(source_shape) == 2:
        rows, cols = source_shape
        positions = tuple((row, row + offset) for row in range(rows) if 0 <= row + offset < cols)
        if not positions:
            raise ValueError("program AD linalg diag offset selects an empty diagonal")
        return positions
    raise ValueError("program AD linalg diag derivative rule requires rank-1 or rank-2 input")


def _program_ad_linalg_diag_shape_from_source(
    source_shape: tuple[int, ...],
    offset: int,
) -> tuple[int, ...]:
    positions = _program_ad_linalg_diag_positions(source_shape, offset)
    if len(source_shape) == 1:
        size = source_shape[0] + abs(offset)
        return (size, size)
    return (len(positions),)


def program_ad_linalg_diag_derivative_rule(
    source_shape: Sequence[int],
    *,
    k: int | np.integer = 0,
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed diagonal primitive signature."""

    static_shape = tuple(int(dimension) for dimension in source_shape)
    if len(static_shape) not in {1, 2}:
        raise ValueError("program AD linalg diag derivative rule requires rank-1 or rank-2 input")
    if any(dimension <= 0 for dimension in static_shape):
        raise ValueError("program AD linalg diag derivative rule dimensions must be positive")
    offset = _program_ad_linalg_offset("diag", k)
    positions = _program_ad_linalg_diag_positions(static_shape, offset)
    source_size = _program_ad_shape_static_size(static_shape)
    output_shape = _program_ad_linalg_diag_shape_from_source(static_shape, offset)
    output_size = _program_ad_shape_static_size(output_shape)

    def split_source(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg diag {name}", values).reshape(-1)
        if vector.size != source_size:
            raise ValueError("program AD linalg diag direct rule values size must match shape")
        return vector.reshape(static_shape)

    def split_output(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg diag {name}", values).reshape(-1)
        if vector.size != output_size:
            raise ValueError("program AD linalg diag VJP cotangent size must match output")
        return vector.reshape(output_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        source = split_source("values", values)
        return _program_ad_float64_vector_result(np.diag(source, k=offset))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        del values
        source_tangent = split_source("tangent", tangent)
        return _program_ad_float64_vector_result(np.diag(source_tangent, k=offset))

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        split_source("values", values)
        cotangent_array = split_output("cotangent", cotangent)
        if len(static_shape) == 1:
            return _program_ad_float64_vector_result(np.diag(cotangent_array, k=offset))
        adjoint = np.zeros(static_shape, dtype=np.float64)
        cotangent_vector = cotangent_array.reshape(-1)
        for index, (row, col) in enumerate(positions):
            adjoint[row, col] += cotangent_vector[index]
        return _program_ad_float64_vector_result(adjoint)

    return CustomDerivativeRule(
        name=(
            "program_ad_linalg_diag_"
            f"{_program_ad_shape_signature(static_shape)}_offset_{offset}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_linalg_diagflat_derivative_rule(
    source_shape: Sequence[int],
    *,
    k: int | np.integer = 0,
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed diagflat primitive signature."""

    static_shape = tuple(int(dimension) for dimension in source_shape)
    if any(dimension <= 0 for dimension in static_shape):
        raise ValueError("program AD linalg diagflat derivative rule dimensions must be positive")
    source_size = _program_ad_shape_static_size(static_shape)
    if source_size <= 0:
        raise ValueError("program AD linalg diagflat derivative rule requires non-empty input")
    offset = _program_ad_linalg_offset("diagflat", k)
    output_shape = (source_size + abs(offset), source_size + abs(offset))
    output_size = _program_ad_shape_static_size(output_shape)

    def split_source(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg diagflat {name}", values).reshape(-1)
        if vector.size != source_size:
            raise ValueError("program AD linalg diagflat direct rule values size must match shape")
        return vector.reshape(static_shape)

    def split_output(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg diagflat {name}", values).reshape(-1)
        if vector.size != output_size:
            raise ValueError("program AD linalg diagflat VJP cotangent size must match output")
        return vector.reshape(output_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        source = split_source("values", values)
        return _program_ad_float64_vector_result(np.diagflat(source, k=offset))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        split_source("values", values)
        tangent_source = split_source("tangent", tangent)
        return _program_ad_float64_vector_result(np.diagflat(tangent_source, k=offset))

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        split_source("values", values)
        cotangent_matrix = split_output("cotangent", cotangent)
        adjoint_flat = np.diag(cotangent_matrix, k=offset)
        if adjoint_flat.size != source_size:
            raise ValueError("program AD linalg diagflat VJP diagonal size must match source")
        return _program_ad_float64_vector_result(adjoint_flat.reshape(static_shape))

    return CustomDerivativeRule(
        name=(
            "program_ad_linalg_diagflat_"
            f"{_program_ad_shape_signature(static_shape)}_offset_{offset}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_linalg_require_symmetric(
    primitive_name: str,
    matrix: NDArray[np.float64],
) -> None:
    if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(f"program AD linalg {primitive_name} requires a symmetric matrix")


def _program_ad_linalg_require_distinct_eigenvalues(
    eigenvalues: NDArray[np.float64],
    primitive_name: str,
) -> None:
    if eigenvalues.size <= 1:
        return
    gaps = np.abs(eigenvalues[:, None] - eigenvalues[None, :])
    strict_gaps = gaps[np.triu_indices(eigenvalues.size, k=1)]
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(strict_gaps)) <= 1.0e-10 * scale:
        raise ValueError(f"program AD linalg {primitive_name} requires distinct eigenvalues")


def _program_ad_linalg_real_simple_eig_decomposition_from_matrix(
    primitive_name: str,
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if matrix.ndim != 2:
        raise ValueError(f"program AD linalg {primitive_name} requires a rank-2 matrix")
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"program AD linalg {primitive_name} requires a square matrix")
    eigenvalues_complex, right_complex = np.linalg.eig(matrix)
    eigenvalue_scale = max(1.0, float(np.max(np.abs(eigenvalues_complex))))
    tolerance = 1.0e-10 * eigenvalue_scale
    if float(np.max(np.abs(np.imag(eigenvalues_complex)))) > tolerance:
        raise ValueError(f"program AD linalg {primitive_name} requires real eigenvalues")
    eigenvalues = np.asarray(np.real(eigenvalues_complex), dtype=np.float64)
    _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, primitive_name)
    eigenvector_scale = max(1.0, float(np.max(np.abs(right_complex))))
    if float(np.max(np.abs(np.imag(right_complex)))) > 1.0e-10 * eigenvector_scale:
        raise ValueError(f"program AD linalg {primitive_name} requires real eigenvectors")
    right_eigenvectors = np.asarray(np.real(right_complex), dtype=np.float64)
    try:
        left_eigenvector_rows = np.linalg.inv(right_eigenvectors)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"program AD linalg {primitive_name} requires a diagonalizable matrix"
        ) from exc
    condition = float(np.linalg.cond(right_eigenvectors))
    if not math.isfinite(condition) or condition > 1.0e10:
        raise ValueError(
            f"program AD linalg {primitive_name} requires a well-conditioned eigenbasis"
        )
    return eigenvalues, right_eigenvectors, left_eigenvector_rows.astype(np.float64)


def _program_ad_linalg_real_simple_eig_decomposition(
    primitive_name: str,
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    matrix = _program_ad_linalg_square_matrix(primitive_name, values)
    eigenvalues, right_eigenvectors, left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition_from_matrix(primitive_name, matrix)
    )
    return matrix, eigenvalues, right_eigenvectors, left_eigenvector_rows


def _program_ad_linalg_require_distinct_positive_singular_values(
    singular_values: NDArray[np.float64],
    primitive_name: str,
) -> None:
    if singular_values.size == 0:
        raise ValueError(
            f"program AD linalg {primitive_name} requires distinct positive singular values"
        )
    scale = max(1.0, float(np.max(np.abs(singular_values))))
    tolerance = 1.0e-10 * scale
    if float(np.min(singular_values)) <= tolerance:
        raise ValueError(
            f"program AD linalg {primitive_name} requires distinct positive singular values"
        )
    if singular_values.size <= 1:
        return
    gaps = np.abs(singular_values[:, None] - singular_values[None, :])
    strict_gaps = gaps[np.triu_indices(singular_values.size, k=1)]
    if float(np.min(strict_gaps)) <= tolerance:
        raise ValueError(
            f"program AD linalg {primitive_name} requires distinct positive singular values"
        )


def _program_ad_linalg_normalise_rcond(value: object) -> float:
    if value is None:
        return 1.0e-15
    if isinstance(value, (TraceADScalar, TraceADArray)):
        raise ValueError("program AD linalg pinv rcond must be static")
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("program AD linalg pinv rcond must be a static real scalar")
    cutoff = float(value)
    if cutoff < 0.0 or not math.isfinite(cutoff):
        raise ValueError("program AD linalg pinv rcond must be finite and non-negative")
    return cutoff


def _program_ad_linalg_require_constant_full_rank(
    matrix: NDArray[np.float64],
    singular_values: NDArray[np.float64],
    *,
    rcond: float,
) -> None:
    rank = min(matrix.shape)
    if singular_values.size != rank:
        raise ValueError("program AD linalg pinv requires rank-2 singular values")
    scale = max(1.0, float(np.max(np.abs(singular_values))))
    threshold = rcond * scale
    if float(np.min(singular_values)) <= threshold:
        raise ValueError(
            "program AD linalg pinv requires a constant full-rank matrix above cutoff"
        )


def _program_ad_linalg_pinv_value_matrix(
    matrix: NDArray[np.float64],
    *,
    rcond: float = 1.0e-15,
) -> NDArray[np.float64]:
    if matrix.ndim != 2:
        raise ValueError("program AD linalg pinv requires a rank-2 matrix")
    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError("program AD linalg pinv requires non-empty matrix dimensions")
    _left, singular_values, _right_h = np.linalg.svd(matrix, full_matrices=False)
    _program_ad_linalg_require_constant_full_rank(matrix, singular_values, rcond=rcond)
    return np.linalg.pinv(matrix, rcond=rcond, hermitian=False).astype(np.float64)


def _program_ad_linalg_pinv_jvp_matrix(
    matrix: NDArray[np.float64],
    pinv: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    if tangent.shape != matrix.shape:
        raise ValueError("program AD linalg pinv tangent shape must match matrix shape")
    left_projector = np.eye(matrix.shape[1], dtype=np.float64) - pinv @ matrix
    right_projector = np.eye(matrix.shape[0], dtype=np.float64) - matrix @ pinv
    return cast(
        NDArray[np.float64],
        (
            -pinv @ tangent @ pinv
            + pinv @ pinv.T @ tangent.T @ right_projector
            + left_projector @ tangent.T @ pinv.T @ pinv
        ).astype(np.float64),
    )


def _program_ad_linalg_pinv_vjp_matrix(
    matrix: NDArray[np.float64],
    pinv: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    if cotangent.shape != pinv.shape:
        raise ValueError("program AD linalg pinv VJP cotangent shape must match output shape")
    left_projector = np.eye(matrix.shape[1], dtype=np.float64) - pinv @ matrix
    right_projector = np.eye(matrix.shape[0], dtype=np.float64) - matrix @ pinv
    return cast(
        NDArray[np.float64],
        (
            -pinv.T @ cotangent @ pinv.T
            + right_projector.T @ cotangent.T @ pinv @ pinv.T
            + pinv.T @ pinv @ cotangent.T @ left_projector.T
        ).astype(np.float64),
    )


def _program_ad_linalg_pinv_square_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("pinv", values)
    return _program_ad_float64_vector_result(_program_ad_linalg_pinv_value_matrix(matrix))


def _program_ad_linalg_pinv_square_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("pinv", values)
    tangent_matrix = _program_ad_linalg_square_matrix("pinv tangent", tangent)
    pinv = _program_ad_linalg_pinv_value_matrix(matrix)
    return _program_ad_float64_vector_result(
        _program_ad_linalg_pinv_jvp_matrix(matrix, pinv, tangent_matrix)
    )


def _program_ad_linalg_pinv_square_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("pinv", values)
    pinv = _program_ad_linalg_pinv_value_matrix(matrix)
    cotangent_matrix = _as_real_numeric_array("program AD linalg pinv cotangent", cotangent)
    if cotangent_matrix.size != pinv.size:
        raise ValueError("program AD linalg pinv VJP cotangent size must match output")
    return _program_ad_float64_vector_result(
        _program_ad_linalg_pinv_vjp_matrix(matrix, pinv, cotangent_matrix.reshape(pinv.shape))
    )


def _program_ad_linalg_uplo(
    value: object,
    primitive_name: str,
) -> Literal["L", "U"]:
    uplo_value = str(value).upper()
    if uplo_value not in {"L", "U"}:
        raise ValueError(f"program AD linalg {primitive_name} requires UPLO='L' or UPLO='U'")
    return cast(Literal["L", "U"], uplo_value)


def _program_ad_linalg_eigvalsh_decomposition(
    primitive_name: str,
    values: NDArray[np.float64],
    *,
    uplo: str = "L",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    matrix = _program_ad_linalg_square_matrix(primitive_name, values)
    _program_ad_linalg_require_symmetric(primitive_name, matrix)
    uplo_value = _program_ad_linalg_uplo(uplo, primitive_name)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix, UPLO=uplo_value)
    _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, primitive_name)
    return matrix, eigenvalues.astype(np.float64), eigenvectors.astype(np.float64)


def _program_ad_linalg_eigvalsh_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    _matrix, eigenvalues, _eigenvectors = _program_ad_linalg_eigvalsh_decomposition(
        "eigvalsh", values
    )
    return _program_ad_float64_vector_result(eigenvalues)


def _program_ad_linalg_eigvals_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    _matrix, eigenvalues, _right_eigenvectors, _left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition("eigvals", values)
    )
    return _program_ad_float64_vector_result(eigenvalues)


def _program_ad_linalg_eigvals_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, _eigenvalues, right_eigenvectors, left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition("eigvals", values)
    )
    tangent_matrix = _program_ad_linalg_square_matrix("eigvals tangent", tangent)
    if tangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg eigvals tangent shape must match matrix shape")
    return _program_ad_float64_vector_result(
        [
            float(left_eigenvector_rows[index, :] @ tangent_matrix @ right_eigenvectors[:, index])
            for index in range(matrix.shape[0])
        ]
    )


def _program_ad_linalg_eigvals_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, _eigenvalues, right_eigenvectors, left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition("eigvals", values)
    )
    cotangent_vector = _as_real_numeric_array(
        "program AD linalg eigvals cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.size != matrix.shape[0]:
        raise ValueError("program AD linalg eigvals VJP cotangent size must match spectrum")
    adjoint = np.zeros_like(matrix, dtype=np.float64)
    for index, weight in enumerate(cotangent_vector):
        adjoint = adjoint + float(weight) * np.outer(
            left_eigenvector_rows[index, :], right_eigenvectors[:, index]
        )
    return _program_ad_float64_vector_result(adjoint)


def _program_ad_linalg_eig_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    _matrix, eigenvalues, right_eigenvectors, _left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition("eig", values)
    )
    return _program_ad_float64_vector_result(
        np.concatenate((eigenvalues, right_eigenvectors.reshape(-1)))
    )


def _program_ad_linalg_eig_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, eigenvalues, right_eigenvectors, left_eigenvector_rows = (
        _program_ad_linalg_real_simple_eig_decomposition("eig", values)
    )
    tangent_matrix = _program_ad_linalg_square_matrix("eig tangent", tangent)
    if tangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg eig tangent shape must match matrix shape")
    eigenvalue_tangent = np.array(
        [
            float(left_eigenvector_rows[index, :] @ tangent_matrix @ right_eigenvectors[:, index])
            for index in range(matrix.shape[0])
        ],
        dtype=np.float64,
    )
    eigenvector_tangent = _program_ad_linalg_eig_eigenvector_jvp_matrix(
        eigenvalues, right_eigenvectors, left_eigenvector_rows, tangent_matrix
    )
    return _program_ad_float64_vector_result(
        np.concatenate((eigenvalue_tangent, eigenvector_tangent.reshape(-1)))
    )


def _program_ad_linalg_eig_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _program_ad_linalg_square_matrix("eig", values)
    cotangent_vector = _as_real_numeric_array(
        "program AD linalg eig cotangent", cotangent
    ).reshape(-1)
    output_size = matrix.shape[0] + matrix.size
    if cotangent_vector.size != output_size:
        raise ValueError("program AD linalg eig VJP cotangent size must match output")
    adjoint = np.zeros_like(matrix, dtype=np.float64)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            basis = np.zeros_like(matrix, dtype=np.float64)
            basis[row, col] = 1.0
            jvp = _program_ad_linalg_eig_jvp(values, basis.reshape(-1))
            adjoint[row, col] = float(jvp @ cotangent_vector)
    return _program_ad_float64_vector_result(adjoint)


def program_ad_linalg_eig_derivative_rule(
    matrix_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed real-simple eig primitive."""

    static_shape = _program_ad_linalg_rank2_shape("eig", matrix_shape)
    if static_shape[0] != static_shape[1]:
        raise ValueError("program AD linalg eig derivative rule requires a square matrix")
    matrix_size = _program_ad_shape_static_size(static_shape)
    output_size = static_shape[0] + matrix_size

    def split(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg eig {name}", values).reshape(-1)
        if vector.size != matrix_size:
            raise ValueError("program AD linalg eig direct rule values size must match shape")
        return vector.reshape(static_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = split("values", values)
        eigenvalues, right_eigenvectors, _left_eigenvector_rows = (
            _program_ad_linalg_real_simple_eig_decomposition_from_matrix("eig", matrix)
        )
        return _program_ad_float64_vector_result(
            np.concatenate((eigenvalues, right_eigenvectors.reshape(-1)))
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = split("values", values)
        tangent_matrix = split("tangent", tangent)
        eigenvalues, right_eigenvectors, left_eigenvector_rows = (
            _program_ad_linalg_real_simple_eig_decomposition_from_matrix("eig", matrix)
        )
        eigenvalue_tangent = np.array(
            [
                float(
                    left_eigenvector_rows[index, :] @ tangent_matrix @ right_eigenvectors[:, index]
                )
                for index in range(static_shape[0])
            ],
            dtype=np.float64,
        )
        eigenvector_tangent = _program_ad_linalg_eig_eigenvector_jvp_matrix(
            eigenvalues, right_eigenvectors, left_eigenvector_rows, tangent_matrix
        )
        return _program_ad_float64_vector_result(
            np.concatenate((eigenvalue_tangent, eigenvector_tangent.reshape(-1)))
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        split("values", values)
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg eig cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != output_size:
            raise ValueError("program AD linalg eig VJP cotangent size must match output")
        adjoint = np.zeros(static_shape, dtype=np.float64)
        for row in range(static_shape[0]):
            for col in range(static_shape[1]):
                basis = np.zeros(static_shape, dtype=np.float64)
                basis[row, col] = 1.0
                adjoint[row, col] = float(jvp_rule(values, basis.reshape(-1)) @ cotangent_vector)
        return _program_ad_float64_vector_result(adjoint)

    return CustomDerivativeRule(
        name=(f"program_ad_linalg_eig_{_program_ad_shape_signature(static_shape)}_direct_rule"),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_linalg_eigvals_derivative_rule(
    matrix_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed real-simple eigvals primitive."""

    static_shape = _program_ad_linalg_rank2_shape("eigvals", matrix_shape)
    if static_shape[0] != static_shape[1]:
        raise ValueError("program AD linalg eigvals derivative rule requires a square matrix")
    matrix_size = _program_ad_shape_static_size(static_shape)

    def split(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg eigvals {name}", values).reshape(-1)
        if vector.size != matrix_size:
            raise ValueError("program AD linalg eigvals direct rule values size must match shape")
        return vector.reshape(static_shape)

    def decompose(
        name: str, values: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        matrix = split(name, values)
        return _program_ad_linalg_real_simple_eig_decomposition_from_matrix("eigvals", matrix)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        eigenvalues, _right_eigenvectors, _left_eigenvector_rows = decompose("values", values)
        return _program_ad_float64_vector_result(eigenvalues)

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _eigenvalues, right_eigenvectors, left_eigenvector_rows = decompose("values", values)
        tangent_matrix = split("tangent", tangent)
        return _program_ad_float64_vector_result(
            [
                float(
                    left_eigenvector_rows[index, :] @ tangent_matrix @ right_eigenvectors[:, index]
                )
                for index in range(static_shape[0])
            ]
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _eigenvalues, right_eigenvectors, left_eigenvector_rows = decompose("values", values)
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg eigvals cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != static_shape[0]:
            raise ValueError("program AD linalg eigvals VJP cotangent size must match spectrum")
        adjoint = np.zeros(static_shape, dtype=np.float64)
        for index, weight in enumerate(cotangent_vector):
            adjoint = adjoint + float(weight) * np.outer(
                left_eigenvector_rows[index, :], right_eigenvectors[:, index]
            )
        return _program_ad_float64_vector_result(adjoint)

    return CustomDerivativeRule(
        name=(
            f"program_ad_linalg_eigvals_{_program_ad_shape_signature(static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_linalg_eigvalsh_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, _eigenvalues, eigenvectors = _program_ad_linalg_eigvalsh_decomposition(
        "eigvalsh", values
    )
    tangent_matrix = _program_ad_linalg_square_matrix("eigvalsh tangent", tangent)
    if tangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg eigvalsh tangent shape must match matrix shape")
    _program_ad_linalg_require_symmetric("eigvalsh tangent", tangent_matrix)
    return _program_ad_float64_vector_result(
        np.array(
            [
                float(eigenvector.T @ tangent_matrix @ eigenvector)
                for eigenvector in eigenvectors.T
            ],
            dtype=np.float64,
        )
    )


def _program_ad_linalg_eigvalsh_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    _matrix, eigenvalues, eigenvectors = _program_ad_linalg_eigvalsh_decomposition(
        "eigvalsh", values
    )
    cotangent_vector = _as_real_numeric_array(
        "program AD linalg eigvalsh cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.size != eigenvalues.size:
        raise ValueError("program AD linalg eigvalsh VJP cotangent size must match eigenvalues")
    adjoint = eigenvectors @ np.diag(cotangent_vector) @ eigenvectors.T
    return _program_ad_float64_vector_result(adjoint)


def _program_ad_linalg_eigh_eigenvector_jvp_matrix(
    eigenvalues: NDArray[np.float64],
    eigenvectors: NDArray[np.float64],
    tangent_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    size = eigenvalues.size
    tangent = np.zeros_like(eigenvectors, dtype=np.float64)
    for column in range(size):
        for other in range(size):
            if other == column:
                continue
            scale = float(
                eigenvectors[:, other].T @ tangent_matrix @ eigenvectors[:, column]
            ) / float(eigenvalues[column] - eigenvalues[other])
            tangent[:, column] = tangent[:, column] + scale * eigenvectors[:, other]
    return tangent


def _program_ad_linalg_eigh_vjp_matrix(
    eigenvalues: NDArray[np.float64],
    eigenvectors: NDArray[np.float64],
    eigenvalue_cotangent: NDArray[np.float64],
    eigenvector_cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    adjoint = eigenvectors @ np.diag(eigenvalue_cotangent) @ eigenvectors.T
    size = eigenvalues.size
    for column in range(size):
        cotangent_column = eigenvector_cotangent[:, column]
        for other in range(size):
            if other == column:
                continue
            scale = float(eigenvectors[:, other].T @ cotangent_column) / float(
                eigenvalues[column] - eigenvalues[other]
            )
            adjoint = adjoint + scale * np.outer(eigenvectors[:, other], eigenvectors[:, column])
    symmetric_adjoint = 0.5 * (adjoint + adjoint.T)
    return cast(NDArray[np.float64], symmetric_adjoint.astype(np.float64))


def _program_ad_linalg_eigh_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    _matrix, eigenvalues, eigenvectors = _program_ad_linalg_eigvalsh_decomposition("eigh", values)
    return _program_ad_float64_vector_result(
        np.concatenate((eigenvalues, eigenvectors.reshape(-1)))
    )


def _program_ad_linalg_eigh_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, eigenvalues, eigenvectors = _program_ad_linalg_eigvalsh_decomposition("eigh", values)
    tangent_matrix = _program_ad_linalg_square_matrix("eigh tangent", tangent)
    if tangent_matrix.shape != matrix.shape:
        raise ValueError("program AD linalg eigh tangent shape must match matrix shape")
    _program_ad_linalg_require_symmetric("eigh tangent", tangent_matrix)
    eigenvalue_tangent = np.array(
        [float(eigenvector.T @ tangent_matrix @ eigenvector) for eigenvector in eigenvectors.T],
        dtype=np.float64,
    )
    eigenvector_tangent = _program_ad_linalg_eigh_eigenvector_jvp_matrix(
        eigenvalues, eigenvectors, tangent_matrix
    )
    return _program_ad_float64_vector_result(
        np.concatenate((eigenvalue_tangent, eigenvector_tangent.reshape(-1)))
    )


def _program_ad_linalg_eigh_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix, eigenvalues, eigenvectors = _program_ad_linalg_eigvalsh_decomposition("eigh", values)
    cotangent_vector = _as_real_numeric_array(
        "program AD linalg eigh cotangent", cotangent
    ).reshape(-1)
    size = matrix.shape[0]
    output_size = size + size * size
    if cotangent_vector.size != output_size:
        raise ValueError("program AD linalg eigh VJP cotangent size must match output")
    eigenvalue_cotangent = cotangent_vector[:size]
    eigenvector_cotangent = cotangent_vector[size:].reshape(size, size)
    return _program_ad_float64_vector_result(
        _program_ad_linalg_eigh_vjp_matrix(
            eigenvalues, eigenvectors, eigenvalue_cotangent, eigenvector_cotangent
        )
    )


def program_ad_linalg_eigh_derivative_rule(
    matrix_shape: Sequence[int],
    *,
    uplo: str = "L",
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed symmetric eigh primitive."""

    static_shape = _program_ad_linalg_rank2_shape("eigh", matrix_shape)
    if static_shape[0] != static_shape[1]:
        raise ValueError("program AD linalg eigh derivative rule requires a square matrix")
    uplo_value = _program_ad_linalg_uplo(uplo, "eigh derivative rule")
    matrix_size = _program_ad_shape_static_size(static_shape)
    output_size = static_shape[0] + matrix_size

    def split(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg eigh {name}", values).reshape(-1)
        if vector.size != matrix_size:
            raise ValueError("program AD linalg eigh direct rule values size must match shape")
        return vector.reshape(static_shape)

    def decompose(
        name: str, values: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        matrix = split(name, values)
        _program_ad_linalg_require_symmetric("eigh", matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix, UPLO=uplo_value)
        _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, "eigh")
        return matrix, eigenvalues, eigenvectors

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        _matrix, eigenvalues, eigenvectors = decompose("values", values)
        return _program_ad_float64_vector_result(
            np.concatenate((eigenvalues, eigenvectors.reshape(-1)))
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _matrix, eigenvalues, eigenvectors = decompose("values", values)
        tangent_matrix = split("tangent", tangent)
        _program_ad_linalg_require_symmetric("eigh tangent", tangent_matrix)
        eigenvalue_tangent = np.array(
            [
                float(eigenvector.T @ tangent_matrix @ eigenvector)
                for eigenvector in eigenvectors.T
            ],
            dtype=np.float64,
        )
        eigenvector_tangent = _program_ad_linalg_eigh_eigenvector_jvp_matrix(
            eigenvalues, eigenvectors, tangent_matrix
        )
        return _program_ad_float64_vector_result(
            np.concatenate((eigenvalue_tangent, eigenvector_tangent.reshape(-1)))
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _matrix, eigenvalues, eigenvectors = decompose("values", values)
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg eigh cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != output_size:
            raise ValueError("program AD linalg eigh VJP cotangent size must match output")
        eigenvalue_cotangent = cotangent_vector[: static_shape[0]]
        eigenvector_cotangent = cotangent_vector[static_shape[0] :].reshape(static_shape)
        return _program_ad_float64_vector_result(
            _program_ad_linalg_eigh_vjp_matrix(
                eigenvalues, eigenvectors, eigenvalue_cotangent, eigenvector_cotangent
            )
        )

    return CustomDerivativeRule(
        name=(f"program_ad_linalg_eigh_{_program_ad_shape_signature(static_shape)}_direct_rule"),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_linalg_eigvalsh_derivative_rule(
    matrix_shape: Sequence[int],
    *,
    uplo: str = "L",
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed symmetric eigvalsh primitive."""

    static_shape = _program_ad_linalg_rank2_shape("eigvalsh", matrix_shape)
    if static_shape[0] != static_shape[1]:
        raise ValueError("program AD linalg eigvalsh derivative rule requires a square matrix")
    uplo_value = _program_ad_linalg_uplo(uplo, "eigvalsh derivative rule")
    matrix_size = _program_ad_shape_static_size(static_shape)

    def split(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg eigvalsh {name}", values).reshape(-1)
        if vector.size != matrix_size:
            raise ValueError("program AD linalg eigvalsh direct rule values size must match shape")
        return vector.reshape(static_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = split("values", values)
        _program_ad_linalg_require_symmetric("eigvalsh", matrix)
        eigenvalues, _eigenvectors = np.linalg.eigh(matrix, UPLO=uplo_value)
        _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, "eigvalsh")
        return _program_ad_float64_vector_result(eigenvalues)

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = split("values", values)
        tangent_matrix = split("tangent", tangent)
        _program_ad_linalg_require_symmetric("eigvalsh", matrix)
        _program_ad_linalg_require_symmetric("eigvalsh tangent", tangent_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix, UPLO=uplo_value)
        _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, "eigvalsh")
        return _program_ad_float64_vector_result(
            np.array(
                [
                    float(eigenvector.T @ tangent_matrix @ eigenvector)
                    for eigenvector in eigenvectors.T
                ],
                dtype=np.float64,
            )
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        matrix = split("values", values)
        _program_ad_linalg_require_symmetric("eigvalsh", matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix, UPLO=uplo_value)
        _program_ad_linalg_require_distinct_eigenvalues(eigenvalues, "eigvalsh")
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg eigvalsh cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != static_shape[0]:
            raise ValueError(
                "program AD linalg eigvalsh VJP cotangent size must match eigenvalues"
            )
        return _program_ad_float64_vector_result(
            eigenvectors @ np.diag(cotangent_vector) @ eigenvectors.T
        )

    return CustomDerivativeRule(
        name=(
            f"program_ad_linalg_eigvalsh_{_program_ad_shape_signature(static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_linalg_svdvals_derivative_rule(
    matrix_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for fixed-shape SVD singular values."""

    static_shape = _program_ad_linalg_rank2_shape("svd", matrix_shape)
    if any(dimension <= 0 for dimension in static_shape):
        raise ValueError("program AD linalg svd derivative rule requires positive dimensions")
    output_size = min(static_shape)
    matrix_size = _program_ad_shape_static_size(static_shape)

    def split(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg svd {name}", values).reshape(-1)
        if vector.size != matrix_size:
            raise ValueError("program AD linalg svd direct rule values size must match shape")
        return vector.reshape(static_shape)

    def decompose(
        name: str,
        values: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        matrix = split(name, values)
        left, singular_values, right_h = np.linalg.svd(matrix, full_matrices=False)
        _program_ad_linalg_require_distinct_positive_singular_values(singular_values, "svd")
        return matrix, left, singular_values, right_h

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        _matrix, _left, singular_values, _right_h = decompose("values", values)
        return _program_ad_float64_vector_result(singular_values)

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _matrix, left, _singular_values, right_h = decompose("values", values)
        tangent_matrix = split("tangent", tangent)
        return _program_ad_float64_vector_result(
            np.array(
                [
                    float(left[:, index].T @ tangent_matrix @ right_h[index, :])
                    for index in range(output_size)
                ],
                dtype=np.float64,
            )
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _matrix, left, _singular_values, right_h = decompose("values", values)
        cotangent_vector = _as_real_numeric_array(
            "program AD linalg svd cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != output_size:
            raise ValueError("program AD linalg svd VJP cotangent size must match singular values")
        return _program_ad_float64_vector_result(left @ np.diag(cotangent_vector) @ right_h)

    return CustomDerivativeRule(
        name=(
            f"program_ad_linalg_svdvals_{_program_ad_shape_signature(static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_linalg_pinv_derivative_rule(
    matrix_shape: Sequence[int],
    *,
    rcond: float | None = None,
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for fixed-shape full-rank pseudoinverse."""

    static_shape = _program_ad_linalg_rank2_shape("pinv", matrix_shape)
    if any(dimension <= 0 for dimension in static_shape):
        raise ValueError("program AD linalg pinv derivative rule requires positive dimensions")
    cutoff = _program_ad_linalg_normalise_rcond(rcond)
    output_shape = (static_shape[1], static_shape[0])
    input_size = _program_ad_shape_static_size(static_shape)
    output_size = _program_ad_shape_static_size(output_shape)

    def split_input(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg pinv {name}", values).reshape(-1)
        if vector.size != input_size:
            raise ValueError("program AD linalg pinv direct rule values size must match shape")
        return vector.reshape(static_shape)

    def split_output(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _as_real_numeric_array(f"program AD linalg pinv {name}", values).reshape(-1)
        if vector.size != output_size:
            raise ValueError("program AD linalg pinv VJP cotangent size must match output")
        return vector.reshape(output_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = split_input("values", values)
        return _program_ad_float64_vector_result(
            _program_ad_linalg_pinv_value_matrix(matrix, rcond=cutoff)
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = split_input("values", values)
        tangent_matrix = split_input("tangent", tangent)
        pinv = _program_ad_linalg_pinv_value_matrix(matrix, rcond=cutoff)
        return _program_ad_float64_vector_result(
            _program_ad_linalg_pinv_jvp_matrix(matrix, pinv, tangent_matrix)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        matrix = split_input("values", values)
        cotangent_matrix = split_output("cotangent", cotangent)
        pinv = _program_ad_linalg_pinv_value_matrix(matrix, rcond=cutoff)
        return _program_ad_float64_vector_result(
            _program_ad_linalg_pinv_vjp_matrix(matrix, pinv, cotangent_matrix)
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_linalg_pinv_"
            f"{_program_ad_shape_signature(static_shape)}_rcond_{cutoff:.3e}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_linalg_derivative_rule(name: str) -> CustomDerivativeRule:
    if name == "det":
        return CustomDerivativeRule(
            name="program_ad_linalg_det_direct_rule",
            value_fn=_program_ad_linalg_det_value,
            jvp_rule=_program_ad_linalg_det_jvp,
            vjp_rule=_program_ad_linalg_det_vjp,
        )
    if name == "inv":
        return CustomDerivativeRule(
            name="program_ad_linalg_inv_direct_rule",
            value_fn=_program_ad_linalg_inv_value,
            jvp_rule=_program_ad_linalg_inv_jvp,
            vjp_rule=_program_ad_linalg_inv_vjp,
        )
    if name == "solve":
        return CustomDerivativeRule(
            name="program_ad_linalg_solve_direct_rule",
            value_fn=_program_ad_linalg_solve_value,
            jvp_rule=_program_ad_linalg_solve_jvp,
            vjp_rule=_program_ad_linalg_solve_vjp,
        )
    if name == "trace":
        return CustomDerivativeRule(
            name="program_ad_linalg_trace_direct_rule",
            value_fn=_program_ad_linalg_trace_value,
            jvp_rule=_program_ad_linalg_trace_jvp,
            vjp_rule=_program_ad_linalg_trace_vjp,
        )
    if name == "eig":
        return CustomDerivativeRule(
            name="program_ad_linalg_eig_direct_rule",
            value_fn=_program_ad_linalg_eig_value,
            jvp_rule=_program_ad_linalg_eig_jvp,
            vjp_rule=_program_ad_linalg_eig_vjp,
        )
    if name == "eigh":
        return CustomDerivativeRule(
            name="program_ad_linalg_eigh_direct_rule",
            value_fn=_program_ad_linalg_eigh_value,
            jvp_rule=_program_ad_linalg_eigh_jvp,
            vjp_rule=_program_ad_linalg_eigh_vjp,
        )
    if name == "eigvalsh":
        return CustomDerivativeRule(
            name="program_ad_linalg_eigvalsh_direct_rule",
            value_fn=_program_ad_linalg_eigvalsh_value,
            jvp_rule=_program_ad_linalg_eigvalsh_jvp,
            vjp_rule=_program_ad_linalg_eigvalsh_vjp,
        )
    if name == "eigvals":
        return CustomDerivativeRule(
            name="program_ad_linalg_eigvals_direct_rule",
            value_fn=_program_ad_linalg_eigvals_value,
            jvp_rule=_program_ad_linalg_eigvals_jvp,
            vjp_rule=_program_ad_linalg_eigvals_vjp,
        )
    if name == "pinv":
        return CustomDerivativeRule(
            name="program_ad_linalg_pinv_square_direct_rule",
            value_fn=_program_ad_linalg_pinv_square_value,
            jvp_rule=_program_ad_linalg_pinv_square_jvp,
            vjp_rule=_program_ad_linalg_pinv_square_vjp,
        )
    return CustomDerivativeRule(
        name=f"program_ad_linalg_{name}_trace_contract",
        value_fn=_program_ad_linalg_direct_value,
        jvp_rule=_program_ad_linalg_direct_jvp,
    )


def _program_ad_linalg_shape_of(value: object) -> tuple[int, ...]:
    if isinstance(value, TraceADArray):
        return value.shape
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _program_ad_linalg_require_matrix_shape(name: str, value: object) -> tuple[int, int]:
    shape = _program_ad_linalg_shape_of(value)
    if len(shape) != 2:
        raise ValueError(f"program AD linalg {name} shape rule requires a rank-2 matrix")
    rows, cols = shape
    if rows != cols:
        raise ValueError(f"program AD linalg {name} shape rule requires a square matrix")
    return rows, cols


def _program_ad_linalg_det_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg det shape rule requires one matrix")
    _program_ad_linalg_require_matrix_shape("det", args[0])
    return ()


def _program_ad_linalg_inv_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg inv shape rule requires one matrix")
    return _program_ad_linalg_require_matrix_shape("inv", args[0])


def _program_ad_linalg_solve_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD linalg solve shape rule requires matrix and right-hand side")
    rows, _cols = _program_ad_linalg_require_matrix_shape("solve", args[0])
    rhs_shape = _program_ad_linalg_shape_of(args[1])
    if len(rhs_shape) == 1:
        if rhs_shape[0] != rows:
            raise ValueError("program AD linalg solve shape rule vector length must match matrix")
        return rhs_shape
    if len(rhs_shape) == 2:
        if rhs_shape[0] != rows:
            raise ValueError("program AD linalg solve shape rule rhs rows must match matrix")
        return rhs_shape
    raise ValueError("program AD linalg solve shape rule requires rank-1 or rank-2 rhs")


def _program_ad_linalg_trace_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 4}:
        raise ValueError(
            "program AD linalg trace shape rule requires matrix and optional static axes"
        )
    shape = _program_ad_linalg_shape_of(args[0])
    if len(shape) != 2:
        raise ValueError("program AD linalg trace shape rule requires a rank-2 matrix")
    offset = 0
    axis1 = 0
    axis2 = 1
    if len(args) == 4:
        offset = _program_ad_linalg_offset("trace", cast(int | np.integer, args[1]))
        axis1 = _program_ad_linalg_offset("trace", cast(int | np.integer, args[2]))
        axis2 = _program_ad_linalg_offset("trace", cast(int | np.integer, args[3]))
    if (axis1, axis2) != (0, 1):
        raise ValueError("program AD linalg trace shape rule supports axis1=0 and axis2=1")
    _program_ad_linalg_trace_positions(shape, offset)
    return ()


def _program_ad_linalg_diag_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD linalg diag shape rule requires source and optional offset")
    shape = _program_ad_linalg_shape_of(args[0])
    offset = 0
    if len(args) == 2:
        offset = _program_ad_linalg_offset("diag", cast(int | np.integer, args[1]))
    if len(shape) not in {1, 2}:
        raise ValueError("program AD linalg diag shape rule requires rank-1 or rank-2 input")
    if any(dimension <= 0 for dimension in shape):
        raise ValueError("program AD linalg diag shape rule dimensions must be positive")
    return _program_ad_linalg_diag_shape_from_source(shape, offset)


def _program_ad_linalg_diagflat_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) not in {1, 2}:
        raise ValueError(
            "program AD linalg diagflat shape rule requires source and optional offset"
        )
    shape = _program_ad_linalg_shape_of(args[0])
    offset = 0
    if len(args) == 2:
        offset = _program_ad_linalg_offset("diagflat", cast(int | np.integer, args[1]))
    source_size = int(np.prod(shape))
    if source_size <= 0:
        raise ValueError("program AD linalg diagflat shape rule requires non-empty input")
    output_size = source_size + abs(offset)
    return (output_size, output_size)


def _program_ad_linalg_matrix_power_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD linalg matrix_power shape rule requires matrix and power")
    if isinstance(args[1], bool) or not isinstance(args[1], (int, np.integer)):
        raise ValueError(
            "program AD linalg matrix_power shape rule requires a static integer power"
        )
    return _program_ad_linalg_require_matrix_shape("matrix_power", args[0])


def _program_ad_linalg_multi_dot_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg multi_dot shape rule requires one operand sequence")
    operands = args[0]
    if isinstance(operands, (TraceADArray, np.ndarray)) or not isinstance(operands, Sequence):
        raise ValueError(
            "program AD linalg multi_dot shape rule requires a static operand sequence"
        )
    shapes = tuple(_program_ad_linalg_shape_of(operand) for operand in operands)
    if len(shapes) < 2:
        raise ValueError("program AD linalg multi_dot shape rule requires at least two operands")
    for index, shape in enumerate(shapes):
        if len(shape) not in {1, 2}:
            raise ValueError(
                "program AD linalg multi_dot shape rule supports rank-1 and rank-2 operands"
            )
        if 0 < index < len(shapes) - 1 and len(shape) != 2:
            raise ValueError(
                "program AD linalg multi_dot shape rule middle operands must be rank-2"
            )

    result_shape = shapes[0]
    for next_shape in shapes[1:]:
        if len(result_shape) == 1 and len(next_shape) == 1:
            if result_shape[0] != next_shape[0]:
                raise ValueError("program AD linalg multi_dot shape rule dimensions must align")
            result_shape = ()
        elif len(result_shape) == 1 and len(next_shape) == 2:
            if result_shape[0] != next_shape[0]:
                raise ValueError("program AD linalg multi_dot shape rule dimensions must align")
            result_shape = (next_shape[1],)
        elif len(result_shape) == 2 and len(next_shape) == 1:
            if result_shape[1] != next_shape[0]:
                raise ValueError("program AD linalg multi_dot shape rule dimensions must align")
            result_shape = (result_shape[0],)
        elif len(result_shape) == 2 and len(next_shape) == 2:
            if result_shape[1] != next_shape[0]:
                raise ValueError("program AD linalg multi_dot shape rule dimensions must align")
            result_shape = (result_shape[0], next_shape[1])
        else:
            raise ValueError(
                "program AD linalg multi_dot shape rule encountered scalar intermediate"
            )
    return result_shape


def _program_ad_linalg_eigh_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg eigh shape rule requires one matrix")
    rows, _cols = _program_ad_linalg_require_matrix_shape("eigh", args[0])
    return (rows, rows, rows)


def _program_ad_linalg_eig_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg eig shape rule requires one matrix")
    rows, _cols = _program_ad_linalg_require_matrix_shape("eig", args[0])
    return (rows, rows, rows)


def _program_ad_linalg_eigvalsh_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg eigvalsh shape rule requires one matrix")
    rows, _cols = _program_ad_linalg_require_matrix_shape("eigvalsh", args[0])
    return (rows,)


def _program_ad_linalg_eigvals_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg eigvals shape rule requires one matrix")
    rows, _cols = _program_ad_linalg_require_matrix_shape("eigvals", args[0])
    return (rows,)


def _program_ad_linalg_svd_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg svd shape rule requires one matrix")
    shape = _program_ad_linalg_shape_of(args[0])
    if len(shape) != 2:
        raise ValueError("program AD linalg svd shape rule requires a rank-2 matrix")
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        raise ValueError("program AD linalg svd shape rule requires non-empty dimensions")
    return (min(rows, cols),)


def _program_ad_linalg_pinv_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg pinv shape rule requires one matrix")
    shape = _program_ad_linalg_shape_of(args[0])
    if len(shape) != 2:
        raise ValueError("program AD linalg pinv shape rule requires a rank-2 matrix")
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        raise ValueError("program AD linalg pinv shape rule requires non-empty dimensions")
    return (cols, rows)


def _program_ad_linalg_dtype_rule(args: tuple[object, ...]) -> str:
    arrays: list[NDArray[np.float64]] = []
    for arg in args:
        if isinstance(arg, TraceADArray):
            continue
        if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes, np.ndarray)):
            for item in arg:
                if isinstance(item, TraceADArray):
                    continue
                arrays.append(_as_real_numeric_array("program AD linalg dtype operand", item))
        elif isinstance(arg, (int, np.integer)) and not isinstance(arg, bool):
            continue
        else:
            arrays.append(_as_real_numeric_array("program AD linalg dtype operand", arg))
    if not arrays:
        return "float64"
    return str(np.result_type(*(array.dtype for array in arrays)))


def _program_ad_linalg_no_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    del args
    return ()


def _program_ad_linalg_matrix_power_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    if len(args) != 2:
        raise ValueError("program AD linalg matrix_power static rule requires matrix and power")
    power = args[1]
    if isinstance(power, bool) or not isinstance(power, (int, np.integer)):
        raise ValueError("program AD linalg matrix_power static rule requires an integer power")
    return (int(power),)


def _program_ad_linalg_trace_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 4}:
        raise ValueError(
            "program AD linalg trace static rule requires matrix and optional static axes"
        )
    shape = _program_ad_linalg_shape_of(args[0])
    if len(shape) != 2:
        raise ValueError("program AD linalg trace static rule requires a rank-2 matrix")
    offset = 0
    axis1 = 0
    axis2 = 1
    if len(args) == 4:
        offset = _program_ad_linalg_offset("trace", cast(int | np.integer, args[1]))
        axis1 = _program_ad_linalg_offset("trace", cast(int | np.integer, args[2]))
        axis2 = _program_ad_linalg_offset("trace", cast(int | np.integer, args[3]))
    if (axis1, axis2) != (0, 1):
        raise ValueError("program AD linalg trace static rule supports axis1=0 and axis2=1")
    _program_ad_linalg_trace_positions(shape, offset)
    return (shape, offset, axis1, axis2)


def _program_ad_linalg_diag_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD linalg diag static rule requires source and optional offset")
    shape = _program_ad_linalg_shape_of(args[0])
    offset = 0
    if len(args) == 2:
        offset = _program_ad_linalg_offset("diag", cast(int | np.integer, args[1]))
    if len(shape) not in {1, 2}:
        raise ValueError("program AD linalg diag static rule requires rank-1 or rank-2 input")
    _program_ad_linalg_diag_shape_from_source(shape, offset)
    return (shape, offset)


def _program_ad_linalg_diagflat_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 2}:
        raise ValueError(
            "program AD linalg diagflat static rule requires source and optional offset"
        )
    shape = _program_ad_linalg_shape_of(args[0])
    offset = 0
    if len(args) == 2:
        offset = _program_ad_linalg_offset("diagflat", cast(int | np.integer, args[1]))
    if int(np.prod(shape)) <= 0:
        raise ValueError("program AD linalg diagflat static rule requires non-empty input")
    return (shape, offset)


def _program_ad_linalg_multi_dot_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 1:
        raise ValueError("program AD linalg multi_dot static rule requires one operand sequence")
    operands = args[0]
    if isinstance(operands, (TraceADArray, np.ndarray)) or not isinstance(operands, Sequence):
        raise ValueError(
            "program AD linalg multi_dot static rule requires a static operand sequence"
        )
    shapes = tuple(_program_ad_linalg_shape_of(operand) for operand in operands)
    if len(shapes) < 2:
        raise ValueError("program AD linalg multi_dot static rule requires at least two operands")
    return shapes


def _program_ad_stencil_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    raise ValueError(
        "program AD stencil primitive contracts are executable only through "
        "operator-intercepted trace dispatch or fixed-shape derivative factories"
    )


def _program_ad_stencil_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    raise ValueError(
        "program AD stencil primitive contracts are executable only through "
        "operator-intercepted trace dispatch or fixed-shape derivative factories"
    )


def _program_ad_stencil_derivative_rule(name: str) -> CustomDerivativeRule:
    if name == "gradient":
        return CustomDerivativeRule(
            name="program_ad_stencil_gradient_trace_contract",
            value_fn=_program_ad_stencil_direct_value,
            jvp_rule=_program_ad_stencil_direct_jvp,
        )
    raise ValueError(f"unsupported program AD stencil primitive {name}")


def _program_ad_stencil_shape_of(arg: object) -> tuple[int, ...]:
    if isinstance(arg, TraceADArray):
        shape = tuple(int(dimension) for dimension in arg.shape)
    else:
        shape = tuple(
            int(dimension)
            for dimension in _as_real_numeric_array("program AD stencil source", arg).shape
        )
    if any(dimension <= 0 for dimension in shape):
        raise ValueError("program AD stencil gradient requires positive source dimensions")
    return shape


def _program_ad_stencil_spacings_arg(arg: object) -> tuple[object, ...]:
    if isinstance(arg, tuple):
        return arg
    if isinstance(arg, list):
        return tuple(arg)
    raise ValueError("program AD stencil gradient static rule requires spacing values as a tuple")


def _program_ad_stencil_gradient_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], tuple[_GradientSpacing, ...], tuple[int, ...], int]:
    if len(args) != 4:
        raise ValueError(
            "program AD stencil gradient static rule requires source, spacings, axis, and edge_order"
        )
    source_shape = _program_ad_stencil_shape_of(args[0])
    edge = _normalise_gradient_edge_order(args[3])
    axes = _normalise_gradient_axes(args[2], len(source_shape))
    spacing_specs = _normalise_gradient_spacings(
        _program_ad_stencil_spacings_arg(args[1]),
        axes,
        source_shape,
    )
    return source_shape, spacing_specs, axes, edge


def _program_ad_stencil_gradient_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape, _spacing_specs, axes, _edge = _program_ad_stencil_gradient_static_parts(args)
    if len(axes) == 1:
        return source_shape
    return (len(axes), *source_shape)


def _program_ad_stencil_dtype_rule(_args: tuple[object, ...]) -> str:
    return "float64"


def _program_ad_stencil_gradient_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    source_shape, spacing_specs, axes, edge = _program_ad_stencil_gradient_static_parts(args)
    return (
        source_shape,
        tuple(_program_ad_gradient_spacing_signature(spacing) for spacing in spacing_specs),
        axes,
        edge,
    )


def _program_ad_stencil_array_output(value: object) -> NDArray[np.float64]:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("program AD stencil batched output must not be empty")
        return np.stack(
            [
                _as_real_numeric_array("program AD stencil batched output", component)
                for component in value
            ],
            axis=0,
        )
    return _as_real_numeric_array("program AD stencil batched output", value)


def _program_ad_stencil_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 4 or len(axes) != 4:
        raise ValueError(
            "program AD stencil gradient batching requires source, spacings, axis, and edge_order"
        )
    if any(axis is not None for axis in axes[1:]):
        raise ValueError(
            "program AD stencil gradient batching keeps spacing, axis, and edge_order static"
        )
    if axes[0] is None:
        raise ValueError("program AD stencil gradient batching requires a mapped source axis")

    source = _as_real_numeric_array("program AD stencil batched source", args[0])
    batch_axis = _normalise_axis("axes[0]", axes[0], source.ndim)
    gradient_axes = _normalise_gradient_axes(args[2], source.ndim)
    if batch_axis in gradient_axes:
        raise ValueError("program AD stencil gradient cannot batch over a differentiated axis")
    adjusted_gradient_axes = tuple(
        axis_index - 1 if axis_index > batch_axis else axis_index for axis_index in gradient_axes
    )
    adjusted_axis_arg: object = (
        adjusted_gradient_axes[0] if len(adjusted_gradient_axes) == 1 else adjusted_gradient_axes
    )

    outputs = [
        _program_ad_stencil_array_output(
            function(
                np.take(source, batch_index, axis=batch_axis),
                args[1],
                adjusted_axis_arg,
                args[3],
            )
        )
        for batch_index in range(source.shape[batch_axis])
    ]
    stacked = np.stack(outputs, axis=0)
    axis_index = _normalise_axis("out_axes", out_axes, stacked.ndim)
    return np.moveaxis(stacked, 0, axis_index)


def _program_ad_stencil_lowering_metadata(name: str) -> Mapping[str, str]:
    if name != "gradient":
        raise ValueError(f"unsupported program AD stencil primitive {name}")
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff stencil dialect interchange; executable lowering blocked",
        "mlir_op": "scpn_diff.stencil.gradient",
        "llvm": "blocked_until_executable_stencil_lowering",
        "rust": "blocked_until_polyglot_stencil_ad",
        "nondifferentiable_boundary": "static_spacing_axis_edge_order",
        "nondifferentiable_boundary_policy": "fail_closed",
        "static_argument_rule": "required",
        "static_derivative_factory": "program_ad_stencil_gradient_derivative_rule",
        "static_signature": "source_shape:ranked_tensor_shape;spacing_axis_edge_order",
    }


def _register_program_ad_stencil_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_STENCIL_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_stencil_derivative_rule(name),
                batching_rule=_program_ad_stencil_batching_rule,
                lowering_metadata=_program_ad_stencil_lowering_metadata(name),
                shape_rule=_program_ad_stencil_gradient_shape,
                dtype_rule=_program_ad_stencil_dtype_rule,
                static_argument_rule=_program_ad_stencil_gradient_static_arguments,
                nondifferentiable_policy=_PROGRAM_AD_STENCIL_POLICY,
                effect="pure",
            )
        )


def _require_program_ad_stencil_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    return _require_program_ad_runtime_contract(
        name,
        family="stencil",
        identities=_PROGRAM_AD_STENCIL_IDENTITIES,
        expected_policy=_PROGRAM_AD_STENCIL_POLICY,
        args=args,
    )


_PROGRAM_AD_LINALG_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "det": _program_ad_linalg_det_shape,
    "inv": _program_ad_linalg_inv_shape,
    "solve": _program_ad_linalg_solve_shape,
    "trace": _program_ad_linalg_trace_shape,
    "diag": _program_ad_linalg_diag_shape,
    "diagflat": _program_ad_linalg_diagflat_shape,
    "matrix_power": _program_ad_linalg_matrix_power_shape,
    "multi_dot": _program_ad_linalg_multi_dot_shape,
    "eig": _program_ad_linalg_eig_shape,
    "eigh": _program_ad_linalg_eigh_shape,
    "eigvals": _program_ad_linalg_eigvals_shape,
    "eigvalsh": _program_ad_linalg_eigvalsh_shape,
    "svd": _program_ad_linalg_svd_shape,
    "pinv": _program_ad_linalg_pinv_shape,
}

_PROGRAM_AD_LINALG_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "det": _program_ad_linalg_no_static_arguments,
    "inv": _program_ad_linalg_no_static_arguments,
    "solve": _program_ad_linalg_no_static_arguments,
    "trace": _program_ad_linalg_trace_static_arguments,
    "diag": _program_ad_linalg_diag_static_arguments,
    "diagflat": _program_ad_linalg_diagflat_static_arguments,
    "matrix_power": _program_ad_linalg_matrix_power_static_arguments,
    "multi_dot": _program_ad_linalg_multi_dot_static_arguments,
    "eig": _program_ad_linalg_no_static_arguments,
    "eigh": _program_ad_linalg_no_static_arguments,
    "eigvals": _program_ad_linalg_no_static_arguments,
    "eigvalsh": _program_ad_linalg_no_static_arguments,
    "svd": _program_ad_linalg_no_static_arguments,
    "pinv": _program_ad_linalg_no_static_arguments,
}


def _program_ad_linalg_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != len(axes):
        raise ValueError("program AD linalg batching axes must match argument count")
    mapped: list[tuple[NDArray[np.float64], int] | None] = []
    batch_size: int | None = None
    for index, (arg, axis) in enumerate(zip(args, axes, strict=True)):
        if axis is None:
            mapped.append(None)
            continue
        array = _as_real_numeric_array(f"program AD linalg batched argument {index}", arg)
        axis_index = _normalise_axis(f"axes[{index}]", axis, array.ndim)
        size = int(array.shape[axis_index])
        if size <= 0:
            raise ValueError("program AD linalg batching axes must be non-empty")
        if batch_size is None:
            batch_size = size
        elif size != batch_size:
            raise ValueError("program AD linalg batching axes must share one batch size")
        mapped.append((array, axis_index))
    if batch_size is None:
        raise ValueError("program AD linalg batching requires at least one mapped axis")

    outputs: list[NDArray[np.float64]] = []
    for batch_index in range(batch_size):
        sliced_args: list[object] = []
        for original, mapped_arg in zip(args, mapped, strict=True):
            if mapped_arg is None:
                sliced_args.append(original)
                continue
            array, axis_index = mapped_arg
            sliced_args.append(np.take(array, batch_index, axis=axis_index))
        outputs.append(
            _as_real_numeric_array("program AD linalg batched output", function(*sliced_args))
        )
    stacked = np.stack(outputs, axis=0)
    axis_index = _normalise_axis("out_axes", out_axes, stacked.ndim)
    return np.moveaxis(stacked, 0, axis_index)


def _program_ad_linalg_lowering_metadata(name: str) -> Mapping[str, str]:
    nondifferentiable_boundaries = {
        "det": "singular_matrix_rank_drop",
        "inv": "singular_matrix_inverse",
        "solve": "singular_or_incompatible_linear_system",
        "trace": "static_diagonal_offset_axis_pair",
        "diag": "static_diagonal_offset_rank",
        "diagflat": "static_flattened_diagonal_offset_rank",
        "matrix_power": "negative_power_singular_matrix",
        "multi_dot": "static_shape_alignment",
        "eig": "real_simple_diagonalizable_eigensystem",
        "eigh": "symmetric_matrix_distinct_eigenvalues",
        "eigvals": "real_simple_diagonalizable_spectrum",
        "eigvalsh": "symmetric_matrix_distinct_eigenvalues",
        "svd": "distinct_positive_singular_values",
        "pinv": "rank_threshold_crossing",
    }
    metadata = {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff linalg dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.linalg.{name}",
        "llvm": "blocked_until_executable_linalg_lowering",
        "rust": "blocked_until_polyglot_linalg_ad",
        "nondifferentiable_boundary": nondifferentiable_boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
        "static_argument_rule": "none",
        "static_derivative_factory": "not_required",
        "static_signature": "none",
        "conditioning_diagnostic": "diagnose_program_ad_linalg_conditioning",
    }
    if name == "solve":
        metadata.update(
            {
                "static_derivative_factory": "program_ad_linalg_solve_derivative_rule",
                "static_signature": "matrix_shape:rank2_square;rhs_shape:rank1_or_rank2",
            }
        )
    elif name == "trace":
        metadata.update(
            {
                "static_argument_rule": "required",
                "static_derivative_factory": "program_ad_linalg_trace_derivative_rule",
                "static_signature": "matrix_shape:rank2;offset_axis_pair",
            }
        )
    elif name == "diag":
        metadata.update(
            {
                "static_argument_rule": "required",
                "static_derivative_factory": "program_ad_linalg_diag_derivative_rule",
                "static_signature": "source_shape:rank1_or_rank2;k",
            }
        )
    elif name == "diagflat":
        metadata.update(
            {
                "static_argument_rule": "required",
                "static_derivative_factory": "program_ad_linalg_diagflat_derivative_rule",
                "static_signature": "source_shape:ranked_tensor_shape;k",
            }
        )
    elif name == "matrix_power":
        metadata.update(
            {
                "static_argument_rule": "required",
                "static_derivative_factory": "program_ad_linalg_matrix_power_derivative_rule",
                "static_signature": "power:i64",
            }
        )
    elif name == "multi_dot":
        metadata.update(
            {
                "static_argument_rule": "required",
                "static_derivative_factory": "program_ad_linalg_multi_dot_derivative_rule",
                "static_signature": "operand_shapes:ranked_tensor_shape_sequence",
            }
        )
    elif name == "eig":
        metadata.update(
            {
                "static_derivative_factory": "program_ad_linalg_eig_derivative_rule",
                "static_signature": "matrix_shape:rank2_square_real_simple_eigensystem",
            }
        )
    elif name == "eigh":
        metadata.update(
            {
                "static_derivative_factory": "program_ad_linalg_eigh_derivative_rule",
                "static_signature": "matrix_shape:rank2_square_symmetric_distinct_spectrum",
            }
        )
    elif name == "eigvalsh":
        metadata.update(
            {
                "static_derivative_factory": "program_ad_linalg_eigvalsh_derivative_rule",
                "static_signature": "matrix_shape:rank2_square_symmetric_distinct_spectrum",
            }
        )
    elif name == "eigvals":
        metadata.update(
            {
                "static_derivative_factory": "program_ad_linalg_eigvals_derivative_rule",
                "static_signature": "matrix_shape:rank2_square_real_simple_spectrum",
            }
        )
    elif name == "svd":
        metadata.update(
            {
                "static_derivative_factory": "program_ad_linalg_svdvals_derivative_rule",
                "static_signature": "matrix_shape:rank2;compute_uv:false;hermitian:false",
            }
        )
    elif name == "pinv":
        metadata.update(
            {
                "static_derivative_factory": "program_ad_linalg_pinv_derivative_rule",
                "static_signature": "matrix_shape:rank2_full_rank;rcond:static_f64",
            }
        )
    return metadata


def _register_program_ad_linalg_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_LINALG_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        rule = _program_ad_linalg_derivative_rule(name)
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=rule,
                batching_rule=_program_ad_linalg_batching_rule,
                lowering_metadata=_program_ad_linalg_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_LINALG_SHAPE_RULES[name],
                dtype_rule=_program_ad_linalg_dtype_rule,
                static_argument_rule=_PROGRAM_AD_LINALG_STATIC_ARGUMENT_RULES[name],
                nondifferentiable_policy=_PROGRAM_AD_LINALG_POLICY,
                effect="pure",
            )
        )


def _require_program_ad_linalg_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    return _require_program_ad_runtime_contract(
        name,
        family="linalg",
        identities=_PROGRAM_AD_LINALG_IDENTITIES,
        expected_policy=_PROGRAM_AD_LINALG_POLICY,
        args=args,
    )


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


def registered_custom_jvp(
    identity: PrimitiveIdentity | str,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    registry: CustomDerivativeRegistry | None = None,
) -> NDArray[np.float64]:
    """Return a JVP by resolving the primitive's registered custom rule."""

    return custom_jvp(
        custom_derivative_rule_for(identity, registry=registry),
        values,
        tangent,
        parameters=parameters,
    )


def registered_custom_vjp(
    identity: PrimitiveIdentity | str,
    values: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    registry: CustomDerivativeRegistry | None = None,
) -> VJPResult:
    """Return a VJP by resolving the primitive's registered custom rule."""

    return custom_vjp(
        custom_derivative_rule_for(identity, registry=registry),
        values,
        cotangent,
        parameters=parameters,
    )


def registered_custom_jacobian(
    identity: PrimitiveIdentity | str,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    registry: CustomDerivativeRegistry | None = None,
) -> JacobianResult:
    """Return a dense Jacobian by resolving the primitive's registered custom rule."""

    return value_and_custom_jacobian(
        custom_derivative_rule_for(identity, registry=registry),
        values,
        parameters=parameters,
    )


@dataclass(frozen=True)
class LevenbergMarquardtStep:
    """Bounded Levenberg-Marquardt candidate step with model diagnostics."""

    gauss_newton: NaturalGradientResult
    step: NDArray[np.float64]
    candidate_values: NDArray[np.float64]
    damping: float
    predicted_reduction: float

    def __post_init__(self) -> None:
        step = _as_real_numeric_array("Levenberg-Marquardt step", self.step)
        candidate_values = _as_real_numeric_array(
            "Levenberg-Marquardt candidate_values",
            self.candidate_values,
        )
        if step.ndim != 1:
            raise ValueError("Levenberg-Marquardt step must be one-dimensional")
        if candidate_values.shape != step.shape:
            raise ValueError("candidate_values shape must match step shape")
        if step.shape != self.gauss_newton.base_gradient.gradient.shape:
            raise ValueError("step shape must match Gauss-Newton gradient shape")
        if not np.all(np.isfinite(step)):
            raise ValueError("Levenberg-Marquardt step must contain only finite values")
        if not np.all(np.isfinite(candidate_values)):
            raise ValueError("candidate_values must contain only finite values")
        damping = _as_real_scalar("Levenberg-Marquardt damping", self.damping)
        if damping < 0.0:
            raise ValueError("Levenberg-Marquardt damping must be finite and non-negative")
        predicted_reduction = _as_real_scalar(
            "Levenberg-Marquardt predicted_reduction",
            self.predicted_reduction,
        )
        if predicted_reduction < -1.0e-12:
            raise ValueError("predicted_reduction must be non-negative")
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "candidate_values", candidate_values)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "predicted_reduction", max(0.0, predicted_reduction))


@dataclass(frozen=True)
class LevenbergMarquardtTrial:
    """Actual-vs-predicted Levenberg-Marquardt acceptance diagnostic."""

    step_result: LevenbergMarquardtStep
    candidate_residual: NDArray[np.float64]
    candidate_value: float
    actual_reduction: float
    reduction_ratio: float
    accepted: bool

    def __post_init__(self) -> None:
        candidate_residual = _as_real_numeric_array(
            "Levenberg-Marquardt candidate_residual",
            self.candidate_residual,
        )
        if candidate_residual.ndim != 1:
            raise ValueError("candidate_residual must be one-dimensional")
        if not np.all(np.isfinite(candidate_residual)):
            raise ValueError("candidate_residual must contain only finite values")
        candidate_value = _as_real_scalar(
            "Levenberg-Marquardt candidate_value",
            self.candidate_value,
        )
        actual_reduction = _as_real_scalar(
            "Levenberg-Marquardt actual_reduction",
            self.actual_reduction,
        )
        reduction_ratio = _as_real_scalar(
            "Levenberg-Marquardt reduction_ratio",
            self.reduction_ratio,
        )
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted flag must be a boolean")
        object.__setattr__(self, "candidate_residual", candidate_residual)
        object.__setattr__(self, "candidate_value", candidate_value)
        object.__setattr__(self, "actual_reduction", actual_reduction)
        object.__setattr__(self, "reduction_ratio", reduction_ratio)


@dataclass(frozen=True)
class LevenbergMarquardtDampingUpdate:
    """Deterministic damping update for Levenberg-Marquardt trust regions."""

    trial: LevenbergMarquardtTrial
    next_damping: float
    action: str

    def __post_init__(self) -> None:
        next_damping = _as_real_scalar(
            "Levenberg-Marquardt next_damping",
            self.next_damping,
        )
        if next_damping < 0.0:
            raise ValueError("next_damping must be finite and non-negative")
        if self.action not in {"accept_decrease", "accept_keep", "reject_increase"}:
            raise ValueError("damping action must be a known Levenberg-Marquardt action")
        object.__setattr__(self, "next_damping", next_damping)


@dataclass(frozen=True)
class LevenbergMarquardtResult:
    """Traceable result from a bounded Levenberg-Marquardt optimization run."""

    values: NDArray[np.float64]
    residual: NDArray[np.float64]
    value_history: tuple[float, ...]
    damping_history: tuple[float, ...]
    accepted_history: tuple[bool, ...]
    steps: int
    converged: bool
    reason: str
    best_values: NDArray[np.float64]
    best_value: float

    def __post_init__(self) -> None:
        values = _as_parameter_array(self.values)
        residual = _as_vector_output(self.residual)
        best_values = _as_parameter_array(self.best_values)
        if best_values.shape != values.shape:
            raise ValueError("LM best values must match result values shape")
        if not self.value_history:
            raise ValueError("LM value history must contain the initial objective")
        value_history = tuple(
            _as_real_scalar("LM objective history value", value) for value in self.value_history
        )
        damping_history = tuple(
            _as_real_scalar("LM damping history value", value) for value in self.damping_history
        )
        accepted_history = tuple(bool(value) for value in self.accepted_history)
        steps = int(self.steps)
        if steps < 0:
            raise ValueError("LM result steps must be non-negative")
        if any(value < 0.0 for value in damping_history):
            raise ValueError("LM damping history must contain finite non-negative values")
        if len(accepted_history) != steps:
            raise ValueError("LM accepted history length must match executed steps")
        if len(damping_history) != steps + 1:
            raise ValueError(
                "LM damping history must include initial damping plus one entry per step"
            )
        if len(value_history) != steps + 1:
            raise ValueError("LM value history must include initial value plus one entry per step")
        best_value = _as_real_scalar("LM best objective", self.best_value)
        if best_value > min(value_history) + 1.0e-12:
            raise ValueError("LM best objective must be no larger than the recorded minimum")
        if self.reason not in {
            "residual_tolerance",
            "step_tolerance",
            "value_tolerance",
            "max_steps",
        }:
            raise ValueError("LM result reason must be a known convergence status")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "value_history", value_history)
        object.__setattr__(self, "damping_history", damping_history)
        object.__setattr__(self, "accepted_history", accepted_history)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "best_values", best_values)
        object.__setattr__(self, "best_value", best_value)


@dataclass(frozen=True)
class LeastSquaresCovarianceResult:
    """Parameter uncertainty estimate from a residual-map Fisher metric."""

    covariance: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    residual_variance: float
    degrees_of_freedom: int
    condition_number: float
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        covariance = _as_real_numeric_array("least-squares covariance", self.covariance)
        standard_errors = _as_real_numeric_array(
            "least-squares standard errors",
            self.standard_errors,
        )
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("least-squares covariance must be a square matrix")
        if standard_errors.ndim != 1 or standard_errors.shape[0] != covariance.shape[0]:
            raise ValueError("standard_errors length must match covariance dimension")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("least-squares covariance must contain only finite values")
        if not np.allclose(covariance, covariance.T, atol=1.0e-10):
            raise ValueError("least-squares covariance must be symmetric")
        if not np.all(np.isfinite(standard_errors)) or np.any(standard_errors < 0.0):
            raise ValueError("standard_errors must contain finite non-negative values")
        residual_variance = _as_real_scalar(
            "least-squares residual_variance",
            self.residual_variance,
        )
        if residual_variance < 0.0:
            raise ValueError("residual_variance must be finite and non-negative")
        degrees_of_freedom = int(self.degrees_of_freedom)
        if degrees_of_freedom < 1:
            raise ValueError("degrees_of_freedom must be positive")
        condition_number = _as_real_scalar(
            "least-squares condition_number",
            self.condition_number,
        )
        if condition_number < 1.0:
            raise ValueError("condition_number must be at least one")
        if len(self.parameter_names) != covariance.shape[0]:
            raise ValueError("parameter_names length must match covariance dimension")
        if len(self.trainable) != covariance.shape[0]:
            raise ValueError("trainable mask length must match covariance dimension")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "standard_errors", standard_errors)
        object.__setattr__(self, "residual_variance", residual_variance)
        object.__setattr__(self, "degrees_of_freedom", degrees_of_freedom)
        object.__setattr__(self, "condition_number", condition_number)


@dataclass(frozen=True)
class FisherVectorProductResult:
    """Matrix-free empirical-Fisher vector product with provenance."""

    value: NDArray[np.float64]
    tangent: NDArray[np.float64]
    product: NDArray[np.float64]
    residual_projection: NDArray[np.float64]
    damping: float
    method: str
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_numeric_array("Fisher-vector value", self.value)
        tangent = _as_real_numeric_array("Fisher-vector tangent", self.tangent)
        product = _as_real_numeric_array("Fisher-vector product", self.product)
        projection = _as_real_numeric_array(
            "Fisher-vector residual_projection",
            self.residual_projection,
        )
        if value.ndim != 1:
            raise ValueError("Fisher-vector value must be one-dimensional")
        if tangent.ndim != 1 or product.shape != tangent.shape:
            raise ValueError("Fisher-vector tangent and product must be one-dimensional matches")
        if projection.shape != value.shape:
            raise ValueError("residual_projection shape must match value shape")
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(projection)):
            raise ValueError("Fisher-vector value and projection must contain only finite values")
        if not np.all(np.isfinite(tangent)) or not np.all(np.isfinite(product)):
            raise ValueError("Fisher-vector tangent and product must contain only finite values")
        damping = _as_real_scalar("Fisher-vector damping", self.damping)
        if damping < 0.0:
            raise ValueError("Fisher-vector damping must be finite and non-negative")
        if not self.method:
            raise ValueError("Fisher-vector method must be non-empty")
        if self.evaluations < 0:
            raise ValueError("Fisher-vector evaluations must be non-negative")
        if len(self.parameter_names) != tangent.size:
            raise ValueError("parameter_names length must match Fisher-vector dimension")
        if len(self.trainable) != tangent.size:
            raise ValueError("trainable mask length must match Fisher-vector dimension")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "tangent", tangent)
        object.__setattr__(self, "product", product)
        object.__setattr__(self, "residual_projection", projection)
        object.__setattr__(self, "damping", damping)


@dataclass(frozen=True)
class FisherConjugateGradientResult:
    """Matrix-free empirical-Fisher conjugate-gradient solve result."""

    solution: NDArray[np.float64]
    residual_norm_history: tuple[float, ...]
    iterations: int
    converged: bool
    tolerance: float
    damping: float
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        solution = _as_real_numeric_array("Fisher-CG solution", self.solution)
        if solution.ndim != 1:
            raise ValueError("Fisher-CG solution must be one-dimensional")
        if not np.all(np.isfinite(solution)):
            raise ValueError("Fisher-CG solution must contain only finite values")
        if not self.residual_norm_history:
            raise ValueError("Fisher-CG residual history must be non-empty")
        residual_history = tuple(
            _as_real_scalar("Fisher-CG residual norm", value)
            for value in self.residual_norm_history
        )
        if any(value < 0.0 for value in residual_history):
            raise ValueError("Fisher-CG residual norms must be finite and non-negative")
        iterations = int(self.iterations)
        if iterations < 0:
            raise ValueError("Fisher-CG iterations must be non-negative")
        if len(residual_history) != iterations + 1:
            raise ValueError("Fisher-CG residual history must include initial residual")
        tolerance = _as_real_scalar("Fisher-CG tolerance", self.tolerance)
        damping = _as_real_scalar("Fisher-CG damping", self.damping)
        if tolerance < 0.0:
            raise ValueError("Fisher-CG tolerance must be finite and non-negative")
        if damping < 0.0:
            raise ValueError("Fisher-CG damping must be finite and non-negative")
        if len(self.parameter_names) != solution.size:
            raise ValueError("parameter_names length must match Fisher-CG dimension")
        if len(self.trainable) != solution.size:
            raise ValueError("trainable mask length must match Fisher-CG dimension")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "solution", solution)
        object.__setattr__(self, "residual_norm_history", residual_history)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "damping", damping)


@dataclass(frozen=True)
class WeightedGradientResult:
    """Weighted scalarisation of multiple scalar gradient results."""

    value: float
    gradient: NDArray[np.float64]
    components: tuple[GradientResult, ...]
    weights: NDArray[np.float64]
    method: str
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("weighted gradient components must be non-empty")
        value = _as_real_scalar("weighted gradient value", self.value)
        gradient = _as_real_numeric_array("weighted gradient", self.gradient)
        weights = _as_real_numeric_array("weighted gradient weights", self.weights)
        if gradient.ndim != 1:
            raise ValueError("weighted gradient must be a one-dimensional array")
        if weights.ndim != 1 or weights.size != len(self.components):
            raise ValueError("weights length must match weighted gradient components")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("weighted gradient must contain only finite values")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must contain only finite values")
        if not self.method:
            raise ValueError("weighted gradient method must be non-empty")
        if self.evaluations < 0:
            raise ValueError("weighted gradient evaluations must be non-negative")
        if len(self.parameter_names) != gradient.size:
            raise ValueError("parameter_names length must match gradient length")
        if len(self.trainable) != gradient.size:
            raise ValueError("trainable mask length must match gradient length")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "weights", weights)


@dataclass(frozen=True)
class DifferentiableOptimizer:
    """Small native gradient-descent optimizer for differentiable SCPN parameters."""

    learning_rate: float = 0.01

    def __post_init__(self) -> None:
        learning_rate = _as_real_scalar("learning_rate", self.learning_rate)
        if learning_rate < 0.0:
            raise ValueError("learning_rate must be finite and non-negative")
        object.__setattr__(self, "learning_rate", learning_rate)

    def step(
        self,
        values: ArrayLike,
        gradient_result: GradientResult,
        *,
        bounds: Sequence[ParameterBounds] | None = None,
        max_gradient_norm: float | None = None,
    ) -> NDArray[np.float64]:
        """Return one gradient-descent update respecting the trainable mask."""

        parameter_values = _as_parameter_array(values)
        bounds_meta = _normalise_bounds(parameter_values, bounds)
        if parameter_values.size != gradient_result.gradient.size:
            raise ValueError("values length must match gradient length")
        trainable = np.asarray(gradient_result.trainable, dtype=bool)
        if trainable.size != parameter_values.size:
            raise ValueError("trainable mask length must match values length")
        gradient = _clip_gradient(
            gradient_result.gradient,
            trainable,
            max_gradient_norm=max_gradient_norm,
        )
        updated: NDArray[np.float64] = parameter_values.copy()
        updated[trainable] -= self.learning_rate * gradient[trainable]
        return _project_bounds(updated, bounds_meta)

    def minimize(
        self,
        objective: ScalarObjective,
        initial_values: ArrayLike,
        *,
        parameters: Sequence[Parameter] | None = None,
        rule: ParameterShiftRule | None = None,
        gradient_method: str = "parameter_shift",
        finite_difference_step: float = 1.0e-6,
        bounds: Sequence[ParameterBounds] | None = None,
        max_gradient_norm: float | None = None,
        max_steps: int = 100,
        gradient_tolerance: float = 1.0e-8,
        value_tolerance: float | None = None,
    ) -> OptimizationResult:
        """Run bounded gradient descent with parameter-shift gradients."""

        if gradient_method not in {"parameter_shift", "finite_difference"}:
            raise ValueError("gradient_method must be 'parameter_shift' or 'finite_difference'")
        finite_difference_step_value = _as_real_scalar(
            "finite_difference_step", finite_difference_step
        )
        if finite_difference_step_value <= 0.0:
            raise ValueError("finite_difference_step must be finite and positive")
        _validate_max_gradient_norm(max_gradient_norm)
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer")
        gradient_tolerance_value = _as_real_scalar("gradient_tolerance", gradient_tolerance)
        if gradient_tolerance_value < 0.0:
            raise ValueError("gradient_tolerance must be finite and non-negative")
        value_tolerance_value = (
            None
            if value_tolerance is None
            else _as_real_scalar("value_tolerance", value_tolerance)
        )
        if value_tolerance_value is not None and value_tolerance_value < 0.0:
            raise ValueError("value_tolerance must be finite and non-negative")

        values = _as_parameter_array(initial_values).copy()
        bounds_meta = _normalise_bounds(values, bounds)
        values = _project_bounds(values, bounds_meta)
        history: list[float] = []
        best_values = values.copy()
        best_value = float("inf")
        previous_value: float | None = None

        for step_index in range(max_steps + 1):
            if gradient_method == "finite_difference":
                gradient_result = value_and_finite_difference_grad(
                    objective,
                    values,
                    parameters=parameters,
                    step=finite_difference_step_value,
                )
            else:
                gradient_result = value_and_parameter_shift_grad(
                    objective,
                    values,
                    parameters=parameters,
                    rule=rule,
                )
            history.append(gradient_result.value)
            if gradient_result.value < best_value:
                best_value = gradient_result.value
                best_values = values.copy()
            trainable = np.asarray(gradient_result.trainable, dtype=bool)
            gradient_norm = float(np.linalg.norm(gradient_result.gradient[trainable], ord=2))
            if gradient_norm <= gradient_tolerance_value:
                return OptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    value_history=tuple(history),
                    steps=step_index,
                    converged=True,
                    reason="gradient_tolerance",
                    best_values=best_values,
                    best_value=best_value,
                )
            if (
                value_tolerance_value is not None
                and previous_value is not None
                and abs(previous_value - gradient_result.value) <= value_tolerance_value
            ):
                return OptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    value_history=tuple(history),
                    steps=step_index,
                    converged=True,
                    reason="value_tolerance",
                    best_values=best_values,
                    best_value=best_value,
                )
            if step_index == max_steps:
                return OptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    value_history=tuple(history),
                    steps=step_index,
                    converged=False,
                    reason="max_steps",
                    best_values=best_values,
                    best_value=best_value,
                )
            previous_value = gradient_result.value
            values = self.step(
                values,
                gradient_result,
                bounds=bounds_meta,
                max_gradient_norm=max_gradient_norm,
            )

        raise RuntimeError("unreachable optimizer state")


@dataclass(frozen=True)
class LevenbergMarquardtOptimizer:
    """Bounded Levenberg-Marquardt optimizer for residual-map objectives."""

    damping: float = 1.0e-3
    max_steps: int = 100
    residual_tolerance: float = 1.0e-8
    step_tolerance: float = 1.0e-8
    value_tolerance: float | None = None
    acceptance_threshold: float = 1.0e-4
    decrease_factor: float = 1.0 / 3.0
    increase_factor: float = 2.0
    min_damping: float = 1.0e-12
    max_damping: float = 1.0e12
    high_quality_ratio: float = 0.75
    finite_difference_step: float = 1.0e-6
    max_step_norm: float | None = None

    def __post_init__(self) -> None:
        damping = _as_real_scalar("Levenberg-Marquardt damping", self.damping)
        if damping < 0.0:
            raise ValueError("Levenberg-Marquardt damping must be finite and non-negative")
        max_steps = int(self.max_steps)
        if max_steps < 1:
            raise ValueError("Levenberg-Marquardt max_steps must be positive")
        residual_tolerance = _as_real_scalar(
            "Levenberg-Marquardt residual_tolerance",
            self.residual_tolerance,
        )
        step_tolerance = _as_real_scalar(
            "Levenberg-Marquardt step_tolerance",
            self.step_tolerance,
        )
        if residual_tolerance < 0.0 or step_tolerance < 0.0:
            raise ValueError("Levenberg-Marquardt tolerances must be finite and non-negative")
        value_tolerance = (
            None
            if self.value_tolerance is None
            else _as_real_scalar("Levenberg-Marquardt value_tolerance", self.value_tolerance)
        )
        if value_tolerance is not None and value_tolerance < 0.0:
            raise ValueError("Levenberg-Marquardt value_tolerance must be finite and non-negative")
        acceptance_threshold = _as_real_scalar(
            "Levenberg-Marquardt acceptance_threshold",
            self.acceptance_threshold,
        )
        if acceptance_threshold < 0.0:
            raise ValueError("Levenberg-Marquardt acceptance_threshold must be non-negative")
        decrease_factor = _as_real_scalar(
            "Levenberg-Marquardt decrease_factor",
            self.decrease_factor,
        )
        increase_factor = _as_real_scalar(
            "Levenberg-Marquardt increase_factor",
            self.increase_factor,
        )
        min_damping = _as_real_scalar("Levenberg-Marquardt min_damping", self.min_damping)
        max_damping = _as_real_scalar("Levenberg-Marquardt max_damping", self.max_damping)
        high_quality_ratio = _as_real_scalar(
            "Levenberg-Marquardt high_quality_ratio",
            self.high_quality_ratio,
        )
        finite_difference_step = _as_real_scalar(
            "Levenberg-Marquardt finite_difference_step",
            self.finite_difference_step,
        )
        if not 0.0 < decrease_factor < 1.0:
            raise ValueError("decrease_factor must be finite and between 0 and 1")
        if increase_factor <= 1.0:
            raise ValueError("increase_factor must be finite and greater than 1")
        if min_damping < 0.0 or max_damping < min_damping:
            raise ValueError("LM damping bounds must be finite and ordered")
        if high_quality_ratio < 0.0:
            raise ValueError("high_quality_ratio must be finite and non-negative")
        if finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be finite and positive")
        max_step_norm = (
            None
            if self.max_step_norm is None
            else _as_real_scalar("Levenberg-Marquardt max_step_norm", self.max_step_norm)
        )
        if max_step_norm is not None and max_step_norm <= 0.0:
            raise ValueError("max_step_norm must be finite and positive")
        object.__setattr__(self, "damping", min(max_damping, max(min_damping, damping)))
        object.__setattr__(self, "max_steps", max_steps)
        object.__setattr__(self, "residual_tolerance", residual_tolerance)
        object.__setattr__(self, "step_tolerance", step_tolerance)
        object.__setattr__(self, "value_tolerance", value_tolerance)
        object.__setattr__(self, "acceptance_threshold", acceptance_threshold)
        object.__setattr__(self, "decrease_factor", decrease_factor)
        object.__setattr__(self, "increase_factor", increase_factor)
        object.__setattr__(self, "min_damping", min_damping)
        object.__setattr__(self, "max_damping", max_damping)
        object.__setattr__(self, "high_quality_ratio", high_quality_ratio)
        object.__setattr__(self, "finite_difference_step", finite_difference_step)
        object.__setattr__(self, "max_step_norm", max_step_norm)

    def minimize(
        self,
        objective: VectorObjective,
        initial_values: ArrayLike,
        *,
        parameters: Sequence[Parameter] | None = None,
        bounds: Sequence[ParameterBounds] | None = None,
        weight_fn: Callable[[NDArray[np.float64]], ArrayLike] | None = None,
        rcond: float = 1.0e-12,
    ) -> LevenbergMarquardtResult:
        """Minimize a vector residual objective with adaptive bounded LM steps."""

        values = _as_parameter_array(initial_values)
        bounds_meta = _normalise_bounds(values, bounds)
        values = _project_bounds(values, bounds_meta)
        damping = self.damping
        jacobian_result = value_and_finite_difference_jacobian(
            objective,
            values,
            parameters=parameters,
            step=self.finite_difference_step,
        )
        weights = self._weights_for(jacobian_result.value, weight_fn)
        current_value = self._weighted_value(jacobian_result.value, weights)
        current_residual = jacobian_result.value
        best_values = values.copy()
        best_value = current_value
        value_history: list[float] = [current_value]
        damping_history: list[float] = [damping]
        accepted_history: list[bool] = []
        reason = "max_steps"
        converged = False

        if float(np.linalg.norm(current_residual, ord=2)) <= self.residual_tolerance:
            return LevenbergMarquardtResult(
                values=values,
                residual=current_residual,
                value_history=tuple(value_history),
                damping_history=tuple(damping_history),
                accepted_history=(),
                steps=0,
                converged=True,
                reason="residual_tolerance",
                best_values=best_values,
                best_value=best_value,
            )

        for _ in range(self.max_steps):
            step_result = levenberg_marquardt_step(
                jacobian_result,
                values,
                weights=weights,
                damping=damping,
                bounds=bounds_meta,
                max_step_norm=self.max_step_norm,
                rcond=rcond,
            )
            trial = evaluate_levenberg_marquardt_step(
                objective,
                step_result,
                weights=weights,
                acceptance_threshold=self.acceptance_threshold,
            )
            update = update_levenberg_marquardt_damping(
                trial,
                decrease_factor=self.decrease_factor,
                increase_factor=self.increase_factor,
                min_damping=self.min_damping,
                max_damping=self.max_damping,
                high_quality_ratio=self.high_quality_ratio,
            )
            accepted_history.append(trial.accepted)
            trainable = np.asarray(jacobian_result.trainable, dtype=bool)
            step_norm = float(np.linalg.norm(step_result.step[trainable], ord=2))
            if trial.accepted:
                values = step_result.candidate_values
                current_residual = trial.candidate_residual
                current_value = trial.candidate_value
                if current_value < best_value:
                    best_value = current_value
                    best_values = values.copy()
                if float(np.linalg.norm(current_residual, ord=2)) <= self.residual_tolerance:
                    reason = "residual_tolerance"
                    converged = True
                elif step_norm <= self.step_tolerance:
                    reason = "step_tolerance"
                    converged = True
                elif (
                    self.value_tolerance is not None
                    and abs(trial.actual_reduction) <= self.value_tolerance
                ):
                    reason = "value_tolerance"
                    converged = True
            damping = update.next_damping
            value_history.append(current_value)
            damping_history.append(damping)
            if converged:
                break
            if trial.accepted:
                jacobian_result = value_and_finite_difference_jacobian(
                    objective,
                    values,
                    parameters=parameters,
                    step=self.finite_difference_step,
                )
                weights = self._weights_for(jacobian_result.value, weight_fn)

        return LevenbergMarquardtResult(
            values=values,
            residual=current_residual,
            value_history=tuple(value_history),
            damping_history=tuple(damping_history),
            accepted_history=tuple(accepted_history),
            steps=len(accepted_history),
            converged=converged,
            reason=reason,
            best_values=best_values,
            best_value=best_value,
        )

    @staticmethod
    def _weighted_value(
        residual: NDArray[np.float64],
        weights: NDArray[np.float64] | None,
    ) -> float:
        if weights is None:
            return 0.5 * float(residual @ residual)
        return 0.5 * float(residual @ (residual * weights))

    @staticmethod
    def _weights_for(
        residual: NDArray[np.float64],
        weight_fn: Callable[[NDArray[np.float64]], ArrayLike] | None,
    ) -> NDArray[np.float64] | None:
        if weight_fn is None:
            return None
        weights = _as_real_numeric_array("LM weights", weight_fn(residual.copy()))
        if weights.ndim != 1 or weights.shape[0] != residual.size:
            raise ValueError("LM weights must be a one-dimensional array matching residual rows")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("LM weights must contain only finite non-negative values")
        return weights


@dataclass(frozen=True)
class NaturalGradientOptimizer:
    """Bounded natural-gradient optimizer for scalar objectives with explicit metrics."""

    learning_rate: float = 0.01
    damping: float = 0.0
    rcond: float = 1.0e-12
    max_step_norm: float | None = None

    def __post_init__(self) -> None:
        learning_rate = _as_real_scalar("natural-gradient learning_rate", self.learning_rate)
        damping = _as_real_scalar("natural-gradient damping", self.damping)
        rcond = _as_real_scalar("natural-gradient rcond", self.rcond)
        if learning_rate < 0.0:
            raise ValueError("natural-gradient learning_rate must be finite and non-negative")
        if damping < 0.0:
            raise ValueError("natural-gradient damping must be finite and non-negative")
        if rcond <= 0.0:
            raise ValueError("natural-gradient rcond must be finite and positive")
        max_step_norm = (
            None
            if self.max_step_norm is None
            else _as_real_scalar("natural-gradient max_step_norm", self.max_step_norm)
        )
        if max_step_norm is not None and max_step_norm <= 0.0:
            raise ValueError("natural-gradient max_step_norm must be finite and positive")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "rcond", rcond)
        object.__setattr__(self, "max_step_norm", max_step_norm)

    def minimize(
        self,
        objective: ScalarObjective,
        initial_values: ArrayLike,
        metric_fn: Callable[[GradientResult, NDArray[np.float64]], ArrayLike],
        *,
        parameters: Sequence[Parameter] | None = None,
        rule: ParameterShiftRule | None = None,
        gradient_method: str = "parameter_shift",
        finite_difference_step: float = 1.0e-6,
        bounds: Sequence[ParameterBounds] | None = None,
        max_steps: int = 100,
        gradient_tolerance: float = 1.0e-8,
        step_tolerance: float = 1.0e-8,
        value_tolerance: float | None = None,
    ) -> NaturalGradientOptimizationResult:
        """Run a bounded natural-gradient descent loop with metric provenance."""

        if gradient_method not in {"parameter_shift", "finite_difference"}:
            raise ValueError("gradient_method must be 'parameter_shift' or 'finite_difference'")
        finite_difference_step_value = _as_real_scalar(
            "finite_difference_step", finite_difference_step
        )
        if finite_difference_step_value <= 0.0:
            raise ValueError("finite_difference_step must be finite and positive")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer")
        gradient_tolerance_value = _as_real_scalar("gradient_tolerance", gradient_tolerance)
        step_tolerance_value = _as_real_scalar("step_tolerance", step_tolerance)
        if gradient_tolerance_value < 0.0 or step_tolerance_value < 0.0:
            raise ValueError("natural-gradient tolerances must be finite and non-negative")
        value_tolerance_value = (
            None
            if value_tolerance is None
            else _as_real_scalar("value_tolerance", value_tolerance)
        )
        if value_tolerance_value is not None and value_tolerance_value < 0.0:
            raise ValueError("value_tolerance must be finite and non-negative")

        values = _as_parameter_array(initial_values).copy()
        bounds_meta = _normalise_bounds(values, bounds)
        values = _project_bounds(values, bounds_meta)
        value_history: list[float] = []
        gradient_norm_history: list[float] = []
        step_norm_history: list[float] = []
        best_values = values.copy()
        best_value = float("inf")
        previous_value: float | None = None

        for step_index in range(max_steps + 1):
            gradient_result = self._gradient(
                objective,
                values,
                parameters=parameters,
                rule=rule,
                gradient_method=gradient_method,
                finite_difference_step=finite_difference_step_value,
            )
            metric = metric_fn(gradient_result, values.copy())
            natural_result = natural_gradient(
                gradient_result,
                metric,
                damping=self.damping,
                rcond=self.rcond,
            )
            trainable = np.asarray(gradient_result.trainable, dtype=bool)
            gradient_norm = float(np.linalg.norm(gradient_result.gradient[trainable], ord=2))
            value_history.append(gradient_result.value)
            gradient_norm_history.append(gradient_norm)
            if gradient_result.value < best_value:
                best_value = gradient_result.value
                best_values = values.copy()
            if gradient_norm <= gradient_tolerance_value:
                return NaturalGradientOptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    final_natural_gradient=natural_result,
                    value_history=tuple(value_history),
                    gradient_norm_history=tuple(gradient_norm_history),
                    natural_step_norm_history=tuple(step_norm_history),
                    steps=step_index,
                    converged=True,
                    reason="gradient_tolerance",
                    best_values=best_values,
                    best_value=best_value,
                )
            if (
                value_tolerance_value is not None
                and previous_value is not None
                and abs(previous_value - gradient_result.value) <= value_tolerance_value
            ):
                return NaturalGradientOptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    final_natural_gradient=natural_result,
                    value_history=tuple(value_history),
                    gradient_norm_history=tuple(gradient_norm_history),
                    natural_step_norm_history=tuple(step_norm_history),
                    steps=step_index,
                    converged=True,
                    reason="value_tolerance",
                    best_values=best_values,
                    best_value=best_value,
                )
            if step_index == max_steps:
                return NaturalGradientOptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    final_natural_gradient=natural_result,
                    value_history=tuple(value_history),
                    gradient_norm_history=tuple(gradient_norm_history),
                    natural_step_norm_history=tuple(step_norm_history),
                    steps=step_index,
                    converged=False,
                    reason="max_steps",
                    best_values=best_values,
                    best_value=best_value,
                )
            step_vector = self._bounded_step(natural_result.natural_gradient, trainable)
            step_norm = float(np.linalg.norm(step_vector[trainable], ord=2))
            if step_norm <= step_tolerance_value:
                return NaturalGradientOptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    final_natural_gradient=natural_result,
                    value_history=tuple(value_history),
                    gradient_norm_history=tuple(gradient_norm_history),
                    natural_step_norm_history=tuple(step_norm_history),
                    steps=step_index,
                    converged=True,
                    reason="step_tolerance",
                    best_values=best_values,
                    best_value=best_value,
                )
            step_norm_history.append(step_norm)
            previous_value = gradient_result.value
            values = _project_bounds(values - step_vector, bounds_meta)

        raise RuntimeError("unreachable natural-gradient optimizer state")

    def _bounded_step(
        self,
        natural_gradient_value: NDArray[np.float64],
        trainable: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        step_vector = self.learning_rate * natural_gradient_value.copy()
        if self.max_step_norm is not None and np.any(trainable):
            norm = float(np.linalg.norm(step_vector[trainable], ord=2))
            if norm > self.max_step_norm:
                step_vector[trainable] *= self.max_step_norm / norm
        step_vector[~trainable] = 0.0
        typed_step: NDArray[np.float64] = step_vector
        return typed_step

    @staticmethod
    def _gradient(
        objective: ScalarObjective,
        values: NDArray[np.float64],
        *,
        parameters: Sequence[Parameter] | None,
        rule: ParameterShiftRule | None,
        gradient_method: str,
        finite_difference_step: float,
    ) -> GradientResult:
        if gradient_method == "finite_difference":
            return value_and_finite_difference_grad(
                objective,
                values,
                parameters=parameters,
                step=finite_difference_step,
            )
        return value_and_parameter_shift_grad(
            objective,
            values,
            parameters=parameters,
            rule=rule,
        )


def armijo_backtracking_line_search(
    objective: ScalarObjective,
    values: ArrayLike,
    gradient_result: GradientResult,
    direction: ArrayLike,
    *,
    bounds: Sequence[ParameterBounds] | None = None,
    initial_step: float = 1.0,
    contraction: float = 0.5,
    sufficient_decrease: float = 1.0e-4,
    max_steps: int = 20,
) -> ArmijoLineSearchResult:
    """Return a bounded Armijo backtracking step for a scalar objective."""

    if not isinstance(gradient_result, GradientResult):
        raise ValueError("line search requires a GradientResult")
    parameter_values = _as_parameter_array(values)
    if parameter_values.size != gradient_result.gradient.size:
        raise ValueError("line-search values length must match gradient length")
    direction_values = _as_parameter_array(direction)
    if direction_values.shape != parameter_values.shape:
        raise ValueError("line-search direction length must match values length")
    initial_step_value = _as_real_scalar("line-search initial_step", initial_step)
    contraction_value = _as_real_scalar("line-search contraction", contraction)
    sufficient_decrease_value = _as_real_scalar(
        "line-search sufficient_decrease",
        sufficient_decrease,
    )
    if initial_step_value <= 0.0:
        raise ValueError("line-search initial_step must be finite and positive")
    if not 0.0 < contraction_value < 1.0:
        raise ValueError("line-search contraction must be finite and between 0 and 1")
    if not 0.0 < sufficient_decrease_value < 1.0:
        raise ValueError("line-search sufficient_decrease must be finite and between 0 and 1")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("line-search max_steps must be a positive integer")
    trainable = np.asarray(gradient_result.trainable, dtype=bool)
    bounds_meta = _normalise_bounds(parameter_values, bounds)
    masked_direction = direction_values.copy()
    masked_direction[~trainable] = 0.0
    directional_derivative = float(gradient_result.gradient @ masked_direction)
    start_value = _as_scalar(objective(parameter_values.copy()))
    evaluations = 1
    history: list[float] = [start_value]
    if directional_derivative >= 0.0 or not np.any(masked_direction[trainable]):
        return ArmijoLineSearchResult(
            values=parameter_values,
            value=start_value,
            step_size=0.0,
            direction=masked_direction,
            directional_derivative=directional_derivative,
            accepted=False,
            evaluations=evaluations,
            value_history=tuple(history),
            reason="non_descent_direction",
            parameter_names=gradient_result.parameter_names,
            trainable=gradient_result.trainable,
        )
    step_size = initial_step_value
    for _ in range(max_steps):
        candidate = _project_bounds(parameter_values + step_size * masked_direction, bounds_meta)
        actual_step = candidate - parameter_values
        actual_derivative = float(gradient_result.gradient @ actual_step)
        candidate_value = _as_scalar(objective(candidate.copy()))
        evaluations += 1
        history.append(candidate_value)
        if candidate_value <= start_value + sufficient_decrease_value * actual_derivative:
            return ArmijoLineSearchResult(
                values=candidate,
                value=candidate_value,
                step_size=step_size,
                direction=masked_direction,
                directional_derivative=directional_derivative,
                accepted=True,
                evaluations=evaluations,
                value_history=tuple(history),
                reason="accepted",
                parameter_names=gradient_result.parameter_names,
                trainable=gradient_result.trainable,
            )
        step_size *= contraction_value
    return ArmijoLineSearchResult(
        values=parameter_values,
        value=start_value,
        step_size=0.0,
        direction=masked_direction,
        directional_derivative=directional_derivative,
        accepted=False,
        evaluations=evaluations,
        value_history=tuple(history),
        reason="max_steps",
        parameter_names=gradient_result.parameter_names,
        trainable=gradient_result.trainable,
    )


def _as_parameter_shift_sample_tensor(
    name: str,
    values: ArrayLike,
    *,
    term_count: int,
) -> NDArray[np.float64]:
    array = _as_real_numeric_array(name, values)
    if term_count == 1 and array.ndim == 1:
        array = array.reshape(1, array.size)
    elif array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_terms, n_parameters)")
    if array.shape[0] != term_count:
        raise ValueError(f"{name} first dimension must match parameter-shift terms")
    if array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one parameter column")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_batch_parameter_array(
    name: str,
    values: ArrayLike,
    parameter_count: int,
) -> NDArray[np.float64]:
    array = _as_real_numeric_array(name, values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional batch")
    if array.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one row")
    if array.shape[1] != parameter_count:
        raise ValueError(f"{name} row length must match parameter length")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_batch_vector_array(
    name: str,
    values: ArrayLike,
    vector_count: int,
) -> NDArray[np.float64]:
    array = _as_real_numeric_array(name, values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional batch")
    if array.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one row")
    if array.shape[1] != vector_count:
        raise ValueError(f"{name} row length must match vector length")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_scalar(value: float | int | np.floating[Any] | NDArray[np.float64]) -> float:
    try:
        scalar = _as_real_scalar("differentiable objective", value)
    except ValueError as exc:
        if "scalar" in str(exc):
            raise ValueError("differentiable objective must return a scalar") from exc
        raise
    if not np.isfinite(scalar):
        raise ValueError("differentiable objective returned a non-finite scalar")
    return scalar


def _as_forward_mode_scalar(value: object) -> DualNumber:
    """Return a scalar dual objective value."""

    if isinstance(value, DualNumber):
        return value
    try:
        return DualNumber(_as_real_scalar("forward-mode objective", value), 0.0)
    except ValueError as exc:
        if "scalar" in str(exc):
            raise ValueError("forward-mode objective must return a scalar") from exc
        raise


def _as_reverse_mode_scalar(value: object) -> ReverseNode:
    """Return a scalar reverse-mode objective value."""

    if isinstance(value, ReverseNode):
        return value
    try:
        return ReverseNode(_as_real_scalar("reverse-mode objective", value))
    except ValueError as exc:
        if "scalar" in str(exc):
            raise ValueError("reverse-mode objective must return a scalar") from exc
        raise


def _reverse_topological_order(root: ReverseNode) -> tuple[ReverseNode, ...]:
    """Return reverse-mode tape nodes in parent-before-child order."""

    ordered: list[ReverseNode] = []
    seen: set[int] = set()

    def visit(node: ReverseNode) -> None:
        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)
        for parent, _local_derivative in node.parents:
            visit(parent)
        ordered.append(node)

    visit(root)
    return tuple(ordered)


def _as_complex_step_scalar(value: object) -> complex:
    """Return a scalar objective value that may carry a complex-step signal."""

    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U"}:
        raise ValueError("complex-step objective must return a scalar")
    try:
        scalar = complex(raw.item())
    except (TypeError, ValueError) as exc:
        raise ValueError("complex-step objective must return a numeric scalar") from exc
    if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag):
        raise ValueError("complex-step objective returned a non-finite scalar")
    return scalar


def _as_vector_output(value: ArrayLike) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("differentiable vector objective", value)
    if vector.ndim != 1:
        raise ValueError("differentiable vector objective must return a one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError("differentiable vector objective returned non-finite values")
    return vector


def _normalise_parameters(
    values: NDArray[np.float64],
    parameters: Sequence[Parameter] | None,
) -> tuple[Parameter, ...]:
    if parameters is None:
        return tuple(Parameter(f"theta_{index}") for index in range(values.size))
    normalised = tuple(parameters)
    if len(normalised) != values.size:
        raise ValueError("parameters length must match values length")
    if len({parameter.name for parameter in normalised}) != len(normalised):
        raise ValueError("parameter names must be unique")
    return normalised


def _normalise_bounds(
    values: NDArray[np.float64],
    bounds: Sequence[ParameterBounds] | None,
) -> tuple[ParameterBounds, ...]:
    if bounds is None:
        return tuple(ParameterBounds() for _ in range(values.size))
    normalised = tuple(bounds)
    if len(normalised) != values.size:
        raise ValueError("bounds length must match values length")
    if any(not isinstance(item, ParameterBounds) for item in normalised):
        raise ValueError("bounds must contain ParameterBounds instances")
    return normalised


def _project_bounds(
    values: NDArray[np.float64],
    bounds: Sequence[ParameterBounds],
) -> NDArray[np.float64]:
    projected = values.copy()
    for index, bound in enumerate(bounds):
        if bound.periodic:
            lower = cast(float, bound.lower)
            upper = cast(float, bound.upper)
            width = upper - lower
            projected[index] = ((projected[index] - lower) % width) + lower
            continue
        if bound.lower is not None and projected[index] < bound.lower:
            projected[index] = bound.lower
        if bound.upper is not None and projected[index] > bound.upper:
            projected[index] = bound.upper
    typed_projected: NDArray[np.float64] = projected
    return typed_projected


def _validate_max_gradient_norm(max_gradient_norm: float | None) -> float | None:
    if max_gradient_norm is None:
        return None
    max_norm = _as_real_scalar("max_gradient_norm", max_gradient_norm)
    if max_norm <= 0.0:
        raise ValueError("max_gradient_norm must be finite and positive")
    return max_norm


def _clip_gradient(
    gradient: NDArray[np.float64],
    trainable: NDArray[np.bool_],
    *,
    max_gradient_norm: float | None,
) -> NDArray[np.float64]:
    max_norm = _validate_max_gradient_norm(max_gradient_norm)
    clipped = gradient.copy()
    if max_norm is None or not np.any(trainable):
        typed_clipped: NDArray[np.float64] = clipped
        return typed_clipped
    trainable_norm = float(np.linalg.norm(clipped[trainable], ord=2))
    if trainable_norm > max_norm:
        clipped[trainable] *= max_norm / trainable_norm
    clipped_gradient: NDArray[np.float64] = clipped
    return clipped_gradient


def parameter_shift_gradient(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> NDArray[np.float64]:
    """Return the parameter-shift gradient of a scalar objective."""

    result = value_and_parameter_shift_grad(
        objective,
        values,
        parameters=parameters,
        rule=rule,
    )
    return result.gradient


def batch_parameter_shift_gradient(
    objectives: Sequence[ScalarObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> NDArray[np.float64]:
    """Return stacked parameter-shift gradients for multiple scalar objectives."""

    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    rows = [
        parameter_shift_gradient(
            objective,
            values,
            parameters=parameters,
            rule=rule,
        )
        for objective in objectives
    ]
    return np.vstack(rows)


def batch_value_and_parameter_shift_grad(
    objectives: Sequence[ScalarObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> tuple[GradientResult, ...]:
    """Return full parameter-shift results for multiple scalar objectives."""

    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    return tuple(
        value_and_parameter_shift_grad(
            objective,
            values,
            parameters=parameters,
            rule=rule,
        )
        for objective in objectives
    )


def value_and_parameter_shift_grad(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> GradientResult:
    """Evaluate a scalar objective and its native parameter-shift gradient."""

    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    shift_rule = rule or ParameterShiftRule()
    terms = shift_rule.terms
    gradient = np.zeros_like(parameter_values)
    base_value = _as_scalar(objective(parameter_values.copy()))
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        for shift, coefficient in terms:
            plus = parameter_values.copy()
            minus = parameter_values.copy()
            plus[index] += shift
            minus[index] -= shift
            plus_value = _as_scalar(objective(plus))
            minus_value = _as_scalar(objective(minus))
            evaluations += 2
            gradient[index] += coefficient * (plus_value - minus_value)

    return GradientResult(
        value=base_value,
        gradient=gradient,
        method="parameter_shift"
        if shift_rule.is_single_term
        else "multi_frequency_parameter_shift",
        shift=shift_rule.shift if shift_rule.is_single_term else None,
        coefficient=shift_rule.coefficient if shift_rule.is_single_term else None,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def parameter_shift_gradient_with_uncertainty(
    plus_values: ArrayLike,
    minus_values: ArrayLike,
    plus_variances: ArrayLike,
    minus_variances: ArrayLike,
    plus_shots: ArrayLike,
    minus_shots: ArrayLike | None = None,
    *,
    value: float = 0.0,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    confidence_level: float = 0.95,
    confidence_z: float = 1.959963984540054,
    failure_policy: GradientFailurePolicy | None = None,
) -> StochasticGradientResult:
    """Propagate independent shot noise through parameter-shift gradients."""

    shift_rule = rule or ParameterShiftRule()
    terms = shift_rule.terms
    term_count = len(terms)
    plus = _as_parameter_shift_sample_tensor(
        "plus_values",
        plus_values,
        term_count=term_count,
    )
    minus = _as_parameter_shift_sample_tensor(
        "minus_values",
        minus_values,
        term_count=term_count,
    )
    plus_var = _as_parameter_shift_sample_tensor(
        "plus_variances",
        plus_variances,
        term_count=term_count,
    )
    minus_var = _as_parameter_shift_sample_tensor(
        "minus_variances",
        minus_variances,
        term_count=term_count,
    )
    plus_count = _as_parameter_shift_sample_tensor(
        "plus_shots",
        plus_shots,
        term_count=term_count,
    )
    minus_count = (
        plus_count.copy()
        if minus_shots is None
        else _as_parameter_shift_sample_tensor(
            "minus_shots",
            minus_shots,
            term_count=term_count,
        )
    )
    if minus.shape != plus.shape:
        raise ValueError("minus_values shape must match plus_values shape")
    if plus_var.shape != plus.shape or minus_var.shape != plus.shape:
        raise ValueError("variance shapes must match plus_values shape")
    if plus_count.shape != plus.shape or minus_count.shape != plus.shape:
        raise ValueError("shot-count shapes must match plus_values shape")
    if np.any(plus_var < 0.0) or np.any(minus_var < 0.0):
        raise ValueError("shot variances must be finite non-negative values")
    if (
        not np.all(plus_count > 0.0)
        or not np.all(minus_count > 0.0)
        or not np.allclose(plus_count, np.round(plus_count))
        or not np.allclose(minus_count, np.round(minus_count))
    ):
        raise ValueError("shot counts must contain positive integers")
    confidence = _as_real_scalar("confidence_level", confidence_level)
    z_value = _as_real_scalar("confidence_z", confidence_z)
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if z_value <= 0.0:
        raise ValueError("confidence_z must be finite and positive")

    parameter_meta = _normalise_parameters(plus[0], parameters)
    gradient = np.zeros(plus.shape[1], dtype=np.float64)
    variance = np.zeros(plus.shape[1], dtype=np.float64)
    records: list[ParameterShiftSampleRecord] = []
    for index, parameter in enumerate(parameter_meta):
        for term_index, (_shift, coefficient) in enumerate(terms):
            gradient_contribution = coefficient * (
                plus[term_index, index] - minus[term_index, index]
            )
            variance_contribution = coefficient**2 * (
                plus_var[term_index, index] / plus_count[term_index, index]
                + minus_var[term_index, index] / minus_count[term_index, index]
            )
            if parameter.trainable:
                gradient[index] += gradient_contribution
                variance[index] += variance_contribution
            records.append(
                ParameterShiftSampleRecord(
                    term_index=term_index,
                    parameter_index=index,
                    parameter_name=parameter.name,
                    trainable=parameter.trainable,
                    shift=_shift,
                    coefficient=coefficient,
                    plus_value=float(plus[term_index, index]),
                    minus_value=float(minus[term_index, index]),
                    plus_variance=float(plus_var[term_index, index]),
                    minus_variance=float(minus_var[term_index, index]),
                    plus_shots=int(plus_count[term_index, index]),
                    minus_shots=int(minus_count[term_index, index]),
                    gradient_contribution=float(
                        gradient_contribution if parameter.trainable else 0.0
                    ),
                    variance_contribution=float(
                        variance_contribution if parameter.trainable else 0.0
                    ),
                )
            )
    standard_error = np.sqrt(variance)
    covariance = np.diag(variance)
    confidence_interval = gradient_confidence_interval(
        gradient,
        standard_error,
        confidence_z=z_value,
        confidence_level=confidence,
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        failure_policy=failure_policy,
    )
    shots = (
        np.vstack([plus_count[0], minus_count[0]])
        if shift_rule.is_single_term
        else np.stack([plus_count, minus_count], axis=1)
    )
    return StochasticGradientResult(
        value=value,
        gradient=gradient,
        standard_error=standard_error,
        covariance=covariance,
        confidence_radius=z_value * standard_error,
        shots=shots,
        confidence_level=confidence,
        method="parameter_shift_shot_noise"
        if shift_rule.is_single_term
        else "multi_frequency_parameter_shift_shot_noise",
        shift=shift_rule.shift if shift_rule.is_single_term else None,
        coefficient=shift_rule.coefficient if shift_rule.is_single_term else None,
        evaluations=2 * term_count * sum(parameter.trainable for parameter in parameter_meta),
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        records=tuple(records),
        claim_boundary=STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY,
        hardware_execution=False,
        confidence_interval=confidence_interval,
        failure_policy_status=confidence_interval.status,
        failure_reasons=confidence_interval.failure_reasons,
    )


def _as_spsa_sample(value: object, *, shots: int | None) -> SPSAObjectiveSample:
    if isinstance(value, SPSAObjectiveSample):
        if shots is not None and value.variance is None:
            raise ValueError("SPSA finite-shot samples must include variance")
        if shots is not None and value.shots is None:
            return SPSAObjectiveSample(
                value=value.value,
                variance=value.variance,
                shots=shots,
                metadata=value.metadata,
            )
        return value
    if shots is not None:
        raise ValueError("SPSA finite-shot objective must return SPSAObjectiveSample")
    return SPSAObjectiveSample(value=_as_real_scalar("SPSA objective value", value))


def _call_spsa_objective(
    objective: Callable[..., object],
    values: NDArray[np.float64],
    shots: int | None,
) -> SPSAObjectiveSample:
    sample = objective(values.copy()) if shots is None else objective(values.copy(), shots)
    return _as_spsa_sample(sample, shots=shots)


def spsa_gradient_estimate(
    objective: Callable[..., object],
    values: ArrayLike,
    *,
    perturbation_radius: float = 0.1,
    repetitions: int = 1,
    seed: int = 0,
    shots: int | None = None,
    parameters: Sequence[Parameter] | None = None,
    confidence_z: float = 1.959963984540054,
    failure_policy: GradientFailurePolicy | None = None,
) -> SPSAGradientResult:
    """Estimate a scalar-objective gradient with seeded SPSA perturbations."""

    parameter_values = _as_parameter_array(values)
    radius = _as_real_scalar("SPSA perturbation_radius", perturbation_radius)
    if radius <= 0.0:
        raise ValueError("SPSA perturbation_radius must be finite and positive")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("SPSA repetitions must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("SPSA seed must be a non-negative integer")
    if shots is not None and (isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0):
        raise ValueError("SPSA shots must be a positive integer or None")
    z_value = _as_real_scalar("SPSA confidence_z", confidence_z)
    if z_value <= 0.0:
        raise ValueError("SPSA confidence_z must be finite and positive")

    parameter_meta = _normalise_parameters(parameter_values, parameters)
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    if not np.any(trainable):
        raise ValueError("SPSA requires at least one trainable parameter")

    rng = np.random.default_rng(seed)
    gradient_estimates = np.zeros((repetitions, parameter_values.size), dtype=np.float64)
    shot_variance = np.zeros(parameter_values.size, dtype=np.float64)
    records: list[SPSAProbeRecord] = []
    total_shots = 0

    for repetition in range(repetitions):
        raw_delta = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=parameter_values.size)
        perturbation = np.where(trainable, raw_delta, 0.0).astype(np.float64)
        plus_parameters = parameter_values + radius * perturbation
        minus_parameters = parameter_values - radius * perturbation
        plus = _call_spsa_objective(objective, plus_parameters, shots)
        minus = _call_spsa_objective(objective, minus_parameters, shots)
        difference = plus.value - minus.value
        estimate = np.zeros(parameter_values.size, dtype=np.float64)
        for index, is_trainable in enumerate(trainable):
            if not is_trainable:
                continue
            estimate[index] = difference / (2.0 * radius * raw_delta[index])
            if shots is not None:
                if plus.variance is None or minus.variance is None:
                    raise ValueError("SPSA finite-shot samples must include variance")
                if plus.shots is None or minus.shots is None:
                    raise ValueError("SPSA finite-shot samples must include shot counts")
                shot_variance[index] += (
                    plus.variance / float(plus.shots) + minus.variance / float(minus.shots)
                ) / (4.0 * radius * radius)
        gradient_estimates[repetition] = estimate
        if shots is not None:
            total_shots += int(cast(int, plus.shots)) + int(cast(int, minus.shots))
        records.append(
            SPSAProbeRecord(
                repetition=repetition,
                perturbation=perturbation,
                plus_parameters=plus_parameters,
                minus_parameters=minus_parameters,
                plus=plus,
                minus=minus,
                gradient_estimate=estimate,
            )
        )

    gradient = np.mean(gradient_estimates, axis=0)
    estimator_variance = (
        np.var(gradient_estimates, axis=0, ddof=1) / repetitions
        if repetitions > 1
        else np.zeros(parameter_values.size, dtype=np.float64)
    )
    variance = estimator_variance + shot_variance / float(repetitions * repetitions)
    variance = np.where(trainable, variance, 0.0)
    standard_error = np.sqrt(variance)
    confidence_interval = gradient_confidence_interval(
        gradient,
        standard_error,
        confidence_z=z_value,
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        failure_policy=failure_policy,
    )
    return SPSAGradientResult(
        gradient=gradient,
        standard_error=standard_error,
        covariance=np.diag(variance),
        confidence_radius=z_value * standard_error,
        records=tuple(records),
        perturbation_radius=radius,
        repetitions=repetitions,
        seed=seed,
        confidence_z=z_value,
        method="finite_shot_spsa" if shots is not None else "spsa",
        evaluations=2 * repetitions,
        total_shots=total_shots if shots is not None else None,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        claim_boundary=(
            "seeded local SPSA gradient estimator for scalar objectives; "
            "finite-shot uncertainty is propagated only when objective samples "
            "provide variances and shot counts; no hardware execution"
        ),
        hardware_execution=False,
        confidence_interval=confidence_interval,
        failure_policy_status=confidence_interval.status,
        failure_reasons=confidence_interval.failure_reasons,
    )


def score_function_gradient_estimate(
    rewards: ArrayLike,
    score_vectors: ArrayLike,
    *,
    baseline: float = 0.0,
    parameters: Sequence[Parameter] | None = None,
    confidence_z: float = 1.959963984540054,
    failure_policy: GradientFailurePolicy | None = None,
) -> ScoreFunctionGradientResult:
    """Estimate a likelihood-ratio gradient from materialised score samples."""

    reward_array = _as_parameter_array(rewards)
    scores = _as_real_numeric_array("score_vectors", score_vectors)
    if scores.ndim != 2:
        raise ValueError("score_vectors must be a two-dimensional sample-by-parameter array")
    if reward_array.size < 2:
        raise ValueError("score-function estimator requires at least two samples")
    if scores.shape[0] != reward_array.size:
        raise ValueError(
            "score_vectors row count must match reward count: "
            f"{scores.shape[0]} != {reward_array.size}"
        )
    baseline_value = _as_real_scalar("score-function baseline", baseline)
    z_value = _as_real_scalar("score-function confidence_z", confidence_z)
    if z_value <= 0.0:
        raise ValueError("score-function confidence_z must be finite and positive")

    parameter_meta = _normalise_parameters(np.zeros(scores.shape[1], dtype=np.float64), parameters)
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    if not np.any(trainable):
        raise ValueError("score-function estimator requires at least one trainable parameter")

    centred_rewards = reward_array - baseline_value
    weighted_scores = centred_rewards[:, None] * scores
    weighted_scores[:, ~trainable] = 0.0
    gradient = np.mean(weighted_scores, axis=0)
    centred_gradients = weighted_scores - gradient
    covariance = centred_gradients.T @ centred_gradients
    covariance /= float((reward_array.size - 1) * reward_array.size)
    covariance[~trainable, :] = 0.0
    covariance[:, ~trainable] = 0.0
    standard_error = np.sqrt(np.diag(covariance))
    confidence_interval = gradient_confidence_interval(
        gradient,
        standard_error,
        confidence_z=z_value,
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        failure_policy=failure_policy,
    )
    records = tuple(
        ScoreFunctionSampleRecord(
            index=index,
            reward=float(reward_array[index]),
            centred_reward=float(centred_rewards[index]),
            score=scores[index],
            weighted_score=weighted_scores[index],
        )
        for index in range(reward_array.size)
    )
    return ScoreFunctionGradientResult(
        gradient=gradient,
        standard_error=standard_error,
        covariance=covariance,
        confidence_radius=z_value * standard_error,
        records=records,
        baseline=baseline_value,
        sample_count=reward_array.size,
        confidence_z=z_value,
        method="score_function_likelihood_ratio",
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        claim_boundary=(
            "materialised likelihood-ratio score-function estimator for "
            "finite scalar rewards and score vectors; no sampler autodiff, "
            "provider callback, or hardware execution"
        ),
        hardware_execution=False,
        confidence_interval=confidence_interval,
        failure_policy_status=confidence_interval.status,
        failure_reasons=confidence_interval.failure_reasons,
    )


def allocate_parameter_shift_shots(
    plus_variances: ArrayLike,
    minus_variances: ArrayLike,
    *,
    target_standard_error: float,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    min_shots: int = 1,
    max_shots_per_evaluation: int | None = None,
) -> ShotAllocationResult:
    """Plan plus/minus shots to meet a target parameter-shift standard error."""

    shift_rule = rule or ParameterShiftRule()
    terms = shift_rule.terms
    term_count = len(terms)
    plus_var = _as_parameter_shift_sample_tensor(
        "plus_variances",
        plus_variances,
        term_count=term_count,
    )
    minus_var = _as_parameter_shift_sample_tensor(
        "minus_variances",
        minus_variances,
        term_count=term_count,
    )
    if minus_var.shape != plus_var.shape:
        raise ValueError("minus_variances shape must match plus_variances shape")
    if np.any(plus_var < 0.0) or np.any(minus_var < 0.0):
        raise ValueError("shot variances must be finite non-negative values")
    target = _as_real_scalar("target_standard_error", target_standard_error)
    if target <= 0.0:
        raise ValueError("target_standard_error must be finite and positive")
    if isinstance(min_shots, bool) or not isinstance(min_shots, int) or min_shots < 1:
        raise ValueError("min_shots must be a positive integer")
    if max_shots_per_evaluation is not None and (
        isinstance(max_shots_per_evaluation, bool)
        or not isinstance(max_shots_per_evaluation, int)
        or max_shots_per_evaluation < min_shots
    ):
        raise ValueError("max_shots_per_evaluation must be an integer >= min_shots")
    parameter_meta = _normalise_parameters(plus_var[0], parameters)
    shot_plan = np.full((term_count, 2, plus_var.shape[1]), float(min_shots), dtype=np.float64)
    variance = np.zeros(plus_var.shape[1], dtype=np.float64)
    target_variance = target**2

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        noises: list[tuple[int, int, float]] = []
        root_sum = 0.0
        for term_index, (_shift, coefficient) in enumerate(terms):
            coefficient_squared = coefficient**2
            plus_noise = coefficient_squared * plus_var[term_index, index]
            minus_noise = coefficient_squared * minus_var[term_index, index]
            noises.append((term_index, 0, float(plus_noise)))
            noises.append((term_index, 1, float(minus_noise)))
            root_sum += float(np.sqrt(plus_noise) + np.sqrt(minus_noise))
        if root_sum > 0.0:
            for term_index, side_index, noise in noises:
                required = np.sqrt(noise) * root_sum / target_variance
                shot_plan[term_index, side_index, index] = max(
                    float(min_shots),
                    float(np.ceil(required)),
                )
        else:
            shot_plan[:, :, index] = float(min_shots)
        if max_shots_per_evaluation is not None:
            shot_plan[:, :, index] = np.minimum(
                shot_plan[:, :, index],
                float(max_shots_per_evaluation),
            )
        for term_index, (_shift, coefficient) in enumerate(terms):
            variance[index] += coefficient**2 * (
                plus_var[term_index, index] / shot_plan[term_index, 0, index]
                + minus_var[term_index, index] / shot_plan[term_index, 1, index]
            )

    standard_error = np.sqrt(variance)
    output_shots = shot_plan[0] if shift_rule.is_single_term else shot_plan
    return ShotAllocationResult(
        shots=output_shots,
        predicted_standard_error=standard_error,
        covariance=np.diag(variance),
        target_standard_error=target,
        total_shots=int(np.sum(output_shots)),
        method="parameter_shift_target_se"
        if shift_rule.is_single_term
        else "multi_frequency_parameter_shift_target_se",
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def finite_difference_gradient(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a central finite-difference gradient for scalar diagnostics."""

    result = value_and_finite_difference_grad(
        objective,
        values,
        parameters=parameters,
        step=step,
    )
    return result.gradient


def complex_step_gradient(
    objective: ComplexStepObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-30,
) -> NDArray[np.float64]:
    """Return a complex-step gradient for real-analytic scalar objectives."""

    result = value_and_complex_step_grad(
        objective,
        values,
        parameters=parameters,
        step=step,
    )
    return result.gradient


def batch_complex_step_gradient(
    objectives: Sequence[ComplexStepObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-30,
) -> NDArray[np.float64]:
    """Return stacked complex-step gradients for real-analytic objectives."""

    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    rows = [
        complex_step_gradient(
            objective,
            values,
            parameters=parameters,
            step=step,
        )
        for objective in objectives
    ]
    return np.vstack(rows)


def batch_value_and_complex_step_grad(
    objectives: Sequence[ComplexStepObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-30,
) -> tuple[GradientResult, ...]:
    """Return full complex-step results for multiple scalar objectives."""

    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    return tuple(
        value_and_complex_step_grad(
            objective,
            values,
            parameters=parameters,
            step=step,
        )
        for objective in objectives
    )


def value_and_forward_mode_grad(
    objective: Callable[[tuple[DualNumber, ...]], object],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> GradientResult:
    """Evaluate a scalar objective and exact forward-mode dual gradient."""

    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    base_duals = tuple(DualNumber(float(value), 0.0) for value in parameter_values)
    base_value = _as_forward_mode_scalar(objective(base_duals)).primal
    gradient = np.zeros_like(parameter_values)
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        dual_values = tuple(
            DualNumber(float(value), 1.0 if basis_index == index else 0.0)
            for basis_index, value in enumerate(parameter_values)
        )
        gradient[index] = _as_forward_mode_scalar(objective(dual_values)).tangent
        evaluations += 1

    return GradientResult(
        value=base_value,
        gradient=gradient,
        method="forward_mode_dual",
        shift=None,
        coefficient=None,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def forward_mode_gradient(
    objective: Callable[[tuple[DualNumber, ...]], object],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return an exact forward-mode dual gradient for scalar objectives."""

    return value_and_forward_mode_grad(
        objective,
        values,
        parameters=parameters,
    ).gradient


def value_and_reverse_mode_grad(
    objective: Callable[[tuple[ReverseNode, ...]], object],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> GradientResult:
    """Evaluate a scalar objective and exact reverse-mode tape gradient."""

    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    reverse_values = tuple(ReverseNode(float(value)) for value in parameter_values)
    output = _as_reverse_mode_scalar(objective(reverse_values))
    tape = _reverse_topological_order(output)
    for node in tape:
        node.adjoint = 0.0
    output.adjoint = 1.0
    for node in reversed(tape):
        for parent, local_derivative in node.parents:
            parent.adjoint += node.adjoint * local_derivative
    gradient = np.array(
        [
            node.adjoint if parameter.trainable else 0.0
            for node, parameter in zip(reverse_values, parameter_meta)
        ],
        dtype=np.float64,
    )
    return GradientResult(
        value=output.primal,
        gradient=gradient,
        method="reverse_mode_tape",
        shift=None,
        coefficient=None,
        evaluations=1,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def reverse_mode_gradient(
    objective: Callable[[tuple[ReverseNode, ...]], object],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return an exact reverse-mode tape gradient for scalar objectives."""

    return value_and_reverse_mode_grad(
        objective,
        values,
        parameters=parameters,
    ).gradient


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


def value_and_jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> JacobianResult:
    """Evaluate a vector objective and Jacobian through the canonical transform API."""

    if method != "finite_difference":
        raise ValueError("Jacobian method must be finite_difference")
    return value_and_finite_difference_jacobian(
        objective,
        values,
        parameters=parameters,
        step=step,
    )


def jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a vector-objective Jacobian through the canonical transform API."""

    return value_and_jacobian(
        objective,
        values,
        parameters=parameters,
        method=method,
        step=step,
    ).jacobian


def value_and_jacfwd(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> JacobianResult:
    """Evaluate a vector objective and Jacobian through forward-Jacobian semantics.

    The current backend is the same central finite-difference Jacobian used by
    ``jacobian``. The separate name establishes transform algebra semantics for
    callers and tests while leaving room for a future true forward-mode Jacobian
    implementation behind the same contract.
    """

    return value_and_jacobian(
        objective,
        values,
        parameters=parameters,
        method=method,
        step=step,
    )


def jacfwd(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a vector-objective Jacobian using forward-Jacobian semantics."""

    return value_and_jacfwd(
        objective,
        values,
        parameters=parameters,
        method=method,
        step=step,
    ).jacobian


def value_and_jacrev(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> JacobianResult:
    """Evaluate a vector objective and Jacobian through reverse-Jacobian semantics.

    Until a true reverse-over-vector backend exists, this is an explicit alias to
    the finite-difference Jacobian contract. It preserves API and composition
    semantics without overclaiming reverse compiler AD.
    """

    return value_and_jacobian(
        objective,
        values,
        parameters=parameters,
        method=method,
        step=step,
    )


def jacrev(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a vector-objective Jacobian using reverse-Jacobian semantics."""

    return value_and_jacrev(
        objective,
        values,
        parameters=parameters,
        method=method,
        step=step,
    ).jacobian


def value_and_hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-4,
) -> HessianResult:
    """Evaluate a scalar objective and Hessian through the canonical transform API."""

    if method != "finite_difference":
        raise ValueError("Hessian method must be finite_difference")
    return value_and_finite_difference_hessian(
        objective,
        values,
        parameters=parameters,
        step=step,
    )


def hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-4,
) -> NDArray[np.float64]:
    """Return a scalar-objective Hessian through the canonical transform API."""

    return value_and_hessian(
        objective,
        values,
        parameters=parameters,
        method=method,
        step=step,
    ).hessian


def dense_to_sparse_matrix(
    matrix: ArrayLike,
    *,
    parameter_names: Sequence[str] | None = None,
    trainable: Sequence[bool] | None = None,
    method: str = "dense_to_sparse",
    tolerance: float = 0.0,
) -> SparseMatrixResult:
    """Convert a dense derivative matrix to a validated coordinate sparse form."""

    matrix_arr = _as_real_numeric_array("sparse source matrix", matrix)
    if matrix_arr.ndim != 2:
        raise ValueError("sparse source matrix must be two-dimensional")
    if not np.all(np.isfinite(matrix_arr)):
        raise ValueError("sparse source matrix must contain only finite values")
    tolerance_value = _as_real_scalar("sparse tolerance", tolerance)
    if tolerance_value < 0.0:
        raise ValueError("sparse tolerance must be finite and non-negative")
    names = (
        tuple(f"p{index}" for index in range(matrix_arr.shape[1]))
        if parameter_names is None
        else tuple(parameter_names)
    )
    trainable_mask = (
        tuple(True for _ in range(matrix_arr.shape[1])) if trainable is None else tuple(trainable)
    )
    row_indices, column_indices = np.nonzero(np.abs(matrix_arr) > tolerance_value)
    values = matrix_arr[row_indices, column_indices]
    return SparseMatrixResult(
        row_indices=cast(NDArray[np.int64], row_indices.astype(np.int64)),
        column_indices=cast(NDArray[np.int64], column_indices.astype(np.int64)),
        values=cast(NDArray[np.float64], values.astype(np.float64)),
        shape=(int(matrix_arr.shape[0]), int(matrix_arr.shape[1])),
        method=method,
        parameter_names=names,
        trainable=trainable_mask,
    )


def sparse_jacobian(
    jacobian_result: JacobianResult,
    *,
    tolerance: float = 0.0,
) -> SparseMatrixResult:
    """Return a coordinate sparse representation of a Jacobian result."""

    if not isinstance(jacobian_result, JacobianResult):
        raise ValueError("sparse_jacobian requires a JacobianResult")
    return dense_to_sparse_matrix(
        jacobian_result.jacobian,
        parameter_names=jacobian_result.parameter_names,
        trainable=jacobian_result.trainable,
        method=f"sparse:{jacobian_result.method}",
        tolerance=tolerance,
    )


def sparse_hessian(
    hessian_result: HessianResult,
    *,
    tolerance: float = 0.0,
) -> SparseMatrixResult:
    """Return a coordinate sparse representation of a Hessian result."""

    if not isinstance(hessian_result, HessianResult):
        raise ValueError("sparse_hessian requires a HessianResult")
    return dense_to_sparse_matrix(
        hessian_result.hessian,
        parameter_names=hessian_result.parameter_names,
        trainable=hessian_result.trainable,
        method=f"sparse:{hessian_result.method}",
        tolerance=tolerance,
    )


def batch_value_and_finite_difference_grad(
    objectives: Sequence[ScalarObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> tuple[GradientResult, ...]:
    """Return full finite-difference results for multiple scalar objectives."""

    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    return tuple(
        value_and_finite_difference_grad(
            objective,
            values,
            parameters=parameters,
            step=step,
        )
        for objective in objectives
    )


def weighted_gradient_sum(
    components: Sequence[GradientResult],
    weights: ArrayLike,
    *,
    method: str = "weighted_sum",
) -> WeightedGradientResult:
    """Combine compatible scalar gradient results by an explicit weight vector."""

    component_tuple = tuple(components)
    if not component_tuple:
        raise ValueError("components must contain at least one GradientResult")
    if any(not isinstance(component, GradientResult) for component in component_tuple):
        raise ValueError("components must contain GradientResult instances")
    weight_arr = _as_real_numeric_array("weights", weights)
    if weight_arr.ndim != 1 or weight_arr.size != len(component_tuple):
        raise ValueError("weights length must match components length")
    if not np.all(np.isfinite(weight_arr)):
        raise ValueError("weights must contain only finite values")
    reference = component_tuple[0]
    for component in component_tuple[1:]:
        if component.gradient.shape != reference.gradient.shape:
            raise ValueError("all component gradients must have matching shapes")
        if component.parameter_names != reference.parameter_names:
            raise ValueError("all component parameter_names must match")
        if component.trainable != reference.trainable:
            raise ValueError("all component trainable masks must match")
    value = float(
        sum(
            float(weight) * component.value
            for weight, component in zip(weight_arr, component_tuple)
        )
    )
    gradient = np.zeros_like(reference.gradient)
    evaluations = 0
    for weight, component in zip(weight_arr, component_tuple):
        gradient += float(weight) * component.gradient
        evaluations += component.evaluations
    return WeightedGradientResult(
        value=value,
        gradient=gradient,
        components=component_tuple,
        weights=weight_arr,
        method=method,
        evaluations=evaluations,
        parameter_names=reference.parameter_names,
        trainable=reference.trainable,
    )


def value_and_complex_step_grad(
    objective: ComplexStepObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-30,
) -> GradientResult:
    """Evaluate a real-analytic scalar objective and complex-step gradient."""

    step_value = _as_real_scalar("complex-step step", step)
    if step_value <= 0.0:
        raise ValueError("complex-step step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    gradient = np.zeros_like(parameter_values)
    base_values = cast(NDArray[np.complex128], parameter_values.astype(np.complex128))
    base_scalar = _as_complex_step_scalar(objective(base_values))
    if base_scalar.imag != 0.0:
        raise ValueError("complex-step objective returned a non-real base scalar")
    base_value = float(base_scalar.real)
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        perturbed = cast(NDArray[np.complex128], parameter_values.astype(np.complex128))
        perturbed[index] += 1j * step_value
        perturbed_value = _as_complex_step_scalar(objective(perturbed))
        evaluations += 1
        gradient[index] = perturbed_value.imag / step_value

    return GradientResult(
        value=base_value,
        gradient=gradient,
        method="complex_step",
        shift=step_value,
        coefficient=1.0 / step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def value_and_finite_difference_grad(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> GradientResult:
    """Evaluate a scalar objective and central finite-difference gradient."""

    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    gradient = np.zeros_like(parameter_values)
    base_value = _as_scalar(objective(parameter_values.copy()))
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        plus = parameter_values.copy()
        minus = parameter_values.copy()
        plus[index] += step_value
        minus[index] -= step_value
        plus_value = _as_scalar(objective(plus))
        minus_value = _as_scalar(objective(minus))
        evaluations += 2
        gradient[index] = (plus_value - minus_value) / (2.0 * step_value)

    return GradientResult(
        value=base_value,
        gradient=gradient,
        method="finite_difference_central",
        shift=step_value,
        coefficient=1.0 / (2.0 * step_value),
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        claim_boundary=FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY,
    )


def finite_difference_jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a central finite-difference Jacobian for vector objectives."""

    return value_and_finite_difference_jacobian(
        objective,
        values,
        parameters=parameters,
        step=step,
    ).jacobian


def value_and_finite_difference_jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> JacobianResult:
    """Evaluate a vector objective and its central finite-difference Jacobian."""

    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    base_value = _as_vector_output(objective(parameter_values.copy()))
    jacobian = np.zeros((base_value.size, parameter_values.size), dtype=np.float64)
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        plus = parameter_values.copy()
        minus = parameter_values.copy()
        plus[index] += step_value
        minus[index] -= step_value
        plus_value = _as_vector_output(objective(plus))
        minus_value = _as_vector_output(objective(minus))
        if plus_value.shape != base_value.shape or minus_value.shape != base_value.shape:
            raise ValueError("vector objective output shape must remain stable")
        evaluations += 2
        jacobian[:, index] = (plus_value - minus_value) / (2.0 * step_value)

    return JacobianResult(
        value=base_value,
        jacobian=jacobian,
        method="finite_difference_central",
        step=step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        claim_boundary=FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY,
    )


def finite_difference_jvp(
    objective: VectorObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a central finite-difference Jacobian-vector product."""

    return value_and_finite_difference_jvp(
        objective,
        values,
        tangent,
        parameters=parameters,
        step=step,
    ).jvp


def value_and_jvp(
    objective: VectorObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> JVPResult:
    """Evaluate a vector objective and canonical Jacobian-vector product transform."""

    if method != "finite_difference":
        raise ValueError("JVP method must be finite_difference")
    return value_and_finite_difference_jvp(
        objective,
        values,
        tangent,
        parameters=parameters,
        step=step,
    )


def jvp(
    objective: VectorObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a canonical Jacobian-vector product transform."""

    return value_and_jvp(
        objective,
        values,
        tangent,
        parameters=parameters,
        method=method,
        step=step,
    ).jvp


def value_and_finite_difference_jvp(
    objective: VectorObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> JVPResult:
    """Evaluate a vector objective and a directional finite-difference JVP."""

    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    tangent_values = _as_parameter_array(tangent)
    if tangent_values.shape != parameter_values.shape:
        raise ValueError("JVP tangent length must match parameter length")
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    masked_tangent = tangent_values.copy()
    masked_tangent[~trainable] = 0.0
    base_value = _as_vector_output(objective(parameter_values.copy()))
    if not np.any(masked_tangent):
        jvp = np.zeros_like(base_value)
        evaluations = 1
    else:
        plus = parameter_values + step_value * masked_tangent
        minus = parameter_values - step_value * masked_tangent
        plus_value = _as_vector_output(objective(plus))
        minus_value = _as_vector_output(objective(minus))
        if plus_value.shape != base_value.shape or minus_value.shape != base_value.shape:
            raise ValueError("vector objective output shape must remain stable")
        jvp = (plus_value - minus_value) / (2.0 * step_value)
        evaluations = 3
    return JVPResult(
        value=base_value,
        jvp=jvp,
        tangent=masked_tangent,
        method="finite_difference_directional",
        step=step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        claim_boundary=FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY,
    )


def batch_finite_difference_jvp(
    objective: VectorObjective,
    values: ArrayLike,
    tangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return stacked finite-difference JVPs for a batch of tangents."""

    results = batch_value_and_finite_difference_jvp(
        objective,
        values,
        tangents,
        parameters=parameters,
        step=step,
    )
    return cast(NDArray[np.float64], np.vstack([result.jvp for result in results]))


def batch_value_and_finite_difference_jvp(
    objective: VectorObjective,
    values: ArrayLike,
    tangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> tuple[JVPResult, ...]:
    """Return one finite-difference JVP result per tangent row."""

    parameter_values = _as_parameter_array(values)
    tangent_batch = _as_batch_parameter_array("JVP tangents", tangents, parameter_values.size)
    return tuple(
        value_and_finite_difference_jvp(
            objective,
            parameter_values,
            tangent,
            parameters=parameters,
            step=step,
        )
        for tangent in tangent_batch
    )


def vector_jacobian_product(
    jacobian: JacobianResult,
    cotangent: ArrayLike,
) -> VJPResult:
    """Contract a validated cotangent with a vector-objective Jacobian."""

    if not isinstance(jacobian, JacobianResult):
        raise ValueError("vector_jacobian_product requires a JacobianResult")
    cotangent_values = _as_vector_output(cotangent)
    if cotangent_values.shape != jacobian.value.shape:
        raise ValueError("VJP cotangent shape must match Jacobian value shape")
    vjp = jacobian.jacobian.T @ cotangent_values
    trainable = np.asarray(jacobian.trainable, dtype=bool)
    vjp[~trainable] = 0.0
    return VJPResult(
        value=jacobian.value,
        cotangent=cotangent_values,
        vjp=vjp,
        method=f"vjp:{jacobian.method}",
        step=jacobian.step,
        evaluations=jacobian.evaluations,
        parameter_names=jacobian.parameter_names,
        trainable=jacobian.trainable,
        claim_boundary=jacobian.claim_boundary,
    )


def _normalise_custom_derivative_parameters(
    values: NDArray[np.float64],
    rule: CustomDerivativeRule,
    parameters: Sequence[Parameter] | None,
) -> tuple[Parameter, ...]:
    """Return explicit parameter metadata for a custom derivative primitive."""

    if parameters is not None:
        return _normalise_parameters(values, parameters)
    if rule.parameter_names:
        if len(rule.parameter_names) != values.size:
            raise ValueError(
                "custom derivative parameter_names length must match parameter length"
            )
        trainable = (
            rule.trainable
            if rule.trainable
            else tuple(True for _ in range(len(rule.parameter_names)))
        )
        if len(trainable) != values.size:
            raise ValueError("custom derivative trainable mask length must match parameter length")
        return tuple(
            Parameter(name, trainable=flag)
            for name, flag in zip(rule.parameter_names, trainable, strict=True)
        )
    return _normalise_parameters(values, None)


def custom_jvp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return an exact custom Jacobian-vector product for a registered primitive."""

    return value_and_custom_jvp(rule, values, tangent, parameters=parameters).jvp


def batch_custom_jvp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    tangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return stacked exact custom JVPs for a batch of tangent vectors."""

    return cast(
        NDArray[np.float64],
        np.vstack(
            [
                result.jvp
                for result in batch_value_and_custom_jvp(
                    rule,
                    values,
                    tangents,
                    parameters=parameters,
                )
            ]
        ),
    )


def batch_value_and_custom_jvp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    tangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> tuple[JVPResult, ...]:
    """Return one exact custom JVP result per tangent row."""

    parameter_values = _as_parameter_array(values)
    tangent_batch = _as_batch_parameter_array(
        "custom JVP tangents", tangents, parameter_values.size
    )
    return tuple(
        value_and_custom_jvp(
            rule,
            parameter_values,
            tangent,
            parameters=parameters,
        )
        for tangent in tangent_batch
    )


def value_and_custom_jvp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> JVPResult:
    """Evaluate a custom primitive and its exact JVP rule."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("custom JVP requires a CustomDerivativeRule")
    if rule.jvp_rule is None:
        raise ValueError("custom derivative rule does not define a JVP rule")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_custom_derivative_parameters(parameter_values, rule, parameters)
    tangent_values = _as_parameter_array(tangent)
    if tangent_values.shape != parameter_values.shape:
        raise ValueError("custom JVP tangent length must match parameter length")
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    masked_tangent = tangent_values.copy()
    masked_tangent[~trainable] = 0.0
    value = _as_vector_output(rule.value_fn(parameter_values.copy()))
    jvp = _as_vector_output(rule.jvp_rule(parameter_values.copy(), masked_tangent.copy()))
    if jvp.shape != value.shape:
        raise ValueError("custom JVP output shape must match primitive value shape")
    return JVPResult(
        value=value,
        jvp=jvp,
        tangent=masked_tangent,
        method=f"custom_jvp:{rule.name}",
        step=0.0,
        evaluations=1,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def custom_vjp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> VJPResult:
    """Return an exact custom vector-Jacobian product for a registered primitive."""

    return value_and_custom_vjp(rule, values, cotangent, parameters=parameters)


def batch_custom_vjp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    cotangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return stacked exact custom VJPs for a batch of cotangent vectors."""

    return cast(
        NDArray[np.float64],
        np.vstack(
            [
                result.vjp
                for result in batch_value_and_custom_vjp(
                    rule,
                    values,
                    cotangents,
                    parameters=parameters,
                )
            ]
        ),
    )


def batch_value_and_custom_vjp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    cotangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> tuple[VJPResult, ...]:
    """Return one exact custom VJP result per cotangent row."""

    parameter_values = _as_parameter_array(values)
    value = _as_vector_output(rule.value_fn(parameter_values.copy()))
    cotangent_batch = _as_batch_vector_array("custom VJP cotangents", cotangents, value.size)
    return tuple(
        value_and_custom_vjp(
            rule,
            parameter_values,
            cotangent,
            parameters=parameters,
        )
        for cotangent in cotangent_batch
    )


def value_and_custom_vjp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> VJPResult:
    """Evaluate a custom primitive and its exact VJP rule."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("custom VJP requires a CustomDerivativeRule")
    if rule.vjp_rule is None:
        raise ValueError("custom derivative rule does not define a VJP rule")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_custom_derivative_parameters(parameter_values, rule, parameters)
    value = _as_vector_output(rule.value_fn(parameter_values.copy()))
    cotangent_values = _as_vector_output(cotangent)
    if cotangent_values.shape != value.shape:
        raise ValueError("custom VJP cotangent shape must match primitive value shape")
    vjp = _as_parameter_array(rule.vjp_rule(parameter_values.copy(), cotangent_values.copy()))
    if vjp.shape != parameter_values.shape:
        raise ValueError("custom VJP output length must match parameter length")
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    masked_vjp = vjp.copy()
    masked_vjp[~trainable] = 0.0
    return VJPResult(
        value=value,
        cotangent=cotangent_values,
        vjp=masked_vjp,
        method=f"custom_vjp:{rule.name}",
        step=0.0,
        evaluations=1,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def check_custom_derivative_consistency(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    tangent: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    finite_difference_step: float = 1.0e-6,
    tolerance: float = 1.0e-5,
) -> CustomDerivativeCheckResult:
    """Check custom JVP/VJP rules against adjoint and finite-difference identities."""

    tolerance_value = _as_real_scalar("custom derivative tolerance", tolerance)
    if tolerance_value < 0.0:
        raise ValueError("custom derivative tolerance must be finite and non-negative")
    step_value = _as_real_scalar(
        "custom derivative finite_difference_step", finite_difference_step
    )
    if step_value <= 0.0:
        raise ValueError("custom derivative finite_difference_step must be finite and positive")
    custom_jvp_result = value_and_custom_jvp(
        rule,
        values,
        tangent,
        parameters=parameters,
    )
    custom_vjp_result = value_and_custom_vjp(
        rule,
        values,
        cotangent,
        parameters=parameters,
    )
    parameter_values = _as_parameter_array(values)
    reference_parameters = tuple(
        Parameter(name, trainable=flag)
        for name, flag in zip(
            custom_jvp_result.parameter_names,
            custom_jvp_result.trainable,
            strict=True,
        )
    )
    reference_jvp = value_and_finite_difference_jvp(
        rule.value_fn,
        parameter_values,
        custom_jvp_result.tangent,
        parameters=reference_parameters,
        step=step_value,
    )
    reference_vjp = finite_difference_vjp(
        rule.value_fn,
        parameter_values,
        custom_vjp_result.cotangent,
        parameters=reference_parameters,
        step=step_value,
    )
    primal_inner = float(np.dot(custom_jvp_result.jvp, custom_vjp_result.cotangent))
    adjoint_inner = float(np.dot(custom_jvp_result.tangent, custom_vjp_result.vjp))
    adjoint_inner_error = abs(primal_inner - adjoint_inner)
    jvp_l2_error = float(np.linalg.norm(custom_jvp_result.jvp - reference_jvp.jvp))
    vjp_l2_error = float(np.linalg.norm(custom_vjp_result.vjp - reference_vjp.vjp))
    passed = (
        adjoint_inner_error <= tolerance_value
        and jvp_l2_error <= tolerance_value
        and vjp_l2_error <= tolerance_value
    )
    return CustomDerivativeCheckResult(
        custom_jvp=custom_jvp_result,
        custom_vjp=custom_vjp_result,
        reference_jvp=reference_jvp,
        reference_vjp=reference_vjp,
        adjoint_inner_error=adjoint_inner_error,
        jvp_l2_error=jvp_l2_error,
        vjp_l2_error=vjp_l2_error,
        tolerance=tolerance_value,
        passed=passed,
    )


def custom_jacobian(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return the exact dense Jacobian implied by a custom derivative rule."""

    return value_and_custom_jacobian(rule, values, parameters=parameters).jacobian


def batch_custom_jacobian(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return stacked exact custom Jacobians for a batch of parameter rows."""

    return cast(
        NDArray[np.float64],
        np.stack(
            [
                result.jacobian
                for result in batch_value_and_custom_jacobian(
                    rule,
                    values,
                    parameters=parameters,
                )
            ],
            axis=0,
        ),
    )


def batch_value_and_custom_jacobian(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> tuple[JacobianResult, ...]:
    """Return one exact custom Jacobian result per parameter row."""

    batch = _as_real_numeric_array("custom Jacobian values", values)
    if batch.ndim != 2:
        raise ValueError("custom Jacobian values must be a two-dimensional batch")
    if not np.all(np.isfinite(batch)):
        raise ValueError("custom Jacobian values must contain only finite values")
    return tuple(
        value_and_custom_jacobian(
            rule,
            row,
            parameters=parameters,
        )
        for row in batch
    )


def value_and_custom_jacobian(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> JacobianResult:
    """Evaluate a custom primitive and materialise its exact dense Jacobian."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("custom Jacobian requires a CustomDerivativeRule")
    if rule.jvp_rule is None and rule.vjp_rule is None:
        raise ValueError("custom derivative rule requires a JVP or VJP rule")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_custom_derivative_parameters(parameter_values, rule, parameters)
    value = _as_vector_output(rule.value_fn(parameter_values.copy()))
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    jacobian_arr = np.zeros((value.size, parameter_values.size), dtype=np.float64)
    evaluations = 1
    if rule.jvp_rule is not None:
        for column, is_trainable in enumerate(trainable):
            if not is_trainable:
                continue
            basis = np.zeros(parameter_values.size, dtype=np.float64)
            basis[column] = 1.0
            jvp = _as_vector_output(rule.jvp_rule(parameter_values.copy(), basis))
            if jvp.shape != value.shape:
                raise ValueError("custom JVP output shape must match primitive value shape")
            jacobian_arr[:, column] = jvp
    else:
        vjp_rule = rule.vjp_rule
        if vjp_rule is None:
            raise ValueError("custom derivative rule requires a JVP or VJP rule")
        for row in range(value.size):
            cotangent = np.zeros(value.size, dtype=np.float64)
            cotangent[row] = 1.0
            vjp = _as_parameter_array(vjp_rule(parameter_values.copy(), cotangent))
            if vjp.shape != parameter_values.shape:
                raise ValueError("custom VJP output length must match parameter length")
            vjp[~trainable] = 0.0
            jacobian_arr[row, :] = vjp
    return JacobianResult(
        value=value,
        jacobian=jacobian_arr,
        method=f"custom_jacobian:{rule.name}",
        step=0.0,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def finite_difference_vjp(
    objective: VectorObjective,
    values: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> VJPResult:
    """Return a finite-difference vector-Jacobian product for a vector objective."""

    jacobian = value_and_finite_difference_jacobian(
        objective,
        values,
        parameters=parameters,
        step=step,
    )
    return vector_jacobian_product(jacobian, cotangent)


def value_and_finite_difference_vjp(
    objective: VectorObjective,
    values: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> VJPResult:
    """Evaluate a vector objective and one finite-difference VJP result."""

    jacobian = value_and_finite_difference_jacobian(
        objective,
        values,
        parameters=parameters,
        step=step,
    )
    return vector_jacobian_product(jacobian, cotangent)


def value_and_vjp(
    objective: VectorObjective,
    values: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> VJPResult:
    """Evaluate a vector objective and canonical vector-Jacobian product transform."""

    if method != "finite_difference":
        raise ValueError("VJP method must be finite_difference")
    return value_and_finite_difference_vjp(
        objective,
        values,
        cotangent,
        parameters=parameters,
        step=step,
    )


def vjp(
    objective: VectorObjective,
    values: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a canonical vector-Jacobian product transform."""

    return value_and_vjp(
        objective,
        values,
        cotangent,
        parameters=parameters,
        method=method,
        step=step,
    ).vjp


def batch_vector_jacobian_product(
    jacobian: JacobianResult,
    cotangents: ArrayLike,
) -> tuple[VJPResult, ...]:
    """Return one vector-Jacobian product per cotangent row."""

    if not isinstance(jacobian, JacobianResult):
        raise ValueError("batch_vector_jacobian_product requires a JacobianResult")
    cotangent_batch = _as_batch_vector_array("VJP cotangents", cotangents, jacobian.value.size)
    return tuple(vector_jacobian_product(jacobian, cotangent) for cotangent in cotangent_batch)


def batch_finite_difference_vjp(
    objective: VectorObjective,
    values: ArrayLike,
    cotangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return stacked finite-difference VJPs for a batch of cotangents."""

    results = batch_value_and_finite_difference_vjp(
        objective,
        values,
        cotangents,
        parameters=parameters,
        step=step,
    )
    return cast(NDArray[np.float64], np.vstack([result.vjp for result in results]))


def batch_value_and_finite_difference_vjp(
    objective: VectorObjective,
    values: ArrayLike,
    cotangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> tuple[VJPResult, ...]:
    """Return one finite-difference VJP result per cotangent row."""

    jacobian = value_and_finite_difference_jacobian(
        objective,
        values,
        parameters=parameters,
        step=step,
    )
    return batch_vector_jacobian_product(jacobian, cotangents)


def finite_difference_hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-4,
) -> NDArray[np.float64]:
    """Return a central finite-difference Hessian for scalar objectives."""

    return value_and_finite_difference_hessian(
        objective,
        values,
        parameters=parameters,
        step=step,
    ).hessian


def value_and_finite_difference_hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-4,
) -> HessianResult:
    """Evaluate a scalar objective and central finite-difference Hessian."""

    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    base_value = _as_scalar(objective(parameter_values.copy()))
    hessian = np.zeros((parameter_values.size, parameter_values.size), dtype=np.float64)
    evaluations = 1

    for row in range(parameter_values.size):
        if not trainable[row]:
            continue
        plus = parameter_values.copy()
        minus = parameter_values.copy()
        plus[row] += step_value
        minus[row] -= step_value
        plus_value = _as_scalar(objective(plus))
        minus_value = _as_scalar(objective(minus))
        evaluations += 2
        hessian[row, row] = (plus_value - 2.0 * base_value + minus_value) / (step_value**2)

        for column in range(row + 1, parameter_values.size):
            if not trainable[column]:
                continue
            plus_plus = parameter_values.copy()
            plus_minus = parameter_values.copy()
            minus_plus = parameter_values.copy()
            minus_minus = parameter_values.copy()
            plus_plus[row] += step_value
            plus_plus[column] += step_value
            plus_minus[row] += step_value
            plus_minus[column] -= step_value
            minus_plus[row] -= step_value
            minus_plus[column] += step_value
            minus_minus[row] -= step_value
            minus_minus[column] -= step_value
            mixed = (
                _as_scalar(objective(plus_plus))
                - _as_scalar(objective(plus_minus))
                - _as_scalar(objective(minus_plus))
                + _as_scalar(objective(minus_minus))
            ) / (4.0 * step_value**2)
            evaluations += 4
            hessian[row, column] = mixed
            hessian[column, row] = mixed

    return HessianResult(
        value=base_value,
        hessian=hessian,
        method="finite_difference_central",
        step=step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        claim_boundary=FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY,
    )


def finite_difference_hvp(
    objective: ScalarObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-5,
) -> NDArray[np.float64]:
    """Return a central finite-difference Hessian-vector product."""

    return value_and_finite_difference_hvp(
        objective,
        values,
        tangent,
        parameters=parameters,
        step=step,
    ).hvp


def value_and_finite_difference_hvp(
    objective: ScalarObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-5,
) -> HVPResult:
    """Evaluate a scalar objective and a directional Hessian-vector product."""

    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    tangent_values = _as_parameter_array(tangent)
    if tangent_values.shape != parameter_values.shape:
        raise ValueError("HVP tangent length must match parameter length")
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    masked_tangent = tangent_values.copy()
    masked_tangent[~trainable] = 0.0
    base_value = _as_scalar(objective(parameter_values.copy()))
    if not np.any(masked_tangent):
        hvp = np.zeros_like(parameter_values)
        evaluations = 1
    else:
        plus = parameter_values + step_value * masked_tangent
        minus = parameter_values - step_value * masked_tangent
        plus_gradient = value_and_finite_difference_grad(
            objective,
            plus,
            parameters=parameter_meta,
            step=step_value,
        )
        minus_gradient = value_and_finite_difference_grad(
            objective,
            minus,
            parameters=parameter_meta,
            step=step_value,
        )
        hvp = (plus_gradient.gradient - minus_gradient.gradient) / (2.0 * step_value)
        hvp[~trainable] = 0.0
        evaluations = 1 + plus_gradient.evaluations + minus_gradient.evaluations
    return HVPResult(
        value=base_value,
        hvp=hvp,
        tangent=masked_tangent,
        method="finite_difference_hvp",
        step=step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        claim_boundary=FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY,
    )


def batch_finite_difference_hvp(
    objective: ScalarObjective,
    values: ArrayLike,
    tangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-5,
) -> NDArray[np.float64]:
    """Return stacked finite-difference HVPs for a batch of tangents."""

    results = batch_value_and_finite_difference_hvp(
        objective,
        values,
        tangents,
        parameters=parameters,
        step=step,
    )
    return cast(NDArray[np.float64], np.vstack([result.hvp for result in results]))


def batch_value_and_finite_difference_hvp(
    objective: ScalarObjective,
    values: ArrayLike,
    tangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-5,
) -> tuple[HVPResult, ...]:
    """Return one finite-difference HVP result per tangent row."""

    parameter_values = _as_parameter_array(values)
    tangent_batch = _as_batch_parameter_array("HVP tangents", tangents, parameter_values.size)
    return tuple(
        value_and_finite_difference_hvp(
            objective,
            parameter_values,
            tangent,
            parameters=parameters,
            step=step,
        )
        for tangent in tangent_batch
    )


def empirical_fisher_metric(
    jacobian: JacobianResult | ArrayLike,
    *,
    weights: ArrayLike | None = None,
    damping: float = 0.0,
) -> NDArray[np.float64]:
    """Return ``J.T @ W @ J + damping * I`` for differentiable residual maps."""

    jacobian_arr = (
        jacobian.jacobian
        if isinstance(jacobian, JacobianResult)
        else _as_real_numeric_array("jacobian", jacobian)
    )
    if jacobian_arr.ndim != 2:
        raise ValueError("jacobian must be a two-dimensional array")
    if not np.all(np.isfinite(jacobian_arr)):
        raise ValueError("jacobian must contain only finite values")
    if weights is None:
        weighted = jacobian_arr
    else:
        weight_arr = _as_real_numeric_array("weights", weights)
        if weight_arr.ndim != 1 or weight_arr.shape[0] != jacobian_arr.shape[0]:
            raise ValueError("weights must be a one-dimensional array matching jacobian rows")
        if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
            raise ValueError("weights must contain only finite non-negative values")
        weighted = jacobian_arr * weight_arr[:, None]
    damping_value = _as_real_scalar("fisher damping", damping)
    if damping_value < 0.0:
        raise ValueError("fisher damping must be finite and non-negative")
    metric = jacobian_arr.T @ weighted
    if damping_value > 0.0:
        metric = metric + damping_value * np.eye(metric.shape[0], dtype=np.float64)
    typed_metric: NDArray[np.float64] = metric
    return typed_metric


def sparse_empirical_fisher_metric(
    jacobian: JacobianResult | ArrayLike,
    *,
    weights: ArrayLike | None = None,
    damping: float = 0.0,
    tolerance: float = 0.0,
) -> SparseMatrixResult:
    """Return a sparse coordinate empirical-Fisher metric."""

    metric = empirical_fisher_metric(jacobian, weights=weights, damping=damping)
    if isinstance(jacobian, JacobianResult):
        parameter_names = jacobian.parameter_names
        trainable = jacobian.trainable
    else:
        parameter_count = metric.shape[1]
        parameter_names = tuple(f"p{index}" for index in range(parameter_count))
        trainable = tuple(True for _ in range(parameter_count))
    return dense_to_sparse_matrix(
        metric,
        parameter_names=parameter_names,
        trainable=trainable,
        method="sparse:empirical_fisher",
        tolerance=tolerance,
    )


def empirical_fisher_vector_product(
    jacobian: JacobianResult,
    tangent: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    damping: float = 0.0,
) -> FisherVectorProductResult:
    """Return matrix-free ``(J.T @ W @ J + damping I) @ tangent``."""

    if not isinstance(jacobian, JacobianResult):
        raise ValueError("empirical_fisher_vector_product requires a JacobianResult")
    tangent_values = _as_parameter_array(tangent)
    if tangent_values.shape[0] != jacobian.jacobian.shape[1]:
        raise ValueError("Fisher-vector tangent length must match Jacobian parameter dimension")
    trainable = np.asarray(jacobian.trainable, dtype=bool)
    masked_tangent = tangent_values.copy()
    masked_tangent[~trainable] = 0.0
    damping_value = _as_real_scalar("Fisher-vector damping", damping)
    if damping_value < 0.0:
        raise ValueError("Fisher-vector damping must be finite and non-negative")
    projection = jacobian.jacobian @ masked_tangent
    if weights is None:
        weighted_projection = projection
    else:
        weight_arr = _as_real_numeric_array("weights", weights)
        if weight_arr.ndim != 1 or weight_arr.shape[0] != projection.size:
            raise ValueError("weights must be a one-dimensional array matching residual rows")
        if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
            raise ValueError("weights must contain only finite non-negative values")
        weighted_projection = projection * weight_arr
    product = jacobian.jacobian.T @ weighted_projection
    if damping_value > 0.0:
        product[trainable] += damping_value * masked_tangent[trainable]
    product[~trainable] = 0.0
    return FisherVectorProductResult(
        value=jacobian.value,
        tangent=masked_tangent,
        product=product,
        residual_projection=projection,
        damping=damping_value,
        method=f"fisher_vector_product:{jacobian.method}",
        evaluations=jacobian.evaluations,
        parameter_names=jacobian.parameter_names,
        trainable=jacobian.trainable,
    )


def empirical_fisher_conjugate_gradient(
    jacobian: JacobianResult,
    rhs: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    damping: float = 1.0e-8,
    tolerance: float = 1.0e-10,
    max_iterations: int | None = None,
) -> FisherConjugateGradientResult:
    """Solve an empirical-Fisher linear system with matrix-free conjugate gradients."""

    if not isinstance(jacobian, JacobianResult):
        raise ValueError("empirical_fisher_conjugate_gradient requires a JacobianResult")
    rhs_values = _as_parameter_array(rhs)
    if rhs_values.shape[0] != jacobian.jacobian.shape[1]:
        raise ValueError("Fisher-CG rhs length must match Jacobian parameter dimension")
    damping_value = _as_real_scalar("Fisher-CG damping", damping)
    if damping_value < 0.0:
        raise ValueError("Fisher-CG damping must be finite and non-negative")
    tolerance_value = _as_real_scalar("Fisher-CG tolerance", tolerance)
    if tolerance_value < 0.0:
        raise ValueError("Fisher-CG tolerance must be finite and non-negative")
    trainable = np.asarray(jacobian.trainable, dtype=bool)
    if max_iterations is None:
        max_iter = max(1, int(np.count_nonzero(trainable)) * 10)
    else:
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError("Fisher-CG max_iterations must be a positive integer")
        max_iter = max_iterations
    solution = np.zeros_like(rhs_values)
    masked_rhs = rhs_values.copy()
    masked_rhs[~trainable] = 0.0
    residual = masked_rhs.copy()
    residual_norm = float(np.linalg.norm(residual[trainable], ord=2))
    residual_history: list[float] = [residual_norm]
    if residual_norm <= tolerance_value or not np.any(trainable):
        return FisherConjugateGradientResult(
            solution=solution,
            residual_norm_history=tuple(residual_history),
            iterations=0,
            converged=True,
            tolerance=tolerance_value,
            damping=damping_value,
            parameter_names=jacobian.parameter_names,
            trainable=jacobian.trainable,
        )

    direction = residual.copy()
    residual_sq = float(residual[trainable] @ residual[trainable])
    converged = False
    iterations = 0
    for iteration in range(1, max_iter + 1):
        product_result = empirical_fisher_vector_product(
            jacobian,
            direction,
            weights=weights,
            damping=damping_value,
        )
        product = product_result.product
        denom = float(direction[trainable] @ product[trainable])
        if denom <= 0.0 or not np.isfinite(denom):
            raise ValueError(
                "Fisher-CG operator must be positive definite on trainable parameters"
            )
        alpha = residual_sq / denom
        solution[trainable] += alpha * direction[trainable]
        residual[trainable] -= alpha * product[trainable]
        new_residual_sq = float(residual[trainable] @ residual[trainable])
        residual_norm = float(np.sqrt(max(new_residual_sq, 0.0)))
        residual_history.append(residual_norm)
        iterations = iteration
        if residual_norm <= tolerance_value:
            converged = True
            break
        beta = new_residual_sq / residual_sq
        direction[trainable] = residual[trainable] + beta * direction[trainable]
        direction[~trainable] = 0.0
        residual_sq = new_residual_sq
    solution[~trainable] = 0.0
    return FisherConjugateGradientResult(
        solution=solution,
        residual_norm_history=tuple(residual_history),
        iterations=iterations,
        converged=converged,
        tolerance=tolerance_value,
        damping=damping_value,
        parameter_names=jacobian.parameter_names,
        trainable=jacobian.trainable,
    )


def least_squares_covariance(
    jacobian: JacobianResult,
    *,
    weights: ArrayLike | None = None,
    residual_variance: float | None = None,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> LeastSquaresCovarianceResult:
    """Estimate parameter covariance from a residual-map Fisher metric."""

    if not isinstance(jacobian, JacobianResult):
        raise ValueError("least_squares_covariance requires a JacobianResult")
    residual = jacobian.value
    trainable = np.asarray(jacobian.trainable, dtype=bool)
    active_count = int(np.count_nonzero(trainable))
    if active_count == 0:
        raise ValueError("least_squares_covariance requires at least one trainable parameter")
    metric = empirical_fisher_metric(jacobian, weights=weights, damping=damping)
    active_metric = metric[np.ix_(trainable, trainable)]
    eigenvalues = np.linalg.eigvalsh(active_metric)
    min_eigenvalue = float(np.min(eigenvalues))
    max_eigenvalue = float(np.max(eigenvalues))
    rcond_value = _as_real_scalar("least-squares rcond", rcond)
    if not 0.0 < rcond_value < 1.0:
        raise ValueError("rcond must be finite and between 0 and 1")
    if min_eigenvalue <= 0.0:
        raise ValueError("least-squares Fisher metric must be positive definite")
    condition_number = max_eigenvalue / min_eigenvalue
    if condition_number > 1.0 / rcond_value:
        raise ValueError("least-squares Fisher metric is ill-conditioned")
    degrees_of_freedom = max(1, residual.size - active_count)
    if residual_variance is None:
        if weights is None:
            weighted_residual = residual
        else:
            weight_arr = _as_real_numeric_array("weights", weights)
            if weight_arr.ndim != 1 or weight_arr.shape[0] != residual.size:
                raise ValueError("weights must be a one-dimensional array matching residual rows")
            if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
                raise ValueError("weights must contain only finite non-negative values")
            weighted_residual = residual * weight_arr
        variance = float(residual @ weighted_residual) / degrees_of_freedom
    else:
        variance = _as_real_scalar("least-squares residual_variance", residual_variance)
        if variance < 0.0:
            raise ValueError("residual_variance must be finite and non-negative")
    active_covariance = np.linalg.inv(active_metric) * variance
    covariance = np.zeros_like(metric)
    covariance[np.ix_(trainable, trainable)] = active_covariance
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return LeastSquaresCovarianceResult(
        covariance=covariance,
        standard_errors=standard_errors,
        residual_variance=variance,
        degrees_of_freedom=degrees_of_freedom,
        condition_number=condition_number,
        parameter_names=jacobian.parameter_names,
        trainable=jacobian.trainable,
    )


def huber_residual_weights(
    residuals: ArrayLike,
    *,
    delta: float = 1.0,
    min_weight: float = 0.0,
) -> NDArray[np.float64]:
    """Return Huber IRLS weights for robust residual-map least squares."""

    residual_arr = _as_vector_output(residuals)
    delta_value = _as_real_scalar("Huber delta", delta)
    if delta_value <= 0.0:
        raise ValueError("Huber delta must be finite and positive")
    min_weight_value = _as_real_scalar("Huber min_weight", min_weight)
    if min_weight_value < 0.0 or min_weight_value > 1.0:
        raise ValueError("Huber min_weight must be finite and in [0, 1]")

    magnitudes = np.abs(residual_arr)
    weights = np.ones_like(residual_arr, dtype=np.float64)
    outliers = magnitudes > delta_value
    weights[outliers] = delta_value / magnitudes[outliers]
    if min_weight_value > 0.0:
        weights = np.maximum(weights, min_weight_value)
    return weights


def soft_l1_residual_weights(
    residuals: ArrayLike,
    *,
    scale: float = 1.0,
    min_weight: float = 0.0,
) -> NDArray[np.float64]:
    """Return smooth Soft-L1 IRLS weights for residual-map least squares."""

    residual_arr = _as_vector_output(residuals)
    scale_value = _as_real_scalar("Soft-L1 scale", scale)
    if scale_value <= 0.0:
        raise ValueError("Soft-L1 scale must be finite and positive")
    min_weight_value = _as_real_scalar("Soft-L1 min_weight", min_weight)
    if min_weight_value < 0.0 or min_weight_value > 1.0:
        raise ValueError("Soft-L1 min_weight must be finite and in [0, 1]")

    scaled = residual_arr / scale_value
    weights = 1.0 / np.sqrt(1.0 + scaled * scaled)
    if min_weight_value > 0.0:
        weights = np.maximum(weights, min_weight_value)
    return weights


def gauss_newton_gradient(
    jacobian: JacobianResult,
    *,
    weights: ArrayLike | None = None,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> NaturalGradientResult:
    """Return the Gauss-Newton-preconditioned least-squares gradient.

    The residual map is read from ``jacobian.value`` and the scalar loss is
    ``0.5 * residual.T @ W @ residual``. The returned ``natural_gradient`` is
    the trainable-subspace solution of ``(J.T @ W @ J + damping * I) @ x =
    J.T @ W @ residual``; subtract it from parameters for a Gauss-Newton
    descent update.
    """

    if not isinstance(jacobian, JacobianResult):
        raise ValueError("gauss-newton gradient requires a JacobianResult")
    jacobian_arr = jacobian.jacobian
    residual = jacobian.value
    if weights is None:
        weighted_residual = residual
    else:
        weight_arr = _as_real_numeric_array("weights", weights)
        if weight_arr.ndim != 1 or weight_arr.shape[0] != residual.size:
            raise ValueError("weights must be a one-dimensional array matching residual rows")
        if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
            raise ValueError("weights must contain only finite non-negative values")
        weighted_residual = residual * weight_arr

    loss_value = 0.5 * float(residual @ weighted_residual)
    gradient = jacobian_arr.T @ weighted_residual
    base_gradient = GradientResult(
        value=loss_value,
        gradient=gradient,
        method=f"gauss_newton:{jacobian.method}",
        shift=None,
        coefficient=None,
        evaluations=jacobian.evaluations,
        parameter_names=jacobian.parameter_names,
        trainable=jacobian.trainable,
    )
    metric = empirical_fisher_metric(jacobian, weights=weights, damping=damping)
    return natural_gradient(base_gradient, metric, damping=0.0, rcond=rcond)


def custom_gauss_newton_gradient(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    weights: ArrayLike | None = None,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> NaturalGradientResult:
    """Return a Gauss-Newton update from an exact custom residual Jacobian."""

    jacobian_result = value_and_custom_jacobian(rule, values, parameters=parameters)
    return gauss_newton_gradient(
        jacobian_result,
        weights=weights,
        damping=damping,
        rcond=rcond,
    )


def levenberg_marquardt_step(
    jacobian: JacobianResult,
    values: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    damping: float = 1.0e-3,
    bounds: Sequence[ParameterBounds] | None = None,
    max_step_norm: float | None = None,
    rcond: float = 1.0e-12,
) -> LevenbergMarquardtStep:
    """Return a bounded Levenberg-Marquardt candidate for residual objectives."""

    current_values = _as_parameter_array(values)
    if current_values.size != jacobian.jacobian.shape[1]:
        raise ValueError("values length must match Jacobian parameter dimension")
    damping_value = _as_real_scalar("Levenberg-Marquardt damping", damping)
    if damping_value < 0.0:
        raise ValueError("Levenberg-Marquardt damping must be finite and non-negative")
    max_step_norm_value = (
        None
        if max_step_norm is None
        else _as_real_scalar("Levenberg-Marquardt max_step_norm", max_step_norm)
    )
    if max_step_norm_value is not None and max_step_norm_value <= 0.0:
        raise ValueError("Levenberg-Marquardt max_step_norm must be finite and positive")

    gauss_newton = gauss_newton_gradient(
        jacobian,
        weights=weights,
        damping=damping_value,
        rcond=rcond,
    )
    step = -gauss_newton.natural_gradient.copy()
    trainable = np.asarray(jacobian.trainable, dtype=bool)
    if max_step_norm_value is not None and np.any(trainable):
        norm = float(np.linalg.norm(step[trainable], ord=2))
        if norm > max_step_norm_value:
            step[trainable] *= max_step_norm_value / norm

    candidate_values = current_values + step
    if bounds is not None:
        candidate_values = _project_bounds(
            candidate_values, _normalise_bounds(current_values, bounds)
        )
        step = candidate_values - current_values

    model_gradient = gauss_newton.base_gradient.gradient
    predicted_reduction = -float(model_gradient @ step + 0.5 * step @ gauss_newton.metric @ step)
    return LevenbergMarquardtStep(
        gauss_newton=gauss_newton,
        step=step,
        candidate_values=candidate_values,
        damping=damping_value,
        predicted_reduction=predicted_reduction,
    )


def custom_levenberg_marquardt_step(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    weights: ArrayLike | None = None,
    damping: float = 1.0e-3,
    bounds: Sequence[ParameterBounds] | None = None,
    max_step_norm: float | None = None,
    rcond: float = 1.0e-12,
) -> LevenbergMarquardtStep:
    """Return an LM candidate using an exact custom residual Jacobian."""

    jacobian_result = value_and_custom_jacobian(rule, values, parameters=parameters)
    return levenberg_marquardt_step(
        jacobian_result,
        values,
        weights=weights,
        damping=damping,
        bounds=bounds,
        max_step_norm=max_step_norm,
        rcond=rcond,
    )


def evaluate_levenberg_marquardt_step(
    objective: VectorObjective,
    step_result: LevenbergMarquardtStep,
    *,
    weights: ArrayLike | None = None,
    acceptance_threshold: float = 1.0e-4,
) -> LevenbergMarquardtTrial:
    """Evaluate actual residual reduction for a Levenberg-Marquardt candidate."""

    threshold = _as_real_scalar("Levenberg-Marquardt acceptance_threshold", acceptance_threshold)
    if threshold < 0.0:
        raise ValueError("Levenberg-Marquardt acceptance_threshold must be non-negative")
    candidate_residual = _as_vector_output(objective(step_result.candidate_values.copy()))
    reference_residual = step_result.gauss_newton.base_gradient.value
    if weights is None:
        candidate_value = 0.5 * float(candidate_residual @ candidate_residual)
    else:
        weight_arr = _as_real_numeric_array("weights", weights)
        if weight_arr.ndim != 1 or weight_arr.shape[0] != candidate_residual.size:
            raise ValueError("weights must be a one-dimensional array matching residual rows")
        if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
            raise ValueError("weights must contain only finite non-negative values")
        candidate_value = 0.5 * float(candidate_residual @ (candidate_residual * weight_arr))
    actual_reduction = reference_residual - candidate_value
    predicted = step_result.predicted_reduction
    reduction_ratio = actual_reduction / predicted if predicted > 0.0 else 0.0
    accepted = predicted > 0.0 and reduction_ratio >= threshold
    return LevenbergMarquardtTrial(
        step_result=step_result,
        candidate_residual=candidate_residual,
        candidate_value=candidate_value,
        actual_reduction=actual_reduction,
        reduction_ratio=reduction_ratio,
        accepted=accepted,
    )


def update_levenberg_marquardt_damping(
    trial: LevenbergMarquardtTrial,
    *,
    decrease_factor: float = 1.0 / 3.0,
    increase_factor: float = 2.0,
    min_damping: float = 1.0e-12,
    max_damping: float = 1.0e12,
    high_quality_ratio: float = 0.75,
) -> LevenbergMarquardtDampingUpdate:
    """Return a bounded trust-region damping update for an LM trial."""

    if not isinstance(trial, LevenbergMarquardtTrial):
        raise ValueError("damping update requires a LevenbergMarquardtTrial")
    decrease = _as_real_scalar("Levenberg-Marquardt decrease_factor", decrease_factor)
    increase = _as_real_scalar("Levenberg-Marquardt increase_factor", increase_factor)
    min_value = _as_real_scalar("Levenberg-Marquardt min_damping", min_damping)
    max_value = _as_real_scalar("Levenberg-Marquardt max_damping", max_damping)
    high_quality = _as_real_scalar(
        "Levenberg-Marquardt high_quality_ratio",
        high_quality_ratio,
    )
    if not 0.0 < decrease < 1.0:
        raise ValueError("decrease_factor must be finite and between 0 and 1")
    if increase <= 1.0:
        raise ValueError("increase_factor must be finite and greater than 1")
    if min_value < 0.0:
        raise ValueError("min_damping must be finite and non-negative")
    if max_value < min_value:
        raise ValueError("max_damping must be greater than or equal to min_damping")
    if high_quality < 0.0:
        raise ValueError("high_quality_ratio must be finite and non-negative")

    current = trial.step_result.damping
    if not trial.accepted:
        return LevenbergMarquardtDampingUpdate(
            trial=trial,
            next_damping=min(max_value, max(min_value, current * increase)),
            action="reject_increase",
        )
    if trial.reduction_ratio >= high_quality:
        return LevenbergMarquardtDampingUpdate(
            trial=trial,
            next_damping=min(max_value, max(min_value, current * decrease)),
            action="accept_decrease",
        )
    return LevenbergMarquardtDampingUpdate(
        trial=trial,
        next_damping=min(max_value, max(min_value, current)),
        action="accept_keep",
    )


def natural_gradient(
    gradient_result: GradientResult,
    metric: ArrayLike,
    *,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> NaturalGradientResult:
    """Solve ``metric @ natural_gradient = gradient`` on trainable parameters."""

    metric_arr = _as_real_numeric_array("natural-gradient metric", metric)
    if metric_arr.ndim != 2 or metric_arr.shape != (
        gradient_result.gradient.size,
        gradient_result.gradient.size,
    ):
        raise ValueError("natural-gradient metric must have shape (n_parameters, n_parameters)")
    if not np.all(np.isfinite(metric_arr)):
        raise ValueError("natural-gradient metric must contain only finite values")
    if not np.allclose(metric_arr, metric_arr.T, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("natural-gradient metric must be symmetric")
    damping_value = _as_real_scalar("natural-gradient damping", damping)
    if damping_value < 0.0:
        raise ValueError("natural-gradient damping must be finite and non-negative")
    rcond_value = _as_real_scalar("natural-gradient rcond", rcond)
    if rcond_value <= 0.0:
        raise ValueError("natural-gradient rcond must be finite and positive")

    trainable = np.asarray(gradient_result.trainable, dtype=bool)
    result = np.zeros_like(gradient_result.gradient)
    if not np.any(trainable):
        return NaturalGradientResult(
            base_gradient=gradient_result,
            metric=metric_arr,
            natural_gradient=result,
            damping=damping_value,
            condition_number=1.0,
        )

    active_metric = metric_arr[np.ix_(trainable, trainable)].copy()
    if damping_value > 0.0:
        active_metric += damping_value * np.eye(active_metric.shape[0], dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(active_metric)
    min_eigenvalue = float(np.min(eigenvalues))
    max_eigenvalue = float(np.max(eigenvalues))
    if min_eigenvalue <= 0.0:
        raise ValueError(
            "natural-gradient metric must be positive definite on trainable parameters"
        )
    condition_number = max_eigenvalue / min_eigenvalue
    if condition_number > 1.0 / rcond_value:
        raise ValueError("natural-gradient metric is ill-conditioned")
    result[trainable] = np.linalg.solve(active_metric, gradient_result.gradient[trainable])
    return NaturalGradientResult(
        base_gradient=gradient_result,
        metric=metric_arr,
        natural_gradient=result,
        damping=damping_value,
        condition_number=condition_number,
    )


def implicit_stationary_sensitivity(
    hessian: ArrayLike,
    cross_derivative: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    hyperparameter_names: Sequence[str] | None = None,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> ImplicitSensitivityResult:
    """Return ``dx*/dalpha = -H^-1 B`` for an implicit stationary system."""

    hessian_arr = _as_real_numeric_array("implicit hessian", hessian)
    cross = _as_real_numeric_array("implicit cross_derivative", cross_derivative)
    if hessian_arr.ndim != 2 or hessian_arr.shape[0] != hessian_arr.shape[1]:
        raise ValueError("implicit hessian must be a square matrix")
    if cross.ndim == 1:
        cross = cross.reshape((-1, 1))
    if cross.ndim != 2 or cross.shape[0] != hessian_arr.shape[0]:
        raise ValueError("implicit cross_derivative row count must match hessian dimension")
    if not np.all(np.isfinite(hessian_arr)) or not np.all(np.isfinite(cross)):
        raise ValueError("implicit operands must contain only finite values")
    if not np.allclose(hessian_arr, hessian_arr.T, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("implicit hessian must be symmetric")
    damping_value = _as_real_scalar("implicit damping", damping)
    if damping_value < 0.0:
        raise ValueError("implicit damping must be finite and non-negative")
    rcond_value = _as_real_scalar("implicit rcond", rcond)
    if rcond_value <= 0.0:
        raise ValueError("implicit rcond must be finite and positive")
    parameter_values = np.zeros(hessian_arr.shape[0], dtype=np.float64)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    hyper_names = (
        tuple(f"alpha{index}" for index in range(cross.shape[1]))
        if hyperparameter_names is None
        else tuple(hyperparameter_names)
    )
    if len(hyper_names) != cross.shape[1]:
        raise ValueError("hyperparameter_names length must match cross_derivative columns")
    trainable = np.asarray([parameter.trainable for parameter in parameter_meta], dtype=bool)
    sensitivity = np.zeros_like(cross)
    condition_number = 1.0
    if np.any(trainable):
        active_hessian = hessian_arr[np.ix_(trainable, trainable)].copy()
        if damping_value > 0.0:
            active_hessian += damping_value * np.eye(active_hessian.shape[0], dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(active_hessian)
        min_eigenvalue = float(np.min(eigenvalues))
        max_eigenvalue = float(np.max(eigenvalues))
        if min_eigenvalue <= 0.0:
            raise ValueError("implicit hessian must be positive definite on trainable parameters")
        condition_number = max_eigenvalue / min_eigenvalue
        if condition_number > 1.0 / rcond_value:
            raise ValueError("implicit hessian is ill-conditioned")
        sensitivity[trainable, :] = -np.linalg.solve(active_hessian, cross[trainable, :])
    return ImplicitSensitivityResult(
        sensitivity=sensitivity,
        hessian=hessian_arr,
        cross_derivative=cross,
        damping=damping_value,
        condition_number=condition_number,
        method="implicit_stationary_sensitivity",
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        hyperparameter_names=hyper_names,
    )


def implicit_fixed_point_sensitivity(
    state_jacobian: ArrayLike,
    parameter_jacobian: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    hyperparameter_names: Sequence[str] | None = None,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> FixedPointSensitivityResult:
    """Return ``dx*/dalpha`` for ``x* = T(x*, alpha)`` fixed-point maps."""

    state = _as_real_numeric_array("fixed-point state_jacobian", state_jacobian)
    parameter = _as_real_numeric_array("fixed-point parameter_jacobian", parameter_jacobian)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("fixed-point state_jacobian must be a square matrix")
    if parameter.ndim == 1:
        parameter = parameter.reshape((-1, 1))
    if parameter.ndim != 2 or parameter.shape[0] != state.shape[0]:
        raise ValueError("fixed-point parameter_jacobian row count must match state dimension")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(parameter)):
        raise ValueError("fixed-point operands must contain only finite values")
    damping_value = _as_real_scalar("fixed-point damping", damping)
    if damping_value < 0.0:
        raise ValueError("fixed-point damping must be finite and non-negative")
    rcond_value = _as_real_scalar("fixed-point rcond", rcond)
    if rcond_value <= 0.0:
        raise ValueError("fixed-point rcond must be finite and positive")
    parameter_values = np.zeros(state.shape[0], dtype=np.float64)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    hyper_names = (
        tuple(f"alpha{index}" for index in range(parameter.shape[1]))
        if hyperparameter_names is None
        else tuple(hyperparameter_names)
    )
    if len(hyper_names) != parameter.shape[1]:
        raise ValueError(
            "hyperparameter_names length must match fixed-point parameter_jacobian columns"
        )
    system_matrix = np.eye(state.shape[0], dtype=np.float64) - state
    if damping_value > 0.0:
        system_matrix = system_matrix + damping_value * np.eye(state.shape[0], dtype=np.float64)
    trainable = np.asarray(
        [parameter_info.trainable for parameter_info in parameter_meta], dtype=bool
    )
    sensitivity = np.zeros_like(parameter)
    condition_number = 1.0
    if np.any(trainable):
        active_system = system_matrix[np.ix_(trainable, trainable)]
        condition_number = float(np.linalg.cond(active_system))
        if not np.isfinite(condition_number) or condition_number > 1.0 / rcond_value:
            raise ValueError("fixed-point system is ill-conditioned")
        sensitivity[trainable, :] = np.linalg.solve(active_system, parameter[trainable, :])
    return FixedPointSensitivityResult(
        sensitivity=sensitivity,
        state_jacobian=state,
        parameter_jacobian=parameter,
        system_matrix=system_matrix,
        damping=damping_value,
        condition_number=condition_number,
        method="implicit_fixed_point_sensitivity",
        parameter_names=tuple(parameter_info.name for parameter_info in parameter_meta),
        trainable=tuple(parameter_info.trainable for parameter_info in parameter_meta),
        hyperparameter_names=hyper_names,
    )


def check_parameter_shift_consistency(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    finite_difference_step: float = 1.0e-6,
    tolerance: float = 1.0e-5,
) -> GradientCheckResult:
    """Compare parameter-shift gradients against central finite differences."""

    tolerance_value = _as_real_scalar("gradient check tolerance", tolerance)
    if tolerance_value < 0.0:
        raise ValueError("gradient check tolerance must be finite and non-negative")
    candidate = value_and_parameter_shift_grad(
        objective,
        values,
        parameters=parameters,
        rule=rule,
    )
    reference = value_and_finite_difference_grad(
        objective,
        values,
        parameters=parameters,
        step=finite_difference_step,
    )
    delta = candidate.gradient - reference.gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta, ord=2))
    value_delta = float(abs(candidate.value - reference.value))
    return GradientCheckResult(
        reference=reference,
        candidate=candidate,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        value_delta=value_delta,
        tolerance=tolerance_value,
        passed=max_abs_error <= tolerance_value,
    )


def is_jax_autodiff_available() -> bool:
    """Return whether JAX autodiff can be imported in the active environment."""

    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
    except ImportError:
        return False
    return True


def jax_value_and_grad(
    objective: Callable[[Any], Any],
    values: ArrayLike,
) -> tuple[float, NDArray[np.float64]]:
    """Evaluate a JAX scalar objective and return ``(value, gradient)``."""

    parameter_values = _as_parameter_array(values)

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError("JAX autodiff is unavailable; install the [jax] extra") from exc

    def wrapped(raw_values: Any) -> Any:
        return objective(raw_values)

    value, gradient = jax.value_and_grad(wrapped)(jnp.asarray(parameter_values))
    result_value = _as_real_scalar("JAX objective value", value)
    result_gradient = _as_real_numeric_array("JAX gradient", gradient)
    if result_gradient.shape != parameter_values.shape:
        raise ValueError("JAX gradient shape must match parameter shape")
    if not np.all(np.isfinite(result_gradient)):
        raise ValueError("JAX gradient must contain only finite values")
    return result_value, result_gradient


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
