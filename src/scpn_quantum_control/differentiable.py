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

import ast
import dis
import inspect
import json
import linecache
import math
import sys
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

ScalarObjective = Callable[[NDArray[np.float64]], float | int | np.floating[Any]]
VectorObjective = Callable[[NDArray[np.float64]], ArrayLike]
ComplexStepObjective = Callable[[NDArray[np.complex128]], object]
CustomJVPRule = Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]
CustomVJPRule = Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]
VMapInAxes = int | None | Sequence[int | None]
PrimitiveBatchingRule = Callable[
    [Callable[..., object], tuple[object, ...], tuple[int | None, ...], int], object
]


@dataclass(frozen=True)
class WholeProgramTraceEvent:
    """One executed Python source line observed during whole-program AD tracing."""

    filename: str
    function_name: str
    line_number: int
    source: str

    def __post_init__(self) -> None:
        if not self.filename:
            raise ValueError("trace event filename must be non-empty")
        if not self.function_name:
            raise ValueError("trace event function_name must be non-empty")
        if self.line_number <= 0:
            raise ValueError("trace event line_number must be positive")
        object.__setattr__(self, "source", str(self.source).strip())


@dataclass(frozen=True)
class WholeProgramIRNode:
    """One operator-intercepted IR node from whole-program AD."""

    index: int
    op: str
    inputs: tuple[str, ...]
    value: float
    tangent: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("IR node index must be non-negative")
        if not self.op:
            raise ValueError("IR node op must be non-empty")
        if any(not isinstance(item, str) or not item for item in self.inputs):
            raise ValueError("IR node inputs must be non-empty strings")
        value = _as_real_scalar("IR node value", self.value)
        tangent = _as_real_numeric_array("IR node tangent", self.tangent)
        if tangent.ndim != 1:
            raise ValueError("IR node tangent must be one-dimensional")
        if not np.all(np.isfinite(tangent)):
            raise ValueError("IR node tangent must contain finite values")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "tangent", tangent)


@dataclass(frozen=True)
class ProgramADSSAValue:
    """One versioned SSA value emitted by program AD graph capture."""

    name: str
    producer: int | None
    version: int
    shape: tuple[int, ...]
    dtype: str
    effect: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("program AD SSA value name must be a non-empty string")
        if self.producer is not None and self.producer < 0:
            raise ValueError("program AD SSA value producer must be non-negative or None")
        if self.version < 0:
            raise ValueError("program AD SSA value version must be non-negative")
        if any(not isinstance(dimension, int) or dimension < 0 for dimension in self.shape):
            raise ValueError("program AD SSA value shape dimensions must be non-negative ints")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("program AD SSA value dtype must be a non-empty string")
        if self.effect is not None and self.effect < 0:
            raise ValueError("program AD SSA value effect must be non-negative or None")


@dataclass(frozen=True)
class ProgramADEffect:
    """One ordered effect or pure operation in program AD graph capture."""

    index: int
    kind: str
    target: str
    inputs: tuple[str, ...]
    version: int
    ordering: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("program AD effect index must be non-negative")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("program AD effect kind must be a non-empty string")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("program AD effect target must be a non-empty string")
        if any(not isinstance(item, str) or not item for item in self.inputs):
            raise ValueError("program AD effect inputs must be non-empty strings")
        if self.version < 0:
            raise ValueError("program AD effect version must be non-negative")
        if self.ordering < 0:
            raise ValueError("program AD effect ordering must be non-negative")


@dataclass(frozen=True)
class ProgramADAliasEdge:
    """One alias or mutation-version edge in program AD graph capture."""

    source: str
    target: str
    kind: str
    version: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("program AD alias source must be a non-empty string")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("program AD alias target must be a non-empty string")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("program AD alias kind must be a non-empty string")
        if self.version < 0:
            raise ValueError("program AD alias version must be non-negative")


@dataclass(frozen=True)
class ProgramADControlRegion:
    """One source or runtime control-flow region in program AD graph capture."""

    index: int
    kind: str
    predicate: str | None
    entered: bool
    source_line: int | None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("program AD control region index must be non-negative")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("program AD control region kind must be a non-empty string")
        if self.predicate is not None and (
            not isinstance(self.predicate, str) or not self.predicate
        ):
            raise ValueError("program AD control region predicate must be non-empty or None")
        if not isinstance(self.entered, bool):
            raise ValueError("program AD control region entered must be a boolean")
        if self.source_line is not None and self.source_line <= 0:
            raise ValueError("program AD control region source_line must be positive or None")


@dataclass(frozen=True)
class ProgramADEffectIR:
    """Deterministic SSA/effect IR emitted by program AD graph capture."""

    ssa_values: tuple[ProgramADSSAValue, ...]
    effects: tuple[ProgramADEffect, ...]
    alias_edges: tuple[ProgramADAliasEdge, ...]
    control_regions: tuple[ProgramADControlRegion, ...]
    serialization: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, ProgramADSSAValue) for value in self.ssa_values):
            raise ValueError("program AD IR ssa_values must contain ProgramADSSAValue entries")
        if any(not isinstance(effect, ProgramADEffect) for effect in self.effects):
            raise ValueError("program AD IR effects must contain ProgramADEffect entries")
        if any(not isinstance(edge, ProgramADAliasEdge) for edge in self.alias_edges):
            raise ValueError("program AD IR alias_edges must contain ProgramADAliasEdge entries")
        if any(not isinstance(region, ProgramADControlRegion) for region in self.control_regions):
            raise ValueError(
                "program AD IR control_regions must contain ProgramADControlRegion entries"
            )
        if not isinstance(self.serialization, str) or not self.serialization:
            raise ValueError("program AD IR serialization must be a non-empty string")


@dataclass(frozen=True)
class ProgramADAdjointResult:
    """Reverse-mode adjoint replay result for a captured program AD graph."""

    gradient: NDArray[np.float64]
    supported: bool
    unsupported_ops: tuple[str, ...]
    method: str
    claim_boundary: str

    def __post_init__(self) -> None:
        gradient = _as_real_numeric_array("program AD adjoint gradient", self.gradient)
        if gradient.ndim != 1:
            raise ValueError("program AD adjoint gradient must be one-dimensional")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("program AD adjoint gradient must contain finite values")
        if not isinstance(self.supported, bool):
            raise ValueError("program AD adjoint supported must be a boolean")
        if any(not isinstance(op, str) or not op for op in self.unsupported_ops):
            raise ValueError("program AD adjoint unsupported_ops must be non-empty strings")
        if self.supported and self.unsupported_ops:
            raise ValueError("program AD adjoint cannot be supported with unsupported ops")
        if not self.method:
            raise ValueError("program AD adjoint method must be non-empty")
        if not self.claim_boundary:
            raise ValueError("program AD adjoint claim_boundary must be non-empty")
        object.__setattr__(self, "gradient", gradient)


@dataclass(frozen=True)
class WholeProgramBytecodeInstruction:
    """One Python bytecode instruction captured for whole-program AD frontend IR."""

    offset: int
    opname: str
    argrepr: str
    line_number: int | None

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("bytecode instruction offset must be non-negative")
        if not self.opname:
            raise ValueError("bytecode instruction opname must be non-empty")
        if not isinstance(self.argrepr, str):
            raise ValueError("bytecode instruction argrepr must be a string")
        if self.line_number is not None and self.line_number <= 0:
            raise ValueError("bytecode instruction line_number must be positive or None")


@dataclass(frozen=True)
class WholeProgramSourceIRFeature:
    """One source-level semantic feature captured for whole-program AD."""

    kind: str
    detail: str
    line_number: int

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("source IR feature kind must be non-empty")
        if not self.detail:
            raise ValueError("source IR feature detail must be non-empty")
        if self.line_number <= 0:
            raise ValueError("source IR feature line_number must be positive")


@dataclass(frozen=True)
class WholeProgramSemanticsReport:
    """Static semantics summary for whole-program AD graph capture."""

    bytecode_frontend: bool
    source_frontend: bool
    graph_capture: bool
    aliasing_observed: bool
    mutation_observed: bool
    loop_observed: bool
    control_flow_observed: bool
    numpy_observed: bool
    differentiation_semantics: str

    def __post_init__(self) -> None:
        for name in (
            "bytecode_frontend",
            "source_frontend",
            "graph_capture",
            "aliasing_observed",
            "mutation_observed",
            "loop_observed",
            "control_flow_observed",
            "numpy_observed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if not self.differentiation_semantics:
            raise ValueError("differentiation_semantics must be non-empty")


@dataclass(frozen=True)
class WholeProgramADResult:
    """Value, gradient, execution trace, and polyglot AD lowering status."""

    value: float
    gradient: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    trace_events: tuple[WholeProgramTraceEvent, ...]
    source: str | None
    control_flow_observed: bool
    numpy_observed: bool
    polyglot_targets: dict[str, str]
    claim_boundary: str
    ir_nodes: tuple[WholeProgramIRNode, ...] = ()
    bytecode_instructions: tuple[WholeProgramBytecodeInstruction, ...] = ()
    source_ir_features: tuple[WholeProgramSourceIRFeature, ...] = ()
    semantics_report: WholeProgramSemanticsReport | None = None
    program_ir: ProgramADEffectIR | None = None
    adjoint_result: ProgramADAdjointResult | None = None

    def __post_init__(self) -> None:
        value = _as_real_scalar("whole-program AD value", self.value)
        gradient = _as_real_numeric_array("whole-program AD gradient", self.gradient)
        if gradient.ndim != 1:
            raise ValueError("whole-program AD gradient must be one-dimensional")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("whole-program AD gradient must contain finite values")
        step = _as_real_scalar("whole-program AD step", self.step)
        if step < 0.0:
            raise ValueError("whole-program AD step must be non-negative")
        if self.evaluations < 1:
            raise ValueError("whole-program AD evaluations must be positive")
        if len(self.parameter_names) != gradient.size:
            raise ValueError("parameter_names length must match gradient length")
        if len(self.trainable) != gradient.size:
            raise ValueError("trainable mask length must match gradient length")
        if any(not isinstance(event, WholeProgramTraceEvent) for event in self.trace_events):
            raise ValueError("trace_events must contain WholeProgramTraceEvent entries")
        if any(not isinstance(node, WholeProgramIRNode) for node in self.ir_nodes):
            raise ValueError("ir_nodes must contain WholeProgramIRNode entries")
        if any(
            not isinstance(instruction, WholeProgramBytecodeInstruction)
            for instruction in self.bytecode_instructions
        ):
            raise ValueError(
                "bytecode_instructions must contain WholeProgramBytecodeInstruction entries"
            )
        if any(
            not isinstance(feature, WholeProgramSourceIRFeature)
            for feature in self.source_ir_features
        ):
            raise ValueError("source_ir_features must contain WholeProgramSourceIRFeature entries")
        if self.semantics_report is not None and not isinstance(
            self.semantics_report, WholeProgramSemanticsReport
        ):
            raise ValueError("semantics_report must be a WholeProgramSemanticsReport or None")
        if self.program_ir is not None and not isinstance(self.program_ir, ProgramADEffectIR):
            raise ValueError("program_ir must be a ProgramADEffectIR or None")
        if self.adjoint_result is not None and not isinstance(
            self.adjoint_result, ProgramADAdjointResult
        ):
            raise ValueError("adjoint_result must be a ProgramADAdjointResult or None")
        if (
            self.adjoint_result is not None
            and self.adjoint_result.gradient.shape != gradient.shape
        ):
            raise ValueError("adjoint_result gradient shape must match forward gradient shape")
        if not isinstance(self.control_flow_observed, bool):
            raise ValueError("control_flow_observed must be a boolean")
        if not isinstance(self.numpy_observed, bool):
            raise ValueError("numpy_observed must be a boolean")
        if not self.polyglot_targets:
            raise ValueError("polyglot_targets must be non-empty")
        if any(not key or not value for key, value in self.polyglot_targets.items()):
            raise ValueError("polyglot target names and status values must be non-empty")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "step", step)


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
    context = _WholeProgramTraceContext(parameter_values.size)
    traced_values: list[TraceADScalar] = []
    for index, (value, parameter) in enumerate(zip(parameter_values, parameter_meta, strict=True)):
        tangent = np.zeros(parameter_values.size, dtype=np.float64)
        if parameter.trainable:
            tangent[index] = 1.0
        traced_values.append(context.make("parameter", (parameter.name,), float(value), tangent))
    raw = objective(TraceADArray(tuple(traced_values), (len(traced_values),), context))
    if isinstance(raw, TraceADArray):
        if raw.shape != ():
            raise ValueError("whole-program objective must return a whole-program AD scalar")
        raw = raw.item()
    if not isinstance(raw, TraceADScalar):
        raise ValueError("whole-program objective must return a whole-program AD scalar")
    source = _objective_source(objective)
    trace_events = (
        _trace_whole_program_objective(cast(ScalarObjective, objective), parameter_values)
        if trace
        else ()
    )
    bytecode_instructions = _objective_bytecode(objective)
    source_ir_features = _source_ir_features(source)
    semantics_report = _whole_program_semantics_report(
        bytecode_instructions=bytecode_instructions,
        source_ir_features=source_ir_features,
        trace_events=trace_events,
        source=source,
        numpy_observed=_source_mentions_numpy(source)
        or any(node.op in {"sin", "cos", "exp", "log"} for node in context.nodes),
        differentiation_semantics=(
            "operator-intercepted exact forward AD over the executed Python program; "
            "loops, branches, local aliasing, list mutation, and supported NumPy scalar "
            "ufuncs execute with derivative-carrying values, while unsupported "
            "derivative-losing operations fail closed"
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
            "loops, local aliasing, list mutation, supported NumPy scalar ufuncs, and "
            "executed-branch control flow with deterministic SSA/effect IR evidence; no "
            "finite-difference fallback and no executable Rust, LLVM, or JIT AD lowering claim"
        ),
        bytecode_instructions=bytecode_instructions,
        source_ir_features=source_ir_features,
        semantics_report=semantics_report,
        program_ir=program_ir,
        adjoint_result=adjoint_result,
    )


class _WholeProgramTraceContext:
    """Mutable builder for whole-program AD IR nodes."""

    def __init__(self, parameter_count: int) -> None:
        self.parameter_count = parameter_count
        self.nodes: list[WholeProgramIRNode] = []
        self.ssa_values: list[ProgramADSSAValue] = []
        self.effects: list[ProgramADEffect] = []
        self.alias_edges: list[ProgramADAliasEdge] = []
        self.control_regions: list[ProgramADControlRegion] = []
        self._value_versions: dict[str, int] = {}
        self._effect_order = 0

    def make(
        self,
        op: str,
        inputs: tuple[str, ...],
        value: float,
        tangent: NDArray[np.float64],
    ) -> TraceADScalar:
        """Create a trace scalar and append its IR node to this AD context."""
        node = WholeProgramIRNode(
            index=len(self.nodes),
            op=op,
            inputs=inputs,
            value=value,
            tangent=tangent.copy(),
        )
        self.nodes.append(node)
        name = f"%{node.index}"
        version = self._next_value_version(name)
        effect = ProgramADEffect(
            index=len(self.effects),
            kind=self._effect_kind(op),
            target=name,
            inputs=inputs,
            version=version,
            ordering=self._effect_order,
        )
        self._effect_order += 1
        self.effects.append(effect)
        self.ssa_values.append(
            ProgramADSSAValue(
                name=name,
                producer=node.index,
                version=version,
                shape=(),
                dtype="float64",
                effect=effect.index,
            )
        )
        if op.startswith("mutation:"):
            target = inputs[0] if inputs else name
            self.alias_edges.append(
                ProgramADAliasEdge(
                    source=target,
                    target=name,
                    kind="mutation_version",
                    version=version,
                )
            )
        if op.startswith("branch:"):
            self.control_regions.append(
                ProgramADControlRegion(
                    index=len(self.control_regions),
                    kind="runtime_branch",
                    predicate=op,
                    entered=bool(value),
                    source_line=None,
                )
            )
        return TraceADScalar(node.value, node.tangent, self, name)

    def program_ir(
        self,
        *,
        source_ir_features: tuple[WholeProgramSourceIRFeature, ...],
        bytecode_instructions: tuple[WholeProgramBytecodeInstruction, ...],
    ) -> ProgramADEffectIR:
        """Build deterministic SSA/effect IR metadata from captured program AD evidence."""

        alias_edges = list(self.alias_edges)
        control_regions = list(self.control_regions)
        for feature in source_ir_features:
            if "alias" in feature.kind:
                alias_edges.append(
                    ProgramADAliasEdge(
                        source=feature.detail,
                        target=f"source:{feature.line_number}",
                        kind=feature.kind,
                        version=len(alias_edges),
                    )
                )
            if any(token in feature.kind for token in ("branch", "control", "loop")):
                control_regions.append(
                    ProgramADControlRegion(
                        index=len(control_regions),
                        kind=f"source_{feature.kind}",
                        predicate=feature.detail,
                        entered=True,
                        source_line=feature.line_number,
                    )
                )
        payload = {
            "format": "program_ad_effect_ir.v1",
            "ssa_values": [
                {
                    "name": value.name,
                    "producer": value.producer,
                    "version": value.version,
                    "shape": value.shape,
                    "dtype": value.dtype,
                    "effect": value.effect,
                }
                for value in self.ssa_values
            ],
            "effects": [
                {
                    "index": effect.index,
                    "kind": effect.kind,
                    "target": effect.target,
                    "inputs": effect.inputs,
                    "version": effect.version,
                    "ordering": effect.ordering,
                }
                for effect in self.effects
            ],
            "alias_edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "kind": edge.kind,
                    "version": edge.version,
                }
                for edge in alias_edges
            ],
            "control_regions": [
                {
                    "index": region.index,
                    "kind": region.kind,
                    "predicate": region.predicate,
                    "entered": region.entered,
                    "source_line": region.source_line,
                }
                for region in control_regions
            ],
            "bytecode_offsets": tuple(instruction.offset for instruction in bytecode_instructions),
        }
        return ProgramADEffectIR(
            ssa_values=tuple(self.ssa_values),
            effects=tuple(self.effects),
            alias_edges=tuple(alias_edges),
            control_regions=tuple(control_regions),
            serialization=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    def _next_value_version(self, name: str) -> int:
        version = self._value_versions.get(name, -1) + 1
        self._value_versions[name] = version
        return version

    @staticmethod
    def _effect_kind(op: str) -> str:
        if op == "parameter":
            return "parameter"
        if op.startswith("branch:"):
            return "control_branch"
        if op.startswith("mutation:"):
            return "mutation"
        if op in {
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
            "abs",
            "clip",
            "where",
        }:
            return "primitive"
        return "pure"


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
    ) -> None:
        if not shape:
            if len(items) != 1:
                raise ValueError("scalar TraceADArray requires exactly one item")
        elif int(np.prod(shape)) != len(items):
            raise ValueError("TraceADArray shape must match item count")
        if any(item.context is not context for item in items):
            raise ValueError("TraceADArray items must belong to the same trace")
        self._items = list(items)
        self.shape = shape
        self.context = context

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
                    tuple(self._items[row * cols : (row + 1) * cols]), (cols,), self.context
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

        return TraceADArray(tuple(self._items), self.shape, self.context)

    def reshape(self, *shape: int | tuple[int, ...]) -> TraceADArray:
        """Return a derivative-preserving reshaped array view."""

        if len(shape) == 1 and isinstance(shape[0], tuple):
            raw_target: object = shape[0]
        else:
            raw_target = shape
        _require_program_ad_shape_contract("reshape", (self, raw_target))
        target = _normalise_trace_reshape_shape(raw_target, self.size)
        return TraceADArray(tuple(self._items), target, self.context)

    def ravel(self) -> TraceADArray:
        """Return a flat view-preserving program AD array."""

        _require_program_ad_shape_contract("ravel", (self,))
        return TraceADArray(tuple(self._items), (self.size,), self.context)

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

        return _trace_variance(self, axis=axis, ddof=ddof)

    def std(self, axis: int | None = None, ddof: int = 0) -> TraceADScalar | TraceADArray:
        """Return a derivative-preserving standard deviation."""

        return _trace_std(self, axis=axis, ddof=ddof)

    def max(self, axis: int | None = None) -> TraceADScalar | TraceADArray:
        """Return a derivative-preserving maximum with tie-safe semantics."""

        return _trace_extreme(self, axis=axis, choose_max=True)

    def min(self, axis: int | None = None) -> TraceADScalar | TraceADArray:
        """Return a derivative-preserving minimum with tie-safe semantics."""

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

        del axis
        _raise_index_selection_boundary()

    def argmin(self, axis: int | None = None) -> NoReturn:
        """Reject nondifferentiable minimum-index selection."""

        del axis
        _raise_index_selection_boundary()

    def __getitem__(self, index: object) -> TraceADScalar | TraceADArray:
        return _trace_array_getitem(self, index)

    def __setitem__(self, index: object, value: object) -> None:
        if self.ndim > 2:
            raise ValueError("whole-program AD array mutation supports arrays with rank <= 2")
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
            raise ValueError("whole-program AD array mutation supports integer indices")
        scalar = _coerce_trace_scalar(value, self.context)
        self.context.make(
            "mutation:setitem",
            (f"%array[{flat_index}]", scalar.name),
            scalar.primal,
            scalar.tangent,
        )
        self._items[flat_index] = scalar

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
    ) -> TraceADScalar | TraceADArray | list[TraceADArray]:
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
        if func in {np.zeros_like, np.ones_like}:
            if len(args) != 1:
                raise ValueError("program AD like-constructors require one reference array")
            _validate_trace_like_constructor_kwargs(kwargs)
            fill_value = 0.0 if func is np.zeros_like else 1.0
            return _trace_like_constant(args[0], fill_value, self.context)
        if func is np.full_like:
            if len(args) != 2:
                raise ValueError("program AD full_like requires reference array and fill value")
            _validate_trace_like_constructor_kwargs(kwargs)
            return _trace_like_constant(args[0], args[1], self.context)
        if func is np.mean:
            if len(args) != 1 or kwargs.keys() - {"axis"}:
                raise ValueError("whole-program AD np.mean supports one array and optional axis")
            return _coerce_trace_array(args[0], self.context).mean(
                axis=cast(int | None, kwargs.get("axis"))
            )
        if func is np.var:
            if len(args) != 1 or kwargs.keys() - {"axis", "ddof"}:
                raise ValueError("program AD np.var supports one array, axis, and ddof")
            return _trace_variance(
                _coerce_trace_array(args[0], self.context),
                axis=cast(int | None, kwargs.get("axis")),
                ddof=kwargs.get("ddof", 0),
            )
        if func is np.std:
            if len(args) != 1 or kwargs.keys() - {"axis", "ddof"}:
                raise ValueError("program AD np.std supports one array, axis, and ddof")
            return _trace_std(
                _coerce_trace_array(args[0], self.context),
                axis=cast(int | None, kwargs.get("axis")),
                ddof=kwargs.get("ddof", 0),
            )
        if func is np.max:
            if len(args) != 1 or kwargs.keys() - {"axis"}:
                raise ValueError("program AD np.max supports one array and optional axis")
            return _trace_extreme(
                _coerce_trace_array(args[0], self.context),
                axis=cast(int | None, kwargs.get("axis")),
                choose_max=True,
            )
        if func is np.min:
            if len(args) != 1 or kwargs.keys() - {"axis"}:
                raise ValueError("program AD np.min supports one array and optional axis")
            return _trace_extreme(
                _coerce_trace_array(args[0], self.context),
                axis=cast(int | None, kwargs.get("axis")),
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
            return _broadcast_trace_array(
                args[0],
                _normalise_trace_broadcast_shape(args[1]),
                self.context,
            )
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
        if func is np.clip:
            if len(args) < 3 or len(args) > 4 or kwargs:
                raise ValueError("whole-program AD np.clip supports array, lower, and upper")
            return _trace_clip(args[0], args[1], args[2], self.context)
        if func is np.linalg.norm:
            if len(args) != 1 or kwargs.keys() - {"ord", "axis"}:
                raise ValueError(
                    "whole-program AD np.linalg.norm supports one array and optional ord/axis"
                )
            return _trace_norm(
                args[0],
                self.context,
                ord_value=kwargs.get("ord"),
                axis=cast(int | None, kwargs.get("axis")),
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
        if func in {
            np.linalg.eig,
            np.linalg.eigh,
            np.linalg.eigvals,
            np.linalg.eigvalsh,
            np.linalg.svd,
            np.linalg.pinv,
        }:
            _raise_spectral_linalg_boundary(func.__name__)
        if func in {np.argmax, np.argmin}:
            _raise_index_selection_boundary()
        if func is np.sort or func is np.argsort:
            raise ValueError(
                "whole-program AD sort/argsort selection semantics are not differentiable "
                "without an explicit primitive policy"
            )
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
) -> TraceADArray:
    array = _coerce_trace_array(reference, context)
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
    items = tuple(array._items[int(item)] for item in selected_array.reshape(-1))
    return TraceADArray(
        items,
        tuple(int(dimension) for dimension in selected_array.shape),
        array.context,
    )


def _normalise_shape_transform_axes(
    name: str, axis: int | tuple[int, ...], *, output_rank: int
) -> tuple[int, ...]:
    axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    if not axes:
        raise ValueError(f"program AD {name} requires at least one axis")
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
    if axis is None:
        target_shape = tuple(dimension for dimension in array.shape if dimension != 1)
        return TraceADArray(tuple(array._items), target_shape, array.context)
    axes = _normalise_shape_transform_axes("squeeze", axis, output_rank=array.ndim)
    for item in axes:
        if array.shape[item] != 1:
            raise ValueError("program AD squeeze axis must have length one")
    target_shape = tuple(
        dimension for index, dimension in enumerate(array.shape) if index not in axes
    )
    return TraceADArray(tuple(array._items), target_shape, array.context)


def _trace_expand_dims(array: TraceADArray, *, axis: int | tuple[int, ...]) -> TraceADArray:
    axis_tuple = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    output_rank = array.ndim + len(axis_tuple)
    axes = _normalise_shape_transform_axes("expand_dims", axis_tuple, output_rank=output_rank)
    shape = list(array.shape)
    for item in axes:
        shape.insert(item, 1)
    return TraceADArray(tuple(array._items), tuple(shape), array.context)


def _trace_atleast_nd(array: TraceADArray, *, rank: int) -> TraceADArray:
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
    else:
        raise ValueError("program AD atleast rank must be 1, 2, or 3")
    return TraceADArray(tuple(array._items), shape, array.context)


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
    first = _normalise_axis_permutation_axis("swapaxes", axis1, rank=array.ndim)
    second = _normalise_axis_permutation_axis("swapaxes", axis2, rank=array.ndim)
    source = np.arange(array.size, dtype=np.int64).reshape(array.shape)
    moved = np.swapaxes(source, first, second)
    return TraceADArray(
        tuple(array._items[int(index)] for index in moved.reshape(-1)),
        tuple(map(int, moved.shape)),
        array.context,
    )


def _trace_moveaxis(
    array: TraceADArray,
    *,
    source: int | tuple[int, ...],
    destination: int | tuple[int, ...],
) -> TraceADArray:
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
    return TraceADArray(
        tuple(array._items[int(index)] for index in moved_indices.reshape(-1)),
        tuple(map(int, moved_indices.shape)),
        array.context,
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
    source = np.arange(array.size, dtype=np.int64).reshape(array.shape)
    if axis is None:
        repeat_counts = _normalise_repeat_counts(repeats, array.size)
        repeated = np.repeat(source.reshape(-1), repeat_counts)
    else:
        axis_index = _normalise_axis_permutation_axis("repeat", axis, rank=array.ndim)
        repeat_counts = _normalise_repeat_counts(repeats, array.shape[axis_index])
        repeated = np.repeat(source, repeat_counts, axis=axis_index)
    return TraceADArray(
        tuple(array._items[int(index)] for index in repeated.reshape(-1)),
        tuple(map(int, repeated.shape)),
        array.context,
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
    reps_tuple = _normalise_tile_reps(reps)
    rank = max(array.ndim, len(reps_tuple))
    source_shape = (1,) * (rank - array.ndim) + array.shape
    reps_aligned = (1,) * (rank - len(reps_tuple)) + reps_tuple
    source = np.arange(array.size, dtype=np.int64).reshape(source_shape)
    tiled = np.tile(source, reps_aligned)
    return TraceADArray(
        tuple(array._items[int(index)] for index in tiled.reshape(-1)),
        tuple(map(int, tiled.shape)),
        array.context,
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
    return TraceADArray(
        tuple(array._items[int(index)] for index in rolled.reshape(-1)),
        tuple(map(int, rolled.shape)),
        array.context,
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
    k_value = _normalise_rot90_k(k)
    axes_value = _normalise_rot90_axes(axes, rank=array.ndim)
    rotated = np.rot90(
        np.arange(array.size, dtype=np.int64).reshape(array.shape),
        k=k_value,
        axes=axes_value,
    )
    return TraceADArray(
        tuple(array._items[int(index)] for index in rotated.reshape(-1)),
        tuple(map(int, rotated.shape)),
        array.context,
    )


def _trace_flip(array: TraceADArray, *, axis: object = None) -> TraceADArray:
    source = np.arange(array.size, dtype=np.int64).reshape(array.shape)
    if axis is None:
        flipped = np.flip(source)
    else:
        axes = _normalise_axis_permutation_axes("flip", axis, rank=array.ndim, role="axis")
        flipped = np.flip(source, axis=axes)
    return TraceADArray(
        tuple(array._items[int(index)] for index in flipped.reshape(-1)),
        tuple(map(int, flipped.shape)),
        array.context,
    )


def _trace_flipud(array: TraceADArray) -> TraceADArray:
    if array.ndim < 1:
        raise ValueError("program AD flipud requires at least rank-1 arrays")
    return _trace_flip(array, axis=0)


def _trace_fliplr(array: TraceADArray) -> TraceADArray:
    if array.ndim < 2:
        raise ValueError("program AD fliplr requires at least rank-2 arrays")
    return _trace_flip(array, axis=1)


def _validate_trace_basic_index(index: object) -> None:
    if isinstance(index, tuple):
        for selector in index:
            _validate_trace_basic_index_selector(selector)
        return
    _validate_trace_basic_index_selector(index)


def _validate_trace_basic_index_selector(selector: object) -> None:
    if isinstance(selector, (TraceADScalar, TraceADArray, np.ndarray, list)):
        raise ValueError("program AD advanced indexing is not supported; indices must be static")
    if isinstance(selector, (bool, np.bool_)):
        raise ValueError("program AD advanced indexing is not supported; indices must be static")
    if selector is Ellipsis or selector is None:
        return
    if isinstance(selector, (int, np.integer)):
        return
    if isinstance(selector, slice):
        for item in (selector.start, selector.stop, selector.step):
            if item is not None and (
                isinstance(item, (bool, np.bool_, TraceADScalar, TraceADArray))
                or not isinstance(item, (int, np.integer))
            ):
                raise ValueError("program AD basic indexing requires static integer slice bounds")
        return
    raise ValueError("program AD advanced indexing is not supported; indices must be static")


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


def _broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Return a NumPy-compatible broadcast shape or fail closed."""

    try:
        return cast(tuple[int, ...], np.broadcast_shapes(*shapes))
    except ValueError as exc:
        raise ValueError(
            "whole-program AD array operands must follow NumPy broadcasting rules"
        ) from exc


def _apply_trace_ufunc(
    ufunc: np.ufunc,
    inputs: tuple[object, ...],
    context: _WholeProgramTraceContext,
) -> TraceADScalar | TraceADArray:
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
    if mode != "raise":
        raise ValueError("program AD np.take currently supports only mode='raise'")
    if isinstance(indices, (TraceADScalar, TraceADArray)):
        raise ValueError("program AD np.take requires static integer indices")
    raw_indices = np.asarray(indices)
    if raw_indices.dtype.kind not in {"i", "u"}:
        raise ValueError("program AD np.take requires static integer indices")
    source = np.arange(array.size, dtype=np.int64).reshape(array.shape)
    try:
        selected = np.take(source, raw_indices, axis=axis, mode="raise")
    except (IndexError, ValueError) as exc:
        raise ValueError("program AD np.take indices must be in bounds") from exc
    selected_array = np.asarray(selected)
    if selected_array.shape == ():
        return array._items[int(selected_array)]
    items = tuple(array._items[int(index)] for index in selected_array.reshape(-1))
    return TraceADArray(items, tuple(int(dim) for dim in selected_array.shape), array.context)


def _raise_index_selection_boundary() -> NoReturn:
    raise ValueError(
        "program AD argmax/argmin index selection semantics require an explicit "
        "nondifferentiable primitive policy"
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
    return TraceADArray(tuple(items), target_shape, context)


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
    if axes == 0:
        lhs = _coerce_trace_array(left, context)
        rhs = _coerce_trace_array(right, context)
        items = tuple(
            left_item * right_item for left_item in lhs._items for right_item in rhs._items
        )
        return TraceADArray(items, lhs.shape + rhs.shape, context)
    if axes == 1:
        lhs = _coerce_trace_array(left, context)
        rhs = _coerce_trace_array(right, context)
        if lhs.ndim == 1 and rhs.ndim == 1:
            return _trace_dot(lhs, rhs, context)
        return _trace_matmul(lhs, rhs, context)
    if axes == ((1,), (0,)):
        return _trace_matmul(left, right, context)
    if axes == ((0,), (0,)):
        return _trace_inner(left, right, context)
    raise ValueError(
        "whole-program AD np.tensordot supports axes 0, axes 1, and one-axis rank<=2 contractions"
    )


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
    raise ValueError(
        "whole-program AD np.einsum supports explicit dot, outer, matmul, trace, and diag forms"
    )


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
    return _trace_det_items(tuple(array._items), rows, context)


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
    determinant = _trace_det_items(tuple(array._items), rows, context)
    if determinant.primal == 0.0:
        raise ValueError("program AD np.linalg.inv requires a nonsingular matrix")
    if rows == 1:
        return TraceADArray((_coerce_trace_scalar(1.0, context) / determinant,), (1, 1), context)
    inverse_items: list[TraceADScalar] = []
    for row in range(rows):
        for col in range(cols):
            minor_items = tuple(
                array._items[minor_row * rows + minor_col]
                for minor_row in range(rows)
                for minor_col in range(cols)
                if minor_row != col and minor_col != row
            )
            cofactor = _trace_det_items(minor_items, rows - 1, context)
            if (row + col) % 2:
                cofactor = -cofactor
            inverse_items.append(cofactor / determinant)
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
    return _trace_matmul(_trace_inv(lhs, context), right, context)


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
    if exponent == 0:
        return _trace_identity_matrix(rows, context)
    base = _trace_inv(array, context) if exponent < 0 else array
    exponent = abs(exponent)
    result = _trace_identity_matrix(rows, context)
    factor = base
    while exponent:
        if exponent & 1:
            product = _trace_matmul(result, factor, context)
            if not isinstance(product, TraceADArray):
                raise ValueError("program AD np.linalg.matrix_power expected matrix product")
            result = product
        exponent >>= 1
        if exponent:
            product = _trace_matmul(factor, factor, context)
            if not isinstance(product, TraceADArray):
                raise ValueError("program AD np.linalg.matrix_power expected matrix product")
            factor = product
    return result


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
    result: TraceADScalar | TraceADArray = arrays[0]
    for operand in arrays[1:]:
        if isinstance(result, TraceADScalar):
            raise ValueError("program AD np.linalg.multi_dot encountered scalar intermediate")
        result = _trace_matmul(result, operand, context)
    return result


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
    diag = _trace_diag(array, context, k=offset)
    if not isinstance(diag, TraceADArray):
        raise ValueError("whole-program AD np.trace diagonal extraction must return an array")
    return cast(TraceADScalar, diag.sum())


def _trace_diag(
    values: object,
    context: _WholeProgramTraceContext,
    *,
    k: int = 0,
) -> TraceADArray:
    array = _coerce_trace_array(values, context)
    if array.ndim == 1:
        size = array.shape[0] + abs(k)
        zero = _trace_constant(0.0, context)
        items: list[TraceADScalar] = []
        for row in range(size):
            for col in range(size):
                source_index = row if k >= 0 else col
                on_diag = (col - row) == k
                items.append(array._items[source_index] if on_diag else zero)
        return TraceADArray(tuple(items), (size, size), context)
    if array.ndim == 2:
        rows, cols = array.shape
        items = []
        for row in range(rows):
            col = row + k
            if 0 <= col < cols:
                items.append(array._items[row * cols + col])
        if not items:
            raise ValueError("whole-program AD np.diag offset selects an empty diagonal")
        return TraceADArray(tuple(items), (len(items),), context)
    raise ValueError("whole-program AD np.diag supports vectors and matrices only")


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
    predicates = _coerce_trace_predicate_array(condition, shape, context)
    items = []
    for predicate, x_item, y_item in zip(
        predicates.predicates, x_array._items, y_array._items, strict=True
    ):
        chosen = x_item if bool(predicate) else y_item
        items.append(
            context.make(
                "where",
                (predicate.label, x_item.name, y_item.name),
                chosen.primal,
                chosen.tangent,
            )
        )
    result = tuple(items)
    return result[0] if shape == () else TraceADArray(result, shape, context)


def _trace_concatenate(
    arrays: Sequence[object],
    context: _WholeProgramTraceContext,
    *,
    axis: int = 0,
) -> TraceADArray:
    if not arrays:
        raise ValueError("whole-program AD np.concatenate requires at least one array")
    trace_arrays = tuple(_coerce_trace_array(array, context) for array in arrays)
    if axis != 0:
        raise ValueError("whole-program AD np.concatenate currently supports axis=0")
    if any(array.ndim != 1 for array in trace_arrays):
        raise ValueError("whole-program AD np.concatenate supports one-dimensional arrays")
    items = tuple(item for array in trace_arrays for item in array._items)
    return TraceADArray(items, (len(items),), context)


def _trace_stack(
    arrays: Sequence[object],
    context: _WholeProgramTraceContext,
    *,
    axis: int = 0,
) -> TraceADArray:
    if not arrays:
        raise ValueError("whole-program AD np.stack requires at least one array")
    trace_arrays = tuple(_coerce_trace_array(array, context) for array in arrays)
    shape = trace_arrays[0].shape
    if any(array.shape != shape for array in trace_arrays):
        raise ValueError("whole-program AD np.stack operands must have matching shapes")
    if len(shape) != 1:
        raise ValueError("whole-program AD np.stack currently supports one-dimensional arrays")
    if axis < 0:
        axis += 2
    if axis == 0:
        items = tuple(item for array in trace_arrays for item in array._items)
        return TraceADArray(items, (len(trace_arrays), shape[0]), context)
    if axis == 1:
        items = tuple(
            trace_arrays[row]._items[col]
            for col in range(shape[0])
            for row in range(len(trace_arrays))
        )
        return TraceADArray(items, (shape[0], len(trace_arrays)), context)
    raise ValueError("whole-program AD np.stack supports axis 0 or 1 for vectors")


def _trace_clip(
    values: object,
    lower: object,
    upper: object,
    context: _WholeProgramTraceContext,
) -> TraceADScalar | TraceADArray:
    value_array = _coerce_trace_array(values, context)
    lower_array = _broadcast_trace_array(lower, value_array.shape, context)
    upper_array = _broadcast_trace_array(upper, value_array.shape, context)
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
    axis: int | None = None,
) -> TraceADScalar:
    if axis is not None:
        raise ValueError("whole-program AD np.linalg.norm supports flattened norms only")
    if ord_value not in {None, 2, 2.0}:
        raise ValueError("whole-program AD np.linalg.norm supports only Euclidean norm")
    array = _coerce_trace_array(values, context)
    squared = array._items[0] * array._items[0]
    for item in array._items[1:]:
        squared = squared + item * item
    norm = np.sqrt(squared)
    if not isinstance(norm, TraceADScalar):
        raise ValueError("whole-program AD norm must return a scalar")
    return norm


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
    """Return the reverse-mode adjoint replay result attached to a program AD result."""

    if not isinstance(result, WholeProgramADResult):
        raise ValueError("program adjoint input must be a WholeProgramADResult")
    if result.adjoint_result is None:
        raise ValueError("program AD result does not contain adjoint replay metadata")
    return result.adjoint_result


def program_adjoint_gradient(result: WholeProgramADResult) -> NDArray[np.float64]:
    """Return a supported reverse-mode adjoint gradient or fail closed."""

    adjoint = program_adjoint_result(result)
    if not adjoint.supported:
        unsupported = ", ".join(adjoint.unsupported_ops)
        raise ValueError(f"program AD adjoint replay unsupported for ops: {unsupported}")
    return cast(NDArray[np.float64], adjoint.gradient.copy())


def _program_adjoint_result_from_nodes(
    *,
    nodes: tuple[WholeProgramIRNode, ...],
    output_name: str,
    parameter_names: tuple[str, ...],
    trainable: tuple[bool, ...],
) -> ProgramADAdjointResult:
    """Replay reverse-mode adjoints over supported scalar program AD IR nodes."""

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
    return ProgramADAdjointResult(
        gradient=gradient,
        supported=supported,
        unsupported_ops=tuple(sorted(unsupported_ops)),
        method="program_adjoint_replay",
        claim_boundary=(
            "reverse-mode adjoint replay over captured scalar program AD IR for supported "
            "pure operations; unsupported operations fail closed without substituting finite "
            "differences or forward tangents"
        ),
    )


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
    if node.op in {"add", "sub", "mul", "div", "pow", "maximum", "minimum", "where", "clip"}:
        return _program_adjoint_binary_or_selection_contributions(node, node_by_name)
    raise ValueError(f"unsupported program AD adjoint op {node.op}")


def _program_adjoint_binary_or_selection_contributions(
    node: WholeProgramIRNode,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> tuple[tuple[str, float], ...]:
    if node.op == "where":
        if len(node.inputs) != 3:
            raise ValueError("where adjoint requires predicate, true value, and false value")
        left_name = node.inputs[1]
        right_name = node.inputs[2]
        left = _program_adjoint_input_value(left_name, node_by_name)
        right = _program_adjoint_input_value(right_name, node_by_name)
        return ((left_name, 1.0),) if node.value == left else ((right_name, 1.0),)
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


def _program_adjoint_input_value(
    name: str,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> float:
    if _program_adjoint_is_ir_value(name):
        if name not in node_by_name:
            raise ValueError(f"program AD adjoint input {name} is missing from IR")
        return node_by_name[name].value
    try:
        return float(name)
    except ValueError:
        if name.startswith("np.float64(") and name.endswith(")"):
            return float(name.removeprefix("np.float64(").removesuffix(")"))
        raise ValueError(f"program AD adjoint literal {name!r} is not numeric") from None


def _program_adjoint_is_ir_value(name: str) -> bool:
    return isinstance(name, str) and name.startswith("%") and name[1:].isdigit()


def _objective_source(objective: Callable[..., object]) -> str | None:
    """Return dedented source for a Python callable when introspection permits."""

    try:
        return textwrap.dedent(inspect.getsource(objective)).strip()
    except (OSError, TypeError):
        return None


def _objective_bytecode(
    objective: Callable[..., object],
) -> tuple[WholeProgramBytecodeInstruction, ...]:
    """Return bytecode frontend IR for a Python objective when available."""

    try:
        instructions = dis.get_instructions(objective)
    except TypeError:
        return ()

    def normalise_line_number(value: int | None) -> int | None:
        if value is None:
            return None
        line_number = int(value)
        return line_number if line_number > 0 else None

    return tuple(
        WholeProgramBytecodeInstruction(
            offset=int(instruction.offset),
            opname=instruction.opname,
            argrepr=instruction.argrepr,
            line_number=normalise_line_number(instruction.starts_line),
        )
        for instruction in instructions
    )


def _source_ir_features(source: str | None) -> tuple[WholeProgramSourceIRFeature, ...]:
    """Return source-level control, alias, mutation, and loop features."""

    if source is None:
        return ()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    features: list[WholeProgramSourceIRFeature] = []

    def add(node: ast.AST, kind: str, detail: str) -> None:
        line_number = int(getattr(node, "lineno", 1) or 1)
        features.append(WholeProgramSourceIRFeature(kind, detail, line_number))

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            add(node, "control_flow", "if")
        elif isinstance(node, ast.IfExp):
            add(node, "control_flow", "if_expression")
        elif isinstance(node, ast.For):
            add(node, "loop", "for")
        elif isinstance(node, ast.While):
            add(node, "loop", "while")
        elif isinstance(node, ast.Break):
            add(node, "loop", "break")
        elif isinstance(node, ast.Continue):
            add(node, "loop", "continue")
        elif isinstance(node, ast.Assign):
            if len(node.targets) > 1 or any(
                isinstance(target, ast.Name) for target in node.targets
            ):
                add(node, "alias_analysis", "assignment_binding")
            if any(_is_mutation_target(target) for target in node.targets):
                add(node, "mutation", "indexed_or_attribute_assignment")
        elif isinstance(node, ast.AugAssign):
            add(node, "mutation", "augmented_assignment")
        elif isinstance(node, ast.Delete):
            add(node, "mutation", "delete")
        elif isinstance(node, ast.Call):
            name = _ast_call_name(node.func)
            if name.startswith("np.") or name.startswith("numpy."):
                add(node, "numpy", name)
            if name.rsplit(".", 1)[-1] in {
                "append",
                "extend",
                "insert",
                "pop",
                "remove",
                "clear",
                "sort",
                "update",
                "add",
            }:
                add(node, "mutation", name)
    return tuple(features)


def _is_mutation_target(node: ast.AST) -> bool:
    return isinstance(node, ast.Subscript | ast.Attribute)


def _ast_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _ast_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _whole_program_semantics_report(
    *,
    bytecode_instructions: tuple[WholeProgramBytecodeInstruction, ...],
    source_ir_features: tuple[WholeProgramSourceIRFeature, ...],
    trace_events: tuple[WholeProgramTraceEvent, ...],
    source: str | None,
    numpy_observed: bool,
    differentiation_semantics: str,
) -> WholeProgramSemanticsReport:
    feature_kinds = {feature.kind for feature in source_ir_features}
    jump_ops = {
        instruction.opname
        for instruction in bytecode_instructions
        if "JUMP" in instruction.opname or instruction.opname in {"FOR_ITER"}
    }
    return WholeProgramSemanticsReport(
        bytecode_frontend=bool(bytecode_instructions),
        source_frontend=source is not None,
        graph_capture=bool(trace_events or bytecode_instructions or source_ir_features),
        aliasing_observed="alias_analysis" in feature_kinds,
        mutation_observed="mutation" in feature_kinds,
        loop_observed="loop" in feature_kinds or "FOR_ITER" in jump_ops,
        control_flow_observed=_source_has_control_flow(source)
        or "control_flow" in feature_kinds
        or bool(jump_ops),
        numpy_observed=numpy_observed or "numpy" in feature_kinds,
        differentiation_semantics=differentiation_semantics,
    )


def _source_has_control_flow(source: str | None) -> bool:
    """Return whether source contains explicit Python control-flow nodes."""

    if source is None:
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return any(token in source for token in ("if ", "for ", "while "))
    return any(
        isinstance(node, (ast.If, ast.For, ast.While, ast.IfExp)) for node in ast.walk(tree)
    )


def _source_mentions_numpy(source: str | None) -> bool:
    """Return whether a source fragment visibly references NumPy."""

    if source is None:
        return False
    return "np." in source or "numpy." in source


def _trace_whole_program_objective(
    objective: ScalarObjective, values: NDArray[np.float64]
) -> tuple[WholeProgramTraceEvent, ...]:
    """Execute ``objective`` once and capture source-line trace events."""

    code = getattr(objective, "__code__", None)
    if code is None:
        return ()
    target_filename = code.co_filename
    events: list[WholeProgramTraceEvent] = []
    seen: set[tuple[str, int, str]] = set()
    previous_trace = sys.gettrace()

    def tracer(frame: Any, event: str, arg: object) -> Any:
        del arg
        if event == "line" and frame.f_code.co_filename == target_filename:
            key = (frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
            if key not in seen:
                seen.add(key)
                events.append(
                    WholeProgramTraceEvent(
                        filename=frame.f_code.co_filename,
                        function_name=frame.f_code.co_name,
                        line_number=frame.f_lineno,
                        source=linecache.getline(frame.f_code.co_filename, frame.f_lineno),
                    )
                )
        return tracer

    sys.settrace(tracer)
    try:
        raw = objective(np.array(values, dtype=np.float64, copy=True))
    finally:
        sys.settrace(previous_trace)
    _as_real_scalar("whole-program traced objective", raw)
    return tuple(events)


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
        mapped: list[tuple[NDArray[np.float64], int] | None] = []
        batch_size: int | None = None
        for index, (arg, axis) in enumerate(zip(args, axes, strict=True)):
            if axis is None:
                mapped.append(None)
                continue
            array = _as_real_numeric_array(f"vmap argument {index}", arg)
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
    if isinstance(first, np.ndarray) or np.isscalar(first):
        arrays = [np.asarray(output) for output in outputs]
        shape = arrays[0].shape
        if any(array.shape != shape for array in arrays):
            raise ValueError("vmap output leaves must have consistent shapes")
        axis = out_axes
        result_rank = arrays[0].ndim + 1
        if axis < 0:
            axis += result_rank
        if axis < 0 or axis >= result_rank:
            raise ValueError("out_axes is out of bounds for stacked output rank")
        stacked = np.stack(arrays, axis=axis)
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


def _as_real_numeric_array(name: str, values: object) -> NDArray[np.float64]:
    """Return a real numeric vector without implicit string/bool/object coercion."""
    try:
        raw = np.asarray(values)
    except ValueError as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc

    if raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must contain real numeric scalars")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric scalars") from exc
    return cast(NDArray[np.float64], array)


def _as_real_scalar(name: str, value: object) -> float:
    """Return an explicit real numeric scalar without implicit coercion."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real numeric scalar")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must be a real numeric scalar")
    scalar = float(raw)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_index_vector(name: str, values: object) -> NDArray[np.int64]:
    """Return a one-dimensional non-negative integer index vector."""

    raw = np.asarray(values)
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain integer indices")
    array = np.asarray(raw, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.any(array < 0):
        raise ValueError(f"{name} must contain non-negative indices")
    return cast(NDArray[np.int64], array)


@dataclass(frozen=True)
class Parameter:
    """One differentiable scalar parameter in an SCPN objective."""

    name: str
    trainable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("parameter name must be non-empty")
        if not isinstance(self.trainable, bool):
            raise ValueError("parameter trainable flag must be a boolean")


@dataclass(frozen=True)
class ParameterBounds:
    """Closed interval constraint for one differentiable scalar parameter."""

    lower: float | None = None
    upper: float | None = None
    periodic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.periodic, bool):
            raise ValueError("periodic flag must be a boolean")
        lower = None if self.lower is None else _as_real_scalar("lower bound", self.lower)
        upper = None if self.upper is None else _as_real_scalar("upper bound", self.upper)
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower bound must be less than or equal to upper bound")
        if self.periodic:
            if lower is None or upper is None:
                raise ValueError("periodic bounds require finite lower and upper bounds")
            if lower == upper:
                raise ValueError("periodic bounds require lower < upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class ParameterShiftRule:
    """Two-point parameter-shift rule for one-generator rotation parameters."""

    shift: float = float(np.pi / 2.0)
    coefficient: float = 0.5

    def __post_init__(self) -> None:
        shift = _as_real_scalar("shift", self.shift)
        coefficient = _as_real_scalar("coefficient", self.coefficient)
        if shift <= 0.0:
            raise ValueError("shift must be finite and positive")
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "coefficient", coefficient)


@dataclass(frozen=True)
class DualNumber:
    """Forward-mode automatic differentiation scalar with one tangent lane."""

    primal: float
    tangent: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "primal", _as_real_scalar("dual primal", self.primal))
        object.__setattr__(self, "tangent", _as_real_scalar("dual tangent", self.tangent))

    @staticmethod
    def coerce(value: object) -> DualNumber:
        """Return a dual number, treating real scalars as zero-tangent constants."""

        if isinstance(value, DualNumber):
            return value
        return DualNumber(_as_real_scalar("dual operand", value), 0.0)

    def __add__(self, other: object) -> DualNumber:
        rhs = DualNumber.coerce(other)
        return DualNumber(self.primal + rhs.primal, self.tangent + rhs.tangent)

    def __radd__(self, other: object) -> DualNumber:
        return self.__add__(other)

    def __sub__(self, other: object) -> DualNumber:
        rhs = DualNumber.coerce(other)
        return DualNumber(self.primal - rhs.primal, self.tangent - rhs.tangent)

    def __rsub__(self, other: object) -> DualNumber:
        lhs = DualNumber.coerce(other)
        return lhs.__sub__(self)

    def __mul__(self, other: object) -> DualNumber:
        rhs = DualNumber.coerce(other)
        return DualNumber(
            self.primal * rhs.primal,
            self.tangent * rhs.primal + self.primal * rhs.tangent,
        )

    def __rmul__(self, other: object) -> DualNumber:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> DualNumber:
        rhs = DualNumber.coerce(other)
        if rhs.primal == 0.0:
            raise ValueError("dual division denominator must be non-zero")
        return DualNumber(
            self.primal / rhs.primal,
            (self.tangent * rhs.primal - self.primal * rhs.tangent) / rhs.primal**2,
        )

    def __rtruediv__(self, other: object) -> DualNumber:
        lhs = DualNumber.coerce(other)
        return lhs.__truediv__(self)

    def __neg__(self) -> DualNumber:
        return DualNumber(-self.primal, -self.tangent)

    def __pow__(self, other: object) -> DualNumber:
        rhs = DualNumber.coerce(other)
        if self.primal <= 0.0 and rhs.tangent != 0.0:
            raise ValueError("dual variable exponent requires positive base")
        primal = self.primal**rhs.primal
        if rhs.tangent == 0.0:
            tangent = rhs.primal * self.primal ** (rhs.primal - 1.0) * self.tangent
        else:
            tangent = primal * (
                rhs.tangent * float(np.log(self.primal)) + rhs.primal * self.tangent / self.primal
            )
        return DualNumber(primal, tangent)

    def __rpow__(self, other: object) -> DualNumber:
        lhs = DualNumber.coerce(other)
        return lhs.__pow__(self)


def dual_sin(value: object) -> DualNumber:
    """Forward-mode sine primitive."""

    arg = DualNumber.coerce(value)
    return DualNumber(float(np.sin(arg.primal)), float(np.cos(arg.primal)) * arg.tangent)


def dual_cos(value: object) -> DualNumber:
    """Forward-mode cosine primitive."""

    arg = DualNumber.coerce(value)
    return DualNumber(float(np.cos(arg.primal)), -float(np.sin(arg.primal)) * arg.tangent)


def dual_exp(value: object) -> DualNumber:
    """Forward-mode exponential primitive."""

    arg = DualNumber.coerce(value)
    primal = float(np.exp(arg.primal))
    return DualNumber(primal, primal * arg.tangent)


def dual_log(value: object) -> DualNumber:
    """Forward-mode natural-log primitive."""

    arg = DualNumber.coerce(value)
    if arg.primal <= 0.0:
        raise ValueError("dual log input must be positive")
    return DualNumber(float(np.log(arg.primal)), arg.tangent / arg.primal)


class ReverseNode:
    """Reverse-mode automatic differentiation scalar with local pullbacks."""

    __slots__ = ("adjoint", "parents", "primal")

    def __init__(
        self,
        primal: float,
        parents: tuple[tuple[ReverseNode, float], ...] = (),
    ) -> None:
        self.primal = _as_real_scalar("reverse primal", primal)
        self.parents = parents
        self.adjoint = 0.0

    @staticmethod
    def coerce(value: object) -> ReverseNode:
        """Return a reverse node, treating real scalars as constants."""

        if isinstance(value, ReverseNode):
            return value
        return ReverseNode(_as_real_scalar("reverse operand", value))

    def __add__(self, other: object) -> ReverseNode:
        rhs = ReverseNode.coerce(other)
        return ReverseNode(self.primal + rhs.primal, ((self, 1.0), (rhs, 1.0)))

    def __radd__(self, other: object) -> ReverseNode:
        return self.__add__(other)

    def __sub__(self, other: object) -> ReverseNode:
        rhs = ReverseNode.coerce(other)
        return ReverseNode(self.primal - rhs.primal, ((self, 1.0), (rhs, -1.0)))

    def __rsub__(self, other: object) -> ReverseNode:
        lhs = ReverseNode.coerce(other)
        return lhs.__sub__(self)

    def __mul__(self, other: object) -> ReverseNode:
        rhs = ReverseNode.coerce(other)
        return ReverseNode(
            self.primal * rhs.primal,
            ((self, rhs.primal), (rhs, self.primal)),
        )

    def __rmul__(self, other: object) -> ReverseNode:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> ReverseNode:
        rhs = ReverseNode.coerce(other)
        if rhs.primal == 0.0:
            raise ValueError("reverse division denominator must be non-zero")
        return ReverseNode(
            self.primal / rhs.primal,
            ((self, 1.0 / rhs.primal), (rhs, -self.primal / rhs.primal**2)),
        )

    def __rtruediv__(self, other: object) -> ReverseNode:
        lhs = ReverseNode.coerce(other)
        return lhs.__truediv__(self)

    def __neg__(self) -> ReverseNode:
        return ReverseNode(-self.primal, ((self, -1.0),))

    def __pow__(self, other: object) -> ReverseNode:
        rhs = ReverseNode.coerce(other)
        if self.primal <= 0.0 and isinstance(other, ReverseNode):
            raise ValueError("reverse variable exponent requires positive base")
        primal = self.primal**rhs.primal
        parents: list[tuple[ReverseNode, float]] = []
        parents.append((self, rhs.primal * self.primal ** (rhs.primal - 1.0)))
        if rhs.parents:
            parents.append((rhs, primal * float(np.log(self.primal))))
        return ReverseNode(primal, tuple(parents))

    def __rpow__(self, other: object) -> ReverseNode:
        lhs = ReverseNode.coerce(other)
        return lhs.__pow__(self)


def reverse_sin(value: object) -> ReverseNode:
    """Reverse-mode sine primitive."""

    arg = ReverseNode.coerce(value)
    return ReverseNode(float(np.sin(arg.primal)), ((arg, float(np.cos(arg.primal))),))


def reverse_cos(value: object) -> ReverseNode:
    """Reverse-mode cosine primitive."""

    arg = ReverseNode.coerce(value)
    return ReverseNode(float(np.cos(arg.primal)), ((arg, -float(np.sin(arg.primal))),))


def reverse_exp(value: object) -> ReverseNode:
    """Reverse-mode exponential primitive."""

    arg = ReverseNode.coerce(value)
    primal = float(np.exp(arg.primal))
    return ReverseNode(primal, ((arg, primal),))


def reverse_log(value: object) -> ReverseNode:
    """Reverse-mode natural-log primitive."""

    arg = ReverseNode.coerce(value)
    if arg.primal <= 0.0:
        raise ValueError("reverse log input must be positive")
    return ReverseNode(float(np.log(arg.primal)), ((arg, 1.0 / arg.primal),))


@dataclass(frozen=True)
class GradientResult:
    """Value, gradient, and provenance returned by a differentiable backend."""

    value: float
    gradient: NDArray[np.float64]
    method: str
    shift: float | None
    coefficient: float | None
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_scalar("gradient result value", self.value)
        gradient = _as_real_numeric_array("gradient", self.gradient)
        if gradient.ndim != 1:
            raise ValueError("gradient must be a one-dimensional array")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("gradient must contain only finite values")
        if not self.method:
            raise ValueError("gradient method must be non-empty")
        shift = None if self.shift is None else _as_real_scalar("gradient shift", self.shift)
        coefficient = (
            None
            if self.coefficient is None
            else _as_real_scalar("gradient coefficient", self.coefficient)
        )
        if shift is not None and shift <= 0.0:
            raise ValueError("gradient shift must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("gradient evaluations must be non-negative")
        if len(self.parameter_names) != gradient.size:
            raise ValueError("parameter_names length must match gradient length")
        if len(self.trainable) != gradient.size:
            raise ValueError("trainable mask length must match gradient length")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "coefficient", coefficient)


@dataclass(frozen=True)
class StochasticGradientResult:
    """Parameter-shift gradient with independent shot-noise uncertainty."""

    value: float
    gradient: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    covariance: NDArray[np.float64]
    confidence_radius: NDArray[np.float64]
    shots: NDArray[np.float64]
    confidence_level: float
    method: str
    shift: float
    coefficient: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_scalar("stochastic gradient value", self.value)
        gradient = _as_parameter_array(self.gradient)
        standard_error = _as_parameter_array(self.standard_error)
        confidence_radius = _as_parameter_array(self.confidence_radius)
        covariance = _as_real_numeric_array("stochastic gradient covariance", self.covariance)
        shots = _as_real_numeric_array("stochastic gradient shots", self.shots)
        confidence_level = _as_real_scalar(
            "stochastic gradient confidence_level",
            self.confidence_level,
        )
        shift = _as_real_scalar("stochastic gradient shift", self.shift)
        coefficient = _as_real_scalar("stochastic gradient coefficient", self.coefficient)
        if standard_error.shape != gradient.shape:
            raise ValueError("standard_error shape must match gradient shape")
        if confidence_radius.shape != gradient.shape:
            raise ValueError("confidence_radius shape must match gradient shape")
        if covariance.shape != (gradient.size, gradient.size):
            raise ValueError("covariance shape must be gradient length squared")
        if shots.shape != (2, gradient.size):
            raise ValueError("shots shape must be (2, gradient length)")
        if not np.all(shots > 0.0) or not np.allclose(shots, np.round(shots)):
            raise ValueError("shots must contain positive integer counts")
        if not np.all(np.isfinite(standard_error)) or np.any(standard_error < 0.0):
            raise ValueError("standard_error must contain finite non-negative values")
        if not np.all(np.isfinite(confidence_radius)) or np.any(confidence_radius < 0.0):
            raise ValueError("confidence_radius must contain finite non-negative values")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("covariance must contain only finite values")
        if confidence_level <= 0.0 or confidence_level >= 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if shift <= 0.0:
            raise ValueError("stochastic gradient shift must be finite and positive")
        if coefficient <= 0.0:
            raise ValueError("stochastic gradient coefficient must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("stochastic gradient evaluations must be non-negative")
        if len(self.parameter_names) != gradient.size:
            raise ValueError("parameter_names length must match gradient length")
        if len(self.trainable) != gradient.size:
            raise ValueError("trainable mask length must match gradient length")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "confidence_radius", confidence_radius)
        object.__setattr__(self, "shots", shots)
        object.__setattr__(self, "confidence_level", confidence_level)
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "coefficient", coefficient)


@dataclass(frozen=True)
class ShotAllocationResult:
    """Per-parameter shot allocation for stochastic parameter-shift gradients."""

    shots: NDArray[np.float64]
    predicted_standard_error: NDArray[np.float64]
    covariance: NDArray[np.float64]
    target_standard_error: float
    total_shots: int
    method: str
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        shots = _as_real_numeric_array("shot allocation shots", self.shots)
        standard_error = _as_parameter_array(self.predicted_standard_error)
        covariance = _as_real_numeric_array("shot allocation covariance", self.covariance)
        target = _as_real_scalar(
            "shot allocation target_standard_error",
            self.target_standard_error,
        )
        if shots.ndim != 2 or shots.shape[0] != 2:
            raise ValueError("shot allocation shots must have shape (2, n_parameters)")
        if standard_error.shape != (shots.shape[1],):
            raise ValueError("predicted_standard_error length must match shot columns")
        if covariance.shape != (shots.shape[1], shots.shape[1]):
            raise ValueError("shot allocation covariance shape must be n_parameters squared")
        if not np.all(shots > 0.0) or not np.allclose(shots, np.round(shots)):
            raise ValueError("shot allocation shots must contain positive integer counts")
        if not np.all(np.isfinite(standard_error)) or np.any(standard_error < 0.0):
            raise ValueError("predicted_standard_error must contain finite non-negative values")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("shot allocation covariance must contain only finite values")
        if target <= 0.0:
            raise ValueError("target_standard_error must be finite and positive")
        total_shots = int(self.total_shots)
        if total_shots != int(np.sum(shots)):
            raise ValueError("total_shots must equal allocated shot sum")
        if not self.method:
            raise ValueError("shot allocation method must be non-empty")
        if len(self.parameter_names) != shots.shape[1]:
            raise ValueError("parameter_names length must match shot columns")
        if len(self.trainable) != shots.shape[1]:
            raise ValueError("trainable mask length must match shot columns")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "shots", shots)
        object.__setattr__(self, "predicted_standard_error", standard_error)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "target_standard_error", target)
        object.__setattr__(self, "total_shots", total_shots)


@dataclass(frozen=True)
class OptimizationResult:
    """Bounded gradient-descent result with convergence provenance."""

    values: NDArray[np.float64]
    final_gradient: GradientResult
    value_history: tuple[float, ...]
    steps: int
    converged: bool
    reason: str
    best_values: NDArray[np.float64] | None = None
    best_value: float | None = None

    def __post_init__(self) -> None:
        values = _as_parameter_array(self.values)
        if values.size != self.final_gradient.gradient.size:
            raise ValueError("optimized values length must match gradient length")
        if not self.value_history:
            raise ValueError("value_history must contain at least one value")
        history = tuple(_as_real_scalar("value_history item", item) for item in self.value_history)
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 0:
            raise ValueError("optimization steps must be a non-negative integer")
        if not isinstance(self.converged, bool):
            raise ValueError("optimization converged flag must be a boolean")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("optimization reason must be non-empty")
        best_values = values if self.best_values is None else _as_parameter_array(self.best_values)
        if best_values.size != values.size:
            raise ValueError("best_values length must match optimized values length")
        best_value = (
            min(history)
            if self.best_value is None
            else _as_real_scalar("best_value", self.best_value)
        )
        if best_value > min(history) + 1.0e-12:
            raise ValueError("best_value must not exceed the minimum value_history entry")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "value_history", history)
        object.__setattr__(self, "best_values", best_values)
        object.__setattr__(self, "best_value", best_value)


@dataclass(frozen=True)
class ArmijoLineSearchResult:
    """Backtracking line-search result with sufficient-decrease provenance."""

    values: NDArray[np.float64]
    value: float
    step_size: float
    direction: NDArray[np.float64]
    directional_derivative: float
    accepted: bool
    evaluations: int
    value_history: tuple[float, ...]
    reason: str
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        values = _as_parameter_array(self.values)
        direction = _as_parameter_array(self.direction)
        if direction.shape != values.shape:
            raise ValueError("line-search direction shape must match values shape")
        value = _as_real_scalar("line-search value", self.value)
        step_size = _as_real_scalar("line-search step_size", self.step_size)
        if step_size < 0.0:
            raise ValueError("line-search step_size must be finite and non-negative")
        directional_derivative = _as_real_scalar(
            "line-search directional_derivative",
            self.directional_derivative,
        )
        if not isinstance(self.accepted, bool):
            raise ValueError("line-search accepted flag must be a boolean")
        if self.evaluations < 0:
            raise ValueError("line-search evaluations must be non-negative")
        if not self.value_history:
            raise ValueError("line-search value_history must be non-empty")
        value_history = tuple(
            _as_real_scalar("line-search value history", item) for item in self.value_history
        )
        if self.reason not in {"accepted", "non_descent_direction", "max_steps"}:
            raise ValueError("line-search reason must be a known status")
        if len(self.parameter_names) != values.size:
            raise ValueError("parameter_names length must match line-search values")
        if len(self.trainable) != values.size:
            raise ValueError("trainable mask length must match line-search values")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "step_size", step_size)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "directional_derivative", directional_derivative)
        object.__setattr__(self, "value_history", value_history)


@dataclass(frozen=True)
class GradientCheckResult:
    """Consistency check between two differentiable gradient estimators."""

    reference: GradientResult
    candidate: GradientResult
    max_abs_error: float
    l2_error: float
    value_delta: float
    tolerance: float
    passed: bool

    def __post_init__(self) -> None:
        if self.reference.gradient.shape != self.candidate.gradient.shape:
            raise ValueError("gradient check operands must have matching shapes")
        max_abs_error = _as_real_scalar("max_abs_error", self.max_abs_error)
        l2_error = _as_real_scalar("l2_error", self.l2_error)
        value_delta = _as_real_scalar("value_delta", self.value_delta)
        tolerance = _as_real_scalar("tolerance", self.tolerance)
        if max_abs_error < 0.0:
            raise ValueError("max_abs_error must be non-negative")
        if l2_error < 0.0:
            raise ValueError("l2_error must be non-negative")
        if value_delta < 0.0:
            raise ValueError("value_delta must be non-negative")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        if not isinstance(self.passed, bool):
            raise ValueError("gradient check passed flag must be a boolean")
        object.__setattr__(self, "max_abs_error", max_abs_error)
        object.__setattr__(self, "l2_error", l2_error)
        object.__setattr__(self, "value_delta", value_delta)
        object.__setattr__(self, "tolerance", tolerance)


@dataclass(frozen=True)
class CustomDerivativeCheckResult:
    """Consistency audit for exact custom JVP/VJP derivative rules."""

    custom_jvp: JVPResult
    custom_vjp: VJPResult
    reference_jvp: JVPResult
    reference_vjp: VJPResult
    adjoint_inner_error: float
    jvp_l2_error: float
    vjp_l2_error: float
    tolerance: float
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.custom_jvp, JVPResult):
            raise ValueError("custom_jvp must be a JVPResult")
        if not isinstance(self.custom_vjp, VJPResult):
            raise ValueError("custom_vjp must be a VJPResult")
        if not isinstance(self.reference_jvp, JVPResult):
            raise ValueError("reference_jvp must be a JVPResult")
        if not isinstance(self.reference_vjp, VJPResult):
            raise ValueError("reference_vjp must be a VJPResult")
        if self.custom_jvp.value.shape != self.reference_jvp.value.shape:
            raise ValueError("custom and reference JVP values must have matching shapes")
        if self.custom_vjp.value.shape != self.reference_vjp.value.shape:
            raise ValueError("custom and reference VJP values must have matching shapes")
        if self.custom_jvp.jvp.shape != self.reference_jvp.jvp.shape:
            raise ValueError("custom and reference JVP outputs must have matching shapes")
        if self.custom_vjp.vjp.shape != self.reference_vjp.vjp.shape:
            raise ValueError("custom and reference VJP outputs must have matching shapes")
        adjoint_inner_error = _as_real_scalar(
            "custom derivative adjoint error",
            self.adjoint_inner_error,
        )
        jvp_l2_error = _as_real_scalar("custom derivative JVP l2 error", self.jvp_l2_error)
        vjp_l2_error = _as_real_scalar("custom derivative VJP l2 error", self.vjp_l2_error)
        tolerance = _as_real_scalar("custom derivative tolerance", self.tolerance)
        if adjoint_inner_error < 0.0 or jvp_l2_error < 0.0 or vjp_l2_error < 0.0:
            raise ValueError("custom derivative errors must be non-negative")
        if tolerance < 0.0:
            raise ValueError("custom derivative tolerance must be finite and non-negative")
        if not isinstance(self.passed, bool):
            raise ValueError("custom derivative passed flag must be a boolean")
        object.__setattr__(self, "adjoint_inner_error", adjoint_inner_error)
        object.__setattr__(self, "jvp_l2_error", jvp_l2_error)
        object.__setattr__(self, "vjp_l2_error", vjp_l2_error)
        object.__setattr__(self, "tolerance", tolerance)


@dataclass(frozen=True)
class JacobianResult:
    """Value, Jacobian, and provenance for a vector-valued objective."""

    value: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_numeric_array("jacobian value", self.value)
        jacobian = _as_real_numeric_array("jacobian", self.jacobian)
        if value.ndim != 1:
            raise ValueError("jacobian value must be a one-dimensional array")
        if jacobian.ndim != 2:
            raise ValueError("jacobian must be a two-dimensional array")
        if jacobian.shape[0] != value.size:
            raise ValueError("jacobian row count must match value length")
        if not np.all(np.isfinite(value)):
            raise ValueError("jacobian value must contain only finite values")
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("jacobian must contain only finite values")
        if not self.method:
            raise ValueError("jacobian method must be non-empty")
        step = _as_real_scalar("jacobian step", self.step)
        if step < 0.0:
            raise ValueError("jacobian step must be finite and non-negative")
        if self.evaluations < 0:
            raise ValueError("jacobian evaluations must be non-negative")
        if len(self.parameter_names) != jacobian.shape[1]:
            raise ValueError("parameter_names length must match jacobian column count")
        if len(self.trainable) != jacobian.shape[1]:
            raise ValueError("trainable mask length must match jacobian column count")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "jacobian", jacobian)
        object.__setattr__(self, "step", step)


@dataclass(frozen=True)
class JVPResult:
    """Jacobian-vector product with directional finite-difference provenance."""

    value: NDArray[np.float64]
    jvp: NDArray[np.float64]
    tangent: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_numeric_array("JVP value", self.value)
        jvp = _as_real_numeric_array("JVP", self.jvp)
        tangent = _as_real_numeric_array("JVP tangent", self.tangent)
        if value.ndim != 1:
            raise ValueError("JVP value must be a one-dimensional array")
        if jvp.shape != value.shape:
            raise ValueError("JVP shape must match value shape")
        if tangent.ndim != 1:
            raise ValueError("JVP tangent must be one-dimensional")
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(jvp)):
            raise ValueError("JVP value and product must contain only finite values")
        if not np.all(np.isfinite(tangent)):
            raise ValueError("JVP tangent must contain only finite values")
        if not self.method:
            raise ValueError("JVP method must be non-empty")
        step = _as_real_scalar("JVP step", self.step)
        if step < 0.0:
            raise ValueError("JVP step must be finite and non-negative")
        if self.evaluations < 0:
            raise ValueError("JVP evaluations must be non-negative")
        if len(self.parameter_names) != tangent.size:
            raise ValueError("parameter_names length must match tangent length")
        if len(self.trainable) != tangent.size:
            raise ValueError("trainable mask length must match tangent length")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "jvp", jvp)
        object.__setattr__(self, "tangent", tangent)
        object.__setattr__(self, "step", step)


@dataclass(frozen=True)
class VJPResult:
    """Vector-Jacobian product with cotangent provenance."""

    value: NDArray[np.float64]
    cotangent: NDArray[np.float64]
    vjp: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_numeric_array("VJP value", self.value)
        cotangent = _as_real_numeric_array("VJP cotangent", self.cotangent)
        vjp = _as_real_numeric_array("VJP", self.vjp)
        if value.ndim != 1:
            raise ValueError("VJP value must be a one-dimensional array")
        if cotangent.shape != value.shape:
            raise ValueError("VJP cotangent shape must match value shape")
        if vjp.ndim != 1:
            raise ValueError("VJP must be one-dimensional")
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(cotangent)):
            raise ValueError("VJP value and cotangent must contain only finite values")
        if not np.all(np.isfinite(vjp)):
            raise ValueError("VJP must contain only finite values")
        if not self.method:
            raise ValueError("VJP method must be non-empty")
        step = _as_real_scalar("VJP step", self.step)
        if step < 0.0:
            raise ValueError("VJP step must be finite and non-negative")
        if self.evaluations < 0:
            raise ValueError("VJP evaluations must be non-negative")
        if len(self.parameter_names) != vjp.size:
            raise ValueError("parameter_names length must match VJP length")
        if len(self.trainable) != vjp.size:
            raise ValueError("trainable mask length must match VJP length")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "cotangent", cotangent)
        object.__setattr__(self, "vjp", vjp)
        object.__setattr__(self, "step", step)


@dataclass(frozen=True)
class HessianResult:
    """Value, Hessian, and provenance for a scalar objective."""

    value: float
    hessian: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_scalar("hessian value", self.value)
        hessian = _as_real_numeric_array("hessian", self.hessian)
        if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
            raise ValueError("hessian must be a square two-dimensional array")
        if not np.all(np.isfinite(hessian)):
            raise ValueError("hessian must contain only finite values")
        if not self.method:
            raise ValueError("hessian method must be non-empty")
        step = _as_real_scalar("hessian step", self.step)
        if step <= 0.0:
            raise ValueError("hessian step must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("hessian evaluations must be non-negative")
        if len(self.parameter_names) != hessian.shape[1]:
            raise ValueError("parameter_names length must match hessian dimension")
        if len(self.trainable) != hessian.shape[1]:
            raise ValueError("trainable mask length must match hessian dimension")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        if not np.allclose(hessian, hessian.T, atol=1.0e-8, rtol=1.0e-8):
            raise ValueError("hessian must be symmetric")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "hessian", hessian)
        object.__setattr__(self, "step", step)


@dataclass(frozen=True)
class SparseMatrixResult:
    """Coordinate sparse derivative matrix with parameter provenance."""

    row_indices: NDArray[np.int64]
    column_indices: NDArray[np.int64]
    values: NDArray[np.float64]
    shape: tuple[int, int]
    method: str
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        rows = _as_index_vector("sparse row_indices", self.row_indices)
        columns = _as_index_vector("sparse column_indices", self.column_indices)
        values = _as_real_numeric_array("sparse values", self.values)
        if values.ndim != 1:
            raise ValueError("sparse values must be one-dimensional")
        if rows.size != columns.size or rows.size != values.size:
            raise ValueError("sparse row, column, and value lengths must match")
        if (
            len(self.shape) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in self.shape)
            or self.shape[0] < 1
            or self.shape[1] < 1
        ):
            raise ValueError("sparse shape must contain two positive integer dimensions")
        if rows.size:
            if int(rows.max()) >= self.shape[0] or int(columns.max()) >= self.shape[1]:
                raise ValueError("sparse indices must be inside matrix shape")
            coordinates = set(zip(rows.tolist(), columns.tolist()))
            if len(coordinates) != rows.size:
                raise ValueError("sparse indices must not contain duplicate coordinates")
        if not np.all(np.isfinite(values)):
            raise ValueError("sparse values must contain only finite values")
        if not self.method:
            raise ValueError("sparse method must be non-empty")
        if len(self.parameter_names) != self.shape[1]:
            raise ValueError("parameter_names length must match sparse column count")
        if len(self.trainable) != self.shape[1]:
            raise ValueError("trainable mask length must match sparse column count")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "column_indices", columns)
        object.__setattr__(self, "values", values)

    @property
    def nnz(self) -> int:
        """Number of explicitly stored non-zero entries."""

        return int(self.values.size)

    def to_dense(self) -> NDArray[np.float64]:
        """Materialise the sparse coordinate matrix as a dense array."""

        dense = np.zeros(self.shape, dtype=np.float64)
        dense[self.row_indices, self.column_indices] = self.values
        return cast(NDArray[np.float64], dense)


@dataclass(frozen=True)
class HVPResult:
    """Hessian-vector product with nested finite-difference provenance."""

    value: float
    hvp: NDArray[np.float64]
    tangent: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_scalar("HVP value", self.value)
        hvp = _as_real_numeric_array("HVP", self.hvp)
        tangent = _as_real_numeric_array("HVP tangent", self.tangent)
        if hvp.ndim != 1:
            raise ValueError("HVP must be one-dimensional")
        if tangent.shape != hvp.shape:
            raise ValueError("HVP tangent shape must match HVP shape")
        if not np.all(np.isfinite(hvp)) or not np.all(np.isfinite(tangent)):
            raise ValueError("HVP and tangent must contain only finite values")
        if not self.method:
            raise ValueError("HVP method must be non-empty")
        step = _as_real_scalar("HVP step", self.step)
        if step <= 0.0:
            raise ValueError("HVP step must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("HVP evaluations must be non-negative")
        if len(self.parameter_names) != hvp.size:
            raise ValueError("parameter_names length must match HVP length")
        if len(self.trainable) != hvp.size:
            raise ValueError("trainable mask length must match HVP length")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "hvp", hvp)
        object.__setattr__(self, "tangent", tangent)
        object.__setattr__(self, "step", step)


@dataclass(frozen=True)
class NaturalGradientResult:
    """Metric-preconditioned gradient with solve provenance."""

    base_gradient: GradientResult
    metric: NDArray[np.float64]
    natural_gradient: NDArray[np.float64]
    damping: float
    condition_number: float

    def __post_init__(self) -> None:
        metric = _as_real_numeric_array("natural-gradient metric", self.metric)
        natural_gradient = _as_real_numeric_array("natural_gradient", self.natural_gradient)
        if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
            raise ValueError("natural-gradient metric must be a square matrix")
        if metric.shape[0] != self.base_gradient.gradient.size:
            raise ValueError("natural-gradient metric dimension must match gradient length")
        if natural_gradient.shape != self.base_gradient.gradient.shape:
            raise ValueError("natural_gradient shape must match gradient shape")
        if not np.all(np.isfinite(metric)):
            raise ValueError("natural-gradient metric must contain only finite values")
        if not np.all(np.isfinite(natural_gradient)):
            raise ValueError("natural_gradient must contain only finite values")
        if not np.allclose(metric, metric.T, atol=1.0e-10, rtol=1.0e-10):
            raise ValueError("natural-gradient metric must be symmetric")
        damping = _as_real_scalar("natural-gradient damping", self.damping)
        if damping < 0.0:
            raise ValueError("natural-gradient damping must be finite and non-negative")
        condition_number = _as_real_scalar(
            "natural-gradient condition_number", self.condition_number
        )
        if condition_number < 1.0:
            raise ValueError("natural-gradient condition_number must be at least 1")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "natural_gradient", natural_gradient)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "condition_number", condition_number)


@dataclass(frozen=True)
class NaturalGradientOptimizationResult:
    """Bounded natural-gradient optimization trace and final state."""

    values: NDArray[np.float64]
    final_gradient: GradientResult
    final_natural_gradient: NaturalGradientResult
    value_history: tuple[float, ...]
    gradient_norm_history: tuple[float, ...]
    natural_step_norm_history: tuple[float, ...]
    steps: int
    converged: bool
    reason: str
    best_values: NDArray[np.float64]
    best_value: float

    def __post_init__(self) -> None:
        values = _as_parameter_array(self.values)
        best_values = _as_parameter_array(self.best_values)
        if best_values.shape != values.shape:
            raise ValueError("natural-gradient best_values shape must match values shape")
        if not isinstance(self.final_gradient, GradientResult):
            raise ValueError("final_gradient must be a GradientResult")
        if not isinstance(self.final_natural_gradient, NaturalGradientResult):
            raise ValueError("final_natural_gradient must be a NaturalGradientResult")
        if not self.value_history:
            raise ValueError("natural-gradient value_history must be non-empty")
        value_history = tuple(
            _as_real_scalar("natural-gradient value history", value)
            for value in self.value_history
        )
        gradient_norm_history = tuple(
            _as_real_scalar("natural-gradient gradient norm history", value)
            for value in self.gradient_norm_history
        )
        step_norm_history = tuple(
            _as_real_scalar("natural-gradient step norm history", value)
            for value in self.natural_step_norm_history
        )
        if any(value < 0.0 for value in gradient_norm_history):
            raise ValueError("gradient_norm_history must contain non-negative values")
        if any(value < 0.0 for value in step_norm_history):
            raise ValueError("natural_step_norm_history must contain non-negative values")
        steps = int(self.steps)
        if steps < 0:
            raise ValueError("natural-gradient steps must be non-negative")
        if len(value_history) != steps + 1:
            raise ValueError("value_history must include initial value plus one per step")
        if len(gradient_norm_history) != steps + 1:
            raise ValueError("gradient_norm_history must include initial value plus one per step")
        if len(step_norm_history) != steps:
            raise ValueError("natural_step_norm_history must include one value per update step")
        if self.reason not in {
            "gradient_tolerance",
            "step_tolerance",
            "value_tolerance",
            "max_steps",
        }:
            raise ValueError("natural-gradient result reason must be known")
        best_value = _as_real_scalar("natural-gradient best_value", self.best_value)
        if best_value > min(value_history) + 1.0e-12:
            raise ValueError("best_value must be no larger than the recorded minimum")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "value_history", value_history)
        object.__setattr__(self, "gradient_norm_history", gradient_norm_history)
        object.__setattr__(self, "natural_step_norm_history", step_norm_history)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "best_values", best_values)
        object.__setattr__(self, "best_value", best_value)


@dataclass(frozen=True)
class ImplicitSensitivityResult:
    """Implicit-function sensitivity for a stationary differentiable system."""

    sensitivity: NDArray[np.float64]
    hessian: NDArray[np.float64]
    cross_derivative: NDArray[np.float64]
    damping: float
    condition_number: float
    method: str
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    hyperparameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        sensitivity = _as_real_numeric_array("implicit sensitivity", self.sensitivity)
        hessian = _as_real_numeric_array("implicit hessian", self.hessian)
        cross = _as_real_numeric_array("implicit cross_derivative", self.cross_derivative)
        if sensitivity.ndim != 2 or hessian.ndim != 2 or cross.ndim != 2:
            raise ValueError("implicit sensitivity operands must be two-dimensional")
        if hessian.shape[0] != hessian.shape[1]:
            raise ValueError("implicit hessian must be square")
        if sensitivity.shape != cross.shape:
            raise ValueError("implicit sensitivity shape must match cross_derivative shape")
        if sensitivity.shape[0] != hessian.shape[0]:
            raise ValueError("implicit sensitivity row count must match hessian dimension")
        if not np.all(np.isfinite(sensitivity)):
            raise ValueError("implicit sensitivity must contain only finite values")
        if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(cross)):
            raise ValueError("implicit operands must contain only finite values")
        if not np.allclose(hessian, hessian.T, atol=1.0e-10, rtol=1.0e-10):
            raise ValueError("implicit hessian must be symmetric")
        damping = _as_real_scalar("implicit damping", self.damping)
        if damping < 0.0:
            raise ValueError("implicit damping must be finite and non-negative")
        condition_number = _as_real_scalar(
            "implicit condition_number",
            self.condition_number,
        )
        if condition_number < 1.0:
            raise ValueError("implicit condition_number must be at least 1")
        if not self.method:
            raise ValueError("implicit method must be non-empty")
        if len(self.parameter_names) != hessian.shape[0]:
            raise ValueError("parameter_names length must match implicit hessian dimension")
        if len(self.trainable) != hessian.shape[0]:
            raise ValueError("trainable mask length must match implicit hessian dimension")
        if len(self.hyperparameter_names) != cross.shape[1]:
            raise ValueError("hyperparameter_names length must match cross_derivative columns")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(name, str) or not name for name in self.hyperparameter_names):
            raise ValueError("hyperparameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "sensitivity", sensitivity)
        object.__setattr__(self, "hessian", hessian)
        object.__setattr__(self, "cross_derivative", cross)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "condition_number", condition_number)


@dataclass(frozen=True)
class FixedPointSensitivityResult:
    """Implicit sensitivity for a converged fixed-point map."""

    sensitivity: NDArray[np.float64]
    state_jacobian: NDArray[np.float64]
    parameter_jacobian: NDArray[np.float64]
    system_matrix: NDArray[np.float64]
    damping: float
    condition_number: float
    method: str
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    hyperparameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        sensitivity = _as_real_numeric_array("fixed-point sensitivity", self.sensitivity)
        state_jacobian = _as_real_numeric_array("fixed-point state_jacobian", self.state_jacobian)
        parameter_jacobian = _as_real_numeric_array(
            "fixed-point parameter_jacobian",
            self.parameter_jacobian,
        )
        system_matrix = _as_real_numeric_array("fixed-point system_matrix", self.system_matrix)
        if (
            sensitivity.ndim != 2
            or state_jacobian.ndim != 2
            or parameter_jacobian.ndim != 2
            or system_matrix.ndim != 2
        ):
            raise ValueError("fixed-point sensitivity operands must be two-dimensional")
        if state_jacobian.shape[0] != state_jacobian.shape[1]:
            raise ValueError("fixed-point state_jacobian must be square")
        if system_matrix.shape != state_jacobian.shape:
            raise ValueError("fixed-point system_matrix shape must match state_jacobian")
        if sensitivity.shape != parameter_jacobian.shape:
            raise ValueError("fixed-point sensitivity shape must match parameter_jacobian")
        if sensitivity.shape[0] != state_jacobian.shape[0]:
            raise ValueError("fixed-point sensitivity row count must match state dimension")
        if not np.all(np.isfinite(sensitivity)):
            raise ValueError("fixed-point sensitivity must contain only finite values")
        if (
            not np.all(np.isfinite(state_jacobian))
            or not np.all(np.isfinite(parameter_jacobian))
            or not np.all(np.isfinite(system_matrix))
        ):
            raise ValueError("fixed-point operands must contain only finite values")
        damping = _as_real_scalar("fixed-point damping", self.damping)
        if damping < 0.0:
            raise ValueError("fixed-point damping must be finite and non-negative")
        condition_number = _as_real_scalar(
            "fixed-point condition_number",
            self.condition_number,
        )
        if condition_number < 1.0:
            raise ValueError("fixed-point condition_number must be at least 1")
        if not self.method:
            raise ValueError("fixed-point method must be non-empty")
        if len(self.parameter_names) != state_jacobian.shape[0]:
            raise ValueError("parameter_names length must match fixed-point state dimension")
        if len(self.trainable) != state_jacobian.shape[0]:
            raise ValueError("trainable mask length must match fixed-point state dimension")
        if len(self.hyperparameter_names) != parameter_jacobian.shape[1]:
            raise ValueError(
                "hyperparameter_names length must match fixed-point parameter_jacobian columns"
            )
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(name, str) or not name for name in self.hyperparameter_names):
            raise ValueError("hyperparameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "sensitivity", sensitivity)
        object.__setattr__(self, "state_jacobian", state_jacobian)
        object.__setattr__(self, "parameter_jacobian", parameter_jacobian)
        object.__setattr__(self, "system_matrix", system_matrix)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "condition_number", condition_number)


@dataclass(frozen=True)
class CustomDerivativeRule:
    """Exact custom derivative rules for one differentiable vector primitive."""

    name: str
    value_fn: VectorObjective
    jvp_rule: CustomJVPRule | None = None
    vjp_rule: CustomVJPRule | None = None
    parameter_names: tuple[str, ...] = ()
    trainable: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("custom derivative rule name must be non-empty")
        if not callable(self.value_fn):
            raise ValueError("custom derivative value_fn must be callable")
        if self.jvp_rule is None and self.vjp_rule is None:
            raise ValueError("custom derivative rule requires a JVP or VJP rule")
        if self.jvp_rule is not None and not callable(self.jvp_rule):
            raise ValueError("custom derivative jvp_rule must be callable")
        if self.vjp_rule is not None and not callable(self.vjp_rule):
            raise ValueError("custom derivative vjp_rule must be callable")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        if (
            self.parameter_names
            and self.trainable
            and len(self.parameter_names) != len(self.trainable)
        ):
            raise ValueError("parameter_names and trainable mask lengths must match")


@dataclass(frozen=True)
class PrimitiveIdentity:
    """Stable typed identity for a differentiable primitive implementation."""

    namespace: str
    name: str
    version: str = "1"

    def __post_init__(self) -> None:
        namespace = _normalise_identity_token("primitive namespace", self.namespace)
        name = _normalise_identity_token("primitive name", self.name)
        version = _normalise_identity_token("primitive version", self.version)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)

    @property
    def key(self) -> str:
        """Return the canonical registry key for this primitive identity."""

        return f"{self.namespace}:{self.name}@{self.version}"

    @staticmethod
    def parse(identity: PrimitiveIdentity | str) -> PrimitiveIdentity:
        """Return a typed identity from an existing identity or canonical string."""

        if isinstance(identity, PrimitiveIdentity):
            return identity
        if not isinstance(identity, str) or not identity:
            raise ValueError("primitive identity must be a PrimitiveIdentity or non-empty string")
        if "@" in identity:
            stem, version = identity.rsplit("@", 1)
        else:
            stem, version = identity, "1"
        if ":" not in stem:
            raise ValueError(
                "primitive identity string must use 'namespace:name[@version]' format"
            )
        namespace, name = stem.split(":", 1)
        return PrimitiveIdentity(namespace=namespace, name=name, version=version)


def _normalise_identity_token(name: str, value: object) -> str:
    """Return a registry-safe identity token."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if any(character.isspace() for character in value):
        raise ValueError(f"{name} must not contain whitespace")
    if any(character in value for character in (":", "@")):
        raise ValueError(f"{name} must not contain ':' or '@'")
    return value


PrimitiveLoweringRule = Callable[[CustomDerivativeRule], object]
PrimitiveShapeRule = Callable[[tuple[object, ...]], tuple[int, ...]]
PrimitiveDTypeRule = Callable[[tuple[object, ...]], str]
PrimitiveStaticArgumentRule = Callable[[tuple[object, ...]], tuple[object, ...]]


@dataclass(frozen=True)
class PrimitiveTransformRule:
    """Combined transform binding for one differentiable primitive identity."""

    identity: PrimitiveIdentity
    derivative_rule: CustomDerivativeRule
    batching_rule: PrimitiveBatchingRule | None = None
    lowering_rule: PrimitiveLoweringRule | None = None
    lowering_metadata: Mapping[str, str] | None = None
    shape_rule: PrimitiveShapeRule | None = None
    dtype_rule: PrimitiveDTypeRule | None = None
    static_argument_rule: PrimitiveStaticArgumentRule | None = None
    nondifferentiable_policy: str = "not_declared"
    effect: str = "pure"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PrimitiveIdentity):
            raise ValueError("transform identity must be a PrimitiveIdentity")
        if not isinstance(self.derivative_rule, CustomDerivativeRule):
            raise ValueError("transform derivative_rule must be a CustomDerivativeRule")
        if self.batching_rule is not None and not callable(self.batching_rule):
            raise ValueError("transform batching_rule must be callable")
        if self.lowering_rule is not None and not callable(self.lowering_rule):
            raise ValueError("transform lowering_rule must be callable")
        if self.shape_rule is not None and not callable(self.shape_rule):
            raise ValueError("transform shape_rule must be callable")
        if self.dtype_rule is not None and not callable(self.dtype_rule):
            raise ValueError("transform dtype_rule must be callable")
        if self.static_argument_rule is not None and not callable(self.static_argument_rule):
            raise ValueError("transform static_argument_rule must be callable")
        if not isinstance(self.nondifferentiable_policy, str) or not self.nondifferentiable_policy:
            raise ValueError("transform nondifferentiable_policy must be a non-empty string")
        if not isinstance(self.effect, str) or not self.effect:
            raise ValueError("transform effect must be a non-empty string")
        metadata = {} if self.lowering_metadata is None else dict(self.lowering_metadata)
        if any(not isinstance(key, str) or not key for key in metadata):
            raise ValueError("lowering metadata keys must be non-empty strings")
        if any(not isinstance(value, str) or not value for value in metadata.values()):
            raise ValueError("lowering metadata values must be non-empty strings")
        object.__setattr__(self, "lowering_metadata", metadata)


@dataclass(frozen=True)
class PrimitiveContract:
    """Unified registered contract for one differentiable primitive identity."""

    identity: PrimitiveIdentity
    derivative_rule: CustomDerivativeRule
    batching_rule: PrimitiveBatchingRule | None
    lowering_rule: PrimitiveLoweringRule | None
    lowering_metadata: Mapping[str, str]
    shape_rule: PrimitiveShapeRule | None
    dtype_rule: PrimitiveDTypeRule | None
    static_argument_rule: PrimitiveStaticArgumentRule | None
    nondifferentiable_policy: str
    effect: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PrimitiveIdentity):
            raise ValueError("contract identity must be a PrimitiveIdentity")
        if not isinstance(self.derivative_rule, CustomDerivativeRule):
            raise ValueError("contract derivative_rule must be a CustomDerivativeRule")
        if self.batching_rule is not None and not callable(self.batching_rule):
            raise ValueError("contract batching_rule must be callable")
        if self.lowering_rule is not None and not callable(self.lowering_rule):
            raise ValueError("contract lowering_rule must be callable")
        if self.shape_rule is not None and not callable(self.shape_rule):
            raise ValueError("contract shape_rule must be callable")
        if self.dtype_rule is not None and not callable(self.dtype_rule):
            raise ValueError("contract dtype_rule must be callable")
        if self.static_argument_rule is not None and not callable(self.static_argument_rule):
            raise ValueError("contract static_argument_rule must be callable")
        if not isinstance(self.nondifferentiable_policy, str) or not self.nondifferentiable_policy:
            raise ValueError("contract nondifferentiable_policy must be a non-empty string")
        if not isinstance(self.effect, str) or not self.effect:
            raise ValueError("contract effect must be a non-empty string")
        metadata = dict(self.lowering_metadata)
        if any(not isinstance(key, str) or not key for key in metadata):
            raise ValueError("contract lowering metadata keys must be non-empty strings")
        if any(not isinstance(value, str) or not value for value in metadata.values()):
            raise ValueError("contract lowering metadata values must be non-empty strings")
        object.__setattr__(self, "lowering_metadata", metadata)

    @staticmethod
    def from_transform(transform: PrimitiveTransformRule) -> PrimitiveContract:
        """Build an immutable primitive contract view from a transform binding."""

        return PrimitiveContract(
            identity=transform.identity,
            derivative_rule=transform.derivative_rule,
            batching_rule=transform.batching_rule,
            lowering_rule=transform.lowering_rule,
            lowering_metadata={}
            if transform.lowering_metadata is None
            else transform.lowering_metadata,
            shape_rule=transform.shape_rule,
            dtype_rule=transform.dtype_rule,
            static_argument_rule=transform.static_argument_rule,
            nondifferentiable_policy=transform.nondifferentiable_policy,
            effect=transform.effect,
        )


class CustomDerivativeRegistry:
    """Conflict-safe registry binding primitive identities to exact rules."""

    def __init__(self, rules: dict[PrimitiveIdentity, CustomDerivativeRule] | None = None) -> None:
        self._rules: dict[PrimitiveIdentity, CustomDerivativeRule] = {}
        self._transforms: dict[PrimitiveIdentity, PrimitiveTransformRule] = {}
        if rules is not None:
            for identity, rule in rules.items():
                self.register(identity, rule)

    def register(
        self,
        identity: PrimitiveIdentity | str,
        rule: CustomDerivativeRule,
        *,
        overwrite: bool = False,
    ) -> CustomDerivativeRule:
        """Register an exact derivative rule for a primitive identity."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        if not isinstance(rule, CustomDerivativeRule):
            raise ValueError("registered custom derivative rule must be a CustomDerivativeRule")
        existing = self._rules.get(primitive_identity)
        if existing is not None and existing != rule and not overwrite:
            raise ValueError(
                f"custom derivative rule already registered for {primitive_identity.key}"
            )
        self._rules[primitive_identity] = rule
        existing_transform = self._transforms.get(primitive_identity)
        if existing_transform is None or existing_transform.derivative_rule != rule:
            self._transforms[primitive_identity] = PrimitiveTransformRule(
                identity=primitive_identity,
                derivative_rule=rule,
                batching_rule=None
                if existing_transform is None
                else existing_transform.batching_rule,
                lowering_rule=None
                if existing_transform is None
                else existing_transform.lowering_rule,
                lowering_metadata={}
                if existing_transform is None
                else existing_transform.lowering_metadata,
                shape_rule=None if existing_transform is None else existing_transform.shape_rule,
                dtype_rule=None if existing_transform is None else existing_transform.dtype_rule,
                static_argument_rule=None
                if existing_transform is None
                else existing_transform.static_argument_rule,
                nondifferentiable_policy="not_declared"
                if existing_transform is None
                else existing_transform.nondifferentiable_policy,
                effect="pure" if existing_transform is None else existing_transform.effect,
            )
        return rule

    def decorator(
        self,
        identity: PrimitiveIdentity | str,
        *,
        overwrite: bool = False,
    ) -> Callable[[CustomDerivativeRule], CustomDerivativeRule]:
        """Return a decorator that registers a CustomDerivativeRule object."""

        def register_rule(rule: CustomDerivativeRule) -> CustomDerivativeRule:
            return self.register(identity, rule, overwrite=overwrite)

        return register_rule

    def register_transform(
        self,
        transform: PrimitiveTransformRule,
        *,
        overwrite: bool = False,
    ) -> PrimitiveTransformRule:
        """Register derivative, batching, and lowering metadata for one primitive."""

        if not isinstance(transform, PrimitiveTransformRule):
            raise ValueError("transform must be a PrimitiveTransformRule")
        existing = self._transforms.get(transform.identity)
        if existing is not None and existing != transform and not overwrite:
            raise ValueError(
                f"primitive transform already registered for {transform.identity.key}"
            )
        self.register(transform.identity, transform.derivative_rule, overwrite=overwrite)
        self._transforms[transform.identity] = transform
        return transform

    def register_batching_rule(
        self,
        identity: PrimitiveIdentity | str,
        batching_rule: PrimitiveBatchingRule,
        *,
        overwrite: bool = False,
    ) -> PrimitiveBatchingRule:
        """Attach a primitive-specific batching rule to an existing identity."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        if not callable(batching_rule):
            raise ValueError("batching_rule must be callable")
        rule = self.require(primitive_identity)
        existing = self._transforms.get(primitive_identity)
        if existing is not None and existing.batching_rule is not None and not overwrite:
            raise ValueError(f"batching rule already registered for {primitive_identity.key}")
        metadata = {} if existing is None else existing.lowering_metadata
        self._transforms[primitive_identity] = PrimitiveTransformRule(
            identity=primitive_identity,
            derivative_rule=rule,
            batching_rule=batching_rule,
            lowering_rule=None if existing is None else existing.lowering_rule,
            lowering_metadata=metadata,
            shape_rule=None if existing is None else existing.shape_rule,
            dtype_rule=None if existing is None else existing.dtype_rule,
            static_argument_rule=None if existing is None else existing.static_argument_rule,
            nondifferentiable_policy="not_declared"
            if existing is None
            else existing.nondifferentiable_policy,
            effect="pure" if existing is None else existing.effect,
        )
        return batching_rule

    def register_lowering_rule(
        self,
        identity: PrimitiveIdentity | str,
        lowering_rule: PrimitiveLoweringRule,
        *,
        overwrite: bool = False,
    ) -> PrimitiveLoweringRule:
        """Attach an executable compiler lowering rule to an existing identity."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        if not callable(lowering_rule):
            raise ValueError("lowering_rule must be callable")
        rule = self.require(primitive_identity)
        existing = self._transforms.get(primitive_identity)
        if existing is not None and existing.lowering_rule is not None and not overwrite:
            raise ValueError(f"lowering rule already registered for {primitive_identity.key}")
        self._transforms[primitive_identity] = PrimitiveTransformRule(
            identity=primitive_identity,
            derivative_rule=rule,
            batching_rule=None if existing is None else existing.batching_rule,
            lowering_rule=lowering_rule,
            lowering_metadata={} if existing is None else existing.lowering_metadata,
            shape_rule=None if existing is None else existing.shape_rule,
            dtype_rule=None if existing is None else existing.dtype_rule,
            static_argument_rule=None if existing is None else existing.static_argument_rule,
            nondifferentiable_policy="not_declared"
            if existing is None
            else existing.nondifferentiable_policy,
            effect="pure" if existing is None else existing.effect,
        )
        return lowering_rule

    def batching_rule_for(self, identity: PrimitiveIdentity | str) -> PrimitiveBatchingRule | None:
        """Return the registered primitive batching rule, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else transform.batching_rule

    def lowering_rule_for(self, identity: PrimitiveIdentity | str) -> PrimitiveLoweringRule | None:
        """Return the registered executable compiler lowering rule, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else transform.lowering_rule

    def shape_rule_for(self, identity: PrimitiveIdentity | str) -> PrimitiveShapeRule | None:
        """Return the registered primitive shape rule, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else transform.shape_rule

    def dtype_rule_for(self, identity: PrimitiveIdentity | str) -> PrimitiveDTypeRule | None:
        """Return the registered primitive dtype rule, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else transform.dtype_rule

    def static_argument_rule_for(
        self, identity: PrimitiveIdentity | str
    ) -> PrimitiveStaticArgumentRule | None:
        """Return the registered primitive static-argument rule, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else transform.static_argument_rule

    def nondifferentiable_policy_for(self, identity: PrimitiveIdentity | str) -> str | None:
        """Return the registered primitive nondifferentiability policy, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else transform.nondifferentiable_policy

    def effect_for(self, identity: PrimitiveIdentity | str) -> str | None:
        """Return the registered primitive effect classification, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else transform.effect

    def contract_for(self, identity: PrimitiveIdentity | str) -> PrimitiveContract | None:
        """Return the unified registered primitive contract, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else PrimitiveContract.from_transform(transform)

    def require_batching_rule(self, identity: PrimitiveIdentity | str) -> PrimitiveBatchingRule:
        """Return a primitive batching rule or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        rule = self.batching_rule_for(primitive_identity)
        if rule is None:
            raise ValueError(f"no batching rule registered for {primitive_identity.key}")
        return rule

    def require_lowering_rule(self, identity: PrimitiveIdentity | str) -> PrimitiveLoweringRule:
        """Return an executable compiler lowering rule or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        rule = self.lowering_rule_for(primitive_identity)
        if rule is None:
            raise ValueError(f"no lowering rule registered for {primitive_identity.key}")
        return rule

    def require_shape_rule(self, identity: PrimitiveIdentity | str) -> PrimitiveShapeRule:
        """Return a primitive shape rule or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        rule = self.shape_rule_for(primitive_identity)
        if rule is None:
            raise ValueError(f"no shape rule registered for {primitive_identity.key}")
        return rule

    def require_dtype_rule(self, identity: PrimitiveIdentity | str) -> PrimitiveDTypeRule:
        """Return a primitive dtype rule or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        rule = self.dtype_rule_for(primitive_identity)
        if rule is None:
            raise ValueError(f"no dtype rule registered for {primitive_identity.key}")
        return rule

    def require_static_argument_rule(
        self, identity: PrimitiveIdentity | str
    ) -> PrimitiveStaticArgumentRule:
        """Return a primitive static-argument rule or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        rule = self.static_argument_rule_for(primitive_identity)
        if rule is None:
            raise ValueError(f"no static argument rule registered for {primitive_identity.key}")
        return rule

    def require_nondifferentiable_policy(self, identity: PrimitiveIdentity | str) -> str:
        """Return a primitive nondifferentiability policy or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        policy = self.nondifferentiable_policy_for(primitive_identity)
        if policy is None or policy == "not_declared":
            raise ValueError(
                f"no nondifferentiable policy registered for {primitive_identity.key}"
            )
        return policy

    def require_effect(self, identity: PrimitiveIdentity | str) -> str:
        """Return a primitive effect classification or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        effect = self.effect_for(primitive_identity)
        if effect is None:
            raise ValueError(f"no effect registered for {primitive_identity.key}")
        return effect

    def require_contract(self, identity: PrimitiveIdentity | str) -> PrimitiveContract:
        """Return a unified primitive contract or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        contract = self.contract_for(primitive_identity)
        if contract is None:
            raise ValueError(f"no primitive contract registered for {primitive_identity.key}")
        return contract

    def require_complete_contract(self, identity: PrimitiveIdentity | str) -> PrimitiveContract:
        """Return a compiler/vectorization-ready primitive contract or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        contract = self.require_contract(primitive_identity)
        missing: list[str] = []
        if contract.batching_rule is None:
            missing.append("batching_rule")
        if contract.lowering_rule is None:
            missing.append("lowering_rule")
        if not contract.lowering_metadata:
            missing.append("lowering_metadata")
        if contract.shape_rule is None:
            missing.append("shape_rule")
        if contract.dtype_rule is None:
            missing.append("dtype_rule")
        if contract.nondifferentiable_policy == "not_declared":
            missing.append("nondifferentiable_policy")
        if not contract.effect:
            missing.append("effect")
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"incomplete primitive contract for {primitive_identity.key}: missing {joined}"
            )
        return contract

    def transform_snapshot(self) -> dict[PrimitiveIdentity, PrimitiveTransformRule]:
        """Return a copy of registered primitive transform bindings."""

        return dict(self._transforms)

    def lookup(self, identity: PrimitiveIdentity | str) -> CustomDerivativeRule | None:
        """Return the registered rule for an identity, if present."""

        return self._rules.get(PrimitiveIdentity.parse(identity))

    def require(self, identity: PrimitiveIdentity | str) -> CustomDerivativeRule:
        """Return the registered rule or fail closed with a useful error."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        rule = self._rules.get(primitive_identity)
        if rule is None:
            raise ValueError(f"no custom derivative rule registered for {primitive_identity.key}")
        return rule

    def unregister(self, identity: PrimitiveIdentity | str) -> CustomDerivativeRule:
        """Remove and return a registered rule."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        try:
            self._transforms.pop(primitive_identity, None)
            return self._rules.pop(primitive_identity)
        except KeyError as exc:
            raise ValueError(
                f"no custom derivative rule registered for {primitive_identity.key}"
            ) from exc

    def snapshot(self) -> dict[PrimitiveIdentity, CustomDerivativeRule]:
        """Return an immutable-by-copy snapshot of registered primitive rules."""

        return dict(self._rules)


DEFAULT_CUSTOM_DERIVATIVE_REGISTRY = CustomDerivativeRegistry()

_PROGRAM_AD_ARRAY_PRIMITIVE_NAMESPACE = "scpn.program_ad.array"
_PROGRAM_AD_ARRAY_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_ARRAY_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_ARRAY_PRIMITIVE_NAMESPACE, name, "1")
    for name in ("getitem", "take")
}

_PROGRAM_AD_SHAPE_PRIMITIVE_NAMESPACE = "scpn.program_ad.shape"
_PROGRAM_AD_SHAPE_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_SHAPE_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_SHAPE_PRIMITIVE_NAMESPACE, name, "1")
    for name in ("reshape", "ravel", "transpose")
}

_PROGRAM_AD_REDUCTION_PRIMITIVE_NAMESPACE = "scpn.program_ad.reduction"
_PROGRAM_AD_REDUCTION_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_REDUCTION_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_REDUCTION_PRIMITIVE_NAMESPACE, name, "1")
    for name in ("sum", "prod", "mean")
}

_PROGRAM_AD_ELEMENTWISE_PRIMITIVE_NAMESPACE = "scpn.program_ad.elementwise"
_PROGRAM_AD_ELEMENTWISE_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_ELEMENTWISE_UNARY_NAMES = (
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
    "negative",
)
_PROGRAM_AD_ELEMENTWISE_BINARY_NAMES = (
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "maximum",
    "minimum",
)
_PROGRAM_AD_ELEMENTWISE_NAMES = (
    *_PROGRAM_AD_ELEMENTWISE_UNARY_NAMES,
    *_PROGRAM_AD_ELEMENTWISE_BINARY_NAMES,
)
_PROGRAM_AD_ELEMENTWISE_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_ELEMENTWISE_PRIMITIVE_NAMESPACE, name, "1")
    for name in _PROGRAM_AD_ELEMENTWISE_NAMES
}

_PROGRAM_AD_PRODUCT_PRIMITIVE_NAMESPACE = "scpn.program_ad.product"
_PROGRAM_AD_PRODUCT_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_PRODUCT_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_PRODUCT_PRIMITIVE_NAMESPACE, name, "1")
    for name in ("dot", "vdot", "matmul")
}

_PROGRAM_AD_CUMULATIVE_PRIMITIVE_NAMESPACE = "scpn.program_ad.cumulative"
_PROGRAM_AD_CUMULATIVE_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_CUMULATIVE_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_CUMULATIVE_PRIMITIVE_NAMESPACE, name, "1")
    for name in ("cumsum", "cumprod", "diff")
}

_PROGRAM_AD_LINALG_PRIMITIVE_NAMESPACE = "scpn.program_ad.linalg"
_PROGRAM_AD_LINALG_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_LINALG_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_LINALG_PRIMITIVE_NAMESPACE, name, "1")
    for name in ("det", "inv", "solve", "matrix_power", "multi_dot")
}


def _program_ad_array_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    raise ValueError(
        "program AD array primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


def _program_ad_array_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    raise ValueError(
        "program AD array primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


def _program_ad_array_derivative_rule(name: str) -> CustomDerivativeRule:
    return CustomDerivativeRule(
        name=f"program_ad_array_{name}_trace_contract",
        value_fn=_program_ad_array_direct_value,
        jvp_rule=_program_ad_array_direct_jvp,
    )


def _program_ad_array_normalise_static_shape(
    primitive_name: str, source_shape: Sequence[int]
) -> tuple[int, ...]:
    shape = tuple(int(dimension) for dimension in source_shape)
    if any(dimension < 0 for dimension in shape):
        raise ValueError(
            f"program AD array {primitive_name} direct rule requires non-negative dimensions"
        )
    return shape


def _program_ad_array_static_size(source_shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in source_shape:
        size *= dimension
    return size


def _program_ad_array_signature(source_shape: tuple[int, ...]) -> str:
    return "scalar" if not source_shape else "x".join(str(dimension) for dimension in source_shape)


def _program_ad_array_vector(
    primitive_name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    expected_size: int,
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array(f"program AD array {primitive_name} {role}", values).reshape(
        -1
    )
    if vector.size != expected_size:
        raise ValueError(
            f"program AD array {primitive_name} direct rule requires {role} "
            f"with {expected_size} values"
        )
    return vector


def _program_ad_array_getitem_flat_indices(
    source_shape: tuple[int, ...], index: object
) -> NDArray[np.int64]:
    _validate_trace_basic_index(index)
    source_indices = np.arange(
        _program_ad_array_static_size(source_shape), dtype=np.int64
    ).reshape(source_shape)
    try:
        selected = source_indices[cast(Any, index)]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "program AD array getitem direct rule requires in-bounds indices"
        ) from exc
    return cast(NDArray[np.int64], np.asarray(selected, dtype=np.int64).reshape(-1))


def _program_ad_array_take_indices(indices: object) -> NDArray[np.int64]:
    raw_indices = np.asarray(indices)
    if raw_indices.dtype.kind not in {"i", "u"}:
        raise ValueError("program AD array take direct rule requires static integer indices")
    return cast(NDArray[np.int64], np.asarray(raw_indices, dtype=np.int64))


def _program_ad_array_take_flat_indices(
    source_shape: tuple[int, ...],
    indices: object,
    axis: int | None,
) -> NDArray[np.int64]:
    source_indices = np.arange(
        _program_ad_array_static_size(source_shape), dtype=np.int64
    ).reshape(source_shape)
    raw_indices = _program_ad_array_take_indices(indices)
    normalised_axis = None if axis is None else _normalise_axis("axis", axis, len(source_shape))
    try:
        selected = np.take(source_indices, raw_indices, axis=normalised_axis, mode="raise")
    except (IndexError, ValueError) as exc:
        raise ValueError("program AD array take direct rule requires in-bounds indices") from exc
    return cast(NDArray[np.int64], np.asarray(selected, dtype=np.int64).reshape(-1))


def _program_ad_array_direct_gather(
    primitive_name: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    flat_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    vector = _program_ad_array_vector(
        primitive_name,
        "values",
        values,
        expected_size=_program_ad_array_static_size(source_shape),
    )
    return _program_ad_float64_vector_result(vector[flat_indices])


def _program_ad_array_direct_gather_jvp(
    primitive_name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    flat_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    _program_ad_array_vector(
        primitive_name,
        "values",
        values,
        expected_size=_program_ad_array_static_size(source_shape),
    )
    tangent_vector = _program_ad_array_vector(
        primitive_name,
        "tangent",
        tangent,
        expected_size=_program_ad_array_static_size(source_shape),
    )
    return _program_ad_float64_vector_result(tangent_vector[flat_indices])


def _program_ad_array_direct_scatter_vjp(
    primitive_name: str,
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    flat_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    source_size = _program_ad_array_static_size(source_shape)
    _program_ad_array_vector(primitive_name, "values", values, expected_size=source_size)
    cotangent_vector = _program_ad_array_vector(
        primitive_name,
        "cotangent",
        cotangent,
        expected_size=int(flat_indices.size),
    )
    result = np.zeros(source_size, dtype=np.float64)
    np.add.at(result, flat_indices, cotangent_vector)
    return result


def program_ad_array_getitem_derivative_rule(
    source_shape: Sequence[int],
    index: object,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed basic-index signature."""

    source = _program_ad_array_normalise_static_shape("getitem", source_shape)
    flat_indices = _program_ad_array_getitem_flat_indices(source, index)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_array_direct_gather(
            "getitem", values, source_shape=source, flat_indices=flat_indices
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_array_direct_gather_jvp(
            "getitem",
            values,
            tangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_array_direct_scatter_vjp(
            "getitem",
            values,
            cotangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    return CustomDerivativeRule(
        name=f"program_ad_array_getitem_{_program_ad_array_signature(source)}_static_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_array_take_derivative_rule(
    source_shape: Sequence[int],
    indices: object,
    *,
    axis: int | None = None,
    mode: str = "raise",
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed NumPy take signature."""

    if mode != "raise":
        raise ValueError("program AD array take direct rule supports only mode='raise'")
    source = _program_ad_array_normalise_static_shape("take", source_shape)
    flat_indices = _program_ad_array_take_flat_indices(source, indices, axis)
    normalised_axis = None if axis is None else _normalise_axis("axis", axis, len(source))
    axis_signature = "flat" if normalised_axis is None else str(normalised_axis)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_array_direct_gather(
            "take", values, source_shape=source, flat_indices=flat_indices
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_array_direct_gather_jvp(
            "take",
            values,
            tangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_array_direct_scatter_vjp(
            "take",
            values,
            cotangent,
            source_shape=source,
            flat_indices=flat_indices,
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_array_take_"
            f"{_program_ad_array_signature(source)}_axis_{axis_signature}_static_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_shape_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    raise ValueError(
        "program AD shape primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


def _program_ad_shape_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    raise ValueError(
        "program AD shape primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


def _program_ad_shape_derivative_rule(name: str) -> CustomDerivativeRule:
    return CustomDerivativeRule(
        name=f"program_ad_shape_{name}_trace_contract",
        value_fn=_program_ad_shape_direct_value,
        jvp_rule=_program_ad_shape_direct_jvp,
    )


def _program_ad_shape_normalise_static_shape(
    primitive_name: str, shape: Sequence[int]
) -> tuple[int, ...]:
    normalised = tuple(int(dimension) for dimension in shape)
    if any(dimension < 0 for dimension in normalised):
        raise ValueError(
            f"program AD shape {primitive_name} direct rule requires non-negative dimensions"
        )
    return normalised


def _program_ad_shape_static_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def _program_ad_shape_signature(shape: tuple[int, ...]) -> str:
    return "scalar" if not shape else "x".join(str(dimension) for dimension in shape)


def _program_ad_shape_vector(
    primitive_name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    expected_size: int,
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array(f"program AD shape {primitive_name} {role}", values).reshape(
        -1
    )
    if vector.size != expected_size:
        raise ValueError(
            f"program AD shape {primitive_name} direct rule requires {role} "
            f"with {expected_size} values"
        )
    return vector


def program_ad_shape_reshape_derivative_rule(
    source_shape: Sequence[int],
    target_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed reshape signature."""

    source = _program_ad_shape_normalise_static_shape("reshape", source_shape)
    target = _program_ad_shape_normalise_static_shape("reshape", target_shape)
    source_size = _program_ad_shape_static_size(source)
    target_size = _program_ad_shape_static_size(target)
    if source_size != target_size:
        raise ValueError(
            "program AD shape reshape direct rule requires source and target "
            "with the same element count"
        )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _program_ad_shape_vector("reshape", "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(vector.reshape(source).reshape(target))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector("reshape", "values", values, expected_size=source_size)
        tangent_vector = _program_ad_shape_vector(
            "reshape", "tangent", tangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(tangent_vector.reshape(source).reshape(target))

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector("reshape", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_shape_vector(
            "reshape", "cotangent", cotangent, expected_size=target_size
        )
        return _program_ad_float64_vector_result(cotangent_vector.reshape(target).reshape(source))

    return CustomDerivativeRule(
        name=(
            "program_ad_shape_reshape_"
            f"{_program_ad_shape_signature(source)}_to_"
            f"{_program_ad_shape_signature(target)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_shape_ravel_derivative_rule(source_shape: Sequence[int]) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed ravel signature."""

    source = _program_ad_shape_normalise_static_shape("ravel", source_shape)
    source_size = _program_ad_shape_static_size(source)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_shape_vector("ravel", "values", values, expected_size=source_size)

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector("ravel", "values", values, expected_size=source_size)
        return _program_ad_shape_vector("ravel", "tangent", tangent, expected_size=source_size)

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector("ravel", "values", values, expected_size=source_size)
        return _program_ad_shape_vector("ravel", "cotangent", cotangent, expected_size=source_size)

    return CustomDerivativeRule(
        name=f"program_ad_shape_ravel_{_program_ad_shape_signature(source)}_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_shape_normalise_static_axes(
    source_shape: tuple[int, ...],
    axes: Sequence[int] | None,
) -> tuple[int, ...]:
    if axes is None:
        return tuple(reversed(range(len(source_shape))))
    normalised = tuple(int(axis) for axis in axes)
    if len(normalised) != len(source_shape) or set(normalised) != set(range(len(source_shape))):
        raise ValueError("program AD shape transpose direct rule requires axes permutation")
    return normalised


def program_ad_shape_transpose_derivative_rule(
    source_shape: Sequence[int],
    axes: Sequence[int] | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed transpose signature."""

    source = _program_ad_shape_normalise_static_shape("transpose", source_shape)
    normalised_axes = _program_ad_shape_normalise_static_axes(source, axes)
    inverse_axes = tuple(int(axis) for axis in np.argsort(normalised_axes))
    source_size = _program_ad_shape_static_size(source)
    target = tuple(source[axis] for axis in normalised_axes)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = _program_ad_shape_vector("transpose", "values", values, expected_size=source_size)
        return _program_ad_float64_vector_result(vector.reshape(source).transpose(normalised_axes))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_shape_vector("transpose", "values", values, expected_size=source_size)
        tangent_vector = _program_ad_shape_vector(
            "transpose", "tangent", tangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            tangent_vector.reshape(source).transpose(normalised_axes)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _program_ad_shape_vector("transpose", "values", values, expected_size=source_size)
        cotangent_vector = _program_ad_shape_vector(
            "transpose", "cotangent", cotangent, expected_size=source_size
        )
        return _program_ad_float64_vector_result(
            cotangent_vector.reshape(target).transpose(inverse_axes)
        )

    axes_signature = "_".join(str(axis) for axis in normalised_axes)
    return CustomDerivativeRule(
        name=(
            "program_ad_shape_transpose_"
            f"{_program_ad_shape_signature(source)}_axes_{axes_signature}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


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


def _program_ad_elementwise_derivative_rule(name: str) -> CustomDerivativeRule:
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
    if name == "matmul":
        return CustomDerivativeRule(
            name="program_ad_product_matmul_direct_rule",
            value_fn=_program_ad_product_matmul_value,
            jvp_rule=_program_ad_product_matmul_jvp,
            vjp_rule=_program_ad_product_matmul_vjp,
        )
    raise ValueError(f"unsupported program AD product primitive {name}")


def _program_ad_cumulative_cumsum_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative cumsum values", values).reshape(-1)
    return np.cumsum(vector).astype(np.float64)


def _program_ad_cumulative_cumsum_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative cumsum values", values).reshape(-1)
    tangent_vector = _as_real_numeric_array(
        "program AD cumulative cumsum tangent", tangent
    ).reshape(-1)
    if tangent_vector.shape != vector.shape:
        raise ValueError("program AD cumulative cumsum tangent shape must match values shape")
    return np.cumsum(tangent_vector).astype(np.float64)


def _program_ad_cumulative_cumsum_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative cumsum values", values).reshape(-1)
    cotangent_vector = _as_real_numeric_array(
        "program AD cumulative cumsum cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.shape != vector.shape:
        raise ValueError("program AD cumulative cumsum cotangent shape must match output shape")
    return _program_ad_float64_vector_result(np.flip(np.cumsum(np.flip(cotangent_vector))))


def _program_ad_cumulative_cumprod_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative cumprod values", values).reshape(-1)
    return np.cumprod(vector).astype(np.float64)


def _program_ad_cumulative_cumprod_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative cumprod values", values).reshape(-1)
    tangent_vector = _as_real_numeric_array(
        "program AD cumulative cumprod tangent", tangent
    ).reshape(-1)
    if tangent_vector.shape != vector.shape:
        raise ValueError("program AD cumulative cumprod tangent shape must match values shape")
    result = np.zeros_like(vector, dtype=np.float64)
    for output_index in range(vector.size):
        total = 0.0
        for tangent_index in range(output_index + 1):
            product = 1.0
            for factor_index in range(output_index + 1):
                product *= (
                    tangent_vector[factor_index]
                    if factor_index == tangent_index
                    else vector[factor_index]
                )
            total += product
        result[output_index] = total
    return result


def _program_ad_cumulative_cumprod_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative cumprod values", values).reshape(-1)
    cotangent_vector = _as_real_numeric_array(
        "program AD cumulative cumprod cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.shape != vector.shape:
        raise ValueError("program AD cumulative cumprod cotangent shape must match output shape")
    result = np.zeros_like(vector, dtype=np.float64)
    for input_index in range(vector.size):
        total = 0.0
        for output_index in range(input_index, vector.size):
            product = 1.0
            for factor_index in range(output_index + 1):
                if factor_index != input_index:
                    product *= vector[factor_index]
            total += cotangent_vector[output_index] * product
        result[input_index] = total
    return result


def _program_ad_cumulative_diff_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative diff values", values).reshape(-1)
    return np.diff(vector).astype(np.float64)


def _program_ad_cumulative_diff_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative diff values", values).reshape(-1)
    tangent_vector = _as_real_numeric_array("program AD cumulative diff tangent", tangent).reshape(
        -1
    )
    if tangent_vector.shape != vector.shape:
        raise ValueError("program AD cumulative diff tangent shape must match values shape")
    return np.diff(tangent_vector).astype(np.float64)


def _program_ad_cumulative_diff_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("program AD cumulative diff values", values).reshape(-1)
    cotangent_vector = _as_real_numeric_array(
        "program AD cumulative diff cotangent", cotangent
    ).reshape(-1)
    if vector.size == 0:
        raise ValueError("program AD cumulative diff direct rule requires at least one value")
    if cotangent_vector.shape != (max(vector.size - 1, 0),):
        raise ValueError("program AD cumulative diff cotangent shape must match output shape")
    result = np.zeros_like(vector, dtype=np.float64)
    if cotangent_vector.size == 0:
        return result
    result[0] = -cotangent_vector[0]
    result[-1] = cotangent_vector[-1]
    if vector.size > 2:
        result[1:-1] = cotangent_vector[:-1] - cotangent_vector[1:]
    return result


def _program_ad_cumulative_derivative_rule(name: str) -> CustomDerivativeRule:
    if name == "cumsum":
        return CustomDerivativeRule(
            name="program_ad_cumulative_cumsum_direct_rule",
            value_fn=_program_ad_cumulative_cumsum_value,
            jvp_rule=_program_ad_cumulative_cumsum_jvp,
            vjp_rule=_program_ad_cumulative_cumsum_vjp,
        )
    if name == "cumprod":
        return CustomDerivativeRule(
            name="program_ad_cumulative_cumprod_direct_rule",
            value_fn=_program_ad_cumulative_cumprod_value,
            jvp_rule=_program_ad_cumulative_cumprod_jvp,
            vjp_rule=_program_ad_cumulative_cumprod_vjp,
        )
    if name == "diff":
        return CustomDerivativeRule(
            name="program_ad_cumulative_diff_direct_rule",
            value_fn=_program_ad_cumulative_diff_value,
            jvp_rule=_program_ad_cumulative_diff_jvp,
            vjp_rule=_program_ad_cumulative_diff_vjp,
        )
    raise ValueError(f"unsupported program AD cumulative primitive {name}")


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
    if mode != "raise":
        raise ValueError("program AD array take shape rule supports only mode='raise'")
    raw_indices = np.asarray(indices)
    if raw_indices.dtype.kind not in {"i", "u"}:
        raise ValueError("program AD array take shape rule requires static integer indices")
    source = np.arange(int(np.prod(_program_ad_array_shape_of(args[0]))), dtype=np.int64).reshape(
        _program_ad_array_shape_of(args[0])
    )
    try:
        selected = np.take(source, raw_indices, axis=axis, mode="raise")
    except (IndexError, ValueError) as exc:
        raise ValueError("program AD array take shape rule indices must be in bounds") from exc
    return tuple(int(dimension) for dimension in np.asarray(selected).shape)


def _program_ad_array_dtype_rule(args: tuple[object, ...]) -> str:
    if not args:
        raise ValueError("program AD array dtype rule requires an array operand")
    return _program_ad_array_dtype_of(args[0])


def _program_ad_shape_reshape_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD shape reshape rule requires array and target shape")
    source_shape = _program_ad_array_shape_of(args[0])
    return _normalise_trace_reshape_shape(args[1], int(np.prod(source_shape)))


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


def _program_ad_reduction_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape = _program_ad_array_shape_of(args[0])
    if int(np.prod(source_shape)) == 0:
        raise ValueError("program AD reduction shape rule requires at least one element")
    axis = _program_ad_reduction_axis(args)
    if axis is None:
        return ()
    normalised_axis = _normalise_axis("axis", axis, len(source_shape))
    return source_shape[:normalised_axis] + source_shape[normalised_axis + 1 :]


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


def _program_ad_product_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) != 2:
        raise ValueError("program AD product dtype rule requires two operands")
    dtypes = tuple(np.dtype(_program_ad_array_dtype_of(arg)) for arg in args)
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
    return (args[1],)


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
    if mode != "raise":
        raise ValueError("program AD array take static rule supports only mode='raise'")
    return (
        tuple(int(index) for index in raw_indices.reshape(-1)),
        None if axis is None else int(axis),
        mode,
    )


def _program_ad_shape_reshape_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 2:
        raise ValueError("program AD shape reshape static rule requires array and target shape")
    source_shape = _program_ad_array_shape_of(args[0])
    return (_normalise_trace_reshape_shape(args[1], int(np.prod(source_shape))),)


def _program_ad_shape_no_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) != 1:
        raise ValueError("program AD shape static rule requires one array")
    return ()


def _program_ad_shape_transpose_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD shape transpose static rule requires array and optional axes")
    source_shape = _program_ad_array_shape_of(args[0])
    axes = _program_ad_shape_normalised_transpose_axes(
        source_shape, args[1] if len(args) == 2 else None
    )
    return () if not axes else (axes,)


def _program_ad_reduction_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    return (_program_ad_reduction_axis(args),)


def _program_ad_elementwise_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD elementwise static rule requires one or two operands")
    return ()


def _program_ad_product_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
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
}

_PROGRAM_AD_ARRAY_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "getitem": _program_ad_array_getitem_static_arguments,
    "take": _program_ad_array_take_static_arguments,
}

_PROGRAM_AD_SHAPE_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "reshape": _program_ad_shape_reshape_shape,
    "ravel": _program_ad_shape_ravel_shape,
    "transpose": _program_ad_shape_transpose_shape,
}

_PROGRAM_AD_SHAPE_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "reshape": _program_ad_shape_reshape_static_arguments,
    "ravel": _program_ad_shape_no_static_arguments,
    "transpose": _program_ad_shape_transpose_static_arguments,
}

_PROGRAM_AD_REDUCTION_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "sum": _program_ad_reduction_shape,
    "prod": _program_ad_reduction_shape,
    "mean": _program_ad_reduction_shape,
}

_PROGRAM_AD_ELEMENTWISE_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    name: _program_ad_elementwise_shape for name in _PROGRAM_AD_ELEMENTWISE_NAMES
}

_PROGRAM_AD_PRODUCT_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "dot": _program_ad_product_dot_shape,
    "vdot": _program_ad_product_vdot_shape,
    "matmul": _program_ad_product_matmul_shape,
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
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff array dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.array.{name}",
        "llvm": "blocked_until_executable_array_lowering",
        "rust": "blocked_until_polyglot_array_ad",
        "static_argument_rule": "required",
        "static_signature": "basic_index" if name == "getitem" else "indices_axis_mode",
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
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff shape dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.shape.{name}",
        "llvm": "blocked_until_executable_shape_lowering",
        "rust": "blocked_until_polyglot_shape_ad",
        "static_argument_rule": "required",
        "static_signature": {
            "reshape": "target_shape",
            "ravel": "none",
            "transpose": "axes",
        }[name],
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
    reduction_axis = _program_ad_reduction_axis(args)
    if reduction_axis is not None:
        reduction_axis = _normalise_axis("reduction axis", reduction_axis, array.ndim)
        if reduction_axis == batch_axis:
            raise ValueError("program AD reduction batching cannot reduce the mapped batch axis")
        if reduction_axis > batch_axis:
            reduction_axis -= 1
    outputs = [
        _as_real_numeric_array(
            "program AD reduction batched output",
            function(np.take(array, batch_index, axis=batch_axis), reduction_axis),
        )
        for batch_index in range(int(array.shape[batch_axis]))
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_reduction_lowering_metadata(name: str) -> Mapping[str, str]:
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff reduction dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.reduction.{name}",
        "llvm": "blocked_until_executable_reduction_lowering",
        "rust": "blocked_until_polyglot_reduction_ad",
        "static_argument_rule": "required",
        "static_signature": "axis",
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
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff elementwise dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.elementwise.{name}",
        "llvm": "blocked_until_executable_elementwise_lowering",
        "rust": "blocked_until_polyglot_elementwise_ad",
        "static_argument_rule": "none",
        "static_signature": "none",
    }


def _program_ad_product_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
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
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff product dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.product.{name}",
        "llvm": "blocked_until_executable_product_lowering",
        "rust": "blocked_until_polyglot_product_ad",
        "static_argument_rule": "none",
        "static_signature": "none",
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
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff cumulative dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.cumulative.{name}",
        "llvm": "blocked_until_executable_cumulative_lowering",
        "rust": "blocked_until_polyglot_cumulative_ad",
        "static_argument_rule": "required",
        "static_signature": "order_axis" if name == "diff" else "axis",
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
                static_argument_rule=_program_ad_reduction_static_arguments,
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


def _require_program_ad_array_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    identity = _PROGRAM_AD_ARRAY_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD array primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_ARRAY_POLICY:
        raise ValueError(f"invalid program AD array primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD array primitive effect for {identity.key}")
    if args is not None:
        _validate_program_ad_primitive_contract_dispatch(contract, args)
    return contract


def _require_program_ad_shape_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    identity = _PROGRAM_AD_SHAPE_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD shape primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_SHAPE_POLICY:
        raise ValueError(f"invalid program AD shape primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD shape primitive effect for {identity.key}")
    if args is not None:
        _validate_program_ad_primitive_contract_dispatch(contract, args)
    return contract


def _require_program_ad_reduction_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    identity = _PROGRAM_AD_REDUCTION_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD reduction primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_REDUCTION_POLICY:
        raise ValueError(f"invalid program AD reduction primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD reduction primitive effect for {identity.key}")
    if args is not None:
        _validate_program_ad_primitive_contract_dispatch(contract, args)
    return contract


def _require_program_ad_elementwise_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    identity = _PROGRAM_AD_ELEMENTWISE_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD elementwise primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_ELEMENTWISE_POLICY:
        raise ValueError(f"invalid program AD elementwise primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD elementwise primitive effect for {identity.key}")
    if args is not None:
        _validate_program_ad_primitive_contract_dispatch(contract, args)
    return contract


def _require_program_ad_product_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    identity = _PROGRAM_AD_PRODUCT_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD product primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_PRODUCT_POLICY:
        raise ValueError(f"invalid program AD product primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD product primitive effect for {identity.key}")
    if args is not None:
        _validate_program_ad_primitive_contract_dispatch(contract, args)
    return contract


def _require_program_ad_cumulative_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    identity = _PROGRAM_AD_CUMULATIVE_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD cumulative primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_CUMULATIVE_POLICY:
        raise ValueError(f"invalid program AD cumulative primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD cumulative primitive effect for {identity.key}")
    if args is not None:
        _validate_program_ad_primitive_contract_dispatch(contract, args)
    return contract


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


_PROGRAM_AD_LINALG_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "det": _program_ad_linalg_det_shape,
    "inv": _program_ad_linalg_inv_shape,
    "solve": _program_ad_linalg_solve_shape,
    "matrix_power": _program_ad_linalg_matrix_power_shape,
    "multi_dot": _program_ad_linalg_multi_dot_shape,
}

_PROGRAM_AD_LINALG_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "det": _program_ad_linalg_no_static_arguments,
    "inv": _program_ad_linalg_no_static_arguments,
    "solve": _program_ad_linalg_no_static_arguments,
    "matrix_power": _program_ad_linalg_matrix_power_static_arguments,
    "multi_dot": _program_ad_linalg_multi_dot_static_arguments,
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
    metadata = {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff linalg dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.linalg.{name}",
        "llvm": "blocked_until_executable_linalg_lowering",
        "rust": "blocked_until_polyglot_linalg_ad",
        "static_argument_rule": "none",
        "static_derivative_factory": "not_required",
        "static_signature": "none",
    }
    if name == "matrix_power":
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
    identity = _PROGRAM_AD_LINALG_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD linalg primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_LINALG_POLICY:
        raise ValueError(f"invalid program AD linalg primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD linalg primitive effect for {identity.key}")
    if args is not None:
        _validate_program_ad_primitive_contract_dispatch(contract, args)
    return contract


_register_program_ad_array_primitive_contracts()
_register_program_ad_shape_primitive_contracts()
_register_program_ad_reduction_primitive_contracts()
_register_program_ad_elementwise_primitive_contracts()
_register_program_ad_product_primitive_contracts()
_register_program_ad_cumulative_primitive_contracts()
_register_program_ad_linalg_primitive_contracts()


def register_custom_derivative_rule(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    overwrite: bool = False,
    registry: CustomDerivativeRegistry | None = None,
) -> CustomDerivativeRule:
    """Register a custom derivative rule in the selected or default registry."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.register(identity, rule, overwrite=overwrite)


def register_primitive_transform_rule(
    transform: PrimitiveTransformRule,
    *,
    overwrite: bool = False,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveTransformRule:
    """Register a combined derivative/batching/lowering transform binding."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.register_transform(transform, overwrite=overwrite)


def register_primitive_batching_rule(
    identity: PrimitiveIdentity | str,
    batching_rule: PrimitiveBatchingRule,
    *,
    overwrite: bool = False,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveBatchingRule:
    """Register a batching rule for an existing primitive derivative rule."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.register_batching_rule(identity, batching_rule, overwrite=overwrite)


def register_primitive_lowering_rule(
    identity: PrimitiveIdentity | str,
    lowering_rule: PrimitiveLoweringRule,
    *,
    overwrite: bool = False,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveLoweringRule:
    """Register an executable compiler lowering rule for an existing primitive."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.register_lowering_rule(identity, lowering_rule, overwrite=overwrite)


def primitive_shape_rule_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveShapeRule:
    """Resolve a primitive shape rule or fail closed."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require_shape_rule(identity)


def primitive_dtype_rule_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveDTypeRule:
    """Resolve a primitive dtype rule or fail closed."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require_dtype_rule(identity)


def primitive_static_argument_rule_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveStaticArgumentRule:
    """Resolve a primitive static-argument rule or fail closed."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require_static_argument_rule(identity)


def primitive_nondifferentiable_policy_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> str:
    """Resolve a primitive nondifferentiability policy or fail closed."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require_nondifferentiable_policy(identity)


def primitive_effect_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> str:
    """Resolve a primitive effect classification or fail closed."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require_effect(identity)


def primitive_contract_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveContract:
    """Resolve a unified primitive transform contract or fail closed."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require_contract(identity)


def primitive_complete_contract_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveContract:
    """Resolve a compiler/vectorization-ready primitive contract or fail closed."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require_complete_contract(identity)


def custom_derivative_rule_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> CustomDerivativeRule:
    """Resolve a custom derivative rule for a primitive identity."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require(identity)


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
        step_vector = cast(NDArray[np.float64], self.learning_rate * natural_gradient_value.copy())
        if self.max_step_norm is not None and np.any(trainable):
            norm = float(np.linalg.norm(step_vector[trainable], ord=2))
            if norm > self.max_step_norm:
                step_vector[trainable] *= self.max_step_norm / norm
        step_vector[~trainable] = 0.0
        return step_vector

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


def _as_parameter_array(values: ArrayLike) -> NDArray[np.float64]:
    array = _as_real_numeric_array("parameters", values)
    if array.ndim != 1:
        raise ValueError("parameters must be a one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError("parameters must contain only finite values")
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
    return cast(NDArray[np.float64], projected)


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
        return cast(NDArray[np.float64], clipped)
    trainable_norm = float(np.linalg.norm(clipped[trainable], ord=2))
    if trainable_norm > max_norm:
        clipped[trainable] *= max_norm / trainable_norm
    return cast(NDArray[np.float64], clipped)


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
    gradient = np.zeros_like(parameter_values)
    base_value = _as_scalar(objective(parameter_values.copy()))
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        plus = parameter_values.copy()
        minus = parameter_values.copy()
        plus[index] += shift_rule.shift
        minus[index] -= shift_rule.shift
        plus_value = _as_scalar(objective(plus))
        minus_value = _as_scalar(objective(minus))
        evaluations += 2
        gradient[index] = shift_rule.coefficient * (plus_value - minus_value)

    return GradientResult(
        value=base_value,
        gradient=gradient,
        method="parameter_shift",
        shift=shift_rule.shift,
        coefficient=shift_rule.coefficient,
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
) -> StochasticGradientResult:
    """Propagate independent shot noise through parameter-shift gradients."""

    plus = _as_parameter_array(plus_values)
    minus = _as_parameter_array(minus_values)
    plus_var = _as_parameter_array(plus_variances)
    minus_var = _as_parameter_array(minus_variances)
    plus_count = _as_parameter_array(plus_shots)
    minus_count = plus_count.copy() if minus_shots is None else _as_parameter_array(minus_shots)
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

    parameter_meta = _normalise_parameters(plus, parameters)
    shift_rule = rule or ParameterShiftRule()
    gradient = np.zeros_like(plus)
    variance = np.zeros_like(plus)
    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        gradient[index] = shift_rule.coefficient * (plus[index] - minus[index])
        variance[index] = shift_rule.coefficient**2 * (
            plus_var[index] / plus_count[index] + minus_var[index] / minus_count[index]
        )
    standard_error = np.sqrt(variance)
    covariance = np.diag(variance)
    return StochasticGradientResult(
        value=value,
        gradient=gradient,
        standard_error=standard_error,
        covariance=covariance,
        confidence_radius=z_value * standard_error,
        shots=np.vstack([plus_count, minus_count]),
        confidence_level=confidence,
        method="parameter_shift_shot_noise",
        shift=shift_rule.shift,
        coefficient=shift_rule.coefficient,
        evaluations=2 * sum(parameter.trainable for parameter in parameter_meta),
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
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

    plus_var = _as_parameter_array(plus_variances)
    minus_var = _as_parameter_array(minus_variances)
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
    parameter_meta = _normalise_parameters(plus_var, parameters)
    shift_rule = rule or ParameterShiftRule()
    shot_plan = np.full((2, plus_var.size), float(min_shots), dtype=np.float64)
    variance = np.zeros_like(plus_var)
    target_variance = target**2
    coefficient_squared = shift_rule.coefficient**2

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        plus_noise = coefficient_squared * plus_var[index]
        minus_noise = coefficient_squared * minus_var[index]
        root_sum = float(np.sqrt(plus_noise) + np.sqrt(minus_noise))
        if root_sum > 0.0:
            plus_required = np.sqrt(plus_noise) * root_sum / target_variance
            minus_required = np.sqrt(minus_noise) * root_sum / target_variance
        else:
            plus_required = float(min_shots)
            minus_required = float(min_shots)
        shot_plan[0, index] = max(float(min_shots), float(np.ceil(plus_required)))
        shot_plan[1, index] = max(float(min_shots), float(np.ceil(minus_required)))
        if max_shots_per_evaluation is not None:
            shot_plan[0, index] = min(shot_plan[0, index], float(max_shots_per_evaluation))
            shot_plan[1, index] = min(shot_plan[1, index], float(max_shots_per_evaluation))
        variance[index] = coefficient_squared * (
            plus_var[index] / shot_plan[0, index] + minus_var[index] / shot_plan[1, index]
        )

    standard_error = np.sqrt(variance)
    return ShotAllocationResult(
        shots=shot_plan,
        predicted_standard_error=standard_error,
        covariance=np.diag(variance),
        target_standard_error=target,
        total_shots=int(np.sum(shot_plan)),
        method="parameter_shift_target_se",
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
) -> GradientResult:
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
    raise ValueError(
        "gradient method must be one of: parameter_shift, finite_difference, complex_step, "
        "forward_mode, reverse_mode"
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
    vjp = cast(NDArray[np.float64], jacobian.jacobian.T @ cotangent_values)
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
    return cast(NDArray[np.float64], metric)


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
    projection = cast(NDArray[np.float64], jacobian.jacobian @ masked_tangent)
    if weights is None:
        weighted_projection = projection
    else:
        weight_arr = _as_real_numeric_array("weights", weights)
        if weight_arr.ndim != 1 or weight_arr.shape[0] != projection.size:
            raise ValueError("weights must be a one-dimensional array matching residual rows")
        if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
            raise ValueError("weights must contain only finite non-negative values")
        weighted_projection = projection * weight_arr
    product = cast(NDArray[np.float64], jacobian.jacobian.T @ weighted_projection)
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
    return cast(NDArray[np.float64], weights)


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
    return cast(NDArray[np.float64], weights)


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
    gradient = cast(NDArray[np.float64], jacobian_arr.T @ weighted_residual)
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
    "FixedPointSensitivityResult",
    "FisherConjugateGradientResult",
    "FisherVectorProductResult",
    "GradientCheckResult",
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
    "ProgramADAdjointResult",
    "ProgramADAliasEdge",
    "ProgramADControlRegion",
    "ProgramADEffect",
    "ProgramADEffectIR",
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
    "SparseMatrixResult",
    "StochasticGradientResult",
    "VJPResult",
    "WeightedGradientResult",
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
    "primitive_complete_contract_for",
    "primitive_contract_for",
    "primitive_dtype_rule_for",
    "primitive_effect_for",
    "primitive_nondifferentiable_policy_for",
    "primitive_shape_rule_for",
    "primitive_static_argument_rule_for",
    "program_ad_array_getitem_derivative_rule",
    "program_ad_array_take_derivative_rule",
    "program_ad_linalg_matrix_power_derivative_rule",
    "program_ad_linalg_multi_dot_derivative_rule",
    "program_ad_reduction_mean_derivative_rule",
    "program_ad_reduction_prod_derivative_rule",
    "program_ad_reduction_sum_derivative_rule",
    "program_ad_shape_ravel_derivative_rule",
    "program_ad_shape_reshape_derivative_rule",
    "program_ad_shape_transpose_derivative_rule",
    "program_adjoint_gradient",
    "program_adjoint_result",
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
    "parameter_shift_gradient_with_uncertainty",
    "update_levenberg_marquardt_damping",
    "weighted_gradient_sum",
    "value_and_grad",
    "whole_program_grad",
    "whole_program_value_and_grad",
    "TraceADArray",
    "TraceADScalar",
    "WholeProgramADResult",
    "WholeProgramBytecodeInstruction",
    "WholeProgramIRNode",
    "WholeProgramSemanticsReport",
    "WholeProgramSourceIRFeature",
    "WholeProgramTraceEvent",
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
