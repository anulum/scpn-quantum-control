# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- whole-program AD trace runtime metadata
"""Runtime trace-context builders for whole-program automatic differentiation."""

from __future__ import annotations

import json
import linecache
import sys
from collections.abc import Callable, Sequence
from types import FrameType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .program_ad_effect_ir import (
    ProgramADAliasEdge,
    ProgramADControlRegion,
    ProgramADEffect,
    ProgramADEffectIR,
    ProgramADPhiNode,
    ProgramADSSAValue,
)
from .whole_program_ad_result import WholeProgramIRNode, WholeProgramTraceEvent
from .whole_program_frontend import (
    WholeProgramBytecodeInstruction,
    WholeProgramSourceIRFeature,
)

if TYPE_CHECKING:
    from .differentiable import TraceADScalar

_TraceScalarFactory = Callable[
    [float, NDArray[np.float64], "_WholeProgramTraceContext", str], "TraceADScalar"
]
_ScalarObjective = Callable[[NDArray[np.float64]], object]


class _WholeProgramTraceContext:
    """Mutable builder for whole-program AD SSA, effect, and alias metadata."""

    def __init__(
        self,
        parameter_count: int,
        *,
        scalar_factory: _TraceScalarFactory | None = None,
    ) -> None:
        self.parameter_count = parameter_count
        self.nodes: list[WholeProgramIRNode] = []
        self.ssa_values: list[ProgramADSSAValue] = []
        self.effects: list[ProgramADEffect] = []
        self.alias_edges: list[ProgramADAliasEdge] = []
        self.control_regions: list[ProgramADControlRegion] = []
        self.phi_nodes: list[ProgramADPhiNode] = []
        self._value_versions: dict[str, int] = {}
        self._effect_order = 0
        self._scalar_factory = scalar_factory

    def bind_scalar_factory(self, scalar_factory: _TraceScalarFactory) -> None:
        """Bind the trace scalar constructor used by facade-owned scalar wrappers."""
        self._scalar_factory = scalar_factory

    def make(
        self,
        op: str,
        inputs: tuple[str, ...],
        value: float,
        tangent: NDArray[np.float64],
    ) -> TraceADScalar:
        """Create a trace scalar and append its IR node to this AD context."""
        if self._scalar_factory is None:
            raise RuntimeError("whole-program trace context has no scalar factory bound")
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
            operation=op,
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
            region_index = len(self.control_regions)
            selected = "executed_true" if bool(value) else "executed_false"
            self.control_regions.append(
                ProgramADControlRegion(
                    index=region_index,
                    kind="runtime_branch",
                    predicate=op,
                    entered=bool(value),
                    source_line=None,
                )
            )
            self.phi_nodes.append(
                ProgramADPhiNode(
                    index=len(self.phi_nodes),
                    target=f"phi:runtime_branch:{region_index}",
                    incoming=("executed_true", "executed_false"),
                    control_region=region_index,
                    selected=selected,
                    source_line=None,
                )
            )
        return self._scalar_factory(node.value, node.tangent, self, name)

    def record_array_view_aliases(
        self,
        op: str,
        source_indices: Sequence[int | None],
        items: Sequence[TraceADScalar],
    ) -> None:
        """Record deterministic metadata for derivative-preserving array views."""
        if len(source_indices) != len(items):
            raise ValueError("program AD view alias source and item counts must match")
        base = f"view:{op}:{len(self.alias_edges)}"
        for output_index, (source_index, item) in enumerate(
            zip(source_indices, items, strict=True)
        ):
            if source_index is not None and source_index < 0:
                raise ValueError("program AD view alias source index must be non-negative")
            if item.context is not self:
                raise ValueError("program AD view alias item belongs to a different trace")
            view_member = f"{base}[{output_index}]"
            version = len(self.alias_edges)
            if source_index is not None:
                self.alias_edges.append(
                    ProgramADAliasEdge(
                        source=f"%array[{source_index}]",
                        target=view_member,
                        kind="view_alias",
                        version=version,
                    )
                )
            self.alias_edges.append(
                ProgramADAliasEdge(
                    source=view_member,
                    target=item.name,
                    kind="view_alias",
                    version=version,
                )
            )

    def program_ir(
        self,
        *,
        source_ir_features: tuple[WholeProgramSourceIRFeature, ...],
        bytecode_instructions: tuple[WholeProgramBytecodeInstruction, ...],
    ) -> ProgramADEffectIR:
        """Build deterministic SSA/effect IR metadata from captured trace evidence."""
        alias_edges = list(self.alias_edges)
        control_regions = list(self.control_regions)
        phi_nodes = list(self.phi_nodes)
        for feature in source_ir_features:
            if feature.kind in {
                "control_path_alias",
                "expression_rebinding_alias",
                "list_alias",
                "local_rebinding_alias",
                "loop_carried_state",
                "object_attribute_alias",
            }:
                source, separator, target = feature.detail.partition("->")
                if not separator or not source or not target:
                    raise ValueError(
                        f"program AD {feature.kind} feature must encode source->target"
                    )
                alias_edges.append(
                    ProgramADAliasEdge(
                        source=source,
                        target=target,
                        kind=feature.kind,
                        version=len(alias_edges),
                    )
                )
                continue
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
                region_index = len(control_regions)
                control_regions.append(
                    ProgramADControlRegion(
                        index=region_index,
                        kind=f"source_{feature.kind}",
                        predicate=feature.detail,
                        entered=True,
                        source_line=feature.line_number,
                    )
                )
                if "loop" in feature.kind:
                    incoming = ("loop_entry", "loop_backedge")
                    selected = "executed_loop_trace"
                else:
                    incoming = ("executed_path", "non_executed_path")
                    selected = "executed_path"
                phi_nodes.append(
                    ProgramADPhiNode(
                        index=len(phi_nodes),
                        target=f"phi:source:{feature.kind}:{feature.line_number}",
                        incoming=incoming,
                        control_region=region_index,
                        selected=selected,
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
                    "operation": effect.operation,
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
            "phi_nodes": [
                {
                    "index": phi.index,
                    "target": phi.target,
                    "incoming": phi.incoming,
                    "control_region": phi.control_region,
                    "selected": phi.selected,
                    "source_line": phi.source_line,
                }
                for phi in phi_nodes
            ],
            "bytecode_offsets": tuple(instruction.offset for instruction in bytecode_instructions),
        }
        return ProgramADEffectIR(
            ssa_values=tuple(self.ssa_values),
            effects=tuple(self.effects),
            alias_edges=tuple(alias_edges),
            control_regions=tuple(control_regions),
            serialization=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            phi_nodes=tuple(phi_nodes),
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


def _trace_whole_program_objective(
    objective: _ScalarObjective, values: NDArray[np.float64]
) -> tuple[WholeProgramTraceEvent, ...]:
    """Execute ``objective`` once and capture source-line trace events."""
    code = getattr(objective, "__code__", None)
    if code is None:
        return ()
    target_filename = code.co_filename
    events: list[WholeProgramTraceEvent] = []
    seen: set[tuple[str, int, str]] = set()
    previous_trace = sys.gettrace()

    def tracer(frame: FrameType, event: str, arg: object) -> Any:
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
    _as_trace_real_scalar("whole-program traced objective", raw)
    return tuple(events)


def _as_trace_real_scalar(name: str, value: object) -> float:
    """Return an explicit finite real scalar for traced objective validation."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real numeric scalar")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must be a real numeric scalar")
    scalar = float(raw)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar
