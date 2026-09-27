# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole program trace runtime module
# scpn-quantum-control -- whole-program AD trace runtime metadata
"""Runtime trace-context builders for whole-program automatic differentiation."""

from __future__ import annotations

import json
import linecache
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from threading import Event
from types import FrameType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .dense_budget import DenseAllocationError
from .execution_memory import (
    ExecutionBuffer,
    ExecutionMemoryPlan,
    dataclass_storage_bytes,
    json_encoded_bytes,
    require_execution_memory,
)
from .execution_reservations import ExecutionMemoryReservation, reserve_execution_memory
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
        max_execution_gib: float | None = None,
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
        self._memory_reservation: ExecutionMemoryReservation | None = None
        self._metadata_bytes = 0
        self._retained_buffers: tuple[ExecutionBuffer, ...] = ()
        self._array_storage_sequence = 0
        self._node_metadata_bytes = sum(
            dataclass_storage_bytes(record_type)
            for record_type in (
                WholeProgramIRNode,
                ProgramADSSAValue,
                ProgramADEffect,
                ProgramADAliasEdge,
                ProgramADControlRegion,
                ProgramADPhiNode,
            )
        ) + sys.getsizeof(np.empty(0, dtype=np.float64))
        self._memory_budget_bytes = require_execution_memory(
            self._memory_plan(max(1, parameter_count)), max_gib=max_execution_gib
        ).budget_bytes

    def _memory_plan(
        self, node_count: int, metadata_bytes: int | None = None
    ) -> ExecutionMemoryPlan:
        size = max(1, self.parameter_count)
        node_count = max(node_count, size)
        if metadata_bytes is None:
            metadata_bytes = (
                self._metadata_bytes
                + max(0, node_count - len(self.nodes)) * self._node_metadata_bytes
            )
        return ExecutionMemoryPlan(
            (
                ExecutionBuffer("parameters", "forward", (size,), "float64"),
                ExecutionBuffer("tangent_temporaries", "intermediate", (size,), "float64", 3),
                ExecutionBuffer(
                    "retained_tangents", "adjoint", (size,), "float64", max(1, node_count)
                ),
                ExecutionBuffer("gradient", "dense_output", (size,), "float64"),
                ExecutionBuffer("trace_numeric_validation", "intermediate", (size,), "bool", 2),
                ExecutionBuffer(
                    "trace_frozen_validation_indices", "intermediate", (size,), "intp"
                ),
                ExecutionBuffer(
                    "trace_frozen_validation_values", "intermediate", (size,), "float64"
                ),
                ExecutionBuffer(
                    "trace_metadata", "intermediate", (max(1, metadata_bytes),), "uint8"
                ),
                *self._retained_buffers,
            )
        )

    def _admit_metadata(self, node_count: int, metadata_bytes: int) -> None:
        plan = self._memory_plan(node_count, metadata_bytes)
        if plan.bytes_required > self._memory_budget_bytes:
            raise DenseAllocationError("adjoint tape exceeds admitted execution memory")
        if self._memory_reservation is not None:
            self._memory_reservation.resize(plan)

    @contextmanager
    def array_storage(
        self, shape: tuple[int, ...], *, workspaces: tuple[ExecutionBuffer, ...] = ()
    ) -> Iterator[ExecutionMemoryReservation]:
        """Admit fixed trace-array list/tuple storage before materialisation.

        Parameters
        ----------
        shape
            Positive output dimensions, checked before native-size products.
        workspaces
            Explicit additional numeric buffers used while constructing the array.

        Yields
        ------
        ExecutionMemoryReservation
            Owner for three simultaneous pointer containers and their headers.

        Notes
        -----
        Fixed source list, constructor tuple and retained list coexist briefly.
        Their peak declaration is conservatively retained through result creation;
        it does not measure Python allocator or numerical-kernel internals.

        """
        identity = self._array_storage_sequence
        self._array_storage_sequence += 1
        plan = ExecutionMemoryPlan(
            (
                ExecutionBuffer(f"array_items_{identity}", "forward", shape, "uintp", 3),
                *(
                    ExecutionBuffer(
                        f"array_workspace_{identity}_{buffer.name}",
                        buffer.role,
                        buffer.shape,
                        buffer.dtype,
                        buffer.count,
                    )
                    for buffer in workspaces
                ),
                ExecutionBuffer(
                    f"array_headers_{identity}",
                    "intermediate",
                    (2 * sys.getsizeof([]) + sys.getsizeof(()) + np.dtype(np.uintp).itemsize,),
                    "uint8",
                ),
            )
        )
        combined = ExecutionMemoryPlan(
            (*self._memory_plan(len(self.nodes)).buffers, *plan.buffers)
        )
        if combined.bytes_required > self._memory_budget_bytes:
            raise DenseAllocationError("trace array storage exceeds admitted execution memory")
        with reserve_execution_memory(plan) as reservation:
            reservation.checkpoint()
            yield reservation
            reservation.checkpoint()
            self.retain_buffers(reservation, plan)

    def retain_buffers(
        self, source: ExecutionMemoryReservation, plan: ExecutionMemoryPlan
    ) -> None:
        """Keep generated buffers charged through later result creation."""
        if self._memory_reservation is None:
            return
        destination_plan = self._memory_plan(len(self.nodes))
        source.handoff(
            self._memory_reservation,
            ExecutionMemoryPlan((*destination_plan.buffers, *plan.buffers)),
        )
        self._retained_buffers += plan.buffers

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
        if tangent.shape != (self.parameter_count,):
            raise ValueError("whole-program tangent shape must match parameter count")
        payload_bytes = (
            sys.getsizeof(op) + sys.getsizeof(inputs) + sum(sys.getsizeof(item) for item in inputs)
        )
        metadata_bytes = self._metadata_bytes + self._node_metadata_bytes + payload_bytes
        self._admit_metadata(len(self.nodes) + 1, metadata_bytes)
        node = WholeProgramIRNode(
            index=len(self.nodes),
            op=op,
            inputs=inputs,
            value=value,
            tangent=tangent.copy(),
        )
        self.nodes.append(node)
        self._metadata_bytes = metadata_bytes
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
        for source_index, item in zip(source_indices, items, strict=True):
            if source_index is not None and source_index < 0:
                raise ValueError("program AD view alias source index must be non-negative")
            if item.context is not self:
                raise ValueError("program AD view alias item belongs to a different trace")
        if len(items) > 0:
            payload_bytes = (
                sys.getsizeof(op)
                + max(sys.getsizeof(item.name) for item in items)
                + 4 * sys.getsizeof(str(sys.maxsize))
            )
            metadata_bytes = self._metadata_bytes + dataclass_storage_bytes(
                ProgramADAliasEdge, count=2 * len(items), payload_bytes=payload_bytes
            )
            self._admit_metadata(len(self.nodes), metadata_bytes)
            self._metadata_bytes = metadata_bytes
        base = f"view:{op}:{len(self.alias_edges)}"
        for output_index, (source_index, item) in enumerate(
            zip(source_indices, items, strict=True)
        ):
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
        feature_bytes = sum(
            dataclass_storage_bytes(
                record_type,
                payload_bytes=sys.getsizeof(feature.kind) + sys.getsizeof(feature.detail),
            )
            for feature in source_ir_features
            for record_type in (ProgramADAliasEdge, ProgramADControlRegion, ProgramADPhiNode)
        )
        projection_bytes = (
            self._metadata_bytes
            + feature_bytes
            + sys.getsizeof(())
            + len(bytecode_instructions) * np.dtype(np.uintp).itemsize
        )
        self._admit_metadata(len(self.nodes), self._metadata_bytes + projection_bytes)
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
        encoded_size = json_encoded_bytes(payload)
        serialization_bytes = (
            3 * encoded_size
            + sys.getsizeof(bytearray(1))
            - 1
            + sys.getsizeof(b"")
            + 2 * sys.getsizeof("")
        )
        admitted_metadata = self._metadata_bytes + projection_bytes + serialization_bytes
        self._admit_metadata(len(self.nodes), admitted_metadata)
        self._metadata_bytes = admitted_metadata
        storage = bytearray(encoded_size)
        offset = 0
        for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":")).iterencode(payload):
            if self._memory_reservation is not None:
                self._memory_reservation.checkpoint()
            encoded_chunk = chunk.encode("ascii")
            end = offset + len(encoded_chunk)
            if end > encoded_size:
                raise ValueError("program IR JSON exceeds its admitted encoding size")
            storage[offset:end] = encoded_chunk
            offset = end
        if offset != encoded_size:
            raise ValueError("program IR JSON does not match its admitted encoding size")
        serialization = storage.decode("ascii")
        if self._memory_reservation is not None:
            self._memory_reservation.checkpoint()
        return ProgramADEffectIR(
            ssa_values=tuple(self.ssa_values),
            effects=tuple(self.effects),
            alias_edges=tuple(alias_edges),
            control_regions=tuple(control_regions),
            serialization=serialization,
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
    objective: _ScalarObjective,
    values: NDArray[np.float64],
    *,
    max_execution_gib: float | None = None,
    deadline_monotonic: float | None = None,
    cancelled: Event | None = None,
    context: _WholeProgramTraceContext | None = None,
) -> tuple[WholeProgramTraceEvent, ...]:
    """Execute ``objective`` once with admitted trace events and owned lifecycle."""
    code = getattr(objective, "__code__", None)
    if code is None:
        return ()
    target_filename = code.co_filename
    events: list[WholeProgramTraceEvent] = []
    seen: set[tuple[str, int, str]] = set()
    previous_trace = sys.gettrace()
    metadata_bytes = sys.getsizeof(events) + sys.getsizeof(seen) + sys.getsizeof((None, 0, None))

    def memory_plan(event_bytes: int) -> ExecutionMemoryPlan:
        return ExecutionMemoryPlan(
            (
                ExecutionBuffer("trace_values", "forward", (max(1, values.size),), "float64", 2),
                ExecutionBuffer("trace_events", "intermediate", (event_bytes,), "uint8"),
            )
        )

    with reserve_execution_memory(
        memory_plan(metadata_bytes),
        max_gib=max_execution_gib,
        deadline_monotonic=deadline_monotonic,
        cancelled=cancelled,
    ) as reservation:

        def tracer(frame: FrameType, event: str, arg: object) -> Any:
            nonlocal metadata_bytes
            del arg
            if event == "line":
                reservation.checkpoint()
            if event == "line" and frame.f_code.co_filename == target_filename:
                key = (frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
                if key not in seen:
                    source = linecache.getline(frame.f_code.co_filename, frame.f_lineno)
                    payload_bytes = (
                        sys.getsizeof(source)
                        + sys.getsizeof(key)
                        + sys.getsizeof({key})
                        + sys.getsizeof(frame.f_lineno)
                        + 2 * np.dtype(np.uintp).itemsize
                    )
                    projected_bytes = metadata_bytes + dataclass_storage_bytes(
                        WholeProgramTraceEvent, payload_bytes=payload_bytes
                    )
                    reservation.resize(memory_plan(projected_bytes))
                    metadata_bytes = projected_bytes
                    seen.add(key)
                    events.append(
                        WholeProgramTraceEvent(
                            filename=frame.f_code.co_filename,
                            function_name=frame.f_code.co_name,
                            line_number=frame.f_lineno,
                            source=source,
                        )
                    )
            return tracer

        sys.settrace(tracer)
        try:
            raw = objective(np.array(values, dtype=np.float64, copy=True))
        finally:
            sys.settrace(previous_trace)
        reservation.checkpoint()
        _as_trace_real_scalar("whole-program traced objective", raw)
        trace_events = tuple(events)
        if context is not None:
            context.retain_buffers(reservation, memory_plan(metadata_bytes))
        return trace_events


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
