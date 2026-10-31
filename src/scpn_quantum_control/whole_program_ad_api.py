# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole program AD API module
# scpn-quantum-control -- whole-program automatic differentiation API
"""Public whole-program automatic differentiation entry points."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import Event
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .canonical_encoding import canonical_digest
from .dense_budget import DenseAllocationError
from .differentiable_parameter_contracts import Parameter, _as_parameter_array
from .differentiable_transform_helpers import _normalise_parameters
from .execution_memory import ExecutionBuffer, ExecutionMemoryPlan, dataclass_storage_bytes
from .execution_reservations import ExecutionMemoryReservation, reserve_execution_memory
from .program_ad_adjoint_generation import _program_adjoint_result_from_nodes
from .program_ad_captured_state import _capture_program_state, _portable_program_state_digest
from .program_ad_tape_binding import (
    _TAPE_DIGEST_BYTES,
    _bind_program_tape,
    _require_bound_program_tape,
)
from .whole_program_ad_result import WholeProgramADResult
from .whole_program_frontend import (
    WholeProgramCompilerFrontendReport,
    WholeProgramUnsupportedSemanticDiagnostic,
    _objective_source,
    _whole_program_semantics_report,
    compile_whole_program_frontend,
)
from .whole_program_trace_runtime import (
    _trace_whole_program_objective,
    _WholeProgramTraceContext,
)
from .whole_program_trace_values import (
    ScalarObjective,
    TraceADArray,
    TraceADScalar,
)


def whole_program_value_and_grad(
    objective: Callable[[Any], object],
    values: ArrayLike,
    parameters: Sequence[Parameter] | None = None,
    *,
    trace: bool = True,
    max_execution_gib: float | None = None,
    deadline_monotonic: float | None = None,
    cancelled: Event | None = None,
) -> WholeProgramADResult:
    """Differentiate an executed Python/NumPy program by operator-intercepted AD.

    Parameters
    ----------
    objective:
        Callable that returns a whole-program AD scalar when executed over
        trace-aware parameter values.
    values:
        One-dimensional real numeric values supplied as a plain NumPy ndarray,
        list, tuple or range. Opaque array protocols and subclasses refuse before
        conversion because their allocation behavior is not bounded here.
    parameters:
        Optional plain list or tuple of exact Parameter records with plain
        string names and boolean trainability. Metadata copies and name storage
        are admitted before copying; opaque sequences and record subclasses
        refuse without invoking their iteration or attribute protocols.
    trace:
        Whether to collect runtime trace events in addition to IR metadata.
    max_execution_gib:
        Optional cap for declared input/conversion, parameter, tangent, retained
        tape, gradient and trace-record buffers. Input storage is admitted before
        NumPy conversion; capacity is observed before initial tangent allocation;
        retained tangent and alias growth refuse before their materialisation.
        Frontend/serialization workspaces, general allocator overhead and
        undeclared user/third-party allocations are not fully covered by this cap.
    deadline_monotonic:
        Absolute monotonic deadline checked before frontend inspection, after
        source inspection and at retained-node checkpoints.
    cancelled:
        Optional cancellation event observed before frontend inspection, after
        source inspection and at retained-node checkpoints.

    Returns
    -------
    WholeProgramADResult
        Exact executed-program value, gradient, source/bytecode metadata, IR
        nodes, frontend report, semantics report, and scalar adjoint replay
        provenance. An in-memory binding checks callable code and captured
        numeric state, and retains the primal, numeric buffers, typed/raw IR and
        adjoint content together. Derivative access refuses changed artifacts.

    Raises
    ------
    ValueError
        If the objective is not callable, fails the source/bytecode frontend
        execution gate, uses unsupported Python semantics, or does not return a
        traceable scalar; also if captured state has unsupported storage or
        changes while the objective is evaluated.

    """
    if not callable(objective):
        raise ValueError("whole-program objective must be callable")
    input_plan, input_count = _parameter_input_memory_plan(values)
    with reserve_execution_memory(
        input_plan,
        max_gib=max_execution_gib,
        deadline_monotonic=deadline_monotonic,
        cancelled=cancelled,
    ) as reservation:
        parameter_values = _as_parameter_array(
            _bounded_parameter_input(
                cast(NDArray[Any] | list[Any] | tuple[Any, ...] | range, values),
                input_count,
                input_plan,
                reservation,
            )
        )
        reservation.checkpoint()
        context = _WholeProgramTraceContext(
            parameter_values.size,
            scalar_factory=TraceADScalar,
            max_execution_gib=max_execution_gib,
        )
        context._retained_buffers = input_plan.buffers
        reservation.resize(context._memory_plan(max(1, parameter_values.size)))
        context._memory_reservation = reservation
        try:
            parameter_meta = _bounded_parameter_metadata(parameter_values, parameters, context)
            frontend_report = compile_whole_program_frontend(objective)
            reservation.checkpoint()
            _require_whole_program_frontend_execution_ready(frontend_report)
            source = _objective_source(objective, context=context)
            reservation.checkpoint()
            captured_state = _capture_program_state(objective, reservation.checkpoint)
            traced_values: list[TraceADScalar] = []
            for index, (value, parameter) in enumerate(
                zip(parameter_values, parameter_meta, strict=True)
            ):
                tangent = np.zeros(parameter_values.size, dtype=np.float64)
                if parameter.trainable:
                    tangent[index] = 1.0
                traced_values.append(
                    context.make("parameter", (parameter.name,), float(value), tangent)
                )
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
                    raise ValueError(
                        "whole-program objective must return a whole-program AD scalar"
                    )
                raw = raw.item()
            if not isinstance(raw, TraceADScalar):
                raise ValueError("whole-program objective must return a whole-program AD scalar")
            captured_state.require_current(reservation.checkpoint)
            reservation.checkpoint()
            trace_events = (
                _trace_whole_program_objective(
                    cast(ScalarObjective, objective),
                    parameter_values,
                    max_execution_gib=max_execution_gib,
                    deadline_monotonic=deadline_monotonic,
                    cancelled=cancelled,
                    context=context,
                )
                if trace
                else ()
            )
            semantics_report = _whole_program_semantics_report(
                bytecode_instructions=frontend_report.bytecode_instructions,
                source_ir_features=frontend_report.source_ir_features,
                trace_events=trace_events,
                source=source,
                accepted_python_semantics=frontend_report.semantics_report.accepted_python_semantics,
                unsupported_python_semantics=(
                    frontend_report.semantics_report.unsupported_python_semantics
                ),
                numpy_observed=frontend_report.semantics_report.numpy_observed
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
                source_ir_features=frontend_report.source_ir_features,
                bytecode_instructions=frontend_report.bytecode_instructions,
            )
            reservation.checkpoint()
            adjoint_result = _program_adjoint_result_from_nodes(
                nodes=tuple(context.nodes),
                output_name=raw.name,
                parameter_names=tuple(parameter.name for parameter in parameter_meta),
                trainable=tuple(parameter.trainable for parameter in parameter_meta),
                program_ir=program_ir,
                max_execution_gib=max_execution_gib,
                deadline_monotonic=deadline_monotonic,
                cancelled=cancelled,
                context=context,
                captured_state=captured_state,
            )
            reservation.checkpoint()
            captured_state.require_current(reservation.checkpoint)
            ad_result = WholeProgramADResult(
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
                bytecode_instructions=frontend_report.bytecode_instructions,
                source_ir_features=frontend_report.source_ir_features,
                semantics_report=semantics_report,
                program_ir=program_ir,
                adjoint_result=adjoint_result,
                frontend_report=frontend_report,
                captured_state=captured_state,
            )
            reservation.checkpoint()
            tape_plan = ExecutionMemoryPlan(
                (
                    ExecutionBuffer(
                        "captured_derivative_digest",
                        "adjoint",
                        (_TAPE_DIGEST_BYTES,),
                        "uint8",
                    ),
                )
            )
            with reserve_execution_memory(tape_plan) as tape_reservation:
                _bind_program_tape(ad_result, tape_reservation.checkpoint)
                context.retain_buffers(tape_reservation, tape_plan)
            captured_state.require_current(reservation.checkpoint)
            return ad_result
        finally:
            context._memory_reservation = None


def _bounded_parameter_metadata(
    values: NDArray[np.float64],
    parameters: Sequence[Parameter] | None,
    context: _WholeProgramTraceContext,
) -> tuple[Parameter, ...]:
    """Snapshot inspectable metadata within the active trace's storage owner."""
    count = int(values.size)
    if parameters is not None:
        if type(parameters) not in (list, tuple):
            raise ValueError("parameter metadata requires a plain list or tuple")
        if len(parameters) != count:
            raise ValueError("parameters length must match values length")
    slots = max(1, count)
    fixed_buffers = (
        ExecutionBuffer(
            "parameter_metadata_records",
            "forward",
            (dataclass_storage_bytes(Parameter, count=slots + 1),),
            "uint8",
        ),
        ExecutionBuffer("parameter_metadata_references", "intermediate", (slots,), "uintp", 4),
        ExecutionBuffer("parameter_metadata_uniqueness", "intermediate", (slots,), "uintp", 8),
        ExecutionBuffer(
            "parameter_metadata_headers",
            "intermediate",
            (sys.getsizeof([]) + sys.getsizeof(()) + sys.getsizeof(set()),),
            "uint8",
            3,
        ),
    )
    name_bytes = 1
    if parameters is None:
        width = len("theta_") + len(str(max(0, count - 1)))
        name_bytes += count * (sys.getsizeof("\U00010000") + 4 * width)
    plan = ExecutionMemoryPlan(
        (
            *fixed_buffers,
            ExecutionBuffer(
                "parameter_metadata_names",
                "forward",
                (name_bytes,),
                "uint8",
            ),
        )
    )
    with reserve_execution_memory(plan) as reservation:
        reservation.checkpoint()
        snapshot: list[Parameter] = [Parameter("theta")] * count
        for index in range(count):
            reservation.checkpoint()
            if parameters is None:
                item = Parameter(f"theta_{index}")
            else:
                try:
                    source = parameters[index]
                except IndexError as error:
                    raise ValueError("parameter metadata changed after admission") from error
                if type(source) is not Parameter:
                    raise ValueError("parameter metadata requires exact Parameter records")
                name, trainable = source.name, source.trainable
                if type(name) is not str or not name or type(trainable) is not bool:
                    raise ValueError(
                        "parameter metadata requires plain names and boolean trainability"
                    )
                name_bytes += sys.getsizeof(name)
                plan = ExecutionMemoryPlan(
                    (
                        *fixed_buffers,
                        ExecutionBuffer(
                            "parameter_metadata_names",
                            "forward",
                            (name_bytes,),
                            "uint8",
                        ),
                    )
                )
                reservation.resize(plan)
                item = Parameter(name, trainable)
            snapshot[index] = item
        if parameters is not None and len(parameters) != count:
            raise ValueError("parameter metadata changed after admission")
        result = _normalise_parameters(values, tuple(snapshot))
        reservation.checkpoint()
        context.retain_buffers(reservation, plan)
        return result


def _parameter_input_memory_plan(values: ArrayLike) -> tuple[ExecutionMemoryPlan, int]:
    """Declare inspectable input and conversion storage without constructing an array."""
    if isinstance(values, np.ndarray):
        if type(values) is not np.ndarray:
            raise ValueError("parameter ndarray subclasses have no bounded conversion contract")
        if values.ndim != 1:
            raise ValueError("parameters must be a one-dimensional sequence")
        if values.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("parameters must contain real numeric scalars")
        count = int(values.size)
        itemsize = max(8, values.dtype.itemsize)
    elif isinstance(values, (list, tuple, range)):
        if type(values) not in (list, tuple, range):
            raise ValueError("parameter sequence subclasses have no bounded conversion contract")
        try:
            count = len(values)
        except OverflowError as exc:
            raise DenseAllocationError("parameter count exceeds native addressability") from exc
        itemsize = max(np.dtype(np.float64).itemsize, np.dtype(np.longdouble).itemsize)
    else:
        raise ValueError("parameters require an inspectable ndarray, list, tuple or range")
    size = max(1, count)
    plan = ExecutionMemoryPlan(
        (
            ExecutionBuffer("parameter_input", "forward", (size, itemsize), "uint8"),
            ExecutionBuffer("parameter_conversion", "intermediate", (size,), "float64", 2),
            ExecutionBuffer("parameter_finite_mask", "intermediate", (size,), "bool"),
            ExecutionBuffer("parameter_sequence_snapshot", "intermediate", (size,), "intp"),
            ExecutionBuffer(
                "parameter_sequence_header", "intermediate", (sys.getsizeof([]),), "uint8"
            ),
            ExecutionBuffer(
                "parameter_array_headers",
                "intermediate",
                (sys.getsizeof(np.empty(0, dtype=np.float64)),),
                "uint8",
                3,
            ),
        )
    )

    return plan, count


def _bounded_parameter_input(
    values: NDArray[Any] | list[Any] | tuple[Any, ...] | range,
    count: int,
    plan: ExecutionMemoryPlan,
    reservation: ExecutionMemoryReservation,
) -> ArrayLike:
    """Snapshot the exact builtin input types already validated by input admission."""
    if isinstance(values, np.ndarray):
        snapshot = np.empty(count, dtype=np.float64)
        if (
            values.shape != (count,)
            or values.dtype.kind not in {"i", "u", "f"}
            or values.dtype.itemsize > plan.buffers[0].shape[1]
        ):
            raise ValueError("parameter input changed after admission")
        np.copyto(snapshot, values, casting="unsafe")
        reservation.checkpoint()
        return snapshot
    copied: list[Any] = [None] * count
    if len(values) != count:
        raise ValueError("parameter input changed after admission")
    for index in range(count):
        reservation.checkpoint()
        try:
            value = values[index]
        except IndexError as exc:
            raise ValueError("parameter input changed after admission") from exc
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError("parameters must contain real numeric scalars")
        if isinstance(value, np.generic) and value.dtype.itemsize > plan.buffers[0].shape[1]:
            raise ValueError("parameter input changed after admission")
        copied[index] = value
    if len(values) != count:
        raise ValueError("parameter input changed after admission")
    return copied


def whole_program_grad(
    objective: Callable[[Any], object],
    values: ArrayLike,
    parameters: Sequence[Parameter] | None = None,
    *,
    trace: bool = True,
    max_execution_gib: float | None = None,
    deadline_monotonic: float | None = None,
    cancelled: Event | None = None,
) -> NDArray[np.float64]:
    """Return only the exact whole-program AD gradient.

    Parameters
    ----------
    objective:
        Callable accepted by :func:`whole_program_value_and_grad`.
    values:
        Initial parameter values.
    parameters:
        Optional metadata that marks trainable parameters and supplies names.
    trace:
        Whether to collect runtime trace events in the underlying result.
    max_execution_gib:
        Optional declared numeric-buffer cap forwarded to the owned AD scope.
    deadline_monotonic:
        Absolute monotonic deadline forwarded to the owned AD scope.
    cancelled:
        Optional cancellation event forwarded to the owned AD scope.

    Returns
    -------
    numpy.ndarray
        Exact whole-program AD gradient as ``float64`` values.

    """
    return whole_program_value_and_grad(
        objective,
        values,
        parameters=parameters,
        trace=trace,
        max_execution_gib=max_execution_gib,
        deadline_monotonic=deadline_monotonic,
        cancelled=cancelled,
    ).gradient


def _require_whole_program_frontend_execution_ready(
    report: WholeProgramCompilerFrontendReport,
) -> None:
    """Reject objective execution unless the source/bytecode frontend is complete."""
    if report.frontend_ready:
        return
    hard_gaps = ", ".join(report.hard_gaps) or "frontend_not_ready"
    details = (
        "whole-program AD frontend execution gate rejected objective: "
        f"function={report.function_name}; frontend_digest={report.frontend_digest}; "
        f"hard_gaps=[{hard_gaps}]"
    )
    if report.unsupported_semantic_diagnostics:
        diagnostics = "; ".join(
            _format_unsupported_frontend_diagnostic(diagnostic)
            for diagnostic in report.unsupported_semantic_diagnostics
        )
        details = f"{details}; unsupported_diagnostics=[{diagnostics}]"
    raise ValueError(details)


def _format_unsupported_frontend_diagnostic(
    diagnostic: WholeProgramUnsupportedSemanticDiagnostic,
) -> str:
    """Return a deterministic one-line unsupported-semantics diagnostic."""
    regions = ",".join(diagnostic.region_ids) or "<none>"
    offsets = ",".join(str(offset) for offset in diagnostic.bytecode_offsets) or "<none>"
    return (
        f"semantic={diagnostic.semantic} detail={diagnostic.detail} "
        f"line={diagnostic.line_number} absolute_line={diagnostic.absolute_line_number} "
        f"regions=[{regions}] bytecode_offsets=[{offsets}]"
    )


WHOLE_PROGRAM_REPLAY_SCHEMA = "whole_program_replay_record.v1"
"""Schema and major version of a saved whole-program replay record."""

WHOLE_PROGRAM_REPLAY_RANDOMNESS = (
    "ambient and generator random draws are refused inside the objective; random "
    "values enter only as parameters or captured numeric state"
)
"""How a replay record binds randomness: no draw can happen during execution."""

_REPLAY_ABSOLUTE_BUDGET = 1.0e-10
_REPLAY_RELATIVE_BUDGET = 1.0e-8
_REPLAY_FIELDS = (
    "interpreter",
    "numpy_version",
    "parameter_names",
    "trainable",
    "parameter_values",
    "trace",
    "program_ir_sha256",
    "captured_state_sha256",
    "effects",
    "adjoint_step_count",
    "value",
    "gradient",
)


class WholeProgramReplayMismatchError(ValueError):
    """A replay disagreed with its saved record; ``fields`` names every difference."""

    def __init__(self, fields: tuple[str, ...], detail: str) -> None:
        """Record the mismatched record fields and a readable explanation.

        Parameters
        ----------
        fields:
            Record field names whose saved and replayed values differ.
        detail:
            Human-readable description of the refusal.

        """
        super().__init__(f"whole-program replay mismatch [{', '.join(fields)}]: {detail}")
        self.fields = fields


@dataclass(frozen=True)
class WholeProgramReplayRecord:
    """Saved identity and derivatives of one differentiated program execution.

    The record binds the program graph (SHA-256 of the effect IR, which also
    fixes the executed branches), the captured state (a process-independent
    digest of the callable, its closures, defaults, globals and numeric
    storage), the ordered effect history, the transform plan (method, parameter
    order, trainable mask, trace flag and adjoint step count) and the runtime
    that produced it. Parameter values, value and gradient are exact float64.

    Attributes
    ----------
    interpreter:
        Interpreter cache tag such as ``cpython-312``; code hashing is only
        comparable within one interpreter version.
    numpy_version:
        NumPy version whose arithmetic produced the saved numbers.
    parameter_names:
        Parameter names in tangent order.
    trainable:
        Trainable mask in tangent order; frozen coordinates have zero gradient.
    parameter_values:
        Exact float64 parameter values the program was executed at.
    trace:
        Whether source-line trace evidence was collected (a second execution).
    program_ir_sha256:
        SHA-256 of the ``program_ad_effect_ir.v1`` serialization.
    captured_state_sha256:
        Portable SHA-256 digest of the captured callable state.
    effects:
        Ordered ``(kind, operation)`` pairs of the effect IR; an effect without
        a recorded operation has ``None``.
    adjoint_step_count:
        Number of generated reverse-adjoint steps, or ``None`` when the
        program has no supported reverse adjoint and only forward mode applies.
    value:
        Exact float64 objective value.
    gradient:
        Exact float64 forward-mode gradient.

    """

    interpreter: str
    numpy_version: str
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    parameter_values: tuple[float, ...]
    trace: bool
    program_ir_sha256: str
    captured_state_sha256: str
    effects: tuple[tuple[str, str | None], ...]
    adjoint_step_count: int | None
    value: float
    gradient: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate exact field types, lengths, digests and finite numbers."""
        for name in ("interpreter", "numpy_version", "program_ir_sha256", "captured_state_sha256"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"replay record {name} must be a non-empty string")
        for name in ("program_ir_sha256", "captured_state_sha256"):
            digest = getattr(self, name)
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"replay record {name} must be a lowercase SHA-256 hex digest")
        if (
            type(self.parameter_names) is not tuple
            or any(type(item) is not str or not item for item in self.parameter_names)
            or len(set(self.parameter_names)) != len(self.parameter_names)
        ):
            raise ValueError("replay record parameter_names must be unique non-empty strings")
        count = len(self.parameter_names)
        if type(self.trainable) is not tuple or any(
            type(item) is not bool for item in self.trainable
        ):
            raise ValueError("replay record trainable must be a tuple of booleans")
        if type(self.trace) is not bool:
            raise ValueError("replay record trace must be a boolean")
        for name in ("parameter_values", "gradient"):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(item) is not float or not math.isfinite(item) for item in values
            ):
                raise ValueError(f"replay record {name} must be a tuple of finite floats")
        if not (len(self.trainable) == len(self.parameter_values) == len(self.gradient) == count):
            raise ValueError("replay record parameter fields must have one entry per parameter")
        if type(self.value) is not float or not math.isfinite(self.value):
            raise ValueError("replay record value must be a finite float")
        if type(self.effects) is not tuple or any(
            type(effect) is not tuple
            or len(effect) != 2
            or type(effect[0]) is not str
            or not effect[0]
            or (effect[1] is not None and (type(effect[1]) is not str or not effect[1]))
            for effect in self.effects
        ):
            raise ValueError("replay record effects must be (kind, operation) pairs")
        if self.adjoint_step_count is not None and (
            type(self.adjoint_step_count) is not int or self.adjoint_step_count < 0
        ):
            raise ValueError("replay record adjoint_step_count must be a non-negative integer")

    @property
    def digest(self) -> str:
        """Return the canonical SHA-256 of the whole record under its schema."""
        return canonical_digest(WHOLE_PROGRAM_REPLAY_SCHEMA, self._body())

    def to_json(self) -> str:
        """Serialize the record with exact float64 hex values and its digest.

        Returns
        -------
        str
            Compact JSON object with ``schema``, ``digest`` and ``record`` keys.

        """
        body = self._body()
        wire: dict[str, object] = {
            **body,
            "parameter_values": [_float_hex(item) for item in self.parameter_values],
            "value": _float_hex(self.value),
            "gradient": [_float_hex(item) for item in self.gradient],
        }
        return json.dumps(
            {"schema": WHOLE_PROGRAM_REPLAY_SCHEMA, "digest": self.digest, "record": wire},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> WholeProgramReplayRecord:
        """Parse a saved record and refuse unknown versions, keys or a wrong digest.

        Parameters
        ----------
        text:
            JSON produced by :meth:`to_json`.

        Returns
        -------
        WholeProgramReplayRecord
            The validated record whose recomputed digest equals the saved one.

        Raises
        ------
        ValueError
            If the text is not the supported schema, has missing or unknown
            keys, malformed values, or a digest that does not match its content.

        """
        if type(text) is not str:
            raise ValueError("replay record JSON must be a string")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("replay record JSON is malformed") from error
        if type(payload) is not dict or set(payload) != {"schema", "digest", "record"}:
            raise ValueError("replay record JSON must contain exactly schema, digest and record")
        if payload["schema"] != WHOLE_PROGRAM_REPLAY_SCHEMA:
            raise ValueError(f"unsupported replay record schema: {payload['schema']!r}")
        wire = payload["record"]
        if type(wire) is not dict or set(wire) != set(_REPLAY_FIELDS):
            raise ValueError("replay record fields do not match the v1 schema")
        record = cls(
            interpreter=wire["interpreter"],
            numpy_version=wire["numpy_version"],
            parameter_names=tuple(_json_list(wire["parameter_names"])),
            trainable=tuple(_json_list(wire["trainable"])),
            parameter_values=tuple(
                _float_from_hex(item) for item in _json_list(wire["parameter_values"])
            ),
            trace=wire["trace"],
            program_ir_sha256=wire["program_ir_sha256"],
            captured_state_sha256=wire["captured_state_sha256"],
            effects=tuple(tuple(_json_list(effect)) for effect in _json_list(wire["effects"])),
            adjoint_step_count=wire["adjoint_step_count"],
            value=_float_from_hex(wire["value"]),
            gradient=tuple(_float_from_hex(item) for item in _json_list(wire["gradient"])),
        )
        if payload["digest"] != record.digest:
            raise ValueError("replay record digest does not match its content")
        return record

    def _body(self) -> dict[str, object]:
        return {
            "interpreter": self.interpreter,
            "numpy_version": self.numpy_version,
            "parameter_names": list(self.parameter_names),
            "trainable": list(self.trainable),
            "parameter_values": list(self.parameter_values),
            "trace": self.trace,
            "program_ir_sha256": self.program_ir_sha256,
            "captured_state_sha256": self.captured_state_sha256,
            "effects": [list(effect) for effect in self.effects],
            "adjoint_step_count": self.adjoint_step_count,
            "value": self.value,
            "gradient": list(self.gradient),
        }


@dataclass(frozen=True)
class WholeProgramReplayReport:
    """Agreement between a saved replay record and a fresh execution.

    Attributes
    ----------
    record:
        Record captured from the replayed execution.
    value_error:
        Absolute difference between saved and replayed values.
    gradient_error:
        Largest absolute forward-gradient difference.
    reverse_gradient_error:
        Largest absolute difference between the saved gradient and the replayed
        reverse-adjoint step stream, or ``None`` when the program has no
        supported reverse adjoint.

    """

    record: WholeProgramReplayRecord
    value_error: float
    gradient_error: float
    reverse_gradient_error: float | None


def capture_whole_program_replay(result: WholeProgramADResult) -> WholeProgramReplayRecord:
    """Save the replay identity of a runtime whole-program AD result.

    Parameters
    ----------
    result:
        Result returned by :func:`whole_program_value_and_grad` whose captured
        state is unchanged since execution.

    Returns
    -------
    WholeProgramReplayRecord
        Program graph, captured state, effect history, transform plan and
        exact numbers of this execution.

    Raises
    ------
    ValueError
        If the result is not a runtime result with its live binding, its
        captured state or tape changed, or its captured state has no portable
        encoding.

    """
    if not isinstance(result, WholeProgramADResult):
        raise ValueError("replay capture requires a WholeProgramADResult")
    binding = result.captured_state
    if binding is None or result.program_ir is None or result.adjoint_result is None:
        raise ValueError("replay capture requires a runtime result with its live binding")
    _require_bound_program_tape(result)
    portable = _portable_program_state_digest(binding.objective)
    if portable is None:
        raise ValueError("captured program state has no portable encoding")
    # The runtime creates one parameter node per value before running the
    # objective, and the bound tape check above refuses any changed node.
    parameters = result.ir_nodes[: len(result.parameter_names)]
    return WholeProgramReplayRecord(
        interpreter=sys.implementation.cache_tag or sys.implementation.name,
        numpy_version=np.__version__,
        parameter_names=tuple(result.parameter_names),
        trainable=tuple(result.trainable),
        parameter_values=tuple(float(node.value) for node in parameters),
        trace=result.evaluations == 2,
        program_ir_sha256=hashlib.sha256(result.program_ir.serialization.encode()).hexdigest(),
        captured_state_sha256=portable.hex(),
        effects=tuple((effect.kind, effect.operation) for effect in result.program_ir.effects),
        adjoint_step_count=(
            len(result.adjoint_result.adjoint_steps)
            if result.adjoint_result.supported is True
            else None
        ),
        value=float(result.value),
        gradient=tuple(float(item) for item in result.gradient),
    )


def replay_whole_program(
    record: WholeProgramReplayRecord,
    objective: Callable[[Any], object],
    *,
    max_execution_gib: float | None = None,
    deadline_monotonic: float | None = None,
    cancelled: Event | None = None,
) -> WholeProgramReplayReport:
    """Re-execute a saved program and require its identity and derivatives to match.

    The runtime and the objective's captured state are checked before the
    objective runs, so a mutated closure, global or default is refused without
    executing it. The program is then differentiated again at the saved
    parameter values; the effect IR, effect history and transform plan must be
    identical, and value, forward gradient and reverse-adjoint replay must agree
    with the saved numbers within ``1e-10 + 1e-8*|saved|``.

    Parameters
    ----------
    record:
        Saved record from :func:`capture_whole_program_replay`.
    objective:
        Callable to replay; it must be the saved program with unchanged state.
    max_execution_gib:
        Optional declared numeric-buffer cap for the replay execution.
    deadline_monotonic:
        Absolute monotonic deadline for the replay execution.
    cancelled:
        Optional cancellation event for the replay execution.

    Returns
    -------
    WholeProgramReplayReport
        Replayed record and the observed numerical differences.

    Raises
    ------
    WholeProgramReplayMismatchError
        If the runtime, captured state, program graph, effects, transform plan
        or numbers differ; ``fields`` names every difference.
    ValueError
        If the record or objective is malformed, or the objective is refused by
        whole-program AD.

    """
    from .program_ad_adjoint import program_adjoint_replay_gradient

    if not isinstance(record, WholeProgramReplayRecord):
        raise ValueError("replay requires a WholeProgramReplayRecord")
    if not callable(objective):
        raise ValueError("whole-program objective must be callable")
    runtime = tuple(
        name
        for name, current in (
            ("interpreter", sys.implementation.cache_tag or sys.implementation.name),
            ("numpy_version", np.__version__),
        )
        if getattr(record, name) != current
    )
    if runtime:
        raise WholeProgramReplayMismatchError(runtime, "the saved runtime differs from this one")
    portable = _portable_program_state_digest(objective)
    if portable is None:
        raise ValueError("captured program state has no portable encoding")
    if portable.hex() != record.captured_state_sha256:
        raise WholeProgramReplayMismatchError(
            ("captured_state_sha256",),
            "captured program state differs from the saved program; it was not executed",
        )
    result = whole_program_value_and_grad(
        objective,
        np.array(record.parameter_values, dtype=np.float64),
        [
            Parameter(name, trainable)
            for name, trainable in zip(record.parameter_names, record.trainable, strict=True)
        ],
        trace=record.trace,
        max_execution_gib=max_execution_gib,
        deadline_monotonic=deadline_monotonic,
        cancelled=cancelled,
    )
    replayed = capture_whole_program_replay(result)
    reverse = (
        None
        if replayed.adjoint_step_count is None
        else program_adjoint_replay_gradient(
            result,
            max_execution_gib=max_execution_gib,
            deadline_monotonic=deadline_monotonic,
            cancelled=cancelled,
        )
    )
    saved_gradient = np.array(record.gradient, dtype=np.float64)
    value_error = abs(replayed.value - record.value)
    gradient_error = float(
        np.max(np.abs(np.array(replayed.gradient) - saved_gradient), initial=0.0)
    )
    reverse_error = (
        None if reverse is None else float(np.max(np.abs(reverse - saved_gradient), initial=0.0))
    )
    mismatched = [
        name
        for name in (
            "parameter_names",
            "trainable",
            "parameter_values",
            "trace",
            "program_ir_sha256",
            "captured_state_sha256",
            "effects",
            "adjoint_step_count",
        )
        if getattr(replayed, name) != getattr(record, name)
    ]
    if not _within_replay_budget(np.array([replayed.value]), np.array([record.value])):
        mismatched.append("value")
    if not _within_replay_budget(np.array(replayed.gradient), saved_gradient):
        mismatched.append("gradient")
    if reverse is not None and not _within_replay_budget(reverse, saved_gradient):
        mismatched.append("reverse_gradient")
    if mismatched:
        raise WholeProgramReplayMismatchError(
            tuple(mismatched), "the replayed execution differs from the saved record"
        )
    return WholeProgramReplayReport(replayed, value_error, gradient_error, reverse_error)


def _within_replay_budget(actual: NDArray[np.float64], saved: NDArray[np.float64]) -> bool:
    """Apply the frozen float64 agreement budget elementwise."""
    return bool(
        np.all(
            np.abs(actual - saved)
            <= _REPLAY_ABSOLUTE_BUDGET + _REPLAY_RELATIVE_BUDGET * np.abs(saved)
        )
    )


def _float_hex(value: float) -> str:
    """Encode a float64 as 16 lowercase big-endian IEEE-754 hex digits."""
    return struct.pack(">d", value).hex()


def _float_from_hex(value: object) -> float:
    """Decode a 16-digit IEEE-754 hex float64 and refuse other encodings."""
    if (
        type(value) is not str
        or len(value) != 16
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("replay record floats must be 16 lowercase hex digits")
    decoded = struct.unpack(">d", bytes.fromhex(value))[0]
    return float(decoded)


def _json_list(value: object) -> list[Any]:
    """Return a JSON array or refuse another type."""
    if type(value) is not list:
        raise ValueError("replay record array field must be a JSON array")
    return value


__all__ = [
    "WHOLE_PROGRAM_REPLAY_RANDOMNESS",
    "WHOLE_PROGRAM_REPLAY_SCHEMA",
    "WholeProgramReplayMismatchError",
    "WholeProgramReplayRecord",
    "WholeProgramReplayReport",
    "capture_whole_program_replay",
    "replay_whole_program",
    "whole_program_grad",
    "whole_program_value_and_grad",
]
