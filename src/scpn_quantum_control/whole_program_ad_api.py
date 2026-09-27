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

import sys
from collections.abc import Callable, Sequence
from threading import Event
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dense_budget import DenseAllocationError
from .differentiable_parameter_contracts import Parameter, _as_parameter_array
from .differentiable_transform_helpers import _normalise_parameters
from .execution_memory import ExecutionBuffer, ExecutionMemoryPlan, dataclass_storage_bytes
from .execution_reservations import ExecutionMemoryReservation, reserve_execution_memory
from .program_ad_adjoint_generation import _program_adjoint_result_from_nodes
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
        provenance.

    Raises
    ------
    ValueError
        If the objective is not callable, fails the source/bytecode frontend
        execution gate, uses unsupported Python semantics, or does not return a
        traceable scalar.

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
            _bounded_parameter_input(values, input_count, input_plan, reservation)
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
        parameter_meta = _bounded_parameter_metadata(parameter_values, parameters, context)
        frontend_report = compile_whole_program_frontend(objective)
        reservation.checkpoint()
        _require_whole_program_frontend_execution_ready(frontend_report)
        source = _objective_source(objective, context=context)
        reservation.checkpoint()
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
                raise ValueError("whole-program objective must return a whole-program AD scalar")
            raw = raw.item()
        if not isinstance(raw, TraceADScalar):
            raise ValueError("whole-program objective must return a whole-program AD scalar")
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
        )
        reservation.checkpoint()
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
        )
        reservation.checkpoint()
        return ad_result


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
    values: ArrayLike,
    count: int,
    plan: ExecutionMemoryPlan,
    reservation: ExecutionMemoryReservation,
) -> ArrayLike:
    """Copy into declared-size private storage before conversion can observe growth."""
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
    if not isinstance(values, (list, tuple, range)):
        raise ValueError("parameter input lost its inspectable conversion contract")
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


__all__ = ["whole_program_grad", "whole_program_value_and_grad"]
