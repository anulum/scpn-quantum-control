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

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_parameter_contracts import Parameter, _as_parameter_array
from .differentiable_transform_helpers import _normalise_parameters
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
) -> WholeProgramADResult:
    """Differentiate an executed Python/NumPy program by operator-intercepted AD.

    Parameters
    ----------
    objective:
        Callable that returns a whole-program AD scalar when executed over
        trace-aware parameter values.
    values:
        Initial parameter values.
    parameters:
        Optional metadata that marks trainable parameters and supplies names.
    trace:
        Whether to collect runtime trace events in addition to IR metadata.

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
    frontend_report = compile_whole_program_frontend(objective)
    _require_whole_program_frontend_execution_ready(frontend_report)
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    source = _objective_source(objective)
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
        bytecode_instructions=frontend_report.bytecode_instructions,
        source_ir_features=frontend_report.source_ir_features,
        semantics_report=semantics_report,
        program_ir=program_ir,
        adjoint_result=adjoint_result,
        frontend_report=frontend_report,
    )


def whole_program_grad(
    objective: Callable[[Any], object],
    values: ArrayLike,
    parameters: Sequence[Parameter] | None = None,
    *,
    trace: bool = True,
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

    Returns
    -------
    numpy.ndarray
        Exact whole-program AD gradient as ``float64`` values.
    """
    return whole_program_value_and_grad(
        objective, values, parameters=parameters, trace=trace
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
