# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
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
    _accepted_python_semantics,
    _objective_bytecode,
    _objective_source,
    _source_ir_features,
    _source_mentions_numpy,
    _unsupported_python_semantics,
    _whole_program_semantics_report,
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
        nodes, semantics report, and scalar adjoint replay provenance.

    Raises
    ------
    ValueError
        If the objective is not callable, uses unsupported Python semantics, or
        does not return a traceable scalar.
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


__all__ = ["whole_program_grad", "whole_program_value_and_grad"]
