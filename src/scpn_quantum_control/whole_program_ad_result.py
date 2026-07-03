# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- whole-program AD result records
"""Whole-program automatic-differentiation result records."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from .program_ad_adjoint import ProgramADAdjointResult
from .program_ad_effect_ir import ProgramADEffectIR
from .whole_program_frontend import (
    WholeProgramBytecodeInstruction,
    WholeProgramCompilerFrontendReport,
    WholeProgramSemanticsReport,
    WholeProgramSourceIRFeature,
)


@dataclass(frozen=True)
class WholeProgramTraceEvent:
    """One executed Python source line observed during whole-program AD tracing."""

    filename: str
    function_name: str
    line_number: int
    source: str

    def __post_init__(self) -> None:
        """Validate trace-event source metadata at construction time."""

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
        """Validate operator-intercepted node metadata at construction time."""

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
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "tangent", tangent)


@dataclass(frozen=True)
class WholeProgramADResult:
    """Value, gradient, frontend gate, adjoint replay contract, and AD status."""

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
    frontend_report: WholeProgramCompilerFrontendReport | None = None

    def __post_init__(self) -> None:
        """Validate whole-program AD result metadata at construction time."""

        value = _as_real_scalar("whole-program AD value", self.value)
        gradient = _as_real_numeric_array("whole-program AD gradient", self.gradient)
        if gradient.ndim != 1:
            raise ValueError("whole-program AD gradient must be one-dimensional")
        step = _as_real_scalar("whole-program AD step", self.step)
        if step < 0.0:
            raise ValueError("whole-program AD step must be non-negative")
        if self.evaluations < 1:
            raise ValueError("whole-program AD evaluations must be positive")
        if len(self.parameter_names) != gradient.size:
            raise ValueError("parameter_names length must match gradient length")
        if len(self.trainable) != gradient.size:
            raise ValueError("trainable mask length must match gradient length")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        _require_zero_frozen_entries("whole-program AD gradient", gradient, self.trainable)
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
        if self.frontend_report is not None and not isinstance(
            self.frontend_report, WholeProgramCompilerFrontendReport
        ):
            raise ValueError(
                "frontend_report must be a WholeProgramCompilerFrontendReport or None"
            )
        if (
            self.adjoint_result is not None
            and self.adjoint_result.gradient.shape != gradient.shape
        ):
            raise ValueError("adjoint_result gradient shape must match forward gradient shape")
        if self.adjoint_result is not None:
            _require_zero_frozen_entries(
                "whole-program AD adjoint gradient",
                self.adjoint_result.gradient,
                self.trainable,
            )
            _require_supported_adjoint_replay_matches_result(
                adjoint_result=self.adjoint_result,
                ir_nodes=self.ir_nodes,
                parameter_names=self.parameter_names,
                trainable=self.trainable,
                forward_gradient=gradient,
            )
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


def _require_zero_frozen_entries(
    name: str,
    values: NDArray[np.float64],
    trainable: tuple[bool, ...],
) -> None:
    frozen = np.logical_not(np.asarray(trainable, dtype=bool))
    if not np.any(frozen):
        return
    selected = values[np.flatnonzero(frozen)]
    if np.any(selected != 0.0):
        raise ValueError(f"{name} must be zero for non-trainable parameters")


def _as_real_numeric_array(name: str, values: object) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _as_real_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _require_supported_adjoint_replay_matches_result(
    *,
    adjoint_result: ProgramADAdjointResult,
    ir_nodes: tuple[WholeProgramIRNode, ...],
    parameter_names: tuple[str, ...],
    trainable: tuple[bool, ...],
    forward_gradient: NDArray[np.float64],
) -> None:
    """Validate supported reverse-adjoint replay against the attached result."""

    if not adjoint_result.supported:
        return

    from .program_ad_adjoint import (
        _PROGRAM_ADJOINT_REPLAY_ATOL,
        _program_adjoint_execute_steps,
    )

    try:
        replay_gradient = _program_adjoint_execute_steps(
            adjoint=adjoint_result,
            ir_nodes=ir_nodes,
            parameter_names=parameter_names,
            trainable=trainable,
        )
    except ValueError as exc:
        raise ValueError(
            f"whole-program AD supported adjoint_result executable replay failed: {exc}"
        ) from exc
    if not np.allclose(
        replay_gradient,
        adjoint_result.gradient,
        rtol=0.0,
        atol=_PROGRAM_ADJOINT_REPLAY_ATOL,
    ):
        raise ValueError(
            "whole-program AD supported adjoint_result executable replay diverged "
            "from attached gradient"
        )
    if not np.allclose(
        replay_gradient,
        forward_gradient,
        rtol=0.0,
        atol=_PROGRAM_ADJOINT_REPLAY_ATOL,
    ):
        raise ValueError(
            "whole-program AD supported adjoint_result executable replay diverged "
            "from forward gradient"
        )


__all__ = ["WholeProgramADResult", "WholeProgramIRNode", "WholeProgramTraceEvent"]
