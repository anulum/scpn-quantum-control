# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD adjoint module
# scpn-quantum-control -- Program AD reverse-adjoint result records and accessors
"""Program AD reverse-adjoint generation result records and accessors."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .differentiable_parameter_contracts import Parameter
    from .whole_program_ad_result import WholeProgramIRNode


_PROGRAM_ADJOINT_REPLAY_ATOL = 1.0e-12


@dataclass(frozen=True)
class ProgramADAdjointStep:
    """One generated reverse-adjoint step over stabilized Program AD IR.

    The step binds a primal SSA value and stabilized effect row to the local
    pullback inputs, finite incoming cotangent, local pullback coefficients,
    emitted contribution cotangents, effect ordering metadata, and any
    unambiguous runtime control/phi row used by reverse-mode adjoint
    generation. Non-executed phi inputs are recorded as blocked adjoints rather
    than replay contributions. It is an auditable generation plan over
    ``program_ad_effect_ir.v1`` metadata; it does not add non-executed branch
    adjoints or executable compiler lowering.
    """

    index: int
    primal_value: str
    primal_effect: int | None
    operation: str
    input_values: tuple[str, ...]
    contribution_inputs: tuple[str, ...]
    supported: bool
    effect_kind: str | None = None
    effect_version: int | None = None
    effect_ordering: int | None = None
    control_region: int | None = None
    control_region_kind: str | None = None
    control_region_entered: bool | None = None
    phi_node: int | None = None
    phi_selected: str | None = None
    non_executed_phi_inputs: tuple[str, ...] = ()
    incoming_cotangent: float = 0.0
    contribution_scales: tuple[float, ...] = ()
    contribution_cotangents: tuple[float, ...] = ()
    unsupported_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate reverse-adjoint step metadata at construction time."""
        if self.index < 0:
            raise ValueError("program AD adjoint step index must be non-negative")
        if not self.primal_value:
            raise ValueError("program AD adjoint step primal_value must be non-empty")
        if self.primal_effect is not None and self.primal_effect < 0:
            raise ValueError("program AD adjoint step primal_effect must be non-negative or None")
        if self.primal_effect is None and (
            self.effect_kind is not None
            or self.effect_version is not None
            or self.effect_ordering is not None
        ):
            raise ValueError("program AD adjoint step effect metadata requires a primal_effect")
        if self.primal_effect is not None:
            if not isinstance(self.effect_kind, str) or not self.effect_kind:
                raise ValueError(
                    "program AD adjoint step effect_kind must be non-empty when "
                    "primal_effect is present"
                )
            if (
                self.effect_version is None
                or isinstance(self.effect_version, bool)
                or not isinstance(self.effect_version, int)
                or self.effect_version < 0
            ):
                raise ValueError(
                    "program AD adjoint step effect_version must be non-negative "
                    "when primal_effect is present"
                )
            if (
                self.effect_ordering is None
                or isinstance(self.effect_ordering, bool)
                or not isinstance(self.effect_ordering, int)
                or self.effect_ordering < 0
            ):
                raise ValueError(
                    "program AD adjoint step effect_ordering must be non-negative "
                    "when primal_effect is present"
                )
        if self.control_region is None:
            if self.control_region_kind is not None or self.control_region_entered is not None:
                raise ValueError(
                    "program AD adjoint step control metadata requires a control_region"
                )
        else:
            if (
                isinstance(self.control_region, bool)
                or not isinstance(self.control_region, int)
                or self.control_region < 0
            ):
                raise ValueError("program AD adjoint step control_region must be non-negative")
            if not isinstance(self.control_region_kind, str) or not self.control_region_kind:
                raise ValueError(
                    "program AD adjoint step control_region_kind must be non-empty "
                    "when control_region is present"
                )
            if not isinstance(self.control_region_entered, bool):
                raise ValueError(
                    "program AD adjoint step control_region_entered must be a boolean "
                    "when control_region is present"
                )
        if self.phi_node is None:
            if self.phi_selected is not None:
                raise ValueError("program AD adjoint step phi metadata requires a phi_node")
        else:
            if self.control_region is None:
                raise ValueError("program AD adjoint step phi metadata requires control metadata")
            if (
                isinstance(self.phi_node, bool)
                or not isinstance(self.phi_node, int)
                or self.phi_node < 0
            ):
                raise ValueError("program AD adjoint step phi_node must be non-negative")
            if not isinstance(self.phi_selected, str) or not self.phi_selected:
                raise ValueError(
                    "program AD adjoint step phi_selected must be non-empty when phi_node "
                    "is present"
                )
        if any(not isinstance(value, str) or not value for value in self.non_executed_phi_inputs):
            raise ValueError(
                "program AD adjoint step non_executed_phi_inputs entries must be non-empty strings"
            )
        if len(set(self.non_executed_phi_inputs)) != len(self.non_executed_phi_inputs):
            raise ValueError("program AD adjoint step non_executed_phi_inputs must be unique")
        if self.non_executed_phi_inputs:
            if self.phi_node is None or self.phi_selected is None:
                raise ValueError(
                    "program AD adjoint step non_executed_phi_inputs requires phi metadata"
                )
            if self.phi_selected in self.non_executed_phi_inputs:
                raise ValueError(
                    "program AD adjoint step non_executed_phi_inputs cannot include "
                    "the selected phi input"
                )
        if not self.operation:
            raise ValueError("program AD adjoint step operation must be non-empty")
        if any(not value for value in self.input_values):
            raise ValueError("program AD adjoint step input_values entries must be non-empty")
        if any(not value for value in self.contribution_inputs):
            raise ValueError(
                "program AD adjoint step contribution_inputs entries must be non-empty"
            )
        if tuple(sorted(set(self.contribution_inputs))) != self.contribution_inputs:
            raise ValueError(
                "program AD adjoint step contribution_inputs must be sorted and unique"
            )
        raw_incoming_cotangent = cast(object, self.incoming_cotangent)
        if isinstance(raw_incoming_cotangent, bool) or not isinstance(
            raw_incoming_cotangent, Real
        ):
            raise ValueError("program AD adjoint step incoming_cotangent must be a finite float")
        incoming_cotangent = float(raw_incoming_cotangent)
        if not np.isfinite(incoming_cotangent):
            raise ValueError("program AD adjoint step incoming_cotangent must be finite")
        object.__setattr__(self, "incoming_cotangent", incoming_cotangent)
        if len(self.contribution_scales) != len(self.contribution_inputs):
            raise ValueError(
                "program AD adjoint step contribution_scales length must match contribution_inputs"
            )
        normalized_scales: list[float] = []
        raw_scales = cast(tuple[object, ...], self.contribution_scales)
        for scale in raw_scales:
            if isinstance(scale, bool) or not isinstance(scale, Real):
                raise ValueError(
                    "program AD adjoint step contribution_scales entries must be finite floats"
                )
            scale_float = float(scale)
            if not np.isfinite(scale_float):
                raise ValueError(
                    "program AD adjoint step contribution_scales entries must be finite"
                )
            normalized_scales.append(scale_float)
        object.__setattr__(self, "contribution_scales", tuple(normalized_scales))
        if len(self.contribution_cotangents) != len(self.contribution_inputs):
            raise ValueError(
                "program AD adjoint step contribution_cotangents length must match "
                "contribution_inputs"
            )
        normalized_cotangents: list[float] = []
        raw_cotangents = cast(tuple[object, ...], self.contribution_cotangents)
        for cotangent in raw_cotangents:
            if isinstance(cotangent, bool) or not isinstance(cotangent, Real):
                raise ValueError(
                    "program AD adjoint step contribution_cotangents entries must be finite floats"
                )
            cotangent_float = float(cotangent)
            if not np.isfinite(cotangent_float):
                raise ValueError(
                    "program AD adjoint step contribution_cotangents entries must be finite"
                )
            normalized_cotangents.append(cotangent_float)
        object.__setattr__(self, "contribution_cotangents", tuple(normalized_cotangents))
        expected_cotangents = tuple(incoming_cotangent * scale for scale in normalized_scales)
        if not np.allclose(
            np.asarray(normalized_cotangents, dtype=np.float64),
            np.asarray(expected_cotangents, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "program AD adjoint step contribution_cotangents must match "
                "incoming_cotangent times contribution_scales"
            )
        if not isinstance(self.supported, bool):
            raise ValueError("program AD adjoint step supported must be a boolean")
        if self.supported and self.unsupported_reason is not None:
            raise ValueError("supported program AD adjoint step cannot carry unsupported_reason")
        if self.unsupported_reason is not None and not self.unsupported_reason:
            raise ValueError(
                "program AD adjoint step unsupported_reason must be non-empty or None"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready reverse-adjoint generation step."""
        return {
            "index": self.index,
            "primal_value": self.primal_value,
            "primal_effect": self.primal_effect,
            "effect_kind": self.effect_kind,
            "effect_version": self.effect_version,
            "effect_ordering": self.effect_ordering,
            "control_region": self.control_region,
            "control_region_kind": self.control_region_kind,
            "control_region_entered": self.control_region_entered,
            "phi_node": self.phi_node,
            "phi_selected": self.phi_selected,
            "non_executed_phi_inputs": list(self.non_executed_phi_inputs),
            "operation": self.operation,
            "input_values": list(self.input_values),
            "contribution_inputs": list(self.contribution_inputs),
            "incoming_cotangent": self.incoming_cotangent,
            "contribution_scales": list(self.contribution_scales),
            "contribution_cotangents": list(self.contribution_cotangents),
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
        }


@dataclass(frozen=True)
class ProgramADAdjointResult:
    """Reverse-mode adjoint generation result for a captured Program AD graph."""

    gradient: NDArray[np.float64]
    supported: bool
    unsupported_ops: tuple[str, ...]
    method: str
    claim_boundary: str
    replay_node_count: int = 0
    replay_effect_count: int = 0
    replay_control_region_count: int = 0
    replay_phi_node_count: int = 0
    executed_branch_replay_count: int = 0
    blocked_non_executed_phi_input_count: int = 0
    replay_ir_format: str = "program_ad_effect_ir.v1"
    adjoint_steps: tuple[ProgramADAdjointStep, ...] = ()

    def __post_init__(self) -> None:
        """Validate reverse-adjoint result metadata at construction time."""
        gradient = _as_real_numeric_array("program AD adjoint gradient", self.gradient)
        if gradient.ndim != 1:
            raise ValueError("program AD adjoint gradient must be one-dimensional")
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
        for name in (
            "replay_node_count",
            "replay_effect_count",
            "replay_control_region_count",
            "replay_phi_node_count",
            "executed_branch_replay_count",
            "blocked_non_executed_phi_input_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"program AD adjoint {name} must be a non-negative integer")
        if not isinstance(self.replay_ir_format, str) or not self.replay_ir_format:
            raise ValueError("program AD adjoint replay_ir_format must be a non-empty string")
        if any(not isinstance(step, ProgramADAdjointStep) for step in self.adjoint_steps):
            raise ValueError(
                "program AD adjoint adjoint_steps must contain ProgramADAdjointStep entries"
            )
        step_indices = tuple(step.index for step in self.adjoint_steps)
        if tuple(range(len(self.adjoint_steps))) != step_indices:
            raise ValueError("program AD adjoint adjoint_steps must be densely indexed")
        unsupported_step_ops = {
            step.operation for step in self.adjoint_steps if not step.supported
        }
        actual_executed_branch_replays = sum(
            1
            for step in self.adjoint_steps
            if step.operation.startswith("branch:")
            and step.control_region_kind == "runtime_branch"
            and step.phi_node is not None
            and step.phi_selected is not None
        )
        if self.executed_branch_replay_count != actual_executed_branch_replays:
            raise ValueError(
                "program AD adjoint executed_branch_replay_count must match "
                "generated branch replay steps"
            )
        actual_blocked_phi_inputs = sum(
            len(step.non_executed_phi_inputs) for step in self.adjoint_steps
        )
        if self.blocked_non_executed_phi_input_count != actual_blocked_phi_inputs:
            raise ValueError(
                "program AD adjoint blocked_non_executed_phi_input_count must match "
                "generated non-executed phi blockers"
            )
        if self.supported and unsupported_step_ops:
            raise ValueError("supported program AD adjoint cannot carry unsupported steps")
        if unsupported_step_ops and not unsupported_step_ops.issubset(set(self.unsupported_ops)):
            raise ValueError(
                "program AD adjoint unsupported steps must be reflected in unsupported_ops"
            )
        object.__setattr__(self, "gradient", gradient)

    @property
    def adjoint_step_count(self) -> int:
        """Return the number of generated reverse-adjoint steps."""
        return len(self.adjoint_steps)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready reverse-adjoint result."""
        return {
            "gradient": self.gradient.tolist(),
            "supported": self.supported,
            "unsupported_ops": list(self.unsupported_ops),
            "method": self.method,
            "claim_boundary": self.claim_boundary,
            "replay_node_count": self.replay_node_count,
            "replay_effect_count": self.replay_effect_count,
            "replay_control_region_count": self.replay_control_region_count,
            "replay_phi_node_count": self.replay_phi_node_count,
            "executed_branch_replay_count": self.executed_branch_replay_count,
            "blocked_non_executed_phi_input_count": self.blocked_non_executed_phi_input_count,
            "replay_ir_format": self.replay_ir_format,
            "adjoint_step_count": self.adjoint_step_count,
            "adjoint_steps": [step.to_dict() for step in self.adjoint_steps],
        }


def _as_real_numeric_array(label: str, value: object) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{label} must contain finite values")
    return array


def program_adjoint_result(result: object) -> ProgramADAdjointResult:
    """Return the reverse-mode adjoint generation result attached to Program AD.

    Parameters
    ----------
    result:
        Whole-program AD result that should carry reverse-adjoint replay
        metadata.

    Returns
    -------
    ProgramADAdjointResult
        Attached reverse-adjoint replay result.

    Raises
    ------
    ValueError
        If ``result`` is not a whole-program AD result or has no attached
        adjoint metadata.
    """
    from .whole_program_ad_result import WholeProgramADResult

    if not isinstance(result, WholeProgramADResult):
        raise ValueError("program adjoint input must be a WholeProgramADResult")
    if result.adjoint_result is None:
        raise ValueError("program AD result does not contain adjoint generation metadata")
    return result.adjoint_result


def program_adjoint_gradient(result: object) -> NDArray[np.float64]:
    """Return a supported reverse-mode adjoint gradient or fail closed.

    Parameters
    ----------
    result:
        Whole-program AD result whose attached reverse-adjoint metadata should
        be supported.

    Returns
    -------
    numpy.ndarray
        Copy of the attached reverse-adjoint gradient.

    Raises
    ------
    ValueError
        If no adjoint metadata is attached or the captured IR has unsupported
        operations.
    """
    adjoint = program_adjoint_result(result)
    if not adjoint.supported:
        unsupported = ", ".join(adjoint.unsupported_ops)
        raise ValueError(f"program AD adjoint generation unsupported for ops: {unsupported}")
    gradient: NDArray[np.float64] = adjoint.gradient.copy()
    return gradient


def program_adjoint_replay_gradient(result: object) -> NDArray[np.float64]:
    """Execute generated Program AD adjoint steps and return the replayed gradient.

    Parameters
    ----------
    result:
        Whole-program AD result carrying supported ``ProgramADAdjointStep``
        rows bound to ``program_ad_effect_ir.v1`` and the captured stabilized
        IR node sequence.

    Returns
    -------
    numpy.ndarray
        Gradient reconstructed by executing the generated reverse-adjoint step
        stream. Non-trainable parameters are preserved as zero entries.

    Raises
    ------
    ValueError
        If the input is not a whole-program result, the adjoint result is
        unsupported, the generated step stream is missing or not bound to the
        captured stabilized IR, or executable replay diverges from the attached
        adjoint gradient.
    """
    from .whole_program_ad_result import WholeProgramADResult

    if not isinstance(result, WholeProgramADResult):
        raise ValueError("program adjoint replay input must be a WholeProgramADResult")
    adjoint = program_adjoint_result(result)
    if not adjoint.supported:
        unsupported = ", ".join(adjoint.unsupported_ops)
        raise ValueError(f"program AD adjoint generation unsupported for ops: {unsupported}")

    replay_gradient = _program_adjoint_execute_steps(
        adjoint=adjoint,
        ir_nodes=result.ir_nodes,
        parameter_names=result.parameter_names,
        trainable=result.trainable,
    )
    if not np.allclose(
        replay_gradient,
        adjoint.gradient,
        rtol=0.0,
        atol=_PROGRAM_ADJOINT_REPLAY_ATOL,
    ):
        raise ValueError("program AD executable adjoint replay diverged from attached gradient")
    return replay_gradient


def program_adjoint_grad(
    objective: Callable[[Any], object],
    values: ArrayLike,
    parameters: Sequence[Parameter] | None = None,
    *,
    trace: bool = True,
) -> NDArray[np.float64]:
    """Return the reverse-mode program AD gradient for supported captured IR.

    Parameters
    ----------
    objective:
        Scalar objective that accepts Program AD trace values.
    values:
        Initial numeric parameter values.
    parameters:
        Optional named parameter metadata. Frozen parameters keep zero
        cotangents in the generated adjoint gradient.
    trace:
        Whether to keep runtime trace-event evidence in the captured
        whole-program result.

    Returns
    -------
    numpy.ndarray
        Reverse-adjoint generation gradient for the captured Program AD IR.

    Raises
    ------
    ValueError
        If the objective does not produce a scalar Program AD result or if the
        captured IR contains unsupported adjoint-generation operations.
    """
    from .differentiable import whole_program_value_and_grad

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
    """Return the objective value and reverse-mode Program AD gradient.

    Parameters
    ----------
    objective:
        Scalar objective that accepts Program AD trace values.
    values:
        Initial numeric parameter values.
    parameters:
        Optional named parameter metadata. Frozen parameters keep zero
        cotangents in the generated adjoint gradient.
    trace:
        Whether to keep runtime trace-event evidence in the captured
        whole-program result.

    Returns
    -------
    tuple[float, numpy.ndarray]
        Objective value and reverse-adjoint generation gradient.

    Raises
    ------
    ValueError
        If the objective does not produce a scalar Program AD result or if the
        captured IR contains unsupported adjoint-generation operations.
    """
    from .differentiable import whole_program_value_and_grad

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=parameters,
        trace=trace,
    )
    return result.value, program_adjoint_gradient(result)


def _program_adjoint_input_value(
    name: str,
    node_by_name: Mapping[str, WholeProgramIRNode],
) -> float:
    """Resolve a Program AD adjoint input token to a primal scalar."""
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
    """Return whether ``name`` is a Program AD SSA value token."""
    return isinstance(name, str) and name.startswith("%") and name[1:].isdigit()


def _program_adjoint_execute_steps(
    *,
    adjoint: ProgramADAdjointResult,
    ir_nodes: tuple[WholeProgramIRNode, ...],
    parameter_names: tuple[str, ...],
    trainable: tuple[bool, ...],
) -> NDArray[np.float64]:
    """Execute a generated reverse-adjoint step stream over captured IR metadata."""
    if adjoint.replay_ir_format != "program_ad_effect_ir.v1":
        raise ValueError("program AD executable adjoint replay requires program_ad_effect_ir.v1")
    if not adjoint.adjoint_steps:
        raise ValueError("program AD executable adjoint replay requires generated adjoint steps")
    if not ir_nodes:
        raise ValueError("program AD executable adjoint replay requires captured IR nodes")
    if adjoint.replay_node_count != len(ir_nodes):
        raise ValueError(
            "program AD executable adjoint replay node count does not match captured IR"
        )
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("program AD executable adjoint replay requires unique parameters")
    expected_primal_values = tuple(f"%{node.index}" for node in reversed(ir_nodes))
    actual_primal_values = tuple(step.primal_value for step in adjoint.adjoint_steps)
    if actual_primal_values != expected_primal_values:
        raise ValueError(
            "program AD executable adjoint replay step stream is not bound to captured IR"
        )
    expected_operations = tuple(node.op for node in reversed(ir_nodes))
    actual_operations = tuple(step.operation for step in adjoint.adjoint_steps)
    if actual_operations != expected_operations:
        raise ValueError(
            "program AD executable adjoint replay operations do not match captured IR"
        )

    root_step = adjoint.adjoint_steps[0]
    cotangents: dict[str, float] = {root_step.primal_value: 1.0}
    parameter_cotangents: dict[str, float] = {}
    parameter_name_set = set(parameter_names)
    for step in adjoint.adjoint_steps:
        if not step.supported:
            raise ValueError(
                "program AD executable adjoint replay cannot execute unsupported step "
                f"{step.operation}"
            )
        incoming = float(cotangents.get(step.primal_value, 0.0))
        _program_adjoint_replay_require_close(
            "incoming cotangent",
            incoming,
            step.incoming_cotangent,
        )
        if step.operation == "parameter":
            if len(step.input_values) != 1:
                raise ValueError(
                    "program AD executable adjoint replay parameter step must name one parameter"
                )
            parameter_name = step.input_values[0]
            if parameter_name not in parameter_name_set:
                raise ValueError(
                    "program AD executable adjoint replay parameter step is not in "
                    "result parameter names"
                )
            parameter_cotangents[parameter_name] = (
                parameter_cotangents.get(parameter_name, 0.0) + incoming
            )
        for input_name, scale, recorded_cotangent in zip(
            step.contribution_inputs,
            step.contribution_scales,
            step.contribution_cotangents,
            strict=True,
        ):
            contribution = incoming * scale
            _program_adjoint_replay_require_close(
                "contribution cotangent",
                contribution,
                recorded_cotangent,
            )
            if _program_adjoint_is_ir_value(input_name):
                cotangents[input_name] = cotangents.get(input_name, 0.0) + contribution

    gradient = np.zeros(len(parameter_names), dtype=np.float64)
    for index, (parameter_name, trainable_flag) in enumerate(
        zip(parameter_names, trainable, strict=True)
    ):
        if trainable_flag:
            gradient[index] = parameter_cotangents.get(parameter_name, 0.0)
    return gradient


def _program_adjoint_replay_require_close(
    label: str,
    actual: float,
    expected: float,
) -> None:
    """Fail closed when replayed scalar cotangent flow diverges from step metadata."""
    if not np.isclose(actual, expected, rtol=0.0, atol=_PROGRAM_ADJOINT_REPLAY_ATOL):
        raise ValueError(
            "program AD executable adjoint replay "
            f"{label} mismatch: expected {expected}, replayed {actual}"
        )


__all__ = [
    "ProgramADAdjointResult",
    "ProgramADAdjointStep",
    "program_adjoint_grad",
    "program_adjoint_gradient",
    "program_adjoint_replay_gradient",
    "program_adjoint_result",
    "program_adjoint_value_and_grad",
]
