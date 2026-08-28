# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum-classical co-design contracts
"""Immutable contracts for the simulator-first co-design co-design loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Final, Protocol

import numpy as np

CODESIGN_SCHEMA: Final[str] = "quantum_classical_codesign.v1"
CODESIGN_CLAIM_BOUNDARY: Final[str] = (
    "deterministic local simulator orchestration over existing phase and control "
    "surfaces; no live QPU, provider submission, operational plasma control, "
    "realtime hardware, stability guarantee, or quantum-advantage claim"
)


class CoDesignMode(str, Enum):
    """Supported directions through the bounded co-design loop."""

    CLASSICAL_TO_QUANTUM = "classical_to_quantum"
    QUANTUM_TO_CLASSICAL = "quantum_to_classical"
    HYBRID_REPLAY = "hybrid_replay"


class SafetyAction(str, Enum):
    """Fail-closed action selected by a loop safety policy."""

    ALLOW = "allow"
    CLAMP = "clamp"
    HOLD = "hold"
    ABORT = "abort"


class StaleGradientAction(str, Enum):
    """Action selected when a gradient is missing or too old."""

    HOLD = "hold"
    ABORT = "abort"


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Capabilities declared by one co-design evaluation backend."""

    backend_id: str
    supports_mid_circuit: bool
    supports_adjoint: bool
    max_latency_ms: float
    hardware: bool = False

    def __post_init__(self) -> None:
        """Validate the backend identity and latency ceiling."""
        if not self.backend_id.strip():
            raise ValueError("backend_id must be non-empty")
        _positive_finite("max_latency_ms", self.max_latency_ms)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready capability mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoopStepInput:
    """Immutable inputs for one co-design update.

    Parameters
    ----------
    step
        Zero-based loop step index.
    observed_at_ms
        Logical time at which the classical measurement was observed.
    apply_at_ms
        Logical time at which the proposed update would be applied.
    parameters
        Current phase-control parameters in radians.
    measurement
        Observed synchronisation order parameter in ``[0, 1]``.
    target_order_parameter
        Requested synchronisation target in ``[0, 1]``.
    mode
        Direction through the co-design loop.

    """

    step: int
    observed_at_ms: float
    apply_at_ms: float
    parameters: tuple[float, ...]
    measurement: float
    target_order_parameter: float
    mode: CoDesignMode = CoDesignMode.CLASSICAL_TO_QUANTUM

    def __post_init__(self) -> None:
        """Validate finite logical time, phase, and measurement inputs."""
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        _finite("observed_at_ms", self.observed_at_ms)
        _finite("apply_at_ms", self.apply_at_ms)
        if self.apply_at_ms < self.observed_at_ms:
            raise ValueError("apply_at_ms must not precede observed_at_ms")
        _finite_vector("parameters", self.parameters)
        _unit_interval("measurement", self.measurement)
        _unit_interval("target_order_parameter", self.target_order_parameter)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready input mapping."""
        return {
            "step": self.step,
            "observed_at_ms": self.observed_at_ms,
            "apply_at_ms": self.apply_at_ms,
            "parameters": list(self.parameters),
            "measurement": self.measurement,
            "target_order_parameter": self.target_order_parameter,
            "mode": self.mode.value,
        }


@dataclass(frozen=True, slots=True)
class StateEstimate:
    """Classical estimator result consumed by the controller."""

    order_parameter: float
    innovation: float
    sample_count: int

    def __post_init__(self) -> None:
        """Validate estimator output."""
        _unit_interval("order_parameter", self.order_parameter)
        _finite("innovation", self.innovation)
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready estimator mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GradientPlanRecord:
    """Bounded gradient-planner decision attached to an evaluation."""

    backend: str
    requested_method: str
    selected_method: str
    supported: bool
    evaluations: int
    requires_hardware_approval: bool
    reasons: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready planner mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QuantumEvaluation:
    """Objective, gradient, and backend telemetry for one simulator evaluation."""

    objective_value: float
    gradient: tuple[float, ...]
    order_parameter: float
    evaluated_at_ms: float
    backend_status: str
    capabilities: BackendCapabilities
    gradient_plan: GradientPlanRecord
    objective_terms: tuple[str, ...]
    open_system_backend: str | None = None
    gradient_source: str = "exact_phase_objective"
    schema: str = CODESIGN_SCHEMA
    claim_boundary: str = CODESIGN_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate finite evaluation telemetry."""
        _finite("objective_value", self.objective_value)
        _finite_vector("gradient", self.gradient)
        _unit_interval("order_parameter", self.order_parameter)
        _finite("evaluated_at_ms", self.evaluated_at_ms)
        if not self.backend_status.strip() or not self.objective_terms:
            raise ValueError("backend_status and objective_terms must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready evaluation mapping."""
        return {
            "schema": self.schema,
            "objective_value": self.objective_value,
            "gradient": list(self.gradient),
            "order_parameter": self.order_parameter,
            "evaluated_at_ms": self.evaluated_at_ms,
            "backend_status": self.backend_status,
            "capabilities": self.capabilities.to_dict(),
            "gradient_plan": self.gradient_plan.to_dict(),
            "objective_terms": list(self.objective_terms),
            "open_system_backend": self.open_system_backend,
            "gradient_source": self.gradient_source,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class ControllerProposal:
    """Unapplied controller proposal awaiting a safety decision."""

    parameters: tuple[float, ...]
    update: tuple[float, ...]
    gain_scale: float

    def __post_init__(self) -> None:
        """Validate proposal dimensions and values."""
        _finite_vector("parameters", self.parameters)
        _finite_vector("update", self.update)
        if len(self.parameters) != len(self.update):
            raise ValueError("controller proposal dimensions must match")
        _positive_finite("gain_scale", self.gain_scale)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready controller proposal."""
        return {
            "parameters": list(self.parameters),
            "update": list(self.update),
            "gain_scale": self.gain_scale,
        }


@dataclass(frozen=True, slots=True)
class LatencyDecision:
    """Decision for a present, missing, or stale gradient."""

    stale: bool
    missing: bool
    age_ms: float | None
    action: SafetyAction
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready latency decision."""
        return asdict(self) | {"action": self.action.value}


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    """Applied fail-closed decision for one proposed controller update."""

    action: SafetyAction
    reason: str
    applied_parameters: tuple[float, ...]
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the applied parameter vector and decision text."""
        _finite_vector("applied_parameters", self.applied_parameters)
        if not self.reason.strip():
            raise ValueError("safety reason must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready safety decision."""
        return {
            "action": self.action.value,
            "reason": self.reason,
            "applied_parameters": list(self.applied_parameters),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True, slots=True)
class ObserverInputs:
    """Optional immutable telemetry from completed observer products."""

    active_sensing_id: str | None = None
    identity_action: str | None = None
    identity_reason: str | None = None
    geometry_gradient_norm: float | None = None
    l16_action: str | None = None
    l16_reason: str | None = None
    adaptive_fim_id: str | None = None
    adaptive_fim_action: str | None = None
    adaptive_fim_lambda_out: float | None = None

    def __post_init__(self) -> None:
        """Validate optional observer values without promoting their claims."""
        if self.identity_action not in {None, "continue", "hold", "abort"}:
            raise ValueError("identity_action must be continue, hold, abort, or None")
        if self.l16_action not in {None, "continue", "adjust", "halt"}:
            raise ValueError("l16_action must be continue, adjust, halt, or None")
        if self.l16_action is None and self.l16_reason is not None:
            raise ValueError("l16_reason requires an l16_action")
        if self.l16_reason is not None and not self.l16_reason.strip():
            raise ValueError("l16_reason must be non-empty when provided")
        if self.adaptive_fim_action not in {None, "decrease", "hold"}:
            raise ValueError("adaptive_fim_action must be decrease, hold, or None")
        fim_values = (
            self.adaptive_fim_id,
            self.adaptive_fim_action,
            self.adaptive_fim_lambda_out,
        )
        if any(value is not None for value in fim_values) and any(
            value is None for value in fim_values
        ):
            raise ValueError("adaptive FIM observer fields must be supplied together")
        if self.adaptive_fim_id is not None and not self.adaptive_fim_id.strip():
            raise ValueError("adaptive_fim_id must be non-empty when provided")
        if self.adaptive_fim_lambda_out is not None:
            value = _finite("adaptive_fim_lambda_out", self.adaptive_fim_lambda_out)
            if value < 0.0:
                raise ValueError("adaptive_fim_lambda_out must be non-negative")
        if self.geometry_gradient_norm is not None:
            value = _finite("geometry_gradient_norm", self.geometry_gradient_norm)
            if value < 0.0:
                raise ValueError("geometry_gradient_norm must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready observer mapping."""
        payload = asdict(self)
        if self.l16_action is None:
            payload.pop("l16_action")
            payload.pop("l16_reason")
        if self.adaptive_fim_id is None:
            payload.pop("adaptive_fim_id")
            payload.pop("adaptive_fim_action")
            payload.pop("adaptive_fim_lambda_out")
        return payload


@dataclass(frozen=True, slots=True)
class LoopStepOutput:
    """Complete immutable result of one co-design loop step."""

    step: int
    estimate: StateEstimate
    evaluation: QuantumEvaluation | None
    proposal: ControllerProposal | None
    latency: LatencyDecision
    safety: SafetyDecision
    observers: ObserverInputs
    mode: CoDesignMode
    schema: str = CODESIGN_SCHEMA
    claim_boundary: str = CODESIGN_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready loop-step mapping."""
        return {
            "schema": self.schema,
            "step": self.step,
            "estimate": self.estimate.to_dict(),
            "evaluation": None if self.evaluation is None else self.evaluation.to_dict(),
            "proposal": None if self.proposal is None else self.proposal.to_dict(),
            "latency": self.latency.to_dict(),
            "safety": self.safety.to_dict(),
            "observers": self.observers.to_dict(),
            "mode": self.mode.value,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PlasmaObjectiveTemplate:
    """Non-operational plasma-relevance framing for a partner-owned plant."""

    template_id: str
    estimated_quantity: str
    controller_proxy: str
    objective: str
    non_operational: bool = True
    partner_validation_required: bool = True
    claim_boundary: str = CODESIGN_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Refuse templates that imply an operational control surface."""
        fields = (self.template_id, self.estimated_quantity, self.controller_proxy, self.objective)
        if not all(value.strip() for value in fields):
            raise ValueError("plasma template fields must be non-empty")
        if not self.non_operational or not self.partner_validation_required:
            raise ValueError("plasma templates must remain non-operational and partner-gated")


class StateEstimatorPort(Protocol):
    """Classical state-estimation boundary consumed by the loop."""

    def update(self, measurement: float) -> StateEstimate:
        """Update the estimator with one order-parameter measurement."""
        ...


class QuantumEvaluationPort(Protocol):
    """Policy-gated quantum-objective evaluation boundary."""

    def evaluate(self, step_input: LoopStepInput) -> QuantumEvaluation | None:
        """Evaluate one objective, or model a missing simulator update."""
        ...


class ControllerPort(Protocol):
    """Classical controller boundary consumed by the loop."""

    def propose(
        self,
        parameters: tuple[float, ...],
        gradient: tuple[float, ...],
        estimate: StateEstimate,
        target_order_parameter: float,
    ) -> ControllerProposal:
        """Propose a controller update without applying it."""
        ...


def plasma_objective_templates() -> tuple[PlasmaObjectiveTemplate, ...]:
    """Return bounded research templates for non-operational plasma framing."""
    return (
        PlasmaObjectiveTemplate(
            template_id="phase_coherence_recovery",
            estimated_quantity="normalised synchronisation order parameter",
            controller_proxy="bounded phase-control parameter update",
            objective="reduce synchronisation-target loss under explicit delay and safety gates",
        ),
        PlasmaObjectiveTemplate(
            template_id="observer_staleness_hold",
            estimated_quantity="age of partner-provided observer telemetry",
            controller_proxy="hold the previous bounded simulator command",
            objective="refuse stale observer-driven updates before a partner plant adapter runs",
        ),
    )


def _finite(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _positive_finite(name: str, value: float) -> float:
    scalar = _finite(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _unit_interval(name: str, value: float) -> float:
    scalar = _finite(name, value)
    if not 0.0 <= scalar <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return scalar


def _finite_vector(name: str, values: tuple[float, ...]) -> tuple[float, ...]:
    if not values or not all(np.isfinite(value) for value in values):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return values


__all__ = [
    "CODESIGN_CLAIM_BOUNDARY",
    "CODESIGN_SCHEMA",
    "BackendCapabilities",
    "CoDesignMode",
    "ControllerPort",
    "ControllerProposal",
    "GradientPlanRecord",
    "LatencyDecision",
    "LoopStepInput",
    "LoopStepOutput",
    "ObserverInputs",
    "PlasmaObjectiveTemplate",
    "QuantumEvaluation",
    "QuantumEvaluationPort",
    "SafetyAction",
    "SafetyDecision",
    "StaleGradientAction",
    "StateEstimate",
    "StateEstimatorPort",
    "plasma_objective_templates",
]
