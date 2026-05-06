# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Feedback submission readiness
"""Provider-neutral S1 feedback submission-readiness packaging.

This module prepares auditable, no-submission execution packages for the S1
hybrid feedback track. It records what a platform must support, what the
candidate dynamic circuit contains, and how much QPU time the experiment is
expected to reserve. It does not read credentials or submit jobs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from qiskit import QuantumCircuit

from ..control.realtime_feedback import RealtimeSyncFeedbackController
from .job_dossier import HardwareJobDossier, build_s1_feedback_job_dossier

FeedbackPlatformKind = Literal[
    "ibm_dynamic_circuits",
    "gate_based_dynamic_circuits",
    "neutral_atom_analog",
    "continuous_variable",
    "simulator",
]
ReadinessStatus = Literal["ready", "blocked", "manual_review"]


@dataclass(frozen=True)
class FeedbackPlatformCapability:
    """Capabilities relevant to S1 feedback-loop execution."""

    name: str
    kind: FeedbackPlatformKind
    max_qubits: int
    supports_mid_circuit_measurement: bool
    supports_conditional_reset: bool
    supports_conditional_rotation: bool
    supports_cross_shot_batches: bool
    supports_native_xy: bool = False
    supports_native_global_feedback: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("platform name must be non-empty")
        if self.max_qubits < 1:
            raise ValueError("max_qubits must be positive")


@dataclass(frozen=True)
class FeedbackBudgetEstimate:
    """Conservative QPU reservation estimate for a feedback package."""

    circuits: int
    shots_per_circuit: int
    repetitions: int
    estimated_execution_seconds: float
    queue_seconds: float = 0.0
    calibration_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.circuits < 1:
            raise ValueError("circuits must be positive")
        if self.shots_per_circuit < 1:
            raise ValueError("shots_per_circuit must be positive")
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        _require_non_negative(self.estimated_execution_seconds, "estimated_execution_seconds")
        _require_non_negative(self.queue_seconds, "queue_seconds")
        _require_non_negative(self.calibration_seconds, "calibration_seconds")

    @property
    def total_reserved_seconds(self) -> float:
        """Return total requested reservation time in seconds."""
        return self.estimated_execution_seconds + self.queue_seconds + self.calibration_seconds


@dataclass(frozen=True)
class FeedbackCircuitSummary:
    """Provider-independent dynamic-circuit summary."""

    n_qubits: int
    n_clbits: int
    depth: int
    operation_counts: Mapping[str, int]
    has_mid_circuit_measurement: bool
    has_conditional_control: bool
    n_rounds: int


@dataclass(frozen=True)
class PlatformReadiness:
    """Readiness decision for one target platform."""

    platform: FeedbackPlatformCapability
    status: ReadinessStatus
    reasons: tuple[str, ...]
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeedbackSubmissionPackage:
    """Complete no-submission S1 feedback package."""

    experiment_id: str
    circuit: FeedbackCircuitSummary
    budget: FeedbackBudgetEstimate
    platform_readiness: tuple[PlatformReadiness, ...]
    claim_boundary: str
    dossier: HardwareJobDossier

    @property
    def ready_platforms(self) -> tuple[str, ...]:
        """Return platforms that satisfy the declared execution requirements."""
        return tuple(
            decision.platform.name
            for decision in self.platform_readiness
            if decision.status == "ready"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the package for JSON manifests or preregistration docs."""
        return {
            "experiment_id": self.experiment_id,
            "circuit": {
                "n_qubits": self.circuit.n_qubits,
                "n_clbits": self.circuit.n_clbits,
                "depth": self.circuit.depth,
                "operation_counts": dict(self.circuit.operation_counts),
                "has_mid_circuit_measurement": self.circuit.has_mid_circuit_measurement,
                "has_conditional_control": self.circuit.has_conditional_control,
                "n_rounds": self.circuit.n_rounds,
            },
            "budget": {
                "circuits": self.budget.circuits,
                "shots_per_circuit": self.budget.shots_per_circuit,
                "repetitions": self.budget.repetitions,
                "estimated_execution_seconds": self.budget.estimated_execution_seconds,
                "queue_seconds": self.budget.queue_seconds,
                "calibration_seconds": self.budget.calibration_seconds,
                "total_reserved_seconds": self.budget.total_reserved_seconds,
            },
            "platform_readiness": [
                {
                    "platform": decision.platform.name,
                    "kind": decision.platform.kind,
                    "status": decision.status,
                    "reasons": list(decision.reasons),
                    "payload": dict(decision.payload),
                }
                for decision in self.platform_readiness
            ],
            "ready_platforms": list(self.ready_platforms),
            "claim_boundary": self.claim_boundary,
            "dossier": self.dossier.to_dict(),
        }


def default_s1_platforms() -> tuple[FeedbackPlatformCapability, ...]:
    """Return conservative platform capability presets for S1 planning."""
    return (
        FeedbackPlatformCapability(
            name="IBM Heron dynamic-circuit backend",
            kind="ibm_dynamic_circuits",
            max_qubits=100,
            supports_mid_circuit_measurement=True,
            supports_conditional_reset=True,
            supports_conditional_rotation=True,
            supports_cross_shot_batches=True,
            notes="Gate-based dynamic-circuit candidate; requires live backend capability check.",
        ),
        FeedbackPlatformCapability(
            name="Generic dynamic-circuit gate backend",
            kind="gate_based_dynamic_circuits",
            max_qubits=32,
            supports_mid_circuit_measurement=True,
            supports_conditional_reset=True,
            supports_conditional_rotation=True,
            supports_cross_shot_batches=True,
            notes="Provider-neutral OpenQASM 3 style execution target.",
        ),
        FeedbackPlatformCapability(
            name="Neutral-atom analogue XY target",
            kind="neutral_atom_analog",
            max_qubits=64,
            supports_mid_circuit_measurement=False,
            supports_conditional_reset=False,
            supports_conditional_rotation=False,
            supports_cross_shot_batches=True,
            supports_native_xy=True,
            notes="Suitable for open-loop/native XY follow-up, not this dynamic-circuit payload.",
        ),
        FeedbackPlatformCapability(
            name="Continuous-variable analogue target",
            kind="continuous_variable",
            max_qubits=16,
            supports_mid_circuit_measurement=False,
            supports_conditional_reset=False,
            supports_conditional_rotation=False,
            supports_cross_shot_batches=True,
            supports_native_xy=True,
            notes="Requires a separate analogue feedback formulation.",
        ),
        FeedbackPlatformCapability(
            name="Local statevector simulator",
            kind="simulator",
            max_qubits=12,
            supports_mid_circuit_measurement=True,
            supports_conditional_reset=True,
            supports_conditional_rotation=True,
            supports_cross_shot_batches=True,
            notes="No-QPU reference target for package validation and latency benchmarking.",
        ),
    )


def build_s1_feedback_submission_package(
    controller: RealtimeSyncFeedbackController,
    *,
    experiment_id: str = "s1_dynamic_feedback_readiness",
    n_rounds: int = 3,
    shots_per_circuit: int = 1024,
    repetitions: int = 12,
    estimated_seconds_per_circuit: float = 1.0,
    platforms: Sequence[FeedbackPlatformCapability] | None = None,
) -> FeedbackSubmissionPackage:
    """Build a provider-neutral no-submission S1 readiness package."""
    if not experiment_id:
        raise ValueError("experiment_id must be non-empty")
    if n_rounds < 1:
        raise ValueError("n_rounds must be positive")
    if shots_per_circuit < 1:
        raise ValueError("shots_per_circuit must be positive")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    _require_non_negative(estimated_seconds_per_circuit, "estimated_seconds_per_circuit")
    circuit = controller.build_monitored_circuit(n_rounds)
    summary = summarise_feedback_circuit(circuit, n_rounds=n_rounds)
    budget = FeedbackBudgetEstimate(
        circuits=1,
        shots_per_circuit=shots_per_circuit,
        repetitions=repetitions,
        estimated_execution_seconds=estimated_seconds_per_circuit * repetitions,
    )
    targets = tuple(platforms or default_s1_platforms())
    decisions = tuple(assess_platform_readiness(platform, summary, budget) for platform in targets)
    claim_boundary = (
        "Readiness package only: no credentials are read, no job is submitted, "
        "and platform readiness does not replace live backend calibration, queue, "
        "depth, or budget approval checks."
    )
    manual_review_platforms = tuple(
        decision.platform.name for decision in decisions if decision.status == "manual_review"
    )
    return FeedbackSubmissionPackage(
        experiment_id=experiment_id,
        circuit=summary,
        budget=budget,
        platform_readiness=decisions,
        claim_boundary=claim_boundary,
        dossier=build_s1_feedback_job_dossier(
            circuit_summary={
                "n_qubits": summary.n_qubits,
                "n_clbits": summary.n_clbits,
                "depth": summary.depth,
                "operation_counts": dict(summary.operation_counts),
                "n_rounds": summary.n_rounds,
            },
            qpu_budget={
                "circuits": budget.circuits,
                "shots_per_circuit": budget.shots_per_circuit,
                "repetitions": budget.repetitions,
                "estimated_execution_seconds": budget.estimated_execution_seconds,
                "total_reserved_seconds": budget.total_reserved_seconds,
            },
            ready_platforms=tuple(
                decision.platform.name for decision in decisions if decision.status == "ready"
            ),
            manual_review_platforms=manual_review_platforms,
        ),
    )


def summarise_feedback_circuit(
    circuit: QuantumCircuit,
    *,
    n_rounds: int,
) -> FeedbackCircuitSummary:
    """Summarise a monitored feedback circuit without provider submission."""
    counts = {str(name): int(count) for name, count in circuit.count_ops().items()}
    has_measure = counts.get("measure", 0) > 0
    has_conditional = _circuit_has_conditionals(circuit)
    return FeedbackCircuitSummary(
        n_qubits=circuit.num_qubits,
        n_clbits=circuit.num_clbits,
        depth=int(circuit.depth()),
        operation_counts=counts,
        has_mid_circuit_measurement=has_measure,
        has_conditional_control=has_conditional,
        n_rounds=n_rounds,
    )


def assess_platform_readiness(
    platform: FeedbackPlatformCapability,
    circuit: FeedbackCircuitSummary,
    budget: FeedbackBudgetEstimate,
) -> PlatformReadiness:
    """Assess whether one platform can execute the S1 dynamic feedback payload."""
    reasons: list[str] = []
    if circuit.n_qubits > platform.max_qubits:
        reasons.append(
            f"requires {circuit.n_qubits} qubits but platform declares {platform.max_qubits}"
        )
    if circuit.has_mid_circuit_measurement and not platform.supports_mid_circuit_measurement:
        reasons.append("payload requires mid-circuit measurement")
    if circuit.has_conditional_control and not platform.supports_conditional_rotation:
        reasons.append("payload requires conditional rotations")
    if "reset" in circuit.operation_counts and not platform.supports_conditional_reset:
        reasons.append("payload requires conditional reset")
    if budget.total_reserved_seconds <= 0.0:
        reasons.append("budget estimate must reserve positive execution time")

    if reasons:
        status: ReadinessStatus = "manual_review" if platform.supports_native_xy else "blocked"
    else:
        status = "ready"
        reasons.append("declared capabilities satisfy the dynamic-circuit payload")

    payload = {
        "kind": platform.kind,
        "n_qubits": circuit.n_qubits,
        "n_clbits": circuit.n_clbits,
        "depth": circuit.depth,
        "operation_counts": dict(circuit.operation_counts),
        "shots_per_circuit": budget.shots_per_circuit,
        "repetitions": budget.repetitions,
        "estimated_execution_seconds": budget.estimated_execution_seconds,
    }
    return PlatformReadiness(
        platform=platform,
        status=status,
        reasons=tuple(reasons),
        payload=payload,
    )


def _circuit_has_conditionals(circuit: QuantumCircuit) -> bool:
    for instruction in circuit.data:
        operation = instruction.operation
        if getattr(operation, "condition", None) is not None:
            return True
        if operation.name in {"if_else", "while_loop", "for_loop", "switch_case"}:
            return True
    return False


def _require_non_negative(value: float, name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
