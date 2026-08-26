# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — adaptive FIM next-experiment proposals
"""Uncertainty-aware, policy-bounded adaptive FIM batch proposals.

The product route consumes disjoint leakage/retention counts, applies a Wilson
score interval and minimum-shot gate, and proposes only a bounded decrease or
hold for a *future* static ``lambda_fim`` batch. hardware-safety approves the complete
paired-arm dry-run plan before any schedule is generated. Nothing here submits
a provider job, applies a controller update, or validates closed-loop efficacy.

The original point-estimate functions remain available as explicitly labelled
compatibility helpers. They are not used by the adaptive-FIM product or evidence lane.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Final, Literal, cast

import numpy as np

from ..hardware_safe_execution import DryRunPlan, dry_run_execution_plan

FeedbackMode = Literal["leakage_suppression", "retention_recovery"]
ProposalDecision = Literal["decrease", "hold"]
PlanOutcome = Literal["allowed_plan", "refused"]
WitnessSource = Literal["unspecified", "synthetic", "simulator", "hardware_replay"]

ADAPTIVE_FIM_SCHEMA: Final[str] = "adaptive_fim_feedback.v3"
ADAPTIVE_FIM_CLAIM_BOUNDARY: Final[str] = (
    "uncertainty-aware batch-level next-experiment proposals under hardware-safe "
    "no-submit dry-run budgets; offline replay is not closed-loop validation; no "
    "provider submission, live QPU feedback, FIM protection, optimal-policy, "
    "hardware-efficacy, realtime control, or quantum-advantage claim"
)


@dataclass(frozen=True, slots=True)
class AdaptiveFIMConfig:
    """Configuration for conservative, clipped ``lambda_fim`` proposals."""

    lambda_min: float = 0.0
    lambda_max: float = 8.0
    step_gain: float = 1.0
    max_delta_per_batch: float = 1.0
    target_leakage: float = 0.0
    target_retention: float = 1.0
    deadband: float = 0.0
    confidence_z: float = 1.959963984540054
    min_shots: int = 256
    mode: FeedbackMode = "leakage_suppression"

    def __post_init__(self) -> None:
        """Validate finite bounds, targets, and uncertainty policy."""
        finite_fields = (
            (self.lambda_min, "lambda_min"),
            (self.lambda_max, "lambda_max"),
            (self.step_gain, "step_gain"),
            (self.max_delta_per_batch, "max_delta_per_batch"),
            (self.target_leakage, "target_leakage"),
            (self.target_retention, "target_retention"),
            (self.deadband, "deadband"),
            (self.confidence_z, "confidence_z"),
        )
        for value, name in finite_fields:
            _require_finite(value, name)
        if self.lambda_min < 0.0:
            raise ValueError("lambda_min must be non-negative")
        if self.lambda_max < self.lambda_min:
            raise ValueError("lambda_max must be >= lambda_min")
        if self.step_gain < 0.0:
            raise ValueError("step_gain must be non-negative")
        if self.max_delta_per_batch <= 0.0:
            raise ValueError("max_delta_per_batch must be positive")
        _require_probability(self.target_leakage, "target_leakage")
        _require_probability(self.target_retention, "target_retention")
        if self.deadband < 0.0 or self.deadband > 1.0:
            raise ValueError("deadband must be in [0, 1]")
        if self.confidence_z <= 0.0:
            raise ValueError("confidence_z must be positive")
        if isinstance(self.min_shots, bool) or self.min_shots <= 0:
            raise ValueError("min_shots must be a positive integer")
        if self.mode not in {"leakage_suppression", "retention_recovery"}:
            raise ValueError("mode must be leakage_suppression or retention_recovery")


@dataclass(frozen=True, slots=True)
class FIMWitness:
    """One observed leakage/retention witness.

    ``leakage_events`` and ``retention_events`` are disjoint categories from
    the same shot block. When supplied, both counts and ``shots`` are required
    and the probability fields must equal the corresponding count fractions.
    Point estimates without counts remain legal only for the compatibility API.
    """

    leakage: float
    retention: float
    depth: int | None = None
    shots: int | None = None
    leakage_events: int | None = None
    retention_events: int | None = None
    source: WitnessSource = "unspecified"
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        """Validate probability, count, and provenance consistency."""
        _require_probability(self.leakage, "leakage")
        _require_probability(self.retention, "retention")
        if self.depth is not None and (
            isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0
        ):
            raise ValueError("depth must be a non-negative integer when provided")
        if self.source not in {"unspecified", "synthetic", "simulator", "hardware_replay"}:
            raise ValueError(f"unknown witness source: {self.source!r}")
        if self.artifact_id is not None and not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty when provided")
        supplied_counts = (self.leakage_events, self.retention_events)
        if any(value is not None for value in supplied_counts) and (
            self.shots is None or any(value is None for value in supplied_counts)
        ):
            raise ValueError("count-bound witnesses require shots and both event counts")
        if self.shots is not None and (
            isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots <= 0
        ):
            raise ValueError("shots must be a positive integer when provided")
        if self.count_bound:
            shots = cast(int, self.shots)
            leakage_events = cast(int, self.leakage_events)
            retention_events = cast(int, self.retention_events)
            for value, name in (
                (leakage_events, "leakage_events"),
                (retention_events, "retention_events"),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{name} must be a non-negative integer")
                if value > shots:
                    raise ValueError(f"{name} must not exceed shots")
            if leakage_events + retention_events > shots:
                raise ValueError("leakage and retention events must be disjoint")
            tolerance = 1e-12
            if not math.isclose(
                self.leakage,
                leakage_events / shots,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError("leakage must match leakage_events / shots")
            if not math.isclose(
                self.retention,
                retention_events / shots,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError("retention must match retention_events / shots")

    @property
    def count_bound(self) -> bool:
        """Return whether a complete count triple is attached."""
        return (
            self.shots is not None
            and self.leakage_events is not None
            and self.retention_events is not None
        )

    @classmethod
    def from_counts(
        cls,
        *,
        leakage_events: int,
        retention_events: int,
        shots: int,
        depth: int | None = None,
        source: WitnessSource = "unspecified",
        artifact_id: str | None = None,
    ) -> FIMWitness:
        """Construct an internally consistent witness from disjoint counts."""
        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer")
        return cls(
            leakage=float(leakage_events / shots),
            retention=float(retention_events / shots),
            depth=depth,
            shots=shots,
            leakage_events=leakage_events,
            retention_events=retention_events,
            source=source,
            artifact_id=artifact_id,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready witness mapping."""
        return asdict(self) | {"count_bound": self.count_bound}


@dataclass(frozen=True, slots=True)
class BinomialInterval:
    """Closed Wilson score interval for one count proportion."""

    events: int
    shots: int
    estimate: float
    lower: float
    upper: float
    confidence_z: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready interval mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AdaptiveFIMStep:
    """One unapplied next-batch proposal and its decision evidence."""

    index: int
    lambda_in: float
    lambda_out: float
    witness: FIMWitness
    error_signal: float
    clipped: bool
    decision: ProposalDecision
    count_qualified: bool
    interval: BinomialInterval | None
    rationale: str
    schema: str = ADAPTIVE_FIM_SCHEMA
    claim_boundary: str = ADAPTIVE_FIM_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate proposal invariants."""
        if self.schema != ADAPTIVE_FIM_SCHEMA:
            raise ValueError(f"unknown adaptive FIM feedback schema: {self.schema!r}")
        if self.claim_boundary != ADAPTIVE_FIM_CLAIM_BOUNDARY:
            raise ValueError("adaptive FIM step claim boundary must not drift")
        if self.index < 0:
            raise ValueError("index must be non-negative")
        for value, name in (
            (self.lambda_in, "lambda_in"),
            (self.lambda_out, "lambda_out"),
            (self.error_signal, "error_signal"),
        ):
            _require_finite(value, name)
        if self.lambda_in < 0.0 or self.lambda_out < 0.0:
            raise ValueError("lambda values must be non-negative")
        if self.decision == "decrease" and not self.lambda_out < self.lambda_in:
            raise ValueError("decrease decisions must reduce lambda")
        if self.decision == "hold" and not math.isclose(self.lambda_out, self.lambda_in):
            raise ValueError("hold decisions must keep lambda fixed")
        if self.count_qualified != (self.interval is not None):
            raise ValueError("count_qualified must match interval presence")
        if not self.rationale.strip():
            raise ValueError("rationale must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready proposal mapping."""
        return {
            "schema": self.schema,
            "index": self.index,
            "lambda_in": self.lambda_in,
            "lambda_out": self.lambda_out,
            "witness": self.witness.to_dict(),
            "error_signal": self.error_signal,
            "clipped": self.clipped,
            "decision": self.decision,
            "count_qualified": self.count_qualified,
            "interval": None if self.interval is None else self.interval.to_dict(),
            "rationale": self.rationale,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveFIMObserverRecord:
    """Bounded co-design telemetry for one unapplied adaptive-FIM proposal."""

    observer_id: str
    action: ProposalDecision
    lambda_in: float
    lambda_out: float
    interval_lower: float | None
    interval_upper: float | None
    shots: int | None
    shot_policy_id: str
    source: WitnessSource
    hardware_execution: bool = False
    claim_boundary: str = ADAPTIVE_FIM_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate adapter telemetry without promoting an applied action."""
        if self.claim_boundary != ADAPTIVE_FIM_CLAIM_BOUNDARY:
            raise ValueError("adaptive FIM observer claim boundary must not drift")
        if not self.observer_id.strip() or not self.shot_policy_id.strip():
            raise ValueError("observer_id and shot_policy_id must be non-empty")
        if self.hardware_execution:
            raise ValueError("adaptive FIM observer records cannot claim hardware execution")
        if (self.interval_lower is None) != (self.interval_upper is None):
            raise ValueError("interval bounds must be both present or both absent")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready observer mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AdaptiveFIMPlan:
    """Complete hardware-safe budget, schedule, and observer decision."""

    outcome: PlanOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    budget: DryRunPlan
    steps: tuple[AdaptiveFIMStep, ...]
    observers: tuple[AdaptiveFIMObserverRecord, ...]
    schema: str = ADAPTIVE_FIM_SCHEMA
    claim_boundary: str = ADAPTIVE_FIM_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate fail-closed product-plan invariants."""
        if self.schema != ADAPTIVE_FIM_SCHEMA:
            raise ValueError(f"unknown adaptive FIM feedback schema: {self.schema!r}")
        if self.claim_boundary != ADAPTIVE_FIM_CLAIM_BOUNDARY:
            raise ValueError("adaptive FIM plan claim boundary must not drift")
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed != (self.outcome == "allowed_plan"):
            raise ValueError("allowed must match outcome")
        if self.allowed and self.blockers:
            raise ValueError("allowed plans cannot list blockers")
        if not self.allowed and (not self.blockers or self.steps or self.observers):
            raise ValueError("refused plans require blockers and no proposals")
        if len(self.steps) != len(self.observers):
            raise ValueError("every schedule step requires one observer record")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready product payload."""
        return {
            "schema": self.schema,
            "outcome": self.outcome,
            "allowed": self.allowed,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "budget": self.budget.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "observers": [observer.to_dict() for observer in self.observers],
            "provider_submission": False,
            "hardware_execution": False,
            "closed_loop_validated": False,
            "claim_boundary": self.claim_boundary,
        }


def wilson_score_interval(events: int, shots: int, *, z: float) -> BinomialInterval:
    """Return the two-sided Wilson score interval for a binomial proportion."""
    if isinstance(events, bool) or not isinstance(events, int) or events < 0:
        raise ValueError("events must be a non-negative integer")
    if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if events > shots:
        raise ValueError("events must not exceed shots")
    _require_finite(z, "z")
    if z <= 0.0:
        raise ValueError("z must be positive")
    estimate = events / shots
    z_squared = z * z
    denominator = 1.0 + z_squared / shots
    centre = (estimate + z_squared / (2.0 * shots)) / denominator
    radius = (
        z
        * math.sqrt(estimate * (1.0 - estimate) / shots + z_squared / (4.0 * shots**2))
        / denominator
    )
    return BinomialInterval(
        events=events,
        shots=shots,
        estimate=float(estimate),
        lower=float(max(0.0, centre - radius)),
        upper=float(min(1.0, centre + radius)),
        confidence_z=float(z),
    )


def propose_next_lambda(
    current_lambda: float,
    witness: FIMWitness,
    config: AdaptiveFIMConfig | None = None,
) -> AdaptiveFIMStep:
    """Return the legacy point-estimate proposal for compatibility only.

    This helper reproduces the original proportional rule. It has no count
    qualification and must not be used as calibration or hardware evidence.
    """
    cfg = config or AdaptiveFIMConfig()
    _validate_current_lambda(current_lambda)
    if cfg.mode == "leakage_suppression":
        error_signal = witness.leakage - cfg.target_leakage
        rationale = "legacy point estimate: reduce lambda when leakage is above target"
    else:
        error_signal = cfg.target_retention - witness.retention
        rationale = "legacy point estimate: reduce lambda when retention is below target"
    proposed = (
        current_lambda
        if abs(error_signal) <= cfg.deadband
        else current_lambda - cfg.step_gain * error_signal
    )
    lambda_out = float(np.clip(proposed, cfg.lambda_min, cfg.lambda_max))
    decision: ProposalDecision = "decrease" if lambda_out < current_lambda else "hold"
    return AdaptiveFIMStep(
        index=0,
        lambda_in=float(current_lambda),
        lambda_out=lambda_out,
        witness=witness,
        error_signal=float(error_signal),
        clipped=not np.isclose(lambda_out, proposed),
        decision=decision,
        count_qualified=False,
        interval=None,
        rationale=rationale,
    )


def propose_count_aware_lambda(
    current_lambda: float,
    witness: FIMWitness,
    config: AdaptiveFIMConfig | None = None,
) -> AdaptiveFIMStep:
    """Return a conservative count-qualified next-batch proposal.

    A harmful-direction interval must exclude the target plus deadband before a
    decrease is proposed. Missing or underpowered counts hold. The product does
    not propose increases after the committed negative FIM hardware result.
    """
    cfg = config or AdaptiveFIMConfig()
    _validate_current_lambda(current_lambda)
    if not witness.count_bound:
        return _hold_step(
            current_lambda,
            witness,
            rationale="hold: count-bound leakage and retention evidence is required",
        )
    shots = cast(int, witness.shots)
    leakage_events = cast(int, witness.leakage_events)
    retention_events = cast(int, witness.retention_events)
    if shots < cfg.min_shots:
        return _hold_step(
            current_lambda,
            witness,
            rationale=(
                f"hold: shots {shots} below min_shots {cfg.min_shots}; "
                "uncertainty gate not qualified"
            ),
        )
    if cfg.mode == "leakage_suppression":
        interval = wilson_score_interval(
            leakage_events,
            shots,
            z=cfg.confidence_z,
        )
        error_signal = interval.lower - (cfg.target_leakage + cfg.deadband)
        rationale = "leakage Wilson lower bound exceeds target plus deadband"
    else:
        interval = wilson_score_interval(
            retention_events,
            shots,
            z=cfg.confidence_z,
        )
        error_signal = (cfg.target_retention - cfg.deadband) - interval.upper
        rationale = "retention Wilson upper bound is below target minus deadband"
    if error_signal <= 0.0 or cfg.step_gain == 0.0:
        return AdaptiveFIMStep(
            index=0,
            lambda_in=float(current_lambda),
            lambda_out=float(current_lambda),
            witness=witness,
            error_signal=float(error_signal),
            clipped=False,
            decision="hold",
            count_qualified=True,
            interval=interval,
            rationale="hold: harmful-direction interval does not clear the decision boundary",
        )
    delta = min(cfg.max_delta_per_batch, cfg.step_gain * error_signal)
    proposed = current_lambda - delta
    lambda_out = float(np.clip(proposed, cfg.lambda_min, cfg.lambda_max))
    decision: ProposalDecision = "decrease" if lambda_out < current_lambda else "hold"
    return AdaptiveFIMStep(
        index=0,
        lambda_in=float(current_lambda),
        lambda_out=lambda_out,
        witness=witness,
        error_signal=float(error_signal),
        clipped=not np.isclose(lambda_out, proposed),
        decision=decision,
        count_qualified=True,
        interval=interval,
        rationale=rationale if decision == "decrease" else "hold: lambda_min reached",
    )


def adaptive_lambda_schedule(
    initial_lambda: float,
    witnesses: list[FIMWitness],
    config: AdaptiveFIMConfig | None = None,
) -> list[AdaptiveFIMStep]:
    """Generate the legacy point-estimate schedule for compatibility only."""
    return _thread_schedule(initial_lambda, witnesses, config, count_aware=False)


def adaptive_count_aware_schedule(
    initial_lambda: float,
    witnesses: tuple[FIMWitness, ...] | list[FIMWitness],
    config: AdaptiveFIMConfig | None = None,
) -> tuple[AdaptiveFIMStep, ...]:
    """Generate a deterministic uncertainty-aware proposal schedule."""
    return tuple(_thread_schedule(initial_lambda, list(witnesses), config, count_aware=True))


def plan_adaptive_fim_schedule(
    initial_lambda: float,
    witnesses: tuple[FIMWitness, ...],
    *,
    policy_id: str,
    shots_per_arm: int,
    config: AdaptiveFIMConfig | None = None,
    request_hardware: bool = False,
) -> AdaptiveFIMPlan:
    """Gate a paired-arm future schedule through hardware-safe policy.

    Each witness corresponds to one planned next batch with a control arm and a
    proposed-lambda arm. Consequently the dry-run budget receives
    ``n_params=len(witnesses)`` and its two-sided evaluation bound accounts for
    exactly two evaluations per batch. Refusal occurs before interval scoring
    or co-design telemetry creation.
    """
    if not witnesses:
        raise ValueError("witnesses must be non-empty")
    budget = dry_run_execution_plan(
        policy_id,
        n_params=len(witnesses),
        shots_per_evaluation=shots_per_arm,
        shift_terms=1,
        would_submit=request_hardware,
    )
    blockers = list(budget.blockers)
    if request_hardware:
        blockers.append(
            "adaptive FIM hardware execution is unavailable on this offline proposal surface"
        )
    if blockers:
        return AdaptiveFIMPlan(
            outcome="refused",
            allowed=False,
            reason="adaptive FIM plan refused before schedule generation",
            blockers=tuple(dict.fromkeys(blockers)),
            budget=budget,
            steps=(),
            observers=(),
        )
    steps = adaptive_count_aware_schedule(initial_lambda, witnesses, config)
    observers = tuple(observer_record_from_step(step, policy_id=policy_id) for step in steps)
    return AdaptiveFIMPlan(
        outcome="allowed_plan",
        allowed=True,
        reason=(
            "offline next-batch proposals allowed under the hardware-safe no-submit "
            "dry-run budget; "
            "no provider submission or controller application occurred"
        ),
        blockers=(),
        budget=budget,
        steps=steps,
        observers=observers,
    )


def observer_record_from_step(
    step: AdaptiveFIMStep,
    *,
    policy_id: str,
) -> AdaptiveFIMObserverRecord:
    """Map one proposal step to bounded co-design observer telemetry."""
    if not policy_id.strip():
        raise ValueError("policy_id must be non-empty")
    interval = step.interval
    return AdaptiveFIMObserverRecord(
        observer_id=f"adaptive_fim:{policy_id}:{step.index}",
        action=step.decision,
        lambda_in=step.lambda_in,
        lambda_out=step.lambda_out,
        interval_lower=None if interval is None else interval.lower,
        interval_upper=None if interval is None else interval.upper,
        shots=step.witness.shots,
        shot_policy_id=policy_id,
        source=step.witness.source,
    )


def _thread_schedule(
    initial_lambda: float,
    witnesses: list[FIMWitness],
    config: AdaptiveFIMConfig | None,
    *,
    count_aware: bool,
) -> list[AdaptiveFIMStep]:
    """Thread lambda through either proposal function."""
    cfg = config or AdaptiveFIMConfig()
    _validate_current_lambda(initial_lambda)
    current = float(initial_lambda)
    steps: list[AdaptiveFIMStep] = []
    for index, witness in enumerate(witnesses):
        candidate = (
            propose_count_aware_lambda(current, witness, cfg)
            if count_aware
            else propose_next_lambda(current, witness, cfg)
        )
        step = replace(candidate, index=index)
        steps.append(step)
        current = step.lambda_out
    return steps


def _hold_step(
    current_lambda: float,
    witness: FIMWitness,
    *,
    rationale: str,
) -> AdaptiveFIMStep:
    """Build a fail-closed unqualified hold step."""
    return AdaptiveFIMStep(
        index=0,
        lambda_in=float(current_lambda),
        lambda_out=float(current_lambda),
        witness=witness,
        error_signal=0.0,
        clipped=False,
        decision="hold",
        count_qualified=False,
        interval=None,
        rationale=rationale,
    )


def _validate_current_lambda(value: float) -> None:
    """Validate a non-negative finite current lambda."""
    _require_finite(value, "current_lambda")
    if value < 0.0:
        raise ValueError("current_lambda must be non-negative")


def _require_finite(value: float, name: str) -> None:
    """Reject non-finite scalar values."""
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_probability(value: float, name: str) -> None:
    """Reject values outside the closed unit interval."""
    _require_finite(value, name)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


__all__ = [
    "ADAPTIVE_FIM_CLAIM_BOUNDARY",
    "ADAPTIVE_FIM_SCHEMA",
    "AdaptiveFIMConfig",
    "AdaptiveFIMObserverRecord",
    "AdaptiveFIMPlan",
    "AdaptiveFIMStep",
    "BinomialInterval",
    "FIMWitness",
    "FeedbackMode",
    "PlanOutcome",
    "ProposalDecision",
    "WitnessSource",
    "adaptive_count_aware_schedule",
    "adaptive_lambda_schedule",
    "observer_record_from_step",
    "plan_adaptive_fim_schedule",
    "propose_count_aware_lambda",
    "propose_next_lambda",
    "wilson_score_interval",
]
