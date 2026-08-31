# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Active sensing / experimental design product
"""Policy-bounded active sensing over existing analytic-design surfaces.

This module ranks synthetic scalar observations by Gaussian expected
information gain, applies the hardware-safe no-submit shot budget before
scoring, runs the existing analytic candidate-design protocol as evidence, and
emits a co-design observer record. It never submits hardware work or promotes
the research-only NV 20 T surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from .benchmarks.s3_design_protocol import (
    default_s3_design_protocol,
    score_s3_candidates,
)
from .hardware_safe_execution import DryRunPlan, dry_run_execution_plan

ACTIVE_SENSING_PRODUCT_SCHEMA: Final[str] = "active_sensing_product.v2"
ACTIVE_SENSING_CLAIM_BOUNDARY: Final[str] = (
    "synthetic local information-gain planning under hardware-safe no-submit shot "
    "budgets; candidate-design evidence is analytic/simulator-only; no adaptive QPU "
    "loop, sensing advantage, or NV 20 T hardware-calibration claim"
)
_ANALYTIC_DESIGN_PROTOCOL_ID: Final[str] = "ml_augmented_pulse_ansatz_design_2026-05-06"
PlanOutcome = Literal["allowed_plan", "refused"]


@dataclass(frozen=True, slots=True)
class SensingInventoryRow:
    """One existing sensing surface and its active-sensing support posture."""

    surface_id: str
    module_path: str
    role: str
    posture: str
    hardware_execution: bool = False


@dataclass(frozen=True, slots=True)
class InformationGainCandidate:
    """Synthetic scalar observation considered by active sensing.

    Parameters
    ----------
    observable_id
        Stable observation identifier.
    prior_variance
        Current scalar posterior variance before measurement.
    sensitivity
        Local derivative of the observation mean with respect to the target.
    noise_variance
        Per-shot observation-noise variance.
    channel
        Co-design observer-channel label.

    """

    observable_id: str
    prior_variance: float
    sensitivity: float
    noise_variance: float
    channel: str = "gradient_observer"

    def __post_init__(self) -> None:
        """Validate finite scalar observation parameters."""
        if not self.observable_id.strip() or not self.channel.strip():
            raise ValueError("observable_id and channel must be non-empty")
        values = (self.prior_variance, self.sensitivity, self.noise_variance)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("candidate parameters must be finite")
        if self.prior_variance <= 0.0 or self.noise_variance <= 0.0:
            raise ValueError("prior_variance and noise_variance must be positive")


@dataclass(frozen=True, slots=True)
class InformationGainScore:
    """Expected posterior reduction for one synthetic scalar observation."""

    observable_id: str
    channel: str
    shots: int
    signal_to_noise: float
    expected_information_gain_nats: float
    posterior_variance: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready score mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnalyticDesignEvidenceRow:
    """One descriptive analytic candidate-design evidence row."""

    analytic_design_protocol_id: str
    candidate_label: str
    family: Literal["ansatz", "pulse"]
    status: str
    score: float
    metrics: dict[str, float | int | str]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready analytic-design evidence mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActiveSensingObserverRecord:
    """Co-design observer-channel record for the selected measurement."""

    observer_id: str
    channel: str
    selected_observable_id: str
    expected_information_gain_nats: float
    posterior_variance: float
    shots: int
    shot_policy_id: str
    analytic_design_protocol_id: str
    hardware_execution: bool = False
    claim_boundary: str = ACTIVE_SENSING_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready observer telemetry mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActiveSensingPlan:
    """Complete policy, ranking, analytic evidence, and observer decision."""

    outcome: PlanOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    budget: DryRunPlan
    scores: tuple[InformationGainScore, ...]
    selected: InformationGainScore | None
    analytic_design_evidence: tuple[AnalyticDesignEvidenceRow, ...]
    observer: ActiveSensingObserverRecord | None
    schema: str = ACTIVE_SENSING_PRODUCT_SCHEMA
    claim_boundary: str = ACTIVE_SENSING_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Reject stale schemas and claim-boundary drift."""
        if self.schema != ACTIVE_SENSING_PRODUCT_SCHEMA:
            raise ValueError("unexpected active-sensing product schema")
        if self.claim_boundary != ACTIVE_SENSING_CLAIM_BOUNDARY:
            raise ValueError("active-sensing claim boundary drift")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready product payload."""
        return {
            "schema": self.schema,
            "outcome": self.outcome,
            "allowed": self.allowed,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "budget": self.budget.to_dict(),
            "scores": [score.to_dict() for score in self.scores],
            "selected": None if self.selected is None else self.selected.to_dict(),
            "analytic_design_evidence": [row.to_dict() for row in self.analytic_design_evidence],
            "observer": None if self.observer is None else self.observer.to_dict(),
            "hardware_execution": False,
            "claim_boundary": self.claim_boundary,
        }


_INVENTORY: Final[tuple[SensingInventoryRow, ...]] = (
    SensingInventoryRow(
        "quantum_fisher_sync_readiness",
        "scpn_quantum_control.analysis.sensing",
        "QFI and sync-order readiness source",
        "local_research",
    ),
    SensingInventoryRow(
        "analytic_candidate_design",
        "scpn_quantum_control.active_sensing_product.AnalyticDesignEvidenceRow",
        "analytic candidate evidence harness",
        "no_qpu",
    ),
    SensingInventoryRow(
        "shot_budget",
        "scpn_quantum_control.hardware_safe_execution",
        "hardware-safe no-submit budget authority",
        "policy_only",
    ),
    SensingInventoryRow(
        "nv_20t",
        "scpn_quantum_control.sensing.nv_magnetometry_20T",
        "research-only hardware-blocked path",
        "hardware_blocked",
    ),
    SensingInventoryRow(
        "codesign_observer",
        "scpn_quantum_control.active_sensing_product.ActiveSensingObserverRecord",
        "ports-over-adapters observer telemetry",
        "adapter_only",
    ),
)


def sensing_surface_inventory() -> tuple[SensingInventoryRow, ...]:
    """Return the frozen active-sensing surface inventory."""
    return _INVENTORY


def score_expected_information_gain(
    candidate: InformationGainCandidate,
    *,
    shots: int,
) -> InformationGainScore:
    """Score one scalar Gaussian measurement in natural-log information units.

    The conjugate Gaussian update gives
    ``I = 0.5 log(1 + shots * variance * sensitivity² / noise_variance)``.
    This is a synthetic local-design score, not empirical hardware evidence.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    signal_to_noise = (
        float(shots)
        * candidate.prior_variance
        * candidate.sensitivity**2
        / candidate.noise_variance
    )
    information_gain = 0.5 * float(np.log1p(signal_to_noise))
    posterior_variance = candidate.prior_variance / (1.0 + signal_to_noise)
    return InformationGainScore(
        observable_id=candidate.observable_id,
        channel=candidate.channel,
        shots=shots,
        signal_to_noise=signal_to_noise,
        expected_information_gain_nats=information_gain,
        posterior_variance=posterior_variance,
    )


def plan_active_sensing(
    candidates: tuple[InformationGainCandidate, ...],
    k_matrix: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    policy_id: str,
    shots_per_observable: int,
    request_hardware: bool = False,
) -> ActiveSensingPlan:
    """Build a complete plan through the budget and analytic-design harnesses.

    Budget enforcement occurs before information scoring or analytic
    circuit/pulse construction. Hardware-adaptive requests are always refused
    on this surface.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")
    labels = [candidate.observable_id for candidate in candidates]
    if len(set(labels)) != len(labels):
        raise ValueError("candidate observable_id values must be unique")
    budget = dry_run_execution_plan(
        policy_id,
        n_params=len(candidates),
        shots_per_evaluation=shots_per_observable,
        shift_terms=1,
        would_submit=request_hardware,
    )
    blockers = list(budget.blockers)
    if request_hardware:
        blockers.append(
            "adaptive hardware sensing requires a separately authorised provider surface"
        )
    if blockers:
        return ActiveSensingPlan(
            outcome="refused",
            allowed=False,
            reason="active sensing refused before evaluation",
            blockers=tuple(dict.fromkeys(blockers)),
            budget=budget,
            scores=(),
            selected=None,
            analytic_design_evidence=(),
            observer=None,
        )

    scores = tuple(
        sorted(
            (
                score_expected_information_gain(candidate, shots=shots_per_observable)
                for candidate in candidates
            ),
            key=lambda row: (-row.expected_information_gain_nats, row.observable_id),
        )
    )
    selected = scores[0]
    protocol = default_s3_design_protocol()
    design_rows = tuple(
        AnalyticDesignEvidenceRow(
            analytic_design_protocol_id=_ANALYTIC_DESIGN_PROTOCOL_ID,
            candidate_label=row.candidate_label,
            family=row.family,
            status=row.status,
            score=row.score,
            metrics=dict(row.metrics),
            claim_boundary=row.claim_boundary,
        )
        for row in score_s3_candidates(protocol, k_matrix, omega)
    )
    observer = ActiveSensingObserverRecord(
        observer_id=f"active-sensing:{policy_id}:{selected.observable_id}",
        channel=selected.channel,
        selected_observable_id=selected.observable_id,
        expected_information_gain_nats=selected.expected_information_gain_nats,
        posterior_variance=selected.posterior_variance,
        shots=selected.shots,
        shot_policy_id=policy_id,
        analytic_design_protocol_id=_ANALYTIC_DESIGN_PROTOCOL_ID,
    )
    return ActiveSensingPlan(
        outcome="allowed_plan",
        allowed=True,
        reason="local synthetic active-sensing plan allowed; no hardware submission occurred",
        blockers=(),
        budget=budget,
        scores=scores,
        selected=selected,
        analytic_design_evidence=design_rows,
        observer=observer,
    )


def demo_information_gain_candidates() -> tuple[InformationGainCandidate, ...]:
    """Return deterministic synthetic candidates for docs and integration tests."""
    return (
        InformationGainCandidate("sync_order", 0.40, 1.20, 0.30),
        InformationGainCandidate("spectral_gap", 0.25, 0.85, 0.20),
        InformationGainCandidate("qfi_peak", 0.55, 1.40, 0.45),
    )


__all__ = [
    "ACTIVE_SENSING_CLAIM_BOUNDARY",
    "ACTIVE_SENSING_PRODUCT_SCHEMA",
    "AnalyticDesignEvidenceRow",
    "ActiveSensingObserverRecord",
    "ActiveSensingPlan",
    "InformationGainCandidate",
    "InformationGainScore",
    "PlanOutcome",
    "SensingInventoryRow",
    "demo_information_gain_candidates",
    "plan_active_sensing",
    "score_expected_information_gain",
    "sensing_surface_inventory",
]
