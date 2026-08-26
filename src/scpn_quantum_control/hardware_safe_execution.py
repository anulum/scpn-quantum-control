# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware-safe gradient execution product
"""Fail-closed hardware-safe gradient execution policy product.

Productises Axis-5 safety: **no-submit default**, shot budgets, honest cost-model
status, dry-run planning, enforce/refuse for would-submit and over-budget, and
structured audit decisions.

Does **not** submit QPU/provider jobs or invent live hardware results. Complements
:mod:`scpn_quantum_control.phase.hardware_gradient_policy` with a versioned
public catalogue + probe surface.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

CostModelStatus = Literal["unavailable", "rate_table", "blocked"]
"""Cost-model honesty vocabulary (never invent vendor rate tables)."""

EnforceMode = Literal["dry_run", "would_submit", "ticketed_prep"]
"""Request modes for enforce / dry-run probes."""

DecisionOutcome = Literal["allowed_plan", "refused", "blocked"]
"""Structured audit / enforce outcomes."""

HARDWARE_SAFE_EXECUTION_SCHEMA: Final[str] = "hardware_safe_execution.v1"
"""JSON schema identifier for serialised policy payloads."""

HARDWARE_SAFE_EXECUTION_CLAIM_BOUNDARY: Final[str] = (
    "hardware-safe execution product only; no-submit is the default; dry-run "
    "plans estimate shots/cost status without provider submission; enforce "
    "refuses would-submit and over-budget without owner-gated allow; this "
    "surface never executes QPU jobs or invents hardware results"
)
"""Shared claim boundary for policies, plans, and audit records."""

# Fixture rate only for the explicit rate_table demo policy (documented test fixture).
_FIXTURE_RATE_USD_PER_SHOT: Final[float] = 0.0  # free fixture; real rates are owner data


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Immutable hardware-safe execution policy record.

    Attributes
    ----------
    policy_id
        Stable catalogue identifier.
    summary
        Short description.
    no_submit
        When true (default product posture), any would-submit path is refused.
    owner_allow_submit
        Explicit owner gate required before would-submit can be considered
        (still does not perform live submission on this surface).
    max_shots_per_evaluation
        Per-evaluation shot ceiling.
    max_total_shots
        Total shot budget for a planned request.
    max_params
        Parameter count ceiling for planning.
    max_shift_terms
        Shift-term ceiling for parameter-shift style plans.
    cost_model_status
        Honest cost-model status (unavailable/blocked without inventing rates).
    cost_usd_per_shot
        Optional fixture rate when ``cost_model_status == rate_table``; else 0.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    policy_id: str
    summary: str
    no_submit: bool
    owner_allow_submit: bool
    max_shots_per_evaluation: int
    max_total_shots: int
    max_params: int
    max_shift_terms: int
    cost_model_status: CostModelStatus
    cost_usd_per_shot: float = 0.0
    as_of: str = "2026-07-23"
    claim_boundary: str = HARDWARE_SAFE_EXECUTION_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate policy invariants."""
        if not self.policy_id or not self.policy_id.strip():
            raise ValueError("policy_id must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.cost_model_status not in {"unavailable", "rate_table", "blocked"}:
            raise ValueError(f"unknown cost_model_status: {self.cost_model_status!r}")
        if self.max_shots_per_evaluation <= 0:
            raise ValueError("max_shots_per_evaluation must be positive")
        if self.max_total_shots <= 0:
            raise ValueError("max_total_shots must be positive")
        if self.max_params <= 0:
            raise ValueError("max_params must be positive")
        if self.max_shift_terms <= 0:
            raise ValueError("max_shift_terms must be positive")
        if self.max_total_shots < self.max_shots_per_evaluation:
            raise ValueError("max_total_shots must be >= max_shots_per_evaluation")
        if self.cost_usd_per_shot < 0:
            raise ValueError("cost_usd_per_shot must be non-negative")
        if self.cost_model_status != "rate_table" and self.cost_usd_per_shot != 0.0:
            raise ValueError("non-rate_table policies must not declare cost_usd_per_shot")
        if self.cost_model_status == "rate_table" and self.cost_usd_per_shot < 0:
            raise ValueError("rate_table requires non-negative cost_usd_per_shot")
        if not self.no_submit and not self.owner_allow_submit:
            raise ValueError(
                "no_submit=False requires owner_allow_submit=True (refuse silent spend posture)"
            )
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this policy."""
        return {
            "policy_id": self.policy_id,
            "summary": self.summary,
            "no_submit": self.no_submit,
            "owner_allow_submit": self.owner_allow_submit,
            "max_shots_per_evaluation": self.max_shots_per_evaluation,
            "max_total_shots": self.max_total_shots,
            "max_params": self.max_params,
            "max_shift_terms": self.max_shift_terms,
            "cost_model_status": self.cost_model_status,
            "cost_usd_per_shot": self.cost_usd_per_shot,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class DryRunPlan:
    """Structured dry-run execution plan (never a live submission).

    Attributes
    ----------
    policy_id
        Policy used for planning.
    n_params
        Requested parameter count.
    shift_terms
        Shift terms for parameter-shift style estimation.
    shots_per_evaluation
        Planned shots per evaluation.
    evaluations
        Estimated evaluation count (``2 * n_params * shift_terms`` style bound).
    estimated_total_shots
        Product of shots and evaluations.
    estimated_cost_usd
        Cost when rate table present; else ``None``.
    cost_model_status
        Honesty status for cost.
    would_submit
        Whether the request intended provider submission.
    outcome
        Plan outcome (allowed_plan vs refused).
    reason
        Human-readable reason.
    blockers
        Non-empty when refused.

    """

    policy_id: str
    n_params: int
    shift_terms: int
    shots_per_evaluation: int
    evaluations: int
    estimated_total_shots: int
    estimated_cost_usd: float | None
    cost_model_status: CostModelStatus
    would_submit: bool
    outcome: DecisionOutcome
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = HARDWARE_SAFE_EXECUTION_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate dry-run plan invariants."""
        if not self.policy_id or not self.policy_id.strip():
            raise ValueError("policy_id must be non-empty")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.outcome not in {"allowed_plan", "refused", "blocked"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if self.outcome == "allowed_plan" and self.blockers:
            raise ValueError("allowed_plan cannot list blockers")
        if self.outcome != "allowed_plan" and not self.blockers:
            raise ValueError("refused/blocked plans require blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")
        if self.n_params <= 0 or self.shift_terms <= 0 or self.shots_per_evaluation <= 0:
            raise ValueError("plan dimensions must be positive")
        if self.evaluations <= 0 or self.estimated_total_shots <= 0:
            raise ValueError("evaluations and estimated_total_shots must be positive")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this plan."""
        return {
            "policy_id": self.policy_id,
            "n_params": self.n_params,
            "shift_terms": self.shift_terms,
            "shots_per_evaluation": self.shots_per_evaluation,
            "evaluations": self.evaluations,
            "estimated_total_shots": self.estimated_total_shots,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cost_model_status": self.cost_model_status,
            "would_submit": self.would_submit,
            "outcome": self.outcome,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class EnforceDecision:
    """Fail-closed enforce decision for an execution request.

    Attributes
    ----------
    policy_id
        Policy applied.
    mode
        Request mode.
    allowed
        Whether planning may proceed (never means live QPU submit succeeded).
    outcome
        Structured outcome label.
    estimated_total_shots
        Estimated total shots for the request (0 when dimensions invalid).
    reason
        Human-readable decision reason.
    blockers
        Open blockers when not allowed.
    audit_id
        Deterministic audit identifier for the decision.

    """

    policy_id: str
    mode: EnforceMode
    allowed: bool
    outcome: DecisionOutcome
    estimated_total_shots: int
    reason: str
    blockers: tuple[str, ...]
    audit_id: str
    claim_boundary: str = HARDWARE_SAFE_EXECUTION_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate enforce-decision invariants."""
        if not self.policy_id or not self.policy_id.strip():
            raise ValueError("policy_id must be non-empty")
        if self.mode not in {"dry_run", "would_submit", "ticketed_prep"}:
            raise ValueError(f"unknown mode: {self.mode!r}")
        if self.outcome not in {"allowed_plan", "refused", "blocked"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if not self.audit_id or not self.audit_id.strip():
            raise ValueError("audit_id must be non-empty")
        if self.allowed and self.blockers:
            raise ValueError("allowed decisions cannot list blockers")
        if not self.allowed and not self.blockers:
            raise ValueError("refused decisions require blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")
        if self.allowed and self.outcome != "allowed_plan":
            raise ValueError("allowed decisions must use outcome=allowed_plan")
        if not self.allowed and self.outcome == "allowed_plan":
            raise ValueError("refused decisions cannot use outcome=allowed_plan")
        if self.estimated_total_shots < 0:
            raise ValueError("estimated_total_shots must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this decision."""
        return {
            "policy_id": self.policy_id,
            "mode": self.mode,
            "allowed": self.allowed,
            "outcome": self.outcome,
            "estimated_total_shots": self.estimated_total_shots,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "audit_id": self.audit_id,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """Secret-free audit log record for a hardware-safe decision.

    Attributes
    ----------
    audit_id
        Deterministic audit identifier.
    policy_id
        Policy applied.
    mode
        Request mode.
    outcome
        Decision outcome.
    estimated_total_shots
        Planned shots.
    reason
        Decision reason (no secrets).

    """

    audit_id: str
    policy_id: str
    mode: EnforceMode
    outcome: DecisionOutcome
    estimated_total_shots: int
    reason: str
    claim_boundary: str = HARDWARE_SAFE_EXECUTION_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate audit record invariants."""
        if not self.audit_id or not self.audit_id.strip():
            raise ValueError("audit_id must be non-empty")
        if not self.policy_id or not self.policy_id.strip():
            raise ValueError("policy_id must be non-empty")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.mode not in {"dry_run", "would_submit", "ticketed_prep"}:
            raise ValueError(f"unknown mode: {self.mode!r}")
        if self.outcome not in {"allowed_plan", "refused", "blocked"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready audit mapping (secret-free by construction)."""
        return {
            "audit_id": self.audit_id,
            "policy_id": self.policy_id,
            "mode": self.mode,
            "outcome": self.outcome,
            "estimated_total_shots": self.estimated_total_shots,
            "reason": self.reason,
            "claim_boundary": self.claim_boundary,
            "contains_secrets": False,
        }


def _policy(
    policy_id: str,
    *,
    summary: str,
    no_submit: bool,
    owner_allow_submit: bool,
    max_shots_per_evaluation: int,
    max_total_shots: int,
    max_params: int = 32,
    max_shift_terms: int = 4,
    cost_model_status: CostModelStatus = "unavailable",
    cost_usd_per_shot: float = 0.0,
) -> ExecutionPolicy:
    """Build one catalogue policy row."""
    return ExecutionPolicy(
        policy_id=policy_id,
        summary=summary,
        no_submit=no_submit,
        owner_allow_submit=owner_allow_submit,
        max_shots_per_evaluation=max_shots_per_evaluation,
        max_total_shots=max_total_shots,
        max_params=max_params,
        max_shift_terms=max_shift_terms,
        cost_model_status=cost_model_status,
        cost_usd_per_shot=cost_usd_per_shot,
    )


_CANONICAL_POLICIES: Final[tuple[ExecutionPolicy, ...]] = (
    _policy(
        "default_no_submit",
        summary=(
            "Product default: no-submit hardware-safe posture with bounded dry-run "
            "shot budgets; cost model unavailable without owner rate table."
        ),
        no_submit=True,
        owner_allow_submit=False,
        max_shots_per_evaluation=1024,
        max_total_shots=65_536,
        cost_model_status="unavailable",
    ),
    _policy(
        "ci_dry_run_only",
        summary=(
            "CI/local dry-run only policy: tight shot budget, no-submit, "
            "cost model blocked (no invent rates in CI)."
        ),
        no_submit=True,
        owner_allow_submit=False,
        max_shots_per_evaluation=256,
        max_total_shots=4_096,
        max_params=8,
        cost_model_status="blocked",
    ),
    _policy(
        "owner_ticketed_prep",
        summary=(
            "Owner-gated ticketed preparation planning: still no live submit on "
            "this surface; owner_allow_submit records explicit gate for prep plans."
        ),
        no_submit=True,  # product surface never submits
        owner_allow_submit=True,
        max_shots_per_evaluation=4096,
        max_total_shots=131_072,
        cost_model_status="rate_table",
        cost_usd_per_shot=_FIXTURE_RATE_USD_PER_SHOT,
    ),
)


def _catalogue_map() -> dict[str, ExecutionPolicy]:
    """Return policy_id → record map; refuse blanks/duplicates."""
    mapping: dict[str, ExecutionPolicy] = {}
    for row in _CANONICAL_POLICIES:
        key = row.policy_id.strip()
        if not key:
            raise RuntimeError("hardware-safe policy catalogue contains blank policy_id")
        if key in mapping:
            raise RuntimeError(f"duplicate policy_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("hardware-safe policy catalogue must be non-empty")
    return mapping


_POLICY_BY_ID: Final[Mapping[str, ExecutionPolicy]] = _catalogue_map()


def list_execution_policy_ids() -> tuple[str, ...]:
    """Return all execution policy identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered policy identifiers.

    """
    return tuple(row.policy_id for row in _CANONICAL_POLICIES)


def get_execution_policy(policy_id: str) -> ExecutionPolicy:
    """Return one policy or raise for blank/unknown identifiers.

    Parameters
    ----------
    policy_id
        Catalogue policy key.

    Returns
    -------
    ExecutionPolicy
        Matching policy.

    Raises
    ------
    ValueError
        If ``policy_id`` is blank or unknown (fail closed).

    """
    if not policy_id or not str(policy_id).strip():
        raise ValueError("policy_id must be a non-empty string")
    key = str(policy_id).strip()
    try:
        return _POLICY_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown execution policy_id {key!r}; refuse invent-green hardware plan "
            f"(known_count={len(_POLICY_BY_ID)})"
        ) from exc


def iter_execution_policies(
    *,
    no_submit: bool | None = None,
) -> tuple[ExecutionPolicy, ...]:
    """Return filtered policies in stable order.

    Parameters
    ----------
    no_submit
        Optional filter on the no-submit flag.

    Returns
    -------
    tuple[ExecutionPolicy, ...]
        Matching policies.

    """
    rows: Iterable[ExecutionPolicy] = _CANONICAL_POLICIES
    if no_submit is not None:
        rows = (row for row in rows if row.no_submit is no_submit)
    return tuple(rows)


def default_execution_policy() -> ExecutionPolicy:
    """Return the product default no-submit policy.

    Returns
    -------
    ExecutionPolicy
        ``default_no_submit`` catalogue row.

    """
    return get_execution_policy("default_no_submit")


def _estimate_evaluations(n_params: int, shift_terms: int) -> int:
    """Bound evaluations for two-sided parameter-shift style planning."""
    return 2 * n_params * shift_terms


def _estimate_shots(
    *,
    n_params: int,
    shift_terms: int,
    shots_per_evaluation: int,
) -> tuple[int, int]:
    """Return ``(evaluations, estimated_total_shots)``."""
    evaluations = _estimate_evaluations(n_params, shift_terms)
    return evaluations, evaluations * shots_per_evaluation


def _cost_for(policy: ExecutionPolicy, total_shots: int) -> float | None:
    """Return estimated cost when a rate table is present; else ``None``."""
    if policy.cost_model_status != "rate_table":
        return None
    return float(total_shots) * float(policy.cost_usd_per_shot)


def dry_run_execution_plan(
    policy_id: str,
    *,
    n_params: int,
    shots_per_evaluation: int | None = None,
    shift_terms: int = 1,
    would_submit: bool = False,
) -> DryRunPlan:
    """Build a structured dry-run plan without provider submission.

    Parameters
    ----------
    policy_id
        Catalogue policy key.
    n_params
        Parameter count.
    shots_per_evaluation
        Shots per evaluation (defaults to policy max when ``None``).
    shift_terms
        Shift-term count.
    would_submit
        Whether the caller intends a live submit (always refused when policy
        ``no_submit`` is true).

    Returns
    -------
    DryRunPlan
        Structured plan or refuse outcome.

    Raises
    ------
    ValueError
        If ``policy_id`` is blank/unknown or dimensions are non-positive.

    """
    policy = get_execution_policy(policy_id)
    if n_params <= 0:
        raise ValueError("n_params must be positive")
    if shift_terms <= 0:
        raise ValueError("shift_terms must be positive")
    shots = (
        policy.max_shots_per_evaluation
        if shots_per_evaluation is None
        else int(shots_per_evaluation)
    )
    if shots <= 0:
        raise ValueError("shots_per_evaluation must be positive")

    evaluations, total = _estimate_shots(
        n_params=n_params,
        shift_terms=shift_terms,
        shots_per_evaluation=shots,
    )
    blockers: list[str] = []
    if would_submit and policy.no_submit:
        blockers.append("policy no_submit=True refuses would-submit path")
    if would_submit and not policy.owner_allow_submit:
        blockers.append("would-submit requires owner_allow_submit gate")
    if n_params > policy.max_params:
        blockers.append(f"n_params {n_params} exceeds policy max_params {policy.max_params}")
    if shift_terms > policy.max_shift_terms:
        blockers.append(
            f"shift_terms {shift_terms} exceeds policy max_shift_terms {policy.max_shift_terms}"
        )
    if shots > policy.max_shots_per_evaluation:
        blockers.append(
            f"shots_per_evaluation {shots} exceeds policy maximum "
            f"{policy.max_shots_per_evaluation}"
        )
    if total > policy.max_total_shots:
        blockers.append(
            f"estimated_total_shots {total} exceeds policy max_total_shots "
            f"{policy.max_total_shots}"
        )

    cost = _cost_for(policy, total)
    if blockers:
        return DryRunPlan(
            policy_id=policy.policy_id,
            n_params=n_params,
            shift_terms=shift_terms,
            shots_per_evaluation=shots,
            evaluations=evaluations,
            estimated_total_shots=total,
            estimated_cost_usd=cost,
            cost_model_status=policy.cost_model_status,
            would_submit=would_submit,
            outcome="refused",
            reason="dry-run plan refused: " + "; ".join(blockers),
            blockers=tuple(blockers),
        )
    return DryRunPlan(
        policy_id=policy.policy_id,
        n_params=n_params,
        shift_terms=shift_terms,
        shots_per_evaluation=shots,
        evaluations=evaluations,
        estimated_total_shots=total,
        estimated_cost_usd=cost,
        cost_model_status=policy.cost_model_status,
        would_submit=would_submit,
        outcome="allowed_plan",
        reason=(
            "dry-run plan allowed under policy; no provider submission occurred "
            f"(cost_model_status={policy.cost_model_status})"
        ),
        blockers=(),
    )


def enforce_execution_request(
    policy_id: str,
    *,
    mode: EnforceMode,
    n_params: int,
    shots_per_evaluation: int | None = None,
    shift_terms: int = 1,
    live_execution_ticket: str = "",
) -> EnforceDecision:
    """Enforce a request against policy (fail-closed; never submits jobs).

    Parameters
    ----------
    policy_id
        Catalogue policy key.
    mode
        ``dry_run``, ``would_submit``, or ``ticketed_prep``.
    n_params
        Parameter count.
    shots_per_evaluation
        Optional shots override.
    shift_terms
        Shift-term count.
    live_execution_ticket
        Optional ticket label (never a secret; blank is treated as missing).

    Returns
    -------
    EnforceDecision
        Allowed only for safe dry-run plans within budget; would-submit refused
        under no-submit defaults.

    Raises
    ------
    ValueError
        If identifiers / dimensions / mode are invalid.

    """
    if mode not in {"dry_run", "would_submit", "ticketed_prep"}:
        raise ValueError(f"unknown mode: {mode!r}")
    policy = get_execution_policy(policy_id)
    would_submit = mode == "would_submit"
    plan = dry_run_execution_plan(
        policy.policy_id,
        n_params=n_params,
        shots_per_evaluation=shots_per_evaluation,
        shift_terms=shift_terms,
        would_submit=would_submit,
    )
    blockers: list[str] = list(plan.blockers)
    if mode == "would_submit":
        blockers.append("hardware-safe surface never performs live QPU submission")
    if mode == "ticketed_prep":
        if not live_execution_ticket or not str(live_execution_ticket).strip():
            blockers.append("ticketed_prep requires non-empty live_execution_ticket")
        if not policy.owner_allow_submit:
            blockers.append("ticketed_prep requires owner_allow_submit policy")
        # Ticketed prep is still a plan-only decision on this surface.
        if plan.outcome == "allowed_plan" and not blockers:
            # Strip would-submit-specific blockers by re-planning dry-run path.
            pass

    # Recompute allow: dry_run allowed_plan only; ticketed_prep allowed only with
    # owner gate + ticket and within budget; would_submit always refused here.
    unique_blockers = tuple(dict.fromkeys(item for item in blockers if item.strip()))
    if mode == "would_submit":
        unique_blockers = tuple(
            dict.fromkeys(
                (
                    *unique_blockers,
                    "would_submit mode refused: no-submit product surface",
                )
            )
        )
        allowed = False
        outcome: DecisionOutcome = "refused"
        reason = "enforce refused would-submit: " + "; ".join(unique_blockers)
    elif mode == "ticketed_prep":
        allowed = (
            plan.outcome == "allowed_plan"
            and policy.owner_allow_submit
            and bool(live_execution_ticket and str(live_execution_ticket).strip())
            and not unique_blockers
        )
        if allowed:
            outcome = "allowed_plan"
            reason = (
                "ticketed preparation plan allowed under owner gate; "
                "no provider submission occurred on this surface"
            )
            unique_blockers = ()
        else:
            outcome = "refused"
            if not unique_blockers:
                unique_blockers = ("ticketed_prep requirements not met",)
            reason = "enforce refused ticketed_prep: " + "; ".join(unique_blockers)
    else:
        # dry_run
        allowed = plan.outcome == "allowed_plan"
        if allowed:
            outcome = "allowed_plan"
            reason = plan.reason
            unique_blockers = ()
        else:
            outcome = "refused"
            reason = "enforce refused dry_run: " + "; ".join(unique_blockers)

    audit_id = (
        f"hse:{policy.policy_id}:{mode}:p{n_params}:s{plan.shots_per_evaluation}:"
        f"t{plan.estimated_total_shots}:{'ok' if allowed else 'no'}"
    )
    return EnforceDecision(
        policy_id=policy.policy_id,
        mode=mode,
        allowed=allowed,
        outcome=outcome,
        estimated_total_shots=plan.estimated_total_shots,
        reason=reason,
        blockers=unique_blockers,
        audit_id=audit_id,
    )


def build_audit_record(decision: EnforceDecision) -> AuditRecord:
    """Build a secret-free audit record from an enforce decision.

    Parameters
    ----------
    decision
        Enforce decision to audit.

    Returns
    -------
    AuditRecord
        Structured audit row.

    """
    return AuditRecord(
        audit_id=decision.audit_id,
        policy_id=decision.policy_id,
        mode=decision.mode,
        outcome=decision.outcome,
        estimated_total_shots=decision.estimated_total_shots,
        reason=decision.reason,
        claim_boundary=decision.claim_boundary,
    )


def build_hardware_safe_execution_registry() -> dict[str, object]:
    """Build the full serialisable hardware-safe execution registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with every policy (no blanks).

    """
    rows = [row.to_dict() for row in _CANONICAL_POLICIES]
    no_submit_count = sum(1 for row in _CANONICAL_POLICIES if row.no_submit)
    return {
        "schema": HARDWARE_SAFE_EXECUTION_SCHEMA,
        "claim_boundary": HARDWARE_SAFE_EXECUTION_CLAIM_BOUNDARY,
        "policy_count": len(rows),
        "no_submit_policy_count": no_submit_count,
        "default_policy_id": "default_no_submit",
        "blank_entry_count": 0,
        "policies": rows,
        "policy_note": (
            "Default posture is no-submit; dry-run plans estimate shots without "
            "provider I/O; would-submit is refused on this product surface."
        ),
    }


def assert_hardware_safe_execution_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers policies without blanks or invent-submit.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_hardware_safe_execution_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-submit defaults appear.

    """
    registry = dict(payload) if payload is not None else build_hardware_safe_execution_registry()
    policies = registry.get("policies")
    if not isinstance(policies, list) or not policies:
        raise ValueError("hardware-safe execution registry must contain a non-empty policies list")
    seen: set[str] = set()
    blank = 0
    default_found = False
    for index, row in enumerate(policies):
        if not isinstance(row, Mapping):
            raise ValueError(f"policy row {index} must be a mapping")
        policy_id = row.get("policy_id")
        no_submit = row.get("no_submit")
        if not policy_id or not str(policy_id).strip():
            blank += 1
            continue
        pid = str(policy_id).strip()
        if pid in seen:
            raise ValueError(f"duplicate policy_id in registry: {pid!r}")
        seen.add(pid)
        if pid == "default_no_submit":
            default_found = True
            if no_submit is not True:
                raise ValueError("default_no_submit must have no_submit=True")
        if no_submit not in {True, False}:
            blank += 1
            continue
        for field in (
            "max_shots_per_evaluation",
            "max_total_shots",
            "max_params",
            "max_shift_terms",
        ):
            value = row.get(field)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"policy {pid!r} has invalid {field}")
    if blank:
        raise ValueError(f"hardware-safe execution registry has {blank} blank or invalid entries")
    if not default_found:
        raise ValueError("hardware-safe execution registry missing default_no_submit")
    expected = set(list_execution_policy_ids())
    if seen != expected:
        raise ValueError(
            f"registry policy set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    policy_count = registry.get("policy_count", -1)
    if not isinstance(policy_count, int) or policy_count != len(policies):
        raise ValueError("policy_count does not match policies list length")
    return registry


__all__ = [
    "HARDWARE_SAFE_EXECUTION_CLAIM_BOUNDARY",
    "HARDWARE_SAFE_EXECUTION_SCHEMA",
    "AuditRecord",
    "CostModelStatus",
    "DecisionOutcome",
    "DryRunPlan",
    "EnforceDecision",
    "EnforceMode",
    "ExecutionPolicy",
    "assert_hardware_safe_execution_integrity",
    "build_audit_record",
    "build_hardware_safe_execution_registry",
    "default_execution_policy",
    "dry_run_execution_plan",
    "enforce_execution_request",
    "get_execution_policy",
    "iter_execution_policies",
    "list_execution_policy_ids",
]
