# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — qpu_compute product surface
"""Fail-closed qpu_compute plan/runtime product surface.

Productises a typed **compute plan** catalogue between algorithms and HALs:
inventory of plan kinds, construction/validation of dry-run plans, refuse for
would-live / hardware_enabled without owner gate, and audit decisions aligned
with hardware-safety :mod:`hardware_safe_execution` no-submit posture.

Composes kernel/backend vocabulary from :mod:`qpu_compute_types`. Does **not**
submit QPU jobs, invent hardware results, or replace provider SDKs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, cast

from .hardware_safe_execution import (
    HARDWARE_SAFE_EXECUTION_CLAIM_BOUNDARY,
    EnforceMode,
    build_audit_record,
    enforce_execution_request,
)
from .qpu_compute_types import SUPPORTED_BACKEND_POLICIES, SUPPORTED_KERNELS

PlanMode = Literal["dry_run", "would_live", "ticketed_prep"]
"""Plan mode vocabulary for product decisions."""

ValidationOutcome = Literal["allowed_plan", "refused"]
"""Structured validation outcomes."""

QPU_COMPUTE_PRODUCT_SCHEMA: Final[str] = "qpu_compute_product.v2"
"""JSON schema identifier for serialised product payloads."""

QPU_COMPUTE_AUDIT_SCHEMA: Final[str] = "qpu_compute_product_audit.v2"
"""JSON schema identifier for secret-free plan-decision audits."""

QPU_COMPUTE_PRODUCT_CLAIM_BOUNDARY: Final[str] = (
    "qpu_compute product only; default posture is dry-run / no-submit; "
    "would_live and hardware_enabled plans are refused without owner gate; "
    "composes qpu_compute_types kernels and hardware-safe audit posture; "
    "never executes QPU jobs or invents hardware results"
)
"""Shared claim boundary for plan kinds, validations, and audits."""


@dataclass(frozen=True, slots=True)
class ComputePlanKind:
    """One versioned compute-plan kind in the product catalogue.

    Attributes
    ----------
    plan_kind_id
        Stable catalogue identifier.
    mode
        Default mode for this kind.
    summary
        Short description.
    default_backend_policy
        Backend policy label (must be supported for dry-run kinds).
    default_hardware_enabled
        Whether hardware is enabled by default (product default is false).
    no_submit
        When true, would-live paths are refused on this surface.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    plan_kind_id: str
    mode: PlanMode
    summary: str
    default_backend_policy: str
    default_hardware_enabled: bool
    no_submit: bool
    as_of: str = "2026-07-24"
    claim_boundary: str = QPU_COMPUTE_PRODUCT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate plan-kind invariants."""
        if not self.plan_kind_id or not self.plan_kind_id.strip():
            raise ValueError("plan_kind_id must be non-empty")
        if self.mode not in {"dry_run", "would_live", "ticketed_prep"}:
            raise ValueError(f"unknown mode: {self.mode!r}")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.default_backend_policy or not self.default_backend_policy.strip():
            raise ValueError("default_backend_policy must be non-empty")
        if self.mode == "dry_run" and self.default_hardware_enabled:
            raise ValueError("dry_run plan kinds must set default_hardware_enabled=False")
        if self.mode == "dry_run" and not self.no_submit:
            raise ValueError("dry_run plan kinds must set no_submit=True")
        if self.mode == "would_live" and not self.no_submit:
            # Product surface still refuses live submit; catalogue may describe intent.
            raise ValueError(
                "product catalogue would_live kinds retain no_submit=True (surface never submits)"
            )
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this plan kind."""
        return {
            "plan_kind_id": self.plan_kind_id,
            "mode": self.mode,
            "summary": self.summary,
            "default_backend_policy": self.default_backend_policy,
            "default_hardware_enabled": self.default_hardware_enabled,
            "no_submit": self.no_submit,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class ComputePlanRecord:
    """A constructed compute plan under product validation.

    Attributes
    ----------
    plan_kind_id
        Catalogue kind used.
    mode
        Effective mode.
    kernel
        Kernel identifier (from SUPPORTED_KERNELS).
    backend_policy
        Backend policy.
    shots
        Shot count.
    hardware_enabled
        Whether hardware spend is requested.
    live_execution_ticket
        Optional ticket label (never a secret).
    no_submit
        Effective no-submit posture.

    """

    plan_kind_id: str
    mode: PlanMode
    kernel: str
    backend_policy: str
    shots: int
    hardware_enabled: bool
    live_execution_ticket: str
    no_submit: bool
    claim_boundary: str = QPU_COMPUTE_PRODUCT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate constructed plan invariants."""
        if not self.plan_kind_id or not self.plan_kind_id.strip():
            raise ValueError("plan_kind_id must be non-empty")
        if self.mode not in {"dry_run", "would_live", "ticketed_prep"}:
            raise ValueError(f"unknown mode: {self.mode!r}")
        if self.kernel not in SUPPORTED_KERNELS:
            raise ValueError(
                f"kernel must be one of {sorted(SUPPORTED_KERNELS)}; got {self.kernel!r}"
            )
        if not self.backend_policy or not self.backend_policy.strip():
            raise ValueError("backend_policy must be non-empty")
        if self.shots < 1:
            raise ValueError("shots must be >= 1")
        if self.live_execution_ticket != self.live_execution_ticket.strip():
            raise ValueError("live_execution_ticket must not have surrounding whitespace")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this plan."""
        return {
            "plan_kind_id": self.plan_kind_id,
            "mode": self.mode,
            "kernel": self.kernel,
            "backend_policy": self.backend_policy,
            "shots": self.shots,
            "hardware_enabled": self.hardware_enabled,
            "live_execution_ticket": self.live_execution_ticket,
            "no_submit": self.no_submit,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class ComputePlanDecision:
    """Fail-closed validation / dry-run decision for a compute plan.

    Attributes
    ----------
    plan_kind_id
        Kind validated.
    outcome
        Allowed plan or refused.
    allowed
        Whether dry-run planning may proceed (never means live QPU ran).
    mode
        Effective mode.
    reason
        Human-readable reason.
    blockers
        Non-empty when refused.
    audit_id
        Deterministic audit identifier.
    hardware_safety_policy_id
        Hardware-safety policy composed into the decision, if any.

    """

    plan_kind_id: str
    outcome: ValidationOutcome
    allowed: bool
    mode: PlanMode
    reason: str
    blockers: tuple[str, ...]
    audit_id: str
    hardware_safety_policy_id: str = ""
    claim_boundary: str = QPU_COMPUTE_PRODUCT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate decision invariants."""
        if not self.plan_kind_id or not self.plan_kind_id.strip():
            raise ValueError("plan_kind_id must be non-empty")
        if self.outcome not in {"allowed_plan", "refused"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if self.mode not in {"dry_run", "would_live", "ticketed_prep"}:
            raise ValueError(f"unknown mode: {self.mode!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if not self.audit_id or not self.audit_id.strip():
            raise ValueError("audit_id must be non-empty")
        if self.allowed and self.outcome != "allowed_plan":
            raise ValueError("allowed decisions must use outcome=allowed_plan")
        if not self.allowed and self.outcome != "refused":
            raise ValueError("refused decisions must use outcome=refused")
        if self.allowed and self.blockers:
            raise ValueError("allowed decisions cannot list blockers")
        if not self.allowed and not self.blockers:
            raise ValueError("refused decisions require blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this decision."""
        return {
            "plan_kind_id": self.plan_kind_id,
            "outcome": self.outcome,
            "allowed": self.allowed,
            "mode": self.mode,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "audit_id": self.audit_id,
            "hardware_safety_policy_id": self.hardware_safety_policy_id,
            "claim_boundary": self.claim_boundary,
        }


def _kind(
    plan_kind_id: str,
    *,
    mode: PlanMode,
    summary: str,
    default_backend_policy: str,
    default_hardware_enabled: bool,
    no_submit: bool,
) -> ComputePlanKind:
    """Build one catalogue plan kind."""
    return ComputePlanKind(
        plan_kind_id=plan_kind_id,
        mode=mode,
        summary=summary,
        default_backend_policy=default_backend_policy,
        default_hardware_enabled=default_hardware_enabled,
        no_submit=no_submit,
    )


_CANONICAL_KINDS: Final[tuple[ComputePlanKind, ...]] = (
    _kind(
        "dry_run_simulator",
        mode="dry_run",
        summary=(
            "Default product plan: local simulator_statevector dry-run; "
            "hardware_enabled=False; hardware-safe no-submit posture."
        ),
        default_backend_policy="simulator_statevector",
        default_hardware_enabled=False,
        no_submit=True,
    ),
    _kind(
        "live_would_submit",
        mode="would_live",
        summary=(
            "Catalogue row describing a would-live intent; product surface "
            "always refuses actual submit (no_submit remains true)."
        ),
        default_backend_policy="simulator_statevector",
        default_hardware_enabled=True,
        no_submit=True,
    ),
    _kind(
        "ticketed_prep_plan",
        mode="ticketed_prep",
        summary=(
            "Owner-gated ticketed preparation plan; still no live submit on "
            "this surface; requires non-empty live_execution_ticket."
        ),
        default_backend_policy="simulator_statevector",
        default_hardware_enabled=False,
        no_submit=True,
    ),
)


def _catalogue_map() -> dict[str, ComputePlanKind]:
    """Return plan_kind_id → kind map; refuse blanks/duplicates."""
    mapping: dict[str, ComputePlanKind] = {}
    for row in _CANONICAL_KINDS:
        key = row.plan_kind_id.strip()
        if not key:
            raise RuntimeError("qpu_compute product catalogue contains blank plan_kind_id")
        if key in mapping:
            raise RuntimeError(f"duplicate plan_kind_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("qpu_compute product catalogue must be non-empty")
    return mapping


_KIND_BY_ID: Final[Mapping[str, ComputePlanKind]] = _catalogue_map()


def list_plan_kind_ids() -> tuple[str, ...]:
    """Return all plan kind identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered plan kind identifiers.

    """
    return tuple(row.plan_kind_id for row in _CANONICAL_KINDS)


def get_plan_kind(plan_kind_id: str) -> ComputePlanKind:
    """Return one plan kind or raise for blank/unknown identifiers.

    Parameters
    ----------
    plan_kind_id
        Catalogue plan kind key.

    Returns
    -------
    ComputePlanKind
        Matching kind.

    Raises
    ------
    ValueError
        If ``plan_kind_id`` is blank or unknown (fail closed).

    """
    if not plan_kind_id or not str(plan_kind_id).strip():
        raise ValueError("plan_kind_id must be a non-empty string")
    key = str(plan_kind_id).strip()
    try:
        return _KIND_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown plan_kind_id {key!r}; refuse invent-green compute plan "
            f"(known_count={len(_KIND_BY_ID)})"
        ) from exc


def iter_plan_kinds(
    *,
    mode: PlanMode | None = None,
) -> tuple[ComputePlanKind, ...]:
    """Return filtered plan kinds in stable order.

    Parameters
    ----------
    mode
        Optional mode filter.

    Returns
    -------
    tuple[ComputePlanKind, ...]
        Matching kinds.

    """
    rows: Iterable[ComputePlanKind] = _CANONICAL_KINDS
    if mode is not None:
        rows = (row for row in rows if row.mode == mode)
    return tuple(rows)


def list_supported_kernels() -> tuple[str, ...]:
    """Return supported kernel identifiers from qpu_compute_types.

    Returns
    -------
    tuple[str, ...]
        Sorted kernel identifiers.

    """
    return tuple(sorted(SUPPORTED_KERNELS))


def list_supported_backend_policies() -> tuple[str, ...]:
    """Return supported backend policies from qpu_compute_types.

    Returns
    -------
    tuple[str, ...]
        Sorted backend policy identifiers.

    """
    return tuple(sorted(SUPPORTED_BACKEND_POLICIES))


def construct_compute_plan(
    plan_kind_id: str,
    *,
    kernel: str = "sync_dla",
    backend_policy: str | None = None,
    shots: int = 1024,
    hardware_enabled: bool | None = None,
    live_execution_ticket: str = "",
) -> ComputePlanRecord:
    """Construct a compute plan record from a catalogue kind.

    Parameters
    ----------
    plan_kind_id
        Catalogue kind key.
    kernel
        Kernel (must be in SUPPORTED_KERNELS).
    backend_policy
        Backend policy (defaults to kind default).
    shots
        Shot count.
    hardware_enabled
        Override kind default hardware flag.
    live_execution_ticket
        Optional ticket label.

    Returns
    -------
    ComputePlanRecord
        Constructed plan (not yet validated for execution).

    Raises
    ------
    ValueError
        If identifiers / dimensions are invalid.

    """
    kind = get_plan_kind(plan_kind_id)
    policy = kind.default_backend_policy if backend_policy is None else str(backend_policy).strip()
    hw = kind.default_hardware_enabled if hardware_enabled is None else bool(hardware_enabled)
    ticket = str(live_execution_ticket).strip()
    return ComputePlanRecord(
        plan_kind_id=kind.plan_kind_id,
        mode=kind.mode,
        kernel=str(kernel).strip(),
        backend_policy=policy,
        shots=int(shots),
        hardware_enabled=hw,
        live_execution_ticket=ticket,
        no_submit=kind.no_submit,
    )


def dry_run_compute_plan(
    plan_kind_id: str,
    *,
    kernel: str = "sync_dla",
    backend_policy: str | None = None,
    shots: int = 1024,
    hardware_enabled: bool | None = None,
    live_execution_ticket: str = "",
) -> ComputePlanDecision:
    """Validate a dry-run compute plan without provider submission.

    Default posture allows ``dry_run_simulator`` with ``simulator_statevector``
    and ``hardware_enabled=False``. Would-live / hardware_enabled requests are
    refused. Ticketed prep requires a non-empty ticket and is plan-only.

    Parameters
    ----------
    plan_kind_id
        Catalogue kind key.
    kernel
        Kernel identifier.
    backend_policy
        Backend policy override.
    shots
        Shot count.
    hardware_enabled
        Hardware flag override.
    live_execution_ticket
        Ticket label for ticketed_prep.

    Returns
    -------
    ComputePlanDecision
        Allowed plan or refused decision (never means live QPU ran).

    Raises
    ------
    ValueError
        If identifiers / dimensions are invalid.

    """
    plan = construct_compute_plan(
        plan_kind_id,
        kernel=kernel,
        backend_policy=backend_policy,
        shots=shots,
        hardware_enabled=hardware_enabled,
        live_execution_ticket=live_execution_ticket,
    )
    blockers: list[str] = []

    if plan.kernel not in SUPPORTED_KERNELS:
        blockers.append(f"unsupported kernel {plan.kernel!r}")
    if plan.backend_policy not in SUPPORTED_BACKEND_POLICIES:
        blockers.append(
            f"unsupported backend_policy {plan.backend_policy!r}; "
            f"product dry-run supports {sorted(SUPPORTED_BACKEND_POLICIES)}"
        )
    if plan.mode == "would_live" or plan.hardware_enabled:
        blockers.append(
            "would_live/hardware_enabled refused on product surface (no-submit default)"
        )
    if plan.mode == "ticketed_prep" and not plan.live_execution_ticket:
        blockers.append("ticketed_prep requires non-empty live_execution_ticket")
    if plan.no_submit and plan.mode == "would_live":
        blockers.append("plan kind no_submit=True refuses would_live execution")

    # Compose hardware-safety enforcement for dry-run / would-submit mapping.
    hardware_safety_mode_raw: str
    if plan.mode == "ticketed_prep":
        hardware_safety_mode_raw = "ticketed_prep"
    elif plan.mode == "would_live" or plan.hardware_enabled:
        hardware_safety_mode_raw = "would_submit"
    else:
        hardware_safety_mode_raw = "dry_run"
    hardware_safety_mode = cast(EnforceMode, hardware_safety_mode_raw)
    hardware_safety_policy = (
        "owner_ticketed_prep" if plan.mode == "ticketed_prep" else "default_no_submit"
    )
    hardware_safety_audit = enforce_execution_request(
        hardware_safety_policy,
        mode=hardware_safety_mode,
        n_params=1,
        shots_per_evaluation=min(plan.shots, 1024),
        live_execution_ticket=plan.live_execution_ticket,
    )
    if not hardware_safety_audit.allowed:
        for item in hardware_safety_audit.blockers:
            if item not in blockers:
                blockers.append(f"hardware_safety_audit:{item}")

    unique = tuple(dict.fromkeys(item for item in blockers if item.strip()))
    audit_id = (
        f"qcp:{plan.plan_kind_id}:{plan.mode}:{plan.kernel}:"
        f"s{plan.shots}:{'ok' if not unique else 'no'}"
    )
    if unique:
        return ComputePlanDecision(
            plan_kind_id=plan.plan_kind_id,
            outcome="refused",
            allowed=False,
            mode=plan.mode,
            reason="qpu_compute product refuse: " + "; ".join(unique),
            blockers=unique,
            audit_id=audit_id,
            hardware_safety_policy_id=hardware_safety_policy,
        )
    return ComputePlanDecision(
        plan_kind_id=plan.plan_kind_id,
        outcome="allowed_plan",
        allowed=True,
        mode=plan.mode,
        reason=(
            "dry-run compute plan allowed under product surface; "
            "no provider submission occurred "
            f"(backend_policy={plan.backend_policy}, kernel={plan.kernel})"
        ),
        blockers=(),
        audit_id=audit_id,
        hardware_safety_policy_id=hardware_safety_policy,
    )


def audit_compute_plan_decision(decision: ComputePlanDecision) -> dict[str, object]:
    """Build a secret-free audit payload for a compute plan decision.

    Includes the hardware-safety audit when a policy was composed.

    Parameters
    ----------
    decision
        Decision from :func:`dry_run_compute_plan`.

    Returns
    -------
    dict[str, object]
        Secret-free audit mapping.

    """
    payload: dict[str, object] = {
        "schema": QPU_COMPUTE_AUDIT_SCHEMA,
        "audit_id": decision.audit_id,
        "plan_kind_id": decision.plan_kind_id,
        "outcome": decision.outcome,
        "allowed": decision.allowed,
        "mode": decision.mode,
        "reason": decision.reason,
        "blockers": list(decision.blockers),
        "hardware_safety_policy_id": decision.hardware_safety_policy_id,
        "claim_boundary": decision.claim_boundary,
        "hardware_safety_claim_boundary": HARDWARE_SAFE_EXECUTION_CLAIM_BOUNDARY,
        "contains_secrets": False,
    }
    # Add the structured hardware-safety audit when a policy is present.
    if decision.hardware_safety_policy_id:
        if decision.mode == "would_live":
            hardware_safety_mode_raw = "would_submit"
        elif decision.mode == "ticketed_prep":
            hardware_safety_mode_raw = "ticketed_prep"
        else:
            hardware_safety_mode_raw = "dry_run"
        hardware_safety_decision = enforce_execution_request(
            decision.hardware_safety_policy_id,
            mode=cast(EnforceMode, hardware_safety_mode_raw),
            n_params=1,
            shots_per_evaluation=64,
            live_execution_ticket=(
                "ticket-audit-placeholder" if decision.mode == "ticketed_prep" else ""
            ),
        )
        audit = build_audit_record(hardware_safety_decision)
        payload["hardware_safety_audit"] = audit.to_dict()
    return payload


def build_qpu_compute_product_registry() -> dict[str, object]:
    """Build the full serialisable qpu_compute product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with every plan kind (no blanks).

    """
    rows = [row.to_dict() for row in _CANONICAL_KINDS]
    dry_run_count = sum(1 for row in _CANONICAL_KINDS if row.mode == "dry_run")
    no_submit_count = sum(1 for row in _CANONICAL_KINDS if row.no_submit)
    return {
        "schema": QPU_COMPUTE_PRODUCT_SCHEMA,
        "claim_boundary": QPU_COMPUTE_PRODUCT_CLAIM_BOUNDARY,
        "plan_kind_count": len(rows),
        "dry_run_kind_count": dry_run_count,
        "no_submit_kind_count": no_submit_count,
        "default_plan_kind_id": "dry_run_simulator",
        "supported_kernels": list(list_supported_kernels()),
        "supported_backend_policies": list(list_supported_backend_policies()),
        "blank_entry_count": 0,
        "plan_kinds": rows,
        "policy_note": (
            "Composes qpu_compute_types kernels/backend policies and a "
            "hardware-safe no-submit posture; full runtime audit wiring and "
            "mass algorithm migration remain outside this bounded registry."
        ),
    }


def assert_qpu_compute_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers plan kinds without blanks or invent-live.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_qpu_compute_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-live defaults appear.

    """
    registry = dict(payload) if payload is not None else build_qpu_compute_product_registry()
    if registry.get("schema") != QPU_COMPUTE_PRODUCT_SCHEMA:
        raise ValueError(f"registry schema must be {QPU_COMPUTE_PRODUCT_SCHEMA}")
    kinds = registry.get("plan_kinds")
    if not isinstance(kinds, list) or not kinds:
        raise ValueError("qpu_compute product registry must contain a non-empty plan_kinds list")
    seen: set[str] = set()
    blank = 0
    default_found = False
    for index, row in enumerate(kinds):
        if not isinstance(row, Mapping):
            raise ValueError(f"plan kind row {index} must be a mapping")
        plan_kind_id = row.get("plan_kind_id")
        no_submit = row.get("no_submit")
        mode = row.get("mode")
        if not plan_kind_id or not str(plan_kind_id).strip():
            blank += 1
            continue
        pid = str(plan_kind_id).strip()
        if pid in seen:
            raise ValueError(f"duplicate plan_kind_id in registry: {pid!r}")
        seen.add(pid)
        if pid == "dry_run_simulator":
            default_found = True
            if no_submit is not True:
                raise ValueError("dry_run_simulator must have no_submit=True")
            if row.get("default_hardware_enabled") is not False:
                raise ValueError("dry_run_simulator must have default_hardware_enabled=False")
        if mode not in {"dry_run", "would_live", "ticketed_prep"}:
            blank += 1
            continue
        if no_submit not in {True, False}:
            blank += 1
            continue
        if no_submit is not True:
            raise ValueError(
                f"plan kind {pid!r} invent-live: product kinds must set no_submit=True"
            )
    if blank:
        raise ValueError(f"qpu_compute product registry has {blank} blank or invalid entries")
    if not default_found:
        raise ValueError("qpu_compute product registry missing dry_run_simulator")
    expected = set(list_plan_kind_ids())
    if seen != expected:
        raise ValueError(
            f"registry plan kind set drift (missing={expected - seen!r}, "
            f"extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    plan_kind_count = registry.get("plan_kind_count", -1)
    if not isinstance(plan_kind_count, int) or plan_kind_count != len(kinds):
        raise ValueError("plan_kind_count does not match plan_kinds list length")
    return registry


__all__ = [
    "QPU_COMPUTE_AUDIT_SCHEMA",
    "QPU_COMPUTE_PRODUCT_CLAIM_BOUNDARY",
    "QPU_COMPUTE_PRODUCT_SCHEMA",
    "ComputePlanDecision",
    "ComputePlanKind",
    "ComputePlanRecord",
    "PlanMode",
    "ValidationOutcome",
    "assert_qpu_compute_product_integrity",
    "audit_compute_plan_decision",
    "build_qpu_compute_product_registry",
    "construct_compute_plan",
    "dry_run_compute_plan",
    "get_plan_kind",
    "iter_plan_kinds",
    "list_plan_kind_ids",
    "list_supported_backend_policies",
    "list_supported_kernels",
]
