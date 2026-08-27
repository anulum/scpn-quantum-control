# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode product surface
"""Fail-closed Phase-QNode **product** catalogue and journey map.

Productises Phase-QNode as a versioned primary quantum programming surface:
public capability/journey inventory, dry-run journey posture, fail-closed
unknown/blank ids, and refuse invent-green hardware / live QPU claims.

Composes honesty from API-stability (workbench not silently SemVer-stable), route-matrix
(route-matrix pointers), and hardware-safety/QPU-compute (no-submit dry-run posture). Does
**not** freeze the entire ``phase/qnode_*`` workbench as a mega-contract and
does not execute provider or QPU jobs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

SupportBadge = Literal[
    "local_dry_run",
    "framework_bridge",
    "provider_boundary",
    "experimental_workbench",
]
"""Support badge vocabulary for product journeys (not invent-green hardware)."""

JourneyOutcome = Literal["allowed_dry_run", "refused"]
"""Structured dry-run journey outcomes."""

PHASE_QNODE_PRODUCT_SCHEMA: Final[str] = "phase_qnode_product.v2"
"""JSON schema identifier for serialised product payloads."""

PHASE_QNODE_PRODUCT_CLAIM_BOUNDARY: Final[str] = (
    "Phase-QNode product surface only; catalogues public journeys and support "
    "badges; ambient phase/qnode_* workbench remains experimental rather than a "
    "frozen SemVer mega-contract; dry-run journeys refuse hardware/QPU spend under "
    "the no-submit policy; does not replace full circuit engines"
)
"""Shared claim boundary for journeys and decisions."""


@dataclass(frozen=True, slots=True)
class PhaseQNodeJourney:
    """One canonical Phase-QNode product journey.

    Attributes
    ----------
    journey_id
        Stable catalogue identifier.
    title
        Human-readable journey title.
    summary
        Short description of the user path.
    module_path
        Primary owning module path (documentation pointer).
    support_badge
        Support posture badge.
    steps
        Ordered journey step labels (build → differentiate → dry-run, …).
    allows_hardware
        Whether this journey claims hardware (product default false).
    route_matrix_pointer
        Optional governed route-family pointer.
    api_stability_class
        Stability honesty class (not invent-stable for workbench).
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    journey_id: str
    title: str
    summary: str
    module_path: str
    support_badge: SupportBadge
    steps: tuple[str, ...]
    allows_hardware: bool = False
    route_matrix_pointer: str = "phase_qnode.local_statevector"
    api_stability_class: str = "experimental_workbench"
    as_of: str = "2026-07-24"
    claim_boundary: str = PHASE_QNODE_PRODUCT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate journey invariants."""
        if not self.journey_id or not self.journey_id.strip():
            raise ValueError("journey_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.module_path or not self.module_path.strip():
            raise ValueError("module_path must be non-empty")
        if self.support_badge not in {
            "local_dry_run",
            "framework_bridge",
            "provider_boundary",
            "experimental_workbench",
        }:
            raise ValueError(f"unknown support_badge: {self.support_badge!r}")
        if not self.steps:
            raise ValueError("steps must be non-empty")
        if any(not step or not str(step).strip() for step in self.steps):
            raise ValueError("steps entries must be non-empty")
        if self.allows_hardware and self.support_badge == "local_dry_run":
            raise ValueError("local_dry_run journeys must set allows_hardware=False")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")
        if not self.api_stability_class or not self.api_stability_class.strip():
            raise ValueError("api_stability_class must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this journey."""
        return {
            "journey_id": self.journey_id,
            "title": self.title,
            "summary": self.summary,
            "module_path": self.module_path,
            "support_badge": self.support_badge,
            "steps": list(self.steps),
            "allows_hardware": self.allows_hardware,
            "route_matrix_pointer": self.route_matrix_pointer,
            "api_stability_class": self.api_stability_class,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PhaseQNodeJourneyDecision:
    """Fail-closed dry-run decision for a product journey.

    Attributes
    ----------
    journey_id
        Journey validated.
    outcome
        Allowed dry-run or refused.
    allowed
        Whether the dry-run journey may proceed (never means QPU ran).
    support_badge
        Badge of the journey.
    reason
        Human-readable reason.
    blockers
        Non-empty when refused.
    steps_completed
        Steps acknowledged in the dry-run posture (not full engine execution).

    """

    journey_id: str
    outcome: JourneyOutcome
    allowed: bool
    support_badge: SupportBadge
    reason: str
    blockers: tuple[str, ...]
    steps_completed: tuple[str, ...]
    claim_boundary: str = PHASE_QNODE_PRODUCT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate decision invariants."""
        if not self.journey_id or not self.journey_id.strip():
            raise ValueError("journey_id must be non-empty")
        if self.outcome not in {"allowed_dry_run", "refused"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed and self.outcome != "allowed_dry_run":
            raise ValueError("allowed decisions must use outcome=allowed_dry_run")
        if not self.allowed and self.outcome != "refused":
            raise ValueError("refused decisions must use outcome=refused")
        if self.allowed and self.blockers:
            raise ValueError("allowed decisions cannot list blockers")
        if not self.allowed and not self.blockers:
            raise ValueError("refused decisions require blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")
        if self.support_badge not in {
            "local_dry_run",
            "framework_bridge",
            "provider_boundary",
            "experimental_workbench",
        }:
            raise ValueError(f"unknown support_badge: {self.support_badge!r}")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this decision."""
        return {
            "journey_id": self.journey_id,
            "outcome": self.outcome,
            "allowed": self.allowed,
            "support_badge": self.support_badge,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "steps_completed": list(self.steps_completed),
            "claim_boundary": self.claim_boundary,
        }


def _journey(
    journey_id: str,
    *,
    title: str,
    summary: str,
    module_path: str,
    support_badge: SupportBadge,
    steps: tuple[str, ...],
    allows_hardware: bool = False,
    route_matrix_pointer: str = "phase_qnode.local_statevector",
    api_stability_class: str = "experimental_workbench",
) -> PhaseQNodeJourney:
    """Build one catalogue journey."""
    return PhaseQNodeJourney(
        journey_id=journey_id,
        title=title,
        summary=summary,
        module_path=module_path,
        support_badge=support_badge,
        steps=steps,
        allows_hardware=allows_hardware,
        route_matrix_pointer=route_matrix_pointer,
        api_stability_class=api_stability_class,
    )


_CANONICAL_JOURNEYS: Final[tuple[PhaseQNodeJourney, ...]] = (
    _journey(
        "build_differentiate_dry_run",
        title="Build → differentiate → dry-run execute",
        summary=(
            "Canonical local journey: construct a registered Phase-QNode circuit, "
            "obtain parameter-shift / local transform gradients, execute dry-run "
            "statevector posture without hardware submission."
        ),
        module_path="scpn_quantum_control.phase.qnode_circuit",
        support_badge="local_dry_run",
        steps=(
            "build_registered_circuit",
            "plan_parameter_shift",
            "differentiate_local",
            "execute_statevector_dry_run",
        ),
        route_matrix_pointer="phase_qnode.local_statevector.parameter_shift",
    ),
    _journey(
        "tape_finite_shot_dry_run",
        title="QNode tape finite-shot dry-run replay",
        summary=(
            "Seeded finite-shot tape replay with shifted-sample provenance; "
            "provider-boundary routes fail closed before hardware submission."
        ),
        module_path="scpn_quantum_control.phase.qnode_tape",
        support_badge="local_dry_run",
        steps=(
            "build_tape_record",
            "seeded_finite_shot_replay",
            "provider_boundary_refuse",
        ),
        route_matrix_pointer="phase_qnode.tape.finite_shot",
    ),
    _journey(
        "local_transform_suite",
        title="Local scalar QNode transforms",
        summary=(
            "Executable local transform evidence for grad, value_and_grad, "
            "hessian, jvp, vjp, jacfwd/jacrev with real-only complex boundaries."
        ),
        module_path="scpn_quantum_control.phase.qnode_transforms",
        support_badge="local_dry_run",
        steps=(
            "select_transform",
            "run_local_transform_evidence",
            "fail_closed_vectorized_provider",
        ),
        route_matrix_pointer="phase_qnode.transforms.local_scalar",
    ),
    _journey(
        "framework_bridge_parity",
        title="Framework bridge parity (JAX/Torch/TF/PL)",
        summary=(
            "Bounded real-framework parity suite with dependency-sparse "
            "classifications; not invent-green hardware parity."
        ),
        module_path="scpn_quantum_control.phase.qnode_framework_parity",
        support_badge="framework_bridge",
        steps=(
            "select_framework_bridge",
            "run_parity_suite_when_installed",
            "report_blocked_missing_deps",
        ),
        route_matrix_pointer="phase_qnode.framework_bridge.parity",
        api_stability_class="experimental_workbench",
    ),
    _journey(
        "provider_transform_boundary",
        title="Provider-callback transform boundary",
        summary=(
            "Provider-callback QNode transform evidence with fail-closed "
            "hardware policy; refuses invent-green QPU submission."
        ),
        module_path="scpn_quantum_control.phase.qnode_provider_transforms",
        support_badge="provider_boundary",
        steps=(
            "provider_callback_transform",
            "finite_shot_uncertainty",
            "hardware_policy_refuse",
        ),
        allows_hardware=False,
        route_matrix_pointer="phase_qnode.provider_transforms.boundary",
        api_stability_class="experimental_workbench",
    ),
)


def _catalogue_map() -> dict[str, PhaseQNodeJourney]:
    """Return journey_id → journey map; refuse blanks/duplicates."""
    mapping: dict[str, PhaseQNodeJourney] = {}
    for row in _CANONICAL_JOURNEYS:
        key = row.journey_id.strip()
        if not key:
            raise RuntimeError("Phase-QNode product catalogue contains blank journey_id")
        if key in mapping:
            raise RuntimeError(f"duplicate journey_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("Phase-QNode product catalogue must be non-empty")
    return mapping


_JOURNEY_BY_ID: Final[Mapping[str, PhaseQNodeJourney]] = _catalogue_map()


def list_phase_qnode_journey_ids() -> tuple[str, ...]:
    """Return all product journey identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered journey identifiers.

    """
    return tuple(row.journey_id for row in _CANONICAL_JOURNEYS)


def get_phase_qnode_journey(journey_id: str) -> PhaseQNodeJourney:
    """Return one journey or raise for blank/unknown identifiers.

    Parameters
    ----------
    journey_id
        Catalogue journey key.

    Returns
    -------
    PhaseQNodeJourney
        Matching journey.

    Raises
    ------
    ValueError
        If ``journey_id`` is blank or unknown (fail closed).

    """
    if not journey_id or not str(journey_id).strip():
        raise ValueError("journey_id must be a non-empty string")
    key = str(journey_id).strip()
    try:
        return _JOURNEY_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown journey_id {key!r}; refuse invent-green Phase-QNode product "
            f"claim (known_count={len(_JOURNEY_BY_ID)})"
        ) from exc


def iter_phase_qnode_journeys(
    *,
    support_badge: SupportBadge | None = None,
) -> tuple[PhaseQNodeJourney, ...]:
    """Return filtered journeys in stable order.

    Parameters
    ----------
    support_badge
        Optional badge filter.

    Returns
    -------
    tuple[PhaseQNodeJourney, ...]
        Matching journeys.

    """
    rows: Iterable[PhaseQNodeJourney] = _CANONICAL_JOURNEYS
    if support_badge is not None:
        rows = (row for row in rows if row.support_badge == support_badge)
    return tuple(rows)


def dry_run_phase_qnode_journey(
    journey_id: str,
    *,
    request_hardware: bool = False,
) -> PhaseQNodeJourneyDecision:
    """Exercise a product journey in dry-run posture (no QPU execution).

    Acknowledges the catalogue journey steps as a structured dry-run plan.
    Requests that ask for hardware are refused. Provider-boundary journeys
    always record a hardware refuse step without invent-green success.

    Parameters
    ----------
    journey_id
        Catalogue journey key.
    request_hardware
        When true, refuse (no invent-green hardware).

    Returns
    -------
    PhaseQNodeJourneyDecision
        Allowed dry-run or refused decision.

    Raises
    ------
    ValueError
        If ``journey_id`` is blank or unknown.

    """
    journey = get_phase_qnode_journey(journey_id)
    blockers: list[str] = []
    if request_hardware or journey.allows_hardware:
        blockers.append(
            "hardware/QPU request refused on Phase-QNode product surface "
            "(composed hardware-safe no-submit posture)"
        )
    if journey.support_badge == "provider_boundary" and request_hardware:
        blockers.append("provider_boundary journeys fail closed before hardware submission")

    if blockers:
        unique = tuple(dict.fromkeys(item for item in blockers if item.strip()))
        return PhaseQNodeJourneyDecision(
            journey_id=journey.journey_id,
            outcome="refused",
            allowed=False,
            support_badge=journey.support_badge,
            reason="Phase-QNode product refuse: " + "; ".join(unique),
            blockers=unique,
            steps_completed=(),
        )

    # Dry-run: acknowledge steps without claiming full engine execution completed.
    steps = journey.steps
    if journey.support_badge == "provider_boundary":
        # Always include explicit refuse-hardware acknowledgment on provider path.
        steps = (*journey.steps,)
    return PhaseQNodeJourneyDecision(
        journey_id=journey.journey_id,
        outcome="allowed_dry_run",
        allowed=True,
        support_badge=journey.support_badge,
        reason=(
            f"dry-run journey {journey.journey_id!r} allowed under product surface; "
            f"module={journey.module_path}; badge={journey.support_badge}; "
            f"stability={journey.api_stability_class}; "
            "no QPU submission occurred"
        ),
        blockers=(),
        steps_completed=steps,
    )


def map_phase_qnode_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of Phase-QNode product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic module map rows for documentation and inventory.

    """
    # Deduplicate module paths preserving journey order.
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for journey in _CANONICAL_JOURNEYS:
        path = journey.module_path
        if path in seen:
            continue
        seen.add(path)
        rows.append(
            {
                "module_path": path,
                "role": "phase_qnode_product_surface",
                "api_stability_class": journey.api_stability_class,
                "support_badge": journey.support_badge,
                "journey_ids": [
                    j.journey_id for j in _CANONICAL_JOURNEYS if j.module_path == path
                ],
                "claim_boundary": PHASE_QNODE_PRODUCT_CLAIM_BOUNDARY,
            }
        )
    return tuple(rows)


def build_phase_qnode_product_registry() -> dict[str, object]:
    """Build the full serialisable Phase-QNode product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with every journey (no blanks).

    """
    journeys = [row.to_dict() for row in _CANONICAL_JOURNEYS]
    local = sum(1 for row in _CANONICAL_JOURNEYS if row.support_badge == "local_dry_run")
    return {
        "schema": PHASE_QNODE_PRODUCT_SCHEMA,
        "claim_boundary": PHASE_QNODE_PRODUCT_CLAIM_BOUNDARY,
        "journey_count": len(journeys),
        "local_dry_run_count": local,
        "default_journey_id": "build_differentiate_dry_run",
        "blank_entry_count": 0,
        "public_surfaces": list(map_phase_qnode_public_surfaces()),
        "journeys": journeys,
        "policy_note": (
            "Phase-QNode product catalogue only; ambient phase/qnode_* exports "
            "remain experimental_workbench unless explicitly promoted through "
            "public API stability; the complete tutorial and badge automation "
            "remain residual."
        ),
    }


def assert_phase_qnode_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers journeys without blanks or invent-hardware.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_phase_qnode_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-hardware rows appear.

    """
    registry = dict(payload) if payload is not None else build_phase_qnode_product_registry()
    if registry.get("schema") != PHASE_QNODE_PRODUCT_SCHEMA:
        raise ValueError(f"registry schema must be {PHASE_QNODE_PRODUCT_SCHEMA}")
    journeys = registry.get("journeys")
    if not isinstance(journeys, list) or not journeys:
        raise ValueError("Phase-QNode product registry must contain a non-empty journeys list")
    seen: set[str] = set()
    blank = 0
    default_found = False
    for index, row in enumerate(journeys):
        if not isinstance(row, Mapping):
            raise ValueError(f"journey row {index} must be a mapping")
        journey_id = row.get("journey_id")
        allows_hardware = row.get("allows_hardware")
        support_badge = row.get("support_badge")
        steps = row.get("steps")
        if not journey_id or not str(journey_id).strip():
            blank += 1
            continue
        jid = str(journey_id).strip()
        if jid in seen:
            raise ValueError(f"duplicate journey_id in registry: {jid!r}")
        seen.add(jid)
        if jid == "build_differentiate_dry_run":
            default_found = True
            if allows_hardware is not False:
                raise ValueError("build_differentiate_dry_run must set allows_hardware=False")
        if support_badge not in {
            "local_dry_run",
            "framework_bridge",
            "provider_boundary",
            "experimental_workbench",
        }:
            blank += 1
            continue
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"journey {jid!r} must have non-empty steps")
        if allows_hardware is True:
            raise ValueError(
                f"journey {jid!r} invent-green hardware: product journeys must "
                "set allows_hardware=False"
            )
    if blank:
        raise ValueError(f"Phase-QNode product registry has {blank} blank or invalid entries")
    if not default_found:
        raise ValueError("Phase-QNode product registry missing build_differentiate_dry_run")
    expected = set(list_phase_qnode_journey_ids())
    if seen != expected:
        raise ValueError(
            f"registry journey set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    journey_count = registry.get("journey_count", -1)
    if not isinstance(journey_count, int) or journey_count != len(journeys):
        raise ValueError("journey_count does not match journeys list length")
    return registry


__all__ = [
    "PHASE_QNODE_PRODUCT_CLAIM_BOUNDARY",
    "PHASE_QNODE_PRODUCT_SCHEMA",
    "JourneyOutcome",
    "PhaseQNodeJourney",
    "PhaseQNodeJourneyDecision",
    "SupportBadge",
    "assert_phase_qnode_product_integrity",
    "build_phase_qnode_product_registry",
    "dry_run_phase_qnode_journey",
    "get_phase_qnode_journey",
    "iter_phase_qnode_journeys",
    "list_phase_qnode_journey_ids",
    "map_phase_qnode_public_surfaces",
]
