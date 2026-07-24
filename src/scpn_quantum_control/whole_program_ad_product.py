# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Whole-program AD product surface (BL-91 / P1)
"""Fail-closed whole-program AD **product** catalogue and journey map (BL-91).

Productises whole-program AD as a versioned frontend → IR → adjoint/replay
capability: public journey inventory, layered architecture map, dry-run
journey posture, fail-closed unknown/blank ids, and refuse invent-green for
unsupported frontends, hardware, polyglot certs, and edge WASM claims.

Composes honesty from BL-97 (workbench not silently SemVer-stable), BL-53
(unsupported frontend fail-closed), BL-46 (metamorphic/formal AD residual),
BL-49 (polyglot parity residual), BL-74 (edge WASM residual), and BL-95
(no invent-green compute). Does **not** freeze every ``whole_program_*`` /
``program_ad_*`` symbol as a mega-contract and does not execute QPU jobs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

SupportBadge = Literal[
    "local_dry_run",
    "frontend_boundary",
    "parity_boundary",
    "edge_boundary",
    "experimental_workbench",
]
"""Support badge vocabulary for whole-program AD product journeys."""

JourneyOutcome = Literal["allowed_dry_run", "refused"]
"""Structured dry-run journey outcomes."""

WHOLE_PROGRAM_AD_PRODUCT_SCHEMA: Final[str] = "whole_program_ad_product.v1"
"""JSON schema identifier for serialised product payloads."""

WHOLE_PROGRAM_AD_PRODUCT_CLAIM_BOUNDARY: Final[str] = (
    "Whole-program AD product surface only; catalogues public journeys and "
    "layered architecture map; ambient whole_program_*/program_ad_* workbench "
    "is not a frozen SemVer mega-contract (BL-97); unsupported frontend cases "
    "fail closed toward BL-53; polyglot parity certs (BL-49) and edge/WASM "
    "(BL-74) remain residual; dry-run journeys refuse invent-green hardware "
    "and unsupported execution (BL-95); does not replace full IR/adjoint engines"
)
"""Shared claim boundary for journeys and decisions."""


@dataclass(frozen=True, slots=True)
class WholeProgramADJourney:
    """One canonical whole-program AD product journey.

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
        Ordered journey step labels (frontend → IR → adjoint dry-run, …).
    allows_hardware
        Whether this journey claims hardware (product default false).
    architecture_layer
        Layered architecture label (frontend, ir, adjoint, product, residual).
    bl53_pointer
        Optional BL-53 unsuitable/unsupported pointer.
    bl97_stability_class
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
    architecture_layer: str = "frontend"
    bl53_pointer: str = ""
    bl97_stability_class: str = "experimental_workbench"
    as_of: str = "2026-07-24"
    claim_boundary: str = WHOLE_PROGRAM_AD_PRODUCT_CLAIM_BOUNDARY

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
            "frontend_boundary",
            "parity_boundary",
            "edge_boundary",
            "experimental_workbench",
        }:
            raise ValueError(f"unknown support_badge: {self.support_badge!r}")
        if not self.steps:
            raise ValueError("steps must be non-empty")
        if any(not step or not str(step).strip() for step in self.steps):
            raise ValueError("steps entries must be non-empty")
        if self.allows_hardware and self.support_badge == "local_dry_run":
            raise ValueError("local_dry_run journeys must set allows_hardware=False")
        if not self.architecture_layer or not self.architecture_layer.strip():
            raise ValueError("architecture_layer must be non-empty")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")
        if not self.bl97_stability_class or not self.bl97_stability_class.strip():
            raise ValueError("bl97_stability_class must be non-empty")

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
            "architecture_layer": self.architecture_layer,
            "bl53_pointer": self.bl53_pointer,
            "bl97_stability_class": self.bl97_stability_class,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class WholeProgramADJourneyDecision:
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
    claim_boundary: str = WHOLE_PROGRAM_AD_PRODUCT_CLAIM_BOUNDARY

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
            "frontend_boundary",
            "parity_boundary",
            "edge_boundary",
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
    architecture_layer: str = "frontend",
    bl53_pointer: str = "",
    bl97_stability_class: str = "experimental_workbench",
) -> WholeProgramADJourney:
    """Build one catalogue journey."""
    return WholeProgramADJourney(
        journey_id=journey_id,
        title=title,
        summary=summary,
        module_path=module_path,
        support_badge=support_badge,
        steps=steps,
        allows_hardware=allows_hardware,
        architecture_layer=architecture_layer,
        bl53_pointer=bl53_pointer,
        bl97_stability_class=bl97_stability_class,
    )


_CANONICAL_JOURNEYS: Final[tuple[WholeProgramADJourney, ...]] = (
    _journey(
        "frontend_compile_dry_run",
        title="Frontend compile → semantics report (dry-run)",
        summary=(
            "Canonical local journey: compile_whole_program_frontend inspects "
            "bytecode/source without executing the objective; produces "
            "frontend_ready or hard-gap diagnostics."
        ),
        module_path="scpn_quantum_control.whole_program_frontend",
        support_badge="local_dry_run",
        architecture_layer="frontend",
        steps=(
            "inspect_objective_source_bytecode",
            "build_frontend_report",
            "evaluate_semantics_gate",
            "emit_frontend_ready_or_hard_gap",
        ),
    ),
    _journey(
        "value_and_grad_local_dry_run",
        title="Frontend-ready → value_and_grad dry-run plan",
        summary=(
            "Plan whole_program_value_and_grad over a frontend-ready objective: "
            "trace-aware parameter injection, adjoint generation, and local "
            "replay posture without invent-green hardware or unsupported "
            "semantics."
        ),
        module_path="scpn_quantum_control.whole_program_ad_api",
        support_badge="local_dry_run",
        architecture_layer="product",
        steps=(
            "require_frontend_ready",
            "trace_objective_parameters",
            "generate_adjoint_plan",
            "record_value_and_grad_dry_run",
        ),
    ),
    _journey(
        "adjoint_replay_local_dry_run",
        title="IR nodes → adjoint replay dry-run",
        summary=(
            "Local adjoint/replay path over program_ad effect IR and result "
            "records; dry-run acknowledges replay steps without claiming "
            "polyglot bit-exact certificates."
        ),
        module_path="scpn_quantum_control.program_ad_adjoint",
        support_badge="local_dry_run",
        architecture_layer="adjoint",
        steps=(
            "materialise_ir_nodes",
            "plan_adjoint_generation",
            "replay_scalar_adjoint_dry_run",
            "attach_result_provenance",
        ),
    ),
    _journey(
        "unsupported_frontend_fail_closed",
        title="Unsupported frontend → BL-53 fail closed",
        summary=(
            "Product path for unsupported Python semantics (async, generators, "
            "context managers, filtered comprehensions, …): refuse invent-green "
            "execution and point at BL-53 unsuitable/anti-silent-wrong registry."
        ),
        module_path="scpn_quantum_control.whole_program_frontend_contracts",
        support_badge="frontend_boundary",
        architecture_layer="frontend",
        bl53_pointer="unsuitable_scenario_registry.unsupported_frontend_semantics",
        steps=(
            "detect_unsupported_semantic",
            "emit_hard_gap_diagnostic",
            "refuse_objective_execution",
            "point_bl53_unsuitable_registry",
        ),
    ),
    _journey(
        "polyglot_parity_boundary",
        title="Polyglot parity certificate boundary (BL-49 residual)",
        summary=(
            "Boundary-only product row for bit-exact polyglot parity certificates. "
            "Does not invent green parity; residual full BL-49 subset open."
        ),
        module_path="scpn_quantum_control.program_ad_rust_bridge",
        support_badge="parity_boundary",
        architecture_layer="residual",
        steps=(
            "declare_parity_boundary",
            "refuse_invent_green_polyglot_cert",
            "point_bl49_residual",
        ),
    ),
    _journey(
        "edge_wasm_boundary",
        title="Edge / WASM AD routing boundary (BL-74 residual)",
        summary=(
            "Boundary-only product row for edge WASM / Julia AD routing. "
            "Refuses invent-green edge execution; residual full BL-74 open."
        ),
        module_path="scpn_quantum_control.whole_program_ad_api",
        support_badge="edge_boundary",
        architecture_layer="residual",
        steps=(
            "declare_edge_boundary",
            "refuse_invent_green_wasm_edge",
            "point_bl74_residual",
        ),
    ),
)

_IR_LAYER_OWNERSHIP_MODULES: Final[tuple[str, ...]] = (
    "scpn_quantum_control.whole_program_ad_result",
    "scpn_quantum_control.program_ad_effect_ir",
    "scpn_quantum_control.program_ad_registry",
)
"""Ambient IR modules always attributed to the architecture IR layer."""


def _catalogue_map() -> dict[str, WholeProgramADJourney]:
    """Return journey_id → journey map; refuse blanks/duplicates."""
    mapping: dict[str, WholeProgramADJourney] = {}
    for row in _CANONICAL_JOURNEYS:
        key = row.journey_id.strip()
        if not key:
            raise RuntimeError("whole-program AD product catalogue contains blank journey_id")
        if key in mapping:
            raise RuntimeError(f"duplicate journey_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("whole-program AD product catalogue must be non-empty")
    return mapping


_JOURNEY_BY_ID: Final[Mapping[str, WholeProgramADJourney]] = _catalogue_map()


def list_whole_program_ad_journey_ids() -> tuple[str, ...]:
    """Return all product journey identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered journey identifiers.
    """
    return tuple(row.journey_id for row in _CANONICAL_JOURNEYS)


def get_whole_program_ad_journey(journey_id: str) -> WholeProgramADJourney:
    """Return one journey or raise for blank/unknown identifiers.

    Parameters
    ----------
    journey_id
        Catalogue journey key.

    Returns
    -------
    WholeProgramADJourney
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
            f"unknown journey_id {key!r}; refuse invent-green whole-program AD "
            f"product claim (known_count={len(_JOURNEY_BY_ID)})"
        ) from exc


def iter_whole_program_ad_journeys(
    *,
    support_badge: SupportBadge | None = None,
    architecture_layer: str | None = None,
) -> tuple[WholeProgramADJourney, ...]:
    """Return filtered journeys in stable order.

    Parameters
    ----------
    support_badge
        Optional badge filter.
    architecture_layer
        Optional architecture layer filter.

    Returns
    -------
    tuple[WholeProgramADJourney, ...]
        Matching journeys.
    """
    rows: Iterable[WholeProgramADJourney] = _CANONICAL_JOURNEYS
    if support_badge is not None:
        rows = (row for row in rows if row.support_badge == support_badge)
    if architecture_layer is not None:
        layer = architecture_layer.strip()
        rows = (row for row in rows if row.architecture_layer == layer)
    return tuple(rows)


def dry_run_whole_program_ad_journey(
    journey_id: str,
    *,
    request_hardware: bool = False,
    request_unsupported_frontend_execute: bool = False,
    request_polyglot_cert: bool = False,
    request_edge_wasm: bool = False,
) -> WholeProgramADJourneyDecision:
    """Exercise a product journey in dry-run posture (no QPU / invent-green).

    Acknowledges the catalogue journey steps as a structured dry-run plan.
    Hardware, unsupported-frontend execute, invent-green polyglot certs, and
    edge WASM requests are refused.

    Parameters
    ----------
    journey_id
        Catalogue journey key.
    request_hardware
        When true, refuse (no invent-green hardware).
    request_unsupported_frontend_execute
        When true, refuse execution of unsupported frontend semantics.
    request_polyglot_cert
        When true, refuse invent-green BL-49 polyglot certificates.
    request_edge_wasm
        When true, refuse invent-green BL-74 edge/WASM routing.

    Returns
    -------
    WholeProgramADJourneyDecision
        Allowed dry-run or refused decision.

    Raises
    ------
    ValueError
        If ``journey_id`` is blank or unknown.
    """
    journey = get_whole_program_ad_journey(journey_id)
    blockers: list[str] = []
    if request_hardware or journey.allows_hardware:
        blockers.append(
            "hardware/QPU request refused on whole-program AD product surface "
            "(compose BL-95 no invent-green compute)"
        )
    if request_unsupported_frontend_execute or (
        journey.support_badge == "frontend_boundary" and request_unsupported_frontend_execute
    ):
        blockers.append(
            "unsupported frontend execute refused; fail closed toward BL-53 "
            f"(pointer={journey.bl53_pointer or 'unsuitable_scenario_registry'})"
        )
    # Frontend-boundary journeys refuse invent-green execute when asked to
    # "run" unsupported paths; dry-run without the flag still plans refuse steps.
    if journey.support_badge == "frontend_boundary" and request_unsupported_frontend_execute:
        blockers.append("frontend_boundary journeys fail closed before objective execution")
    if request_polyglot_cert or (
        journey.support_badge == "parity_boundary" and request_polyglot_cert
    ):
        blockers.append(
            "polyglot parity certificate invent-green refused "
            "(BL-49 residual; product boundary only)"
        )
    if request_edge_wasm or (journey.support_badge == "edge_boundary" and request_edge_wasm):
        blockers.append("edge/WASM invent-green refused (BL-74 residual; product boundary only)")
    # Boundary journeys that are purely residual always refuse invent-green
    # "cert complete" claims when their residual flag is set; dry-run without
    # residual claim flags remains allowed as a boundary map step.
    if journey.support_badge == "parity_boundary" and request_polyglot_cert:
        blockers.append("parity_boundary journeys do not ship full BL-49 certs here")
    if journey.support_badge == "edge_boundary" and request_edge_wasm:
        blockers.append("edge_boundary journeys do not ship full BL-74 routing here")

    if blockers:
        unique = tuple(dict.fromkeys(item for item in blockers if item.strip()))
        return WholeProgramADJourneyDecision(
            journey_id=journey.journey_id,
            outcome="refused",
            allowed=False,
            support_badge=journey.support_badge,
            reason="whole-program AD product refuse: " + "; ".join(unique),
            blockers=unique,
            steps_completed=(),
        )

    return WholeProgramADJourneyDecision(
        journey_id=journey.journey_id,
        outcome="allowed_dry_run",
        allowed=True,
        support_badge=journey.support_badge,
        reason=(
            f"dry-run journey {journey.journey_id!r} allowed under product surface; "
            f"module={journey.module_path}; layer={journey.architecture_layer}; "
            f"badge={journey.support_badge}; stability={journey.bl97_stability_class}; "
            "no QPU submission and no invent-green residual cert occurred"
        ),
        blockers=(),
        steps_completed=journey.steps,
    )


def map_whole_program_ad_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of whole-program AD product modules (S91.0/S91.1).

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic module map rows for documentation and inventory.
    """
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
                "role": "whole_program_ad_product_surface",
                "architecture_layer": journey.architecture_layer,
                "bl97_stability_class": journey.bl97_stability_class,
                "support_badge": journey.support_badge,
                "journey_ids": [
                    j.journey_id for j in _CANONICAL_JOURNEYS if j.module_path == path
                ],
                "claim_boundary": WHOLE_PROGRAM_AD_PRODUCT_CLAIM_BOUNDARY,
            }
        )
    return tuple(rows)


def map_whole_program_ad_architecture_layers() -> tuple[dict[str, object], ...]:
    """Return layered architecture map for whole-program AD (S91.0).

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic layer rows (frontend → ir → adjoint → product → residual).
    """
    order = ("frontend", "ir", "adjoint", "product", "residual")
    layer_modules: dict[str, list[str]] = {name: [] for name in order}
    # Explicit IR layer ownership (not every journey is IR-primary).
    layer_modules["ir"].extend(_IR_LAYER_OWNERSHIP_MODULES)
    for journey in _CANONICAL_JOURNEYS:
        layer = journey.architecture_layer
        if layer not in layer_modules:
            layer_modules[layer] = []
        if journey.module_path not in layer_modules[layer]:
            layer_modules[layer].append(journey.module_path)
    rows: list[dict[str, object]] = []
    for layer in order:
        modules = layer_modules.get(layer, [])
        if not modules and layer == "ir":
            continue
        rows.append(
            {
                "layer": layer,
                "module_paths": list(modules),
                "journey_ids": [
                    j.journey_id for j in _CANONICAL_JOURNEYS if j.architecture_layer == layer
                ],
                "claim_boundary": WHOLE_PROGRAM_AD_PRODUCT_CLAIM_BOUNDARY,
            }
        )
    return tuple(rows)


def build_whole_program_ad_product_registry() -> dict[str, object]:
    """Build the full serialisable whole-program AD product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with every journey (no blanks).
    """
    journeys = [row.to_dict() for row in _CANONICAL_JOURNEYS]
    local = sum(1 for row in _CANONICAL_JOURNEYS if row.support_badge == "local_dry_run")
    return {
        "schema": WHOLE_PROGRAM_AD_PRODUCT_SCHEMA,
        "claim_boundary": WHOLE_PROGRAM_AD_PRODUCT_CLAIM_BOUNDARY,
        "journey_count": len(journeys),
        "local_dry_run_count": local,
        "default_journey_id": "frontend_compile_dry_run",
        "blank_entry_count": 0,
        "public_surfaces": list(map_whole_program_ad_public_surfaces()),
        "architecture_layers": list(map_whole_program_ad_architecture_layers()),
        "journeys": journeys,
        "policy_note": (
            "Whole-program AD product catalogue only; ambient whole_program_* / "
            "program_ad_* exports remain experimental_workbench under BL-97 "
            "unless promoted; S91.3 BL-49 polyglot cert subset and S91.4 BL-74 "
            "edge/WASM residual open honestly."
        ),
    }


def assert_whole_program_ad_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers journeys without blanks or invent-hardware.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_whole_program_ad_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-hardware rows appear.
    """
    registry = dict(payload) if payload is not None else build_whole_program_ad_product_registry()
    journeys = registry.get("journeys")
    if not isinstance(journeys, list) or not journeys:
        raise ValueError(
            "whole-program AD product registry must contain a non-empty journeys list"
        )
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
        architecture_layer = row.get("architecture_layer")
        if not journey_id or not str(journey_id).strip():
            blank += 1
            continue
        jid = str(journey_id).strip()
        if jid in seen:
            raise ValueError(f"duplicate journey_id in registry: {jid!r}")
        seen.add(jid)
        if jid == "frontend_compile_dry_run":
            default_found = True
            if allows_hardware is not False:
                raise ValueError("frontend_compile_dry_run must set allows_hardware=False")
        if support_badge not in {
            "local_dry_run",
            "frontend_boundary",
            "parity_boundary",
            "edge_boundary",
            "experimental_workbench",
        }:
            blank += 1
            continue
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"journey {jid!r} must have non-empty steps")
        if not architecture_layer or not str(architecture_layer).strip():
            raise ValueError(f"journey {jid!r} must have architecture_layer")
        if allows_hardware is True:
            raise ValueError(
                f"journey {jid!r} invent-green hardware: product journeys must "
                "set allows_hardware=False"
            )
        if jid == "unsupported_frontend_fail_closed":
            pointer = row.get("bl53_pointer")
            if not pointer or not str(pointer).strip():
                raise ValueError("unsupported_frontend_fail_closed must carry bl53_pointer")
    if blank:
        raise ValueError(f"whole-program AD product registry has {blank} blank or invalid entries")
    if not default_found:
        raise ValueError("whole-program AD product registry missing frontend_compile_dry_run")
    expected = set(list_whole_program_ad_journey_ids())
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
    layers = registry.get("architecture_layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("architecture_layers must be a non-empty list")
    return registry


__all__ = [
    "WHOLE_PROGRAM_AD_PRODUCT_CLAIM_BOUNDARY",
    "WHOLE_PROGRAM_AD_PRODUCT_SCHEMA",
    "JourneyOutcome",
    "SupportBadge",
    "WholeProgramADJourney",
    "WholeProgramADJourneyDecision",
    "assert_whole_program_ad_product_integrity",
    "build_whole_program_ad_product_registry",
    "dry_run_whole_program_ad_journey",
    "get_whole_program_ad_journey",
    "iter_whole_program_ad_journeys",
    "list_whole_program_ad_journey_ids",
    "map_whole_program_ad_architecture_layers",
    "map_whole_program_ad_public_surfaces",
]
