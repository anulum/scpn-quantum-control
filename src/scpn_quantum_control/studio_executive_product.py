# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio executive + coverage frontier product
"""Fail-closed **Studio executive + coverage frontier** product surface.

Productises executive verb→route catalogue honesty and a coverage-frontier
score (honesty × answer-rate — not all-boundary theatre):

* versioned verb catalogue over ambient :mod:`studio.verbs` /
  :func:`studio.executive.resolve_verb_contract` with route-matrix route pointers;
* fail-closed blank/unknown verbs and invent-green unsupported routes;
* materialised coverage-frontier probe with finite honesty/answer-rate fields
  and ``invent_green_full_coverage=false`` when boundary abstentions exist;
* refuse invent-green “100% route coverage” claims that hide refuse rates.

Does **not** re-architect Studio UI, invent full reproduction-kit export
automation, or claim federation depth complete.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

VerbKind = Literal[
    "compile",
    "simulate",
    "analyse",
    "validate",
    "benchmark",
    "replay",
    "differentiate",
    "mitigate",
    "execute",
]
"""Executive verb names on the QUANTUM studio contract."""

SupportPosture = Literal[
    "local_research",
    "live_hardware_gated",
    "policy_only",
]
"""Support posture badges for executive verb rows."""

PathDecisionOutcome = Literal["allowed", "refused"]
"""Structured path-eligibility outcomes."""

STUDIO_EXECUTIVE_PRODUCT_SCHEMA: Final[str] = "studio_executive_product.v2"
"""JSON schema identifier for serialised product payloads."""

STUDIO_EXECUTIVE_CLAIM_BOUNDARY: Final[str] = (
    "Studio executive + coverage frontier product surface only; catalogues "
    "executive verbs with governed-route pointers and materialises honesty×answer-rate "
    "coverage-frontier probes; invent_green_full_coverage=false when boundary "
    "abstentions exist; refuses invent-green unsupported routes and hidden refuse "
    "rates; does not claim full reproduction-kit export or Studio UI redesign"
)
"""Shared claim boundary for studio executive product payloads."""


@dataclass(frozen=True, slots=True)
class ExecutiveVerbRow:
    """One executive verb catalogue row.

    Attributes
    ----------
    verb_id
        Stable verb identifier (matches federation verb name).
    title
        Human-readable title.
    summary
        Short description.
    route_matrix_pointer
        Governed-route matrix pointer for this verb family.
    unsuitable_scenario_pointer
        Unsuitable-scenario / anti-silent-wrong honesty pointer.
    support_posture
        Support posture badge.
    requires_approval
        Whether live/certified verbs require explicit approval.
    allows_live_hardware
        Whether this verb may dispatch live hardware (execute only).
    backends
        Declared backend tokens.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    verb_id: str
    title: str
    summary: str
    route_matrix_pointer: str
    unsuitable_scenario_pointer: str
    support_posture: SupportPosture
    requires_approval: bool
    allows_live_hardware: bool
    backends: tuple[str, ...]
    as_of: str = "2026-07-24"
    claim_boundary: str = STUDIO_EXECUTIVE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate executive verb row invariants."""
        if not self.verb_id or not self.verb_id.strip():
            raise ValueError("verb_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.route_matrix_pointer or not self.route_matrix_pointer.strip():
            raise ValueError("route_matrix_pointer must be non-empty")
        if not self.unsuitable_scenario_pointer or not self.unsuitable_scenario_pointer.strip():
            raise ValueError("unsuitable_scenario_pointer must be non-empty")
        if self.support_posture not in {
            "local_research",
            "live_hardware_gated",
            "policy_only",
        }:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")
        if self.allows_live_hardware and self.verb_id != "execute":
            raise ValueError("only the execute verb may set allows_live_hardware=True")
        if self.allows_live_hardware and not self.requires_approval:
            raise ValueError("live-hardware verbs must require_approval=True")
        if not self.backends:
            raise ValueError("backends must be non-empty")
        if any(not item or not str(item).strip() for item in self.backends):
            raise ValueError("backends entries must be non-empty")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this row."""
        return {
            "verb_id": self.verb_id,
            "title": self.title,
            "summary": self.summary,
            "route_matrix_pointer": self.route_matrix_pointer,
            "unsuitable_scenario_pointer": self.unsuitable_scenario_pointer,
            "support_posture": self.support_posture,
            "requires_approval": self.requires_approval,
            "allows_live_hardware": self.allows_live_hardware,
            "backends": list(self.backends),
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PathEligibilityDecision:
    """Fail-closed path eligibility for studio executive product use.

    Attributes
    ----------
    outcome
        Allowed or refused.
    allowed
        Whether the path may proceed under this product.
    reason
        Human-readable reason.
    blockers
        Non-empty when refused.

    """

    outcome: PathDecisionOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = STUDIO_EXECUTIVE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate path eligibility invariants."""
        if self.outcome not in {"allowed", "refused"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed and self.outcome != "allowed":
            raise ValueError("allowed decisions must use outcome=allowed")
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
            "outcome": self.outcome,
            "allowed": self.allowed,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MaterialisedCoverageFrontierProbe:
    """Materialised honesty×answer-rate coverage frontier probe.

    Attributes
    ----------
    total_claims
        Total claims measured.
    answered_confident
        Confident answers (not abstentions).
    honest_abstentions
        Honest boundary/refuse abstentions (counted in honesty).
    answer_rate
        ``answered_confident / total_claims`` (0 when total is 0).
    honesty_rate
        ``(answered_confident + honest_abstentions) / total`` — honest bookkeeping.
    frontier_score
        ``honesty_rate * answer_rate`` (useful coverage under honesty).
    invent_green_full_coverage
        Must be False whenever honest abstentions > 0 or answer_rate < 1.
    off_frontier
        Whether improvable candidates remain unanswered.
    demo_label
        Demo fixture label.

    """

    total_claims: int
    answered_confident: int
    honest_abstentions: int
    answer_rate: float
    honesty_rate: float
    frontier_score: float
    invent_green_full_coverage: bool
    off_frontier: bool
    demo_label: str
    claim_boundary: str = STUDIO_EXECUTIVE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate coverage frontier probe invariants."""
        if self.total_claims < 0:
            raise ValueError("total_claims must be non-negative")
        if self.answered_confident < 0 or self.honest_abstentions < 0:
            raise ValueError("answered_confident and honest_abstentions must be non-negative")
        if self.answered_confident + self.honest_abstentions > self.total_claims:
            raise ValueError("answered + abstentions cannot exceed total_claims")
        if not 0.0 <= self.answer_rate <= 1.0:
            raise ValueError("answer_rate must be in [0, 1]")
        if not 0.0 <= self.honesty_rate <= 1.0:
            raise ValueError("honesty_rate must be in [0, 1]")
        if not 0.0 <= self.frontier_score <= 1.0:
            raise ValueError("frontier_score must be in [0, 1]")
        if self.invent_green_full_coverage:
            raise ValueError(
                "product probe must set invent_green_full_coverage=False "
                "(never invent-green 100% coverage while abstentions exist)"
            )
        if not self.demo_label or not self.demo_label.strip():
            raise ValueError("demo_label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "total_claims": self.total_claims,
            "answered_confident": self.answered_confident,
            "honest_abstentions": self.honest_abstentions,
            "answer_rate": self.answer_rate,
            "honesty_rate": self.honesty_rate,
            "frontier_score": self.frontier_score,
            "invent_green_full_coverage": self.invent_green_full_coverage,
            "off_frontier": self.off_frontier,
            "demo_label": self.demo_label,
            "claim_boundary": self.claim_boundary,
        }


def _row(
    verb_id: str,
    *,
    title: str,
    summary: str,
    route_matrix_pointer: str,
    support_posture: SupportPosture,
    requires_approval: bool,
    allows_live_hardware: bool,
    backends: tuple[str, ...],
) -> ExecutiveVerbRow:
    """Build one verb catalogue row."""
    return ExecutiveVerbRow(
        verb_id=verb_id,
        title=title,
        summary=summary,
        route_matrix_pointer=route_matrix_pointer,
        unsuitable_scenario_pointer="unsuitable_scenario_registry + metamorphic_ad_verification",
        support_posture=support_posture,
        requires_approval=requires_approval,
        allows_live_hardware=allows_live_hardware,
        backends=backends,
    )


_ROUTE_MAP: Final[dict[str, str]] = {
    "compile": "governed_route:studio.compile.kuramoto_xy",
    "simulate": "governed_route:studio.simulate.quantum_evolution",
    "analyse": "governed_route:studio.analyse.sync_witness",
    "validate": "governed_route:studio.validate.physics_parity",
    "benchmark": "governed_route:studio.benchmark.databank",
    "replay": "governed_route:studio.replay.evidence",
    "differentiate": "governed_route:studio.differentiate.program_ad",
    "mitigate": "governed_route:studio.mitigate.readout_zne",
    "execute": "governed_route:studio.execute.qpu_hal_gated",
}
_TITLES: Final[dict[str, str]] = {
    "compile": "Compile phase networks",
    "simulate": "Simulate quantum evolution",
    "analyse": "Analyse synchronisation / DLA",
    "validate": "Validate physics / parity",
    "benchmark": "Benchmark databank",
    "replay": "Replay evidence packs",
    "differentiate": "Differentiate programmes",
    "mitigate": "Mitigate readout / noise",
    "execute": "Execute on QPU (approval-gated)",
}
_SUMMARIES: Final[dict[str, str]] = {
    "compile": "Compile K_nm/omega networks into XY/XXZ Hamiltonians and circuits.",
    "simulate": "Evolve state on simulator backends (no live QPU).",
    "analyse": "Sync analysis, DLA parity, coupling invariants.",
    "validate": "Physics validation and readiness checks.",
    "benchmark": "Native/speedup databank measurements.",
    "replay": "Evidence-pack replay without re-submission.",
    "differentiate": "Value-and-grad / adjoint over compiled phase programmes.",
    "mitigate": "ZNE/PEC/readout mitigation with uncertainty.",
    "execute": "Live hardware via approval-gated provider HAL (certified).",
}
# Product-local fallback when Studio platform extra is absent (Python 3.11 CI).
# execute is the only live-hardware verb; backends mirror ambient contracts.
_FALLBACK_VERB_SPECS: Final[
    tuple[tuple[str, SupportPosture, bool, bool, tuple[str, ...]], ...]
] = (
    ("compile", "local_research", False, False, ("numpy", "jax", "torch")),
    ("simulate", "local_research", False, False, ("numpy", "jax", "torch", "qiskit")),
    ("analyse", "local_research", False, False, ("numpy",)),
    ("validate", "local_research", False, False, ("numpy",)),
    ("benchmark", "local_research", False, False, ("numpy",)),
    ("replay", "local_research", False, False, ("numpy",)),
    ("differentiate", "local_research", False, False, ("numpy", "jax", "torch")),
    ("mitigate", "local_research", False, False, ("numpy",)),
    ("execute", "live_hardware_gated", True, True, ("qiskit", "braket", "iqm")),
)


def _build_fallback_canonical_verbs() -> tuple[ExecutiveVerbRow, ...]:
    """Build verb catalogue without importing the optional Studio platform."""
    rows: list[ExecutiveVerbRow] = []
    for name, posture, requires_approval, live, backends in _FALLBACK_VERB_SPECS:
        rows.append(
            _row(
                name,
                title=_TITLES.get(name, name),
                summary=_SUMMARIES.get(name, f"Studio verb {name}"),
                route_matrix_pointer=_ROUTE_MAP.get(name, f"governed_route:studio.{name}"),
                support_posture=posture,
                requires_approval=requires_approval,
                allows_live_hardware=live,
                backends=backends,
            )
        )
    if not rows:
        raise RuntimeError("studio executive catalogue must be non-empty")
    return tuple(rows)


def _build_canonical_verbs() -> tuple[ExecutiveVerbRow, ...]:
    """Build verb catalogue from ambient federation contracts when available.

    Falls back to the product-local catalogue when ``scpn_studio_platform`` is
    not installed (base Python 3.11 CI matrix; Studio extra is ≥3.12 only).
    """
    try:
        from .studio.executive import resolve_verb_contract
        from .studio.verbs import QUANTUM_VERBS
    except ImportError:
        return _build_fallback_canonical_verbs()

    rows: list[ExecutiveVerbRow] = []
    for declared in QUANTUM_VERBS:
        name = str(declared.name)
        contract = resolve_verb_contract(name)
        live = contract.side_effect == "LIVE_HARDWARE"
        rows.append(
            _row(
                name,
                title=_TITLES.get(name, name),
                summary=_SUMMARIES.get(name, f"Studio verb {name}"),
                route_matrix_pointer=_ROUTE_MAP.get(name, f"governed_route:studio.{name}"),
                support_posture=("live_hardware_gated" if live else "local_research"),
                requires_approval=bool(contract.requires_approval),
                allows_live_hardware=live,
                backends=tuple(contract.backends),
            )
        )
    if not rows:
        raise RuntimeError("studio executive catalogue must be non-empty")
    return tuple(rows)


_CANONICAL_VERBS: Final[tuple[ExecutiveVerbRow, ...]] = _build_canonical_verbs()


def _catalogue_map() -> dict[str, ExecutiveVerbRow]:
    """Return verb_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, ExecutiveVerbRow] = {}
    for row in _CANONICAL_VERBS:
        key = row.verb_id.strip()
        if not key:
            raise RuntimeError("studio executive catalogue contains blank verb_id")
        if key in mapping:
            raise RuntimeError(f"duplicate verb_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("studio executive catalogue must be non-empty")
    return mapping


_VERB_BY_ID: Final[Mapping[str, ExecutiveVerbRow]] = _catalogue_map()


def list_executive_verb_ids() -> tuple[str, ...]:
    """Return all executive verb identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered verb identifiers.

    """
    return tuple(row.verb_id for row in _CANONICAL_VERBS)


def get_executive_verb(verb_id: str) -> ExecutiveVerbRow:
    """Return one verb row or raise for blank/unknown identifiers.

    Parameters
    ----------
    verb_id
        Catalogue verb key.

    Returns
    -------
    ExecutiveVerbRow
        Matching row.

    Raises
    ------
    ValueError
        If ``verb_id`` is blank or unknown (fail closed).

    """
    if not verb_id or not str(verb_id).strip():
        raise ValueError("verb_id must be a non-empty string")
    key = str(verb_id).strip()
    try:
        return _VERB_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown verb_id {key!r}; refuse invent-green studio executive "
            f"product claim (known_count={len(_VERB_BY_ID)})"
        ) from exc


def iter_executive_verbs(
    *,
    support_posture: SupportPosture | None = None,
) -> tuple[ExecutiveVerbRow, ...]:
    """Return filtered verb rows in stable order.

    Parameters
    ----------
    support_posture
        Optional posture filter.

    Returns
    -------
    tuple[ExecutiveVerbRow, ...]
        Matching rows.

    """
    rows: Sequence[ExecutiveVerbRow] = _CANONICAL_VERBS
    if support_posture is not None:
        rows = tuple(row for row in rows if row.support_posture == support_posture)
    return tuple(rows)


def decide_executive_path(
    verb_id: str,
    *,
    request_unsupported_route: bool = False,
    invent_green_full_coverage: bool = False,
    approval_present: bool = False,
) -> PathEligibilityDecision:
    """Decide whether an executive verb path may proceed.

    Parameters
    ----------
    verb_id
        Verb to validate (blank/unknown fail closed).
    request_unsupported_route
        When true, refuse under the governed-route and unsuitable-scenario policy.
    invent_green_full_coverage
        When true, refuse invent-green 100% coverage claims.
    approval_present
        Whether explicit approval is present for gated verbs.

    Returns
    -------
    PathEligibilityDecision
        Allowed or refused decision with blockers.

    Raises
    ------
    ValueError
        If ``verb_id`` is blank or unknown.

    """
    row = get_executive_verb(verb_id)
    blockers: list[str] = []
    if request_unsupported_route:
        blockers.append(
            f"unsupported route invent-green refused for verb {row.verb_id!r} "
            f"(governed route={row.route_matrix_pointer}; unsuitable scenario="
            f"{row.unsuitable_scenario_pointer})"
        )
    if invent_green_full_coverage:
        blockers.append(
            "invent-green full coverage claim refused "
            "(coverage frontier must expose honesty×answer-rate; hide refuse rates forbidden)"
        )
    if row.requires_approval and not approval_present:
        blockers.append(
            f"verb {row.verb_id!r} requires explicit approval before execution "
            f"(side-effect gated; posture={row.support_posture})"
        )
    if blockers:
        unique = tuple(dict.fromkeys(item for item in blockers if item.strip()))
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="studio executive product refuse: " + "; ".join(unique),
            blockers=unique,
        )
    return PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason=(
            f"studio executive path allowed for verb {row.verb_id!r} "
            f"(route={row.route_matrix_pointer}; live_hardware={row.allows_live_hardware})"
        ),
        blockers=(),
    )


def compute_coverage_frontier_score(
    *,
    total_claims: int,
    answered_confident: int,
    honest_abstentions: int,
    improvable_candidates: int = 0,
) -> MaterialisedCoverageFrontierProbe:
    """Compute honesty×answer-rate coverage frontier score.

    Parameters
    ----------
    total_claims
        Total claims measured (must be positive for a materialised demo).
    answered_confident
        Confident answers.
    honest_abstentions
        Honest boundary refuse / abstention counts.
    improvable_candidates
        Unanswered candidates that could improve with evidence (off-frontier).

    Returns
    -------
    MaterialisedCoverageFrontierProbe
        Finite primary observables with invent_green_full_coverage=False.

    Raises
    ------
    ValueError
        If counts are inconsistent.

    """
    if total_claims <= 0:
        raise ValueError("total_claims must be positive for materialised frontier probe")
    if improvable_candidates < 0:
        raise ValueError("improvable_candidates must be non-negative")
    if answered_confident + honest_abstentions + improvable_candidates > total_claims:
        raise ValueError("partition of claims exceeds total_claims")

    answer_rate = answered_confident / total_claims
    honesty_rate = (answered_confident + honest_abstentions) / total_claims
    frontier_score = honesty_rate * answer_rate
    # Never invent-green full coverage while any abstention or incomplete answer rate.
    invent_green = False
    return MaterialisedCoverageFrontierProbe(
        total_claims=total_claims,
        answered_confident=answered_confident,
        honest_abstentions=honest_abstentions,
        answer_rate=float(answer_rate),
        honesty_rate=float(honesty_rate),
        frontier_score=float(frontier_score),
        invent_green_full_coverage=invent_green,
        off_frontier=improvable_candidates > 0,
        demo_label="synthetic_honesty_answer_rate_frontier",
    )


def materialise_demo_coverage_frontier_probe() -> MaterialisedCoverageFrontierProbe:
    """Materialise a deterministic coverage-frontier demo probe.

    Uses a fixed synthetic ledger partition: 10 total, 3 answered, 5 honest
    abstentions, 2 improvable candidates → answer_rate=0.3, honesty_rate=0.8,
    frontier_score=0.24, invent_green_full_coverage=False, off_frontier=True.

    Returns
    -------
    MaterialisedCoverageFrontierProbe
        Finite primary observables.

    Raises
    ------
    ValueError
        If scoring validation fails.

    """
    return compute_coverage_frontier_score(
        total_claims=10,
        answered_confident=3,
        honest_abstentions=5,
        improvable_candidates=2,
    )


def map_studio_executive_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of studio executive product modules.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    return (
        {
            "module_path": "scpn_quantum_control.studio_executive_product",
            "role": "studio_executive_product_surface",
            "support_posture": "local_research",
            "verb_ids": list(list_executive_verb_ids()),
            "invent_green_full_coverage": False,
            "claim_boundary": STUDIO_EXECUTIVE_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.studio.verbs",
            "role": "ambient_federation_verb_spine",
            "support_posture": "local_research",
            "symbol_name": "QUANTUM_VERBS",
            "claim_boundary": STUDIO_EXECUTIVE_CLAIM_BOUNDARY,
        },
        {
            "module_path": "scpn_quantum_control.studio.coverage_frontier",
            "role": "ambient_coverage_frontier_policy",
            "support_posture": "policy_only",
            "symbol_name": "measure_coverage_frontier",
            "claim_boundary": STUDIO_EXECUTIVE_CLAIM_BOUNDARY,
        },
    )


def build_studio_executive_product_registry() -> dict[str, object]:
    """Build the full serialisable studio executive product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with verbs (no blanks).

    """
    verbs = [row.to_dict() for row in _CANONICAL_VERBS]
    return {
        "schema": STUDIO_EXECUTIVE_PRODUCT_SCHEMA,
        "claim_boundary": STUDIO_EXECUTIVE_CLAIM_BOUNDARY,
        "verb_count": len(verbs),
        "blank_entry_count": 0,
        "default_verb_id": "differentiate",
        "invent_green_full_coverage_policy": False,
        "public_surfaces": list(map_studio_executive_public_surfaces()),
        "verbs": verbs,
        "policy_note": (
            "Studio executive product catalogue only; ambient studio.verbs / "
            "executive spine / coverage_frontier remain the implementation; "
            "full reproduction-kit evidence export remains open; invent_green_full_coverage "
            "forbidden when honesty×answer-rate shows abstentions."
        ),
    }


def assert_studio_executive_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers verbs without blanks or invent-green coverage.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_studio_executive_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-green policies appear.

    """
    registry = dict(payload) if payload is not None else build_studio_executive_product_registry()
    if registry.get("schema") != STUDIO_EXECUTIVE_PRODUCT_SCHEMA:
        raise ValueError("studio executive product schema mismatch")
    verbs = registry.get("verbs")
    if not isinstance(verbs, list) or not verbs:
        raise ValueError("studio executive product registry must contain a non-empty verbs list")
    seen: set[str] = set()
    blank = 0
    default_found = False
    execute_found = False
    for index, row in enumerate(verbs):
        if not isinstance(row, Mapping):
            raise ValueError(f"verb row {index} must be a mapping")
        verb_id = row.get("verb_id")
        route_matrix_row = row.get("route_matrix_pointer")
        allows_live = row.get("allows_live_hardware")
        backends = row.get("backends")
        if not verb_id or not str(verb_id).strip():
            blank += 1
            continue
        vid = str(verb_id).strip()
        if vid in seen:
            raise ValueError(f"duplicate verb_id in registry: {vid!r}")
        seen.add(vid)
        if vid == "differentiate":
            default_found = True
        if vid == "execute":
            execute_found = True
        if not route_matrix_row or not str(route_matrix_row).strip():
            raise ValueError(f"verb {vid!r} must have route_matrix_pointer")
        if not isinstance(backends, list) or not backends:
            raise ValueError(f"verb {vid!r} must have non-empty backends list")
        if allows_live is True and vid != "execute":
            raise ValueError(
                f"verb {vid!r} invent-green live hardware: only execute may allow live hardware"
            )
    if blank:
        raise ValueError(f"studio executive product registry has {blank} blank or invalid entries")
    if not default_found:
        raise ValueError("studio executive product registry missing differentiate")
    if not execute_found:
        raise ValueError("studio executive product registry missing execute")
    expected = set(list_executive_verb_ids())
    if seen != expected:
        raise ValueError(
            f"registry verb set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    verb_count = registry.get("verb_count", -1)
    if not isinstance(verb_count, int) or verb_count != len(verbs):
        raise ValueError("verb_count does not match verbs list length")
    invent_policy = registry.get("invent_green_full_coverage_policy", True)
    if invent_policy is not False:
        raise ValueError("invent_green_full_coverage_policy must be False")
    return registry


__all__ = [
    "STUDIO_EXECUTIVE_CLAIM_BOUNDARY",
    "STUDIO_EXECUTIVE_PRODUCT_SCHEMA",
    "ExecutiveVerbRow",
    "MaterialisedCoverageFrontierProbe",
    "PathDecisionOutcome",
    "PathEligibilityDecision",
    "SupportPosture",
    "VerbKind",
    "assert_studio_executive_product_integrity",
    "build_studio_executive_product_registry",
    "compute_coverage_frontier_score",
    "decide_executive_path",
    "get_executive_verb",
    "iter_executive_verbs",
    "list_executive_verb_ids",
    "map_studio_executive_public_surfaces",
    "materialise_demo_coverage_frontier_probe",
]
