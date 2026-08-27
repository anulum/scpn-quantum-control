# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — scorecard acceptance engine
"""Fail-closed baseline-scorecard acceptance / promotion engine.

Productises the eleven differentiable baseline-scorecard categories as a
versioned acceptance surface: list/query status, evidence pointers, and blockers;
refuse invent-green ``exceeds_baseline`` / ``at_baseline`` promotions without
required evidence packages.

Honest ``behind_baseline`` rows are expected until claim-ledger promotion and
isolated artefacts exist. This module composes category ids from
:mod:`differentiable_baseline_scorecard` and does not rewrite the committed
scorecard artefact.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from .differentiable_baseline_scorecard import (
    REQUIRED_BASELINE_CATEGORIES,
    DifferentiableBaselineCategory,
    DifferentiableBaselineStatus,
)

ScorecardEngineStatus = DifferentiableBaselineStatus
"""Re-export status vocabulary used by the baseline scorecard."""

SCORECARD_ACCEPTANCE_ENGINE_SCHEMA: Final[str] = "scorecard_acceptance_engine.v1"
"""JSON schema identifier for serialised engine payloads."""

SCORECARD_ACCEPTANCE_CLAIM_BOUNDARY: Final[str] = (
    "scorecard acceptance engine only; behind_baseline is an honest inventory "
    "state; promote refuses invent-green at_baseline/exceeds_baseline without "
    "required evidence digests and language-safe claims"
)
"""Shared claim boundary for category rows and promote results."""

_REQUIRED_EVIDENCE_KINDS: Final[tuple[str, ...]] = (
    "claim_ledger_promoted_row",
    "external_baseline_comparison",
)


@dataclass(frozen=True, slots=True)
class ScorecardCategoryRecord:
    """One scorecard category under the acceptance engine.

    Attributes
    ----------
    category_id
        One of the eleven baseline-scorecard category identifiers.
    status
        Current honest status (default inventory is ``behind_baseline``).
    summary
        Short description of the category.
    evidence_ids
        Attached evidence identifiers (empty until promotion packages exist).
    blockers
        Human-readable blockers remaining before promotion is allowed.
    required_evidence
        Evidence kinds required to consider promotion.
    as_of
        Inventory date label (ISO-like string, not a runtime clock claim).
    claim_boundary
        Non-promotional claim boundary.

    """

    category_id: DifferentiableBaselineCategory
    status: ScorecardEngineStatus
    summary: str
    evidence_ids: tuple[str, ...]
    blockers: tuple[str, ...]
    required_evidence: tuple[str, ...]
    as_of: str = "2026-07-23"
    claim_boundary: str = SCORECARD_ACCEPTANCE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate category-record invariants."""
        if self.category_id not in REQUIRED_BASELINE_CATEGORIES:
            raise ValueError(f"unknown scorecard category: {self.category_id!r}")
        if self.status not in {
            "behind_baseline",
            "at_baseline",
            "exceeds_baseline",
            "not_comparable",
        }:
            raise ValueError(f"unknown scorecard status: {self.status!r}")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")
        if any(not item or not item.strip() for item in self.evidence_ids):
            raise ValueError("evidence_ids must be non-empty strings when present")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers must be non-empty strings when present")
        if any(not item or not item.strip() for item in self.required_evidence):
            raise ValueError("required_evidence entries must be non-empty")
        if self.status == "behind_baseline" and not self.blockers:
            raise ValueError("behind_baseline rows require at least one blocker")
        if self.status in {"at_baseline", "exceeds_baseline"} and self.blockers:
            raise ValueError("promoted statuses must not carry open blockers")
        if self.status in {"at_baseline", "exceeds_baseline"} and not self.evidence_ids:
            raise ValueError("promoted statuses require evidence_ids")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this category record."""
        return {
            "category_id": self.category_id,
            "status": self.status,
            "summary": self.summary,
            "evidence_ids": list(self.evidence_ids),
            "blockers": list(self.blockers),
            "required_evidence": list(self.required_evidence),
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class PromoteDecision:
    """Result of a fail-closed promotion attempt.

    Attributes
    ----------
    category_id
        Target category.
    allowed
        Whether promotion was accepted (false for invent-green refusals).
    from_status
        Status before the attempt.
    to_status
        Requested target status.
    reason
        Human-readable decision reason.
    missing_evidence
        Required evidence kinds still missing.

    """

    category_id: str
    allowed: bool
    from_status: ScorecardEngineStatus
    to_status: ScorecardEngineStatus
    reason: str
    missing_evidence: tuple[str, ...] = ()
    claim_boundary: str = SCORECARD_ACCEPTANCE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate promote-decision invariants."""
        if not self.category_id or not self.category_id.strip():
            raise ValueError("category_id must be non-empty")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed and self.missing_evidence:
            raise ValueError("allowed promotions cannot list missing_evidence")
        if any(not item or not item.strip() for item in self.missing_evidence):
            raise ValueError("missing_evidence entries must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this decision."""
        return {
            "category_id": self.category_id,
            "allowed": self.allowed,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "reason": self.reason,
            "missing_evidence": list(self.missing_evidence),
            "claim_boundary": self.claim_boundary,
        }


def _behind(
    category_id: DifferentiableBaselineCategory,
    summary: str,
    *,
    blockers: Sequence[str],
) -> ScorecardCategoryRecord:
    """Build one honest behind_baseline inventory row."""
    return ScorecardCategoryRecord(
        category_id=category_id,
        status="behind_baseline",
        summary=summary,
        evidence_ids=(),
        blockers=tuple(blockers),
        required_evidence=_REQUIRED_EVIDENCE_KINDS,
    )


# Inventory: all eleven categories start honest behind_baseline until evidence packages land.
_CANONICAL_CATEGORIES: Final[tuple[ScorecardCategoryRecord, ...]] = (
    _behind(
        "jax_native_transforms",
        "JAX native transforms / value_and_grad parity vs external baseline.",
        blockers=(
            "promoted claim-ledger row absent",
            "external JAX baseline comparison package incomplete",
        ),
    ),
    _behind(
        "pytorch_autograd_compile",
        "PyTorch autograd / compile routes vs external baseline.",
        blockers=(
            "promoted claim-ledger row absent",
            "fullgraph compile evidence not registered",
        ),
    ),
    _behind(
        "pennylane_qnode_device_plugin",
        "PennyLane QNode / device-plugin gradient surface.",
        blockers=(
            "promoted claim-ledger row absent",
            "hardware-plugin path remains permanent boundary",
        ),
    ),
    _behind(
        "qiskit_runtime_provider_gradients",
        "Qiskit Runtime / provider gradient workflows.",
        blockers=(
            "promoted claim-ledger row absent",
            "provider Runtime evidence chain incomplete",
        ),
    ),
    _behind(
        "catalyst_compiler_workflows",
        "Catalyst compiler AD workflows (qjit/MLIR/QIR).",
        blockers=(
            "promoted claim-ledger row absent",
            "competitor boundary rows document missing batching rules",
        ),
    ),
    _behind(
        "enzyme_compiler_ad",
        "Enzyme / MLIR compiler AD kernels.",
        blockers=(
            "promoted claim-ledger row absent",
            "isolated compiler benchmark promotion package incomplete",
        ),
    ),
    _behind(
        "rust_native_program_ad",
        "Rust native Program AD registry parity.",
        blockers=(
            "promoted claim-ledger row absent",
            "fuzz/parity promotion package incomplete",
        ),
    ),
    _behind(
        "provider_hardware_gradients",
        "Provider / hardware gradient preparation (no live invent-green).",
        blockers=(
            "promoted claim-ledger row absent",
            "owner-ticket hardware evidence chain incomplete",
        ),
    ),
    _behind(
        "benchmark_promotion",
        "Isolated affinity benchmark promotion gate.",
        blockers=(
            "promoted claim-ledger row absent",
            "isolated_affinity artefacts incomplete",
        ),
    ),
    _behind(
        "docs_api_maintainability",
        "Public docs / API maintainability scorecard evidence.",
        blockers=(
            "promoted claim-ledger row absent",
            "public API stability package incomplete",
        ),
    ),
    _behind(
        "adoption_licensing",
        "Adoption / licensing clarity scorecard evidence.",
        blockers=(
            "promoted claim-ledger row absent",
            "adoption licensing evidence incomplete",
        ),
    ),
)


def _catalogue_map() -> dict[str, ScorecardCategoryRecord]:
    """Return category_id → record map; enforce full required coverage."""
    mapping: dict[str, ScorecardCategoryRecord] = {
        str(row.category_id): row for row in _CANONICAL_CATEGORIES
    }
    missing = set(REQUIRED_BASELINE_CATEGORIES) - set(mapping)
    if missing or len(mapping) != len(REQUIRED_BASELINE_CATEGORIES):
        raise RuntimeError(
            "scorecard acceptance catalogue must cover all required categories "
            f"(missing={missing!r})"
        )
    return mapping


_CATEGORY_BY_ID: Final[Mapping[str, ScorecardCategoryRecord]] = _catalogue_map()


def list_scorecard_category_ids() -> tuple[str, ...]:
    """Return all required scorecard category identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered category identifiers.

    """
    return tuple(row.category_id for row in _CANONICAL_CATEGORIES)


def get_scorecard_category(category_id: str) -> ScorecardCategoryRecord:
    """Return one category row or raise for unknown identifiers.

    Parameters
    ----------
    category_id
        Scorecard category key.

    Returns
    -------
    ScorecardCategoryRecord
        Matching catalogue row.

    Raises
    ------
    ValueError
        If ``category_id`` is blank or unknown.

    """
    if not category_id or not str(category_id).strip():
        raise ValueError("category_id must be a non-empty string")
    key = str(category_id).strip()
    try:
        return _CATEGORY_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown scorecard category_id {key!r}; refuse invent-green promotion "
            f"(known_count={len(_CATEGORY_BY_ID)})"
        ) from exc


def iter_scorecard_categories(
    *,
    status: ScorecardEngineStatus | None = None,
) -> tuple[ScorecardCategoryRecord, ...]:
    """Return filtered category rows in stable order.

    Parameters
    ----------
    status
        Optional status filter.

    Returns
    -------
    tuple[ScorecardCategoryRecord, ...]
        Matching rows.

    """
    rows: Iterable[ScorecardCategoryRecord] = _CANONICAL_CATEGORIES
    if status is not None:
        rows = (row for row in rows if row.status == status)
    return tuple(rows)


def build_scorecard_acceptance_registry() -> dict[str, object]:
    """Build the full serialisable scorecard acceptance registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with every required category (no blanks).

    """
    rows = [row.to_dict() for row in _CANONICAL_CATEGORIES]
    behind = sum(1 for row in _CANONICAL_CATEGORIES if row.status == "behind_baseline")
    ready = sum(
        1 for row in _CANONICAL_CATEGORIES if row.status in {"at_baseline", "exceeds_baseline"}
    )
    return {
        "schema": SCORECARD_ACCEPTANCE_ENGINE_SCHEMA,
        "claim_boundary": SCORECARD_ACCEPTANCE_CLAIM_BOUNDARY,
        "category_count": len(rows),
        "behind_baseline_count": behind,
        "ready_category_count": ready,
        "blank_entry_count": 0,
        "categories": rows,
    }


def promote_scorecard_category(
    category_id: str,
    *,
    target_status: ScorecardEngineStatus,
    evidence_ids: Sequence[str] = (),
    language_claim: str = "",
) -> PromoteDecision:
    """Attempt fail-closed promotion of one scorecard category.

    Promotion to ``at_baseline`` / ``exceeds_baseline`` requires:

    * non-empty ``evidence_ids`` covering required evidence kinds by label
      convention (each required kind name must appear as a substring of some
      evidence id), and
    * ``language_claim`` free of unbounded promotional phrases.

    Parameters
    ----------
    category_id
        Category to promote.
    target_status
        Requested status.
    evidence_ids
        Evidence package identifiers supplied by the caller.
    language_claim
        Optional claim text to language-gate.

    Returns
    -------
    PromoteDecision
        Allowed or refused decision (never invent-green).

    Raises
    ------
    ValueError
        If ``category_id`` / ``target_status`` are invalid.

    """
    record = get_scorecard_category(category_id)
    if target_status not in {
        "behind_baseline",
        "at_baseline",
        "exceeds_baseline",
        "not_comparable",
    }:
        raise ValueError(f"unknown target_status: {target_status!r}")

    if target_status == "behind_baseline":
        return PromoteDecision(
            category_id=record.category_id,
            allowed=True,
            from_status=record.status,
            to_status="behind_baseline",
            reason="demote/keep behind_baseline is always allowed (honest inventory)",
        )

    if target_status == "not_comparable":
        if not evidence_ids:
            return PromoteDecision(
                category_id=record.category_id,
                allowed=False,
                from_status=record.status,
                to_status=target_status,
                reason=(
                    "not_comparable requires evidence that comparison is impossible; "
                    "refuse invent-green empty evidence"
                ),
                missing_evidence=("comparison_impossibility_note",),
            )
        return PromoteDecision(
            category_id=record.category_id,
            allowed=True,
            from_status=record.status,
            to_status="not_comparable",
            reason="not_comparable accepted with supplied evidence labels",
        )

    # at_baseline / exceeds_baseline
    supplied = tuple(str(item).strip() for item in evidence_ids if str(item).strip())
    missing = tuple(
        kind
        for kind in record.required_evidence
        if not any(kind in evidence_id for evidence_id in supplied)
    )
    if missing:
        return PromoteDecision(
            category_id=record.category_id,
            allowed=False,
            from_status=record.status,
            to_status=target_status,
            reason=(f"refuse invent-green promotion: required evidence kinds missing {missing}"),
            missing_evidence=missing,
        )

    claim = (language_claim or "").lower()
    banned = (
        "category of its own",
        "world-class",
        "state-of-the-art",
        "promotion-ready",
        "production performance",
    )
    hits = tuple(phrase for phrase in banned if phrase in claim)
    if hits:
        return PromoteDecision(
            category_id=record.category_id,
            allowed=False,
            from_status=record.status,
            to_status=target_status,
            reason=(
                "refuse invent-green promotion language: "
                + ", ".join(hits)
                + "; use bounded non-promotional wording"
            ),
            missing_evidence=(),
        )

    return PromoteDecision(
        category_id=record.category_id,
        allowed=True,
        from_status=record.status,
        to_status=target_status,
        reason=(
            "promotion criteria satisfied for engine decision only; "
            "committed scorecard artefact regeneration remains a separate slice"
        ),
    )


def assert_scorecard_acceptance_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry payload covers all required categories without blanks.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_scorecard_acceptance_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-green ready rows without evidence appear.

    """
    registry = dict(payload) if payload is not None else build_scorecard_acceptance_registry()
    categories = registry.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("scorecard acceptance registry must contain a non-empty categories list")
    seen: set[str] = set()
    blank = 0
    for index, row in enumerate(categories):
        if not isinstance(row, Mapping):
            raise ValueError(f"category row {index} must be a mapping")
        category_id = row.get("category_id")
        status = row.get("status")
        if not category_id:
            blank += 1
            continue
        if category_id not in REQUIRED_BASELINE_CATEGORIES:
            raise ValueError(f"unknown category_id in registry: {category_id!r}")
        seen.add(str(category_id))
        if status not in {
            "behind_baseline",
            "at_baseline",
            "exceeds_baseline",
            "not_comparable",
        }:
            blank += 1
            continue
        if status == "behind_baseline":
            blockers = row.get("blockers")
            if not isinstance(blockers, list) or not blockers:
                raise ValueError(f"category {category_id!r} is behind_baseline without blockers")
        if status in {"at_baseline", "exceeds_baseline"}:
            evidence = row.get("evidence_ids")
            if not isinstance(evidence, list) or not evidence:
                raise ValueError(f"category {category_id!r} is promoted without evidence_ids")
    if blank:
        raise ValueError(f"scorecard acceptance registry has {blank} blank or invalid entries")
    missing = set(REQUIRED_BASELINE_CATEGORIES) - seen
    if missing:
        raise ValueError(f"scorecard acceptance registry missing categories: {missing}")
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    category_count = registry.get("category_count", -1)
    if not isinstance(category_count, int) or category_count != len(categories):
        raise ValueError("category_count does not match categories list length")
    return registry


__all__ = [
    "SCORECARD_ACCEPTANCE_CLAIM_BOUNDARY",
    "SCORECARD_ACCEPTANCE_ENGINE_SCHEMA",
    "PromoteDecision",
    "ScorecardCategoryRecord",
    "ScorecardEngineStatus",
    "assert_scorecard_acceptance_integrity",
    "build_scorecard_acceptance_registry",
    "get_scorecard_category",
    "iter_scorecard_categories",
    "list_scorecard_category_ids",
    "promote_scorecard_category",
]
