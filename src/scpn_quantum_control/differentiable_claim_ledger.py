# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable claim ledger.
"""Claim ledger for bounded differentiable Phase-QNode evidence."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PromotionStatus = Literal["promoted", "SOTA-candidate", "hard_gap", "blocked"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER_PATH = REPO_ROOT / "data" / "differentiable_phase_qnode" / "claim_ledger.json"


@dataclass(frozen=True)
class ClaimLedgerRow:
    """One bounded claim and its evidence surfaces."""

    claim_id: str
    claim_text: str
    implementation_surface: tuple[str, ...]
    test_surface: tuple[str, ...]
    docs_surface: tuple[str, ...]
    evidence_artifact_ids: tuple[str, ...]
    benchmark_artifact_ids: tuple[str, ...]
    known_gaps: tuple[str, ...]
    promotion_status: PromotionStatus
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("claim_id must be non-empty")
        if not self.claim_text:
            raise ValueError("claim_text must be non-empty")
        if self.promotion_status not in {"promoted", "SOTA-candidate", "hard_gap", "blocked"}:
            raise ValueError("promotion_status is unknown")
        for field_name in (
            "implementation_surface",
            "test_surface",
            "docs_surface",
            "known_gaps",
        ):
            value = getattr(self, field_name)
            if not value or any(not item for item in value):
                raise ValueError(f"{field_name} must contain non-empty entries")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    @classmethod
    def from_dict(cls, payload: MappingLike) -> ClaimLedgerRow:
        """Build a row from JSON data."""

        return cls(
            claim_id=str(payload["claim_id"]),
            claim_text=str(payload["claim_text"]),
            implementation_surface=tuple(str(item) for item in payload["implementation_surface"]),
            test_surface=tuple(str(item) for item in payload["test_surface"]),
            docs_surface=tuple(str(item) for item in payload["docs_surface"]),
            evidence_artifact_ids=tuple(str(item) for item in payload["evidence_artifact_ids"]),
            benchmark_artifact_ids=_string_tuple(
                payload.get("benchmark_artifact_ids") or payload["evidence_artifact_ids"]
            ),
            known_gaps=tuple(str(item) for item in payload["known_gaps"]),
            promotion_status=str(payload["promotion_status"]),  # type: ignore[arg-type]
            claim_boundary=str(payload["claim_boundary"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready row."""

        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "implementation_surface": list(self.implementation_surface),
            "test_surface": list(self.test_surface),
            "docs_surface": list(self.docs_surface),
            "evidence_artifact_ids": list(self.evidence_artifact_ids),
            "benchmark_artifact_ids": list(self.benchmark_artifact_ids),
            "known_gaps": list(self.known_gaps),
            "promotion_status": self.promotion_status,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class ClaimLedger:
    """Loaded differentiable claim ledger."""

    schema: str
    artifact_id: str
    rows: tuple[ClaimLedgerRow, ...]

    def __iter__(self):
        return iter(self.rows)


@dataclass(frozen=True)
class ClaimLedgerValidation:
    """Claim-ledger validation result."""

    passed: bool
    errors: tuple[str, ...]


MappingLike = dict[str, Any]


def _string_tuple(value: Any) -> tuple[str, ...]:
    """Return a tuple of strings from a JSON list-like value."""

    if not isinstance(value, list | tuple):
        raise ValueError("expected a list-like JSON value")
    return tuple(str(item) for item in value)


def load_differentiable_claim_ledger(path: Path = DEFAULT_LEDGER_PATH) -> ClaimLedger:
    """Load the committed differentiable claim ledger."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return ClaimLedger(
        schema=str(payload["schema"]),
        artifact_id=str(payload["artifact_id"]),
        rows=tuple(ClaimLedgerRow.from_dict(row) for row in payload["claims"]),
    )


def validate_claim_ledger(
    rows_or_ledger: ClaimLedger | Sequence[ClaimLedgerRow],
    *,
    artifact_statuses: MappingLike | None = None,
) -> ClaimLedgerValidation:
    """Validate claim ledger promotion invariants."""

    rows = (
        rows_or_ledger.rows if isinstance(rows_or_ledger, ClaimLedger) else tuple(rows_or_ledger)
    )
    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.claim_id in seen:
            errors.append(f"{row.claim_id}: duplicate claim_id")
        seen.add(row.claim_id)
        if row.promotion_status == "promoted" and not row.evidence_artifact_ids:
            errors.append(f"{row.claim_id}: promoted claims require at least one artefact ID")
        if row.promotion_status == "promoted" and not row.benchmark_artifact_ids:
            errors.append(f"{row.claim_id}: promoted claims require benchmark evidence IDs")
        if (
            row.promotion_status in {"promoted", "SOTA-candidate"}
            and not row.evidence_artifact_ids
        ):
            errors.append(f"{row.claim_id}: candidate claims require artefact ID evidence")
        if artifact_statuses and row.promotion_status == "promoted":
            for artifact_id in row.evidence_artifact_ids:
                if artifact_statuses.get(artifact_id) != "passed":
                    errors.append(f"{row.claim_id}: artefact {artifact_id} is not passed")
    return ClaimLedgerValidation(passed=not errors, errors=tuple(errors))


def validate_public_language_against_ledger(
    rows_or_ledger: ClaimLedger | Sequence[ClaimLedgerRow],
    public_texts: Iterable[str],
) -> ClaimLedgerValidation:
    """Reject public SOTA wording when the ledger has no promoted claim."""

    rows = (
        rows_or_ledger.rows if isinstance(rows_or_ledger, ClaimLedger) else tuple(rows_or_ledger)
    )
    has_promoted = any(row.promotion_status == "promoted" for row in rows)
    if has_promoted:
        return ClaimLedgerValidation(passed=True, errors=())
    banned = ("world-leading", "state-of-the-art", "SOTA", "production performance")
    errors: list[str] = []
    for text in public_texts:
        for phrase in banned:
            if phrase in text and "SOTA-candidate" not in text:
                errors.append(f"public wording exceeds claim ledger: {phrase}")
    return ClaimLedgerValidation(passed=not errors, errors=tuple(errors))


def render_claim_ledger_markdown(rows_or_ledger: ClaimLedger | Iterable[ClaimLedgerRow]) -> str:
    """Render a compact Markdown summary for reviewers."""

    rows = (
        rows_or_ledger.rows if isinstance(rows_or_ledger, ClaimLedger) else tuple(rows_or_ledger)
    )
    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "SCPN Quantum Control — Differentiable Phase-QNode Claim Ledger",
        "-->",
        "",
        "# Differentiable Phase-QNode Claim Ledger",
        "",
        "| Claim | Status | Artefact IDs | Benchmark IDs | Known gaps |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {claim} | {status} | {artefacts} | {benchmarks} | {gaps} |".format(
                claim=row.claim_id,
                status=row.promotion_status,
                artefacts=", ".join(row.evidence_artifact_ids) or "none",
                benchmarks=", ".join(row.benchmark_artifact_ids) or "none",
                gaps="<br>".join(row.known_gaps),
            )
        )
    lines.append("")
    lines.append(
        "Bounded language: the differentiable lane is SOTA-candidate unless isolated "
        "CI benchmark evidence and external comparison artefacts pass."
    )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "ClaimLedger",
    "ClaimLedgerRow",
    "ClaimLedgerValidation",
    "DEFAULT_LEDGER_PATH",
    "load_differentiable_claim_ledger",
    "render_claim_ledger_markdown",
    "validate_claim_ledger",
    "validate_public_language_against_ledger",
]
