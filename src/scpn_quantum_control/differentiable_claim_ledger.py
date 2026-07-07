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
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PromotionStatus = Literal["promoted", "SOTA-candidate", "hard_gap", "blocked"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER_PATH = REPO_ROOT / "data" / "differentiable_phase_qnode" / "claim_ledger.json"
DEFAULT_CAPABILITY_MANIFEST_PATH = REPO_ROOT / "docs" / "_generated" / "capability_manifest.json"
SUPPORT_SURFACE_ALIGNMENT_SCHEMA = "scpn_qc_differentiable_support_surface_alignment_v1"
DEFAULT_SUPPORT_SURFACE_ALIGNMENT_PATH = (
    REPO_ROOT
    / "data"
    / "differentiable_phase_qnode"
    / "differentiable_support_surface_alignment_20260627.json"
)


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
        """Validate row identity, status, surfaces, and boundary text."""
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

    def __iter__(self) -> Iterator[ClaimLedgerRow]:
        """Iterate over claim-ledger rows."""
        return iter(self.rows)


@dataclass(frozen=True)
class ClaimLedgerValidation:
    """Claim-ledger validation result."""

    passed: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class DifferentiableSupportSurfaceAlignment:
    """Docs/API/generated-manifest alignment result for differentiable claims."""

    passed: bool
    errors: tuple[str, ...]
    checked_claim_ids: tuple[str, ...]
    checked_paths: tuple[str, ...]
    claim_boundary: str
    schema: str = SUPPORT_SURFACE_ALIGNMENT_SCHEMA
    artifact_id: str = "diff-support-surface-alignment-audit-v1"

    @classmethod
    def from_dict(cls, payload: MappingLike) -> DifferentiableSupportSurfaceAlignment:
        """Build support-surface alignment evidence from JSON data."""
        schema = str(payload.get("schema", SUPPORT_SURFACE_ALIGNMENT_SCHEMA))
        if schema != SUPPORT_SURFACE_ALIGNMENT_SCHEMA:
            raise ValueError(f"unknown support-surface alignment schema: {schema}")
        return cls(
            passed=bool(payload["passed"]),
            errors=_string_tuple(payload["errors"]),
            checked_claim_ids=_string_tuple(payload["checked_claim_ids"]),
            checked_paths=_string_tuple(payload["checked_paths"]),
            claim_boundary=str(payload["claim_boundary"]),
            schema=schema,
            artifact_id=str(payload.get("artifact_id", "diff-support-surface-alignment-audit-v1")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready support-surface alignment evidence."""
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "passed": self.passed,
            "errors": list(self.errors),
            "checked_claim_ids": list(self.checked_claim_ids),
            "checked_paths": list(self.checked_paths),
            "claim_boundary": self.claim_boundary,
        }


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


def load_differentiable_support_surface_alignment(
    path: Path = DEFAULT_SUPPORT_SURFACE_ALIGNMENT_PATH,
) -> DifferentiableSupportSurfaceAlignment:
    """Load the committed differentiable support-surface alignment artefact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DifferentiableSupportSurfaceAlignment.from_dict(payload)


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
    """Reject public promotional wording when the ledger has no promoted claim."""
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


def validate_differentiable_support_surface_alignment(
    rows: Sequence[ClaimLedgerRow] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    manifest_path: Path = DEFAULT_CAPABILITY_MANIFEST_PATH,
) -> DifferentiableSupportSurfaceAlignment:
    """Validate that claim-ledger surfaces exist and match generated inventory."""
    ledger = load_differentiable_claim_ledger(ledger_path) if rows is None else None
    claim_rows = ledger.rows if ledger is not None else tuple(rows or ())
    manifest_paths, manifest_errors = _load_manifest_paths(manifest_path)
    errors: list[str] = list(manifest_errors)
    ledger_validation = validate_claim_ledger(claim_rows)
    errors.extend(ledger_validation.errors)

    checked_paths: set[str] = {str(manifest_path.relative_to(repo_root))}
    for row in claim_rows:
        for surface_name, paths in (
            ("implementation_surface", row.implementation_surface),
            ("test_surface", row.test_surface),
            ("docs_surface", row.docs_surface),
        ):
            for path in paths:
                checked_paths.add(path)
                if not (repo_root / path).exists():
                    errors.append(f"{row.claim_id}: {surface_name} path does not exist: {path}")
                if _requires_manifest_membership(path) and path not in manifest_paths:
                    errors.append(
                        f"{row.claim_id}: {surface_name} path is absent from "
                        f"generated capability manifest: {path}"
                    )

    return DifferentiableSupportSurfaceAlignment(
        passed=not errors,
        errors=tuple(errors),
        checked_claim_ids=tuple(row.claim_id for row in claim_rows),
        checked_paths=tuple(sorted(checked_paths)),
        claim_boundary=(
            "support-surface alignment audit only; verifies claim-ledger paths "
            "against the generated capability manifest without promoting "
            "differentiable performance, hardware, or provider claims"
        ),
    )


def _load_manifest_paths(manifest_path: Path) -> tuple[set[str], tuple[str, ...]]:
    if not manifest_path.exists():
        return set(), (f"generated capability manifest is missing: {manifest_path}",)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return set(), (f"generated capability manifest is not valid JSON: {exc}",)
    return _collect_manifest_paths(payload), ()


def _collect_manifest_paths(value: Any) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            paths.update(_collect_manifest_paths(child))
    elif isinstance(value, list | tuple):
        for child in value:
            paths.update(_collect_manifest_paths(child))
    elif isinstance(value, str) and _looks_like_repo_path(value):
        paths.add(value)
    return paths


def _looks_like_repo_path(value: str) -> bool:
    return (
        "/" in value
        and not value.startswith(("http://", "https://"))
        and not value.endswith("/")
        and any(value.startswith(prefix) for prefix in ("src/", "tests/", "docs/", "scripts/"))
    )


def _requires_manifest_membership(path: str) -> bool:
    return path.startswith(("src/", "tests/", "docs/")) and path != "README.md"


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
        "Bounded language: the differentiable lane remains a promotion candidate unless isolated "
        "CI benchmark evidence and external comparison artefacts pass."
    )
    return "\n".join(lines)


def render_differentiable_support_surface_alignment_markdown(
    alignment: DifferentiableSupportSurfaceAlignment,
) -> str:
    """Render support-surface alignment evidence for reviewers."""
    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "SCPN Quantum Control — Differentiable support-surface alignment",
        "-->",
        "",
        "# Differentiable Support-Surface Alignment",
        "",
        f"- Schema: `{alignment.schema}`",
        f"- Artifact ID: `{alignment.artifact_id}`",
        f"- `passed`: `{alignment.passed}`",
        f"- Claim boundary: {alignment.claim_boundary}",
        "",
        "## Checked Claims",
        "",
        "| Claim ID |",
        "|---|",
    ]
    for claim_id in alignment.checked_claim_ids:
        lines.append(f"| `{claim_id}` |")
    lines.extend(
        [
            "",
            "## Checked Paths",
            "",
            "| Path |",
            "|---|",
        ]
    )
    for path in alignment.checked_paths:
        lines.append(f"| `{path}` |")
    if alignment.errors:
        lines.extend(["", "## Errors", "", "| Error |", "|---|"])
        for error in alignment.errors:
            lines.append(f"| {error} |")
    return "\n".join(lines)


def render_public_claim_table(rows_or_ledger: ClaimLedger | Iterable[ClaimLedgerRow]) -> str:
    """Render public-safe differentiable claim wording from the ledger."""
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
        "SCPN Quantum Control — Differentiable Public Claim Table",
        "-->",
        "",
        "# Differentiable Public Claim Table",
        "",
        "This table is generated from the differentiable claim ledger. It is the",
        "public wording boundary for current differentiable-programming claims.",
        "",
        "| Claim ID | Public status | Public-safe wording | Do not claim yet | Evidence |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        safe_status = _public_status(row)
        lines.append(
            f"| `{row.claim_id}` | `{safe_status}` | {_public_safe_wording(row)} | "
            f"{_public_blocked_wording(row)} | {_public_evidence_wording(row)} |"
        )
    lines.extend(
        [
            "",
            "Global boundary: no differentiable row is promoted until the claim ledger,",
            "external comparison rows, and isolated CI benchmark artefacts all pass.",
        ]
    )
    return "\n".join(lines)


def validate_public_claim_table(
    rows_or_ledger: ClaimLedger | Iterable[ClaimLedgerRow],
    markdown: str,
) -> ClaimLedgerValidation:
    """Validate that the public claim table stays within ledger boundaries."""
    rows = (
        rows_or_ledger.rows if isinstance(rows_or_ledger, ClaimLedger) else tuple(rows_or_ledger)
    )
    errors: list[str] = []
    public_language = validate_public_language_against_ledger(rows, (markdown,))
    errors.extend(public_language.errors)
    for row in rows:
        if f"`{row.claim_id}`" not in markdown:
            errors.append(f"{row.claim_id}: missing public claim-table row")
        if row.claim_boundary not in markdown and row.promotion_status == "promoted":
            errors.append(f"{row.claim_id}: promoted row must include exact claim boundary")
    if any(row.promotion_status != "promoted" for row in rows):
        required = (
            "No hardware, provider, QPU, GPU, production-performance, or isolated_affinity claim."
        )
        if required not in markdown:
            errors.append("public claim table is missing the non-promotional boundary")
    return ClaimLedgerValidation(passed=not errors, errors=tuple(errors))


def _public_status(row: ClaimLedgerRow) -> str:
    if row.promotion_status == "promoted":
        return "promoted"
    if row.promotion_status == "hard_gap":
        return "blocked"
    return "bounded-candidate"


def _public_safe_wording(row: ClaimLedgerRow) -> str:
    if row.promotion_status == "promoted":
        return row.claim_text
    return (
        "Evidence-backed candidate surface for the bounded differentiable lane; "
        "use the listed artefacts and claim boundary when discussing scope."
    )


def _public_blocked_wording(row: ClaimLedgerRow) -> str:
    if row.promotion_status == "promoted":
        return "Do not extend beyond the exact claim boundary without a new ledger row."
    return "No hardware, provider, QPU, GPU, production-performance, or isolated_affinity claim."


def _public_evidence_wording(row: ClaimLedgerRow) -> str:
    artefacts = ", ".join(f"`{artifact}`" for artifact in row.evidence_artifact_ids)
    benchmarks = ", ".join(f"`{artifact}`" for artifact in row.benchmark_artifact_ids)
    return f"Artefacts: {artefacts}; benchmark IDs: {benchmarks}."


__all__ = [
    "ClaimLedger",
    "ClaimLedgerRow",
    "ClaimLedgerValidation",
    "DEFAULT_CAPABILITY_MANIFEST_PATH",
    "DEFAULT_LEDGER_PATH",
    "DEFAULT_SUPPORT_SURFACE_ALIGNMENT_PATH",
    "DifferentiableSupportSurfaceAlignment",
    "SUPPORT_SURFACE_ALIGNMENT_SCHEMA",
    "load_differentiable_support_surface_alignment",
    "load_differentiable_claim_ledger",
    "render_claim_ledger_markdown",
    "render_differentiable_support_surface_alignment_markdown",
    "render_public_claim_table",
    "validate_differentiable_support_surface_alignment",
    "validate_claim_ledger",
    "validate_public_claim_table",
    "validate_public_language_against_ledger",
]
