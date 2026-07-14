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
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from .differentiable_claim_rendering import (
    NON_PROMOTIONAL_BOUNDARY as _NON_PROMOTIONAL_BOUNDARY,
)
from .differentiable_claim_rendering import (
    render_claim_ledger_markdown,
    render_differentiable_support_surface_alignment_markdown,
    render_public_claim_table,
)

PromotionStatus = Literal["promoted", "bounded_candidate", "hard_gap", "blocked"]

REPO_ROOT = Path(__file__).resolve().parents[2]
CLAIM_LEDGER_SCHEMA = "scpn_qc_differentiable_claim_ledger_v1"
CLAIM_LEDGER_ARTIFACT_ID = "diff-qnode-claim-ledger-v1"
DEFAULT_LEDGER_PATH = REPO_ROOT / "data" / "differentiable_phase_qnode" / "claim_ledger.json"
DEFAULT_CAPABILITY_MANIFEST_PATH = REPO_ROOT / "docs" / "_generated" / "capability_manifest.json"
SUPPORT_SURFACE_ALIGNMENT_SCHEMA = "scpn_qc_differentiable_support_surface_alignment_v1"
SUPPORT_SURFACE_ALIGNMENT_ARTIFACT_ID = "diff-support-surface-alignment-audit-v1"
DEFAULT_SUPPORT_SURFACE_ALIGNMENT_PATH = (
    REPO_ROOT
    / "data"
    / "differentiable_phase_qnode"
    / "differentiable_support_surface_alignment_20260627.json"
)

_PROMOTION_STATUSES = frozenset({"promoted", "bounded_candidate", "hard_gap", "blocked"})
_CLAIM_ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]*\Z")
_BANNED_PUBLIC_PHRASES = (
    "world-leading",
    "state-of-the-art",
    "sota",
    "production performance",
)
_CLAIM_LEDGER_METADATA = {
    "SPDX-License-Identifier": "AGPL-3.0-or-later",
    "commercial_license": "available",
    "contact": "www.anulum.li | protoscience@anulum.li",
    "copyright_code": "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
    "copyright_concepts": "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
    "orcid": "0009-0009-3560-0851",
    "title": "SCPN Quantum Control — Differentiable Phase-QNode Claim Ledger",
}


@dataclass(frozen=True)
class ClaimLedgerRow:
    """Describe one bounded claim and its evidence surfaces.

    Parameters
    ----------
    claim_id
        Stable lowercase snake- or kebab-case identity for the claim.
    claim_text
        Human-readable statement governed by the row.
    implementation_surface
        Repository-relative production paths supporting the claim.
    test_surface
        Repository-relative test paths exercising the production surfaces.
    docs_surface
        Repository-relative documentation and evidence paths.
    evidence_artifact_ids
        Stable identities for supporting evidence artefacts.
    benchmark_artifact_ids
        Stable identities for benchmark evidence.
    known_gaps
        Explicit limitations that bound the claim.
    promotion_status
        Current promotion state.
    claim_boundary
        Exact wording that limits public use of the claim.
    """

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
        """Validate row identity, status, surfaces, and boundary text.

        Raises
        ------
        ValueError
            If an identity, status, surface, artefact ID, gap, or boundary is
            malformed.
        """
        claim_id = _nonblank_string(self.claim_id, "claim_id")
        _nonblank_string(self.claim_text, "claim_text")
        _nonblank_string(self.claim_boundary, "claim_boundary")
        if _CLAIM_ID_PATTERN.fullmatch(claim_id) is None:
            raise ValueError("claim_id must use lower snake or kebab case")
        if (
            not isinstance(self.promotion_status, str)
            or self.promotion_status not in _PROMOTION_STATUSES
        ):
            raise ValueError("promotion_status is unknown")
        for field_name, value, allow_empty in (
            ("implementation_surface", self.implementation_surface, False),
            ("test_surface", self.test_surface, False),
            ("docs_surface", self.docs_surface, False),
            ("evidence_artifact_ids", self.evidence_artifact_ids, True),
            ("benchmark_artifact_ids", self.benchmark_artifact_ids, True),
            ("known_gaps", self.known_gaps, False),
        ):
            _string_tuple(
                value,
                field_name=field_name,
                allow_empty=allow_empty,
                allow_list=False,
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ClaimLedgerRow:
        """Build a row from decoded JSON data.

        Parameters
        ----------
        payload
            Decoded claim-row mapping.

        Returns
        -------
        ClaimLedgerRow
            Validated immutable row.

        Raises
        ------
        ValueError
            If a required field is missing or has the wrong type or value.
        """
        benchmark_ids = (
            _required_value(payload, "benchmark_artifact_ids", "claim row")
            if "benchmark_artifact_ids" in payload
            else _required_value(payload, "evidence_artifact_ids", "claim row")
        )
        status = _required_string(payload, "promotion_status", "claim row")
        return cls(
            claim_id=_required_string(payload, "claim_id", "claim row"),
            claim_text=_required_string(payload, "claim_text", "claim row"),
            implementation_surface=_string_tuple(
                _required_value(payload, "implementation_surface", "claim row"),
                field_name="implementation_surface",
                allow_empty=False,
            ),
            test_surface=_string_tuple(
                _required_value(payload, "test_surface", "claim row"),
                field_name="test_surface",
                allow_empty=False,
            ),
            docs_surface=_string_tuple(
                _required_value(payload, "docs_surface", "claim row"),
                field_name="docs_surface",
                allow_empty=False,
            ),
            evidence_artifact_ids=_string_tuple(
                _required_value(payload, "evidence_artifact_ids", "claim row"),
                field_name="evidence_artifact_ids",
            ),
            benchmark_artifact_ids=_string_tuple(
                benchmark_ids,
                field_name="benchmark_artifact_ids",
            ),
            known_gaps=_string_tuple(
                _required_value(payload, "known_gaps", "claim row"),
                field_name="known_gaps",
                allow_empty=False,
            ),
            promotion_status=cast(PromotionStatus, status),
            claim_boundary=_required_string(payload, "claim_boundary", "claim row"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready row.

        Returns
        -------
        dict[str, object]
            Mapping that preserves the committed row schema.
        """
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
    """Hold a validated differentiable claim ledger.

    Parameters
    ----------
    schema
        Ledger schema identity.
    artifact_id
        Stable ledger artefact identity.
    rows
        Non-empty ordered claim rows.
    """

    schema: str
    artifact_id: str
    rows: tuple[ClaimLedgerRow, ...]

    def __post_init__(self) -> None:
        """Validate ledger identity, row types, and row uniqueness.

        Raises
        ------
        ValueError
            If the ledger identity or row collection is malformed.
        """
        if _nonblank_string(self.schema, "schema") != CLAIM_LEDGER_SCHEMA:
            raise ValueError(f"unknown claim-ledger schema: {self.schema}")
        if _nonblank_string(self.artifact_id, "artifact_id") != CLAIM_LEDGER_ARTIFACT_ID:
            raise ValueError(f"unknown claim-ledger artifact_id: {self.artifact_id}")
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValueError("claim ledger rows must be a non-empty tuple")
        if any(not isinstance(row, ClaimLedgerRow) for row in self.rows):
            raise ValueError("claim ledger rows must contain ClaimLedgerRow values")
        claim_ids = tuple(row.claim_id for row in self.rows)
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("claim ledger rows contain duplicate claim_id values")

    def __iter__(self) -> Iterator[ClaimLedgerRow]:
        """Iterate over claim-ledger rows.

        Returns
        -------
        Iterator[ClaimLedgerRow]
            Rows in committed order.
        """
        return iter(self.rows)

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-ready ledger.

        Returns
        -------
        dict[str, object]
            Mapping compatible with the committed ledger schema.
        """
        return {
            **_CLAIM_LEDGER_METADATA,
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "claims": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class ClaimLedgerValidation:
    """Report a claim-ledger validation result.

    Parameters
    ----------
    passed
        Whether every checked invariant passed.
    errors
        Ordered validation errors; empty only for a passing result.
    """

    passed: bool
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate result coherence.

        Raises
        ------
        ValueError
            If the Boolean, error tuple, or pass/error relationship is invalid.
        """
        if type(self.passed) is not bool:
            raise ValueError("passed must be a bool")
        _string_tuple(self.errors, field_name="errors", allow_list=False)
        if self.passed == bool(self.errors):
            raise ValueError("passed must be true exactly when errors is empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready validation result.

        Returns
        -------
        dict[str, object]
            Serialized pass flag and ordered errors.
        """
        return {"passed": self.passed, "errors": list(self.errors)}


@dataclass(frozen=True)
class DifferentiableSupportSurfaceAlignment:
    """Report docs, API, and generated-manifest claim alignment.

    Parameters
    ----------
    passed
        Whether every alignment check passed.
    errors
        Ordered alignment errors.
    checked_claim_ids
        Claim identities included in the audit.
    checked_paths
        Repository surfaces included in the audit.
    claim_boundary
        Non-promotional interpretation boundary.
    schema
        Alignment schema identity.
    artifact_id
        Stable alignment artefact identity.
    """

    passed: bool
    errors: tuple[str, ...]
    checked_claim_ids: tuple[str, ...]
    checked_paths: tuple[str, ...]
    claim_boundary: str
    schema: str = SUPPORT_SURFACE_ALIGNMENT_SCHEMA
    artifact_id: str = SUPPORT_SURFACE_ALIGNMENT_ARTIFACT_ID

    def __post_init__(self) -> None:
        """Validate alignment identity, collections, and result coherence.

        Raises
        ------
        ValueError
            If an identity, collection, Boolean, or pass/error relationship is
            malformed.
        """
        if type(self.passed) is not bool:
            raise ValueError("passed must be a bool")
        _string_tuple(self.errors, field_name="errors", allow_list=False)
        _string_tuple(
            self.checked_claim_ids,
            field_name="checked_claim_ids",
            allow_list=False,
        )
        _string_tuple(self.checked_paths, field_name="checked_paths", allow_list=False)
        _nonblank_string(self.claim_boundary, "claim_boundary")
        if _nonblank_string(self.schema, "schema") != SUPPORT_SURFACE_ALIGNMENT_SCHEMA:
            raise ValueError(f"unknown support-surface alignment schema: {self.schema}")
        if (
            _nonblank_string(self.artifact_id, "artifact_id")
            != SUPPORT_SURFACE_ALIGNMENT_ARTIFACT_ID
        ):
            raise ValueError(f"unknown support-surface alignment artifact_id: {self.artifact_id}")
        if self.passed == bool(self.errors):
            raise ValueError("passed must be true exactly when errors is empty")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> DifferentiableSupportSurfaceAlignment:
        """Build support-surface alignment evidence from decoded JSON data.

        Parameters
        ----------
        payload
            Decoded alignment mapping.

        Returns
        -------
        DifferentiableSupportSurfaceAlignment
            Validated immutable alignment evidence.

        Raises
        ------
        ValueError
            If a required field is missing or malformed.
        """
        return cls(
            passed=_required_bool(payload, "passed", "support-surface alignment"),
            errors=_string_tuple(
                _required_value(payload, "errors", "support-surface alignment"),
                field_name="errors",
            ),
            checked_claim_ids=_string_tuple(
                _required_value(payload, "checked_claim_ids", "support-surface alignment"),
                field_name="checked_claim_ids",
            ),
            checked_paths=_string_tuple(
                _required_value(payload, "checked_paths", "support-surface alignment"),
                field_name="checked_paths",
            ),
            claim_boundary=_required_string(
                payload,
                "claim_boundary",
                "support-surface alignment",
            ),
            schema=_required_string(payload, "schema", "support-surface alignment"),
            artifact_id=_required_string(
                payload,
                "artifact_id",
                "support-surface alignment",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready support-surface alignment evidence.

        Returns
        -------
        dict[str, object]
            Mapping compatible with the committed alignment schema.
        """
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "passed": self.passed,
            "errors": list(self.errors),
            "checked_claim_ids": list(self.checked_claim_ids),
            "checked_paths": list(self.checked_paths),
            "claim_boundary": self.claim_boundary,
        }


def _mapping(value: object, context: str) -> Mapping[str, object]:
    """Return a string-keyed mapping or reject the decoded JSON value."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object")
    return cast(Mapping[str, object], value)


def _required_value(payload: Mapping[str, object], field_name: str, context: str) -> object:
    """Return a required mapping value with a descriptive missing-field error."""
    if field_name not in payload:
        raise ValueError(f"{context} is missing required field: {field_name}")
    return payload[field_name]


def _nonblank_string(value: object, field_name: str) -> str:
    """Return an exact non-blank string without coercion."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _required_string(payload: Mapping[str, object], field_name: str, context: str) -> str:
    """Return a required non-blank string from a mapping."""
    return _nonblank_string(_required_value(payload, field_name, context), field_name)


def _required_bool(payload: Mapping[str, object], field_name: str, context: str) -> bool:
    """Return a required exact Boolean from a mapping."""
    value = _required_value(payload, field_name, context)
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a bool")
    return value


def _string_tuple(
    value: object,
    *,
    field_name: str,
    allow_empty: bool = True,
    allow_list: bool = True,
) -> tuple[str, ...]:
    """Validate and return a unique tuple of non-blank strings."""
    if not isinstance(value, tuple) and not (allow_list and isinstance(value, list)):
        raise ValueError(f"{field_name} must be a list-like JSON value")
    items = tuple(value)
    if not allow_empty and not items:
        raise ValueError(f"{field_name} must contain non-empty entries")
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError(f"{field_name} must contain non-empty string entries")
    string_items = cast(tuple[str, ...], items)
    if len(set(string_items)) != len(string_items):
        raise ValueError(f"{field_name} must not contain duplicate entries")
    return string_items


def _claim_rows(
    rows_or_ledger: ClaimLedger | Iterable[ClaimLedgerRow],
) -> tuple[ClaimLedgerRow, ...]:
    """Return ledger rows while rejecting invalid runtime iterable contents."""
    rows = (
        rows_or_ledger.rows if isinstance(rows_or_ledger, ClaimLedger) else tuple(rows_or_ledger)
    )
    if any(not isinstance(row, ClaimLedgerRow) for row in rows):
        raise ValueError("claim rows must contain ClaimLedgerRow values")
    return rows


def _validation_result(errors: Iterable[str]) -> ClaimLedgerValidation:
    """Build a validation result after removing duplicate diagnostics."""
    unique_errors = tuple(dict.fromkeys(errors))
    return ClaimLedgerValidation(passed=not unique_errors, errors=unique_errors)


def load_differentiable_claim_ledger(path: Path = DEFAULT_LEDGER_PATH) -> ClaimLedger:
    """Load and validate a differentiable claim ledger.

    Parameters
    ----------
    path
        JSON ledger path.

    Returns
    -------
    ClaimLedger
        Validated ledger with ordered rows.

    Raises
    ------
    OSError
        If the ledger cannot be read.
    json.JSONDecodeError
        If the ledger is not valid JSON.
    ValueError
        If the decoded JSON violates the ledger contract.
    """
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    payload = _mapping(decoded, "claim ledger")
    for field_name, expected in _CLAIM_LEDGER_METADATA.items():
        if _required_string(payload, field_name, "claim ledger") != expected:
            raise ValueError(f"claim ledger {field_name} does not match the canonical value")
    claims = _required_value(payload, "claims", "claim ledger")
    if not isinstance(claims, list):
        raise ValueError("claim ledger claims must be a JSON array")
    return ClaimLedger(
        schema=_required_string(payload, "schema", "claim ledger"),
        artifact_id=_required_string(payload, "artifact_id", "claim ledger"),
        rows=tuple(
            ClaimLedgerRow.from_dict(_mapping(row, f"claim ledger row {index}"))
            for index, row in enumerate(claims)
        ),
    )


def load_differentiable_support_surface_alignment(
    path: Path = DEFAULT_SUPPORT_SURFACE_ALIGNMENT_PATH,
) -> DifferentiableSupportSurfaceAlignment:
    """Load and validate support-surface alignment evidence.

    Parameters
    ----------
    path
        JSON alignment-evidence path.

    Returns
    -------
    DifferentiableSupportSurfaceAlignment
        Validated alignment evidence.

    Raises
    ------
    OSError
        If the artefact cannot be read.
    json.JSONDecodeError
        If the artefact is not valid JSON.
    ValueError
        If the decoded JSON violates the alignment contract.
    """
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    payload = _mapping(decoded, "support-surface alignment")
    return DifferentiableSupportSurfaceAlignment.from_dict(payload)


def validate_claim_ledger(
    rows_or_ledger: ClaimLedger | Sequence[ClaimLedgerRow],
    *,
    artifact_statuses: Mapping[str, str] | None = None,
) -> ClaimLedgerValidation:
    """Validate claim-ledger promotion invariants.

    Parameters
    ----------
    rows_or_ledger
        Validated ledger or ordered rows to inspect.
    artifact_statuses
        Optional artefact-ID to status mapping. When supplied, every evidence
        and benchmark ID on a promoted row must map to ``"passed"``; an empty
        mapping therefore fails closed.

    Returns
    -------
    ClaimLedgerValidation
        Pass flag and deterministic errors.
    """
    rows = _claim_rows(rows_or_ledger)
    errors: list[str] = []
    if not rows:
        errors.append("claim ledger must contain at least one row")
    seen: set[str] = set()
    for row in rows:
        if row.claim_id in seen:
            errors.append(f"{row.claim_id}: duplicate claim_id")
        seen.add(row.claim_id)
        if row.promotion_status == "promoted" and not row.evidence_artifact_ids:
            errors.append(f"{row.claim_id}: promoted claims require at least one artefact ID")
        if row.promotion_status == "promoted" and not row.benchmark_artifact_ids:
            errors.append(f"{row.claim_id}: promoted claims require benchmark evidence IDs")
        if row.promotion_status == "bounded_candidate" and not row.evidence_artifact_ids:
            errors.append(f"{row.claim_id}: candidate claims require artefact ID evidence")
        if artifact_statuses is not None and row.promotion_status == "promoted":
            artifact_ids = dict.fromkeys((*row.evidence_artifact_ids, *row.benchmark_artifact_ids))
            for artifact_id in artifact_ids:
                if artifact_statuses.get(artifact_id) != "passed":
                    errors.append(f"{row.claim_id}: artefact {artifact_id} is not passed")
    return _validation_result(errors)


def validate_public_language_against_ledger(
    rows_or_ledger: ClaimLedger | Sequence[ClaimLedgerRow],
    public_texts: Iterable[str],
) -> ClaimLedgerValidation:
    """Reject promotional wording unless every ledger row is promoted.

    The validator has no category-to-text mapping, so a single promoted row
    cannot safely authorise promotional wording for a mixed ledger.

    Parameters
    ----------
    rows_or_ledger
        Ledger or rows governing the public text.
    public_texts
        Public strings to inspect case-insensitively.

    Returns
    -------
    ClaimLedgerValidation
        Pass flag and one error per banned phrase occurrence.
    """
    rows = _claim_rows(rows_or_ledger)
    if rows and all(row.promotion_status == "promoted" for row in rows):
        return _validation_result(())
    errors: list[str] = []
    for text in public_texts:
        folded = text.casefold()
        for phrase in _BANNED_PUBLIC_PHRASES:
            if phrase in folded:
                errors.append(f"public wording exceeds claim ledger: {phrase}")
    return _validation_result(errors)


def validate_differentiable_support_surface_alignment(
    rows: Sequence[ClaimLedgerRow] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    manifest_path: Path = DEFAULT_CAPABILITY_MANIFEST_PATH,
) -> DifferentiableSupportSurfaceAlignment:
    """Validate claim surfaces against the repository and generated inventory.

    Parameters
    ----------
    rows
        Optional rows to validate instead of loading the committed ledger.
    repo_root
        Repository root used for path containment and existence checks.
    ledger_path
        Ledger loaded when ``rows`` is omitted.
    manifest_path
        Generated capability manifest whose path inventory must include source,
        test, and documentation surfaces.

    Returns
    -------
    DifferentiableSupportSurfaceAlignment
        Deterministic alignment evidence. Unsafe, missing, or unregistered paths
        produce a failed result without being dereferenced outside the repository.
    """
    ledger = load_differentiable_claim_ledger(ledger_path) if rows is None else None
    claim_rows = ledger.rows if ledger is not None else _claim_rows(rows or ())
    manifest_relative = _repo_relative_path(manifest_path, repo_root)
    if manifest_relative is None:
        manifest_paths: set[str] = set()
        errors = [f"generated capability manifest is outside repository: {manifest_path}"]
        checked_paths: set[str] = set()
    else:
        manifest_paths, manifest_errors = _load_manifest_paths(manifest_path)
        errors = list(manifest_errors)
        checked_paths = {manifest_relative}
    ledger_validation = validate_claim_ledger(claim_rows)
    errors.extend(ledger_validation.errors)

    for row in claim_rows:
        for surface_name, paths in (
            ("implementation_surface", row.implementation_surface),
            ("test_surface", row.test_surface),
            ("docs_surface", row.docs_surface),
        ):
            for path in paths:
                checked_paths.add(path)
                resolved = _safe_repo_surface(repo_root, path)
                if resolved is None:
                    errors.append(
                        f"{row.claim_id}: {surface_name} path is not a safe "
                        f"repository-relative path: {path}"
                    )
                    continue
                if not resolved.exists():
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
    """Load a capability manifest and collect its repository paths."""
    if not manifest_path.is_file():
        return set(), (f"generated capability manifest is missing: {manifest_path}",)
    try:
        decoded: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return set(), (f"generated capability manifest is not valid JSON: {exc}",)
    except OSError as exc:
        return set(), (f"generated capability manifest cannot be read: {exc}",)
    try:
        payload = _mapping(decoded, "generated capability manifest")
    except ValueError as exc:
        return set(), (str(exc),)
    return _collect_manifest_paths(payload), ()


def _collect_manifest_paths(value: object) -> set[str]:
    """Collect repository-shaped strings from nested manifest data."""
    paths: set[str] = set()
    if isinstance(value, Mapping):
        for child in value.values():
            paths.update(_collect_manifest_paths(child))
    elif isinstance(value, list | tuple):
        for child in value:
            paths.update(_collect_manifest_paths(child))
    elif isinstance(value, str) and _looks_like_repo_path(value):
        paths.add(value)
    return paths


def _looks_like_repo_path(value: str) -> bool:
    """Return whether a manifest string is a safe supported repository path."""
    return (
        "/" in value
        and not value.startswith(("http://", "https://"))
        and not value.endswith("/")
        and any(value.startswith(prefix) for prefix in ("src/", "tests/", "docs/", "scripts/"))
        and _is_safe_repo_relative_path(value)
    )


def _requires_manifest_membership(path: str) -> bool:
    """Return whether a claim surface must occur in the capability manifest."""
    return path.startswith(("src/", "tests/", "docs/")) and path != "README.md"


def _is_safe_repo_relative_path(value: str) -> bool:
    """Reject absolute, ambiguous, Windows-style, and traversing paths."""
    parts = value.split("/")
    return (
        bool(value)
        and value == value.strip()
        and not value.startswith(("/", "\\"))
        and "\\" not in value
        and all(part not in {"", ".", ".."} for part in parts)
        and not parts[0].endswith(":")
    )


def _repo_relative_path(path: Path, repo_root: Path) -> str | None:
    """Return a resolved repository-relative path when containment holds."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return None


def _safe_repo_surface(repo_root: Path, relative: str) -> Path | None:
    """Resolve a safe repository-relative surface without permitting escape."""
    if not _is_safe_repo_relative_path(relative):
        return None
    candidate = (repo_root / relative).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return candidate


def validate_public_claim_table(
    rows_or_ledger: ClaimLedger | Iterable[ClaimLedgerRow],
    markdown: str,
) -> ClaimLedgerValidation:
    """Validate that a public claim table matches ledger boundaries exactly.

    Parameters
    ----------
    rows_or_ledger
        Ledger or ordered claim rows governing the table.
    markdown
        Public table text to validate.

    Returns
    -------
    ClaimLedgerValidation
        Pass flag and claim-specific drift errors.
    """
    rows = _claim_rows(rows_or_ledger)
    errors = list(validate_claim_ledger(rows).errors)
    public_language = validate_public_language_against_ledger(rows, (markdown,))
    errors.extend(public_language.errors)
    canonical = render_public_claim_table(rows)
    canonical_lines = canonical.splitlines()
    markdown_lines = markdown.splitlines()
    for row in rows:
        marker = f"`{row.claim_id}`"
        if marker not in markdown:
            errors.append(f"{row.claim_id}: missing public claim-table row")
        expected_row = next(line for line in canonical_lines if line.startswith(f"| {marker} |"))
        if markdown_lines.count(expected_row) != 1:
            errors.append(f"{row.claim_id}: public claim-table row does not exactly match ledger")
            if row.promotion_status == "promoted":
                errors.append(f"{row.claim_id}: promoted row must include exact claim boundary")
    if (
        any(row.promotion_status != "promoted" for row in rows)
        and _NON_PROMOTIONAL_BOUNDARY not in markdown
    ):
        errors.append("public claim table is missing the non-promotional boundary")
    if markdown != canonical:
        errors.append("public claim table differs from the canonical ledger rendering")
    return _validation_result(errors)


__all__ = [
    "CLAIM_LEDGER_ARTIFACT_ID",
    "CLAIM_LEDGER_SCHEMA",
    "ClaimLedger",
    "ClaimLedgerRow",
    "ClaimLedgerValidation",
    "DEFAULT_CAPABILITY_MANIFEST_PATH",
    "DEFAULT_LEDGER_PATH",
    "DEFAULT_SUPPORT_SURFACE_ALIGNMENT_PATH",
    "DifferentiableSupportSurfaceAlignment",
    "PromotionStatus",
    "SUPPORT_SURFACE_ALIGNMENT_ARTIFACT_ID",
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
