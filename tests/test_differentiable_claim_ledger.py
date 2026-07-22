# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Claim Ledger Tests
"""Tests for differentiable evidence claim-ledger invariants."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.differentiable_claim_ledger import (
    CLAIM_LEDGER_ARTIFACT_ID,
    CLAIM_LEDGER_SCHEMA,
    DEFAULT_LEDGER_PATH,
    DEFAULT_SUPPORT_SURFACE_ALIGNMENT_PATH,
    SUPPORT_SURFACE_ALIGNMENT_ARTIFACT_ID,
    ClaimLedger,
    ClaimLedgerRow,
    ClaimLedgerValidation,
    DifferentiableSupportSurfaceAlignment,
    PromotionStatus,
    load_differentiable_claim_ledger,
    load_differentiable_support_surface_alignment,
    render_claim_ledger_markdown,
    render_differentiable_support_surface_alignment_markdown,
    render_public_claim_table,
    validate_claim_ledger,
    validate_differentiable_support_surface_alignment,
    validate_public_claim_table,
    validate_public_language_against_ledger,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _valid_claim_row(
    claim_id: str = "claim",
    *,
    promotion_status: PromotionStatus = "bounded_candidate",
    evidence_artifact_ids: tuple[str, ...] = ("artefact-1",),
    benchmark_artifact_ids: tuple[str, ...] = ("benchmark-1",),
    claim_boundary: str = "bounded claim boundary",
) -> ClaimLedgerRow:
    """Build a minimally valid claim-ledger row for validation-edge tests."""
    return ClaimLedgerRow(
        claim_id=claim_id,
        claim_text=f"{claim_id} bounded claim",
        implementation_surface=("src/scpn_quantum_control/differentiable_claim_ledger.py",),
        test_surface=("tests/test_differentiable_claim_ledger.py",),
        docs_surface=("docs/differentiable_api.md",),
        evidence_artifact_ids=evidence_artifact_ids,
        benchmark_artifact_ids=benchmark_artifact_ids,
        known_gaps=("external evidence pending",),
        promotion_status=promotion_status,
        claim_boundary=claim_boundary,
    )


def test_committed_claim_ledger_has_required_rows_and_artefact_ids() -> None:
    """The committed ledger round-trips through its strict public schema."""
    ledger = load_differentiable_claim_ledger()
    validation = validate_claim_ledger(ledger)

    assert validation.passed
    assert ledger.schema == CLAIM_LEDGER_SCHEMA
    assert ledger.artifact_id == CLAIM_LEDGER_ARTIFACT_ID
    assert ledger.to_dict() == json.loads(DEFAULT_LEDGER_PATH.read_text(encoding="utf-8"))
    claim_ids = {row.claim_id for row in ledger.rows}
    assert {
        "framework_overlay_parity",
        "ci_benchmark_evidence",
        "external_framework_comparison",
        "phase_qnode_claim_boundary",
        "differentiable_architecture_rustification_map",
        "differentiable_dependency_environment_map",
        "differentiable_isolated_benchmark_plan",
    } <= claim_ids
    for row in ledger.rows:
        assert row.implementation_surface
        assert row.test_surface
        assert row.docs_surface
        assert row.benchmark_artifact_ids
        assert row.known_gaps
        if row.promotion_status == "promoted":
            assert row.evidence_artifact_ids


def test_claim_ledger_row_rejects_empty_required_fields() -> None:
    """Claim rows reject empty identities, surfaces, and boundaries."""
    row = _valid_claim_row()

    with pytest.raises(ValueError, match="claim_id must be non-empty"):
        replace(row, claim_id="")
    with pytest.raises(ValueError, match="claim_text must be non-empty"):
        replace(row, claim_text="")
    with pytest.raises(ValueError, match="implementation_surface must contain non-empty entries"):
        replace(row, implementation_surface=())
    with pytest.raises(ValueError, match="claim_boundary must be non-empty"):
        replace(row, claim_boundary="")


def test_claim_ledger_iterates_rows() -> None:
    """The ledger object exposes its rows through iteration."""
    ledger = load_differentiable_claim_ledger()

    assert tuple(iter(ledger)) == ledger.rows


def test_claim_ledger_rejects_promoted_row_without_artefact_id() -> None:
    row = ClaimLedgerRow(
        claim_id="bad_claim",
        claim_text="Unsupported promoted claim",
        implementation_surface=("src/scpn_quantum_control/example.py",),
        test_surface=("tests/test_example.py",),
        docs_surface=("docs/example.md",),
        evidence_artifact_ids=(),
        benchmark_artifact_ids=(),
        known_gaps=("none recorded",),
        promotion_status="promoted",
        claim_boundary="must not promote without evidence",
    )

    validation = validate_claim_ledger([row])

    assert not validation.passed
    assert "bad_claim" in validation.errors[0]
    assert "artefact ID" in validation.errors[0]


def test_claim_ledger_rejects_promoted_row_without_benchmark_id() -> None:
    """Promoted claims must name benchmark evidence IDs."""
    row = _valid_claim_row(
        "bad_claim",
        promotion_status="promoted",
        benchmark_artifact_ids=(),
    )

    validation = validate_claim_ledger([row])

    assert not validation.passed
    assert "benchmark evidence IDs" in validation.errors[0]


def test_claim_ledger_validation_reports_duplicates_and_candidate_without_evidence() -> None:
    """Duplicate claim IDs and candidate rows without artefacts are rejected."""
    row = _valid_claim_row("duplicate", evidence_artifact_ids=())

    validation = validate_claim_ledger([row, row])

    assert not validation.passed
    assert any("duplicate claim_id" in error for error in validation.errors)
    assert any(
        "candidate claims require artefact ID evidence" in error for error in validation.errors
    )


def test_claim_ledger_rejects_promoted_row_without_passing_artefacts() -> None:
    row = ClaimLedgerRow(
        claim_id="bad_claim",
        claim_text="Unsupported promoted claim",
        implementation_surface=("src/scpn_quantum_control/example.py",),
        test_surface=("tests/test_example.py",),
        docs_surface=("docs/example.md",),
        evidence_artifact_ids=("missing",),
        benchmark_artifact_ids=("missing",),
        known_gaps=("none recorded",),
        promotion_status="promoted",
        claim_boundary="must not promote without evidence",
    )

    validation = validate_claim_ledger([row], artifact_statuses={"missing": "failed"})

    assert not validation.passed
    assert "not passed" in validation.errors[0]


def test_public_language_allows_promoted_ledger() -> None:
    """Once a ledger has promoted evidence, public-language validation delegates to it."""
    row = _valid_claim_row("promoted_claim", promotion_status="promoted")

    validation = validate_public_language_against_ledger(
        [row],
        ("This is a world-leading differentiable quantum control claim.",),
    )

    assert validation.passed
    assert validation.errors == ()


def test_claim_ledger_markdown_summary_maps_rows_to_status(tmp_path: Path) -> None:
    ledger = load_differentiable_claim_ledger()
    markdown = render_claim_ledger_markdown(ledger)
    output = tmp_path / "summary.md"
    output.write_text(markdown, encoding="utf-8")

    text = output.read_text(encoding="utf-8")
    assert "| Claim | Status | Artefact IDs | Benchmark IDs | Known gaps |" in text
    assert "framework_overlay_parity" in text
    assert "differentiable_architecture_rustification_map" in text
    assert "differentiable_dependency_environment_map" in text
    assert "differentiable_isolated_benchmark_plan" in text
    assert "bounded_candidate" in text


def test_support_surface_alignment_from_dict_edges() -> None:
    """Support-surface alignment loader validates schema and list-like fields."""
    payload = {
        "schema": "scpn_qc_differentiable_support_surface_alignment_v1",
        "artifact_id": SUPPORT_SURFACE_ALIGNMENT_ARTIFACT_ID,
        "passed": True,
        "errors": [],
        "checked_claim_ids": ["claim"],
        "checked_paths": ["docs/differentiable_api.md"],
        "claim_boundary": "support-surface alignment audit only",
    }

    alignment = DifferentiableSupportSurfaceAlignment.from_dict(payload)

    assert alignment.to_dict()["checked_claim_ids"] == ["claim"]
    with pytest.raises(ValueError, match="unknown support-surface alignment schema"):
        DifferentiableSupportSurfaceAlignment.from_dict({**payload, "schema": "bad.v1"})
    with pytest.raises(ValueError, match="errors must be a list-like JSON value"):
        DifferentiableSupportSurfaceAlignment.from_dict({**payload, "errors": "bad"})


def test_claim_ledger_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="promotion_status"):
        ClaimLedgerRow(
            claim_id="unknown",
            claim_text="Unknown status",
            implementation_surface=("src/x.py",),
            test_surface=("tests/test_x.py",),
            docs_surface=("docs/x.md",),
            evidence_artifact_ids=("artefact-1",),
            benchmark_artifact_ids=("artefact-1",),
            known_gaps=("none",),
            promotion_status=cast(PromotionStatus, "done"),
            claim_boundary="bounded",
        )


def test_public_language_cannot_exceed_unpromoted_ledger() -> None:
    ledger = load_differentiable_claim_ledger()

    validation = validate_public_language_against_ledger(
        ledger,
        ("This is a world-leading differentiable quantum control claim.",),
    )

    assert not validation.passed
    assert "world-leading" in validation.errors[0]


def test_public_claim_table_is_generated_from_committed_ledger() -> None:
    ledger = load_differentiable_claim_ledger()
    markdown = render_public_claim_table(ledger)
    validation = validate_public_claim_table(ledger, markdown)

    assert validation.passed
    assert "# Differentiable Public Claim Table" in markdown
    assert "`framework_overlay_parity`" in markdown
    assert "`differentiable_architecture_rustification_map`" in markdown
    assert "`differentiable_dependency_environment_map`" in markdown
    assert "`differentiable_isolated_benchmark_plan`" in markdown
    assert "`external_validation_environment_lock`" in markdown
    assert "bounded-candidate" in markdown
    assert "No hardware, provider, QPU, GPU, production-performance" in markdown
    for row in ledger:
        assert row.claim_text in markdown
        assert row.claim_boundary in markdown
        assert all(gap in markdown for gap in row.known_gaps)


def test_public_claim_table_validator_rejects_missing_rows() -> None:
    ledger = load_differentiable_claim_ledger()

    validation = validate_public_claim_table(ledger, "# Differentiable Public Claim Table\n")

    assert not validation.passed
    assert any("missing public claim-table row" in error for error in validation.errors)


def test_public_claim_table_handles_promoted_and_hard_gap_rows() -> None:
    """Public claim tables render promoted and hard-gap status branches."""
    promoted = _valid_claim_row(
        "promoted_claim",
        promotion_status="promoted",
        claim_boundary="exact promoted boundary",
    )
    hard_gap = _valid_claim_row("gap_claim", promotion_status="hard_gap")
    rows = (promoted, hard_gap)
    markdown = render_public_claim_table(rows)

    assert "`promoted`" in markdown
    assert "`blocked`" in markdown
    assert "exact promoted boundary" in markdown
    assert "Hard-gap statement (not promoted)" in markdown
    assert hard_gap.claim_text in markdown
    assert hard_gap.claim_boundary in markdown
    assert all(gap in markdown for gap in hard_gap.known_gaps)
    assert validate_public_claim_table(rows, markdown).passed

    invalid = validate_public_claim_table(
        rows,
        markdown.replace("exact promoted boundary", "missing promoted boundary"),
    )

    assert not invalid.passed
    assert any(
        "promoted row must include exact claim boundary" in error for error in invalid.errors
    )


def test_support_surface_alignment_audit_matches_committed_manifest_and_ledger() -> None:
    alignment = validate_differentiable_support_surface_alignment()

    assert alignment.passed
    assert {
        "framework_overlay_parity",
        "ci_benchmark_evidence",
        "external_framework_comparison",
        "phase_qnode_claim_boundary",
        "differentiable_architecture_rustification_map",
        "differentiable_dependency_environment_map",
        "differentiable_isolated_benchmark_plan",
    } <= set(alignment.checked_claim_ids)
    assert "README.md" in alignment.checked_paths
    assert "docs/differentiable_programming.md" in alignment.checked_paths
    assert "docs/_generated/capability_manifest.json" in alignment.checked_paths
    assert "support-surface alignment audit" in alignment.claim_boundary


def test_committed_support_surface_alignment_artifact_matches_rerun() -> None:
    committed = load_differentiable_support_surface_alignment()
    rerun = validate_differentiable_support_surface_alignment()

    assert DEFAULT_SUPPORT_SURFACE_ALIGNMENT_PATH.exists()
    assert committed == rerun
    assert committed.passed
    assert "differentiable_baseline_scorecard" in committed.checked_claim_ids
    assert "differentiable_rust_python_inventory" in committed.checked_claim_ids
    assert "data/differentiable_phase_qnode/claim_ledger.md" in committed.checked_paths


def test_support_surface_alignment_markdown_lists_claim_boundary() -> None:
    alignment = load_differentiable_support_surface_alignment()
    markdown = render_differentiable_support_surface_alignment_markdown(alignment)

    assert "# Differentiable Support-Surface Alignment" in markdown
    assert "`passed`: `True`" in markdown
    assert "support-surface alignment audit" in markdown
    assert "differentiable_dependency_environment_map" in markdown
    assert "docs/_generated/capability_manifest.json" in markdown


def test_support_surface_alignment_audit_rejects_missing_manifest_path() -> None:
    row = ClaimLedgerRow(
        claim_id="missing_manifest",
        claim_text="A bounded differentiable surface that is not in the manifest.",
        implementation_surface=("src/scpn_quantum_control/differentiable_claim_ledger.py",),
        test_surface=("tests/test_differentiable_claim_ledger.py",),
        docs_surface=("docs/not_in_manifest.md",),
        evidence_artifact_ids=("artifact",),
        benchmark_artifact_ids=("artifact",),
        known_gaps=("none",),
        promotion_status="bounded_candidate",
        claim_boundary="bounded",
    )

    alignment = validate_differentiable_support_surface_alignment(rows=[row])

    assert not alignment.passed
    assert any("docs/not_in_manifest.md" in error for error in alignment.errors)


def test_support_surface_alignment_audit_reports_missing_and_invalid_manifest(
    tmp_path: Path,
) -> None:
    """Missing or invalid generated manifests are reported as audit errors."""
    missing = tmp_path / "docs" / "_generated" / "capability_manifest.json"
    missing.parent.mkdir(parents=True)

    missing_alignment = validate_differentiable_support_surface_alignment(
        rows=[],
        repo_root=tmp_path,
        manifest_path=missing,
    )
    assert not missing_alignment.passed
    assert any(
        "generated capability manifest is missing" in error for error in missing_alignment.errors
    )

    missing.write_text("{not json", encoding="utf-8")
    invalid_alignment = validate_differentiable_support_surface_alignment(
        rows=[],
        repo_root=tmp_path,
        manifest_path=missing,
    )
    assert not invalid_alignment.passed
    assert any("not valid JSON" in error for error in invalid_alignment.errors)


def test_support_surface_alignment_markdown_renders_errors() -> None:
    """Support-surface alignment markdown includes audit errors when present."""
    alignment = DifferentiableSupportSurfaceAlignment(
        passed=False,
        errors=("bad path",),
        checked_claim_ids=("claim",),
        checked_paths=("docs/missing.md",),
        claim_boundary="support-surface alignment audit only",
    )

    markdown = render_differentiable_support_surface_alignment_markdown(alignment)

    assert "## Errors" in markdown
    assert "bad path" in markdown


def test_claim_row_construction_rejects_runtime_type_and_sequence_drift() -> None:
    """Direct row construction rejects coercion, blanks, lists, and duplicates."""
    row = _valid_claim_row()

    with pytest.raises(ValueError, match="claim_id must be non-empty"):
        replace(row, claim_id=cast(str, 7))
    with pytest.raises(ValueError, match="claim_id must use lower snake or kebab case"):
        replace(row, claim_id="Claim-ID")
    with pytest.raises(ValueError, match="evidence_artifact_ids must be a list-like"):
        replace(
            row,
            evidence_artifact_ids=cast(tuple[str, ...], ["artefact-1"]),
        )
    with pytest.raises(ValueError, match="known_gaps must contain non-empty string entries"):
        replace(row, known_gaps=(" ",))
    with pytest.raises(ValueError, match="test_surface must contain non-empty string entries"):
        replace(row, test_surface=(cast(str, 3),))
    with pytest.raises(ValueError, match="docs_surface must not contain duplicate entries"):
        replace(row, docs_surface=("docs/a.md", "docs/a.md"))


def test_claim_row_json_parser_preserves_explicit_empty_benchmark_ids() -> None:
    """An explicit empty benchmark list never falls back to evidence IDs."""
    payload = _valid_claim_row().to_dict()
    payload["benchmark_artifact_ids"] = []

    explicit_empty = ClaimLedgerRow.from_dict(payload)
    del payload["benchmark_artifact_ids"]
    legacy_fallback = ClaimLedgerRow.from_dict(payload)

    assert explicit_empty.benchmark_artifact_ids == ()
    assert legacy_fallback.benchmark_artifact_ids == explicit_empty.evidence_artifact_ids
    payload["claim_id"] = 7
    with pytest.raises(ValueError, match="claim_id must be non-empty"):
        ClaimLedgerRow.from_dict(payload)
    payload["claim_id"] = "claim"
    payload["evidence_artifact_ids"] = "artefact-1"
    with pytest.raises(ValueError, match="evidence_artifact_ids must be a list-like"):
        ClaimLedgerRow.from_dict(payload)


def test_claim_ledger_constructor_rejects_identity_row_and_uniqueness_drift() -> None:
    """Ledger construction locks schema, artefact identity, rows, and claim IDs."""
    row = _valid_claim_row()

    with pytest.raises(ValueError, match="unknown claim-ledger schema"):
        ClaimLedger(schema="bad.v1", artifact_id=CLAIM_LEDGER_ARTIFACT_ID, rows=(row,))
    with pytest.raises(ValueError, match="unknown claim-ledger artifact_id"):
        ClaimLedger(schema=CLAIM_LEDGER_SCHEMA, artifact_id="bad", rows=(row,))
    with pytest.raises(ValueError, match="non-empty tuple"):
        ClaimLedger(schema=CLAIM_LEDGER_SCHEMA, artifact_id=CLAIM_LEDGER_ARTIFACT_ID, rows=())
    with pytest.raises(ValueError, match="non-empty tuple"):
        ClaimLedger(
            schema=CLAIM_LEDGER_SCHEMA,
            artifact_id=CLAIM_LEDGER_ARTIFACT_ID,
            rows=cast(tuple[ClaimLedgerRow, ...], [row]),
        )
    with pytest.raises(ValueError, match="ClaimLedgerRow values"):
        ClaimLedger(
            schema=CLAIM_LEDGER_SCHEMA,
            artifact_id=CLAIM_LEDGER_ARTIFACT_ID,
            rows=(cast(ClaimLedgerRow, object()),),
        )
    with pytest.raises(ValueError, match="duplicate claim_id"):
        ClaimLedger(
            schema=CLAIM_LEDGER_SCHEMA,
            artifact_id=CLAIM_LEDGER_ARTIFACT_ID,
            rows=(row, row),
        )


def test_claim_ledger_loader_rejects_malformed_top_level_and_row_shapes(
    tmp_path: Path,
) -> None:
    """The file loader reports malformed JSON shapes through the public API."""
    path = tmp_path / "claim-ledger.json"
    committed = cast(
        dict[str, object],
        json.loads(DEFAULT_LEDGER_PATH.read_text(encoding="utf-8")),
    )
    missing_schema = dict(committed)
    del missing_schema["schema"]
    wrong_claims = dict(committed)
    wrong_claims["claims"] = {}
    scalar_row = dict(committed)
    scalar_row["claims"] = [3]
    empty_rows = dict(committed)
    empty_rows["claims"] = []
    stale_title = dict(committed)
    stale_title["title"] = "stale"
    malformed: tuple[tuple[object, str], ...] = (
        ([], "claim ledger must be a JSON object"),
        (missing_schema, "missing required field"),
        (wrong_claims, "claims must be a JSON array"),
        (scalar_row, "claim ledger row 0 must be a JSON object"),
        (empty_rows, "non-empty tuple"),
        (stale_title, "title does not match the canonical value"),
    )

    for payload, message in malformed:
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_differentiable_claim_ledger(path)


def test_claim_validation_result_serialization_and_coherence() -> None:
    """Validation results serialize deterministically and reject contradictions."""
    passed = validate_claim_ledger([_valid_claim_row()])

    assert passed.to_dict() == {"passed": True, "errors": []}
    with pytest.raises(ValueError, match="passed must be a bool"):
        ClaimLedgerValidation(passed=cast(bool, 1), errors=())
    with pytest.raises(ValueError, match="true exactly when errors is empty"):
        ClaimLedgerValidation(passed=True, errors=("error",))
    with pytest.raises(ValueError, match="true exactly when errors is empty"):
        ClaimLedgerValidation(passed=False, errors=())
    with pytest.raises(ValueError, match="errors must not contain duplicate entries"):
        ClaimLedgerValidation(passed=False, errors=("error", "error"))
    with pytest.raises(ValueError, match="ClaimLedgerRow values"):
        validate_claim_ledger(cast(list[ClaimLedgerRow], [object()]))


def test_promoted_claim_status_map_fails_closed_when_empty_or_incomplete() -> None:
    """Supplied artefact statuses cover evidence and benchmark IDs without truthiness gaps."""
    row = _valid_claim_row("promoted", promotion_status="promoted")

    empty = validate_claim_ledger([row], artifact_statuses={})
    evidence_only = validate_claim_ledger(
        [row],
        artifact_statuses={"artefact-1": "passed"},
    )
    passed = validate_claim_ledger(
        [row],
        artifact_statuses={"artefact-1": "passed", "benchmark-1": "passed"},
    )

    assert not empty.passed
    assert {"artefact-1", "benchmark-1"} <= {
        error.split("artefact ", 1)[1].split(" is not passed", 1)[0] for error in empty.errors
    }
    assert not evidence_only.passed
    assert any("benchmark-1" in error for error in evidence_only.errors)
    assert passed.passed
    assert validate_claim_ledger([]).errors == ("claim ledger must contain at least one row",)


def test_public_language_requires_every_row_promoted_and_is_case_insensitive() -> None:
    """One promoted row or a candidate marker cannot bypass banned public wording."""
    promoted = _valid_claim_row("promoted", promotion_status="promoted")
    candidate = _valid_claim_row("candidate")
    text = "STATE-OF-THE-ART bounded_candidate production performance"

    validation = validate_public_language_against_ledger(
        (promoted, candidate),
        (text, text),
    )

    assert not validation.passed
    assert validation.errors == (
        "public wording exceeds claim ledger: state-of-the-art",
        "public wording exceeds claim ledger: production performance",
    )


def test_support_alignment_parser_rejects_boolean_identity_and_result_drift() -> None:
    """Alignment evidence rejects truthy strings, stale IDs, and incoherent results."""
    payload: dict[str, object] = {
        "schema": "scpn_qc_differentiable_support_surface_alignment_v1",
        "artifact_id": SUPPORT_SURFACE_ALIGNMENT_ARTIFACT_ID,
        "passed": True,
        "errors": [],
        "checked_claim_ids": ["claim"],
        "checked_paths": ["docs/differentiable_api.md"],
        "claim_boundary": "support-surface alignment audit only",
    }

    with pytest.raises(ValueError, match="passed must be a bool"):
        DifferentiableSupportSurfaceAlignment.from_dict({**payload, "passed": "false"})
    with pytest.raises(ValueError, match="passed must be a bool"):
        DifferentiableSupportSurfaceAlignment(
            passed=cast(bool, 1),
            errors=(),
            checked_claim_ids=("claim",),
            checked_paths=("docs/differentiable_api.md",),
            claim_boundary="support-surface alignment audit only",
        )
    with pytest.raises(ValueError, match="unknown support-surface alignment artifact_id"):
        DifferentiableSupportSurfaceAlignment.from_dict({**payload, "artifact_id": "stale"})
    with pytest.raises(ValueError, match="true exactly when errors is empty"):
        DifferentiableSupportSurfaceAlignment.from_dict({**payload, "passed": False, "errors": []})
    with pytest.raises(ValueError, match="missing required field: artifact_id"):
        without_artifact = dict(payload)
        del without_artifact["artifact_id"]
        DifferentiableSupportSurfaceAlignment.from_dict(without_artifact)


def test_support_alignment_rejects_unsafe_and_symlink_escape_paths(tmp_path: Path) -> None:
    """Alignment validation never dereferences unsafe or escaping claim surfaces."""
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = repo / "manifest.json"
    manifest.write_text('{"paths": []}', encoding="utf-8")
    base = _valid_claim_row("unsafe")

    for unsafe in ("../outside.md", "/absolute.md", "docs\\windows.md", "docs//gap.md"):
        alignment = validate_differentiable_support_surface_alignment(
            rows=[replace(base, docs_surface=(unsafe,))],
            repo_root=repo,
            manifest_path=manifest,
        )
        assert any("not a safe repository-relative path" in error for error in alignment.errors)

    external = tmp_path / "external"
    external.mkdir()
    (external / "surface.py").write_text("pass\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "link").symlink_to(external, target_is_directory=True)
    alignment = validate_differentiable_support_surface_alignment(
        rows=[replace(base, implementation_surface=("src/link/surface.py",))],
        repo_root=repo,
        manifest_path=manifest,
    )

    assert any("not a safe repository-relative path" in error for error in alignment.errors)


def test_support_alignment_rejects_outside_and_scalar_manifests(tmp_path: Path) -> None:
    """Capability manifests must be contained JSON objects, not external scalars."""
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    outside_result = validate_differentiable_support_surface_alignment(
        rows=[_valid_claim_row()],
        repo_root=repo,
        manifest_path=outside,
    )
    assert any("manifest is outside repository" in error for error in outside_result.errors)

    manifest = repo / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    scalar_result = validate_differentiable_support_surface_alignment(
        rows=[_valid_claim_row()],
        repo_root=repo,
        manifest_path=manifest,
    )
    assert any("manifest must be a JSON object" in error for error in scalar_result.errors)

    manifest.write_text("{}", encoding="utf-8")
    manifest.chmod(0)
    try:
        unreadable_result = validate_differentiable_support_surface_alignment(
            rows=[_valid_claim_row()],
            repo_root=repo,
            manifest_path=manifest,
        )
    finally:
        manifest.chmod(0o600)
    assert any("manifest cannot be read" in error for error in unreadable_result.errors)


def test_support_alignment_loader_rejects_non_object_json(tmp_path: Path) -> None:
    """The alignment file loader requires a top-level JSON object."""
    path = tmp_path / "alignment.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="support-surface alignment must be a JSON object"):
        load_differentiable_support_surface_alignment(path)


def test_public_claim_table_validation_rejects_duplicate_and_header_drift() -> None:
    """The validator requires one canonical row and the complete generated table."""
    row = _valid_claim_row()
    canonical = render_public_claim_table((row,))
    duplicated = canonical.replace(
        "\n\nGlobal boundary:",
        f"\n{canonical.splitlines()[17]}\n\nGlobal boundary:",
    )

    duplicate_validation = validate_public_claim_table((row,), duplicated)
    header_validation = validate_public_claim_table(
        (row,),
        canonical.replace("# Differentiable Public Claim Table", "# Claims"),
    )

    assert not duplicate_validation.passed
    assert any("does not exactly match ledger" in error for error in duplicate_validation.errors)
    assert not header_validation.passed
    assert any(
        "differs from the canonical ledger rendering" in error
        for error in header_validation.errors
    )
