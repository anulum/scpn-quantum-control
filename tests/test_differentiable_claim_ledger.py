# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Claim Ledger Tests
"""Tests for differentiable evidence claim-ledger invariants."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.differentiable_claim_ledger import (
    DEFAULT_SUPPORT_SURFACE_ALIGNMENT_PATH,
    ClaimLedgerRow,
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
    ledger = load_differentiable_claim_ledger()
    validation = validate_claim_ledger(ledger)

    assert validation.passed
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
        "artifact_id": "artifact",
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
    with pytest.raises(ValueError, match="expected a list-like JSON value"):
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
