# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Claim Ledger Tests
"""Tests for differentiable evidence claim-ledger invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_quantum_control.differentiable_claim_ledger import (
    ClaimLedgerRow,
    load_differentiable_claim_ledger,
    render_claim_ledger_markdown,
    render_public_claim_table,
    validate_claim_ledger,
    validate_differentiable_support_surface_alignment,
    validate_public_claim_table,
    validate_public_language_against_ledger,
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
    } <= claim_ids
    for row in ledger.rows:
        assert row.implementation_surface
        assert row.test_surface
        assert row.docs_surface
        assert row.benchmark_artifact_ids
        assert row.known_gaps
        if row.promotion_status == "promoted":
            assert row.evidence_artifact_ids


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


def test_claim_ledger_markdown_summary_maps_rows_to_status(tmp_path: Path) -> None:
    ledger = load_differentiable_claim_ledger()
    markdown = render_claim_ledger_markdown(ledger)
    output = tmp_path / "summary.md"
    output.write_text(markdown, encoding="utf-8")

    text = output.read_text(encoding="utf-8")
    assert "| Claim | Status | Artefact IDs | Benchmark IDs | Known gaps |" in text
    assert "framework_overlay_parity" in text
    assert "SOTA-candidate" in text


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
            promotion_status="done",
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
    assert "`external_validation_environment_lock`" in markdown
    assert "bounded-candidate" in markdown
    assert "No hardware, provider, QPU, GPU, production-performance" in markdown


def test_public_claim_table_validator_rejects_missing_rows() -> None:
    ledger = load_differentiable_claim_ledger()

    validation = validate_public_claim_table(ledger, "# Differentiable Public Claim Table\n")

    assert not validation.passed
    assert any("missing public claim-table row" in error for error in validation.errors)


def test_support_surface_alignment_audit_matches_committed_manifest_and_ledger() -> None:
    alignment = validate_differentiable_support_surface_alignment()

    assert alignment.passed
    assert {
        "framework_overlay_parity",
        "ci_benchmark_evidence",
        "external_framework_comparison",
        "phase_qnode_claim_boundary",
    } <= set(alignment.checked_claim_ids)
    assert "README.md" in alignment.checked_paths
    assert "docs/differentiable_programming.md" in alignment.checked_paths
    assert "docs/_generated/capability_manifest.json" in alignment.checked_paths
    assert "support-surface alignment audit" in alignment.claim_boundary


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
        promotion_status="SOTA-candidate",
        claim_boundary="bounded",
    )

    alignment = validate_differentiable_support_surface_alignment(rows=[row])

    assert not alignment.passed
    assert any("docs/not_in_manifest.md" in error for error in alignment.errors)
