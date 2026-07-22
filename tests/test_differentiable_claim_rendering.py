# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Claim Markdown Rendering Tests
"""Tests for deterministic differentiable claim Markdown rendering."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from scpn_quantum_control.differentiable_claim_ledger import (
    ClaimLedgerRow,
    DifferentiableSupportSurfaceAlignment,
    PromotionStatus,
    load_differentiable_claim_ledger,
    load_differentiable_support_surface_alignment,
)
from scpn_quantum_control.differentiable_claim_rendering import (
    render_claim_ledger_markdown,
    render_differentiable_support_surface_alignment_markdown,
    render_public_claim_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "differentiable_phase_qnode"


def _row(
    claim_id: str,
    status: PromotionStatus,
    *,
    claim_text: str = "bounded claim",
    boundary: str = "bounded claim boundary",
    evidence: tuple[str, ...] = ("artefact-1",),
    benchmarks: tuple[str, ...] = ("benchmark-1",),
    gaps: tuple[str, ...] = ("external evidence pending",),
) -> ClaimLedgerRow:
    """Build a valid row for renderer branch and escaping tests."""
    return ClaimLedgerRow(
        claim_id=claim_id,
        claim_text=claim_text,
        implementation_surface=("src/scpn_quantum_control/differentiable_claim_rendering.py",),
        test_surface=("tests/test_differentiable_claim_rendering.py",),
        docs_surface=("docs/differentiable_programming.md",),
        evidence_artifact_ids=evidence,
        benchmark_artifact_ids=benchmarks,
        known_gaps=gaps,
        promotion_status=status,
        claim_boundary=boundary,
    )


def test_committed_markdown_artifacts_match_public_renderers_byte_for_byte() -> None:
    """All committed claim Markdown is the exact output of public renderers."""
    ledger = load_differentiable_claim_ledger()
    alignment = load_differentiable_support_surface_alignment()

    assert (DATA_ROOT / "claim_ledger.md").read_text(
        encoding="utf-8"
    ) == render_claim_ledger_markdown(ledger)
    assert (DATA_ROOT / "public_claim_table_20260616.md").read_text(
        encoding="utf-8"
    ) == render_public_claim_table(ledger)
    assert (DATA_ROOT / "differentiable_support_surface_alignment_20260627.md").read_text(
        encoding="utf-8"
    ) == render_differentiable_support_surface_alignment_markdown(alignment)


def test_claim_summary_folds_lines_and_escapes_table_delimiters() -> None:
    """Claim summaries cannot inject rows through artefact IDs or gap text."""
    row = _row(
        "escaped_claim",
        "bounded_candidate",
        evidence=("artefact|id",),
        benchmarks=(),
        gaps=("line one\nline two | bounded",),
    )

    markdown = render_claim_ledger_markdown((row,))

    assert "artefact\\|id" in markdown
    assert "line one line two \\| bounded" in markdown
    assert "| none |" in markdown
    assert render_claim_ledger_markdown(()).endswith("external comparison artefacts pass.\n")


def test_public_table_renders_every_status_and_escapes_promoted_content() -> None:
    """Public rows cover each status while containing text and code-span controls."""
    promoted = _row(
        "promoted_claim",
        "promoted",
        claim_text="Promoted | claim\nnext",
        boundary="`boundary` | exact",
        evidence=("`artefact`|id",),
    )
    rows = (
        promoted,
        _row("candidate_claim", "bounded_candidate"),
        _row("gap_claim", "hard_gap", evidence=(), benchmarks=()),
        _row("blocked_claim", "blocked", evidence=(), benchmarks=()),
    )

    markdown = render_public_claim_table(rows)
    promoted_line = next(
        line for line in markdown.splitlines() if line.startswith("| `promoted_claim` |")
    )

    assert "Promoted \\| claim next" in promoted_line
    assert "`boundary` \\| exact" in promoted_line
    assert "`` `artefact`\\|id ``" in promoted_line
    assert markdown.count("| `blocked` |") == 2
    assert "| `bounded-candidate` |" in markdown
    assert "Candidate statement (not promoted): bounded claim" in markdown
    assert "Hard-gap statement (not promoted): bounded claim" in markdown
    assert "Blocked statement (not promoted): bounded claim" in markdown
    assert "Claim boundary: bounded claim boundary" in markdown
    assert "Open gaps: external evidence pending" in markdown
    assert "Artefacts: none; benchmark IDs: none." in markdown
    assert "Global boundary:" in render_public_claim_table(())


def test_public_table_rejects_unknown_status_instead_of_downgrading_it() -> None:
    """Unknown internal states cannot silently appear as bounded candidates."""
    row = SimpleNamespace(
        claim_id="unknown_status",
        claim_text="bounded claim",
        evidence_artifact_ids=("artefact-1",),
        benchmark_artifact_ids=("benchmark-1",),
        known_gaps=("external evidence pending",),
        promotion_status=cast(PromotionStatus, "unknown"),
        claim_boundary="bounded claim boundary",
    )

    with pytest.raises(ValueError, match="unknown promotion status: 'unknown'"):
        render_public_claim_table((row,))


def test_alignment_renderer_contains_code_spans_and_error_cells() -> None:
    """Alignment identities use safe code spans and diagnostics stay in one cell."""
    alignment = DifferentiableSupportSurfaceAlignment(
        passed=False,
        errors=("line one\nline two | error",),
        checked_claim_ids=("`claim`",),
        checked_paths=("docs/path|name.md",),
        claim_boundary="support | alignment",
    )

    markdown = render_differentiable_support_surface_alignment_markdown(alignment)

    assert "| `` `claim` `` |" in markdown
    assert "docs/path\\|name.md" in markdown
    assert "line one line two \\| error" in markdown
    assert "support \\| alignment" in markdown
    assert "## Errors" in markdown
