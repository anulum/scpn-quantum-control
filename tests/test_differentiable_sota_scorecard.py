# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable SOTA scorecard tests
"""Tests for differentiable state-of-art scorecard governance."""

from __future__ import annotations

from typing import get_args

from scpn_quantum_control import (
    DifferentiableSOTACategory,
    DifferentiableSOTAScorecardRow,
    differentiable_api,
    render_differentiable_sota_scorecard_markdown,
    run_differentiable_sota_scorecard,
)
from scpn_quantum_control.differentiable_sota_scorecard import (
    REQUIRED_SOTA_CATEGORIES,
    validate_differentiable_sota_scorecard,
)


def test_differentiable_sota_scorecard_records_all_required_categories() -> None:
    """The committed scorecard must cover every category from the TODO lane."""
    scorecard = run_differentiable_sota_scorecard()

    assert scorecard.schema == "scpn_qc_differentiable_sota_scorecard_v1"
    assert scorecard.promotion_ready is False
    assert scorecard.ready_category_count == 0
    assert scorecard.total_category_count == len(REQUIRED_SOTA_CATEGORIES)
    assert {row.category for row in scorecard.rows} == set(REQUIRED_SOTA_CATEGORIES)
    assert any(row.category == "catalyst_compiler_workflows" for row in scorecard.rows)
    assert any(row.category == "rust_native_program_ad" for row in scorecard.rows)
    assert "SOTA-candidate" in scorecard.claim_boundary


def test_differentiable_sota_scorecard_rows_are_claim_bounded() -> None:
    """Rows must explain why category promotion is blocked."""
    scorecard = run_differentiable_sota_scorecard()
    rows = {row.category: row for row in scorecard.rows}

    assert rows["jax_native_transforms"].status == "behind_baseline"
    assert rows["jax_native_transforms"].claim_ids == (
        "external_framework_comparison",
        "phase_qnode_claim_boundary",
    )
    assert any("isolated benchmark" in reason for reason in rows["benchmark_promotion"].blockers)
    assert "Catalyst" in rows["catalyst_compiler_workflows"].baseline
    assert rows["provider_hardware_gradients"].claim_ids == ("phase_qnode_claim_boundary",)
    assert rows["rust_native_program_ad"].status == "behind_baseline"


def test_differentiable_sota_scorecard_validation_rejects_unpromoted_ready_rows() -> None:
    """At-baseline or exceedance rows require promoted ledger evidence."""
    scorecard = run_differentiable_sota_scorecard()
    first = scorecard.rows[0]
    invalid_row = DifferentiableSOTAScorecardRow(
        category=first.category,
        baseline=first.baseline,
        current_evidence=first.current_evidence,
        status="at_baseline",
        claim_ids=first.claim_ids,
        implementation_surface=first.implementation_surface,
        test_surface=first.test_surface,
        docs_surface=first.docs_surface,
        benchmark_artifact_ids=first.benchmark_artifact_ids,
        blockers=(),
        next_hardening_rounds=first.next_hardening_rounds,
        claim_boundary=first.claim_boundary,
    )
    invalid_scorecard = type(scorecard)(
        schema=scorecard.schema,
        artifact_id=scorecard.artifact_id,
        rows=(invalid_row, *scorecard.rows[1:]),
        promotion_ready=False,
        ready_category_count=1,
        total_category_count=scorecard.total_category_count,
        claim_boundary=scorecard.claim_boundary,
    )

    validation = validate_differentiable_sota_scorecard(invalid_scorecard)

    assert not validation.passed
    assert any("requires promoted ledger rows" in error for error in validation.errors)


def test_differentiable_sota_scorecard_markdown_and_facade_dispatch() -> None:
    """The scorecard must render and dispatch through the unified API."""
    scorecard = run_differentiable_sota_scorecard()
    markdown = render_differentiable_sota_scorecard_markdown(scorecard)
    result = differentiable_api("sota_scorecard")

    assert "# Differentiable SOTA Scorecard" in markdown
    assert "catalyst_compiler_workflows" in markdown
    assert "SOTA-candidate" in markdown
    assert result.operation == "sota_scorecard"
    assert result.supported is False
    assert result.payload["promotion_ready"] is False
    assert result.payload["total_category_count"] == len(REQUIRED_SOTA_CATEGORIES)
    assert "SOTA-candidate" in result.claim_boundary


def test_differentiable_sota_scorecard_exports_are_public() -> None:
    """Top-level package exports must expose the scorecard types and runner."""
    scorecard = run_differentiable_sota_scorecard()

    assert isinstance(scorecard.rows[0].category, str)
    assert isinstance(scorecard.rows[0], DifferentiableSOTAScorecardRow)
    assert "jax_native_transforms" in get_args(DifferentiableSOTACategory)
