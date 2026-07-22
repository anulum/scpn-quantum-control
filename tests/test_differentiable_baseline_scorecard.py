# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable baseline scorecard tests
"""Tests for differentiable baseline scorecard governance."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast, get_args

import pytest

from scpn_quantum_control import (
    DifferentiableBaselineCategory,
    DifferentiableBaselineScorecardRow,
    audit_differentiable_promotion_language,
    differentiable_api,
    render_differentiable_baseline_scorecard_markdown,
    run_differentiable_baseline_scorecard,
)
from scpn_quantum_control.differentiable_baseline_scorecard import (
    REQUIRED_BASELINE_CATEGORIES,
    DifferentiableBaselineStatus,
    validate_differentiable_baseline_scorecard,
)
from scpn_quantum_control.differentiable_baseline_scorecard import (
    DifferentiableBaselineCategory as BaselineCategory,
)
from scpn_quantum_control.differentiable_claim_ledger import (
    ClaimLedger,
    load_differentiable_claim_ledger,
)


def test_differentiable_baseline_scorecard_records_all_required_categories() -> None:
    """The committed scorecard must cover every category from the TODO lane."""
    scorecard = run_differentiable_baseline_scorecard()

    assert scorecard.schema == "scpn_qc_differentiable_baseline_scorecard_v1"
    assert scorecard.promotion_ready is False
    assert scorecard.ready_category_count == 0
    assert scorecard.total_category_count == len(REQUIRED_BASELINE_CATEGORIES)
    assert {row.category for row in scorecard.rows} == set(REQUIRED_BASELINE_CATEGORIES)
    assert any(row.category == "catalyst_compiler_workflows" for row in scorecard.rows)
    assert any(row.category == "rust_native_program_ad" for row in scorecard.rows)
    assert "promotion candidate" in scorecard.claim_boundary


def test_differentiable_baseline_scorecard_rows_are_claim_bounded() -> None:
    """Rows must explain why category promotion is blocked."""
    scorecard = run_differentiable_baseline_scorecard()
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


def test_differentiable_baseline_scorecard_row_rejects_invalid_fields() -> None:
    """Row construction rejects unknown vocabulary and empty evidence fields."""
    row = run_differentiable_baseline_scorecard().rows[0]

    with pytest.raises(ValueError, match="unknown baseline category"):
        replace(row, category=cast(BaselineCategory, "unknown_category"))
    with pytest.raises(ValueError, match="unknown baseline status"):
        replace(row, status=cast(DifferentiableBaselineStatus, "done"))
    with pytest.raises(ValueError, match="baseline must be non-empty"):
        replace(row, baseline=" ")
    with pytest.raises(ValueError, match="baseline must be non-empty"):
        replace(row, baseline=cast(str, 1))
    with pytest.raises(ValueError, match="claim_ids must contain non-empty entries"):
        replace(row, claim_ids=())
    with pytest.raises(ValueError, match="claim_ids must contain non-empty entries"):
        replace(row, claim_ids=cast(tuple[str, ...], ["claim"]))
    with pytest.raises(ValueError, match="claim_ids must contain non-empty entries"):
        replace(row, claim_ids=(cast(str, 1),))
    with pytest.raises(ValueError, match="blockers must contain non-empty entries"):
        replace(row, blockers=(cast(str, 1),))


def test_differentiable_baseline_scorecard_row_rejects_inconsistent_blockers() -> None:
    """Ready rows cannot carry blockers, and behind-baseline rows must list them."""
    row = run_differentiable_baseline_scorecard().rows[0]

    with pytest.raises(ValueError, match="ready scorecard rows must not carry blockers"):
        replace(row, status="at_baseline")
    with pytest.raises(ValueError, match="behind-baseline scorecard rows must list blockers"):
        replace(row, blockers=())
    with pytest.raises(ValueError, match="claim_ids must contain unique values"):
        replace(row, claim_ids=(row.claim_ids[0], row.claim_ids[0]))
    with pytest.raises(ValueError, match="blockers must contain unique values"):
        replace(row, blockers=(row.blockers[0], row.blockers[0]))
    with pytest.raises(ValueError, match="claim_boundary is not canonical"):
        replace(row, claim_boundary="not a promotion")


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"schema": " "}, "schema must be non-empty"),
        ({"artifact_id": 1}, "artifact_id must be non-empty"),
        ({"rows": []}, "rows must be a non-empty row tuple"),
        ({"rows": ()}, "rows must be a non-empty row tuple"),
        ({"rows": ("row",)}, "rows must be a non-empty row tuple"),
        ({"promotion_ready": 1}, "promotion_ready must be boolean"),
        ({"ready_category_count": True}, "must be a non-negative integer"),
        ({"total_category_count": -1}, "must be a non-negative integer"),
        ({"claim_boundary": "candidate"}, "claim_boundary is not canonical"),
    ),
)
def test_differentiable_baseline_scorecard_rejects_structural_drift(
    changes: dict[str, object],
    message: str,
) -> None:
    """Scorecard construction must reject coercion and malformed row containers."""
    scorecard = run_differentiable_baseline_scorecard()

    with pytest.raises(ValueError, match=message):
        replace(scorecard, **changes)

    with pytest.raises(ValueError, match="categories must contain unique values"):
        replace(scorecard, rows=(scorecard.rows[0], scorecard.rows[0]))


def test_differentiable_baseline_scorecard_validation_rejects_unpromoted_ready_rows() -> None:
    """At-baseline or exceedance rows require promoted ledger evidence."""
    scorecard = run_differentiable_baseline_scorecard()
    first = scorecard.rows[0]
    invalid_row = DifferentiableBaselineScorecardRow(
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

    validation = validate_differentiable_baseline_scorecard(invalid_scorecard)

    assert not validation.passed
    assert any("requires promoted ledger rows" in error for error in validation.errors)


def test_differentiable_baseline_scorecard_validation_reports_metadata_errors(
    tmp_path: Path,
) -> None:
    """Scorecard validation reports schema, count, ordering, and path drift."""
    scorecard = run_differentiable_baseline_scorecard()
    first = replace(
        scorecard.rows[0],
        claim_ids=("missing_claim",),
        docs_surface=("docs/missing_scorecard_path.md",),
    )
    invalid_scorecard = type(scorecard)(
        schema="bad.schema",
        artifact_id="bad-artifact",
        rows=(scorecard.rows[1], first, *scorecard.rows[2:]),
        promotion_ready=True,
        ready_category_count=99,
        total_category_count=99,
        claim_boundary=scorecard.claim_boundary,
    )

    validation = validate_differentiable_baseline_scorecard(
        invalid_scorecard,
        repo_root=tmp_path,
    )
    payload = validation.to_dict()

    assert not validation.passed
    assert payload["passed"] is False
    assert any("unexpected scorecard schema" in error for error in validation.errors)
    assert any("unexpected scorecard artifact_id" in error for error in validation.errors)
    assert any("categories must match" in error for error in validation.errors)
    assert any("total_category_count" in error for error in validation.errors)
    assert any("ready_category_count" in error for error in validation.errors)
    assert any("promotion_ready" in error for error in validation.errors)
    assert any("unknown claim-ledger row: missing_claim" in error for error in validation.errors)
    assert any("docs/missing_scorecard_path.md" in error for error in validation.errors)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"passed": 1}, "passed must be boolean"),
        ({"passed": False}, "passed must be true exactly when errors are empty"),
        ({"errors": ["error"]}, "errors must contain non-empty entries"),
        ({"errors": ("",)}, "errors must contain non-empty entries"),
        ({"checked_categories": ["jax_native_transforms"]}, "must contain non-empty entries"),
        (
            {"checked_categories": ("jax_native_transforms", "jax_native_transforms")},
            "checked_categories must contain unique values",
        ),
        ({"checked_categories": ("unknown",)}, "contains unknown category"),
        ({"checked_claim_ids": ("",)}, "checked_claim_ids must contain non-empty entries"),
        (
            {"checked_claim_ids": ("claim", "claim")},
            "checked_claim_ids must contain unique values",
        ),
        ({"checked_paths": ("",)}, "checked_paths must contain non-empty entries"),
        ({"checked_paths": ("path", "path")}, "checked_paths must contain unique values"),
        ({"claim_boundary": "validated"}, "claim_boundary is not canonical"),
    ),
)
def test_differentiable_baseline_scorecard_validation_result_rejects_drift(
    changes: dict[str, object],
    message: str,
) -> None:
    """Validation result construction must preserve exact checked evidence."""
    validation = validate_differentiable_baseline_scorecard(
        run_differentiable_baseline_scorecard()
    )
    assert validation.passed

    with pytest.raises(ValueError, match=message):
        replace(validation, **changes)


def test_differentiable_baseline_scorecard_rejects_external_evidence_paths(
    tmp_path: Path,
) -> None:
    """Claim evidence must be a repository-contained regular file."""
    scorecard = run_differentiable_baseline_scorecard()
    outside = tmp_path.parent / "outside-scorecard-evidence.md"
    outside.write_text("untrusted evidence\n", encoding="utf-8")
    escaped = replace(
        scorecard.rows[0],
        docs_surface=(
            "../outside-scorecard-evidence.md",
            str(outside),
            r"docs\outside-scorecard-evidence.md",
        ),
    )
    candidate = replace(scorecard, rows=(escaped, *scorecard.rows[1:]))

    validation = validate_differentiable_baseline_scorecard(
        candidate,
        repo_root=tmp_path,
    )

    assert not validation.passed
    assert any(
        "evidence path is unsafe: ../outside-scorecard-evidence.md" in error
        for error in validation.errors
    )
    assert any(f"evidence path is unsafe: {outside}" in error for error in validation.errors)
    assert any(
        r"evidence path is unsafe: docs\outside-scorecard-evidence.md" in error
        for error in validation.errors
    )


def test_differentiable_baseline_scorecard_rejects_directory_evidence(
    tmp_path: Path,
) -> None:
    """Existing directories cannot masquerade as scorecard evidence files."""
    scorecard = run_differentiable_baseline_scorecard()
    (tmp_path / "docs").mkdir()
    directory_row = replace(scorecard.rows[0], docs_surface=("docs",))
    candidate = replace(scorecard, rows=(directory_row, *scorecard.rows[1:]))

    validation = validate_differentiable_baseline_scorecard(
        candidate,
        repo_root=tmp_path,
    )

    assert not validation.passed
    assert any("evidence path does not exist: docs" in error for error in validation.errors)


def test_differentiable_promotion_language_rejects_unbacked_public_claims() -> None:
    """Public promotional wording must fail until scorecard and ledger rows are promoted."""
    audit = audit_differentiable_promotion_language(
        public_texts={
            "README.md": ("The differentiable stack is world-leading for JAX native transforms.")
        }
    )

    assert not audit.passed
    assert audit.checked_paths == ("README.md",)
    assert audit.checked_promotional_categories == ("jax_native_transforms",)
    assert any("jax_native_transforms" in error for error in audit.errors)


def test_differentiable_promotion_language_maps_category_markers() -> None:
    """Category marker words route promotional wording to the matching scorecard row."""
    audit = audit_differentiable_promotion_language(
        public_texts={"README.md": "The torch.compile route is promotion-ready."}
    )

    assert not audit.passed
    assert audit.checked_promotional_categories == ("pytorch_autograd_compile",)


def test_differentiable_promotion_language_checks_all_categories_without_category_hint() -> None:
    """Unscoped promotional wording is checked against every scorecard category."""
    audit = audit_differentiable_promotion_language(
        public_texts={"README.md": "This route is promotion-ready."}
    )
    payload = audit.to_dict()

    assert not audit.passed
    assert payload["passed"] is False
    assert set(audit.checked_promotional_categories) == set(REQUIRED_BASELINE_CATEGORIES)


def test_differentiable_promotion_language_allows_ready_promoted_category() -> None:
    """A ready scorecard row with promoted ledger rows may pass the language gate."""
    committed = load_differentiable_claim_ledger()
    promoted = ClaimLedger(
        schema=committed.schema,
        artifact_id=committed.artifact_id,
        rows=tuple(replace(row, promotion_status="promoted") for row in committed.rows),
    )
    scorecard = run_differentiable_baseline_scorecard(ledger=promoted)
    ready_first = replace(scorecard.rows[0], status="at_baseline", blockers=())
    ready_scorecard = type(scorecard)(
        schema=scorecard.schema,
        artifact_id=scorecard.artifact_id,
        rows=(ready_first, *scorecard.rows[1:]),
        promotion_ready=False,
        ready_category_count=1,
        total_category_count=scorecard.total_category_count,
        claim_boundary=scorecard.claim_boundary,
    )

    audit = audit_differentiable_promotion_language(
        public_texts={"README.md": "JAX native transforms are at_baseline."},
        scorecard=ready_scorecard,
        ledger=promoted,
    )

    assert audit.passed
    assert audit.checked_promotional_categories == ("jax_native_transforms",)


def test_differentiable_promotion_language_loads_existing_public_paths(
    tmp_path: Path,
) -> None:
    """The audit loads existing configured paths and fails on missing paths."""
    docs = tmp_path / "docs"
    docs.mkdir()
    present = docs / "page.md"
    present.write_text("This route remains bounded_candidate.\n", encoding="utf-8")

    audit = audit_differentiable_promotion_language(
        public_paths=("docs/page.md", "docs/missing.md"),
        repo_root=tmp_path,
    )

    assert not audit.passed
    assert audit.checked_paths == ("docs/page.md",)
    assert any(
        "public promotion path is missing or not a file: docs/missing.md" in error
        for error in audit.errors
    )


def test_differentiable_promotion_language_rejects_paths_outside_repository(
    tmp_path: Path,
) -> None:
    """Configured claim scans cannot traverse or use absolute public paths."""
    outside = tmp_path.parent / "outside-claim.md"
    outside.write_text("The route is world-leading.\n", encoding="utf-8")

    audit = audit_differentiable_promotion_language(
        public_paths=("../outside-claim.md", str(outside)),
        repo_root=tmp_path,
    )

    assert not audit.passed
    assert audit.checked_paths == ()
    assert any("escapes repository: ../outside-claim.md" in error for error in audit.errors)
    assert any("unsafe public promotion path:" in error for error in audit.errors)


def test_differentiable_promotion_language_reports_unreadable_public_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable configured public surface fails closed with its path."""
    public_path = tmp_path / "public.md"
    public_path.write_text("bounded_candidate\n", encoding="utf-8")
    ledger = load_differentiable_claim_ledger()

    def fail_read_text(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        del path, encoding, errors
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", fail_read_text)
    audit = audit_differentiable_promotion_language(
        public_paths=("public.md",),
        repo_root=tmp_path,
        ledger=ledger,
    )

    assert not audit.passed
    assert audit.checked_paths == ()
    assert any(
        "public promotion path cannot be read: public.md: permission denied" in error
        for error in audit.errors
    )


def test_differentiable_promotion_language_allows_bounded_candidate_wording() -> None:
    """Bounded candidate language may describe the governance lane without promotion."""
    audit = audit_differentiable_promotion_language(
        public_texts={
            "docs/differentiable_programming.md": (
                "The differentiable Phase-QNode lane remains behind_baseline "
                "until isolated benchmark evidence and promoted ledger rows exist."
            )
        }
    )

    assert audit.passed
    assert audit.errors == ()
    assert audit.checked_promotional_categories == ()


def test_differentiable_promotion_language_reports_missing_referenced_row() -> None:
    """An incomplete scorecard must return a finding instead of raising on explicit wording."""
    scorecard = run_differentiable_baseline_scorecard()
    incomplete = replace(scorecard, rows=scorecard.rows[1:])

    audit = audit_differentiable_promotion_language(
        public_texts={"README.md": "The JAX native transforms are world-leading."},
        scorecard=incomplete,
    )

    assert not audit.passed
    assert any(
        "scorecard row is missing: jax_native_transforms" in error for error in audit.errors
    )


@pytest.mark.parametrize(
    "public_texts",
    (
        {"": "bounded_candidate"},
        {"README.md": 1},
        cast(Any, []),
    ),
)
def test_differentiable_promotion_language_rejects_malformed_injected_texts(
    public_texts: object,
) -> None:
    """Injected public-text evidence must use exact non-empty string mappings."""
    with pytest.raises(
        ValueError, match="public_texts must map non-empty string paths to strings"
    ):
        audit_differentiable_promotion_language(public_texts=cast(Any, public_texts))


def test_differentiable_promotion_language_rejects_non_string_configured_path(
    tmp_path: Path,
) -> None:
    """Configured scan paths must fail closed when their runtime type is not a string."""
    audit = audit_differentiable_promotion_language(
        public_paths=cast(Any, (7,)),
        repo_root=tmp_path,
    )

    assert not audit.passed
    assert "unsafe public promotion path: 7" in audit.errors


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"passed": 1}, "passed must be boolean"),
        ({"passed": False}, "passed must be true exactly when errors are empty"),
        ({"errors": ["error"]}, "errors must contain non-empty entries"),
        ({"checked_paths": ["README.md"]}, "checked_paths must contain non-empty entries"),
        ({"checked_paths": ("README.md", "README.md")}, "must contain unique values"),
        ({"checked_promotional_categories": ("",)}, "must contain non-empty entries"),
        (
            {"checked_promotional_categories": ("jax_native_transforms",) * 2},
            "must contain unique values",
        ),
        ({"checked_promotional_categories": ("unknown",)}, "contains unknown category"),
        ({"checked_claim_ids": ("",)}, "checked_claim_ids must contain non-empty entries"),
        ({"checked_claim_ids": ("claim", "claim")}, "must contain unique values"),
        ({"claim_boundary": "audited"}, "claim_boundary is not canonical"),
    ),
)
def test_differentiable_promotion_language_result_rejects_drift(
    changes: dict[str, object],
    message: str,
) -> None:
    """Promotion-audit results must preserve typed, unique, fail-closed evidence."""
    audit = audit_differentiable_promotion_language(
        public_texts={"README.md": "The route remains bounded_candidate."}
    )
    assert audit.passed

    with pytest.raises(ValueError, match=message):
        replace(audit, **changes)


def test_differentiable_baseline_scorecard_markdown_and_facade_dispatch() -> None:
    """The scorecard must render and dispatch through the unified API."""
    scorecard = run_differentiable_baseline_scorecard()
    markdown = render_differentiable_baseline_scorecard_markdown(scorecard)
    result = differentiable_api("baseline_scorecard")

    assert "# Differentiable Baseline Scorecard" in markdown
    assert "catalyst_compiler_workflows" in markdown
    assert "promotion candidate" in markdown
    assert result.operation == "baseline_scorecard"
    assert result.supported is False
    assert result.payload["promotion_ready"] is False
    assert result.payload["total_category_count"] == len(REQUIRED_BASELINE_CATEGORIES)
    assert "promotion candidate" in result.claim_boundary


def test_differentiable_baseline_scorecard_exports_are_public() -> None:
    """Top-level package exports must expose the scorecard types and runner."""
    scorecard = run_differentiable_baseline_scorecard()

    assert isinstance(scorecard.rows[0].category, str)
    assert isinstance(scorecard.rows[0], DifferentiableBaselineScorecardRow)
    assert "jax_native_transforms" in get_args(DifferentiableBaselineCategory)
