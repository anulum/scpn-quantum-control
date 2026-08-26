# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for scorecard acceptance engine
"""Real-surface tests for ``scpn_quantum_control.scorecard_acceptance_engine``."""

from __future__ import annotations

from typing import cast

import pytest

import scpn_quantum_control.scorecard_acceptance_engine as scorecard_acceptance_engine
from scpn_quantum_control.differentiable_baseline_scorecard import (
    REQUIRED_BASELINE_CATEGORIES,
    DifferentiableBaselineCategory,
)
from scpn_quantum_control.scorecard_acceptance_engine import (
    SCORECARD_ACCEPTANCE_CLAIM_BOUNDARY,
    SCORECARD_ACCEPTANCE_ENGINE_SCHEMA,
    PromoteDecision,
    ScorecardCategoryRecord,
    ScorecardEngineStatus,
    assert_scorecard_acceptance_integrity,
    build_scorecard_acceptance_registry,
    get_scorecard_category,
    iter_scorecard_categories,
    list_scorecard_category_ids,
    promote_scorecard_category,
)


def _category_rows(registry: dict[str, object]) -> list[dict[str, object]]:
    """Return explicitly narrowed mutable copies of registry category rows."""
    raw_rows = registry["categories"]
    if not isinstance(raw_rows, list):
        raise AssertionError("registry categories must be a list")
    rows: list[dict[str, object]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise AssertionError("registry category rows must be dictionaries")
        rows.append({str(key): value for key, value in raw_row.items()})
    return rows


def test_list_covers_all_required_categories() -> None:
    """Cover every required category exactly once in stable order."""
    ids = list_scorecard_category_ids()
    assert len(ids) == len(REQUIRED_BASELINE_CATEGORIES)
    assert set(ids) == set(REQUIRED_BASELINE_CATEGORIES)
    assert ids == list_scorecard_category_ids()


def test_all_inventory_rows_honestly_behind_baseline() -> None:
    """Keep the canonical inventory honestly behind baseline with blockers."""
    rows = iter_scorecard_categories()
    assert len(rows) == 11
    assert all(row.status == "behind_baseline" for row in rows)
    assert all(row.blockers for row in rows)
    behind = iter_scorecard_categories(status="behind_baseline")
    assert len(behind) == 11


def test_get_known_and_unknown() -> None:
    """Resolve known categories and reject blank or unknown identifiers."""
    row = get_scorecard_category("jax_native_transforms")
    assert row.category_id == "jax_native_transforms"
    assert row.claim_boundary == SCORECARD_ACCEPTANCE_CLAIM_BOUNDARY
    assert row.required_evidence
    with pytest.raises(ValueError, match="non-empty"):
        get_scorecard_category("  ")
    with pytest.raises(ValueError, match="unknown scorecard category_id"):
        get_scorecard_category("not_a_category")


def test_build_registry_counts() -> None:
    """Build a complete schema-tagged registry with consistent counts."""
    registry = build_scorecard_acceptance_registry()
    assert registry["schema"] == SCORECARD_ACCEPTANCE_ENGINE_SCHEMA
    assert registry["category_count"] == 11
    assert registry["behind_baseline_count"] == 11
    assert registry["ready_category_count"] == 0
    assert registry["blank_entry_count"] == 0
    validated = assert_scorecard_acceptance_integrity(registry)
    assert validated["category_count"] == 11


def test_promote_refuses_without_evidence() -> None:
    """Refuse ready-state promotion when no evidence labels are supplied."""
    decision = promote_scorecard_category(
        "benchmark_promotion",
        target_status="exceeds_baseline",
        evidence_ids=(),
    )
    assert decision.allowed is False
    assert decision.missing_evidence
    assert "refuse invent-green" in decision.reason


def test_promote_refuses_partial_evidence_and_bad_language() -> None:
    """Refuse partial evidence and unbounded promotional language."""
    partial = promote_scorecard_category(
        "jax_native_transforms",
        target_status="at_baseline",
        evidence_ids=("claim_ledger_promoted_row:demo",),
    )
    assert partial.allowed is False
    assert "external_baseline_comparison" in partial.missing_evidence

    full_ids = (
        "claim_ledger_promoted_row:demo",
        "external_baseline_comparison:demo",
    )
    bad_lang = promote_scorecard_category(
        "jax_native_transforms",
        target_status="at_baseline",
        evidence_ids=full_ids,
        language_claim="We are state-of-the-art and a category of its own",
    )
    assert bad_lang.allowed is False
    assert "language" in bad_lang.reason.lower() or "invent-green" in bad_lang.reason


def test_promote_allows_with_evidence_and_demote() -> None:
    """Allow complete evidence, honest demotion, and evidenced non-comparison."""
    full_ids = (
        "claim_ledger_promoted_row:demo",
        "external_baseline_comparison:demo",
    )
    ok = promote_scorecard_category(
        "rust_native_program_ad",
        target_status="at_baseline",
        evidence_ids=full_ids,
        language_claim="bounded local parity evidence only",
    )
    assert ok.allowed is True
    assert ok.to_status == "at_baseline"

    demote = promote_scorecard_category(
        "rust_native_program_ad",
        target_status="behind_baseline",
    )
    assert demote.allowed is True

    not_comp = promote_scorecard_category(
        "provider_hardware_gradients",
        target_status="not_comparable",
        evidence_ids=("comparison_impossibility_note:no_qpu_access",),
    )
    assert not_comp.allowed is True

    not_comp_fail = promote_scorecard_category(
        "provider_hardware_gradients",
        target_status="not_comparable",
        evidence_ids=(),
    )
    assert not_comp_fail.allowed is False


def test_promote_unknown_target_status() -> None:
    """Reject target statuses outside the closed vocabulary."""
    with pytest.raises(ValueError, match="unknown target_status"):
        promote_scorecard_category(
            "jax_native_transforms",
            target_status=cast(ScorecardEngineStatus, "green"),
        )


def test_record_validation() -> None:
    """Enforce every immutable scorecard-category record invariant."""
    with pytest.raises(ValueError, match="unknown scorecard category"):
        ScorecardCategoryRecord(
            category_id=cast(DifferentiableBaselineCategory, "nope"),
            status="behind_baseline",
            summary="s",
            evidence_ids=(),
            blockers=("b",),
            required_evidence=("e",),
        )
    with pytest.raises(ValueError, match="unknown scorecard status"):
        ScorecardCategoryRecord(
            category_id="jax_native_transforms",
            status=cast(ScorecardEngineStatus, "green"),
            summary="s",
            evidence_ids=(),
            blockers=("b",),
            required_evidence=("e",),
        )
    with pytest.raises(ValueError, match="summary"):
        ScorecardCategoryRecord(
            category_id="jax_native_transforms",
            status="behind_baseline",
            summary="  ",
            evidence_ids=(),
            blockers=("b",),
            required_evidence=("e",),
        )
    with pytest.raises(ValueError, match="as_of"):
        ScorecardCategoryRecord(
            category_id="jax_native_transforms",
            status="behind_baseline",
            summary="s",
            evidence_ids=(),
            blockers=("b",),
            required_evidence=("e",),
            as_of="",
        )
    with pytest.raises(ValueError, match="behind_baseline rows require"):
        ScorecardCategoryRecord(
            category_id="jax_native_transforms",
            status="behind_baseline",
            summary="s",
            evidence_ids=(),
            blockers=(),
            required_evidence=("e",),
        )
    with pytest.raises(ValueError, match="must not carry open blockers"):
        ScorecardCategoryRecord(
            category_id="jax_native_transforms",
            status="at_baseline",
            summary="s",
            evidence_ids=("ev",),
            blockers=("still blocked",),
            required_evidence=("e",),
        )
    with pytest.raises(ValueError, match="require evidence_ids"):
        ScorecardCategoryRecord(
            category_id="jax_native_transforms",
            status="exceeds_baseline",
            summary="s",
            evidence_ids=(),
            blockers=(),
            required_evidence=("e",),
        )
    with pytest.raises(ValueError, match="evidence_ids"):
        ScorecardCategoryRecord(
            category_id="jax_native_transforms",
            status="behind_baseline",
            summary="s",
            evidence_ids=("",),
            blockers=("b",),
            required_evidence=("e",),
        )
    with pytest.raises(ValueError, match="blockers"):
        ScorecardCategoryRecord(
            category_id="jax_native_transforms",
            status="behind_baseline",
            summary="s",
            evidence_ids=(),
            blockers=("  ",),
            required_evidence=("e",),
        )
    with pytest.raises(ValueError, match="required_evidence"):
        ScorecardCategoryRecord(
            category_id="jax_native_transforms",
            status="behind_baseline",
            summary="s",
            evidence_ids=(),
            blockers=("b",),
            required_evidence=("",),
        )


def test_promote_decision_validation() -> None:
    """Enforce non-empty and internally consistent promotion decisions."""
    with pytest.raises(ValueError, match="category_id"):
        PromoteDecision(
            category_id="",
            allowed=False,
            from_status="behind_baseline",
            to_status="at_baseline",
            reason="r",
        )
    with pytest.raises(ValueError, match="reason"):
        PromoteDecision(
            category_id="c",
            allowed=False,
            from_status="behind_baseline",
            to_status="at_baseline",
            reason="",
        )
    with pytest.raises(ValueError, match="missing_evidence"):
        PromoteDecision(
            category_id="c",
            allowed=True,
            from_status="behind_baseline",
            to_status="at_baseline",
            reason="r",
            missing_evidence=("x",),
        )
    with pytest.raises(ValueError, match="missing_evidence entries"):
        PromoteDecision(
            category_id="c",
            allowed=False,
            from_status="behind_baseline",
            to_status="at_baseline",
            reason="r",
            missing_evidence=("  ",),
        )


def test_assert_integrity_rejects_invalid() -> None:
    """Reject malformed, incomplete, blank, or unsupported registry rows."""
    with pytest.raises(ValueError, match="non-empty categories"):
        assert_scorecard_acceptance_integrity({"categories": []})
    with pytest.raises(ValueError, match="blank"):
        assert_scorecard_acceptance_integrity(
            {
                "categories": [{"category_id": "", "status": "behind_baseline"}],
                "blank_entry_count": 0,
                "category_count": 1,
            }
        )
    with pytest.raises(ValueError, match="unknown category"):
        assert_scorecard_acceptance_integrity(
            {
                "categories": [
                    {
                        "category_id": "nope",
                        "status": "behind_baseline",
                        "blockers": ["b"],
                    }
                ],
                "blank_entry_count": 0,
                "category_count": 1,
            }
        )
    good = get_scorecard_category("jax_native_transforms").to_dict()
    with pytest.raises(ValueError, match="missing categories"):
        assert_scorecard_acceptance_integrity(
            {
                "categories": [good],
                "blank_entry_count": 0,
                "category_count": 1,
            }
        )
    full = build_scorecard_acceptance_registry()
    full_rows = _category_rows(full)
    # mutate one row to promoted without evidence
    cats = [dict(row) for row in full_rows]
    cats[0] = {
        **cats[0],
        "status": "at_baseline",
        "blockers": [],
        "evidence_ids": [],
    }
    with pytest.raises(ValueError, match="without evidence"):
        assert_scorecard_acceptance_integrity(
            {
                "categories": cats,
                "blank_entry_count": 0,
                "category_count": len(cats),
            }
        )
    # promoted with non-list evidence_ids
    cats2 = [dict(row) for row in full_rows]
    cats2[0] = {
        **cats2[0],
        "status": "exceeds_baseline",
        "blockers": [],
        "evidence_ids": "not-a-list",
    }
    with pytest.raises(ValueError, match="without evidence"):
        assert_scorecard_acceptance_integrity(
            {
                "categories": cats2,
                "blank_entry_count": 0,
                "category_count": len(cats2),
            }
        )
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_scorecard_acceptance_integrity({**full, "blank_entry_count": 1})
    with pytest.raises(ValueError, match="category_count"):
        assert_scorecard_acceptance_integrity({**full, "category_count": 0})
    with pytest.raises(ValueError, match="mapping"):
        assert_scorecard_acceptance_integrity(
            {
                "categories": ["x"],
                "blank_entry_count": 0,
                "category_count": 1,
            }
        )
    with pytest.raises(ValueError, match="blank"):
        assert_scorecard_acceptance_integrity(
            {
                "categories": [
                    {
                        "category_id": "jax_native_transforms",
                        "status": "not-a-status",
                        "blockers": ["b"],
                    }
                ]
                + [
                    get_scorecard_category(cid).to_dict()
                    for cid in list_scorecard_category_ids()
                    if cid != "jax_native_transforms"
                ],
                "blank_entry_count": 0,
                "category_count": 11,
            }
        )
    # behind without blockers
    bad_behind = {
        **get_scorecard_category("pytorch_autograd_compile").to_dict(),
        "blockers": [],
    }
    with pytest.raises(ValueError, match="without blockers"):
        assert_scorecard_acceptance_integrity(
            {
                "categories": [
                    bad_behind if row["category_id"] == "pytorch_autograd_compile" else row
                    for row in full_rows
                ],
                "blank_entry_count": 0,
                "category_count": 11,
            }
        )


def test_catalogue_map_coverage_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a canonical catalogue that omits required categories."""
    row = get_scorecard_category("jax_native_transforms")
    monkeypatch.setattr(
        scorecard_acceptance_engine,
        "_CANONICAL_CATEGORIES",
        (row,),
    )
    with pytest.raises(RuntimeError, match="cover all required"):
        scorecard_acceptance_engine._catalogue_map()


def test_to_dict_round_trips() -> None:
    """Serialize category records and promotion decisions to public mappings."""
    row = get_scorecard_category("docs_api_maintainability")
    payload = row.to_dict()
    assert payload["category_id"] == "docs_api_maintainability"
    assert isinstance(payload["blockers"], list)
    decision = promote_scorecard_category(
        "docs_api_maintainability",
        target_status="exceeds_baseline",
    )
    assert decision.to_dict()["allowed"] is False


def test_integrity_accepts_promoted_rows_with_evidence() -> None:
    """Integrity accepts at_baseline / exceeds_baseline rows that carry evidence_ids."""
    registry = build_scorecard_acceptance_registry()
    rows = _category_rows(registry)
    # Promote two rows with evidence so the evidence-present branch is covered.
    rows[0]["status"] = "at_baseline"
    rows[0]["blockers"] = []
    rows[0]["evidence_ids"] = ["ledger:at_baseline_demo"]
    rows[1]["status"] = "exceeds_baseline"
    rows[1]["blockers"] = []
    rows[1]["evidence_ids"] = ["ledger:exceeds_demo"]
    promoted = dict(registry)
    promoted["categories"] = rows
    validated = assert_scorecard_acceptance_integrity(promoted)
    assert validated["blank_entry_count"] == 0
