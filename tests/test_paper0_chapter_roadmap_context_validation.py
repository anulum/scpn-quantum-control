# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 chapter roadmap context fixtures
"""Tests for Paper 0 chapter roadmap context fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.chapter_roadmap_context_validation import (
    ChapterRoadmapContextConfig,
    chapter_roadmap_entries,
    classify_chapter_roadmap_context,
    validate_chapter_roadmap_context_fixture,
)


def test_chapter_roadmap_entries_preserve_major_validation_targets() -> None:
    entries = chapter_roadmap_entries()
    joined = " ".join(entries)

    assert len(entries) == 18
    assert "Unified Phase Dynamics Equation" in joined
    assert "Fisher Information Metric" in joined
    assert "Multi-Scale Quantum Error Correction" in joined
    assert "Recursive Optimisation Hamiltonian" in joined


def test_chapter_roadmap_classifier_rejects_evidence_promotion() -> None:
    classification = classify_chapter_roadmap_context(
        blank_marker_count=5,
        numbering_inconsistency_present=True,
        requested_status="context_boundary",
    )

    assert classification == "numbering_inconsistent_roadmap_context"

    with pytest.raises(
        ValueError, match="chapter roadmap cannot be promoted as validation evidence"
    ):
        classify_chapter_roadmap_context(
            blank_marker_count=5,
            numbering_inconsistency_present=True,
            requested_status="validation_evidence",
        )
    with pytest.raises(ValueError, match="blank_marker_count cannot be negative"):
        classify_chapter_roadmap_context(
            blank_marker_count=-1,
            numbering_inconsistency_present=True,
            requested_status="context_boundary",
        )


def test_chapter_roadmap_context_fixture_preserves_scope_and_counts() -> None:
    result = validate_chapter_roadmap_context_fixture()

    assert result.hardware_status == "source_context_no_experiment"
    assert result.source_ledger_span == ("P0R00105", "P0R00217")
    assert result.part_count == 5
    assert result.chapter_count == 18
    assert result.blank_marker_count == 5
    assert result.numbering_inconsistency_present is True
    assert result.context_classification == "numbering_inconsistent_roadmap_context"
    assert result.null_controls["roadmap_as_validation_evidence_rejection_label"] == 1.0
    assert "not validation evidence" in result.claim_boundary

    with pytest.raises(ValueError, match="expected_chapter_count must equal 18"):
        ChapterRoadmapContextConfig(expected_chapter_count=17)
