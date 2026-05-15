# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 positioning preface context fixtures
"""Tests for Paper 0 Positioning Preface context fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.positioning_preface_context_validation import (
    PositioningPrefaceContextConfig,
    classify_positioning_preface_context,
    positioning_preface_targets,
    validate_positioning_preface_context_fixture,
)


def test_positioning_preface_targets_preserve_source_positioning() -> None:
    targets = positioning_preface_targets()
    joined = " ".join(targets)

    assert len(targets) == 5
    assert "Field Architecture" in joined
    assert "Consciousness Engineering" in joined
    assert "Noetic Field Theory" in joined
    assert "two frequencies" in joined


def test_positioning_preface_classifier_rejects_validation_evidence_promotion() -> None:
    classification = classify_positioning_preface_context(
        blank_separator_count=2,
        image_marker_count=1,
        has_dual_register_author_note=True,
        requested_status="context_boundary",
    )

    assert classification == "dual_register_preface_context"

    with pytest.raises(ValueError, match="positioning preface cannot be validation evidence"):
        classify_positioning_preface_context(
            blank_separator_count=2,
            image_marker_count=1,
            has_dual_register_author_note=True,
            requested_status="validation_evidence",
        )
    with pytest.raises(ValueError, match="blank_separator_count cannot be negative"):
        classify_positioning_preface_context(
            blank_separator_count=-1,
            image_marker_count=1,
            has_dual_register_author_note=True,
            requested_status="context_boundary",
        )


def test_positioning_preface_context_fixture_preserves_scope_and_counts() -> None:
    result = validate_positioning_preface_context_fixture()

    assert result.hardware_status == "source_context_no_experiment"
    assert result.source_ledger_span == ("P0R00249", "P0R00267")
    assert result.blank_separator_count == 2
    assert result.image_marker_count == 1
    assert result.part_i_boundary == "P0R00268"
    assert result.context_classification == "dual_register_preface_context"
    assert result.null_controls["preface_as_validation_evidence_rejection_label"] == 1.0
    assert "not validation evidence" in result.claim_boundary

    with pytest.raises(ValueError, match="expected_image_marker_count must be non-negative"):
        PositioningPrefaceContextConfig(expected_image_marker_count=-1)
