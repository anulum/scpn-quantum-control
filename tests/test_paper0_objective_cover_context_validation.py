# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 objective cover context fixtures
"""Tests for Paper 0 objective and cover context fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.objective_cover_context_validation import (
    ObjectiveCoverContextConfig,
    classify_objective_cover_context,
    objective_cover_targets,
    validate_objective_cover_context_fixture,
)


def test_objective_cover_targets_preserve_source_positioning() -> None:
    targets = objective_cover_targets()
    joined = " ".join(targets)

    assert len(targets) == 5
    assert "Multi-Scale Interaction of Consciousness and Biology" in joined
    assert "cosmic-scale active inference engine" in joined
    assert "Meta Metatron Cycle" in joined
    assert "Book II" in joined


def test_objective_cover_classifier_rejects_validation_evidence_promotion() -> None:
    classification = classify_objective_cover_context(
        image_marker_count=3,
        has_cyclic_operator_positioning=True,
        requested_status="context_boundary",
    )

    assert classification == "cover_positioning_context"

    with pytest.raises(
        ValueError, match="objective and cover context cannot be validation evidence"
    ):
        classify_objective_cover_context(
            image_marker_count=3,
            has_cyclic_operator_positioning=True,
            requested_status="validation_evidence",
        )
    with pytest.raises(ValueError, match="image_marker_count cannot be negative"):
        classify_objective_cover_context(
            image_marker_count=-1,
            has_cyclic_operator_positioning=True,
            requested_status="context_boundary",
        )


def test_objective_cover_context_fixture_preserves_scope_and_counts() -> None:
    result = validate_objective_cover_context_fixture()

    assert result.hardware_status == "source_context_no_experiment"
    assert result.source_ledger_span == ("P0R00218", "P0R00248")
    assert result.image_marker_count == 3
    assert result.collection_book_count == 5
    assert result.positioning_preface_boundary == "P0R00249"
    assert result.context_classification == "cover_positioning_context"
    assert result.null_controls["cover_as_validation_evidence_rejection_label"] == 1.0
    assert "not validation evidence" in result.claim_boundary

    with pytest.raises(ValueError, match="expected_image_marker_count must be non-negative"):
        ObjectiveCoverContextConfig(expected_image_marker_count=-1)
