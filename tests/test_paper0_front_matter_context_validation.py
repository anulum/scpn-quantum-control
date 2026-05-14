# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 front matter context fixtures
"""Tests for Paper 0 front matter and ToC context fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.front_matter_context_validation import (
    FrontMatterContextConfig,
    classify_toc_context,
    collection_books,
    validate_front_matter_context_fixture,
)


def test_collection_books_preserve_source_order_and_book_ii_position() -> None:
    books = collection_books()

    assert len(books) == 5
    assert books[1] == "Book II - The Sentient-Consciousness Projection Network"
    assert books[-1] == "Book V - VIBRANA"


def test_toc_context_classifier_rejects_empirical_evidence_status() -> None:
    classification = classify_toc_context(
        blank_placeholder_count=45,
        fragmented_warning_present=True,
        requested_status="context_boundary",
    )

    assert classification == "fragmented_context_boundary"

    with pytest.raises(ValueError, match="front matter cannot be promoted as empirical evidence"):
        classify_toc_context(
            blank_placeholder_count=45,
            fragmented_warning_present=True,
            requested_status="empirical_evidence",
        )
    with pytest.raises(ValueError, match="blank_placeholder_count cannot be negative"):
        classify_toc_context(
            blank_placeholder_count=-1,
            fragmented_warning_present=True,
            requested_status="context_boundary",
        )


def test_front_matter_context_fixture_preserves_scope_and_counts() -> None:
    result = validate_front_matter_context_fixture()

    assert result.hardware_status == "source_context_no_experiment"
    assert result.source_ledger_span == ("P0R00018", "P0R00104")
    assert result.collection_book_count == 5
    assert result.layer_monograph_count == 16
    assert result.validation_suite_paper_count == 4
    assert result.blank_placeholder_count == 45
    assert result.fragmented_toc_warning_present is True
    assert result.context_classification == "fragmented_context_boundary"
    assert result.null_controls["toc_as_empirical_evidence_rejection_label"] == 1.0
    assert "not validation evidence" in result.claim_boundary

    with pytest.raises(ValueError, match="expected_blank_placeholder_count must be non-negative"):
        FrontMatterContextConfig(expected_blank_placeholder_count=-1)
