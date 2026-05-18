# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Notes on correspondence (non-obligatory analogues). validation tests
"""Tests for Paper 0 Notes on correspondence (non-obligatory analogues). source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.notes_on_correspondence_non_obligatory_analogues_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    NotesOnCorrespondenceNonObligatoryAnaloguesConfig,
    classify_notes_on_correspondence_non_obligatory_analogues_component,
    notes_on_correspondence_non_obligatory_analogues_labels,
    validate_notes_on_correspondence_non_obligatory_analogues_fixture,
)


def test_notes_on_correspondence_non_obligatory_analogues_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_notes_on_correspondence_non_obligatory_analogues_fixture()
    assert result.source_ledger_span == ("P0R04009", "P0R04028")
    assert result.source_record_count == 20
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04029"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_notes_on_correspondence_non_obligatory_analogues_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04009"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04028"


def test_notes_on_correspondence_non_obligatory_analogues_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "notes_on_correspondence_non_obligatory_analogues",
        "terminology_and_notation_effective_immediately_revision_11_00",
        "consequences",
    ):
        assert (
            classify_notes_on_correspondence_non_obligatory_analogues_component(component)
            == f"{component}_source_boundary"
        )
    labels = notes_on_correspondence_non_obligatory_analogues_labels()
    assert labels["section"] == "Notes on correspondence (non-obligatory analogues)."
    assert labels["next_boundary"] == "P0R04029"


def test_notes_on_correspondence_non_obligatory_analogues_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 20"):
        NotesOnCorrespondenceNonObligatoryAnaloguesConfig(expected_source_record_count=19)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        NotesOnCorrespondenceNonObligatoryAnaloguesConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04029"):
        NotesOnCorrespondenceNonObligatoryAnaloguesConfig(next_source_boundary="P0R04028")
    with pytest.raises(
        ValueError, match="unknown notes_on_correspondence_non_obligatory_analogues component"
    ):
        classify_notes_on_correspondence_non_obligatory_analogues_component(
            "empirical_validation_claim"
        )
