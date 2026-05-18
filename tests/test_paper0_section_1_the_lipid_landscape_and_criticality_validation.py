# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Lipid Landscape and Criticality validation tests
"""Tests for Paper 0 1. The Lipid Landscape and Criticality source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_1_the_lipid_landscape_and_criticality_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section1TheLipidLandscapeAndCriticalityConfig,
    classify_section_1_the_lipid_landscape_and_criticality_component,
    section_1_the_lipid_landscape_and_criticality_labels,
    validate_section_1_the_lipid_landscape_and_criticality_fixture,
)


def test_section_1_the_lipid_landscape_and_criticality_fixture_preserves_source_boundary() -> None:
    result = validate_section_1_the_lipid_landscape_and_criticality_fixture()
    assert result.source_ledger_span == ("P0R04746", "P0R04753")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04754"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_1_the_lipid_landscape_and_criticality_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04746"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04753"


def test_section_1_the_lipid_landscape_and_criticality_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "1_the_lipid_landscape_and_criticality",
        "2_the_central_role_of_cholesterol",
        "3_lipid_rafts_the_organising_platforms_for_iet",
    ):
        assert (
            classify_section_1_the_lipid_landscape_and_criticality_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_1_the_lipid_landscape_and_criticality_labels()
    assert labels["section"] == "1. The Lipid Landscape and Criticality"
    assert labels["next_boundary"] == "P0R04754"


def test_section_1_the_lipid_landscape_and_criticality_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section1TheLipidLandscapeAndCriticalityConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section1TheLipidLandscapeAndCriticalityConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04754"):
        Section1TheLipidLandscapeAndCriticalityConfig(next_source_boundary="P0R04753")
    with pytest.raises(
        ValueError, match="unknown section_1_the_lipid_landscape_and_criticality component"
    ):
        classify_section_1_the_lipid_landscape_and_criticality_component(
            "empirical_validation_claim"
        )
