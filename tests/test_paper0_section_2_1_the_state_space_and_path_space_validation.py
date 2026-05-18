# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2.1 The State Space and Path Space validation tests
"""Tests for Paper 0 2.1 The State Space and Path Space source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_1_the_state_space_and_path_space_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section21TheStateSpaceAndPathSpaceConfig,
    classify_section_2_1_the_state_space_and_path_space_component,
    section_2_1_the_state_space_and_path_space_labels,
    validate_section_2_1_the_state_space_and_path_space_fixture,
)


def test_section_2_1_the_state_space_and_path_space_fixture_preserves_source_boundary() -> None:
    result = validate_section_2_1_the_state_space_and_path_space_fixture()
    assert result.source_ledger_span == ("P0R03753", "P0R03761")
    assert result.source_record_count == 9
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03762"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_1_the_state_space_and_path_space_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03753"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03761"


def test_section_2_1_the_state_space_and_path_space_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("2_1_the_state_space_and_path_space", "2_2_the_path_integral_measure"):
        assert (
            classify_section_2_1_the_state_space_and_path_space_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_2_1_the_state_space_and_path_space_labels()
    assert labels["section"] == "2.1 The State Space and Path Space"
    assert labels["next_boundary"] == "P0R03762"


def test_section_2_1_the_state_space_and_path_space_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        Section21TheStateSpaceAndPathSpaceConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section21TheStateSpaceAndPathSpaceConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03762"):
        Section21TheStateSpaceAndPathSpaceConfig(next_source_boundary="P0R03761")
    with pytest.raises(
        ValueError, match="unknown section_2_1_the_state_space_and_path_space component"
    ):
        classify_section_2_1_the_state_space_and_path_space_component("empirical_validation_claim")
