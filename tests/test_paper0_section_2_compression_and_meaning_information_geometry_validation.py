# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. Compression and Meaning (Information Geometry): validation tests
"""Tests for Paper 0 2. Compression and Meaning (Information Geometry): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_compression_and_meaning_information_geometry_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2CompressionAndMeaningInformationGeometryConfig,
    classify_section_2_compression_and_meaning_information_geometry_component,
    section_2_compression_and_meaning_information_geometry_labels,
    validate_section_2_compression_and_meaning_information_geometry_fixture,
)


def test_section_2_compression_and_meaning_information_geometry_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_compression_and_meaning_information_geometry_fixture()
    assert result.source_ledger_span == ("P0R03232", "P0R03240")
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R03241"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_compression_and_meaning_information_geometry_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03232"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03240"


def test_section_2_compression_and_meaning_information_geometry_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "2_compression_and_meaning_information_geometry",
        "iv_inter_layer_transition_operators_ilto",
        "key_operators",
    ):
        assert (
            classify_section_2_compression_and_meaning_information_geometry_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_2_compression_and_meaning_information_geometry_labels()
    assert labels["section"] == "2. Compression and Meaning (Information Geometry):"
    assert labels["next_boundary"] == "P0R03241"


def test_section_2_compression_and_meaning_information_geometry_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        Section2CompressionAndMeaningInformationGeometryConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section2CompressionAndMeaningInformationGeometryConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03241"):
        Section2CompressionAndMeaningInformationGeometryConfig(next_source_boundary="P0R03240")
    with pytest.raises(
        ValueError,
        match="unknown section_2_compression_and_meaning_information_geometry component",
    ):
        classify_section_2_compression_and_meaning_information_geometry_component(
            "empirical_validation_claim"
        )
