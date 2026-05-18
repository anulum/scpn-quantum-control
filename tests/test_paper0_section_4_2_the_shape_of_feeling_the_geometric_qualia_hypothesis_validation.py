# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4.2 The Shape of Feeling: The Geometric Qualia Hypothesis validation tests
"""Tests for Paper 0 4.2 The Shape of Feeling: The Geometric Qualia Hypothesis source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section42TheShapeOfFeelingTheGeometricQualiaHypothesisConfig,
    classify_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_component,
    section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_labels,
    validate_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_fixture,
)


def test_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_fixture()
    assert result.source_ledger_span == ("P0R03386", "P0R03399")
    assert result.source_record_count == 14
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03400"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03386"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03399"


def test_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis",
        "confronting_the_hard_problem_a_mathematical_resolution",
    ):
        assert (
            classify_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_labels()
    assert labels["section"] == "4.2 The Shape of Feeling: The Geometric Qualia Hypothesis"
    assert labels["next_boundary"] == "P0R03400"


def test_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 14"):
        Section42TheShapeOfFeelingTheGeometricQualiaHypothesisConfig(
            expected_source_record_count=13
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section42TheShapeOfFeelingTheGeometricQualiaHypothesisConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03400"):
        Section42TheShapeOfFeelingTheGeometricQualiaHypothesisConfig(
            next_source_boundary="P0R03399"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis component",
    ):
        classify_section_4_2_the_shape_of_feeling_the_geometric_qualia_hypothesis_component(
            "empirical_validation_claim"
        )
