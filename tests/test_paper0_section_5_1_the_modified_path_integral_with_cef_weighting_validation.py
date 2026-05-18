# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 5.1 The Modified Path Integral with CEF Weighting validation tests
"""Tests for Paper 0 5.1 The Modified Path Integral with CEF Weighting source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_5_1_the_modified_path_integral_with_cef_weighting_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section51TheModifiedPathIntegralWithCefWeightingConfig,
    classify_section_5_1_the_modified_path_integral_with_cef_weighting_component,
    section_5_1_the_modified_path_integral_with_cef_weighting_labels,
    validate_section_5_1_the_modified_path_integral_with_cef_weighting_fixture,
)


def test_section_5_1_the_modified_path_integral_with_cef_weighting_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_5_1_the_modified_path_integral_with_cef_weighting_fixture()
    assert result.source_ledger_span == ("P0R03869", "P0R03931")
    assert result.source_record_count == 63
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03932"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_5_1_the_modified_path_integral_with_cef_weighting_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03869"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03931"


def test_section_5_1_the_modified_path_integral_with_cef_weighting_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "5_1_the_modified_path_integral_with_cef_weighting",
        "5_2_new_falsifiable_predictions",
    ):
        assert (
            classify_section_5_1_the_modified_path_integral_with_cef_weighting_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_5_1_the_modified_path_integral_with_cef_weighting_labels()
    assert labels["section"] == "5.1 The Modified Path Integral with CEF Weighting"
    assert labels["next_boundary"] == "P0R03932"


def test_section_5_1_the_modified_path_integral_with_cef_weighting_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 63"):
        Section51TheModifiedPathIntegralWithCefWeightingConfig(expected_source_record_count=62)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section51TheModifiedPathIntegralWithCefWeightingConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03932"):
        Section51TheModifiedPathIntegralWithCefWeightingConfig(next_source_boundary="P0R03931")
    with pytest.raises(
        ValueError,
        match="unknown section_5_1_the_modified_path_integral_with_cef_weighting component",
    ):
        classify_section_5_1_the_modified_path_integral_with_cef_weighting_component(
            "empirical_validation_claim"
        )
