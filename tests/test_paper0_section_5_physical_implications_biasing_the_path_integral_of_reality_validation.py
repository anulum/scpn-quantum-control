# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 5. Physical Implications: Biasing the Path Integral of Reality validation tests
"""Tests for Paper 0 5. Physical Implications: Biasing the Path Integral of Reality source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_5_physical_implications_biasing_the_path_integral_of_reality_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section5PhysicalImplicationsBiasingThePathIntegralOfRealityConfig,
    classify_section_5_physical_implications_biasing_the_path_integral_of_reality_component,
    section_5_physical_implications_biasing_the_path_integral_of_reality_labels,
    validate_section_5_physical_implications_biasing_the_path_integral_of_reality_fixture,
)


def test_section_5_physical_implications_biasing_the_path_integral_of_reality_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_section_5_physical_implications_biasing_the_path_integral_of_reality_fixture()
    )
    assert result.source_ledger_span == ("P0R03848", "P0R03868")
    assert result.source_record_count == 21
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03869"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_5_physical_implications_biasing_the_path_integral_of_reality_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03848"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03868"


def test_section_5_physical_implications_biasing_the_path_integral_of_reality_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("5_physical_implications_biasing_the_path_integral_of_reality",):
        assert (
            classify_section_5_physical_implications_biasing_the_path_integral_of_reality_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_5_physical_implications_biasing_the_path_integral_of_reality_labels()
    assert labels["section"] == "5. Physical Implications: Biasing the Path Integral of Reality"
    assert labels["next_boundary"] == "P0R03869"


def test_section_5_physical_implications_biasing_the_path_integral_of_reality_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 21"):
        Section5PhysicalImplicationsBiasingThePathIntegralOfRealityConfig(
            expected_source_record_count=20
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        Section5PhysicalImplicationsBiasingThePathIntegralOfRealityConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03869"):
        Section5PhysicalImplicationsBiasingThePathIntegralOfRealityConfig(
            next_source_boundary="P0R03868"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_5_physical_implications_biasing_the_path_integral_of_reality component",
    ):
        classify_section_5_physical_implications_biasing_the_path_integral_of_reality_component(
            "empirical_validation_claim"
        )
