# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Cytoskeleton-Water Interface and QEC: validation tests
"""Tests for Paper 0 1. The Cytoskeleton-Water Interface and QEC: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_1_the_cytoskeleton_water_interface_and_qec_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section1TheCytoskeletonWaterInterfaceAndQecConfig,
    classify_section_1_the_cytoskeleton_water_interface_and_qec_component,
    section_1_the_cytoskeleton_water_interface_and_qec_labels,
    validate_section_1_the_cytoskeleton_water_interface_and_qec_fixture,
)


def test_section_1_the_cytoskeleton_water_interface_and_qec_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_1_the_cytoskeleton_water_interface_and_qec_fixture()
    assert result.source_ledger_span == ("P0R04794", "P0R04801")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04802"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_1_the_cytoskeleton_water_interface_and_qec_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04794"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04801"


def test_section_1_the_cytoskeleton_water_interface_and_qec_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "1_the_cytoskeleton_water_interface_and_qec",
        "2_endogenous_em_fields_and_biophotons",
        "3_spin_chemistry_and_the_radical_pair_mechanism_rpm",
    ):
        assert (
            classify_section_1_the_cytoskeleton_water_interface_and_qec_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_1_the_cytoskeleton_water_interface_and_qec_labels()
    assert labels["section"] == "1. The Cytoskeleton-Water Interface and QEC:"
    assert labels["next_boundary"] == "P0R04802"


def test_section_1_the_cytoskeleton_water_interface_and_qec_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section1TheCytoskeletonWaterInterfaceAndQecConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section1TheCytoskeletonWaterInterfaceAndQecConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04802"):
        Section1TheCytoskeletonWaterInterfaceAndQecConfig(next_source_boundary="P0R04801")
    with pytest.raises(
        ValueError, match="unknown section_1_the_cytoskeleton_water_interface_and_qec component"
    ):
        classify_section_1_the_cytoskeleton_water_interface_and_qec_component(
            "empirical_validation_claim"
        )
