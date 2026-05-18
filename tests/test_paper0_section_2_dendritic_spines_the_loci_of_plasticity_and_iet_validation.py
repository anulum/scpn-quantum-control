# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. Dendritic Spines: The Loci of Plasticity and IET validation tests
"""Tests for Paper 0 2. Dendritic Spines: The Loci of Plasticity and IET source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_dendritic_spines_the_loci_of_plasticity_and_iet_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2DendriticSpinesTheLociOfPlasticityAndIetConfig,
    classify_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_component,
    section_2_dendritic_spines_the_loci_of_plasticity_and_iet_labels,
    validate_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_fixture,
)


def test_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_fixture()
    assert result.source_ledger_span == ("P0R04703", "P0R04710")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04711"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04703"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04710"


def test_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "2_dendritic_spines_the_loci_of_plasticity_and_iet",
        "3_the_axon_and_the_axon_initial_segment_ais_the_decision_point",
        "ii_the_chemical_milieu_and_the_primacy_of_water_l1_l2",
    ):
        assert (
            classify_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_2_dendritic_spines_the_loci_of_plasticity_and_iet_labels()
    assert labels["section"] == "2. Dendritic Spines: The Loci of Plasticity and IET"
    assert labels["next_boundary"] == "P0R04711"


def test_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section2DendriticSpinesTheLociOfPlasticityAndIetConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section2DendriticSpinesTheLociOfPlasticityAndIetConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04711"):
        Section2DendriticSpinesTheLociOfPlasticityAndIetConfig(next_source_boundary="P0R04710")
    with pytest.raises(
        ValueError,
        match="unknown section_2_dendritic_spines_the_loci_of_plasticity_and_iet component",
    ):
        classify_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_component(
            "empirical_validation_claim"
        )
