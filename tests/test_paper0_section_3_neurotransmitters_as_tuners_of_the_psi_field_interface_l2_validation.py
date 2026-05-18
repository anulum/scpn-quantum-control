# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. Neurotransmitters as Tuners of the Psi-Field Interface (L2): validation tests
"""Tests for Paper 0 3. Neurotransmitters as Tuners of the Psi-Field Interface (L2): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section3NeurotransmittersAsTunersOfThePsiFieldInterfaceL2Config,
    classify_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_component,
    section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_labels,
    validate_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_fixture,
)


def test_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_fixture()
    assert result.source_ledger_span == ("P0R04649", "P0R04656")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04657"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04649"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04656"


def test_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2",
        "4_the_coherent_milieu_csf_and_the_glymphatic_system",
        "ii_the_architecture_of_structure_and_plasticity_domain_i_l3",
    ):
        assert (
            classify_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_labels()
    assert labels["section"] == "3. Neurotransmitters as Tuners of the Psi-Field Interface (L2):"
    assert labels["next_boundary"] == "P0R04657"


def test_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section3NeurotransmittersAsTunersOfThePsiFieldInterfaceL2Config(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section3NeurotransmittersAsTunersOfThePsiFieldInterfaceL2Config(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04657"):
        Section3NeurotransmittersAsTunersOfThePsiFieldInterfaceL2Config(
            next_source_boundary="P0R04656"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2 component",
    ):
        classify_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_component(
            "empirical_validation_claim"
        )
