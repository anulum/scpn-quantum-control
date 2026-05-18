# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. Examination of The Architecture of Structure and Plasticity (Domain I: L3) validation tests
"""Tests for Paper 0 II. Examination of The Architecture of Structure and Plasticity (Domain I: L3) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainIConfig,
    classify_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_component,
    ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_labels,
    validate_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_fixture,
)


def test_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_fixture()
    )
    assert result.source_ledger_span == ("P0R04560", "P0R04571")
    assert result.source_record_count == 12
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04572"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04560"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04571"


def test_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i",
        "the_optimised_connectome_the_geometric_scaffold_of_thought",
        "the_active_role_of_glia_the_slow_control_network",
    ):
        assert (
            classify_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_labels()
    assert (
        labels["section"]
        == "II. Examination of The Architecture of Structure and Plasticity (Domain I: L3)"
    )
    assert labels["next_boundary"] == "P0R04572"


def test_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 12"):
        IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainIConfig(
            expected_source_record_count=11
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainIConfig(
            expected_component_count=4
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04572"):
        IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainIConfig(
            next_source_boundary="P0R04571"
        )
    with pytest.raises(
        ValueError,
        match="unknown ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i component",
    ):
        classify_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_component(
            "empirical_validation_claim"
        )
