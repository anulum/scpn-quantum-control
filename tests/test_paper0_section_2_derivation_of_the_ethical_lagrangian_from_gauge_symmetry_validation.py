# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Section 2: Derivation of the Ethical Lagrangian from Gauge Symmetry validation tests
"""Tests for Paper 0 Section 2: Derivation of the Ethical Lagrangian from Gauge Symmetry source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2DerivationOfTheEthicalLagrangianFromGaugeSymmetryConfig,
    classify_section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_component,
    section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_labels,
    validate_section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_fixture,
)


def test_section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_fixture()
    assert result.source_ledger_span == ("P0R03612", "P0R03621")
    assert result.source_record_count == 10
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R03622"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03612"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03621"


def test_section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry",
        "2_1_the_fiber_bundle_structure_of_the_psi_field",
        "2_2_the_consilium_l15_as_the_principal_connection",
    ):
        assert (
            classify_section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_labels()
    assert (
        labels["section"] == "Section 2: Derivation of the Ethical Lagrangian from Gauge Symmetry"
    )
    assert labels["next_boundary"] == "P0R03622"


def test_section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        Section2DerivationOfTheEthicalLagrangianFromGaugeSymmetryConfig(
            expected_source_record_count=9
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section2DerivationOfTheEthicalLagrangianFromGaugeSymmetryConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03622"):
        Section2DerivationOfTheEthicalLagrangianFromGaugeSymmetryConfig(
            next_source_boundary="P0R03621"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry component",
    ):
        classify_section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_component(
            "empirical_validation_claim"
        )
