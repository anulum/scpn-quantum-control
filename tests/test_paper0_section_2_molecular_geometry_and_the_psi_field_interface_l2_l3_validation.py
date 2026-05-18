# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. Molecular Geometry and the Psi-Field Interface (L2/L3): validation tests
"""Tests for Paper 0 2. Molecular Geometry and the Psi-Field Interface (L2/L3): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2MolecularGeometryAndThePsiFieldInterfaceL2L3Config,
    classify_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_component,
    section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_labels,
    validate_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_fixture,
)


def test_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_fixture()
    assert result.source_ledger_span == ("P0R04824", "P0R04831")
    assert result.source_record_count == 8
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R04832"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04824"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04831"


def test_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "2_molecular_geometry_and_the_psi_field_interface_l2_l3",
        "3_cellular_geometry_tensegrity_and_fractals_l3",
        "iii_meso_scale_geometry_circuits_columns_and_fields_l3_l4",
        "1_the_geometry_of_the_bioelectric_field_l3",
    ):
        assert (
            classify_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_labels()
    assert labels["section"] == "2. Molecular Geometry and the Psi-Field Interface (L2/L3):"
    assert labels["next_boundary"] == "P0R04832"


def test_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section2MolecularGeometryAndThePsiFieldInterfaceL2L3Config(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        Section2MolecularGeometryAndThePsiFieldInterfaceL2L3Config(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04832"):
        Section2MolecularGeometryAndThePsiFieldInterfaceL2L3Config(next_source_boundary="P0R04831")
    with pytest.raises(
        ValueError,
        match="unknown section_2_molecular_geometry_and_the_psi_field_interface_l2_l3 component",
    ):
        classify_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_component(
            "empirical_validation_claim"
        )
