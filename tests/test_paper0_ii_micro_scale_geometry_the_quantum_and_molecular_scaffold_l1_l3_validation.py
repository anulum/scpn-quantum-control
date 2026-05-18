# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. Micro-Scale Geometry: The Quantum and Molecular Scaffold (L1-L3) validation tests
"""Tests for Paper 0 II. Micro-Scale Geometry: The Quantum and Molecular Scaffold (L1-L3) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3Config,
    classify_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_component,
    ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_labels,
    validate_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_fixture,
)


def test_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_fixture()
    assert result.source_ledger_span == ("P0R04813", "P0R04823")
    assert result.source_record_count == 11
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04824"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04813"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04823"


def test_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3",
        "1_the_geometry_of_the_quantum_substrate_l1",
    ):
        assert (
            classify_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_labels()
    assert (
        labels["section"] == "II. Micro-Scale Geometry: The Quantum and Molecular Scaffold (L1-L3)"
    )
    assert labels["next_boundary"] == "P0R04824"


def test_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3Config(
            expected_source_record_count=10
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3Config(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04824"):
        IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3Config(
            next_source_boundary="P0R04823"
        )
    with pytest.raises(
        ValueError,
        match="unknown ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3 component",
    ):
        classify_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_component(
            "empirical_validation_claim"
        )
