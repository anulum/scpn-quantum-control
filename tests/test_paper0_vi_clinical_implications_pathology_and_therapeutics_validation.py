# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 VI. Clinical Implications: Pathology and Therapeutics validation tests
"""Tests for Paper 0 VI. Clinical Implications: Pathology and Therapeutics source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.vi_clinical_implications_pathology_and_therapeutics_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ViClinicalImplicationsPathologyAndTherapeuticsConfig,
    classify_vi_clinical_implications_pathology_and_therapeutics_component,
    validate_vi_clinical_implications_pathology_and_therapeutics_fixture,
    vi_clinical_implications_pathology_and_therapeutics_labels,
)


def test_vi_clinical_implications_pathology_and_therapeutics_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_vi_clinical_implications_pathology_and_therapeutics_fixture()
    assert result.source_ledger_span == ("P0R04693", "P0R04702")
    assert result.source_record_count == 10
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R04703"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_vi_clinical_implications_pathology_and_therapeutics_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04693"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04702"


def test_vi_clinical_implications_pathology_and_therapeutics_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "vi_clinical_implications_pathology_and_therapeutics",
        "the_ultra_detailed_architecture_of_the_neuron_within_the_scpn",
        "i_neuronal_geometry_and_the_physics_of_information_flow_l3_l4",
        "1_the_dendritic_arbour_the_antenna_of_the_neuron",
    ):
        assert (
            classify_vi_clinical_implications_pathology_and_therapeutics_component(component)
            == f"{component}_source_boundary"
        )
    labels = vi_clinical_implications_pathology_and_therapeutics_labels()
    assert labels["section"] == "VI. Clinical Implications: Pathology and Therapeutics"
    assert labels["next_boundary"] == "P0R04703"


def test_vi_clinical_implications_pathology_and_therapeutics_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        ViClinicalImplicationsPathologyAndTherapeuticsConfig(expected_source_record_count=9)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        ViClinicalImplicationsPathologyAndTherapeuticsConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04703"):
        ViClinicalImplicationsPathologyAndTherapeuticsConfig(next_source_boundary="P0R04702")
    with pytest.raises(
        ValueError, match="unknown vi_clinical_implications_pathology_and_therapeutics component"
    ):
        classify_vi_clinical_implications_pathology_and_therapeutics_component(
            "empirical_validation_claim"
        )
