# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Case Study: The Layer 3 (Genomic-Morphogenetic) Transduction Pathway validation tests
"""Tests for Paper 0 Case Study: The Layer 3 (Genomic-Morphogenetic) Transduction Pathway source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    CaseStudyTheLayer3GenomicMorphogeneticTransductionPathwayConfig,
    case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_labels,
    classify_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_component,
    validate_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_fixture,
)


def test_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_fixture()
    assert result.source_ledger_span == ("P0R02088", "P0R02097")
    assert result.source_record_count == 10
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02098"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02088"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02097"


def test_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("case_study_the_layer_3_genomic_morphogenetic_transduction_pathway",):
        assert (
            classify_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_labels()
    assert (
        labels["section"] == "Case Study: The Layer 3 (Genomic-Morphogenetic) Transduction Pathway"
    )
    assert labels["next_boundary"] == "P0R02098"


def test_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        CaseStudyTheLayer3GenomicMorphogeneticTransductionPathwayConfig(
            expected_source_record_count=9
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        CaseStudyTheLayer3GenomicMorphogeneticTransductionPathwayConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02098"):
        CaseStudyTheLayer3GenomicMorphogeneticTransductionPathwayConfig(
            next_source_boundary="P0R02097"
        )
    with pytest.raises(
        ValueError,
        match="unknown case_study_the_layer_3_genomic_morphogenetic_transduction_pathway component",
    ):
        classify_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_component(
            "empirical_validation_claim"
        )
