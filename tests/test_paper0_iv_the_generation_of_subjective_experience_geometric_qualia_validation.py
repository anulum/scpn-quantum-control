# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. The Generation of Subjective Experience (Geometric Qualia) validation tests
"""Tests for Paper 0 IV. The Generation of Subjective Experience (Geometric Qualia) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.iv_the_generation_of_subjective_experience_geometric_qualia_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IvTheGenerationOfSubjectiveExperienceGeometricQualiaConfig,
    classify_iv_the_generation_of_subjective_experience_geometric_qualia_component,
    iv_the_generation_of_subjective_experience_geometric_qualia_labels,
    validate_iv_the_generation_of_subjective_experience_geometric_qualia_fixture,
)


def test_iv_the_generation_of_subjective_experience_geometric_qualia_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_iv_the_generation_of_subjective_experience_geometric_qualia_fixture()
    assert result.source_ledger_span == ("P0R06015", "P0R06022")
    assert result.source_record_count == 8
    assert result.component_count == 5
    assert result.next_source_boundary == "P0R06023"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_iv_the_generation_of_subjective_experience_geometric_qualia_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06015"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06022"


def test_iv_the_generation_of_subjective_experience_geometric_qualia_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "iv_the_generation_of_subjective_experience_geometric_qualia",
        "the_ontology_of_experience",
        "formalisation_of_geometric_qualia",
        "the_metric_tensor_gmu_encodes_the_valence_the_intrinsic_curvature_ricci",
        "the_connection_defines_the_flow_of_experience_stream_of_consciousness_vi",
    ):
        assert (
            classify_iv_the_generation_of_subjective_experience_geometric_qualia_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = iv_the_generation_of_subjective_experience_geometric_qualia_labels()
    assert labels["section"] == "IV. The Generation of Subjective Experience (Geometric Qualia)"
    assert labels["next_boundary"] == "P0R06023"


def test_iv_the_generation_of_subjective_experience_geometric_qualia_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        IvTheGenerationOfSubjectiveExperienceGeometricQualiaConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 5"):
        IvTheGenerationOfSubjectiveExperienceGeometricQualiaConfig(expected_component_count=6)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06023"):
        IvTheGenerationOfSubjectiveExperienceGeometricQualiaConfig(next_source_boundary="P0R06022")
    with pytest.raises(
        ValueError,
        match="unknown iv_the_generation_of_subjective_experience_geometric_qualia component",
    ):
        classify_iv_the_generation_of_subjective_experience_geometric_qualia_component(
            "empirical_validation_claim"
        )
