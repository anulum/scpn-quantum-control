# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 meta-framework Psi coupling validation tests
"""Tests for Paper 0 meta-framework/Psi-coupling validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.meta_framework_psi_coupling_validation import (
    MetaFrameworkPsiCouplingConfig,
    classify_meta_framework_psi_coupling_component,
    meta_framework_psi_coupling_labels,
    validate_meta_framework_psi_coupling_fixture,
)


def test_meta_framework_psi_coupling_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 67"):
        MetaFrameworkPsiCouplingConfig(expected_source_record_count=66)

    with pytest.raises(ValueError, match="expected_blank_record_count must equal 2"):
        MetaFrameworkPsiCouplingConfig(expected_blank_record_count=1)

    with pytest.raises(ValueError, match="expected_image_or_figure_record_count must equal 6"):
        MetaFrameworkPsiCouplingConfig(expected_image_or_figure_record_count=5)

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00905"):
        MetaFrameworkPsiCouplingConfig(next_source_boundary="P0R00904")


def test_meta_framework_psi_coupling_classifiers_are_source_bounded() -> None:
    assert (
        classify_meta_framework_psi_coupling_component("meta_framework_boundary")
        == "meta_framework_integrations_boundary"
    )
    assert (
        classify_meta_framework_psi_coupling_component("predictive_coding_loop")
        == "fibre_bundle_belief_state_and_tripartite_active_inference_loop"
    )
    assert (
        classify_meta_framework_psi_coupling_component("psi_interaction_hamiltonian")
        == "h_int_minus_lambda_psis_sigma_coupling_statement"
    )
    assert (
        classify_meta_framework_psi_coupling_component("coupling_projection")
        == "total_space_to_fibre_projection_and_g_to_h_transduction"
    )
    assert (
        classify_meta_framework_psi_coupling_component("formal_ontology_restatement")
        == "psi_x_in_e_and_pi_e_to_m_formal_restatement"
    )
    assert (
        classify_meta_framework_psi_coupling_component("figure_and_image_records")
        == "image_and_caption_records_preserved_not_validation_evidence"
    )
    assert (
        classify_meta_framework_psi_coupling_component("repeated_ontology_block")
        == "repeated_tripartite_ontology_block_preserved"
    )

    with pytest.raises(ValueError, match="unknown meta-framework Psi-coupling component"):
        classify_meta_framework_psi_coupling_component("category_theory")


def test_meta_framework_psi_coupling_fixture_preserves_claim_boundary_and_null_controls() -> None:
    result = validate_meta_framework_psi_coupling_fixture()

    assert result.source_ledger_span == ("P0R00838", "P0R00904")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 67
    assert result.predictive_coding_record_count == 14
    assert result.psi_coupling_record_count == 25
    assert result.formal_restatement_record_count == 22
    assert result.image_or_figure_record_count == 6
    assert result.blank_record_count == 2
    assert result.next_source_boundary == "P0R00905"
    assert result.null_controls == {
        "meta_framework_psi_coupling_is_source_claim_not_empirical_evidence": 1.0,
        "image_and_figure_records_are_not_promoted_as_validation_evidence": 1.0,
        "blank_records_p0r00875_p0r00897_are_preserved": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_meta_framework_psi_coupling_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00838"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00904"


def test_meta_framework_psi_coupling_labels_name_category_theory_boundary() -> None:
    labels = meta_framework_psi_coupling_labels()

    assert labels["section"] == "Meta-Framework Integrations"
    assert labels["predictive_coding"] == "fibre bundle state space of beliefs"
    assert labels["hamiltonian"] == "H_int = -lambda * Psis * sigma"
    assert labels["projection"] == "total space to fibre projection with G to H transduction"
    assert labels["source_integrity"] == "P0R00875 blank; P0R00897 blank"
    assert labels["next_boundary"] == "1.5 The Universal Grammar: A Category-Theoretic Foundation"
