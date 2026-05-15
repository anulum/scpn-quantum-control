# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 tripartite ontology validation tests
"""Tests for Paper 0 tripartite ontology validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.tripartite_ontology_validation import (
    TripartiteOntologyConfig,
    classify_tripartite_ontology_component,
    tripartite_ontology_labels,
    validate_tripartite_ontology_fixture,
)


def test_tripartite_ontology_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 20"):
        TripartiteOntologyConfig(expected_source_record_count=19)

    with pytest.raises(ValueError, match="expected_blank_record_count must equal 2"):
        TripartiteOntologyConfig(expected_blank_record_count=1)

    with pytest.raises(ValueError, match="expected_information_form_count must equal 3"):
        TripartiteOntologyConfig(expected_information_form_count=2)

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00838"):
        TripartiteOntologyConfig(next_source_boundary="P0R00837")


def test_tripartite_ontology_classifiers_are_source_bounded() -> None:
    assert (
        classify_tripartite_ontology_component("section_boundary")
        == "tripartite_ontology_section_with_blank_boundary_record"
    )
    assert (
        classify_tripartite_ontology_component("psi_fibre_bundle")
        == "psi_field_section_of_fibre_bundle_over_spacetime"
    )
    assert (
        classify_tripartite_ontology_component("information_forms")
        == "phi_g_h_tripartite_information_ontology"
    )
    assert (
        classify_tripartite_ontology_component("bidirectional_transduction")
        == "phi_g_h_downward_and_h_g_phi_upward_transduction"
    )
    assert (
        classify_tripartite_ontology_component("grounded_platonism")
        == "mathematics_as_source_field_layer_13_logic"
    )
    assert (
        classify_tripartite_ontology_component("explanatory_analogies")
        == "lay_analogy_records_preserved_not_validation_evidence"
    )

    with pytest.raises(ValueError, match="unknown tripartite ontology component"):
        classify_tripartite_ontology_component("meta_framework_integrations")


def test_tripartite_ontology_fixture_preserves_claim_boundary_and_null_controls() -> None:
    result = validate_tripartite_ontology_fixture()

    assert result.source_ledger_span == ("P0R00818", "P0R00837")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 20
    assert result.blank_record_count == 2
    assert result.formal_ontology_record_count == 12
    assert result.explanatory_analogy_record_count == 6
    assert result.information_form_count == 3
    assert result.transduction_direction_count == 2
    assert result.next_source_boundary == "P0R00838"
    assert result.null_controls == {
        "tripartite_ontology_is_source_claim_not_empirical_evidence": 1.0,
        "blank_records_p0r00819_p0r00837_are_preserved": 1.0,
        "explanatory_analogies_are_not_promoted_as_validation_evidence": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_tripartite_ontology_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00818"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00837"


def test_tripartite_ontology_labels_name_meta_framework_boundary() -> None:
    labels = tripartite_ontology_labels()

    assert labels["section"] == "1.4 Tripartite Ontology: The Substance of Information"
    assert labels["psi_geometry"] == "Psi-field section of fibre bundle pi:E->M"
    assert labels["information_forms"] == "Phi experiential, G semantic/geometric, H syntactic"
    assert labels["transduction"] == "Phi -> G -> H and H -> G -> Phi"
    assert labels["source_integrity"] == "P0R00819 blank; P0R00837 blank"
    assert labels["next_boundary"] == "Meta-Framework Integrations"
