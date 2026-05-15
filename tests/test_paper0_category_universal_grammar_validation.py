# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 category universal grammar validation tests
"""Tests for Paper 0 category/universal-grammar validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.category_universal_grammar_validation import (
    CategoryUniversalGrammarConfig,
    category_universal_grammar_labels,
    classify_category_universal_grammar_component,
    validate_category_universal_grammar_fixture,
)


def test_category_universal_grammar_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 82"):
        CategoryUniversalGrammarConfig(expected_source_record_count=81)
    with pytest.raises(ValueError, match="expected_blank_record_count must equal 3"):
        CategoryUniversalGrammarConfig(expected_blank_record_count=2)
    with pytest.raises(ValueError, match="expected_image_record_count must equal 5"):
        CategoryUniversalGrammarConfig(expected_image_record_count=4)
    with pytest.raises(ValueError, match="expected_figure_caption_record_count must equal 11"):
        CategoryUniversalGrammarConfig(expected_figure_caption_record_count=10)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00987"):
        CategoryUniversalGrammarConfig(next_source_boundary="P0R00986")


def test_category_universal_grammar_classifiers_are_source_bounded() -> None:
    assert (
        classify_category_universal_grammar_component("section_boundary")
        == "category_theory_universal_grammar_boundary"
    )
    assert (
        classify_category_universal_grammar_component("category_objects_morphisms")
        == "scpn_category_objects_morphisms_identity_composition"
    )
    assert (
        classify_category_universal_grammar_component("functor_bridge")
        == "consciousness_physics_functors_and_natural_transformation"
    )
    assert (
        classify_category_universal_grammar_component("topos_kan_foundation")
        == "topos_classifier_exponential_and_kan_inference_foundation"
    )
    assert (
        classify_category_universal_grammar_component("explanatory_analogies")
        == "lay_analogies_preserved_not_validation_evidence"
    )
    assert (
        classify_category_universal_grammar_component("meta_framework_integrations")
        == "category_predictive_coding_and_psi_coupling_integrations"
    )
    assert (
        classify_category_universal_grammar_component("formal_diagram_records")
        == "image_caption_and_diagram_records_preserved_not_evidence"
    )
    assert (
        classify_category_universal_grammar_component("kan_extension_inference")
        == "lan_ran_and_psi_inferred_missing_data_formulas"
    )
    with pytest.raises(ValueError, match="unknown category universal-grammar component"):
        classify_category_universal_grammar_component("physical_sector")


def test_category_universal_grammar_fixture_preserves_claim_boundary_and_null_controls() -> None:
    result = validate_category_universal_grammar_fixture()

    assert result.source_ledger_span == ("P0R00905", "P0R00986")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 82
    assert result.blank_record_count == 3
    assert result.image_record_count == 5
    assert result.figure_caption_record_count == 11
    assert result.formal_category_record_count == 27
    assert result.meta_framework_record_count == 15
    assert result.next_source_boundary == "P0R00987"
    assert result.null_controls == {
        "category_universal_grammar_is_source_claim_not_empirical_evidence": 1.0,
        "image_and_caption_records_are_not_promoted_as_validation_evidence": 1.0,
        "blank_records_p0r00915_p0r00968_p0r00978_are_preserved": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_category_universal_grammar_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00905"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00986"


def test_category_universal_grammar_labels_name_physical_sector_boundary() -> None:
    labels = category_universal_grammar_labels()

    assert labels["section"] == "1.5 The Universal Grammar: A Category-Theoretic Foundation"
    assert labels["category"] == "SCPN category with objects L1...L15 and morphisms f: Li -> Lj"
    assert labels["functors"] == "F: Consciousness -> Physics; G: Physics -> Consciousness"
    assert labels["naturality"] == "eta: F => G"
    assert labels["kan"] == "Psi_inferred = Lan_physical(Psi_true)"
    assert labels["next_boundary"] == "Part II: The Physical Sector"
