# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 category grammar fixture tests
"""Tests for Paper 0 category grammar formal-consistency fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.category_grammar_validation import (
    CategoryGrammarConfig,
    compose_morphisms,
    finite_category_law_summary,
    identity_morphism,
    subobject_truth_values,
    validate_category_grammar_fixture,
    validate_functorial_mapping,
)


def test_category_laws_hold_for_finite_layer_fixture() -> None:
    f = ("L1", "L2", "f12")
    g = ("L2", "L3", "f23")
    h = ("L3", "L4", "f34")

    assert identity_morphism("L1") == ("L1", "L1", "id_L1")
    assert compose_morphisms(identity_morphism("L1"), f) == f
    assert compose_morphisms(f, identity_morphism("L2")) == f
    assert compose_morphisms(compose_morphisms(f, g), h) == compose_morphisms(
        f, compose_morphisms(g, h)
    )

    summary = finite_category_law_summary(("L1", "L2", "L3", "L4"))

    assert summary["identity_law"] is True
    assert summary["associativity_law"] is True
    assert summary["layer_count"] == 4


def test_category_grammar_functor_and_truth_value_guards() -> None:
    assert subobject_truth_values() == ("true", "false", "uncertain")
    assert (
        validate_functorial_mapping(
            object_map={"L1": "physics:L1", "L2": "physics:L2", "L3": "physics:L3"},
            morphisms=(("L1", "L2", "f12"), ("L2", "L3", "f23")),
        )
        is True
    )

    with pytest.raises(ValueError, match="morphisms are not composable"):
        compose_morphisms(("L1", "L2", "f12"), ("L4", "L5", "f45"))
    with pytest.raises(ValueError, match="layer_count must be at least 2"):
        CategoryGrammarConfig(layer_count=1)
    with pytest.raises(ValueError, match="object_map is missing morphism endpoint"):
        validate_functorial_mapping(
            object_map={"L1": "physics:L1"},
            morphisms=(("L1", "L2", "f12"),),
        )


def test_category_grammar_fixture_preserves_claim_boundary() -> None:
    result = validate_category_grammar_fixture()

    assert result.spec_keys == (
        "integration_synthesis.category_grammar.block_boundary",
        "integration_synthesis.category_grammar.scpn_category",
        "integration_synthesis.category_grammar.functorial_mappings",
        "integration_synthesis.category_grammar.topos_internal_logic",
        "integration_synthesis.category_grammar.kan_inference_mechanism",
        "integration_synthesis.category_grammar.string_diagram_calculus",
        "integration_synthesis.category_grammar.upde_category_application",
        "integration_synthesis.category_grammar.theorem_obligation_boundary",
    )
    assert result.hardware_status == "formal_consistency_fixture_no_execution"
    assert result.source_ledger_span == ("P0R06815", "P0R06877")
    assert result.layer_count == 16
    assert result.category_laws["identity_law"] is True
    assert result.category_laws["associativity_law"] is True
    assert result.functorial_mapping_valid is True
    assert result.natural_transformation_boundary == "formal square-commutation obligation"
    assert result.truth_values == ("true", "false", "uncertain")
    assert result.null_controls["noncomposable_morphism_rejection_label"] == 1.0
    assert result.null_controls["missing_functor_endpoint_rejection_label"] == 1.0
    assert result.null_controls["unsupported_empirical_claim_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
