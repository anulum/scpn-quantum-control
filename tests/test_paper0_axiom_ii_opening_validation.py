# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom II opening validation tests
"""Tests for Paper 0 Axiom II opening and interaction-axiom validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.axiom_ii_opening_validation import (
    AxiomIIOpeningConfig,
    axiom_ii_opening_labels,
    classify_axiom_ii_component,
    validate_axiom_ii_opening_fixture,
)


def test_axiom_ii_opening_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_heading_record_count must equal 4"):
        AxiomIIOpeningConfig(expected_heading_record_count=3)

    with pytest.raises(ValueError, match="expected_axiom_statement_count must equal 1"):
        AxiomIIOpeningConfig(expected_axiom_statement_count=0)

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00770"):
        AxiomIIOpeningConfig(next_source_boundary="P0R00769")


def test_axiom_ii_opening_classifiers_are_source_bounded() -> None:
    assert classify_axiom_ii_component("section_headings") == "axiom_ii_navigation_headings"
    assert classify_axiom_ii_component("source_material") == "interactions_as_information_geometry"
    assert (
        classify_axiom_ii_component("ontology_to_dynamics")
        == "psi_field_substance_to_interaction_language"
    )
    assert (
        classify_axiom_ii_component("interaction_axiom")
        == "informational_geometric_falsifiable_hypothesis"
    )

    with pytest.raises(ValueError, match="unknown Axiom II opening component"):
        classify_axiom_ii_component("fim_solution_detail")


def test_axiom_ii_opening_fixture_preserves_claim_boundary_and_null_controls() -> None:
    result = validate_axiom_ii_opening_fixture()

    assert result.source_ledger_span == ("P0R00761", "P0R00769")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 9
    assert result.heading_record_count == 4
    assert result.axiom_statement_count == 1
    assert result.falsifiability_boundary_count == 1
    assert result.next_source_boundary == "P0R00770"
    assert result.null_controls == {
        "axiom_statement_is_not_empirical_validation": 1.0,
        "falsifiability_boundary_requires_downstream_protocols": 1.0,
        "information_geometry_claim_is_source_hypothesis_only": 1.0,
    }
    assert (
        result.problem_metadata["protocol_state"] == "source_axiom_ii_opening_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00761"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00769"


def test_axiom_ii_opening_labels_name_infoton_boundary() -> None:
    labels = axiom_ii_opening_labels()

    assert labels["section"] == "Axiom II: The Language of Information Geometry"
    assert labels["axiom"] == "Axiom II: The Axiom of Interaction (Information Geometry)"
    assert labels["next_boundary"] == 'The Central Problem: The Geometry of the "Infoton"'
