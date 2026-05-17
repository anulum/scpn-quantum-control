# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 phenomenological Lagrangian validation tests
"""Tests for Paper 0 phenomenological Lagrangian validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.phenomenological_lagrangian_validation import (
    PhenomenologicalLagrangianConfig,
    classify_phenomenological_lagrangian_component,
    phenomenological_lagrangian_labels,
    validate_phenomenological_lagrangian_fixture,
)


def test_phenomenological_lagrangian_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 51"):
        PhenomenologicalLagrangianConfig(expected_source_record_count=50)
    with pytest.raises(ValueError, match="expected_component_count must equal 5"):
        PhenomenologicalLagrangianConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01384"):
        PhenomenologicalLagrangianConfig(next_source_boundary="P0R01383")


def test_phenomenological_lagrangian_classifiers_are_source_bounded() -> None:
    assert (
        classify_phenomenological_lagrangian_component("section_opening_dual_coupling")
        == "phenomenological_scaffold_dual_coupling_boundary"
    )
    assert (
        classify_phenomenological_lagrangian_component("predictive_coding_free_energy")
        == "heuristic_free_energy_mapping_boundary"
    )
    assert (
        classify_phenomenological_lagrangian_component("black_box_interaction")
        == "black_box_h_int_limitation_boundary"
    )
    assert (
        classify_phenomenological_lagrangian_component("master_interaction_terms")
        == "early_master_interaction_equation_boundary"
    )
    assert (
        classify_phenomenological_lagrangian_component("architecture_stationary_action")
        == "stationary_action_upde_scaffold_boundary"
    )
    with pytest.raises(ValueError, match="unknown phenomenological Lagrangian component"):
        classify_phenomenological_lagrangian_component("gauge_derivation")


def test_phenomenological_lagrangian_fixture_preserves_claim_boundary() -> None:
    result = validate_phenomenological_lagrangian_fixture()

    assert result.source_ledger_span == ("P0R01333", "P0R01383")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 51
    assert result.component_count == 5
    assert result.next_source_boundary == "P0R01384"
    assert result.null_controls == {
        "phenomenological_scaffold_is_not_final_gauge_derivation": 1.0,
        "black_box_h_int_must_not_satisfy_symmetry_derived_boundary": 1.0,
        "stationary_action_scaffold_is_not_complete_upde_proof": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_phenomenological_lagrangian_scaffold_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01333"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01383"


def test_phenomenological_lagrangian_labels_name_next_derivation_boundary() -> None:
    labels = phenomenological_lagrangian_labels()

    assert labels["section"] == "The Phenomenological Formulation: An Evolutionary Starting Point"
    assert labels["total_lagrangian"] == "L_Total = L_Psi + L_Physical + L_Int"
    assert labels["interaction_split"] == "L_Int = L_Geometric + L_Informational"
    assert labels["next_boundary"] == "Deriving the Master Interaction Lagrangian"
