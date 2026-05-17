# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 final LInt SM interface validation tests
"""Tests for Paper 0 final LInt and SM-interface validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.final_lint_sm_interface_validation import (
    FinalLIntSMInterfaceConfig,
    classify_final_lint_sm_interface_component,
    final_lint_sm_interface_labels,
    validate_final_lint_sm_interface_fixture,
)


def test_final_lint_sm_interface_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 72"):
        FinalLIntSMInterfaceConfig(expected_source_record_count=71)
    with pytest.raises(ValueError, match="expected_component_count must equal 6"):
        FinalLIntSMInterfaceConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01582"):
        FinalLIntSMInterfaceConfig(next_source_boundary="P0R01581")


def test_final_lint_sm_interface_classifiers_are_source_bounded() -> None:
    assert (
        classify_final_lint_sm_interface_component("final_lint_dual_clause")
        == "final_lint_geometric_informational_dual_clause_boundary"
    )
    assert (
        classify_final_lint_sm_interface_component("free_energy_and_h_int_mapping")
        == "free_energy_h_int_mapping_source_boundary"
    )
    assert (
        classify_final_lint_sm_interface_component("foundational_physics_equations")
        == "compact_lint_foundational_physics_equation_boundary"
    )
    assert (
        classify_final_lint_sm_interface_component("standard_model_indirect_coupling")
        == "standard_model_indirect_coupling_no_direct_force_boundary"
    )
    assert (
        classify_final_lint_sm_interface_component("predictive_interface_mapping")
        == "predictive_downward_causation_interface_mapping_boundary"
    )
    assert (
        classify_final_lint_sm_interface_component("downstream_sm_manifestations")
        == "downstream_sm_manifestation_exploratory_hypothesis_boundary"
    )
    with pytest.raises(ValueError, match="unknown final LInt SM-interface component"):
        classify_final_lint_sm_interface_component("symmetry_cascade")


def test_final_lint_sm_interface_fixture_preserves_claim_boundary() -> None:
    result = validate_final_lint_sm_interface_fixture()

    assert result.source_ledger_span == ("P0R01510", "P0R01581")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 72
    assert result.component_count == 6
    assert result.next_source_boundary == "P0R01582"
    assert result.null_controls == {
        "direct_standard_model_force_carrier_claim_rejected": 1.0,
        "weak_force_and_alp_extensions_remain_exploratory_hypotheses": 1.0,
        "free_energy_mapping_is_not_measured_cost_function": 1.0,
        "prediction_mapping_is_not_observed_probability_bias": 1.0,
        "diagrammatic_sm_interface_is_not_experimental_evidence": 1.0,
    }
    assert (
        result.problem_metadata["protocol_state"]
        == "source_final_lint_sm_interface_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01510"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01581"


def test_final_lint_sm_interface_labels_name_next_symmetry_boundary() -> None:
    labels = final_lint_sm_interface_labels()

    assert labels["section"] == "The Master Interaction Lagrangian (Derived from First Principles)"
    assert labels["lint"] == "L_Int = L_Geometric + L_Informational"
    assert labels["heuristic"] == "H_int = -lambda * Psi_s * sigma"
    assert (
        labels["next_boundary"] == "How Reality Gets Its Structure: A Cascade of Broken Symmetries"
    )
