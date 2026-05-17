# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 derived Lagrangian detail validation tests
"""Tests for Paper 0 derived Lagrangian detail validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.derived_lagrangian_detail_validation import (
    DerivedLagrangianDetailConfig,
    classify_derived_lagrangian_detail_component,
    derived_lagrangian_detail_labels,
    validate_derived_lagrangian_detail_fixture,
)


def test_derived_lagrangian_detail_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 88"):
        DerivedLagrangianDetailConfig(expected_source_record_count=87)
    with pytest.raises(ValueError, match="expected_component_count must equal 7"):
        DerivedLagrangianDetailConfig(expected_component_count=6)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01510"):
        DerivedLagrangianDetailConfig(next_source_boundary="P0R01509")


def test_derived_lagrangian_detail_classifiers_are_source_bounded() -> None:
    assert (
        classify_derived_lagrangian_detail_component("derived_lint_split")
        == "derived_lint_prime_informational_geometric_split_boundary"
    )
    assert (
        classify_derived_lagrangian_detail_component("informational_lagrangian_fim_kinetics")
        == "informational_lagrangian_fim_gauge_kinetic_boundary"
    )
    assert (
        classify_derived_lagrangian_detail_component("operational_pullback_protocol")
        == "statistical_bundle_fim_pullback_gauge_protocol_boundary"
    )
    assert (
        classify_derived_lagrangian_detail_component("observable_l4_l5_prediction")
        == "observable_l4_l5_fim_prediction_only_boundary"
    )
    assert (
        classify_derived_lagrangian_detail_component("neural_fim_covariance_strategy")
        == "full_covariance_neural_fim_strategy_boundary"
    )
    assert (
        classify_derived_lagrangian_detail_component("domain_constraints_local_physics")
        == "eft_lorentz_locality_causality_pullback_boundary"
    )
    assert (
        classify_derived_lagrangian_detail_component("geometric_constants_predictions")
        == "geometric_coupling_constants_prediction_target_boundary"
    )
    with pytest.raises(ValueError, match="unknown derived Lagrangian detail component"):
        classify_derived_lagrangian_detail_component("final_lint_restart")


def test_derived_lagrangian_detail_fixture_preserves_claim_boundary() -> None:
    result = validate_derived_lagrangian_detail_fixture()

    assert result.source_ledger_span == ("P0R01422", "P0R01509")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 88
    assert result.component_count == 7
    assert result.next_source_boundary == "P0R01510"
    assert result.null_controls == {
        "spacetime_metric_contraction_rejected_for_fim_gauge_kinetic_term": 1.0,
        "nv_centre_sensor_prediction_is_not_observed_evidence": 1.0,
        "nonlocal_or_acausal_pullback_dependency_rejected": 1.0,
        "psi_higgs_and_alp_targets_are_not_detected_particles": 1.0,
        "mean_only_or_diagonal_fim_shortcut_rejected": 1.0,
    }
    assert (
        result.problem_metadata["protocol_state"]
        == "source_derived_lagrangian_detail_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01422"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01509"


def test_derived_lagrangian_detail_labels_name_next_restart_boundary() -> None:
    labels = derived_lagrangian_detail_labels()

    assert labels["section"] == "The Master Interaction Lagrangian (Derived from First Principles)"
    assert labels["lint_split"] == "L_Int_prime = L_Informational_prime + L_Geometric_prime"
    assert labels["geometric"] == "L_Geometric_prime = -xi R Psi* Psi"
    assert (
        labels["next_boundary"]
        == "The Master Interaction Lagrangian (Derived from First Principles) restart"
    )
