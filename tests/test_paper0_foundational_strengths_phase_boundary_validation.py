# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 foundational strengths phase boundary validation tests
"""Tests for Paper 0 foundational-strengths and phase-boundary validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.foundational_strengths_phase_boundary_validation import (
    FoundationalStrengthsPhaseBoundaryConfig,
    classify_foundational_strengths_phase_boundary_component,
    foundational_strengths_phase_boundary_labels,
    validate_foundational_strengths_phase_boundary_fixture,
)


def test_foundational_strengths_phase_boundary_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 53"):
        FoundationalStrengthsPhaseBoundaryConfig(expected_source_record_count=52)
    with pytest.raises(ValueError, match="expected_component_count must equal 6"):
        FoundationalStrengthsPhaseBoundaryConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01242"):
        FoundationalStrengthsPhaseBoundaryConfig(next_source_boundary="P0R01241")


def test_foundational_strengths_phase_boundary_classifiers_are_source_bounded() -> None:
    assert (
        classify_foundational_strengths_phase_boundary_component("foundational_strengths")
        == "predictive_constrained_explanatory_falsifiable_source_claim_boundary"
    )
    assert (
        classify_foundational_strengths_phase_boundary_component("architecture_integration")
        == "fifteen_layer_architecture_integration_source_claim_boundary"
    )
    assert (
        classify_foundational_strengths_phase_boundary_component(
            "future_research_and_parameter_constraints"
        )
        == "future_research_parameter_constraint_queue_boundary"
    )
    assert (
        classify_foundational_strengths_phase_boundary_component("modulus_phase_decomposition")
        == "single_psi_modulus_phase_regime_decomposition"
    )
    assert (
        classify_foundational_strengths_phase_boundary_component("axion_analogy_and_em_interface")
        == "axion_analogy_phase_sensitive_em_interface_boundary"
    )
    assert (
        classify_foundational_strengths_phase_boundary_component(
            "gauge_choice_and_kinetic_phase_boundary"
        )
        == "fixed_phase_gravity_vs_released_phase_quantum_regime_boundary"
    )
    with pytest.raises(
        ValueError, match="unknown foundational-strengths phase-boundary component"
    ):
        classify_foundational_strengths_phase_boundary_component("operational_pullback")


def test_foundational_strengths_phase_boundary_fixture_preserves_claim_boundary() -> None:
    result = validate_foundational_strengths_phase_boundary_fixture()

    assert result.source_ledger_span == ("P0R01189", "P0R01241")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 53
    assert result.component_count == 6
    assert result.next_source_boundary == "P0R01242"
    assert result.null_controls == {
        "foundational_strengths_are_source_claims_not_validation_evidence": 1.0,
        "parameter_values_g_and_v_remain_unconstrained_without_external_bounds": 1.0,
        "fixed_phase_gravity_and_phase_varying_quantum_regimes_must_not_be_mixed": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_foundational_strengths_phase_boundary_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01189"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01241"


def test_foundational_strengths_phase_boundary_labels_name_ssb_boundary() -> None:
    labels = foundational_strengths_phase_boundary_labels()

    assert labels["section"] == "Foundational Strengths of the SCPN Lagrangian"
    assert labels["parameter_boundary"] == "g and v remain unconstrained source parameters"
    assert labels["phase_split"] == "Psi = (v + h) exp(i theta)"
    assert labels["alp_interface"] == "L_a_gamma_gamma = g_a_gamma_gamma a F F_tilde"
    assert labels["next_boundary"] == "2.3 The Physics of Form: Spontaneous Symmetry Breaking"
