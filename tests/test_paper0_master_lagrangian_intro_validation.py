# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 master Lagrangian intro validation tests
"""Tests for Paper 0 master-interaction-Lagrangian introduction validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.master_lagrangian_intro_validation import (
    MasterLagrangianIntroConfig,
    classify_master_lagrangian_intro_component,
    master_lagrangian_intro_labels,
    validate_master_lagrangian_intro_fixture,
)


def test_master_lagrangian_intro_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 31"):
        MasterLagrangianIntroConfig(expected_source_record_count=30)
    with pytest.raises(ValueError, match="expected_blank_record_count must equal 2"):
        MasterLagrangianIntroConfig(expected_blank_record_count=1)
    with pytest.raises(ValueError, match="expected_meta_framework_record_count must equal 16"):
        MasterLagrangianIntroConfig(expected_meta_framework_record_count=15)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01018"):
        MasterLagrangianIntroConfig(next_source_boundary="P0R01017")


def test_master_lagrangian_intro_classifiers_are_source_bounded() -> None:
    assert (
        classify_master_lagrangian_intro_component("part_ii_boundary")
        == "physical_sector_and_master_lagrangian_section_boundary"
    )
    assert (
        classify_master_lagrangian_intro_component("first_principles_framing")
        == "phenomenological_to_first_principles_claim_boundary"
    )
    assert (
        classify_master_lagrangian_intro_component("two_stream_derivation")
        == "u1_informational_and_curved_spacetime_geometric_derivation_claims"
    )
    assert (
        classify_master_lagrangian_intro_component("explanatory_analogies")
        == "lay_analogies_preserved_not_validation_evidence"
    )
    assert (
        classify_master_lagrangian_intro_component("gauge_inference_integration")
        == "gauge_invariance_and_infoton_prediction_error_inference_mapping"
    )
    assert (
        classify_master_lagrangian_intro_component("psis_coupling_gauge_interpretation")
        == "h_int_gauge_interaction_noether_current_and_dual_coupling_mapping"
    )
    with pytest.raises(ValueError, match="unknown master-Lagrangian-intro component"):
        classify_master_lagrangian_intro_component("covariant_derivative_detail")


def test_master_lagrangian_intro_fixture_preserves_claim_boundary_and_null_controls() -> None:
    result = validate_master_lagrangian_intro_fixture()

    assert result.source_ledger_span == ("P0R00987", "P0R01017")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 31
    assert result.blank_record_count == 2
    assert result.introduction_record_count == 13
    assert result.meta_framework_record_count == 16
    assert result.gauge_inference_record_count == 6
    assert result.psis_coupling_record_count == 9
    assert result.next_source_boundary == "P0R01018"
    assert result.null_controls == {
        "master_lagrangian_intro_is_source_claim_not_empirical_evidence": 1.0,
        "first_principles_language_is_not_proof_without_derivation_fixture": 1.0,
        "blank_records_p0r00988_p0r01001_are_preserved": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_master_lagrangian_intro_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00987"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01017"


def test_master_lagrangian_intro_labels_name_gauge_derivation_boundary() -> None:
    labels = master_lagrangian_intro_labels()

    assert labels["section"] == (
        "2.1 Master Interaction Lagrangian: Derivation from First Principles"
    )
    assert labels["informational"] == "local U(1) gauge invariance of complex scalar Psi"
    assert labels["mediator"] == "infoton gauge boson A_mu"
    assert labels["current"] == "J_mu = i(Psi* partial_mu Psi - Psi partial_mu Psi*)"
    assert labels["coupling"] == "H_int = -lambda * Psis * sigma with lambda = g"
    assert labels["next_boundary"] == "A Gauge-Principle Derivation of the Psi-Field"
