# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 predicted particles validation tests
"""Tests for Paper 0 infoton and Psi-Higgs prediction validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.predicted_particles_infoton_psi_higgs_validation import (
    PredictedParticlesInfotonPsiHiggsConfig,
    classify_predicted_particles_infoton_psi_higgs_component,
    predicted_particles_infoton_psi_higgs_labels,
    validate_predicted_particles_infoton_psi_higgs_fixture,
)


def test_predicted_particles_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 26"):
        PredictedParticlesInfotonPsiHiggsConfig(expected_source_record_count=25)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        PredictedParticlesInfotonPsiHiggsConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01623"):
        PredictedParticlesInfotonPsiHiggsConfig(next_source_boundary="P0R01622")


def test_predicted_particles_classifiers_are_source_bounded() -> None:
    assert (
        classify_predicted_particles_infoton_psi_higgs_component("particle_prediction_opening")
        == "u1_ssb_infoton_psi_higgs_prediction_boundary"
    )
    assert (
        classify_predicted_particles_infoton_psi_higgs_component("search_strategy_summary")
        == "collider_cosmology_search_strategy_claim_boundary"
    )
    assert (
        classify_predicted_particles_infoton_psi_higgs_component("active_inference_mapping")
        == "active_inference_particle_role_mapping_boundary"
    )
    assert (
        classify_predicted_particles_infoton_psi_higgs_component("h_int_falsifiability_bridge")
        == "h_int_parameter_falsifiability_bridge_boundary"
    )
    with pytest.raises(ValueError, match="unknown predicted-particles component"):
        classify_predicted_particles_infoton_psi_higgs_component("infoton_derivation")


def test_predicted_particles_fixture_preserves_claim_boundary() -> None:
    result = validate_predicted_particles_infoton_psi_higgs_fixture()

    assert result.source_ledger_span == ("P0R01597", "P0R01622")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 26
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R01623"
    assert result.null_controls == {
        "predicted_particles_are_not_observed_discoveries": 1.0,
        "lhc_and_lisa_search_channels_are_not_completed_evidence": 1.0,
        "active_inference_particle_mapping_is_not_neural_measurement": 1.0,
        "h_int_falsifiability_bridge_is_not_empirical_detection": 1.0,
    }
    assert (
        result.problem_metadata["protocol_state"]
        == "source_predicted_particles_infoton_psi_higgs_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01597"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01622"


def test_predicted_particles_labels_name_next_infoton_derivation_boundary() -> None:
    labels = predicted_particles_infoton_psi_higgs_labels()

    assert labels["section"] == "Predicted Particles: The Infoton and the Psi-Higgs Boson"
    assert labels["infoton_mass"] == "m_A = g v"
    assert labels["psi_higgs_mass"] == "m_h = sqrt(2 lambda) v"
    assert labels["search_channels"] == "LHC and gravitational-wave signatures"
    assert labels["interaction"] == "H_int = -lambda * Psi_s * sigma"
    assert labels["next_boundary"] == "Derivation of the Infoton's Properties"
