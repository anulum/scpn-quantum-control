# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 derived interaction opening validation tests
"""Tests for Paper 0 derived interaction opening validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.derived_interaction_opening_validation import (
    DerivedInteractionOpeningConfig,
    classify_derived_interaction_opening_component,
    derived_interaction_opening_labels,
    validate_derived_interaction_opening_fixture,
)


def test_derived_interaction_opening_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 38"):
        DerivedInteractionOpeningConfig(expected_source_record_count=37)
    with pytest.raises(ValueError, match="expected_component_count must equal 5"):
        DerivedInteractionOpeningConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01422"):
        DerivedInteractionOpeningConfig(next_source_boundary="P0R01421")


def test_derived_interaction_opening_classifiers_are_source_bounded() -> None:
    assert (
        classify_derived_interaction_opening_component("gauge_theory_grounding")
        == "complex_scalar_u1_infoton_grounding_boundary"
    )
    assert (
        classify_derived_interaction_opening_component("predictive_coding_mapping")
        == "psi_charge_belief_infoton_prediction_error_mapping_boundary"
    )
    assert (
        classify_derived_interaction_opening_component("h_int_gauge_identification")
        == "h_int_to_u1_gauge_interaction_identification_boundary"
    )
    assert (
        classify_derived_interaction_opening_component("intrinsic_properties_quantum_numbers")
        == "spin_zero_psi_charge_infoton_fim_quantum_number_boundary"
    )
    assert (
        classify_derived_interaction_opening_component("gauge_principle_nonabelian_boundary")
        == "local_u1_su_n_hypothesis_fim_dynamics_boundary"
    )
    with pytest.raises(ValueError, match="unknown derived interaction opening component"):
        classify_derived_interaction_opening_component("derived_lagrangian_terms")


def test_derived_interaction_opening_fixture_preserves_claim_boundary() -> None:
    result = validate_derived_interaction_opening_fixture()

    assert result.source_ledger_span == ("P0R01384", "P0R01421")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 38
    assert result.component_count == 5
    assert result.next_source_boundary == "P0R01422"
    assert result.null_controls == {
        "phenomenological_black_box_h_int_rejected_for_derived_boundary": 1.0,
        "predictive_coding_mapping_is_not_observed_infoton_signal": 1.0,
        "su_n_qualia_confinement_remains_hypothesis_not_established_gauge_group": 1.0,
        "diagram_caption_is_not_particle_detection_evidence": 1.0,
    }
    assert (
        result.problem_metadata["protocol_state"]
        == "source_derived_interaction_opening_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01384"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01421"


def test_derived_interaction_opening_labels_name_next_lagrangian_boundary() -> None:
    labels = derived_interaction_opening_labels()

    assert labels["section"] == "Deriving the Master Interaction Lagrangian"
    assert labels["interaction"] == "L_interaction = i g A_mu J_mu"
    assert labels["current"] == "J_mu = Psi* partial_mu Psi - Psi partial_mu Psi*"
    assert (
        labels["next_boundary"]
        == "The Master Interaction Lagrangian (Derived from First Principles)"
    )
