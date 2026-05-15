# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I family predictions validation tests
"""Tests for Paper 0 Axiom I family-satisfaction and prediction validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.axiom_i_family_predictions_validation import (
    AxiomIFamilyPredictionsConfig,
    axiom_i_family_predictions_labels,
    classify_conditional_prediction,
    classify_rejected_family,
    validate_axiom_i_family_predictions_fixture,
)


def test_family_predictions_config_rejects_boundary_drift() -> None:
    with pytest.raises(ValueError, match="expected_conditional_prediction_count must equal 3"):
        AxiomIFamilyPredictionsConfig(expected_conditional_prediction_count=2)

    with pytest.raises(ValueError, match="expected_rejected_model_class_count must equal 5"):
        AxiomIFamilyPredictionsConfig(expected_rejected_model_class_count=4)

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00757"):
        AxiomIFamilyPredictionsConfig(next_source_boundary="P0R00756")


def test_family_predictions_classifiers_are_source_bounded() -> None:
    assert classify_rejected_family("real_scalar") == "lacks_phase_charge_and_solitons"
    assert classify_rejected_family("global_u1") == "long_range_goldstone_no_locality"
    assert classify_rejected_family("vector_tensor") == "unnecessary_lorentz_structure"
    assert classify_rejected_family("spinor") == "spin_representation_not_universal_fibre"
    assert classify_rejected_family("non_abelian_minimal") == "deferred_su_n_field_content"
    assert classify_conditional_prediction("psi_charge") == "conserved_noether_current_q_psi"
    assert classify_conditional_prediction("massive_infoton") == "ssb_mass_m_a_equals_g_v"
    assert classify_conditional_prediction("psi_higgs") == "massive_radial_spin0_excitation"

    with pytest.raises(ValueError, match="unknown rejected model family"):
        classify_rejected_family("unbounded_scalar")
    with pytest.raises(ValueError, match="unknown conditional prediction"):
        classify_conditional_prediction("dark_photon")


def test_family_predictions_fixture_preserves_risk_boundary_and_null_controls() -> None:
    result = validate_axiom_i_family_predictions_fixture()

    assert result.source_ledger_span == ("P0R00747", "P0R00756")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.conditional_prediction_count == 3
    assert result.rejected_model_class_count == 5
    assert result.blank_separator_count == 1
    assert result.next_source_boundary == "P0R00757"
    assert result.null_controls == {
        "conditional_predictions_are_not_observed_results": 1.0,
        "rejected_model_classes_remain_source_boundary_claims": 1.0,
        "su_n_extension_header_is_not_promoted_model_selection": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_family_predictions_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00747"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00756"


def test_family_predictions_labels_name_su_n_extension_boundary() -> None:
    labels = axiom_i_family_predictions_labels()

    assert labels["section"] == "Why this family satisfies (i)-(iii)"
    assert (
        labels["decision_rule"] == "model-class escalation or replacement after contrary evidence"
    )
    assert labels["next_boundary"] == "Extension to SU(N) Qualia Confinement"
