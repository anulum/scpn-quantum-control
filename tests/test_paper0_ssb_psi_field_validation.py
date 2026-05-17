# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 SSB Psi-field validation tests
"""Tests for Paper 0 SSB Psi-field validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ssb_psi_field_validation import (
    SSBPsiFieldConfig,
    classify_ssb_psi_field_component,
    ssb_psi_field_labels,
    validate_ssb_psi_field_fixture,
)


def test_ssb_psi_field_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 61"):
        SSBPsiFieldConfig(expected_source_record_count=60)
    with pytest.raises(ValueError, match="expected_component_count must equal 8"):
        SSBPsiFieldConfig(expected_component_count=7)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01333"):
        SSBPsiFieldConfig(next_source_boundary="P0R01332")


def test_ssb_psi_field_classifiers_are_source_bounded() -> None:
    assert (
        classify_ssb_psi_field_component("section_overview_and_three_implications")
        == "ssb_psi_field_section_overview_boundary"
    )
    assert (
        classify_ssb_psi_field_component("popular_context_short_range_particle_self")
        == "popular_context_not_validation_boundary"
    )
    assert (
        classify_ssb_psi_field_component("predictive_coding_core_belief")
        == "predictive_coding_mapping_source_boundary"
    )
    assert (
        classify_ssb_psi_field_component("psi_s_coupling_integration")
        == "h_int_psi_s_sigma_coupling_integration_boundary"
    )
    assert (
        classify_ssb_psi_field_component("mexican_hat_vacuum_selection")
        == "mexican_hat_potential_and_vacuum_selection_boundary"
    )
    assert (
        classify_ssb_psi_field_component("eft_sextic_stability_and_mass")
        == "eft_sextic_stability_and_radial_mass_boundary"
    )
    assert (
        classify_ssb_psi_field_component("global_goldstone_boundary")
        == "global_u1_goldstone_counterfactual_boundary"
    )
    assert (
        classify_ssb_psi_field_component("local_higgs_architecture_implications")
        == "local_u1_higgs_infoton_architecture_boundary"
    )
    with pytest.raises(ValueError, match="unknown SSB Psi-field component"):
        classify_ssb_psi_field_component("phenomenological_lagrangian")


def test_ssb_psi_field_fixture_preserves_claim_boundary() -> None:
    result = validate_ssb_psi_field_fixture()

    assert result.source_ledger_span == ("P0R01272", "P0R01332")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 61
    assert result.component_count == 8
    assert result.next_source_boundary == "P0R01333"
    assert result.null_controls == {
        "psi_higgs_prediction_is_not_particle_detection": 1.0,
        "global_goldstone_case_must_not_be_mixed_with_local_higgs_case": 1.0,
        "quartic_only_potential_rejected_for_eft_stability_boundary": 1.0,
        "popular_context_is_not_empirical_validation": 1.0,
    }
    assert (
        result.problem_metadata["protocol_state"]
        == "source_ssb_psi_field_mechanism_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01272"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01332"


def test_ssb_psi_field_labels_name_next_phenomenological_boundary() -> None:
    labels = ssb_psi_field_labels()

    assert (
        labels["section"] == "The Physics of Form: Spontaneous Symmetry Breaking and the Psi-Field"
    )
    assert labels["potential"] == "V(|Psi|) = -mu^2 |Psi|^2 + lambda |Psi|^4"
    assert labels["eft_potential"] == (
        "V(|Psi|) = -mu^2 |Psi|^2 + lambda |Psi|^4 + (gamma/Lambda^2)|Psi|^6"
    )
    assert labels["higgs_mass"] == "m_A = sqrt(2) g v"
    assert (
        labels["next_boundary"]
        == "The Phenomenological Formulation: An Evolutionary Starting Point"
    )
