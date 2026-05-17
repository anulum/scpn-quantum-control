# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Non-Abelian qualia field validation tests
"""Tests for Paper 0 Non-Abelian qualia-field validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.non_abelian_qualia_field_validation import (
    NonAbelianQualiaFieldConfig,
    classify_non_abelian_qualia_field_component,
    non_abelian_qualia_field_labels,
    validate_non_abelian_qualia_field_fixture,
)


def test_non_abelian_qualia_field_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 32"):
        NonAbelianQualiaFieldConfig(expected_source_record_count=31)
    with pytest.raises(ValueError, match="expected_anomaly_condition_record_count must equal 10"):
        NonAbelianQualiaFieldConfig(expected_anomaly_condition_record_count=9)
    with pytest.raises(ValueError, match="expected_confinement_record_count must equal 9"):
        NonAbelianQualiaFieldConfig(expected_confinement_record_count=8)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01135"):
        NonAbelianQualiaFieldConfig(next_source_boundary="P0R01134")


def test_non_abelian_qualia_field_classifiers_are_source_bounded() -> None:
    assert (
        classify_non_abelian_qualia_field_component("boundary_and_rationale")
        == "u1_to_su_n_qualia_field_hypothesis_boundary"
    )
    assert (
        classify_non_abelian_qualia_field_component("self_interacting_gauge_bosons")
        == "non_abelian_gauge_boson_multiplicity_and_self_interaction"
    )
    assert (
        classify_non_abelian_qualia_field_component("anomaly_cancellation_condition")
        == "su_n_qualia_colour_anomaly_cancellation_constraint"
    )
    assert (
        classify_non_abelian_qualia_field_component("confinement_binding_boundary")
        == "qcd_analogue_confinement_and_binding_problem_claim_boundary"
    )
    assert (
        classify_non_abelian_qualia_field_component("topological_entanglement_resolution")
        == "topological_entanglement_and_qualia_ball_prediction_boundary"
    )
    with pytest.raises(ValueError, match="unknown Non-Abelian qualia-field component"):
        classify_non_abelian_qualia_field_component("geometric_coupling")


def test_non_abelian_qualia_field_fixture_preserves_claim_boundary_and_null_controls() -> None:
    result = validate_non_abelian_qualia_field_fixture()

    assert result.source_ledger_span == ("P0R01103", "P0R01134")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 32
    assert result.structural_record_count == 2
    assert result.context_record_count == 13
    assert result.claim_record_count == 13
    assert result.validation_target_record_count == 4
    assert result.anomaly_condition_record_count == 10
    assert result.confinement_record_count == 9
    assert result.topological_entanglement_record_count == 7
    assert result.next_source_boundary == "P0R01135"
    assert result.null_controls == {
        "non_abelian_qualia_field_is_source_hypothesis_not_empirical_evidence": 1.0,
        "qcd_analogy_alone_is_not_biological_confinement_validation": 1.0,
        "trivial_singlet_neutrality_is_preserved_as_rejected_mapping": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_non_abelian_qualia_field_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01103"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01134"


def test_non_abelian_qualia_field_labels_name_geometric_coupling_boundary() -> None:
    labels = non_abelian_qualia_field_labels()

    assert labels["section"] == "Beyond U(1): The Hypothesis of a Non-Abelian Qualia Field"
    assert labels["field_strength"] == "F_mu_nu^a includes g f_abc A_mu^b A_nu^c"
    assert labels["anomaly_condition"] == "sum d_abc q_i_a q_i_b q_i_c == 0"
    assert labels["confinement"] == "Qualia confinement remains QCD-analogue source claim"
    assert labels["next_boundary"] == "Consistency Conditions and the Origin of Geometric Coupling"
