# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 operational pullback protocol validation tests
"""Tests for Paper 0 operational-pullback protocol validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.operational_pullback_protocol_validation import (
    OperationalPullbackProtocolConfig,
    classify_operational_pullback_protocol_component,
    operational_pullback_protocol_labels,
    validate_operational_pullback_protocol_fixture,
)


def test_operational_pullback_protocol_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 30"):
        OperationalPullbackProtocolConfig(expected_source_record_count=29)
    with pytest.raises(ValueError, match="expected_component_count must equal 6"):
        OperationalPullbackProtocolConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01272"):
        OperationalPullbackProtocolConfig(next_source_boundary="P0R01271")


def test_operational_pullback_protocol_classifiers_are_source_bounded() -> None:
    assert (
        classify_operational_pullback_protocol_component("section_and_protocol_boundary")
        == "ssb_section_operational_pullback_protocol_boundary"
    )
    assert (
        classify_operational_pullback_protocol_component("statistical_bundle_and_fim")
        == "statistical_bundle_section_and_fim_source_definition"
    )
    assert (
        classify_operational_pullback_protocol_component("spacetime_pullback_and_normalisation")
        == "fim_spacetime_pullback_and_lambda_i_normalisation"
    )
    assert (
        classify_operational_pullback_protocol_component("observable_sections_and_l4_l5_case")
        == "observable_sections_l4_l5_case_and_nv_prediction_boundary"
    )
    assert (
        classify_operational_pullback_protocol_component("full_covariance_fim_strategy")
        == "full_covariance_fim_computation_requirement"
    )
    assert (
        classify_operational_pullback_protocol_component("eft_lorentz_locality_constraints")
        == "eft_lorentz_locality_constraint_boundary"
    )
    with pytest.raises(ValueError, match="unknown operational-pullback protocol component"):
        classify_operational_pullback_protocol_component("ssb_psi_field")


def test_operational_pullback_protocol_fixture_preserves_claim_boundary() -> None:
    result = validate_operational_pullback_protocol_fixture()

    assert result.source_ledger_span == ("P0R01242", "P0R01271")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 30
    assert result.component_count == 6
    assert result.next_source_boundary == "P0R01272"
    assert result.null_controls == {
        "operational_pullback_protocol_is_source_protocol_not_measurement": 1.0,
        "nv_centre_prediction_is_not_experimental_evidence": 1.0,
        "diagonal_or_mean_only_fim_shortcut_rejected_for_full_covariance_protocol": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_operational_pullback_protocol_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01242"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01271"


def test_operational_pullback_protocol_labels_name_next_psi_boundary() -> None:
    labels = operational_pullback_protocol_labels()

    assert labels["section"] == "2.3 The Physics of Form: Spontaneous Symmetry Breaking"
    assert labels["protocol"] == "Operational Pullback Protocol Revision 11.00"
    assert labels["fim"] == "I_ij(theta) = E[partial_i log p partial_j log p]"
    assert labels["pullback"] == "g_F_mu_nu = partial_mu theta I partial_nu theta"
    assert labels["next_boundary"] == (
        "The Physics of Form: Spontaneous Symmetry Breaking and the Psi-Field"
    )
