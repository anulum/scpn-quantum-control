# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I meta-coupling validation tests
"""Tests for Paper 0 Axiom I meta-framework and coupling source validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.axiom_i_meta_coupling_validation import (
    AxiomIMetaCouplingConfig,
    axiom_i_meta_coupling_labels,
    classify_coupling_requirement,
    classify_predictive_hardware_role,
    validate_axiom_i_meta_coupling_fixture,
)


def test_meta_coupling_config_rejects_boundary_drift() -> None:
    with pytest.raises(ValueError, match="expected_interaction_component_count must equal 3"):
        AxiomIMetaCouplingConfig(expected_interaction_component_count=2)

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00733"):
        AxiomIMetaCouplingConfig(next_source_boundary="P0R00734")


def test_meta_coupling_classifiers_are_source_bounded() -> None:
    assert classify_predictive_hardware_role("spin0") == "minimal_processor_components"
    assert classify_predictive_hardware_role("phase") == "belief_prior_carrier_updated_by_u1"
    assert classify_predictive_hardware_role("soliton") == "persistent_deep_prior_self"
    assert classify_coupling_requirement("psi_s") == "complex_scalar_stability_intentionality"
    assert classify_coupling_requirement("gauge") == "local_well_behaved_infoton_mediation"
    assert classify_coupling_requirement("sigma") == "stable_charge_supported_q_ball_self"

    with pytest.raises(ValueError, match="unknown predictive-hardware role"):
        classify_predictive_hardware_role("memory_bus")
    with pytest.raises(ValueError, match="unknown coupling requirement"):
        classify_coupling_requirement("bare_contact")


def test_meta_coupling_fixture_preserves_null_controls_and_labels() -> None:
    result = validate_axiom_i_meta_coupling_fixture()

    assert result.source_ledger_span == ("P0R00717", "P0R00732")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.predictive_hardware_role_count == 3
    assert result.interaction_component_count == 3
    assert result.next_source_boundary == "P0R00733"
    assert result.null_controls == {
        "predictive_coding_hardware_is_not_empirical_validation": 1.0,
        "hint_component_justification_requires_lagrangian_tests": 1.0,
        "necessity_language_requires_downstream_falsification": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == "source_meta_coupling_only_no_experiment"
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00717"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00732"


def test_meta_coupling_labels_name_minimal_lagrangian_boundary() -> None:
    labels = axiom_i_meta_coupling_labels()

    assert labels["section"] == "Meta-Framework Integrations and Psi-s Coupling"
    assert labels["h_int"] == "H_int = -lambda * Psi_s * sigma"
    assert (
        labels["next_boundary"]
        == "Model-Class Justification: From Axiom 1 to a Minimal Psi-Field Lagrangian"
    )
