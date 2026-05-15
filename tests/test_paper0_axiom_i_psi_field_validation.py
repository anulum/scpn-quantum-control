# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I Psi-field validation tests
"""Tests for source-accounting validation of Paper 0 Axiom I records."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.axiom_i_psi_field_validation import (
    AxiomIPsiFieldConfig,
    axiom_i_psi_field_labels,
    classify_axiom_i_role,
    classify_psi_field_claim,
    validate_axiom_i_psi_field_fixture,
)


def test_axiom_i_psi_field_config_rejects_boundary_drift() -> None:
    with pytest.raises(ValueError, match="expected_blank_separator_count must equal 2"):
        AxiomIPsiFieldConfig(expected_blank_separator_count=1)

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00703"):
        AxiomIPsiFieldConfig(next_source_boundary="P0R00704")


def test_axiom_i_and_psi_field_classifiers_are_source_bounded() -> None:
    assert classify_axiom_i_role("ontological_primitive") == ("metaphysical_generative_postulate")
    assert classify_axiom_i_role("psi_field") == ("universal_complex_scalar_field_formalisation")
    assert classify_axiom_i_role("generative_model") == "cosmic_priors_physical_substrate"
    assert classify_axiom_i_role("hint_ground") == "psi_s_ontological_ground_not_peer_field"
    assert classify_psi_field_claim("not_emergent") == "matter_emerges_from_psi_not_reverse"
    assert classify_psi_field_claim("complex_scalar") == "universal_complex_scalar_field"
    assert classify_psi_field_claim("hierarchical_definition") == (
        "ontological_physical_experiential_layers"
    )

    with pytest.raises(ValueError, match="unknown Axiom I role"):
        classify_axiom_i_role("empirical_result")
    with pytest.raises(ValueError, match="unknown Psi-field claim"):
        classify_psi_field_claim("measured_signal")


def test_axiom_i_psi_field_fixture_preserves_null_controls_and_labels() -> None:
    result = validate_axiom_i_psi_field_fixture()

    assert result.source_ledger_span == ("P0R00670", "P0R00702")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.axiom_i_role_count == 4
    assert result.psi_field_claim_count == 3
    assert result.blank_separator_count == 2
    assert result.next_source_boundary == "P0R00703"
    assert result.null_controls == {
        "axiom_i_is_not_empirical_result": 1.0,
        "psi_field_formalisation_requires_downstream_model_tests": 1.0,
        "image_boundary_is_not_validation_evidence": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == "source_axiom_i_map_only_no_experiment"
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00670"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00702"


def test_axiom_i_psi_field_labels_name_next_model_class_boundary() -> None:
    labels = axiom_i_psi_field_labels()

    assert labels["section"] == "Axiom I: The Primacy of Consciousness (Psi)"
    assert labels["h_int"] == "H_int = -lambda * Psi_s * sigma"
    assert labels["next_boundary"] == "Model-Class Justification: From Axiom to Lagrangian"
