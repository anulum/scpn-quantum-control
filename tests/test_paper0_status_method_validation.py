# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Status and Method fixtures
"""Tests for Paper 0 Status and Method validation fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.status_method_validation import (
    StatusMethodConfig,
    classify_scientific_inference_step,
    methodology_commitment_catalogue,
    normalise_status_method_interaction_formula,
    validate_status_method_fixture,
)


def test_status_method_interaction_formula_normaliser_preserves_parameters() -> None:
    assert normalise_status_method_interaction_formula("H_int = -λ * Ψs * σ") == (
        "H_int = -lambda * Psi_s * sigma"
    )

    with pytest.raises(ValueError, match="unsupported Status and Method interaction formula"):
        normalise_status_method_interaction_formula("H_int = -lambda * sigma")


def test_methodology_commitment_catalogue_preserves_operational_rules() -> None:
    catalogue = methodology_commitment_catalogue()

    assert catalogue["falsifiability_first"].source_record == "P0R00360"
    assert catalogue["falsifiability_first"].operational_role == "admission_gate"
    assert catalogue["hypothesis_registry"].source_record == "P0R00360"
    assert catalogue["tiered_status"].operational_role == "claim_status_tracking"
    assert catalogue["versioning_and_correction"].operational_role == "model_update"


def test_scientific_inference_classifier_rejects_unknown_step() -> None:
    assert classify_scientific_inference_step("theory") == "generative_model"
    assert classify_scientific_inference_step("experiment") == "sensory_evidence"
    assert classify_scientific_inference_step("falsification") == "prediction_error"
    assert classify_scientific_inference_step("revision") == "model_update"

    with pytest.raises(ValueError, match="unknown scientific inference step"):
        classify_scientific_inference_step("dogma")


def test_status_method_fixture_preserves_scope_and_quality_controls() -> None:
    result = validate_status_method_fixture()

    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_ledger_span == ("P0R00358", "P0R00390")
    assert result.blank_separator_count == 2
    assert result.interaction_formula == "H_int = -lambda * Psi_s * sigma"
    assert result.next_boundary == "P0R00391"
    assert result.null_controls["doctrine_promotion_rejection_label"] == 1.0
    assert result.null_controls["untestable_sigma_rejection_label"] == 1.0
    assert result.null_controls["analogy_without_empirical_handle_rejection_label"] == 1.0

    with pytest.raises(ValueError, match="expected_blank_separator_count must equal 2"):
        StatusMethodConfig(expected_blank_separator_count=1)
