# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Preface I rigour fixtures
"""Tests for Paper 0 Preface I methodological-rigour fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.preface_i_rigour_validation import (
    PrefaceIRigourConfig,
    classify_hpc_application,
    discipline_role_catalogue,
    normalise_interaction_formula,
    validate_preface_i_rigour_fixture,
)


def test_interaction_formula_normaliser_preserves_source_parameters() -> None:
    assert normalise_interaction_formula("H_int = -λ * Ψs * σ") == (
        "H_int = -lambda * Psi_s * sigma"
    )
    assert normalise_interaction_formula("H_int = -lambda * Psi_s * sigma") == (
        "H_int = -lambda * Psi_s * sigma"
    )

    with pytest.raises(ValueError, match="unsupported interaction formula"):
        normalise_interaction_formula("H_int = -lambda * Psi_s")


def test_discipline_role_catalogue_preserves_preface_i_distinctions() -> None:
    catalogue = discipline_role_catalogue()

    assert catalogue["field_architecture"].source_records == ("P0R00319", "P0R00323", "P0R00327")
    assert catalogue["field_architecture"].hpc_role == "generative_model_structure"
    assert "identifying and characterising sigma" in catalogue["field_architecture"].sigma_role
    assert catalogue["consciousness_engineering"].source_records == (
        "P0R00320",
        "P0R00324",
        "P0R00328",
    )
    assert catalogue["consciousness_engineering"].hpc_role == "prediction_error_modulation"
    assert "designing and controlling sigma" in catalogue["consciousness_engineering"].sigma_role


def test_hpc_application_classifier_rejects_unknown_discipline() -> None:
    assert classify_hpc_application("field_architecture") == "generative_model_structure"
    assert classify_hpc_application("consciousness_engineering") == "prediction_error_modulation"

    with pytest.raises(ValueError, match="unknown Preface I discipline"):
        classify_hpc_application("metaphysical_commentary")


def test_preface_i_rigour_fixture_preserves_scope_and_counts() -> None:
    result = validate_preface_i_rigour_fixture()

    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_ledger_span == ("P0R00307", "P0R00332")
    assert result.blank_separator_count == 2
    assert result.interaction_formula == "H_int = -lambda * Psi_s * sigma"
    assert result.preface_ii_boundary == "P0R00333"
    assert result.null_controls["metaphysics_without_formalism_rejection_label"] == 1.0
    assert result.null_controls["empirical_validation_overclaim_rejection_label"] == 1.0

    with pytest.raises(ValueError, match="expected_blank_separator_count must equal 2"):
        PrefaceIRigourConfig(expected_blank_separator_count=1)
