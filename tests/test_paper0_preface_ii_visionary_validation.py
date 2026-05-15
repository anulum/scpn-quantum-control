# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Preface II visionary fixtures
"""Tests for Paper 0 Preface II visionary-register fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.preface_ii_visionary_validation import (
    PrefaceIIVisionaryConfig,
    classify_visionary_active_inference_role,
    normalise_visionary_interaction_formula,
    validate_preface_ii_visionary_fixture,
    visionary_operator_catalogue,
)


def test_visionary_interaction_formula_normaliser_preserves_source_parameters() -> None:
    assert normalise_visionary_interaction_formula("H_int = -λ * Ψs * σ") == (
        "H_int = -lambda * Psi_s * sigma"
    )
    assert normalise_visionary_interaction_formula("H_int = -lambda * Psi_s * sigma") == (
        "H_int = -lambda * Psi_s * sigma"
    )

    with pytest.raises(ValueError, match="unsupported visionary interaction formula"):
        normalise_visionary_interaction_formula("H_int = -lambda * sigma")


def test_visionary_operator_catalogue_preserves_source_roles() -> None:
    catalogue = visionary_operator_catalogue()

    assert catalogue["projection_lattices"].source_records == ("P0R00345", "P0R00353")
    assert catalogue["projection_lattices"].active_inference_role == "prior_pathway"
    assert catalogue["resonance_hubs"].source_records == ("P0R00349", "P0R00353")
    assert catalogue["resonance_hubs"].sigma_role == "coherent sigma coupling site"
    assert catalogue["vibrational_codes"].source_records == ("P0R00346", "P0R00354", "P0R00356")
    assert catalogue["vibrational_codes"].sigma_role == "designed sigma organisation"


def test_visionary_active_inference_classifier_rejects_unknown_operator() -> None:
    assert classify_visionary_active_inference_role("projection_lattices") == "prior_pathway"
    assert classify_visionary_active_inference_role("vibrational_codes") == (
        "generative_model_intervention"
    )

    with pytest.raises(ValueError, match="unknown Preface II operator"):
        classify_visionary_active_inference_role("rhetorical_overclaim")


def test_preface_ii_visionary_fixture_preserves_scope_and_counts() -> None:
    result = validate_preface_ii_visionary_fixture()

    assert result.hardware_status == "source_visionary_register_no_experiment"
    assert result.source_ledger_span == ("P0R00333", "P0R00357")
    assert result.blank_separator_count == 1
    assert result.interaction_formula == "H_int = -lambda * Psi_s * sigma"
    assert result.status_method_boundary == "P0R00358"
    assert result.null_controls["manifesto_as_empirical_evidence_rejection_label"] == 1.0
    assert result.null_controls["sigma_design_without_testability_rejection_label"] == 1.0

    with pytest.raises(ValueError, match="expected_blank_separator_count must equal 1"):
        PrefaceIIVisionaryConfig(expected_blank_separator_count=2)
