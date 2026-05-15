# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Foreword coupling fixtures
"""Tests for Paper 0 Foreword coupling fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.foreword_coupling_validation import (
    ForewordCouplingConfig,
    classify_predictive_coding_channel,
    interaction_hamiltonian,
    sigma_layer_catalogue,
    validate_foreword_coupling_fixture,
)


def test_interaction_hamiltonian_preserves_source_sign_and_parameters() -> None:
    assert interaction_hamiltonian(lambda_coupling=2.0, psi_s=3.0, sigma=5.0) == -30.0
    assert interaction_hamiltonian(lambda_coupling=0.5, psi_s=2 + 1j, sigma=4.0) == -(4 + 2j)

    with pytest.raises(ValueError, match="lambda_coupling must be finite"):
        interaction_hamiltonian(lambda_coupling=float("nan"), psi_s=1.0, sigma=1.0)


def test_sigma_layer_catalogue_preserves_layer_examples() -> None:
    catalogue = sigma_layer_catalogue()

    assert catalogue["L1"].source_record == "P0R00293"
    assert "dipole" in catalogue["L1"].collective_state_variable
    assert catalogue["L2"].source_record == "P0R00294"
    assert "gamma" in catalogue["L2"].collective_state_variable
    assert catalogue["L6"].source_record == "P0R00295"
    assert "global temperature" in catalogue["L6"].collective_state_variable


def test_predictive_coding_channel_classifier_rejects_unknown_channel() -> None:
    assert classify_predictive_coding_channel("downward_projection") == "generative_model"
    assert classify_predictive_coding_channel("upward_feedback") == "prediction_error_flow"

    with pytest.raises(ValueError, match="unknown predictive-coding channel"):
        classify_predictive_coding_channel("sideways_projection")


def test_foreword_coupling_fixture_preserves_scope_and_counts() -> None:
    result = validate_foreword_coupling_fixture()

    assert result.hardware_status == "source_formula_no_experiment"
    assert result.source_ledger_span == ("P0R00268", "P0R00306")
    assert result.sigma_layer_example_count == 3
    assert result.image_marker_count == 1
    assert result.preface_i_boundary == "P0R00307"
    assert result.null_controls["unknown_channel_rejection_label"] == 1.0
    assert result.sample_hamiltonian_value == -30.0

    with pytest.raises(ValueError, match="expected_sigma_layer_example_count must equal 3"):
        ForewordCouplingConfig(expected_sigma_layer_example_count=2)
