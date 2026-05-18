# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Prediction III: Topos-Theoretic Cognitive Hesitation (The $\Omega$-State) validation tests
r"""Tests for Paper 0 Prediction III: Topos-Theoretic Cognitive Hesitation (The $\Omega$-State) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    PredictionIiiToposTheoreticCognitiveHesitationTheOmegaStateConfig,
    classify_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_component,
    prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_labels,
    validate_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_fixture,
)


def test_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_fixture()
    assert result.source_ledger_span == ("P0R05124", "P0R05142")
    assert result.source_record_count == 19
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R05143"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05124"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05142"


def test_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",):
        assert (
            classify_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_labels()
    assert (
        labels["section"]
        == "Prediction III: Topos-Theoretic Cognitive Hesitation (The $\\Omega$-State)"
    )
    assert labels["next_boundary"] == "P0R05143"


def test_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 19"):
        PredictionIiiToposTheoreticCognitiveHesitationTheOmegaStateConfig(
            expected_source_record_count=18
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        PredictionIiiToposTheoreticCognitiveHesitationTheOmegaStateConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05143"):
        PredictionIiiToposTheoreticCognitiveHesitationTheOmegaStateConfig(
            next_source_boundary="P0R05142"
        )
    with pytest.raises(
        ValueError,
        match="unknown prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state component",
    ):
        classify_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_component(
            "empirical_validation_claim"
        )
