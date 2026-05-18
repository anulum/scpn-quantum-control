# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Prediction II: Causal Entropic Force Signatures in Quantum Randomness validation tests
"""Tests for Paper 0 Prediction II: Causal Entropic Force Signatures in Quantum Randomness source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    PredictionIiCausalEntropicForceSignaturesInQuantumRandomnessConfig,
    classify_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_component,
    prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_labels,
    validate_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_fixture,
)


def test_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_fixture()
    )
    assert result.source_ledger_span == ("P0R05202", "P0R05216")
    assert result.source_record_count == 15
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05217"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05202"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05216"


def test_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "prediction_ii_causal_entropic_force_signatures_in_quantum_randomness",
        "theoretical_derivation",
        "predicted_signature",
    ):
        assert (
            classify_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_labels()
    assert (
        labels["section"]
        == "Prediction II: Causal Entropic Force Signatures in Quantum Randomness"
    )
    assert labels["next_boundary"] == "P0R05217"


def test_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 15"):
        PredictionIiCausalEntropicForceSignaturesInQuantumRandomnessConfig(
            expected_source_record_count=14
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        PredictionIiCausalEntropicForceSignaturesInQuantumRandomnessConfig(
            expected_component_count=4
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05217"):
        PredictionIiCausalEntropicForceSignaturesInQuantumRandomnessConfig(
            next_source_boundary="P0R05216"
        )
    with pytest.raises(
        ValueError,
        match="unknown prediction_ii_causal_entropic_force_signatures_in_quantum_randomness component",
    ):
        classify_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_component(
            "empirical_validation_claim"
        )
