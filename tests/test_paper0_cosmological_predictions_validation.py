# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 cosmological predictions fixture tests
"""Tests for Paper 0 preregisterable cosmological predictions fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.cosmological_predictions_validation import (
    CosmologicalPredictionsConfig,
    cross_prediction_tension,
    prediction_catalogue,
    priority_ranking,
    validate_cosmological_predictions_fixture,
)


def test_prediction_catalogue_preserves_five_preregisterable_targets() -> None:
    catalogue = prediction_catalogue()

    assert tuple(item.prediction_id for item in catalogue) == (
        "28.1",
        "28.2",
        "28.3",
        "28.4",
        "28.5",
    )
    assert all(item.test_protocol for item in catalogue)
    assert all(item.null_result for item in catalogue)
    assert priority_ranking() == ("28.1", "28.2", "28.5", "28.3", "28.4")


def test_cross_prediction_tension_flags_inconsistent_confirmations() -> None:
    assert cross_prediction_tension({"28.1": True, "28.5": False}) == (
        "cmb-confirmed-quantum-retention-null"
    )
    assert cross_prediction_tension({"28.3": True, "28.5": False}) == (
        "observer-entropy-confirmed-retention-null"
    )
    assert cross_prediction_tension({"28.2": True, "28.1": False}) == ("gw-confirmed-cmb-null")
    assert cross_prediction_tension({"28.1": False, "28.2": False, "28.3": False}) is None


def test_cosmological_predictions_fixture_preserves_claim_boundary() -> None:
    result = validate_cosmological_predictions_fixture()

    assert result.hardware_status == "preregistration_protocol_no_execution"
    assert result.source_ledger_span == ("P0R06949", "P0R07005")
    assert result.prediction_count == 5
    assert result.priority_ranking == ("28.1", "28.2", "28.5", "28.3", "28.4")
    assert result.cross_consistency_rules == 4
    assert result.null_controls["missing_null_result_rejection_label"] == 1.0
    assert result.null_controls["invalid_priority_ranking_rejection_label"] == 1.0
    assert result.null_controls["unsupported_confirmation_claim_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary

    with pytest.raises(ValueError, match="expected_prediction_count must be at least 1"):
        CosmologicalPredictionsConfig(expected_prediction_count=0)
