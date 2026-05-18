# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Predicted Signatures and Analysis Protocol validation tests
"""Tests for Paper 0 Predicted Signatures and Analysis Protocol source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.predicted_signatures_and_analysis_protocol_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    PredictedSignaturesAndAnalysisProtocolConfig,
    classify_predicted_signatures_and_analysis_protocol_component,
    predicted_signatures_and_analysis_protocol_labels,
    validate_predicted_signatures_and_analysis_protocol_fixture,
)


def test_predicted_signatures_and_analysis_protocol_fixture_preserves_source_boundary() -> None:
    result = validate_predicted_signatures_and_analysis_protocol_fixture()
    assert result.source_ledger_span == ("P0R05264", "P0R05272")
    assert result.source_record_count == 9
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R05273"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_predicted_signatures_and_analysis_protocol_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05264"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05272"


def test_predicted_signatures_and_analysis_protocol_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("predicted_signatures_and_analysis_protocol",):
        assert (
            classify_predicted_signatures_and_analysis_protocol_component(component)
            == f"{component}_source_boundary"
        )
    labels = predicted_signatures_and_analysis_protocol_labels()
    assert labels["section"] == "Predicted Signatures and Analysis Protocol"
    assert labels["next_boundary"] == "P0R05273"


def test_predicted_signatures_and_analysis_protocol_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        PredictedSignaturesAndAnalysisProtocolConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        PredictedSignaturesAndAnalysisProtocolConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05273"):
        PredictedSignaturesAndAnalysisProtocolConfig(next_source_boundary="P0R05272")
    with pytest.raises(
        ValueError, match="unknown predicted_signatures_and_analysis_protocol component"
    ):
        classify_predicted_signatures_and_analysis_protocol_component("empirical_validation_claim")
