# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Predictive Coding Integration validation tests
"""Tests for Paper 0 Predictive Coding Integration source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.predictive_coding_integration_p0r03059_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    PredictiveCodingIntegrationP0r03059Config,
    classify_predictive_coding_integration_p0r03059_component,
    predictive_coding_integration_p0r03059_labels,
    validate_predictive_coding_integration_p0r03059_fixture,
)


def test_predictive_coding_integration_p0r03059_fixture_preserves_source_boundary() -> None:
    result = validate_predictive_coding_integration_p0r03059_fixture()
    assert result.source_ledger_span == ("P0R03059", "P0R03066")
    assert result.source_record_count == 8
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R03067"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_predictive_coding_integration_p0r03059_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03059"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03066"


def test_predictive_coding_integration_p0r03059_classification_and_labels_are_explicit() -> None:
    for component in (
        "predictive_coding_integration",
        "the_psi_field_as_a_precision_enhancer",
        "psis_field_coupling_integration",
        "h_int_as_a_stabilising_potential",
    ):
        assert (
            classify_predictive_coding_integration_p0r03059_component(component)
            == f"{component}_source_boundary"
        )
    labels = predictive_coding_integration_p0r03059_labels()
    assert labels["section"] == "Predictive Coding Integration"
    assert labels["next_boundary"] == "P0R03067"


def test_predictive_coding_integration_p0r03059_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        PredictiveCodingIntegrationP0r03059Config(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        PredictiveCodingIntegrationP0r03059Config(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03067"):
        PredictiveCodingIntegrationP0r03059Config(next_source_boundary="P0R03066")
    with pytest.raises(
        ValueError, match="unknown predictive_coding_integration_p0r03059 component"
    ):
        classify_predictive_coding_integration_p0r03059_component("empirical_validation_claim")
