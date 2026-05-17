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

from scpn_quantum_control.paper0.predictive_coding_integration_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    PredictiveCodingIntegrationConfig,
    classify_predictive_coding_integration_component,
    predictive_coding_integration_labels,
    validate_predictive_coding_integration_fixture,
)


def test_predictive_coding_integration_fixture_preserves_source_boundary() -> None:
    result = validate_predictive_coding_integration_fixture()
    assert result.source_ledger_span == ("P0R01763", "P0R01770")
    assert result.source_record_count == 8
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R01771"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_predictive_coding_integration_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R01763"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R01770"


def test_predictive_coding_integration_classification_and_labels_are_explicit() -> None:
    for component in (
        "predictive_coding_integration",
        "the_potential_v_psi_as_the_landscape_of_priors",
        "the_sextic_term_as_a_sanity_check",
        "psis_field_coupling_integration",
    ):
        assert (
            classify_predictive_coding_integration_component(component)
            == f"{component}_source_boundary"
        )
    labels = predictive_coding_integration_labels()
    assert labels["section"] == "Predictive Coding Integration"
    assert labels["next_boundary"] == "P0R01771"


def test_predictive_coding_integration_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        PredictiveCodingIntegrationConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        PredictiveCodingIntegrationConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01771"):
        PredictiveCodingIntegrationConfig(next_source_boundary="P0R01770")
    with pytest.raises(ValueError, match="unknown predictive_coding_integration component"):
        classify_predictive_coding_integration_component("empirical_validation_claim")
