# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Integration of CEF into the Path Integral Formalism: validation tests
"""Tests for Paper 0 Integration of CEF into the Path Integral Formalism: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.integration_of_cef_into_the_path_integral_formalism_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IntegrationOfCefIntoThePathIntegralFormalismConfig,
    classify_integration_of_cef_into_the_path_integral_formalism_component,
    integration_of_cef_into_the_path_integral_formalism_labels,
    validate_integration_of_cef_into_the_path_integral_formalism_fixture,
)


def test_integration_of_cef_into_the_path_integral_formalism_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_integration_of_cef_into_the_path_integral_formalism_fixture()
    assert result.source_ledger_span == ("P0R03704", "P0R03714")
    assert result.source_record_count == 11
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03715"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_integration_of_cef_into_the_path_integral_formalism_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03704"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03714"


def test_integration_of_cef_into_the_path_integral_formalism_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "integration_of_cef_into_the_path_integral_formalism",
        "how_consciousness_shapes_reality_focusing_the_quantum_world",
    ):
        assert (
            classify_integration_of_cef_into_the_path_integral_formalism_component(component)
            == f"{component}_source_boundary"
        )
    labels = integration_of_cef_into_the_path_integral_formalism_labels()
    assert labels["section"] == "Integration of CEF into the Path Integral Formalism:"
    assert labels["next_boundary"] == "P0R03715"


def test_integration_of_cef_into_the_path_integral_formalism_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        IntegrationOfCefIntoThePathIntegralFormalismConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        IntegrationOfCefIntoThePathIntegralFormalismConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03715"):
        IntegrationOfCefIntoThePathIntegralFormalismConfig(next_source_boundary="P0R03714")
    with pytest.raises(
        ValueError, match="unknown integration_of_cef_into_the_path_integral_formalism component"
    ):
        classify_integration_of_cef_into_the_path_integral_formalism_component(
            "empirical_validation_claim"
        )
