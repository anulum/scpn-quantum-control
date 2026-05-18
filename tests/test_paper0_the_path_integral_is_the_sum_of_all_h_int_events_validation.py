# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Path Integral is the Sum of all H_int events: validation tests
"""Tests for Paper 0 The Path Integral is the Sum of all H_int events: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_path_integral_is_the_sum_of_all_h_int_events_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ThePathIntegralIsTheSumOfAllHIntEventsConfig,
    classify_the_path_integral_is_the_sum_of_all_h_int_events_component,
    the_path_integral_is_the_sum_of_all_h_int_events_labels,
    validate_the_path_integral_is_the_sum_of_all_h_int_events_fixture,
)


def test_the_path_integral_is_the_sum_of_all_h_int_events_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_path_integral_is_the_sum_of_all_h_int_events_fixture()
    assert result.source_ledger_span == ("P0R03673", "P0R03703")
    assert result.source_record_count == 31
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R03704"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_path_integral_is_the_sum_of_all_h_int_events_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03673"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03703"


def test_the_path_integral_is_the_sum_of_all_h_int_events_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_path_integral_is_the_sum_of_all_h_int_events",
        "cef_biases_the_coupling",
        "ethics_as_causal_entropic_forces_cef",
    ):
        assert (
            classify_the_path_integral_is_the_sum_of_all_h_int_events_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_path_integral_is_the_sum_of_all_h_int_events_labels()
    assert labels["section"] == "The Path Integral is the Sum of all H_int events:"
    assert labels["next_boundary"] == "P0R03704"


def test_the_path_integral_is_the_sum_of_all_h_int_events_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 31"):
        ThePathIntegralIsTheSumOfAllHIntEventsConfig(expected_source_record_count=30)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        ThePathIntegralIsTheSumOfAllHIntEventsConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03704"):
        ThePathIntegralIsTheSumOfAllHIntEventsConfig(next_source_boundary="P0R03703")
    with pytest.raises(
        ValueError, match="unknown the_path_integral_is_the_sum_of_all_h_int_events component"
    ):
        classify_the_path_integral_is_the_sum_of_all_h_int_events_component(
            "empirical_validation_claim"
        )
