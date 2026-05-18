# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Binding Integral is H_int: validation tests
"""Tests for Paper 0 The Binding Integral is H_int: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_binding_integral_is_h_int_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheBindingIntegralIsHIntConfig,
    classify_the_binding_integral_is_h_int_component,
    the_binding_integral_is_h_int_labels,
    validate_the_binding_integral_is_h_int_fixture,
)


def test_the_binding_integral_is_h_int_fixture_preserves_source_boundary() -> None:
    result = validate_the_binding_integral_is_h_int_fixture()
    assert result.source_ledger_span == ("P0R03410", "P0R03417")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03418"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_binding_integral_is_h_int_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03410"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03417"


def test_the_binding_integral_is_h_int_classification_and_labels_are_explicit() -> None:
    for component in ("the_binding_integral_is_h_int", "the_coupling_creates_experience"):
        assert (
            classify_the_binding_integral_is_h_int_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_binding_integral_is_h_int_labels()
    assert labels["section"] == "The Binding Integral is H_int:"
    assert labels["next_boundary"] == "P0R03418"


def test_the_binding_integral_is_h_int_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        TheBindingIntegralIsHIntConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TheBindingIntegralIsHIntConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03418"):
        TheBindingIntegralIsHIntConfig(next_source_boundary="P0R03417")
    with pytest.raises(ValueError, match="unknown the_binding_integral_is_h_int component"):
        classify_the_binding_integral_is_h_int_component("empirical_validation_claim")
