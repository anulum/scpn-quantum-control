# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 A Map of Reality validation tests
"""Tests for Paper 0 A Map of Reality source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.a_map_of_reality_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    AMapOfRealityConfig,
    a_map_of_reality_labels,
    classify_a_map_of_reality_component,
    validate_a_map_of_reality_fixture,
)


def test_a_map_of_reality_fixture_preserves_source_boundary() -> None:
    result = validate_a_map_of_reality_fixture()
    assert result.source_ledger_span == ("P0R02031", "P0R02041")
    assert result.source_record_count == 11
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R02042"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"] == "source_a_map_of_reality_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02031"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02041"


def test_a_map_of_reality_classification_and_labels_are_explicit() -> None:
    for component in (
        "a_map_of_reality",
        "from_field_to_function_the_need_for_an_architecture",
        "the_sentient_consciousness_projection_network_scpn",
    ):
        assert classify_a_map_of_reality_component(component) == f"{component}_source_boundary"
    labels = a_map_of_reality_labels()
    assert labels["section"] == "A Map of Reality"
    assert labels["next_boundary"] == "P0R02042"


def test_a_map_of_reality_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        AMapOfRealityConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        AMapOfRealityConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02042"):
        AMapOfRealityConfig(next_source_boundary="P0R02041")
    with pytest.raises(ValueError, match="unknown a_map_of_reality component"):
        classify_a_map_of_reality_component("empirical_validation_claim")
