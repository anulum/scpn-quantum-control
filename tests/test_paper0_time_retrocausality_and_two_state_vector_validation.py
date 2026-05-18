# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Time, Retrocausality, and Two-State Vector validation tests
"""Tests for Paper 0  Time, Retrocausality, and Two-State Vector source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.time_retrocausality_and_two_state_vector_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TimeRetrocausalityAndTwoStateVectorConfig,
    classify_time_retrocausality_and_two_state_vector_component,
    time_retrocausality_and_two_state_vector_labels,
    validate_time_retrocausality_and_two_state_vector_fixture,
)


def test_time_retrocausality_and_two_state_vector_fixture_preserves_source_boundary() -> None:
    result = validate_time_retrocausality_and_two_state_vector_fixture()
    assert result.source_ledger_span == ("P0R05902", "P0R05909")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05910"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_time_retrocausality_and_two_state_vector_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05902"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05909"


def test_time_retrocausality_and_two_state_vector_classification_and_labels_are_explicit() -> None:
    for component in ("time_retrocausality_and_two_state_vector", "memory_holography"):
        assert (
            classify_time_retrocausality_and_two_state_vector_component(component)
            == f"{component}_source_boundary"
        )
    labels = time_retrocausality_and_two_state_vector_labels()
    assert labels["section"] == " Time, Retrocausality, and Two-State Vector"
    assert labels["next_boundary"] == "P0R05910"


def test_time_retrocausality_and_two_state_vector_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        TimeRetrocausalityAndTwoStateVectorConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TimeRetrocausalityAndTwoStateVectorConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05910"):
        TimeRetrocausalityAndTwoStateVectorConfig(next_source_boundary="P0R05909")
    with pytest.raises(
        ValueError, match="unknown time_retrocausality_and_two_state_vector component"
    ):
        classify_time_retrocausality_and_two_state_vector_component("empirical_validation_claim")
