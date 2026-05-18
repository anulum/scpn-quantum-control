# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Ultimate Feedback Loop: validation tests
"""Tests for Paper 0 The Ultimate Feedback Loop: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_ultimate_feedback_loop_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheUltimateFeedbackLoopConfig,
    classify_the_ultimate_feedback_loop_component,
    the_ultimate_feedback_loop_labels,
    validate_the_ultimate_feedback_loop_fixture,
)


def test_the_ultimate_feedback_loop_fixture_preserves_source_boundary() -> None:
    result = validate_the_ultimate_feedback_loop_fixture()
    assert result.source_ledger_span == ("P0R03067", "P0R03075")
    assert result.source_record_count == 9
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03076"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_ultimate_feedback_loop_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03067"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03075"


def test_the_ultimate_feedback_loop_classification_and_labels_are_explicit() -> None:
    for component in (
        "the_ultimate_feedback_loop",
        "the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel",
    ):
        assert (
            classify_the_ultimate_feedback_loop_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_ultimate_feedback_loop_labels()
    assert labels["section"] == "The Ultimate Feedback Loop:"
    assert labels["next_boundary"] == "P0R03076"


def test_the_ultimate_feedback_loop_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        TheUltimateFeedbackLoopConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TheUltimateFeedbackLoopConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03076"):
        TheUltimateFeedbackLoopConfig(next_source_boundary="P0R03075")
    with pytest.raises(ValueError, match="unknown the_ultimate_feedback_loop component"):
        classify_the_ultimate_feedback_loop_component("empirical_validation_claim")
