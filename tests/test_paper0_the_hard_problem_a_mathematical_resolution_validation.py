# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Hard Problem: A Mathematical Resolution validation tests
"""Tests for Paper 0 The Hard Problem: A Mathematical Resolution source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_hard_problem_a_mathematical_resolution_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheHardProblemAMathematicalResolutionConfig,
    classify_the_hard_problem_a_mathematical_resolution_component,
    the_hard_problem_a_mathematical_resolution_labels,
    validate_the_hard_problem_a_mathematical_resolution_fixture,
)


def test_the_hard_problem_a_mathematical_resolution_fixture_preserves_source_boundary() -> None:
    result = validate_the_hard_problem_a_mathematical_resolution_fixture()
    assert result.source_ledger_span == ("P0R03418", "P0R03426")
    assert result.source_record_count == 9
    assert result.component_count == 5
    assert result.next_source_boundary == "P0R03427"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_hard_problem_a_mathematical_resolution_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03418"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03426"


def test_the_hard_problem_a_mathematical_resolution_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_hard_problem_a_mathematical_resolution",
        "the_hard_problem_reformulated",
        "traditional",
        "scpn",
        "key_insight",
    ):
        assert (
            classify_the_hard_problem_a_mathematical_resolution_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_hard_problem_a_mathematical_resolution_labels()
    assert labels["section"] == "The Hard Problem: A Mathematical Resolution"
    assert labels["next_boundary"] == "P0R03427"


def test_the_hard_problem_a_mathematical_resolution_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        TheHardProblemAMathematicalResolutionConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 5"):
        TheHardProblemAMathematicalResolutionConfig(expected_component_count=6)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03427"):
        TheHardProblemAMathematicalResolutionConfig(next_source_boundary="P0R03426")
    with pytest.raises(
        ValueError, match="unknown the_hard_problem_a_mathematical_resolution component"
    ):
        classify_the_hard_problem_a_mathematical_resolution_component("empirical_validation_claim")
