# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Binding Problem Solution: validation tests
"""Tests for Paper 0 The Binding Problem Solution: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_binding_problem_solution_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheBindingProblemSolutionConfig,
    classify_the_binding_problem_solution_component,
    the_binding_problem_solution_labels,
    validate_the_binding_problem_solution_fixture,
)


def test_the_binding_problem_solution_fixture_preserves_source_boundary() -> None:
    result = validate_the_binding_problem_solution_fixture()
    assert result.source_ledger_span == ("P0R03440", "P0R03452")
    assert result.source_record_count == 13
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03453"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_binding_problem_solution_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03440"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03452"


def test_the_binding_problem_solution_classification_and_labels_are_explicit() -> None:
    for component in ("the_binding_problem_solution", "mary_s_room_resolution"):
        assert (
            classify_the_binding_problem_solution_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_binding_problem_solution_labels()
    assert labels["section"] == "The Binding Problem Solution:"
    assert labels["next_boundary"] == "P0R03453"


def test_the_binding_problem_solution_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 13"):
        TheBindingProblemSolutionConfig(expected_source_record_count=12)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TheBindingProblemSolutionConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03453"):
        TheBindingProblemSolutionConfig(next_source_boundary="P0R03452")
    with pytest.raises(ValueError, match="unknown the_binding_problem_solution component"):
        classify_the_binding_problem_solution_component("empirical_validation_claim")
