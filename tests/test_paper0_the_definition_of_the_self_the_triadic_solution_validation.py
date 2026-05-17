# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Definition of the Self: The Triadic Solution validation tests
"""Tests for Paper 0 The Definition of the Self: The Triadic Solution source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_definition_of_the_self_the_triadic_solution_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheDefinitionOfTheSelfTheTriadicSolutionConfig,
    classify_the_definition_of_the_self_the_triadic_solution_component,
    the_definition_of_the_self_the_triadic_solution_labels,
    validate_the_definition_of_the_self_the_triadic_solution_fixture,
)


def test_the_definition_of_the_self_the_triadic_solution_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_definition_of_the_self_the_triadic_solution_fixture()
    assert result.source_ledger_span == ("P0R01831", "P0R01894")
    assert result.source_record_count == 64
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R01895"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_definition_of_the_self_the_triadic_solution_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R01831"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R01894"


def test_the_definition_of_the_self_the_triadic_solution_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_definition_of_the_self_the_triadic_solution",):
        assert (
            classify_the_definition_of_the_self_the_triadic_solution_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_definition_of_the_self_the_triadic_solution_labels()
    assert labels["section"] == "The Definition of the Self: The Triadic Solution"
    assert labels["next_boundary"] == "P0R01895"


def test_the_definition_of_the_self_the_triadic_solution_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 64"):
        TheDefinitionOfTheSelfTheTriadicSolutionConfig(expected_source_record_count=63)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheDefinitionOfTheSelfTheTriadicSolutionConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01895"):
        TheDefinitionOfTheSelfTheTriadicSolutionConfig(next_source_boundary="P0R01894")
    with pytest.raises(
        ValueError, match="unknown the_definition_of_the_self_the_triadic_solution component"
    ):
        classify_the_definition_of_the_self_the_triadic_solution_component(
            "empirical_validation_claim"
        )
