# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Maximizing  as the Goal of Coupling: validation tests
"""Tests for Paper 0 Maximizing  as the Goal of Coupling: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.maximizing_as_the_goal_of_coupling_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MaximizingAsTheGoalOfCouplingConfig,
    classify_maximizing_as_the_goal_of_coupling_component,
    maximizing_as_the_goal_of_coupling_labels,
    validate_maximizing_as_the_goal_of_coupling_fixture,
)


def test_maximizing_as_the_goal_of_coupling_fixture_preserves_source_boundary() -> None:
    result = validate_maximizing_as_the_goal_of_coupling_fixture()
    assert result.source_ledger_span == ("P0R03539", "P0R03554")
    assert result.source_record_count == 16
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R03555"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_maximizing_as_the_goal_of_coupling_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03539"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03554"


def test_maximizing_as_the_goal_of_coupling_classification_and_labels_are_explicit() -> None:
    for component in (
        "maximizing_as_the_goal_of_coupling",
        "integration_with_integrated_information_theory_4_0",
        "iit_4_0_integration",
        "bridging_scpn_with_iit_4_0_s_mathematical_framework",
    ):
        assert (
            classify_maximizing_as_the_goal_of_coupling_component(component)
            == f"{component}_source_boundary"
        )
    labels = maximizing_as_the_goal_of_coupling_labels()
    assert labels["section"] == "Maximizing  as the Goal of Coupling:"
    assert labels["next_boundary"] == "P0R03555"


def test_maximizing_as_the_goal_of_coupling_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 16"):
        MaximizingAsTheGoalOfCouplingConfig(expected_source_record_count=15)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        MaximizingAsTheGoalOfCouplingConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03555"):
        MaximizingAsTheGoalOfCouplingConfig(next_source_boundary="P0R03554")
    with pytest.raises(ValueError, match="unknown maximizing_as_the_goal_of_coupling component"):
        classify_maximizing_as_the_goal_of_coupling_component("empirical_validation_claim")
