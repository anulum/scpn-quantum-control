# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Two-Timescale Controller: Stability and Exploration validation tests
"""Tests for Paper 0 The Two-Timescale Controller: Stability and Exploration source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_two_timescale_controller_stability_and_exploration_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheTwoTimescaleControllerStabilityAndExplorationConfig,
    classify_the_two_timescale_controller_stability_and_exploration_component,
    the_two_timescale_controller_stability_and_exploration_labels,
    validate_the_two_timescale_controller_stability_and_exploration_fixture,
)


def test_the_two_timescale_controller_stability_and_exploration_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_two_timescale_controller_stability_and_exploration_fixture()
    assert result.source_ledger_span == ("P0R02915", "P0R02922")
    assert result.source_record_count == 8
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02923"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_two_timescale_controller_stability_and_exploration_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02915"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02922"


def test_the_two_timescale_controller_stability_and_exploration_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_two_timescale_controller_stability_and_exploration",):
        assert (
            classify_the_two_timescale_controller_stability_and_exploration_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_two_timescale_controller_stability_and_exploration_labels()
    assert labels["section"] == "The Two-Timescale Controller: Stability and Exploration"
    assert labels["next_boundary"] == "P0R02923"


def test_the_two_timescale_controller_stability_and_exploration_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        TheTwoTimescaleControllerStabilityAndExplorationConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheTwoTimescaleControllerStabilityAndExplorationConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02923"):
        TheTwoTimescaleControllerStabilityAndExplorationConfig(next_source_boundary="P0R02922")
    with pytest.raises(
        ValueError,
        match="unknown the_two_timescale_controller_stability_and_exploration component",
    ):
        classify_the_two_timescale_controller_stability_and_exploration_component(
            "empirical_validation_claim"
        )
