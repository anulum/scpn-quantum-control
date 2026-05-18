# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Overarching Principles and System Dynamics validation tests
"""Tests for Paper 0 Overarching Principles and System Dynamics source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.overarching_principles_and_system_dynamics_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    OverarchingPrinciplesAndSystemDynamicsConfig,
    classify_overarching_principles_and_system_dynamics_component,
    overarching_principles_and_system_dynamics_labels,
    validate_overarching_principles_and_system_dynamics_fixture,
)


def test_overarching_principles_and_system_dynamics_fixture_preserves_source_boundary() -> None:
    result = validate_overarching_principles_and_system_dynamics_fixture()
    assert result.source_ledger_span == ("P0R05919", "P0R05927")
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05928"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_overarching_principles_and_system_dynamics_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05919"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05927"


def test_overarching_principles_and_system_dynamics_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "overarching_principles_and_system_dynamics",
        "computational_unifier",
        "layer_5_strange_loop_as_active_inference_engine_sn_precision_control_oct",
    ):
        assert (
            classify_overarching_principles_and_system_dynamics_component(component)
            == f"{component}_source_boundary"
        )
    labels = overarching_principles_and_system_dynamics_labels()
    assert labels["section"] == "Overarching Principles and System Dynamics"
    assert labels["next_boundary"] == "P0R05928"


def test_overarching_principles_and_system_dynamics_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        OverarchingPrinciplesAndSystemDynamicsConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        OverarchingPrinciplesAndSystemDynamicsConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05928"):
        OverarchingPrinciplesAndSystemDynamicsConfig(next_source_boundary="P0R05927")
    with pytest.raises(
        ValueError, match="unknown overarching_principles_and_system_dynamics component"
    ):
        classify_overarching_principles_and_system_dynamics_component("empirical_validation_claim")
