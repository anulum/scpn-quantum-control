# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 VIII. The Evolutionary Trajectory of the Brain-Body System validation tests
"""Tests for Paper 0 VIII. The Evolutionary Trajectory of the Brain-Body System source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.viii_the_evolutionary_trajectory_of_the_brain_body_system_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemConfig,
    classify_viii_the_evolutionary_trajectory_of_the_brain_body_system_component,
    validate_viii_the_evolutionary_trajectory_of_the_brain_body_system_fixture,
    viii_the_evolutionary_trajectory_of_the_brain_body_system_labels,
)


def test_viii_the_evolutionary_trajectory_of_the_brain_body_system_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_viii_the_evolutionary_trajectory_of_the_brain_body_system_fixture()
    assert result.source_ledger_span == ("P0R05039", "P0R05049")
    assert result.source_record_count == 11
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05050"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_viii_the_evolutionary_trajectory_of_the_brain_body_system_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05039"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05049"


def test_viii_the_evolutionary_trajectory_of_the_brain_body_system_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "viii_the_evolutionary_trajectory_of_the_brain_body_system",
        "ix_advanced_therapeutic_interventions_tuning_the_embodied_scpn",
    ):
        assert (
            classify_viii_the_evolutionary_trajectory_of_the_brain_body_system_component(component)
            == f"{component}_source_boundary"
        )
    labels = viii_the_evolutionary_trajectory_of_the_brain_body_system_labels()
    assert labels["section"] == "VIII. The Evolutionary Trajectory of the Brain-Body System"
    assert labels["next_boundary"] == "P0R05050"


def test_viii_the_evolutionary_trajectory_of_the_brain_body_system_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05050"):
        ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemConfig(next_source_boundary="P0R05049")
    with pytest.raises(
        ValueError,
        match="unknown viii_the_evolutionary_trajectory_of_the_brain_body_system component",
    ):
        classify_viii_the_evolutionary_trajectory_of_the_brain_body_system_component(
            "empirical_validation_claim"
        )
