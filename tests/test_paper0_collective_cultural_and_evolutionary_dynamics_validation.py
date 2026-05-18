# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Collective, Cultural, and Evolutionary Dynamics validation tests
"""Tests for Paper 0  Collective, Cultural, and Evolutionary Dynamics source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.collective_cultural_and_evolutionary_dynamics_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    CollectiveCulturalAndEvolutionaryDynamicsConfig,
    classify_collective_cultural_and_evolutionary_dynamics_component,
    collective_cultural_and_evolutionary_dynamics_labels,
    validate_collective_cultural_and_evolutionary_dynamics_fixture,
)


def test_collective_cultural_and_evolutionary_dynamics_fixture_preserves_source_boundary() -> None:
    result = validate_collective_cultural_and_evolutionary_dynamics_fixture()
    assert result.source_ledger_span == ("P0R05818", "P0R05825")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05826"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_collective_cultural_and_evolutionary_dynamics_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05818"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05825"


def test_collective_cultural_and_evolutionary_dynamics_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "collective_cultural_and_evolutionary_dynamics",
        "consciousness_anomalous_data",
    ):
        assert (
            classify_collective_cultural_and_evolutionary_dynamics_component(component)
            == f"{component}_source_boundary"
        )
    labels = collective_cultural_and_evolutionary_dynamics_labels()
    assert labels["section"] == " Collective, Cultural, and Evolutionary Dynamics"
    assert labels["next_boundary"] == "P0R05826"


def test_collective_cultural_and_evolutionary_dynamics_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        CollectiveCulturalAndEvolutionaryDynamicsConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        CollectiveCulturalAndEvolutionaryDynamicsConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05826"):
        CollectiveCulturalAndEvolutionaryDynamicsConfig(next_source_boundary="P0R05825")
    with pytest.raises(
        ValueError, match="unknown collective_cultural_and_evolutionary_dynamics component"
    ):
        classify_collective_cultural_and_evolutionary_dynamics_component(
            "empirical_validation_claim"
        )
