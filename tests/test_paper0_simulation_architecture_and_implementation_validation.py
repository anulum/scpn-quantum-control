# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Simulation Architecture and Implementation validation tests
"""Tests for Paper 0 Simulation Architecture and Implementation source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.simulation_architecture_and_implementation_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    SimulationArchitectureAndImplementationConfig,
    classify_simulation_architecture_and_implementation_component,
    simulation_architecture_and_implementation_labels,
    validate_simulation_architecture_and_implementation_fixture,
)


def test_simulation_architecture_and_implementation_fixture_preserves_source_boundary() -> None:
    result = validate_simulation_architecture_and_implementation_fixture()
    assert result.source_ledger_span == ("P0R05245", "P0R05255")
    assert result.source_record_count == 11
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R05256"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_simulation_architecture_and_implementation_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05245"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05255"


def test_simulation_architecture_and_implementation_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("simulation_architecture_and_implementation",):
        assert (
            classify_simulation_architecture_and_implementation_component(component)
            == f"{component}_source_boundary"
        )
    labels = simulation_architecture_and_implementation_labels()
    assert labels["section"] == "Simulation Architecture and Implementation"
    assert labels["next_boundary"] == "P0R05256"


def test_simulation_architecture_and_implementation_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        SimulationArchitectureAndImplementationConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        SimulationArchitectureAndImplementationConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05256"):
        SimulationArchitectureAndImplementationConfig(next_source_boundary="P0R05255")
    with pytest.raises(
        ValueError, match="unknown simulation_architecture_and_implementation component"
    ):
        classify_simulation_architecture_and_implementation_component("empirical_validation_claim")
