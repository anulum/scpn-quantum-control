# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Part III: System Architecture & Network Dynamics validation tests
"""Tests for Paper 0 Part III: System Architecture & Network Dynamics source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.part_iii_system_architecture_network_dynamics_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    PartIiiSystemArchitectureNetworkDynamicsConfig,
    classify_part_iii_system_architecture_network_dynamics_component,
    part_iii_system_architecture_network_dynamics_labels,
    validate_part_iii_system_architecture_network_dynamics_fixture,
)


def test_part_iii_system_architecture_network_dynamics_fixture_preserves_source_boundary() -> None:
    result = validate_part_iii_system_architecture_network_dynamics_fixture()
    assert result.source_ledger_span == ("P0R02011", "P0R02030")
    assert result.source_record_count == 20
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R02031"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_part_iii_system_architecture_network_dynamics_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02011"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02030"


def test_part_iii_system_architecture_network_dynamics_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "part_iii_system_architecture_network_dynamics",
        "3_1_the_master_diagram_visualising_the_15_layers_6_domains",
    ):
        assert (
            classify_part_iii_system_architecture_network_dynamics_component(component)
            == f"{component}_source_boundary"
        )
    labels = part_iii_system_architecture_network_dynamics_labels()
    assert labels["section"] == "Part III: System Architecture & Network Dynamics"
    assert labels["next_boundary"] == "P0R02031"


def test_part_iii_system_architecture_network_dynamics_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 20"):
        PartIiiSystemArchitectureNetworkDynamicsConfig(expected_source_record_count=19)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        PartIiiSystemArchitectureNetworkDynamicsConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02031"):
        PartIiiSystemArchitectureNetworkDynamicsConfig(next_source_boundary="P0R02030")
    with pytest.raises(
        ValueError, match="unknown part_iii_system_architecture_network_dynamics component"
    ):
        classify_part_iii_system_architecture_network_dynamics_component(
            "empirical_validation_claim"
        )
