# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Topology & Geometry in Consciousness Models validation tests
"""Tests for Paper 0  Topology & Geometry in Consciousness Models source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.topology_geometry_in_consciousness_models_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TopologyGeometryInConsciousnessModelsConfig,
    classify_topology_geometry_in_consciousness_models_component,
    topology_geometry_in_consciousness_models_labels,
    validate_topology_geometry_in_consciousness_models_fixture,
)


def test_topology_geometry_in_consciousness_models_fixture_preserves_source_boundary() -> None:
    result = validate_topology_geometry_in_consciousness_models_fixture()
    assert result.source_ledger_span == ("P0R05876", "P0R05885")
    assert result.source_record_count == 10
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05886"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_topology_geometry_in_consciousness_models_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05876"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05885"


def test_topology_geometry_in_consciousness_models_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "topology_geometry_in_consciousness_models",
        "source_component",
        "cybernetics_control",
    ):
        assert (
            classify_topology_geometry_in_consciousness_models_component(component)
            == f"{component}_source_boundary"
        )
    labels = topology_geometry_in_consciousness_models_labels()
    assert labels["section"] == " Topology & Geometry in Consciousness Models"
    assert labels["next_boundary"] == "P0R05886"


def test_topology_geometry_in_consciousness_models_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        TopologyGeometryInConsciousnessModelsConfig(expected_source_record_count=9)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        TopologyGeometryInConsciousnessModelsConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05886"):
        TopologyGeometryInConsciousnessModelsConfig(next_source_boundary="P0R05885")
    with pytest.raises(
        ValueError, match="unknown topology_geometry_in_consciousness_models component"
    ):
        classify_topology_geometry_in_consciousness_models_component("empirical_validation_claim")
