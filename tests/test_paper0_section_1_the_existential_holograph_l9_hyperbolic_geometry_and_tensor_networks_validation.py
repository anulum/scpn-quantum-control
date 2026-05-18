# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Existential Holograph (L9): Hyperbolic Geometry and Tensor Networks validation tests
"""Tests for Paper 0 1. The Existential Holograph (L9): Hyperbolic Geometry and Tensor Networks source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section1TheExistentialHolographL9HyperbolicGeometryAndTensorNetworksConfig,
    classify_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_component,
    section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_labels,
    validate_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_fixture,
)


def test_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_fixture()
    assert result.source_ledger_span == ("P0R04441", "P0R04453")
    assert result.source_record_count == 13
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04454"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04441"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04453"


def test_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks",
        "resolving_the_holographic_geometry_mera_as_an_information_geometric_bulk",
    ):
        assert (
            classify_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = (
        section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_labels()
    )
    assert (
        labels["section"]
        == "1. The Existential Holograph (L9): Hyperbolic Geometry and Tensor Networks"
    )
    assert labels["next_boundary"] == "P0R04454"


def test_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 13"):
        Section1TheExistentialHolographL9HyperbolicGeometryAndTensorNetworksConfig(
            expected_source_record_count=12
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section1TheExistentialHolographL9HyperbolicGeometryAndTensorNetworksConfig(
            expected_component_count=3
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04454"):
        Section1TheExistentialHolographL9HyperbolicGeometryAndTensorNetworksConfig(
            next_source_boundary="P0R04453"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks component",
    ):
        classify_section_1_the_existential_holograph_l9_hyperbolic_geometry_and_tensor_networks_component(
            "empirical_validation_claim"
        )
