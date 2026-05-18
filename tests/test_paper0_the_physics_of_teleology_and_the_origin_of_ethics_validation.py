# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Physics of Teleology and the Origin of Ethics validation tests
"""Tests for Paper 0 The Physics of Teleology and the Origin of Ethics source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_physics_of_teleology_and_the_origin_of_ethics_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ThePhysicsOfTeleologyAndTheOriginOfEthicsConfig,
    classify_the_physics_of_teleology_and_the_origin_of_ethics_component,
    the_physics_of_teleology_and_the_origin_of_ethics_labels,
    validate_the_physics_of_teleology_and_the_origin_of_ethics_fixture,
)


def test_the_physics_of_teleology_and_the_origin_of_ethics_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_physics_of_teleology_and_the_origin_of_ethics_fixture()
    assert result.source_ledger_span == ("P0R06088", "P0R06098")
    assert result.source_record_count == 11
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R06099"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_physics_of_teleology_and_the_origin_of_ethics_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06088"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06098"


def test_the_physics_of_teleology_and_the_origin_of_ethics_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_physics_of_teleology_and_the_origin_of_ethics",
        "i_the_ontological_origin_of_ethics_gauge_theory_derivation",
        "ii_the_principle_of_ethical_least_action_pela",
    ):
        assert (
            classify_the_physics_of_teleology_and_the_origin_of_ethics_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_physics_of_teleology_and_the_origin_of_ethics_labels()
    assert labels["section"] == "The Physics of Teleology and the Origin of Ethics"
    assert labels["next_boundary"] == "P0R06099"


def test_the_physics_of_teleology_and_the_origin_of_ethics_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        ThePhysicsOfTeleologyAndTheOriginOfEthicsConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        ThePhysicsOfTeleologyAndTheOriginOfEthicsConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06099"):
        ThePhysicsOfTeleologyAndTheOriginOfEthicsConfig(next_source_boundary="P0R06098")
    with pytest.raises(
        ValueError, match="unknown the_physics_of_teleology_and_the_origin_of_ethics component"
    ):
        classify_the_physics_of_teleology_and_the_origin_of_ethics_component(
            "empirical_validation_claim"
        )
