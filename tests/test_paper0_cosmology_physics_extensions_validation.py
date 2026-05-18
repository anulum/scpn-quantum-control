# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Cosmology & Physics Extensions validation tests
"""Tests for Paper 0  Cosmology & Physics Extensions source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.cosmology_physics_extensions_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    CosmologyPhysicsExtensionsConfig,
    classify_cosmology_physics_extensions_component,
    cosmology_physics_extensions_labels,
    validate_cosmology_physics_extensions_fixture,
)


def test_cosmology_physics_extensions_fixture_preserves_source_boundary() -> None:
    result = validate_cosmology_physics_extensions_fixture()
    assert result.source_ledger_span == ("P0R05665", "P0R05672")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05673"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_cosmology_physics_extensions_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05665"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05672"


def test_cosmology_physics_extensions_classification_and_labels_are_explicit() -> None:
    for component in ("cosmology_physics_extensions", "cosmological_quantum_foundations"):
        assert (
            classify_cosmology_physics_extensions_component(component)
            == f"{component}_source_boundary"
        )
    labels = cosmology_physics_extensions_labels()
    assert labels["section"] == " Cosmology & Physics Extensions"
    assert labels["next_boundary"] == "P0R05673"


def test_cosmology_physics_extensions_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        CosmologyPhysicsExtensionsConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        CosmologyPhysicsExtensionsConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05673"):
        CosmologyPhysicsExtensionsConfig(next_source_boundary="P0R05672")
    with pytest.raises(ValueError, match="unknown cosmology_physics_extensions component"):
        classify_cosmology_physics_extensions_component("empirical_validation_claim")
