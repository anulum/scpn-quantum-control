# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Complexity Economics & Social Physics validation tests
"""Tests for Paper 0  Complexity Economics & Social Physics source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.complexity_economics_social_physics_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ComplexityEconomicsSocialPhysicsConfig,
    classify_complexity_economics_social_physics_component,
    complexity_economics_social_physics_labels,
    validate_complexity_economics_social_physics_fixture,
)


def test_complexity_economics_social_physics_fixture_preserves_source_boundary() -> None:
    result = validate_complexity_economics_social_physics_fixture()
    assert result.source_ledger_span == ("P0R05886", "P0R05893")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05894"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_complexity_economics_social_physics_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05886"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05893"


def test_complexity_economics_social_physics_classification_and_labels_are_explicit() -> None:
    for component in ("complexity_economics_social_physics", "biosemiotics_meaning"):
        assert (
            classify_complexity_economics_social_physics_component(component)
            == f"{component}_source_boundary"
        )
    labels = complexity_economics_social_physics_labels()
    assert labels["section"] == " Complexity Economics & Social Physics"
    assert labels["next_boundary"] == "P0R05894"


def test_complexity_economics_social_physics_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        ComplexityEconomicsSocialPhysicsConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        ComplexityEconomicsSocialPhysicsConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05894"):
        ComplexityEconomicsSocialPhysicsConfig(next_source_boundary="P0R05893")
    with pytest.raises(ValueError, match="unknown complexity_economics_social_physics component"):
        classify_complexity_economics_social_physics_component("empirical_validation_claim")
