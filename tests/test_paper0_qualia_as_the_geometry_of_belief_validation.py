# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Qualia as the Geometry of Belief: validation tests
"""Tests for Paper 0 Qualia as the Geometry of Belief: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.qualia_as_the_geometry_of_belief_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    QualiaAsTheGeometryOfBeliefConfig,
    classify_qualia_as_the_geometry_of_belief_component,
    qualia_as_the_geometry_of_belief_labels,
    validate_qualia_as_the_geometry_of_belief_fixture,
)


def test_qualia_as_the_geometry_of_belief_fixture_preserves_source_boundary() -> None:
    result = validate_qualia_as_the_geometry_of_belief_fixture()
    assert result.source_ledger_span == ("P0R03462", "P0R03469")
    assert result.source_record_count == 8
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R03470"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_qualia_as_the_geometry_of_belief_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03462"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03469"


def test_qualia_as_the_geometry_of_belief_classification_and_labels_are_explicit() -> None:
    for component in (
        "qualia_as_the_geometry_of_belief",
        "psis_field_coupling_integration",
        "sigma_is_the_geometry_of_the_manifold",
        "the_coupling_is_the_experience_of_geometry",
    ):
        assert (
            classify_qualia_as_the_geometry_of_belief_component(component)
            == f"{component}_source_boundary"
        )
    labels = qualia_as_the_geometry_of_belief_labels()
    assert labels["section"] == "Qualia as the Geometry of Belief:"
    assert labels["next_boundary"] == "P0R03470"


def test_qualia_as_the_geometry_of_belief_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        QualiaAsTheGeometryOfBeliefConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        QualiaAsTheGeometryOfBeliefConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03470"):
        QualiaAsTheGeometryOfBeliefConfig(next_source_boundary="P0R03469")
    with pytest.raises(ValueError, match="unknown qualia_as_the_geometry_of_belief component"):
        classify_qualia_as_the_geometry_of_belief_component("empirical_validation_claim")
