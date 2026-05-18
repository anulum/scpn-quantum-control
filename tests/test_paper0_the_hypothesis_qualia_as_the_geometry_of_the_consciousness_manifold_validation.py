# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Hypothesis: Qualia as the Geometry of the Consciousness Manifold validation tests
"""Tests for Paper 0 The Hypothesis: Qualia as the Geometry of the Consciousness Manifold source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldConfig,
    classify_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_component,
    the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_labels,
    validate_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_fixture,
)


def test_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_fixture()
    assert result.source_ledger_span == ("P0R03453", "P0R03461")
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R03462"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03453"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03461"


def test_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold",
        "meta_framework_integrations",
        "predictive_coding_integration",
    ):
        assert (
            classify_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_labels()
    assert (
        labels["section"] == "The Hypothesis: Qualia as the Geometry of the Consciousness Manifold"
    )
    assert labels["next_boundary"] == "P0R03462"


def test_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldConfig(
            expected_source_record_count=8
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldConfig(
            expected_component_count=4
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03462"):
        TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldConfig(
            next_source_boundary="P0R03461"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold component",
    ):
        classify_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_component(
            "empirical_validation_claim"
        )
