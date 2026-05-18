# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Information-Geometric Coarse-Graining Lemma: Constructing the Macrostates Manifold validation tests
"""Tests for Paper 0 The Information-Geometric Coarse-Graining Lemma: Constructing the Macrostates Manifold source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_information_geometric_coarse_graining_lemma_constructing_the_macrost_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheInformationGeometricCoarseGrainingLemmaConstructingTheMacrostConfig,
    classify_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_component,
    the_information_geometric_coarse_graining_lemma_constructing_the_macrost_labels,
    validate_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_fixture,
)


def test_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_fixture()
    )
    assert result.source_ledger_span == ("P0R04151", "P0R04167")
    assert result.source_record_count == 17
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R04168"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04151"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04167"


def test_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_information_geometric_coarse_graining_lemma_constructing_the_macrost",):
        assert (
            classify_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_information_geometric_coarse_graining_lemma_constructing_the_macrost_labels()
    assert (
        labels["section"]
        == "The Information-Geometric Coarse-Graining Lemma: Constructing the Macrostates Manifold"
    )
    assert labels["next_boundary"] == "P0R04168"


def test_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 17"):
        TheInformationGeometricCoarseGrainingLemmaConstructingTheMacrostConfig(
            expected_source_record_count=16
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheInformationGeometricCoarseGrainingLemmaConstructingTheMacrostConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04168"):
        TheInformationGeometricCoarseGrainingLemmaConstructingTheMacrostConfig(
            next_source_boundary="P0R04167"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_information_geometric_coarse_graining_lemma_constructing_the_macrost component",
    ):
        classify_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_component(
            "empirical_validation_claim"
        )
