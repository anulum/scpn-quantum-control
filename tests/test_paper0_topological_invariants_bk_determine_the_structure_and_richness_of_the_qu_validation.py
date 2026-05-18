# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Topological Invariants (bk): Determine the structure and richness of the qualia (Betti numbers). validation tests
"""Tests for Paper 0 Topological Invariants (bk): Determine the structure and richness of the qualia (Betti numbers). source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TopologicalInvariantsBkDetermineTheStructureAndRichnessOfTheQuConfig,
    classify_topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_component,
    topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_labels,
    validate_topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_fixture,
)


def test_topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_fixture()
    )
    assert result.source_ledger_span == ("P0R06023", "P0R06030")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R06031"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06023"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06030"


def test_topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "topological_invariants_bk_determine_the_structure_and_richness_of_the_qu",
        "v_the_scpn_evolutionary_synthesis",
        "1_the_adaptive_potential_landscape_apl",
    ):
        assert (
            classify_topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_labels()
    assert (
        labels["section"]
        == "Topological Invariants (bk): Determine the structure and richness of the qualia (Betti numbers)."
    )
    assert labels["next_boundary"] == "P0R06031"


def test_topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        TopologicalInvariantsBkDetermineTheStructureAndRichnessOfTheQuConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        TopologicalInvariantsBkDetermineTheStructureAndRichnessOfTheQuConfig(
            expected_component_count=4
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06031"):
        TopologicalInvariantsBkDetermineTheStructureAndRichnessOfTheQuConfig(
            next_source_boundary="P0R06030"
        )
    with pytest.raises(
        ValueError,
        match="unknown topological_invariants_bk_determine_the_structure_and_richness_of_the_qu component",
    ):
        classify_topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_component(
            "empirical_validation_claim"
        )
