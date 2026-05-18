# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. The Strange Loop (L5): The Geometry of Self-Reference validation tests
"""Tests for Paper 0 2. The Strange Loop (L5): The Geometry of Self-Reference source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_the_strange_loop_l5_the_geometry_of_self_reference_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2TheStrangeLoopL5TheGeometryOfSelfReferenceConfig,
    classify_section_2_the_strange_loop_l5_the_geometry_of_self_reference_component,
    section_2_the_strange_loop_l5_the_geometry_of_self_reference_labels,
    validate_section_2_the_strange_loop_l5_the_geometry_of_self_reference_fixture,
)


def test_section_2_the_strange_loop_l5_the_geometry_of_self_reference_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_the_strange_loop_l5_the_geometry_of_self_reference_fixture()
    assert result.source_ledger_span == ("P0R04433", "P0R04440")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04441"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_the_strange_loop_l5_the_geometry_of_self_reference_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04433"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04440"


def test_section_2_the_strange_loop_l5_the_geometry_of_self_reference_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "2_the_strange_loop_l5_the_geometry_of_self_reference",
        "3_symbols_as_geometric_operators_l7",
        "vi_the_geometry_of_memory_and_spacetime_domain_iii_l9_l10",
    ):
        assert (
            classify_section_2_the_strange_loop_l5_the_geometry_of_self_reference_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_2_the_strange_loop_l5_the_geometry_of_self_reference_labels()
    assert labels["section"] == "2. The Strange Loop (L5): The Geometry of Self-Reference"
    assert labels["next_boundary"] == "P0R04441"


def test_section_2_the_strange_loop_l5_the_geometry_of_self_reference_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section2TheStrangeLoopL5TheGeometryOfSelfReferenceConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section2TheStrangeLoopL5TheGeometryOfSelfReferenceConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04441"):
        Section2TheStrangeLoopL5TheGeometryOfSelfReferenceConfig(next_source_boundary="P0R04440")
    with pytest.raises(
        ValueError,
        match="unknown section_2_the_strange_loop_l5_the_geometry_of_self_reference component",
    ):
        classify_section_2_the_strange_loop_l5_the_geometry_of_self_reference_component(
            "empirical_validation_claim"
        )
