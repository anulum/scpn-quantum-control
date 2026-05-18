# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4.5 The Strange Loop of Closure - Meta-Layer 16 and The Anulum validation tests
"""Tests for Paper 0 4.5 The Strange Loop of Closure - Meta-Layer 16 and The Anulum source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section45TheStrangeLoopOfClosureMetaLayer16AndTheAnulumConfig,
    classify_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_component,
    section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_labels,
    validate_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_fixture,
)


def test_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_fixture()
    )
    assert result.source_ledger_span == ("P0R04216", "P0R04223")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04224"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04216"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04223"


def test_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum",
        "a_note_on_cybernetic_closure",
    ):
        assert (
            classify_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_labels()
    assert labels["section"] == "4.5 The Strange Loop of Closure - Meta-Layer 16 and The Anulum"
    assert labels["next_boundary"] == "P0R04224"


def test_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section45TheStrangeLoopOfClosureMetaLayer16AndTheAnulumConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section45TheStrangeLoopOfClosureMetaLayer16AndTheAnulumConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04224"):
        Section45TheStrangeLoopOfClosureMetaLayer16AndTheAnulumConfig(
            next_source_boundary="P0R04223"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum component",
    ):
        classify_section_4_5_the_strange_loop_of_closure_meta_layer_16_and_the_anulum_component(
            "empirical_validation_claim"
        )
