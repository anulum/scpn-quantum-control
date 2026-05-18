# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 I. The Architecture of Time: The Meta Metatron Cycle and Retrocausality validation tests
"""Tests for Paper 0 I. The Architecture of Time: The Meta Metatron Cycle and Retrocausality source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ITheArchitectureOfTimeTheMetaMetatronCycleAndRetrocausalityConfig,
    classify_i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_component,
    i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_labels,
    validate_i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_fixture,
)


def test_i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_fixture()
    )
    assert result.source_ledger_span == ("P0R05928", "P0R05935")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05936"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05928"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05935"


def test_i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality",
        "1_the_cyclic_operator_and_reversibility",
        "2_emergence_of_the_arrow_of_time",
    ):
        assert (
            classify_i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_labels()
    assert (
        labels["section"]
        == "I. The Architecture of Time: The Meta Metatron Cycle and Retrocausality"
    )
    assert labels["next_boundary"] == "P0R05936"


def test_i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        ITheArchitectureOfTimeTheMetaMetatronCycleAndRetrocausalityConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        ITheArchitectureOfTimeTheMetaMetatronCycleAndRetrocausalityConfig(
            expected_component_count=4
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05936"):
        ITheArchitectureOfTimeTheMetaMetatronCycleAndRetrocausalityConfig(
            next_source_boundary="P0R05935"
        )
    with pytest.raises(
        ValueError,
        match="unknown i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality component",
    ):
        classify_i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_component(
            "empirical_validation_claim"
        )
