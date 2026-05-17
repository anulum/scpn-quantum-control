# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Overarching Dynamic Principles and the Mathematical Spine validation tests
"""Tests for Paper 0 Overarching Dynamic Principles and the Mathematical Spine source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.overarching_dynamic_principles_and_the_mathematical_spine_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    OverarchingDynamicPrinciplesAndTheMathematicalSpineConfig,
    classify_overarching_dynamic_principles_and_the_mathematical_spine_component,
    overarching_dynamic_principles_and_the_mathematical_spine_labels,
    validate_overarching_dynamic_principles_and_the_mathematical_spine_fixture,
)


def test_overarching_dynamic_principles_and_the_mathematical_spine_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_overarching_dynamic_principles_and_the_mathematical_spine_fixture()
    assert result.source_ledger_span == ("P0R02502", "P0R02512")
    assert result.source_record_count == 11
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R02513"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_overarching_dynamic_principles_and_the_mathematical_spine_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02502"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02512"


def test_overarching_dynamic_principles_and_the_mathematical_spine_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "overarching_dynamic_principles_and_the_mathematical_spine",
        "i_the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn",
    ):
        assert (
            classify_overarching_dynamic_principles_and_the_mathematical_spine_component(component)
            == f"{component}_source_boundary"
        )
    labels = overarching_dynamic_principles_and_the_mathematical_spine_labels()
    assert labels["section"] == "Overarching Dynamic Principles and the Mathematical Spine"
    assert labels["next_boundary"] == "P0R02513"


def test_overarching_dynamic_principles_and_the_mathematical_spine_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        OverarchingDynamicPrinciplesAndTheMathematicalSpineConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        OverarchingDynamicPrinciplesAndTheMathematicalSpineConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02513"):
        OverarchingDynamicPrinciplesAndTheMathematicalSpineConfig(next_source_boundary="P0R02512")
    with pytest.raises(
        ValueError,
        match="unknown overarching_dynamic_principles_and_the_mathematical_spine component",
    ):
        classify_overarching_dynamic_principles_and_the_mathematical_spine_component(
            "empirical_validation_claim"
        )
