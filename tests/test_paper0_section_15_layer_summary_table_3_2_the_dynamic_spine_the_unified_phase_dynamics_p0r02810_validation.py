# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint validation tests
"""Tests for Paper 0 15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsP0r02810Config,
    classify_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_component,
    section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_labels,
    validate_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_fixture,
)


def test_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_fixture()
    assert result.source_ledger_span == ("P0R02810", "P0R02830")
    assert result.source_record_count == 21
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02831"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02810"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02830"


def test_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",):
        assert (
            classify_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_labels()
    assert (
        labels["section"]
        == "15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint"
    )
    assert labels["next_boundary"] == "P0R02831"


def test_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 21"):
        Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsP0r02810Config(
            expected_source_record_count=20
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsP0r02810Config(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02831"):
        Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsP0r02810Config(
            next_source_boundary="P0R02830"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810 component",
    ):
        classify_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_component(
            "empirical_validation_claim"
        )
