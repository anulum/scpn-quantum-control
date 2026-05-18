# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. L4 Impact (Dampening Dynamics and Shifting Criticality): validation tests
"""Tests for Paper 0 2. L4 Impact (Dampening Dynamics and Shifting Criticality): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_l4_impact_dampening_dynamics_and_shifting_criticality_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2L4ImpactDampeningDynamicsAndShiftingCriticalityConfig,
    classify_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_component,
    section_2_l4_impact_dampening_dynamics_and_shifting_criticality_labels,
    validate_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_fixture,
)


def test_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_fixture()
    assert result.source_ledger_span == ("P0R05083", "P0R05090")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05091"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05083"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05090"


def test_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "2_l4_impact_dampening_dynamics_and_shifting_criticality",
        "3_l5_impact_analgesia_and_geometric_remodelling",
    ):
        assert (
            classify_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_2_l4_impact_dampening_dynamics_and_shifting_criticality_labels()
    assert labels["section"] == "2. L4 Impact (Dampening Dynamics and Shifting Criticality):"
    assert labels["next_boundary"] == "P0R05091"


def test_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section2L4ImpactDampeningDynamicsAndShiftingCriticalityConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section2L4ImpactDampeningDynamicsAndShiftingCriticalityConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05091"):
        Section2L4ImpactDampeningDynamicsAndShiftingCriticalityConfig(
            next_source_boundary="P0R05090"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_2_l4_impact_dampening_dynamics_and_shifting_criticality component",
    ):
        classify_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_component(
            "empirical_validation_claim"
        )
