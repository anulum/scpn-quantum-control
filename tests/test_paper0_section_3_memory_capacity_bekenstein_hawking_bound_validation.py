# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. Memory Capacity (Bekenstein-Hawking Bound) validation tests
"""Tests for Paper 0 3. Memory Capacity (Bekenstein-Hawking Bound) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_3_memory_capacity_bekenstein_hawking_bound_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section3MemoryCapacityBekensteinHawkingBoundConfig,
    classify_section_3_memory_capacity_bekenstein_hawking_bound_component,
    section_3_memory_capacity_bekenstein_hawking_bound_labels,
    validate_section_3_memory_capacity_bekenstein_hawking_bound_fixture,
)


def test_section_3_memory_capacity_bekenstein_hawking_bound_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_3_memory_capacity_bekenstein_hawking_bound_fixture()
    assert result.source_ledger_span == ("P0R02257", "P0R02277")
    assert result.source_record_count == 21
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R02278"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_3_memory_capacity_bekenstein_hawking_bound_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02257"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02277"


def test_section_3_memory_capacity_bekenstein_hawking_bound_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "3_memory_capacity_bekenstein_hawking_bound",
        "p0r02263",
        "4_emergent_spacetime_ryu_takayanagi_formula",
    ):
        assert (
            classify_section_3_memory_capacity_bekenstein_hawking_bound_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_3_memory_capacity_bekenstein_hawking_bound_labels()
    assert labels["section"] == "3. Memory Capacity (Bekenstein-Hawking Bound)"
    assert labels["next_boundary"] == "P0R02278"


def test_section_3_memory_capacity_bekenstein_hawking_bound_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 21"):
        Section3MemoryCapacityBekensteinHawkingBoundConfig(expected_source_record_count=20)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section3MemoryCapacityBekensteinHawkingBoundConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02278"):
        Section3MemoryCapacityBekensteinHawkingBoundConfig(next_source_boundary="P0R02277")
    with pytest.raises(
        ValueError, match="unknown section_3_memory_capacity_bekenstein_hawking_bound component"
    ):
        classify_section_3_memory_capacity_bekenstein_hawking_bound_component(
            "empirical_validation_claim"
        )
