# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Two-timescale structure (definitions). validation tests
"""Tests for Paper 0 Two-timescale structure (definitions). source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.two_timescale_structure_definitions_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TwoTimescaleStructureDefinitionsConfig,
    classify_two_timescale_structure_definitions_component,
    two_timescale_structure_definitions_labels,
    validate_two_timescale_structure_definitions_fixture,
)


def test_two_timescale_structure_definitions_fixture_preserves_source_boundary() -> None:
    result = validate_two_timescale_structure_definitions_fixture()
    assert result.source_ledger_span == ("P0R02958", "P0R02966")
    assert result.source_record_count == 9
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R02967"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_two_timescale_structure_definitions_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02958"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02966"


def test_two_timescale_structure_definitions_classification_and_labels_are_explicit() -> None:
    for component in (
        "two_timescale_structure_definitions",
        "gain_scheduling_via_the_affective_field",
    ):
        assert (
            classify_two_timescale_structure_definitions_component(component)
            == f"{component}_source_boundary"
        )
    labels = two_timescale_structure_definitions_labels()
    assert labels["section"] == "Two-timescale structure (definitions)."
    assert labels["next_boundary"] == "P0R02967"


def test_two_timescale_structure_definitions_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        TwoTimescaleStructureDefinitionsConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TwoTimescaleStructureDefinitionsConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02967"):
        TwoTimescaleStructureDefinitionsConfig(next_source_boundary="P0R02966")
    with pytest.raises(ValueError, match="unknown two_timescale_structure_definitions component"):
        classify_two_timescale_structure_definitions_component("empirical_validation_claim")
