# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. The Central Void (The Source): validation tests
"""Tests for Paper 0 2. The Central Void (The Source): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_the_central_void_the_source_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2TheCentralVoidTheSourceConfig,
    classify_section_2_the_central_void_the_source_component,
    section_2_the_central_void_the_source_labels,
    validate_section_2_the_central_void_the_source_fixture,
)


def test_section_2_the_central_void_the_source_fixture_preserves_source_boundary() -> None:
    result = validate_section_2_the_central_void_the_source_fixture()
    assert result.source_ledger_span == ("P0R02566", "P0R02579")
    assert result.source_record_count == 14
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R02580"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_the_central_void_the_source_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02566"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02579"


def test_section_2_the_central_void_the_source_classification_and_labels_are_explicit() -> None:
    for component in (
        "2_the_central_void_the_source",
        "3_the_surface_dynamics_criticality_and_coherence",
        "4_the_attractor_l15_l8",
    ):
        assert (
            classify_section_2_the_central_void_the_source_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_2_the_central_void_the_source_labels()
    assert labels["section"] == "2. The Central Void (The Source):"
    assert labels["next_boundary"] == "P0R02580"


def test_section_2_the_central_void_the_source_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 14"):
        Section2TheCentralVoidTheSourceConfig(expected_source_record_count=13)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section2TheCentralVoidTheSourceConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02580"):
        Section2TheCentralVoidTheSourceConfig(next_source_boundary="P0R02579")
    with pytest.raises(
        ValueError, match="unknown section_2_the_central_void_the_source component"
    ):
        classify_section_2_the_central_void_the_source_component("empirical_validation_claim")
