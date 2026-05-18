# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. Impedance Matching: validation tests
"""Tests for Paper 0 2. Impedance Matching: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_impedance_matching_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2ImpedanceMatchingConfig,
    classify_section_2_impedance_matching_component,
    section_2_impedance_matching_labels,
    validate_section_2_impedance_matching_fixture,
)


def test_section_2_impedance_matching_fixture_preserves_source_boundary() -> None:
    result = validate_section_2_impedance_matching_fixture()
    assert result.source_ledger_span == ("P0R05641", "P0R05649")
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05650"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_impedance_matching_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05641"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05649"


def test_section_2_impedance_matching_classification_and_labels_are_explicit() -> None:
    for component in (
        "2_impedance_matching",
        "citations_cross_domain_anchors",
        "cross_domain_anchors_core_dynamics",
    ):
        assert (
            classify_section_2_impedance_matching_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_2_impedance_matching_labels()
    assert labels["section"] == "2. Impedance Matching:"
    assert labels["next_boundary"] == "P0R05650"


def test_section_2_impedance_matching_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        Section2ImpedanceMatchingConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section2ImpedanceMatchingConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05650"):
        Section2ImpedanceMatchingConfig(next_source_boundary="P0R05649")
    with pytest.raises(ValueError, match="unknown section_2_impedance_matching component"):
        classify_section_2_impedance_matching_component("empirical_validation_claim")
