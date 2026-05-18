# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Computational & AI Alignment validation tests
"""Tests for Paper 0  Computational & AI Alignment source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.computational_ai_alignment_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ComputationalAiAlignmentConfig,
    classify_computational_ai_alignment_component,
    computational_ai_alignment_labels,
    validate_computational_ai_alignment_fixture,
)


def test_computational_ai_alignment_fixture_preserves_source_boundary() -> None:
    result = validate_computational_ai_alignment_fixture()
    assert result.source_ledger_span == ("P0R05852", "P0R05859")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05860"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_computational_ai_alignment_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05852"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05859"


def test_computational_ai_alignment_classification_and_labels_are_explicit() -> None:
    for component in ("computational_ai_alignment", "complex_systems_criticality"):
        assert (
            classify_computational_ai_alignment_component(component)
            == f"{component}_source_boundary"
        )
    labels = computational_ai_alignment_labels()
    assert labels["section"] == " Computational & AI Alignment"
    assert labels["next_boundary"] == "P0R05860"


def test_computational_ai_alignment_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        ComputationalAiAlignmentConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        ComputationalAiAlignmentConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05860"):
        ComputationalAiAlignmentConfig(next_source_boundary="P0R05859")
    with pytest.raises(ValueError, match="unknown computational_ai_alignment component"):
        classify_computational_ai_alignment_component("empirical_validation_claim")
