# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Mathematical Bridge: validation tests
"""Tests for Paper 0 Mathematical Bridge: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.mathematical_bridge_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MathematicalBridgeConfig,
    classify_mathematical_bridge_component,
    mathematical_bridge_labels,
    validate_mathematical_bridge_fixture,
)


def test_mathematical_bridge_fixture_preserves_source_boundary() -> None:
    result = validate_mathematical_bridge_fixture()
    assert result.source_ledger_span == ("P0R03427", "P0R03439")
    assert result.source_record_count == 13
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03440"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_mathematical_bridge_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03427"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03439"


def test_mathematical_bridge_classification_and_labels_are_explicit() -> None:
    for component in ("mathematical_bridge", "why_there_s_something_it_s_like"):
        assert classify_mathematical_bridge_component(component) == f"{component}_source_boundary"
    labels = mathematical_bridge_labels()
    assert labels["section"] == "Mathematical Bridge:"
    assert labels["next_boundary"] == "P0R03440"


def test_mathematical_bridge_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 13"):
        MathematicalBridgeConfig(expected_source_record_count=12)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        MathematicalBridgeConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03440"):
        MathematicalBridgeConfig(next_source_boundary="P0R03439")
    with pytest.raises(ValueError, match="unknown mathematical_bridge component"):
        classify_mathematical_bridge_component("empirical_validation_claim")
