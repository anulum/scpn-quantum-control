# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Philosophy & Ethics Anchors validation tests
"""Tests for Paper 0  Philosophy & Ethics Anchors source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.philosophy_ethics_anchors_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    PhilosophyEthicsAnchorsConfig,
    classify_philosophy_ethics_anchors_component,
    philosophy_ethics_anchors_labels,
    validate_philosophy_ethics_anchors_fixture,
)


def test_philosophy_ethics_anchors_fixture_preserves_source_boundary() -> None:
    result = validate_philosophy_ethics_anchors_fixture()
    assert result.source_ledger_span == ("P0R05770", "P0R05777")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05778"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_philosophy_ethics_anchors_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05770"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05777"


def test_philosophy_ethics_anchors_classification_and_labels_are_explicit() -> None:
    for component in ("philosophy_ethics_anchors", "philosophy_of_science_methodology"):
        assert (
            classify_philosophy_ethics_anchors_component(component)
            == f"{component}_source_boundary"
        )
    labels = philosophy_ethics_anchors_labels()
    assert labels["section"] == " Philosophy & Ethics Anchors"
    assert labels["next_boundary"] == "P0R05778"


def test_philosophy_ethics_anchors_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        PhilosophyEthicsAnchorsConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        PhilosophyEthicsAnchorsConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05778"):
        PhilosophyEthicsAnchorsConfig(next_source_boundary="P0R05777")
    with pytest.raises(ValueError, match="unknown philosophy_ethics_anchors component"):
        classify_philosophy_ethics_anchors_component("empirical_validation_claim")
