# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Systems & Cybernetics validation tests
"""Tests for Paper 0  Systems & Cybernetics source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.systems_cybernetics_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    SystemsCyberneticsConfig,
    classify_systems_cybernetics_component,
    systems_cybernetics_labels,
    validate_systems_cybernetics_fixture,
)


def test_systems_cybernetics_fixture_preserves_source_boundary() -> None:
    result = validate_systems_cybernetics_fixture()
    assert result.source_ledger_span == ("P0R05844", "P0R05851")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05852"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_systems_cybernetics_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05844"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05851"


def test_systems_cybernetics_classification_and_labels_are_explicit() -> None:
    for component in ("systems_cybernetics", "cognitive_science_language"):
        assert classify_systems_cybernetics_component(component) == f"{component}_source_boundary"
    labels = systems_cybernetics_labels()
    assert labels["section"] == " Systems & Cybernetics"
    assert labels["next_boundary"] == "P0R05852"


def test_systems_cybernetics_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        SystemsCyberneticsConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        SystemsCyberneticsConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05852"):
        SystemsCyberneticsConfig(next_source_boundary="P0R05851")
    with pytest.raises(ValueError, match="unknown systems_cybernetics component"):
        classify_systems_cybernetics_component("empirical_validation_claim")
