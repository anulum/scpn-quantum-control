# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Components: validation tests
"""Tests for Paper 0 Components: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.components_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ComponentsConfig,
    classify_components_component,
    components_labels,
    validate_components_fixture,
)


def test_components_fixture_preserves_source_boundary() -> None:
    result = validate_components_fixture()
    assert result.source_ledger_span == ("P0R01779", "P0R01791")
    assert result.source_record_count == 13
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R01792"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert result.problem_metadata["protocol_state"] == "source_components_only_no_experiment"
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R01779"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R01791"


def test_components_classification_and_labels_are_explicit() -> None:
    for component in ("components",):
        assert classify_components_component(component) == f"{component}_source_boundary"
    labels = components_labels()
    assert labels["section"] == "Components:"
    assert labels["next_boundary"] == "P0R01792"


def test_components_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 13"):
        ComponentsConfig(expected_source_record_count=12)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        ComponentsConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01792"):
        ComponentsConfig(next_source_boundary="P0R01791")
    with pytest.raises(ValueError, match="unknown components component"):
        classify_components_component("empirical_validation_claim")
