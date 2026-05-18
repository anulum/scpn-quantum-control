# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Ethics & Teleology validation tests
"""Tests for Paper 0  Ethics & Teleology source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ethics_teleology_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    EthicsTeleologyConfig,
    classify_ethics_teleology_component,
    ethics_teleology_labels,
    validate_ethics_teleology_fixture,
)


def test_ethics_teleology_fixture_preserves_source_boundary() -> None:
    result = validate_ethics_teleology_fixture()
    assert result.source_ledger_span == ("P0R05762", "P0R05769")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05770"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"] == "source_ethics_teleology_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05762"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05769"


def test_ethics_teleology_classification_and_labels_are_explicit() -> None:
    for component in ("ethics_teleology", "ethics_evolution_systems"):
        assert classify_ethics_teleology_component(component) == f"{component}_source_boundary"
    labels = ethics_teleology_labels()
    assert labels["section"] == " Ethics & Teleology"
    assert labels["next_boundary"] == "P0R05770"


def test_ethics_teleology_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        EthicsTeleologyConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        EthicsTeleologyConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05770"):
        EthicsTeleologyConfig(next_source_boundary="P0R05769")
    with pytest.raises(ValueError, match="unknown ethics_teleology component"):
        classify_ethics_teleology_component("empirical_validation_claim")
