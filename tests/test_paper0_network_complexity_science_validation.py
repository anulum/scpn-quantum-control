# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Network & Complexity Science validation tests
"""Tests for Paper 0  Network & Complexity Science source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.network_complexity_science_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    NetworkComplexityScienceConfig,
    classify_network_complexity_science_component,
    network_complexity_science_labels,
    validate_network_complexity_science_fixture,
)


def test_network_complexity_science_fixture_preserves_source_boundary() -> None:
    result = validate_network_complexity_science_fixture()
    assert result.source_ledger_span == ("P0R05810", "P0R05817")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05818"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_network_complexity_science_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05810"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05817"


def test_network_complexity_science_classification_and_labels_are_explicit() -> None:
    for component in ("network_complexity_science", "cognitive_computational_foundations"):
        assert (
            classify_network_complexity_science_component(component)
            == f"{component}_source_boundary"
        )
    labels = network_complexity_science_labels()
    assert labels["section"] == " Network & Complexity Science"
    assert labels["next_boundary"] == "P0R05818"


def test_network_complexity_science_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        NetworkComplexityScienceConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        NetworkComplexityScienceConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05818"):
        NetworkComplexityScienceConfig(next_source_boundary="P0R05817")
    with pytest.raises(ValueError, match="unknown network_complexity_science component"):
        classify_network_complexity_science_component("empirical_validation_claim")
