# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Functional Implications: validation tests
"""Tests for Paper 0 Functional Implications: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.functional_implications_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    FunctionalImplicationsConfig,
    classify_functional_implications_component,
    functional_implications_labels,
    validate_functional_implications_fixture,
)


def test_functional_implications_fixture_preserves_source_boundary() -> None:
    result = validate_functional_implications_fixture()
    assert result.source_ledger_span == ("P0R02859", "P0R02868")
    assert result.source_record_count == 10
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R02869"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_functional_implications_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02859"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02868"


def test_functional_implications_classification_and_labels_are_explicit() -> None:
    for component in (
        "functional_implications",
        "maximised_information_capacity",
        "efficient_communication",
    ):
        assert (
            classify_functional_implications_component(component) == f"{component}_source_boundary"
        )
    labels = functional_implications_labels()
    assert labels["section"] == "Functional Implications:"
    assert labels["next_boundary"] == "P0R02869"


def test_functional_implications_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        FunctionalImplicationsConfig(expected_source_record_count=9)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        FunctionalImplicationsConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02869"):
        FunctionalImplicationsConfig(next_source_boundary="P0R02868")
    with pytest.raises(ValueError, match="unknown functional_implications component"):
        classify_functional_implications_component("empirical_validation_claim")
