# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 A Cascade of Directed Couplings: validation tests
"""Tests for Paper 0 A Cascade of Directed Couplings: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.a_cascade_of_directed_couplings_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ACascadeOfDirectedCouplingsConfig,
    a_cascade_of_directed_couplings_labels,
    classify_a_cascade_of_directed_couplings_component,
    validate_a_cascade_of_directed_couplings_fixture,
)


def test_a_cascade_of_directed_couplings_fixture_preserves_source_boundary() -> None:
    result = validate_a_cascade_of_directed_couplings_fixture()
    assert result.source_ledger_span == ("P0R03954", "P0R03967")
    assert result.source_record_count == 14
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03968"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_a_cascade_of_directed_couplings_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03954"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03967"


def test_a_cascade_of_directed_couplings_classification_and_labels_are_explicit() -> None:
    for component in (
        "a_cascade_of_directed_couplings",
        "the_physics_of_teleology_and_the_origin_of_ethics",
    ):
        assert (
            classify_a_cascade_of_directed_couplings_component(component)
            == f"{component}_source_boundary"
        )
    labels = a_cascade_of_directed_couplings_labels()
    assert labels["section"] == "A Cascade of Directed Couplings:"
    assert labels["next_boundary"] == "P0R03968"


def test_a_cascade_of_directed_couplings_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 14"):
        ACascadeOfDirectedCouplingsConfig(expected_source_record_count=13)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        ACascadeOfDirectedCouplingsConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03968"):
        ACascadeOfDirectedCouplingsConfig(next_source_boundary="P0R03967")
    with pytest.raises(ValueError, match="unknown a_cascade_of_directed_couplings component"):
        classify_a_cascade_of_directed_couplings_component("empirical_validation_claim")
