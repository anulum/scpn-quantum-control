# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Resolution of the Observability Paradox (B-HJB) validation tests
"""Tests for Paper 0 Resolution of the Observability Paradox (B-HJB) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.resolution_of_the_observability_paradox_b_hjb_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ResolutionOfTheObservabilityParadoxBHjbConfig,
    classify_resolution_of_the_observability_paradox_b_hjb_component,
    resolution_of_the_observability_paradox_b_hjb_labels,
    validate_resolution_of_the_observability_paradox_b_hjb_fixture,
)


def test_resolution_of_the_observability_paradox_b_hjb_fixture_preserves_source_boundary() -> None:
    result = validate_resolution_of_the_observability_paradox_b_hjb_fixture()
    assert result.source_ledger_span == ("P0R02468", "P0R02484")
    assert result.source_record_count == 17
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R02485"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_resolution_of_the_observability_paradox_b_hjb_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02468"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02484"


def test_resolution_of_the_observability_paradox_b_hjb_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "resolution_of_the_observability_paradox_b_hjb",
        "overarching_dynamic_principles_a_synopsis",
    ):
        assert (
            classify_resolution_of_the_observability_paradox_b_hjb_component(component)
            == f"{component}_source_boundary"
        )
    labels = resolution_of_the_observability_paradox_b_hjb_labels()
    assert labels["section"] == "Resolution of the Observability Paradox (B-HJB)"
    assert labels["next_boundary"] == "P0R02485"


def test_resolution_of_the_observability_paradox_b_hjb_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 17"):
        ResolutionOfTheObservabilityParadoxBHjbConfig(expected_source_record_count=16)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        ResolutionOfTheObservabilityParadoxBHjbConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02485"):
        ResolutionOfTheObservabilityParadoxBHjbConfig(next_source_boundary="P0R02484")
    with pytest.raises(
        ValueError, match="unknown resolution_of_the_observability_paradox_b_hjb component"
    ):
        classify_resolution_of_the_observability_paradox_b_hjb_component(
            "empirical_validation_claim"
        )
