# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Foundation of the Biological Substrate validation tests
"""Tests for Paper 0 The Foundation of the Biological Substrate source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_foundation_of_the_biological_substrate_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheFoundationOfTheBiologicalSubstrateConfig,
    classify_the_foundation_of_the_biological_substrate_component,
    the_foundation_of_the_biological_substrate_labels,
    validate_the_foundation_of_the_biological_substrate_fixture,
)


def test_the_foundation_of_the_biological_substrate_fixture_preserves_source_boundary() -> None:
    result = validate_the_foundation_of_the_biological_substrate_fixture()
    assert result.source_ledger_span == ("P0R05306", "P0R05313")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05314"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_foundation_of_the_biological_substrate_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05306"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05313"


def test_the_foundation_of_the_biological_substrate_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_foundation_of_the_biological_substrate",
        "i_the_qed_of_water_coherence_domains_cds",
        "ii_the_emergence_of_life_abiogenesis_within_the_scpn",
    ):
        assert (
            classify_the_foundation_of_the_biological_substrate_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_foundation_of_the_biological_substrate_labels()
    assert labels["section"] == "The Foundation of the Biological Substrate"
    assert labels["next_boundary"] == "P0R05314"


def test_the_foundation_of_the_biological_substrate_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        TheFoundationOfTheBiologicalSubstrateConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        TheFoundationOfTheBiologicalSubstrateConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05314"):
        TheFoundationOfTheBiologicalSubstrateConfig(next_source_boundary="P0R05313")
    with pytest.raises(
        ValueError, match="unknown the_foundation_of_the_biological_substrate component"
    ):
        classify_the_foundation_of_the_biological_substrate_component("empirical_validation_claim")
