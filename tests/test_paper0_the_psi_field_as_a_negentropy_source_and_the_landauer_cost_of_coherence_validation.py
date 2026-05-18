# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Psi-Field as a Negentropy Source and the Landauer Cost of Coherence validation tests
"""Tests for Paper 0 The Psi-Field as a Negentropy Source and the Landauer Cost of Coherence source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ThePsiFieldAsANegentropySourceAndTheLandauerCostOfCoherenceConfig,
    classify_the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence_component,
    the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence_labels,
    validate_the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence_fixture,
)


def test_the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence_fixture()
    )
    assert result.source_ledger_span == ("P0R05953", "P0R05963")
    assert result.source_record_count == 11
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05964"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05953"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05963"


def test_the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence",
        "satisfying_the_generalised_second_law_gsl_the_cost_of_erasure",
        "the_thermodynamic_balance",
    ):
        assert (
            classify_the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence_labels()
    assert (
        labels["section"]
        == "The Psi-Field as a Negentropy Source and the Landauer Cost of Coherence"
    )
    assert labels["next_boundary"] == "P0R05964"


def test_the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        ThePsiFieldAsANegentropySourceAndTheLandauerCostOfCoherenceConfig(
            expected_source_record_count=10
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        ThePsiFieldAsANegentropySourceAndTheLandauerCostOfCoherenceConfig(
            expected_component_count=4
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05964"):
        ThePsiFieldAsANegentropySourceAndTheLandauerCostOfCoherenceConfig(
            next_source_boundary="P0R05963"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence component",
    ):
        classify_the_psi_field_as_a_negentropy_source_and_the_landauer_cost_of_coherence_component(
            "empirical_validation_claim"
        )
