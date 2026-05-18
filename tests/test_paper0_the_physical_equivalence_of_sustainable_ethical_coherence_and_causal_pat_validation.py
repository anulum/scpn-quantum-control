# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Physical Equivalence of Sustainable Ethical Coherence and Causal Path Entropy validation tests
"""Tests for Paper 0 The Physical Equivalence of Sustainable Ethical Coherence and Causal Path Entropy source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ThePhysicalEquivalenceOfSustainableEthicalCoherenceAndCausalPatConfig,
    classify_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_component,
    the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_labels,
    validate_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_fixture,
)


def test_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_fixture()
    )
    assert result.source_ledger_span == ("P0R03723", "P0R03736")
    assert result.source_record_count == 14
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03737"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03723"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03736"


def test_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat",
        "1_introduction_from_teleological_principle_to_thermodynamic_imperative",
    ):
        assert (
            classify_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_labels()
    assert (
        labels["section"]
        == "The Physical Equivalence of Sustainable Ethical Coherence and Causal Path Entropy"
    )
    assert labels["next_boundary"] == "P0R03737"


def test_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 14"):
        ThePhysicalEquivalenceOfSustainableEthicalCoherenceAndCausalPatConfig(
            expected_source_record_count=13
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        ThePhysicalEquivalenceOfSustainableEthicalCoherenceAndCausalPatConfig(
            expected_component_count=3
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03737"):
        ThePhysicalEquivalenceOfSustainableEthicalCoherenceAndCausalPatConfig(
            next_source_boundary="P0R03736"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat component",
    ):
        classify_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_component(
            "empirical_validation_claim"
        )
