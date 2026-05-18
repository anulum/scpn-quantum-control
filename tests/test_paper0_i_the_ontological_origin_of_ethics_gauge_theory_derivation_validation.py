# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 I. The Ontological Origin of Ethics (Gauge Theory Derivation) validation tests
"""Tests for Paper 0 I. The Ontological Origin of Ethics (Gauge Theory Derivation) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.i_the_ontological_origin_of_ethics_gauge_theory_derivation_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ITheOntologicalOriginOfEthicsGaugeTheoryDerivationConfig,
    classify_i_the_ontological_origin_of_ethics_gauge_theory_derivation_component,
    i_the_ontological_origin_of_ethics_gauge_theory_derivation_labels,
    validate_i_the_ontological_origin_of_ethics_gauge_theory_derivation_fixture,
)


def test_i_the_ontological_origin_of_ethics_gauge_theory_derivation_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_i_the_ontological_origin_of_ethics_gauge_theory_derivation_fixture()
    assert result.source_ledger_span == ("P0R03968", "P0R03980")
    assert result.source_record_count == 13
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03981"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_i_the_ontological_origin_of_ethics_gauge_theory_derivation_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03968"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03980"


def test_i_the_ontological_origin_of_ethics_gauge_theory_derivation_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("i_the_ontological_origin_of_ethics_gauge_theory_derivation",):
        assert (
            classify_i_the_ontological_origin_of_ethics_gauge_theory_derivation_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = i_the_ontological_origin_of_ethics_gauge_theory_derivation_labels()
    assert labels["section"] == "I. The Ontological Origin of Ethics (Gauge Theory Derivation)"
    assert labels["next_boundary"] == "P0R03981"


def test_i_the_ontological_origin_of_ethics_gauge_theory_derivation_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 13"):
        ITheOntologicalOriginOfEthicsGaugeTheoryDerivationConfig(expected_source_record_count=12)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        ITheOntologicalOriginOfEthicsGaugeTheoryDerivationConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03981"):
        ITheOntologicalOriginOfEthicsGaugeTheoryDerivationConfig(next_source_boundary="P0R03980")
    with pytest.raises(
        ValueError,
        match="unknown i_the_ontological_origin_of_ethics_gauge_theory_derivation component",
    ):
        classify_i_the_ontological_origin_of_ethics_gauge_theory_derivation_component(
            "empirical_validation_claim"
        )
