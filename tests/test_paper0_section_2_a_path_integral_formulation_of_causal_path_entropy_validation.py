# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. A Path Integral Formulation of Causal Path Entropy validation tests
"""Tests for Paper 0 2. A Path Integral Formulation of Causal Path Entropy source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_a_path_integral_formulation_of_causal_path_entropy_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2APathIntegralFormulationOfCausalPathEntropyConfig,
    classify_section_2_a_path_integral_formulation_of_causal_path_entropy_component,
    section_2_a_path_integral_formulation_of_causal_path_entropy_labels,
    validate_section_2_a_path_integral_formulation_of_causal_path_entropy_fixture,
)


def test_section_2_a_path_integral_formulation_of_causal_path_entropy_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_a_path_integral_formulation_of_causal_path_entropy_fixture()
    assert result.source_ledger_span == ("P0R03737", "P0R03752")
    assert result.source_record_count == 16
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03753"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_a_path_integral_formulation_of_causal_path_entropy_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03737"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03752"


def test_section_2_a_path_integral_formulation_of_causal_path_entropy_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("2_a_path_integral_formulation_of_causal_path_entropy",):
        assert (
            classify_section_2_a_path_integral_formulation_of_causal_path_entropy_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_2_a_path_integral_formulation_of_causal_path_entropy_labels()
    assert labels["section"] == "2. A Path Integral Formulation of Causal Path Entropy"
    assert labels["next_boundary"] == "P0R03753"


def test_section_2_a_path_integral_formulation_of_causal_path_entropy_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 16"):
        Section2APathIntegralFormulationOfCausalPathEntropyConfig(expected_source_record_count=15)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        Section2APathIntegralFormulationOfCausalPathEntropyConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03753"):
        Section2APathIntegralFormulationOfCausalPathEntropyConfig(next_source_boundary="P0R03752")
    with pytest.raises(
        ValueError,
        match="unknown section_2_a_path_integral_formulation_of_causal_path_entropy component",
    ):
        classify_section_2_a_path_integral_formulation_of_causal_path_entropy_component(
            "empirical_validation_claim"
        )
