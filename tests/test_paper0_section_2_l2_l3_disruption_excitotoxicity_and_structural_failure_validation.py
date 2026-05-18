# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. L2/L3 Disruption (Excitotoxicity and Structural Failure): validation tests
"""Tests for Paper 0 2. L2/L3 Disruption (Excitotoxicity and Structural Failure): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2L2L3DisruptionExcitotoxicityAndStructuralFailureConfig,
    classify_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_component,
    section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_labels,
    validate_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_fixture,
)


def test_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_fixture()
    assert result.source_ledger_span == ("P0R05058", "P0R05065")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05066"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05058"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05065"


def test_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "2_l2_l3_disruption_excitotoxicity_and_structural_failure",
        "3_l4_disruption_dyscritia_and_desynchronization",
    ):
        assert (
            classify_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_labels()
    assert labels["section"] == "2. L2/L3 Disruption (Excitotoxicity and Structural Failure):"
    assert labels["next_boundary"] == "P0R05066"


def test_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section2L2L3DisruptionExcitotoxicityAndStructuralFailureConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section2L2L3DisruptionExcitotoxicityAndStructuralFailureConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05066"):
        Section2L2L3DisruptionExcitotoxicityAndStructuralFailureConfig(
            next_source_boundary="P0R05065"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_2_l2_l3_disruption_excitotoxicity_and_structural_failure component",
    ):
        classify_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_component(
            "empirical_validation_claim"
        )
