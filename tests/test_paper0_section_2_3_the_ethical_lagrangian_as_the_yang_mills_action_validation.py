# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2.3. The Ethical Lagrangian as the Yang-Mills Action validation tests
"""Tests for Paper 0 2.3. The Ethical Lagrangian as the Yang-Mills Action source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section23TheEthicalLagrangianAsTheYangMillsActionConfig,
    classify_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_component,
    section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_labels,
    validate_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_fixture,
)


def test_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_fixture()
    assert result.source_ledger_span == ("P0R03622", "P0R03629")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03630"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03622"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03629"


def test_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "2_3_the_ethical_lagrangian_as_the_yang_mills_action",
        "section_3_the_conserved_ethical_charge_and_its_physical_basis",
    ):
        assert (
            classify_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_labels()
    assert labels["section"] == "2.3. The Ethical Lagrangian as the Yang-Mills Action"
    assert labels["next_boundary"] == "P0R03630"


def test_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section23TheEthicalLagrangianAsTheYangMillsActionConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section23TheEthicalLagrangianAsTheYangMillsActionConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03630"):
        Section23TheEthicalLagrangianAsTheYangMillsActionConfig(next_source_boundary="P0R03629")
    with pytest.raises(
        ValueError,
        match="unknown section_2_3_the_ethical_lagrangian_as_the_yang_mills_action component",
    ):
        classify_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_component(
            "empirical_validation_claim"
        )
