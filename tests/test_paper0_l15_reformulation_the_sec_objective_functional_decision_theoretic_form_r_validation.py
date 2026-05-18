# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 L15 Reformulation: The SEC Objective Functional (Decision-Theoretic Form) (revision 11.00) validation tests
"""Tests for Paper 0 L15 Reformulation: The SEC Objective Functional (Decision-Theoretic Form) (revision 11.00) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRConfig,
    classify_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_component,
    l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_labels,
    validate_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_fixture,
)


def test_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_fixture()
    )
    assert result.source_ledger_span == ("P0R03981", "P0R04000")
    assert result.source_record_count == 20
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04001"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03981"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04000"


def test_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r",
        "definition_canonical_form",
    ):
        assert (
            classify_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_labels()
    assert (
        labels["section"]
        == "L15 Reformulation: The SEC Objective Functional (Decision-Theoretic Form) (revision 11.00)"
    )
    assert labels["next_boundary"] == "P0R04001"


def test_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 20"):
        L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRConfig(
            expected_source_record_count=19
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRConfig(
            expected_component_count=3
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04001"):
        L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRConfig(
            next_source_boundary="P0R04000"
        )
    with pytest.raises(
        ValueError,
        match="unknown l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r component",
    ):
        classify_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_component(
            "empirical_validation_claim"
        )
