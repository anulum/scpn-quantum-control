# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 H_int as the Selection Operator: validation tests
"""Tests for Paper 0 H_int as the Selection Operator: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.h_int_as_the_selection_operator_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    HIntAsTheSelectionOperatorConfig,
    classify_h_int_as_the_selection_operator_component,
    h_int_as_the_selection_operator_labels,
    validate_h_int_as_the_selection_operator_fixture,
)


def test_h_int_as_the_selection_operator_fixture_preserves_source_boundary() -> None:
    result = validate_h_int_as_the_selection_operator_fixture()
    assert result.source_ledger_span == ("P0R03324", "P0R03331")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03332"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_h_int_as_the_selection_operator_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03324"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03331"


def test_h_int_as_the_selection_operator_classification_and_labels_are_explicit() -> None:
    for component in ("h_int_as_the_selection_operator", "qsr_as_the_energy_transducer"):
        assert (
            classify_h_int_as_the_selection_operator_component(component)
            == f"{component}_source_boundary"
        )
    labels = h_int_as_the_selection_operator_labels()
    assert labels["section"] == "H_int as the Selection Operator:"
    assert labels["next_boundary"] == "P0R03332"


def test_h_int_as_the_selection_operator_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        HIntAsTheSelectionOperatorConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        HIntAsTheSelectionOperatorConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03332"):
        HIntAsTheSelectionOperatorConfig(next_source_boundary="P0R03331")
    with pytest.raises(ValueError, match="unknown h_int_as_the_selection_operator component"):
        classify_h_int_as_the_selection_operator_component("empirical_validation_claim")
