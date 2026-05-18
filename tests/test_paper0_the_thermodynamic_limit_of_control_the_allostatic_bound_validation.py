# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Thermodynamic Limit of Control: The Allostatic Bound validation tests
"""Tests for Paper 0 The Thermodynamic Limit of Control: The Allostatic Bound source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_thermodynamic_limit_of_control_the_allostatic_bound_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheThermodynamicLimitOfControlTheAllostaticBoundConfig,
    classify_the_thermodynamic_limit_of_control_the_allostatic_bound_component,
    the_thermodynamic_limit_of_control_the_allostatic_bound_labels,
    validate_the_thermodynamic_limit_of_control_the_allostatic_bound_fixture,
)


def test_the_thermodynamic_limit_of_control_the_allostatic_bound_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_thermodynamic_limit_of_control_the_allostatic_bound_fixture()
    assert result.source_ledger_span == ("P0R05408", "P0R05419")
    assert result.source_record_count == 12
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R05420"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_thermodynamic_limit_of_control_the_allostatic_bound_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05408"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05419"


def test_the_thermodynamic_limit_of_control_the_allostatic_bound_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_thermodynamic_limit_of_control_the_allostatic_bound",):
        assert (
            classify_the_thermodynamic_limit_of_control_the_allostatic_bound_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_thermodynamic_limit_of_control_the_allostatic_bound_labels()
    assert labels["section"] == "The Thermodynamic Limit of Control: The Allostatic Bound"
    assert labels["next_boundary"] == "P0R05420"


def test_the_thermodynamic_limit_of_control_the_allostatic_bound_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 12"):
        TheThermodynamicLimitOfControlTheAllostaticBoundConfig(expected_source_record_count=11)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheThermodynamicLimitOfControlTheAllostaticBoundConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05420"):
        TheThermodynamicLimitOfControlTheAllostaticBoundConfig(next_source_boundary="P0R05419")
    with pytest.raises(
        ValueError,
        match="unknown the_thermodynamic_limit_of_control_the_allostatic_bound component",
    ):
        classify_the_thermodynamic_limit_of_control_the_allostatic_bound_component(
            "empirical_validation_claim"
        )
