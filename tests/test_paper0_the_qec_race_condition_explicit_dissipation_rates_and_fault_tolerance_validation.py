# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The QEC Race Condition: Explicit Dissipation Rates and Fault Tolerance validation tests
"""Tests for Paper 0 The QEC Race Condition: Explicit Dissipation Rates and Fault Tolerance source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheQecRaceConditionExplicitDissipationRatesAndFaultToleranceConfig,
    classify_the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance_component,
    the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance_labels,
    validate_the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance_fixture,
)


def test_the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance_fixture()
    )
    assert result.source_ledger_span == ("P0R03099", "P0R03121")
    assert result.source_record_count == 23
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03122"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03099"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03121"


def test_the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance",):
        assert (
            classify_the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance_labels()
    assert (
        labels["section"]
        == "The QEC Race Condition: Explicit Dissipation Rates and Fault Tolerance"
    )
    assert labels["next_boundary"] == "P0R03122"


def test_the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 23"):
        TheQecRaceConditionExplicitDissipationRatesAndFaultToleranceConfig(
            expected_source_record_count=22
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheQecRaceConditionExplicitDissipationRatesAndFaultToleranceConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03122"):
        TheQecRaceConditionExplicitDissipationRatesAndFaultToleranceConfig(
            next_source_boundary="P0R03121"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance component",
    ):
        classify_the_qec_race_condition_explicit_dissipation_rates_and_fault_tolerance_component(
            "empirical_validation_claim"
        )
