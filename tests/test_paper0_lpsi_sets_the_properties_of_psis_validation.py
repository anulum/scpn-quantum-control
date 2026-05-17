# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 LPsi Sets the Properties of Psis: validation tests
"""Tests for Paper 0 LPsi Sets the Properties of Psis: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.lpsi_sets_the_properties_of_psis_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    LpsiSetsThePropertiesOfPsisConfig,
    classify_lpsi_sets_the_properties_of_psis_component,
    lpsi_sets_the_properties_of_psis_labels,
    validate_lpsi_sets_the_properties_of_psis_fixture,
)


def test_lpsi_sets_the_properties_of_psis_fixture_preserves_source_boundary() -> None:
    result = validate_lpsi_sets_the_properties_of_psis_fixture()
    assert result.source_ledger_span == ("P0R01771", "P0R01778")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R01779"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_lpsi_sets_the_properties_of_psis_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R01771"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R01778"


def test_lpsi_sets_the_properties_of_psis_classification_and_labels_are_explicit() -> None:
    for component in (
        "lpsi_sets_the_properties_of_psis",
        "a_stable_vacuum_for_a_stable_interaction",
        "the_intrinsic_dynamics_of_the_psi_field_lpsi",
    ):
        assert (
            classify_lpsi_sets_the_properties_of_psis_component(component)
            == f"{component}_source_boundary"
        )
    labels = lpsi_sets_the_properties_of_psis_labels()
    assert labels["section"] == "LPsi Sets the Properties of Psis:"
    assert labels["next_boundary"] == "P0R01779"


def test_lpsi_sets_the_properties_of_psis_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        LpsiSetsThePropertiesOfPsisConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        LpsiSetsThePropertiesOfPsisConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01779"):
        LpsiSetsThePropertiesOfPsisConfig(next_source_boundary="P0R01778")
    with pytest.raises(ValueError, match="unknown lpsi_sets_the_properties_of_psis component"):
        classify_lpsi_sets_the_properties_of_psis_component("empirical_validation_claim")
