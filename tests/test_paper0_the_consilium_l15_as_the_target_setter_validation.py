# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Consilium (L15) as the Target Setter: validation tests
"""Tests for Paper 0 The Consilium (L15) as the Target Setter: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_consilium_l15_as_the_target_setter_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheConsiliumL15AsTheTargetSetterConfig,
    classify_the_consilium_l15_as_the_target_setter_component,
    the_consilium_l15_as_the_target_setter_labels,
    validate_the_consilium_l15_as_the_target_setter_fixture,
)


def test_the_consilium_l15_as_the_target_setter_fixture_preserves_source_boundary() -> None:
    result = validate_the_consilium_l15_as_the_target_setter_fixture()
    assert result.source_ledger_span == ("P0R04131", "P0R04150")
    assert result.source_record_count == 20
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04151"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_consilium_l15_as_the_target_setter_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04131"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04150"


def test_the_consilium_l15_as_the_target_setter_classification_and_labels_are_explicit() -> None:
    for component in (
        "the_consilium_l15_as_the_target_setter",
        "qualia_capacity_q_as_a_key_component_of_sigma_universe",
    ):
        assert (
            classify_the_consilium_l15_as_the_target_setter_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_consilium_l15_as_the_target_setter_labels()
    assert labels["section"] == "The Consilium (L15) as the Target Setter:"
    assert labels["next_boundary"] == "P0R04151"


def test_the_consilium_l15_as_the_target_setter_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 20"):
        TheConsiliumL15AsTheTargetSetterConfig(expected_source_record_count=19)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TheConsiliumL15AsTheTargetSetterConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04151"):
        TheConsiliumL15AsTheTargetSetterConfig(next_source_boundary="P0R04150")
    with pytest.raises(
        ValueError, match="unknown the_consilium_l15_as_the_target_setter component"
    ):
        classify_the_consilium_l15_as_the_target_setter_component("empirical_validation_claim")
