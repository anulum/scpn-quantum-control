# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Psis Field as the Target-Setter: validation tests
"""Tests for Paper 0 The Psis Field as the Target-Setter: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_psis_field_as_the_target_setter_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ThePsisFieldAsTheTargetSetterConfig,
    classify_the_psis_field_as_the_target_setter_component,
    the_psis_field_as_the_target_setter_labels,
    validate_the_psis_field_as_the_target_setter_fixture,
)


def test_the_psis_field_as_the_target_setter_fixture_preserves_source_boundary() -> None:
    result = validate_the_psis_field_as_the_target_setter_fixture()
    assert result.source_ledger_span == ("P0R02904", "P0R02914")
    assert result.source_record_count == 11
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R02915"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_psis_field_as_the_target_setter_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02904"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02914"


def test_the_psis_field_as_the_target_setter_classification_and_labels_are_explicit() -> None:
    for component in (
        "the_psis_field_as_the_target_setter",
        "sigma_is_the_lyapunov_function_itself",
        "homeostatic_quasicritical_controller",
    ):
        assert (
            classify_the_psis_field_as_the_target_setter_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_psis_field_as_the_target_setter_labels()
    assert labels["section"] == "The Psis Field as the Target-Setter:"
    assert labels["next_boundary"] == "P0R02915"


def test_the_psis_field_as_the_target_setter_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        ThePsisFieldAsTheTargetSetterConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        ThePsisFieldAsTheTargetSetterConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02915"):
        ThePsisFieldAsTheTargetSetterConfig(next_source_boundary="P0R02914")
    with pytest.raises(ValueError, match="unknown the_psis_field_as_the_target_setter component"):
        classify_the_psis_field_as_the_target_setter_component("empirical_validation_claim")
