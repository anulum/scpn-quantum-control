# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The QEC Imperative and the Role of the Psi-Field validation tests
"""Tests for Paper 0 The QEC Imperative and the Role of the Psi-Field source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_qec_imperative_and_the_role_of_the_psi_field_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheQecImperativeAndTheRoleOfThePsiFieldConfig,
    classify_the_qec_imperative_and_the_role_of_the_psi_field_component,
    the_qec_imperative_and_the_role_of_the_psi_field_labels,
    validate_the_qec_imperative_and_the_role_of_the_psi_field_fixture,
)


def test_the_qec_imperative_and_the_role_of_the_psi_field_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_qec_imperative_and_the_role_of_the_psi_field_fixture()
    assert result.source_ledger_span == ("P0R03051", "P0R03058")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03059"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_qec_imperative_and_the_role_of_the_psi_field_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03051"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03058"


def test_the_qec_imperative_and_the_role_of_the_psi_field_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_qec_imperative_and_the_role_of_the_psi_field",
        "meta_framework_integrations",
    ):
        assert (
            classify_the_qec_imperative_and_the_role_of_the_psi_field_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_qec_imperative_and_the_role_of_the_psi_field_labels()
    assert labels["section"] == "The QEC Imperative and the Role of the Psi-Field"
    assert labels["next_boundary"] == "P0R03059"


def test_the_qec_imperative_and_the_role_of_the_psi_field_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        TheQecImperativeAndTheRoleOfThePsiFieldConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TheQecImperativeAndTheRoleOfThePsiFieldConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03059"):
        TheQecImperativeAndTheRoleOfThePsiFieldConfig(next_source_boundary="P0R03058")
    with pytest.raises(
        ValueError, match="unknown the_qec_imperative_and_the_role_of_the_psi_field component"
    ):
        classify_the_qec_imperative_and_the_role_of_the_psi_field_component(
            "empirical_validation_claim"
        )
