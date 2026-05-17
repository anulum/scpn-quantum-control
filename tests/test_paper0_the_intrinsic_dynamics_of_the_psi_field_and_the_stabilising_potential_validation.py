# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Intrinsic Dynamics of the Psi-Field and the Stabilising Potential validation tests
"""Tests for Paper 0 The Intrinsic Dynamics of the Psi-Field and the Stabilising Potential source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialConfig,
    classify_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_component,
    the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_labels,
    validate_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_fixture,
)


def test_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_fixture()
    )
    assert result.source_ledger_span == ("P0R01755", "P0R01762")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R01763"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R01755"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R01762"


def test_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential",
        "meta_framework_integrations",
    ):
        assert (
            classify_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_labels()
    assert (
        labels["section"]
        == "The Intrinsic Dynamics of the Psi-Field and the Stabilising Potential"
    )
    assert labels["next_boundary"] == "P0R01763"


def test_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01763"):
        TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialConfig(
            next_source_boundary="P0R01762"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential component",
    ):
        classify_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_component(
            "empirical_validation_claim"
        )
