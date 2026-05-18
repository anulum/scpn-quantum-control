# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality validation tests
"""Tests for Paper 0 The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliConfig,
    classify_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_component,
    the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_labels,
    validate_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_fixture,
)


def test_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_fixture()
    )
    assert result.source_ledger_span == ("P0R05366", "P0R05379")
    assert result.source_record_count == 14
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R05380"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05366"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05379"


def test_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",):
        assert (
            classify_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_labels()
    assert (
        labels["section"]
        == "The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality"
    )
    assert labels["next_boundary"] == "P0R05380"


def test_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 14"):
        TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliConfig(
            expected_source_record_count=13
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05380"):
        TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliConfig(
            next_source_boundary="P0R05379"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali component",
    ):
        classify_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_component(
            "empirical_validation_claim"
        )
