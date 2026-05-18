# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Slow Control Network: Glial Homeostasis and Neuronal Criticality validation tests
"""Tests for Paper 0 The Slow Control Network: Glial Homeostasis and Neuronal Criticality source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_slow_control_network_glial_homeostasis_and_neuronal_criticality_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheSlowControlNetworkGlialHomeostasisAndNeuronalCriticalityConfig,
    classify_the_slow_control_network_glial_homeostasis_and_neuronal_criticality_component,
    the_slow_control_network_glial_homeostasis_and_neuronal_criticality_labels,
    validate_the_slow_control_network_glial_homeostasis_and_neuronal_criticality_fixture,
)


def test_the_slow_control_network_glial_homeostasis_and_neuronal_criticality_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_slow_control_network_glial_homeostasis_and_neuronal_criticality_fixture()
    assert result.source_ledger_span == ("P0R05380", "P0R05389")
    assert result.source_record_count == 10
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R05390"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_slow_control_network_glial_homeostasis_and_neuronal_criticality_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05380"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05389"


def test_the_slow_control_network_glial_homeostasis_and_neuronal_criticality_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_slow_control_network_glial_homeostasis_and_neuronal_criticality",):
        assert (
            classify_the_slow_control_network_glial_homeostasis_and_neuronal_criticality_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_slow_control_network_glial_homeostasis_and_neuronal_criticality_labels()
    assert (
        labels["section"] == "The Slow Control Network: Glial Homeostasis and Neuronal Criticality"
    )
    assert labels["next_boundary"] == "P0R05390"


def test_the_slow_control_network_glial_homeostasis_and_neuronal_criticality_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        TheSlowControlNetworkGlialHomeostasisAndNeuronalCriticalityConfig(
            expected_source_record_count=9
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheSlowControlNetworkGlialHomeostasisAndNeuronalCriticalityConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05390"):
        TheSlowControlNetworkGlialHomeostasisAndNeuronalCriticalityConfig(
            next_source_boundary="P0R05389"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_slow_control_network_glial_homeostasis_and_neuronal_criticality component",
    ):
        classify_the_slow_control_network_glial_homeostasis_and_neuronal_criticality_component(
            "empirical_validation_claim"
        )
