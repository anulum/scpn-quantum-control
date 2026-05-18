# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Proposed Experimental Protocol: NV-Center Quantum Sensing of Neuronal Culture Complexity validation tests
"""Tests for Paper 0 Proposed Experimental Protocol: NV-Center Quantum Sensing of Neuronal Culture Complexity source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ProposedExperimentalProtocolNvCenterQuantumSensingOfNeuronalCulConfig,
    classify_proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul_component,
    proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul_labels,
    validate_proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul_fixture,
)


def test_proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul_fixture()
    )
    assert result.source_ledger_span == ("P0R05182", "P0R05190")
    assert result.source_record_count == 9
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05191"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05182"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05190"


def test_proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul",
        "apparatus",
    ):
        assert (
            classify_proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul_labels()
    assert (
        labels["section"]
        == "Proposed Experimental Protocol: NV-Center Quantum Sensing of Neuronal Culture Complexity"
    )
    assert labels["next_boundary"] == "P0R05191"


def test_proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        ProposedExperimentalProtocolNvCenterQuantumSensingOfNeuronalCulConfig(
            expected_source_record_count=8
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        ProposedExperimentalProtocolNvCenterQuantumSensingOfNeuronalCulConfig(
            expected_component_count=3
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05191"):
        ProposedExperimentalProtocolNvCenterQuantumSensingOfNeuronalCulConfig(
            next_source_boundary="P0R05190"
        )
    with pytest.raises(
        ValueError,
        match="unknown proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul component",
    ):
        classify_proposed_experimental_protocol_nv_center_quantum_sensing_of_neuronal_cul_component(
            "empirical_validation_claim"
        )
