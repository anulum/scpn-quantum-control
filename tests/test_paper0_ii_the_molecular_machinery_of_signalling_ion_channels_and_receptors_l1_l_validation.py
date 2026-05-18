# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. The Molecular Machinery of Signalling: Ion Channels and Receptors (L1/L2) validation tests
"""Tests for Paper 0 II. The Molecular Machinery of Signalling: Ion Channels and Receptors (L1/L2) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LConfig,
    classify_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_component,
    ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_labels,
    validate_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_fixture,
)


def test_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_fixture()
    )
    assert result.source_ledger_span == ("P0R04769", "P0R04777")
    assert result.source_record_count == 9
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R04778"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04769"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04777"


def test_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l",
        "1_the_architecture_of_gating_and_iet",
        "2_quantum_effects_in_selectivity_and_binding_l1_l2",
        "3_qze_and_attentional_stabilisation",
    ):
        assert (
            classify_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_labels()
    assert (
        labels["section"]
        == "II. The Molecular Machinery of Signalling: Ion Channels and Receptors (L1/L2)"
    )
    assert labels["next_boundary"] == "P0R04778"


def test_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LConfig(
            expected_source_record_count=8
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LConfig(
            expected_component_count=5
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04778"):
        IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LConfig(
            next_source_boundary="P0R04777"
        )
    with pytest.raises(
        ValueError,
        match="unknown ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l component",
    ):
        classify_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_component(
            "empirical_validation_claim"
        )
