# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Electrodynamic Interface of Consciousness (CEMI and IIIEF) validation tests
"""Tests for Paper 0 The Electrodynamic Interface of Consciousness (CEMI and IIIEF) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_electrodynamic_interface_of_consciousness_cemi_and_iiief_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheElectrodynamicInterfaceOfConsciousnessCemiAndIiiefConfig,
    classify_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_component,
    the_electrodynamic_interface_of_consciousness_cemi_and_iiief_labels,
    validate_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_fixture,
)


def test_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_fixture()
    assert result.source_ledger_span == ("P0R05420", "P0R05429")
    assert result.source_record_count == 10
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R05430"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05420"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05429"


def test_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_electrodynamic_interface_of_consciousness_cemi_and_iiief",):
        assert (
            classify_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_electrodynamic_interface_of_consciousness_cemi_and_iiief_labels()
    assert labels["section"] == "The Electrodynamic Interface of Consciousness (CEMI and IIIEF)"
    assert labels["next_boundary"] == "P0R05430"


def test_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        TheElectrodynamicInterfaceOfConsciousnessCemiAndIiiefConfig(expected_source_record_count=9)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheElectrodynamicInterfaceOfConsciousnessCemiAndIiiefConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05430"):
        TheElectrodynamicInterfaceOfConsciousnessCemiAndIiiefConfig(
            next_source_boundary="P0R05429"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_electrodynamic_interface_of_consciousness_cemi_and_iiief component",
    ):
        classify_the_electrodynamic_interface_of_consciousness_cemi_and_iiief_component(
            "empirical_validation_claim"
        )
