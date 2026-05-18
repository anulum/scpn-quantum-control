# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Mathematical Bridge: Deriving the UPDE from Free Energy Minimisation validation tests
"""Tests for Paper 0 The Mathematical Bridge: Deriving the UPDE from Free Energy Minimisation source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheMathematicalBridgeDerivingTheUpdeFromFreeEnergyMinimisationConfig,
    classify_the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_component,
    the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_labels,
    validate_the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_fixture,
)


def test_the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_fixture()
    )
    assert result.source_ledger_span == ("P0R06164", "P0R06178")
    assert result.source_record_count == 15
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R06179"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06164"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06178"


def test_the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",):
        assert (
            classify_the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_labels()
    assert (
        labels["section"]
        == "The Mathematical Bridge: Deriving the UPDE from Free Energy Minimisation"
    )
    assert labels["next_boundary"] == "P0R06179"


def test_the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 15"):
        TheMathematicalBridgeDerivingTheUpdeFromFreeEnergyMinimisationConfig(
            expected_source_record_count=14
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheMathematicalBridgeDerivingTheUpdeFromFreeEnergyMinimisationConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06179"):
        TheMathematicalBridgeDerivingTheUpdeFromFreeEnergyMinimisationConfig(
            next_source_boundary="P0R06178"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation component",
    ):
        classify_the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_component(
            "empirical_validation_claim"
        )
