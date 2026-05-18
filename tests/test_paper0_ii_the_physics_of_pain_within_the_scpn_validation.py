# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. The Physics of Pain within the SCPN validation tests
"""Tests for Paper 0 II. The Physics of Pain within the SCPN source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ii_the_physics_of_pain_within_the_scpn_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IiThePhysicsOfPainWithinTheScpnConfig,
    classify_ii_the_physics_of_pain_within_the_scpn_component,
    ii_the_physics_of_pain_within_the_scpn_labels,
    validate_ii_the_physics_of_pain_within_the_scpn_fixture,
)


def test_ii_the_physics_of_pain_within_the_scpn_fixture_preserves_source_boundary() -> None:
    result = validate_ii_the_physics_of_pain_within_the_scpn_fixture()
    assert result.source_ledger_span == ("P0R05075", "P0R05082")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05083"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_ii_the_physics_of_pain_within_the_scpn_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05075"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05082"


def test_ii_the_physics_of_pain_within_the_scpn_classification_and_labels_are_explicit() -> None:
    for component in (
        "ii_the_physics_of_pain_within_the_scpn",
        "iii_intervention_intravenous_morphine_opioid_agonism",
        "1_l2_modulation_the_molecular_brake_and_iet_interface",
    ):
        assert (
            classify_ii_the_physics_of_pain_within_the_scpn_component(component)
            == f"{component}_source_boundary"
        )
    labels = ii_the_physics_of_pain_within_the_scpn_labels()
    assert labels["section"] == "II. The Physics of Pain within the SCPN"
    assert labels["next_boundary"] == "P0R05083"


def test_ii_the_physics_of_pain_within_the_scpn_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        IiThePhysicsOfPainWithinTheScpnConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        IiThePhysicsOfPainWithinTheScpnConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05083"):
        IiThePhysicsOfPainWithinTheScpnConfig(next_source_boundary="P0R05082")
    with pytest.raises(
        ValueError, match="unknown ii_the_physics_of_pain_within_the_scpn component"
    ):
        classify_ii_the_physics_of_pain_within_the_scpn_component("empirical_validation_claim")
