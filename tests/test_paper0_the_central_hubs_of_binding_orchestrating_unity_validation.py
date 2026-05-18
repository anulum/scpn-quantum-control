# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Central Hubs of Binding: Orchestrating Unity validation tests
"""Tests for Paper 0 The Central Hubs of Binding: Orchestrating Unity source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_central_hubs_of_binding_orchestrating_unity_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheCentralHubsOfBindingOrchestratingUnityConfig,
    classify_the_central_hubs_of_binding_orchestrating_unity_component,
    the_central_hubs_of_binding_orchestrating_unity_labels,
    validate_the_central_hubs_of_binding_orchestrating_unity_fixture,
)


def test_the_central_hubs_of_binding_orchestrating_unity_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_central_hubs_of_binding_orchestrating_unity_fixture()
    assert result.source_ledger_span == ("P0R04598", "P0R04606")
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04607"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_central_hubs_of_binding_orchestrating_unity_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04598"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04606"


def test_the_central_hubs_of_binding_orchestrating_unity_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_central_hubs_of_binding_orchestrating_unity",
        "introduction_to_the_integrative_systems_the_embodied_brain",
        "v_examination_of_the_integrative_systems_the_embodied_brain",
    ):
        assert (
            classify_the_central_hubs_of_binding_orchestrating_unity_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_central_hubs_of_binding_orchestrating_unity_labels()
    assert labels["section"] == "The Central Hubs of Binding: Orchestrating Unity"
    assert labels["next_boundary"] == "P0R04607"


def test_the_central_hubs_of_binding_orchestrating_unity_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        TheCentralHubsOfBindingOrchestratingUnityConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        TheCentralHubsOfBindingOrchestratingUnityConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04607"):
        TheCentralHubsOfBindingOrchestratingUnityConfig(next_source_boundary="P0R04606")
    with pytest.raises(
        ValueError, match="unknown the_central_hubs_of_binding_orchestrating_unity component"
    ):
        classify_the_central_hubs_of_binding_orchestrating_unity_component(
            "empirical_validation_claim"
        )
