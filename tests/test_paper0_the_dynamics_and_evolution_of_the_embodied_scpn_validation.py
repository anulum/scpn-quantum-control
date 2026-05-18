# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Dynamics and Evolution of the Embodied SCPN validation tests
"""Tests for Paper 0 The Dynamics and Evolution of the Embodied SCPN source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_dynamics_and_evolution_of_the_embodied_scpn_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheDynamicsAndEvolutionOfTheEmbodiedScpnConfig,
    classify_the_dynamics_and_evolution_of_the_embodied_scpn_component,
    the_dynamics_and_evolution_of_the_embodied_scpn_labels,
    validate_the_dynamics_and_evolution_of_the_embodied_scpn_fixture,
)


def test_the_dynamics_and_evolution_of_the_embodied_scpn_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_dynamics_and_evolution_of_the_embodied_scpn_fixture()
    assert result.source_ledger_span == ("P0R04956", "P0R04966")
    assert result.source_record_count == 11
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04967"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_dynamics_and_evolution_of_the_embodied_scpn_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04956"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04966"


def test_the_dynamics_and_evolution_of_the_embodied_scpn_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_dynamics_and_evolution_of_the_embodied_scpn",
        "i_the_dynamics_of_consciousness_in_the_brain_body_system",
        "1_the_embodied_upde_the_symphony_of_the_self",
    ):
        assert (
            classify_the_dynamics_and_evolution_of_the_embodied_scpn_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_dynamics_and_evolution_of_the_embodied_scpn_labels()
    assert labels["section"] == "The Dynamics and Evolution of the Embodied SCPN"
    assert labels["next_boundary"] == "P0R04967"


def test_the_dynamics_and_evolution_of_the_embodied_scpn_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        TheDynamicsAndEvolutionOfTheEmbodiedScpnConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        TheDynamicsAndEvolutionOfTheEmbodiedScpnConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04967"):
        TheDynamicsAndEvolutionOfTheEmbodiedScpnConfig(next_source_boundary="P0R04966")
    with pytest.raises(
        ValueError, match="unknown the_dynamics_and_evolution_of_the_embodied_scpn component"
    ):
        classify_the_dynamics_and_evolution_of_the_embodied_scpn_component(
            "empirical_validation_claim"
        )
