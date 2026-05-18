# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 I. The Brain's Protective Scaffold and Fluid Dynamics validation tests
"""Tests for Paper 0 I. The Brain's Protective Scaffold and Fluid Dynamics source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.i_the_brain_s_protective_scaffold_and_fluid_dynamics_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ITheBrainSProtectiveScaffoldAndFluidDynamicsConfig,
    classify_i_the_brain_s_protective_scaffold_and_fluid_dynamics_component,
    i_the_brain_s_protective_scaffold_and_fluid_dynamics_labels,
    validate_i_the_brain_s_protective_scaffold_and_fluid_dynamics_fixture,
)


def test_i_the_brain_s_protective_scaffold_and_fluid_dynamics_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_i_the_brain_s_protective_scaffold_and_fluid_dynamics_fixture()
    assert result.source_ledger_span == ("P0R04858", "P0R04870")
    assert result.source_record_count == 13
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04871"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_i_the_brain_s_protective_scaffold_and_fluid_dynamics_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04858"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04870"


def test_i_the_brain_s_protective_scaffold_and_fluid_dynamics_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "i_the_brain_s_protective_scaffold_and_fluid_dynamics",
        "1_the_meninges_and_cranial_vault_geometric_and_immune_boundaries_l3",
        "2_the_blood_brain_barrier_bbb_and_neurovascular_unit_nvu_l2_l3",
    ):
        assert (
            classify_i_the_brain_s_protective_scaffold_and_fluid_dynamics_component(component)
            == f"{component}_source_boundary"
        )
    labels = i_the_brain_s_protective_scaffold_and_fluid_dynamics_labels()
    assert labels["section"] == "I. The Brain's Protective Scaffold and Fluid Dynamics"
    assert labels["next_boundary"] == "P0R04871"


def test_i_the_brain_s_protective_scaffold_and_fluid_dynamics_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 13"):
        ITheBrainSProtectiveScaffoldAndFluidDynamicsConfig(expected_source_record_count=12)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        ITheBrainSProtectiveScaffoldAndFluidDynamicsConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04871"):
        ITheBrainSProtectiveScaffoldAndFluidDynamicsConfig(next_source_boundary="P0R04870")
    with pytest.raises(
        ValueError, match="unknown i_the_brain_s_protective_scaffold_and_fluid_dynamics component"
    ):
        classify_i_the_brain_s_protective_scaffold_and_fluid_dynamics_component(
            "empirical_validation_claim"
        )
