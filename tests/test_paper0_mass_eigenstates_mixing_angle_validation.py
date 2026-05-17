# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 mass eigenstates mixing-angle validation tests
"""Tests for Paper 0 mass-eigenstates mixing-angle validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.mass_eigenstates_mixing_angle_validation import (
    MassEigenstatesMixingAngleConfig,
    classify_mass_eigenstates_mixing_angle_component,
    mass_eigenstates_mixing_angle_labels,
    validate_mass_eigenstates_mixing_angle_fixture,
)


def test_mass_eigenstates_mixing_angle_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 15"):
        MassEigenstatesMixingAngleConfig(expected_source_record_count=14)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        MassEigenstatesMixingAngleConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01684"):
        MassEigenstatesMixingAngleConfig(next_source_boundary="P0R01683")


def test_mass_eigenstates_mixing_angle_classifiers_are_source_bounded() -> None:
    assert (
        classify_mass_eigenstates_mixing_angle_component("mass_eigenstate_rotation")
        == "orthogonal_mass_eigenstate_rotation_boundary"
    )
    assert (
        classify_mass_eigenstates_mixing_angle_component("lhc_invisible_decay_bound")
        == "lhc_invisible_higgs_branching_bound_boundary"
    )
    assert (
        classify_mass_eigenstates_mixing_angle_component("perturbative_target_boundary")
        == "perturbative_lambda_mix_search_target_boundary"
    )
    with pytest.raises(ValueError, match="unknown mass-eigenstates mixing-angle component"):
        classify_mass_eigenstates_mixing_angle_component("lhc_phenomenology")


def test_mass_eigenstates_mixing_angle_fixture_preserves_claim_boundary() -> None:
    result = validate_mass_eigenstates_mixing_angle_fixture()

    assert result.source_ledger_span == ("P0R01669", "P0R01683")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 15
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R01684"
    assert result.null_controls == {
        "rotation_formalism_is_not_measured_higgs_mixing": 1.0,
        "lhc_invisible_bound_is_constraint_not_psi_sector_detection": 1.0,
        "working_bound_is_not_model_confirmation": 1.0,
    }
    assert (
        result.problem_metadata["protocol_state"]
        == "source_mass_eigenstates_mixing_angle_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01669"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01683"


def test_mass_eigenstates_mixing_angle_labels_name_next_lhc_strategy_boundary() -> None:
    labels = mass_eigenstates_mixing_angle_labels()

    assert labels["section"] == "Mass Eigenstates and the Mixing Angle (theta)"
    assert labels["rotation"] == "[h_SM, h_Psi]^T = R(theta) [h_bare, h_Psi,bare]^T"
    assert labels["working_bound"] == "sin theta lesssim 0.31"
    assert labels["next_boundary"] == "Phenomenology and Search Strategies at the LHC"
