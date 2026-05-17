# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 symmetry cascade validation tests
"""Tests for Paper 0 symmetry-cascade validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.symmetry_cascade_validation import (
    SymmetryCascadeConfig,
    classify_symmetry_cascade_component,
    symmetry_cascade_labels,
    validate_symmetry_cascade_fixture,
)


def test_symmetry_cascade_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 15"):
        SymmetryCascadeConfig(expected_source_record_count=14)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        SymmetryCascadeConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01597"):
        SymmetryCascadeConfig(next_source_boundary="P0R01596")


def test_symmetry_cascade_classifiers_are_source_bounded() -> None:
    assert (
        classify_symmetry_cascade_component("cascade_opening")
        == "source_field_symmetry_cascade_opening_boundary"
    )
    assert (
        classify_symmetry_cascade_component("three_breaks_architecture")
        == "three_breaks_architecture_claim_boundary"
    )
    assert (
        classify_symmetry_cascade_component("psi_field_potential_stability")
        == "psi_field_potential_stable_vacuum_boundary"
    )
    assert (
        classify_symmetry_cascade_component("world_interface_summary")
        == "geometric_informational_world_interface_summary_boundary"
    )
    with pytest.raises(ValueError, match="unknown symmetry-cascade component"):
        classify_symmetry_cascade_component("infoton_prediction")


def test_symmetry_cascade_fixture_preserves_claim_boundary() -> None:
    result = validate_symmetry_cascade_fixture()

    assert result.source_ledger_span == ("P0R01582", "P0R01596")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 15
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R01597"
    assert result.null_controls == {
        "balanced_pen_analogy_is_not_physical_derivation": 1.0,
        "prime_directive_law_selection_remains_source_claim_not_validation": 1.0,
        "direct_new_force_claim_rejected_for_world_interface_summary": 1.0,
        "stable_vacuum_language_is_not_measured_potential": 1.0,
    }
    assert (
        result.problem_metadata["protocol_state"] == "source_symmetry_cascade_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01582"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01596"


def test_symmetry_cascade_labels_name_next_particle_boundary() -> None:
    labels = symmetry_cascade_labels()

    assert labels["section"] == "How Reality Gets Its Structure: A Cascade of Broken Symmetries"
    assert labels["three_breaks"] == "laws, selves, actualisation"
    assert labels["potential"] == "Mexican-hat stable valley"
    assert labels["interfaces"] == "geometric and informational interfaces"
    assert labels["next_boundary"] == "Predicted Particles: The Infoton and the Psi-Higgs Boson"
