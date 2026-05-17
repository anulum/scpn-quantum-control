# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 geometric coupling consistency validation tests
"""Tests for Paper 0 geometric-coupling consistency validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.geometric_coupling_consistency_validation import (
    GeometricCouplingConsistencyConfig,
    classify_geometric_coupling_consistency_component,
    geometric_coupling_consistency_labels,
    validate_geometric_coupling_consistency_fixture,
)


def test_geometric_coupling_consistency_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 54"):
        GeometricCouplingConsistencyConfig(expected_source_record_count=53)
    with pytest.raises(ValueError, match="expected_component_count must equal 6"):
        GeometricCouplingConsistencyConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="expected_math_id_count must equal 3"):
        GeometricCouplingConsistencyConfig(expected_math_id_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01189"):
        GeometricCouplingConsistencyConfig(next_source_boundary="P0R01188")


def test_geometric_coupling_consistency_classifiers_are_source_bounded() -> None:
    assert (
        classify_geometric_coupling_consistency_component("coupling_problem_boundary")
        == "internal_gauge_symmetry_not_spacetime_curvature_coupling"
    )
    assert (
        classify_geometric_coupling_consistency_component("minimal_curved_spacetime_coupling")
        == "minimal_scalar_curved_spacetime_coupling_and_limitation"
    )
    assert (
        classify_geometric_coupling_consistency_component("non_minimal_consistency_condition")
        == "conformal_and_renormalizability_non_minimal_coupling_argument"
    )
    assert (
        classify_geometric_coupling_consistency_component("derived_geometric_lagrangian")
        == "derived_scalar_curvature_interaction_source_equation"
    )
    assert (
        classify_geometric_coupling_consistency_component("complete_covariant_action")
        == "generally_covariant_gauge_invariant_total_action_boundary"
    )
    assert (
        classify_geometric_coupling_consistency_component("interpretation_prediction_comments")
        == "interpretation_infoton_prediction_and_derivation_comment_boundary"
    )
    with pytest.raises(ValueError, match="unknown geometric-coupling consistency component"):
        classify_geometric_coupling_consistency_component("foundational_strengths")


def test_geometric_coupling_consistency_fixture_preserves_claim_boundary_and_null_controls() -> (
    None
):
    result = validate_geometric_coupling_consistency_fixture()

    assert result.source_ledger_span == ("P0R01135", "P0R01188")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 54
    assert result.component_count == 6
    assert result.math_ids == ("EQ0010", "EQ0011", "EQ0012")
    assert result.image_ids == ("IMG0020",)
    assert result.table_ids == ("TBL002",)
    assert result.next_source_boundary == "P0R01189"
    assert result.null_controls == {
        "geometric_coupling_consistency_is_source_derivation_not_empirical_evidence": 1.0,
        "minimal_coupling_alone_does_not_satisfy_direct_curvature_coupling": 1.0,
        "infoton_prediction_is_not_detector_evidence": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_geometric_coupling_consistency_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01135"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01188"


def test_geometric_coupling_consistency_labels_name_foundational_strengths_boundary() -> None:
    labels = geometric_coupling_consistency_labels()

    assert labels["section"] == "Consistency Conditions and the Origin of Geometric Coupling"
    assert labels["minimal_coupling"] == "L_Psi_curved uses g_mu_nu and covariant derivatives"
    assert labels["non_minimal_coupling"] == "L_non_minimal = - xi R Psi^* Psi"
    assert labels["derived_geometric_term"] == "L_Geometric_prime = - g_PsiG R Psi^* Psi"
    assert labels["next_boundary"] == "Foundational Strengths of the SCPN Lagrangian"
