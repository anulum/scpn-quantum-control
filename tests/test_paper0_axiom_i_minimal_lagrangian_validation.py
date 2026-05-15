# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I minimal Lagrangian validation tests
"""Tests for Paper 0 Axiom I minimal Lagrangian source validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.axiom_i_minimal_lagrangian_validation import (
    AxiomIMinimalLagrangianConfig,
    axiom_i_minimal_lagrangian_labels,
    classify_lagrangian_operator,
    classify_minimal_lagrangian_requirement,
    validate_axiom_i_minimal_lagrangian_fixture,
)


def test_minimal_lagrangian_config_rejects_boundary_drift() -> None:
    with pytest.raises(ValueError, match="expected_minimal_criterion_count must equal 3"):
        AxiomIMinimalLagrangianConfig(expected_minimal_criterion_count=2)

    with pytest.raises(ValueError, match="expected_equation_record_count must equal 4"):
        AxiomIMinimalLagrangianConfig(expected_equation_record_count=3)

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00747"):
        AxiomIMinimalLagrangianConfig(next_source_boundary="P0R00748")


def test_minimal_lagrangian_classifiers_are_source_bounded() -> None:
    assert classify_minimal_lagrangian_requirement("spin0") == "single_irreducible_spin0_dof"
    assert classify_minimal_lagrangian_requirement("phase") == "intentional_phase_variable"
    assert classify_minimal_lagrangian_requirement("soliton") == (
        "stable_finite_energy_organismal_soliton"
    )
    assert classify_lagrangian_operator("kinetic") == "covariant_psi_kinetic_term"
    assert classify_lagrangian_operator("potential") == "quartic_ssb_potential"
    assert classify_lagrangian_operator("curvature") == "nonminimal_curvature_coupling"
    assert classify_lagrangian_operator("infoton") == "pulled_back_information_metric_dynamics"

    with pytest.raises(ValueError, match="unknown minimal Lagrangian requirement"):
        classify_minimal_lagrangian_requirement("spin2")
    with pytest.raises(ValueError, match="unknown Lagrangian operator"):
        classify_lagrangian_operator("dissipator")


def test_minimal_lagrangian_fixture_preserves_equation_boundary_and_null_controls() -> None:
    result = validate_axiom_i_minimal_lagrangian_fixture()

    assert result.source_ledger_span == ("P0R00733", "P0R00746")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.minimal_criterion_count == 3
    assert result.lagrangian_operator_count == 4
    assert result.equation_record_count == 4
    assert result.next_source_boundary == "P0R00747"
    assert result.null_controls == {
        "minimal_lagrangian_is_source_formula_not_empirical_fit": 1.0,
        "curvature_and_information_metric_terms_require_downstream_tests": 1.0,
        "ssb_boundedness_claim_requires_model_validation": 1.0,
    }
    assert (
        result.problem_metadata["protocol_state"] == "source_minimal_lagrangian_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00733"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00746"


def test_minimal_lagrangian_labels_name_criteria_satisfaction_boundary() -> None:
    labels = axiom_i_minimal_lagrangian_labels()

    assert (
        labels["section"]
        == "Model-Class Justification: From Axiom 1 to a Minimal Psi-Field Lagrangian"
    )
    assert labels["lagrangian"] == "L_min = |D_mu Psi|^2 - V(|Psi|) - 1/4 g_F F F - xi R |Psi|^2"
    assert labels["next_boundary"] == "Why this family satisfies (i)-(iii)"
