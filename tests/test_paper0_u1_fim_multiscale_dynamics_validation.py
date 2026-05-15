# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 U1 FIM multiscale dynamics fixtures
"""Tests for Paper 0 U(1)/FIM and multiscale-dynamics fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.u1_fim_multiscale_dynamics_validation import (
    U1FIMMultiscaleDynamicsConfig,
    classify_upde_component,
    classify_validation_boundary,
    covariant_derivative_formula,
    validate_u1_fim_multiscale_dynamics_fixture,
)


def test_upde_component_classifier_preserves_core_terms() -> None:
    assert classify_upde_component("intrinsic_dynamics") == "omega_i_layer_timescale"
    assert classify_upde_component("intra_layer_coupling") == "kij_layer_synchronisation"
    assert classify_upde_component("inter_layer_coupling") == "cross_layer_causal_flow"

    with pytest.raises(ValueError, match="unknown UPDE component"):
        classify_upde_component("missing_noise_term")


def test_validation_boundary_classifier_separates_speculative_claims() -> None:
    assert classify_validation_boundary("fim_geometry") == "information_geometry_claim"
    assert classify_validation_boundary("ms_qec") == "requires_quantitative_biophysical_validation"
    assert (
        classify_validation_boundary("sfh_analogue")
        == "conceptual_convergence_not_independent_proof"
    )

    with pytest.raises(ValueError, match="unknown validation boundary"):
        classify_validation_boundary("unbounded_confirmation")


def test_covariant_derivative_formula_preserves_source_expression() -> None:
    assert covariant_derivative_formula() == "D_mu = partial_mu - i g A_mu"


def test_u1_fim_multiscale_fixture_preserves_scope_counts_and_boundaries() -> None:
    result = validate_u1_fim_multiscale_dynamics_fixture()

    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_ledger_span == ("P0R00506", "P0R00544")
    assert result.blank_separator_count == 1
    assert result.upde_component_count == 3
    assert result.validation_boundary_count == 3
    assert result.next_source_boundary == "P0R00545"
    assert result.null_controls["informational_lagrangian_is_not_empirical_validation"] == 1.0
    assert result.null_controls["sentience_field_convergence_is_not_independent_proof"] == 1.0

    with pytest.raises(ValueError, match="expected_blank_separator_count must equal 1"):
        U1FIMMultiscaleDynamicsConfig(expected_blank_separator_count=2)
