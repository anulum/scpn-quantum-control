# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 gauge-principle derivation validation tests
"""Tests for Paper 0 gauge-principle derivation validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.gauge_principle_derivation_validation import (
    GaugePrincipleDerivationConfig,
    classify_gauge_principle_derivation_component,
    gauge_principle_derivation_labels,
    validate_gauge_principle_derivation_fixture,
)


def test_gauge_principle_derivation_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 60"):
        GaugePrincipleDerivationConfig(expected_source_record_count=59)
    with pytest.raises(ValueError, match="expected_blank_record_count must equal 1"):
        GaugePrincipleDerivationConfig(expected_blank_record_count=0)
    with pytest.raises(ValueError, match="expected_image_record_count must equal 1"):
        GaugePrincipleDerivationConfig(expected_image_record_count=0)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01078"):
        GaugePrincipleDerivationConfig(next_source_boundary="P0R01077")


def test_gauge_principle_derivation_classifiers_are_source_bounded() -> None:
    assert (
        classify_gauge_principle_derivation_component("derivation_boundary")
        == "gauge_principle_derivation_section_boundary"
    )
    assert (
        classify_gauge_principle_derivation_component("phenomenology_symmetry_roadmap")
        == "phenomenological_lagrangian_critique_and_symmetry_roadmap"
    )
    assert (
        classify_gauge_principle_derivation_component("free_scalar_global_u1")
        == "complex_scalar_free_lagrangian_and_global_u1_symmetry"
    )
    assert (
        classify_gauge_principle_derivation_component("local_u1_derivative_failure")
        == "local_phase_promotion_and_ordinary_derivative_failure"
    )
    assert (
        classify_gauge_principle_derivation_component("covariant_derivative_minimal_coupling")
        == "covariant_derivative_gauge_transform_and_minimal_coupling"
    )
    assert (
        classify_gauge_principle_derivation_component("fim_gauge_dynamics")
        == "fim_metric_informational_gauge_dynamics_source_claim"
    )
    with pytest.raises(ValueError, match="unknown gauge-principle-derivation component"):
        classify_gauge_principle_derivation_component("lorentz_covariance_resolution")


def test_gauge_principle_derivation_fixture_preserves_claim_boundary_and_null_controls() -> None:
    result = validate_gauge_principle_derivation_fixture()

    assert result.source_ledger_span == ("P0R01018", "P0R01077")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 60
    assert result.blank_record_count == 1
    assert result.image_record_count == 1
    assert result.phenomenology_symmetry_record_count == 17
    assert result.free_scalar_record_count == 12
    assert result.local_u1_record_count == 7
    assert result.covariant_derivative_record_count == 14
    assert result.fim_dynamics_record_count == 10
    assert result.next_source_boundary == "P0R01078"
    assert result.null_controls == {
        "gauge_principle_derivation_is_source_claim_not_empirical_evidence": 1.0,
        "fim_metric_replacement_is_not_lorentz_safe_until_next_slice": 1.0,
        "blank_record_p0r01046_and_image_p0r01076_are_preserved": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_gauge_principle_derivation_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01018"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01077"


def test_gauge_principle_derivation_labels_name_lorentz_boundary() -> None:
    labels = gauge_principle_derivation_labels()

    assert labels["section"] == "A Gauge-Principle Derivation of the Psi-Field"
    assert labels["free_lagrangian"] == "L_Psi = (partial_mu Psi)* (partial^mu Psi) - V(|Psi|)"
    assert labels["local_phase"] == "Psi(x) -> Psi'(x) = exp(i alpha(x)) Psi(x)"
    assert labels["covariant_derivative"] == "D_mu = partial_mu - i g A_mu"
    assert labels["field_strength"] == "F_mu_nu = partial_mu A_nu - partial_nu A_mu"
    assert labels["next_boundary"] == "Formal Resolution of Lorentz Covariance"
