# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom III Ntilde invariance-law validation tests
"""Tests for Paper 0 Axiom III Ntilde-invariance-law validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.axiom_iii_ntilde_invariance_law_validation import (
    AxiomIIINtildeInvarianceLawConfig,
    axiom_iii_ntilde_invariance_law_labels,
    classify_ntilde_invariance_law_component,
    validate_axiom_iii_ntilde_invariance_law_fixture,
)


def test_ntilde_invariance_law_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        AxiomIIINtildeInvarianceLawConfig(expected_source_record_count=10)

    with pytest.raises(ValueError, match="expected_variable_definition_count must equal 3"):
        AxiomIIINtildeInvarianceLawConfig(expected_variable_definition_count=2)

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00811"):
        AxiomIIINtildeInvarianceLawConfig(next_source_boundary="P0R00810")


def test_ntilde_invariance_law_classifiers_are_source_bounded() -> None:
    assert (
        classify_ntilde_invariance_law_component("physical_law_identification")
        == "teleological_drive_identified_with_dimensionless_ntilde_invariant"
    )
    assert (
        classify_ntilde_invariance_law_component("invariant_ratio_equation")
        == "ntilde_power_over_reversible_information_processing_cost"
    )
    assert (
        classify_ntilde_invariance_law_component("variable_definitions")
        == "power_information_rate_and_reversible_cost_per_bit"
    )
    assert (
        classify_ntilde_invariance_law_component("unity_threshold_limit")
        == "ntilde_unity_reversible_efficiency_limit"
    )

    with pytest.raises(ValueError, match="unknown Ntilde-invariance-law component"):
        classify_ntilde_invariance_law_component("sec_equivalence")


def test_ntilde_invariance_law_fixture_preserves_claim_boundary_and_null_controls() -> None:
    result = validate_axiom_iii_ntilde_invariance_law_fixture()

    assert result.source_ledger_span == ("P0R00800", "P0R00810")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 11
    assert result.invariant_definition_count == 3
    assert result.variable_definition_count == 3
    assert result.threshold_equation_count == 1
    assert result.reversible_limit_count == 1
    assert result.next_source_boundary == "P0R00811"
    assert result.null_controls == {
        "ntilde_invariance_law_is_source_claim_not_empirical_evidence": 1.0,
        "petrasek_2025_reference_requires_bibliographic_trace": 1.0,
        "unity_threshold_requires_downstream_operational_validation": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_axiom_iii_ntilde_invariance_law_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00800"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00810"


def test_ntilde_invariance_law_labels_name_sec_equivalence_boundary() -> None:
    labels = axiom_iii_ntilde_invariance_law_labels()

    assert labels["section"] == "Formal Physical Definition: The tilde_N_t Invariance Law"
    assert labels["invariant"] == (
        "tilde_N_t = P / (epsilon_b dot_I) = (E/t) / ((Delta F_rev / Delta I) dot_I)"
    )
    assert labels["threshold"] == "tilde_N_t -> 1"
    assert labels["next_boundary"] == "Equivalence of SEC and the tilde_N_t = 1 State"
