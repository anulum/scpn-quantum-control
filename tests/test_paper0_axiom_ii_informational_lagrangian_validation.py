# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom II informational Lagrangian validation tests
"""Tests for Paper 0 Axiom II informational-Lagrangian validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.axiom_ii_informational_lagrangian_validation import (
    AxiomIIInformationalLagrangianConfig,
    axiom_ii_informational_lagrangian_labels,
    classify_informational_lagrangian_component,
    validate_axiom_ii_informational_lagrangian_fixture,
)


def test_informational_lagrangian_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        AxiomIIInformationalLagrangianConfig(expected_source_record_count=8)

    with pytest.raises(ValueError, match="expected_gauge_equation_count must equal 2"):
        AxiomIIInformationalLagrangianConfig(expected_gauge_equation_count=1)

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00791"):
        AxiomIIInformationalLagrangianConfig(next_source_boundary="P0R00790")


def test_informational_lagrangian_classifiers_are_source_bounded() -> None:
    assert (
        classify_informational_lagrangian_component("kinetic_term_modification")
        == "infoton_kinetic_term_uses_pulled_back_information_metric"
    )
    assert (
        classify_informational_lagrangian_component("standard_gauge_baseline")
        == "standard_gauge_lagrangian_spacetime_metric_baseline"
    )
    assert (
        classify_informational_lagrangian_component("scpn_gauge_lagrangian")
        == "scpn_gauge_lagrangian_information_metric_substitution"
    )
    assert (
        classify_informational_lagrangian_component("operational_pullback_protocol")
        == "chapter6_pullback_protocol_falsifiability_bridge"
    )

    with pytest.raises(ValueError, match="unknown informational-Lagrangian component"):
        classify_informational_lagrangian_component("axiom_iii")


def test_informational_lagrangian_fixture_preserves_claim_boundary_and_null_controls() -> None:
    result = validate_axiom_ii_informational_lagrangian_fixture()

    assert result.source_ledger_span == ("P0R00782", "P0R00790")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 9
    assert result.gauge_equation_count == 2
    assert result.standard_gauge_baseline_count == 3
    assert result.scpn_gauge_count == 3
    assert result.pullback_protocol_count == 1
    assert result.next_source_boundary == "P0R00791"
    assert result.null_controls == {
        "informational_lagrangian_is_source_formula_not_empirical_evidence": 1.0,
        "pulled_back_fim_requires_chapter6_operational_protocol": 1.0,
        "falsifiability_claim_requires_downstream_bridge_tests": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_axiom_ii_informational_lagrangian_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00782"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00790"


def test_informational_lagrangian_labels_name_axiom_iii_boundary() -> None:
    labels = axiom_ii_informational_lagrangian_labels()

    assert labels["section"] == "Formal Consequence: The Informational Lagrangian"
    assert labels["standard_lagrangian"] == (
        "L_gauge = -1/4 g^{mu alpha} g^{nu beta} F_{mu nu} F_{alpha beta}"
    )
    assert labels["scpn_lagrangian"] == (
        "L_gauge = -1/4 tilde_g_F^{mu alpha} tilde_g_F^{nu beta} F_{mu nu} F_{alpha beta}"
    )
    assert labels["next_boundary"] == "Axiom III: The Drive of Teleological Optimisation"
