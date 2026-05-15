# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom II infoton geometry validation tests
"""Tests for Paper 0 Axiom II infoton-geometry validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.axiom_ii_infoton_geometry_validation import (
    AxiomIIInfotonGeometryConfig,
    axiom_ii_infoton_geometry_labels,
    classify_infoton_geometry_component,
    validate_axiom_ii_infoton_geometry_fixture,
)


def test_infoton_geometry_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 5"):
        AxiomIIInfotonGeometryConfig(expected_source_record_count=4)

    with pytest.raises(ValueError, match="expected_gauge_necessity_count must equal 1"):
        AxiomIIInfotonGeometryConfig(expected_gauge_necessity_count=0)

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00775"):
        AxiomIIInfotonGeometryConfig(next_source_boundary="P0R00774")


def test_infoton_geometry_classifiers_are_source_bounded() -> None:
    assert classify_infoton_geometry_component("problem_heading") == "infoton_geometry_problem"
    assert (
        classify_infoton_geometry_component("gauge_necessity")
        == "u1_local_complex_field_requires_spin1_infoton"
    )
    assert (
        classify_infoton_geometry_component("spacetime_baseline")
        == "standard_em_spacetime_metric_kinetic_term"
    )
    assert (
        classify_infoton_geometry_component("fim_dynamics") == "infoton_dynamics_governed_by_fim"
    )

    with pytest.raises(ValueError, match="unknown infoton-geometry component"):
        classify_infoton_geometry_component("fim_solution_detail")


def test_infoton_geometry_fixture_preserves_claim_boundary_and_null_controls() -> None:
    result = validate_axiom_ii_infoton_geometry_fixture()

    assert result.source_ledger_span == ("P0R00770", "P0R00774")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 5
    assert result.gauge_necessity_count == 1
    assert result.baseline_lagrangian_count == 1
    assert result.fim_claim_count == 1
    assert result.next_source_boundary == "P0R00775"
    assert result.null_controls == {
        "gauge_necessity_is_source_derivation_pointer_not_rederived_here": 1.0,
        "standard_em_lagrangian_is_baseline_not_scpn_result": 1.0,
        "fim_dynamics_claim_requires_downstream_validation": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_axiom_ii_infoton_geometry_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00770"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00774"


def test_infoton_geometry_labels_name_fim_solution_boundary() -> None:
    labels = axiom_ii_infoton_geometry_labels()

    assert labels["section"] == 'The Central Problem: The Geometry of the "Infoton"'
    assert labels["baseline_lagrangian"] == "L_EM = -1/4 F_mu_nu F^mu_nu"
    assert labels["next_boundary"] == "The Fisher Information Metric (FIM) as the Solution"
