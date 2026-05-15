# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom II FIM solution validation tests
"""Tests for Paper 0 Axiom II FIM-solution validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.axiom_ii_fim_solution_validation import (
    AxiomIIFIMSolutionConfig,
    axiom_ii_fim_solution_labels,
    classify_fim_solution_component,
    validate_axiom_ii_fim_solution_fixture,
)


def test_fim_solution_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 7"):
        AxiomIIFIMSolutionConfig(expected_source_record_count=6)

    with pytest.raises(ValueError, match="expected_physical_statement_count must equal 2"):
        AxiomIIFIMSolutionConfig(expected_physical_statement_count=1)

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00782"):
        AxiomIIFIMSolutionConfig(next_source_boundary="P0R00781")


def test_fim_solution_classifiers_are_source_bounded() -> None:
    assert (
        classify_fim_solution_component("metric_definition") == "fim_statistical_manifold_metric"
    )
    assert (
        classify_fim_solution_component("informational_interaction")
        == "infoton_propagates_through_information_geometry"
    )
    assert (
        classify_fim_solution_component("complexity_coupling")
        == "coupling_strength_tracks_informational_complexity"
    )
    assert (
        classify_fim_solution_component("fep_hpc_upde_synthesis")
        == "shared_information_geometry_language"
    )

    with pytest.raises(ValueError, match="unknown FIM-solution component"):
        classify_fim_solution_component("lagrangian_detail")


def test_fim_solution_fixture_preserves_claim_boundary_and_null_controls() -> None:
    result = validate_axiom_ii_fim_solution_fixture()

    assert result.source_ledger_span == ("P0R00775", "P0R00781")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 7
    assert result.metric_definition_count == 1
    assert result.physical_statement_count == 2
    assert result.synthesis_statement_count == 2
    assert result.next_source_boundary == "P0R00782"
    assert result.null_controls == {
        "fim_natural_metric_statement_is_source_claim_not_proof": 1.0,
        "complexity_coupling_requires_downstream_operational_metric": 1.0,
        "fep_hpc_upde_synthesis_is_not_empirical_validation": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_axiom_ii_fim_solution_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00775"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00781"


def test_fim_solution_labels_name_lagrangian_boundary() -> None:
    labels = axiom_ii_fim_solution_labels()

    assert labels["section"] == "The Fisher Information Metric (FIM) as the Solution"
    assert labels["metric"] == "natural unique Riemannian metric on a statistical manifold"
    assert labels["next_boundary"] == "Formal Consequence: The Informational Lagrangian"
