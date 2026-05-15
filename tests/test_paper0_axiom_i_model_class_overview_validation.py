# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I model-class overview validation tests
"""Tests for source-accounting validation of Axiom I model-class overview records."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.axiom_i_model_class_overview_validation import (
    AxiomIModelClassOverviewConfig,
    axiom_i_model_class_overview_labels,
    classify_model_class_choice,
    classify_selection_criterion,
    validate_axiom_i_model_class_overview_fixture,
)


def test_model_class_overview_config_rejects_boundary_drift() -> None:
    with pytest.raises(ValueError, match="expected_selection_criterion_count must equal 3"):
        AxiomIModelClassOverviewConfig(expected_selection_criterion_count=2)

    with pytest.raises(ValueError, match="expected_blank_separator_count must equal 1"):
        AxiomIModelClassOverviewConfig(expected_blank_separator_count=0)

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00717"):
        AxiomIModelClassOverviewConfig(next_source_boundary="P0R00718")


def test_model_class_overview_classifiers_are_source_bounded() -> None:
    assert classify_selection_criterion("spin0") == "irreducible_spin0_degree_of_freedom"
    assert classify_selection_criterion("phase") == "intentional_phase_variable"
    assert classify_selection_criterion("soliton") == "stable_finite_energy_self_structure"
    assert classify_model_class_choice("complex_scalar") == "minimal_spin0_amplitude_phase_carrier"
    assert classify_model_class_choice("local_u1") == "local_gauge_phase_agency_via_infoton"
    assert classify_model_class_choice("ssb_potential") == "mexican_hat_stability_mechanism"
    assert classify_model_class_choice("rejected_alternatives") == (
        "real_global_vector_spinor_alternatives_rejected"
    )

    with pytest.raises(ValueError, match="unknown model-class criterion"):
        classify_selection_criterion("extra_dimension")
    with pytest.raises(ValueError, match="unknown model-class choice"):
        classify_model_class_choice("unbounded_potential")


def test_model_class_overview_fixture_preserves_predictions_and_null_controls() -> None:
    result = validate_axiom_i_model_class_overview_fixture()

    assert result.source_ledger_span == ("P0R00703", "P0R00716")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.selection_criterion_count == 3
    assert result.model_class_choice_count == 4
    assert result.blank_separator_count == 1
    assert result.next_source_boundary == "P0R00717"
    assert result.null_controls == {
        "model_class_overview_is_not_empirical_validation": 1.0,
        "rejected_alternatives_require_downstream_falsification": 1.0,
        "pedagogical_metaphors_are_not_model_terms": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_model_class_overview_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00703"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00716"


def test_model_class_overview_labels_name_meta_framework_boundary() -> None:
    labels = axiom_i_model_class_overview_labels()

    assert labels["section"] == "Model-Class Justification: From Axiom to Lagrangian"
    assert labels["selected_family"] == "complex scalar field with local U(1) and SSB"
    assert labels["next_boundary"] == "Meta-Framework Integrations"
