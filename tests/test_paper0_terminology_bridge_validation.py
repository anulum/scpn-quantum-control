# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 terminology bridge validation tests
"""Tests for source-accounting checks around the terminology bridge."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.terminology_bridge_validation import (
    TerminologyBridgeConfig,
    classify_mainstream_anchor,
    classify_pela_boundary,
    terminology_bridge_labels,
    validate_terminology_bridge_fixture,
)


def test_mainstream_anchor_classification_preserves_domain_mapping() -> None:
    assert classify_mainstream_anchor("psi_field") == "field_theory_section_of_fibre_bundle"
    assert classify_mainstream_anchor("upde") == "nonlinear_coupled_oscillator_model"
    assert classify_mainstream_anchor("geometric_qualia") == "topology_metric_state_manifold_tda"
    assert (
        classify_mainstream_anchor("pela") == "yang_mills_like_regulariser_not_literal_gauge_law"
    )

    with pytest.raises(ValueError, match="unknown terminology anchor"):
        classify_mainstream_anchor("unknown")


def test_pela_boundary_classification_rejects_literal_equivalence() -> None:
    assert classify_pela_boundary("role") == "supervisory_optimisation_prior"
    assert classify_pela_boundary("h_int_effect") == "sets_boundary_conditions_or_tunes_parameters"
    assert classify_pela_boundary("gauge_status") == "analogy_not_deductive_derivation"
    assert classify_pela_boundary("simulation_status") == "stress_testable_control_functional"

    with pytest.raises(ValueError, match="unknown PELA boundary"):
        classify_pela_boundary("force_term")


def test_terminology_bridge_fixture_is_source_bounded() -> None:
    result = validate_terminology_bridge_fixture()

    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_ledger_span == ("P0R00610", "P0R00634")
    assert result.mainstream_anchor_count == 4
    assert result.analogy_boundary_count == 2
    assert result.next_source_boundary == "P0R00635"
    assert result.null_controls["anchor_map_is_not_validation_evidence"] == 1.0
    assert result.null_controls["yang_mills_similarity_is_not_deductive_equivalence"] == 1.0
    assert result.null_controls["pela_does_not_add_force_term_to_h_int"] == 1.0

    labels = terminology_bridge_labels()
    assert labels["bridge"] == "Terminology Bridge"
    assert labels["predictive_coding"] == "precision-weighted priors"
    assert labels["h_int"] == "H_int = -lambda * Psi_s * sigma"
    assert result.as_dict()["problem_metadata"]["protocol_state"] == (
        "source_anchor_map_only_no_experiment"
    )


def test_terminology_bridge_config_rejects_wrong_source_counts() -> None:
    with pytest.raises(ValueError, match="expected_mainstream_anchor_count must equal 4"):
        TerminologyBridgeConfig(expected_mainstream_anchor_count=3)
    with pytest.raises(ValueError, match="expected_analogy_boundary_count must equal 2"):
        TerminologyBridgeConfig(expected_analogy_boundary_count=1)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00635"):
        TerminologyBridgeConfig(next_source_boundary="P0R00636")
