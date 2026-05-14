# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Layer 5 Active Inference fixture tests
"""Tests for Paper 0 Layer 5 Active Inference simulator fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.l5_active_inference_validation import (
    L5ActiveInferenceConfig,
    attractor_boundary_score,
    fep_loop_consistency_score,
    triple_network_precision_score,
    validate_l5_active_inference_fixture,
)


def test_attractor_boundary_score_requires_topological_components() -> None:
    complete = attractor_boundary_score(
        basin_coverage=0.88,
        loop_consistency=0.84,
        saddle_transition_support=0.82,
    )
    missing_saddle = attractor_boundary_score(
        basin_coverage=0.88,
        loop_consistency=0.84,
        saddle_transition_support=0.0,
    )

    assert complete > missing_saddle
    assert complete > L5ActiveInferenceConfig().attractor_threshold


def test_triple_network_precision_score_penalises_missing_salience_gate() -> None:
    complete = triple_network_precision_score(
        downward_prediction=0.86,
        upward_prediction_error=0.83,
        salience_precision_gate=0.81,
        cen_dmn_switching=0.79,
    )
    missing_gate = triple_network_precision_score(
        downward_prediction=0.86,
        upward_prediction_error=0.83,
        salience_precision_gate=0.0,
        cen_dmn_switching=0.79,
    )

    assert complete > missing_gate
    assert complete > L5ActiveInferenceConfig().triple_network_threshold


def test_fep_loop_consistency_requires_all_perception_action_terms() -> None:
    complete = fep_loop_consistency_score(
        free_energy_bound=True,
        prediction_error=True,
        action_policy=True,
        belief_update=True,
    )
    partial = fep_loop_consistency_score(
        free_energy_bound=True,
        prediction_error=True,
        action_policy=False,
        belief_update=False,
    )

    assert complete == pytest.approx(1.0)
    assert partial == pytest.approx(0.5)


def test_l5_active_inference_fixture_preserves_boundaries_and_controls() -> None:
    with pytest.raises(ValueError, match="attractor_threshold must be finite and positive"):
        L5ActiveInferenceConfig(attractor_threshold=0.0)
    with pytest.raises(ValueError, match="inputs must be in \\[0, 1\\]"):
        attractor_boundary_score(
            basin_coverage=1.1,
            loop_consistency=0.84,
            saddle_transition_support=0.82,
        )

    result = validate_l5_active_inference_fixture()

    assert result.spec_keys == (
        "l5_active_inference.attractor_geometry",
        "l5_active_inference.hpc_triple_network_loop",
        "l5_active_inference.fep_perception_action_loop",
        "l5_active_inference.cosmic_prior_boundary",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06434", "P0R06449")
    assert result.attractor_score > result.config_thresholds["attractor_threshold"]
    assert result.triple_network_score > result.config_thresholds["triple_network_threshold"]
    assert result.fep_loop_score == pytest.approx(1.0)
    assert result.null_controls["missing_saddle_boundary_rejection_label"] == 1.0
    assert result.null_controls["missing_salience_gate_rejection_label"] == 1.0
    assert result.null_controls["cosmic_prior_empirical_claim_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
