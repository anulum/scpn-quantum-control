# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Layer 5 four-stroke fixture tests
"""Tests for Paper 0 Layer 5 four-stroke engine fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.l5_four_stroke_validation import (
    L5FourStrokeConfig,
    l5_coherence_metric,
    policy_precision_weights,
    prediction_error,
    sleep_consolidation_sigma_update,
    validate_l5_four_stroke_fixture,
)


def test_policy_precision_weights_are_normalised_and_reward_sensitive() -> None:
    weights = policy_precision_weights(
        reward_predictions=np.array([0.1, 1.2, -0.2], dtype=np.float64),
        precision=3.0,
    )

    assert weights.sum() == pytest.approx(1.0)
    assert weights[1] > weights[0] > weights[2]


def test_prediction_error_and_coherence_metric_preserve_source_equation() -> None:
    residual = prediction_error(
        sensory_input=np.array([1.0, 0.5, -0.25], dtype=np.float64),
        prediction=np.array([0.75, 0.25, -0.5], dtype=np.float64),
    )
    coherent = l5_coherence_metric(
        theta_bg=np.array([0.2, 0.3, 0.4], dtype=np.float64),
        theta_cb=np.array([0.1, 0.15, 0.2], dtype=np.float64),
        theta_ctx=np.array([0.1, 0.15, 0.2], dtype=np.float64),
    )
    decohered = l5_coherence_metric(
        theta_bg=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        theta_cb=np.array([0.2, 0.3, 0.5], dtype=np.float64),
        theta_ctx=np.array([0.4, 0.1, 0.7], dtype=np.float64),
    )

    assert residual.tolist() == pytest.approx([0.25, 0.25, 0.25])
    assert coherent == pytest.approx(1.0)
    assert 0.0 <= decohered < coherent


def test_sleep_consolidation_moves_sigma_toward_criticality() -> None:
    updated = sleep_consolidation_sigma_update(
        sigma=1.4,
        homeostatic_gain=0.25,
        resetting_noise=0.0,
    )

    assert updated == pytest.approx(1.3)


def test_l5_four_stroke_fixture_preserves_boundaries_and_controls() -> None:
    with pytest.raises(ValueError, match="precision must be finite and positive"):
        policy_precision_weights(
            reward_predictions=np.array([0.1, 0.2], dtype=np.float64),
            precision=0.0,
        )
    with pytest.raises(ValueError, match="vectors must have the same shape"):
        prediction_error(
            sensory_input=np.array([1.0, 2.0], dtype=np.float64),
            prediction=np.array([1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="homeostatic_gain must be in \\[0, 1\\]"):
        L5FourStrokeConfig(homeostatic_gain=1.5)

    result = validate_l5_four_stroke_fixture()

    assert result.spec_keys == (
        "l5_four_stroke.engine_framing",
        "l5_four_stroke.policy_selection",
        "l5_four_stroke.prediction_generation",
        "l5_four_stroke.error_processing",
        "l5_four_stroke.model_consolidation",
        "l5_four_stroke.upde_coherence_prediction",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06582", "P0R06614")
    assert result.selected_policy_index == 1
    assert result.prediction_error_norm > 0.0
    assert 0.0 <= result.l5_coherence <= 1.0
    assert abs(result.post_sleep_sigma - 1.0) < abs(result.pre_sleep_sigma - 1.0)
    assert result.null_controls["shape_mismatch_rejection_label"] == 1.0
    assert result.null_controls["invalid_precision_rejection_label"] == 1.0
    assert result.null_controls["unsupported_tms_evidence_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
