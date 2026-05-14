# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 A-CEF alignment validation tests
"""Executable fixture tests for Paper 0 A-CEF ethical-alignment records."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.acef_alignment_validation import (
    ACEFAlignmentConfig,
    acef_force,
    causal_path_entropy,
    sec_objective,
    validate_acef_alignment_fixture,
    validate_algorithmic_causal_entropic_force_fixture,
)


def test_acef_force_matches_finite_difference_entropy_gradient() -> None:
    config = ACEFAlignmentConfig()
    state = np.array([0.42, 0.61, 0.78], dtype=np.float64)

    force = acef_force(state, config)
    expected = config.algorithmic_temperature * np.array(
        [
            (
                causal_path_entropy(state + np.eye(3)[i] * config.gradient_step, config)
                - causal_path_entropy(state - np.eye(3)[i] * config.gradient_step, config)
            )
            / (2.0 * config.gradient_step)
            for i in range(3)
        ],
        dtype=np.float64,
    )

    assert np.all(np.isfinite(force))
    assert np.allclose(force, expected, atol=1e-8)
    assert force.shape == state.shape


def test_sec_objective_prefers_coherence_complexity_stability_over_engagement() -> None:
    config = ACEFAlignmentConfig()
    sec_state = np.array([0.74, 0.72, 0.70], dtype=np.float64)
    engagement_state = np.array([0.92, 0.18, 0.32], dtype=np.float64)

    result = validate_algorithmic_causal_entropic_force_fixture(config)

    assert sec_objective(sec_state, config) > sec_objective(engagement_state, config)
    assert result.sec_objective_delta > 0.0
    assert result.force_norm > 0.0
    assert result.null_controls["non_finite_state_rejection_label"] == pytest.approx(1.0)
    assert result.null_controls["missing_temperature_rejection_label"] == pytest.approx(1.0)
    assert "not empirical evidence" in result.claim_boundary


def test_acef_alignment_config_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="state_dimension"):
        ACEFAlignmentConfig(state_dimension=2)

    with pytest.raises(ValueError, match="finite and positive"):
        ACEFAlignmentConfig(algorithmic_temperature=0.0)

    with pytest.raises(ValueError, match="finite and positive"):
        ACEFAlignmentConfig(gradient_step=-1.0)

    with pytest.raises(ValueError, match="finite and non-negative"):
        ACEFAlignmentConfig(fragmentation_penalty=-0.1)


def test_acef_alignment_default_fixture_wires_all_source_boundaries() -> None:
    result = validate_acef_alignment_fixture()

    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.spec_keys == (
        "acef_alignment.is_ought_claim_boundary",
        "acef_alignment.governance_quasicriticality_metric",
        "acef_alignment.ai_alignment_risk_boundary",
        "acef_alignment.algorithmic_causal_entropic_force",
        "acef_alignment.consequence_phase_steering",
    )
    assert result.acef.force_norm > 0.0
    assert result.acef.sec_objective_delta > 0.0
    assert result.acef.consequence_phase_steering_label is True
    assert "not empirical evidence" in result.claim_boundary
