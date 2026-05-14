# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Layer 5 Active Inference math fixture tests
"""Tests for Paper 0 Layer 5 Active Inference mathematical fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.l5_active_inference_math_validation import (
    L5ActiveInferenceMathConfig,
    expected_free_energy,
    layer_free_energy_terms,
    message_passing_update,
    precision_weighted_update,
    validate_l5_active_inference_math_fixture,
)


def test_layer_free_energy_terms_match_kl_minus_expected_log_likelihood() -> None:
    terms = layer_free_energy_terms(
        q_psi=np.array([0.2, 0.5, 0.3]),
        p_psi=np.array([0.25, 0.45, 0.3]),
        p_o_given_psi=np.array([0.7, 0.4, 0.9]),
    )

    assert terms.total_free_energy == pytest.approx(terms.complexity_kl + terms.accuracy_loss)
    assert terms.complexity_kl >= 0.0
    assert terms.accuracy_loss >= 0.0


def test_message_passing_update_preserves_upward_downward_delta_equations() -> None:
    update = message_passing_update(
        observation=np.array([1.2, 0.8]),
        generated_prediction=np.array([1.0, 0.9]),
        local_mu=np.array([0.4, 0.6]),
        parent_prediction=np.array([0.3, 0.7]),
        kappa=0.25,
    )

    np.testing.assert_allclose(update.upward_error, np.array([0.2, -0.1]))
    np.testing.assert_allclose(update.downward_error, np.array([0.1, -0.1]))
    np.testing.assert_allclose(update.delta_mu, np.array([-0.075, 0.05]))


def test_expected_free_energy_and_argmin_policy_are_source_bounded() -> None:
    scores = expected_free_energy(
        ambiguity=np.array([0.4, 0.3, 0.2]),
        divergence_from_prior=np.array([0.2, 0.1, 0.5]),
    )

    assert scores.expected_free_energy.tolist() == pytest.approx([0.6, 0.4, 0.7])
    assert scores.selected_policy_index == 1


def test_precision_weighted_update_uses_source_inverse_precision_formula() -> None:
    precision = np.diag([2.0, 4.0])
    update = precision_weighted_update(
        precision_matrix=precision,
        prediction_error=np.array([0.5, 1.0]),
    )

    np.testing.assert_allclose(update.delta_mu, np.array([0.25, 0.25]))
    assert update.source_formula_consistency_warning is True


def test_l5_active_inference_math_fixture_preserves_boundaries_and_controls() -> None:
    with pytest.raises(ValueError, match="kappa must be finite and positive"):
        L5ActiveInferenceMathConfig(kappa=0.0)
    with pytest.raises(ValueError, match="probability vectors must have the same shape"):
        layer_free_energy_terms(
            q_psi=np.array([0.5, 0.5]),
            p_psi=np.array([0.5, 0.5, 0.0]),
            p_o_given_psi=np.array([0.8, 0.7]),
        )

    result = validate_l5_active_inference_math_fixture()

    assert result.spec_keys == (
        "l5_active_inference_math.generative_hierarchy",
        "l5_active_inference_math.layer_free_energy",
        "l5_active_inference_math.message_passing_update",
        "l5_active_inference_math.action_and_precision_control",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06450", "P0R06484")
    assert result.free_energy_residual == pytest.approx(0.0)
    assert result.selected_policy_index == 1
    assert result.null_controls["shape_mismatch_rejection_label"] == 1.0
    assert result.null_controls["non_positive_likelihood_rejection_label"] == 1.0
    assert result.null_controls["singular_precision_rejection_label"] == 1.0
    assert result.null_controls["source_precision_wording_warning_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
