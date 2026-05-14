# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 ethical-imperative validation tests
"""Executable fixture tests for Paper 0 Ethical Imperative records."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ethical_imperative_validation import (
    EthicalImperativeConfig,
    civilisation_choice_label,
    feedback_tuning_score,
    governance_beyond_borders_score,
    validate_ethical_imperative_fixture,
    validate_governance_beyond_borders_fixture,
)


def test_civilisation_choice_labels_alignment_fragmentation_and_collapse() -> None:
    config = EthicalImperativeConfig()

    assert (
        civilisation_choice_label(
            coherence=0.84, fragmentation=0.18, collapse_entropy=0.22, config=config
        )
        == "alignment_global_coherence"
    )
    assert (
        civilisation_choice_label(
            coherence=0.43, fragmentation=0.78, collapse_entropy=0.41, config=config
        )
        == "fragmentation_societal_spin_glass"
    )
    assert (
        civilisation_choice_label(
            coherence=0.18, fragmentation=0.31, collapse_entropy=0.86, config=config
        )
        == "collapse_entropy_death"
    )


def test_governance_beyond_borders_requires_all_three_protocols() -> None:
    config = EthicalImperativeConfig()

    result = validate_governance_beyond_borders_fixture(config)

    assert governance_beyond_borders_score(config) > config.governance_threshold
    assert result.governance_score > config.governance_threshold
    assert result.null_controls["missing_entropy_budget_rejection_label"] == pytest.approx(1.0)
    assert result.null_controls[
        "missing_global_coherence_metric_rejection_label"
    ] == pytest.approx(1.0)
    assert result.null_controls["missing_recursive_review_rejection_label"] == pytest.approx(1.0)
    assert "not empirical evidence" in result.claim_boundary


def test_feedback_tuning_score_prefers_tuned_recursive_loop() -> None:
    config = EthicalImperativeConfig()

    tuned = feedback_tuning_score(
        loop_gain=0.72, damping=0.68, layer16_closure=0.81, config=config
    )
    untuned = feedback_tuning_score(
        loop_gain=0.95, damping=0.12, layer16_closure=0.18, config=config
    )

    assert tuned > untuned
    assert tuned > config.feedback_threshold
    assert untuned < config.feedback_threshold


def test_ethical_imperative_config_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        EthicalImperativeConfig(governance_threshold=0.0)

    with pytest.raises(ValueError, match="finite and non-negative"):
        EthicalImperativeConfig(entropy_budget_weight=-0.1)

    with pytest.raises(ValueError, match="threshold ordering"):
        EthicalImperativeConfig(alignment_threshold=0.2, collapse_entropy_threshold=0.1)


def test_ethical_imperative_default_fixture_wires_restatement_boundary() -> None:
    result = validate_ethical_imperative_fixture()

    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.spec_keys == (
        "ethical_imperative.ethics_physics_restatement",
        "ethical_imperative.civilisation_choice_phase_boundary",
        "ethical_imperative.consciousness_engineering_call_boundary",
        "ethical_imperative.governance_beyond_borders_protocol",
        "ethical_imperative.feedback_loop_tuning_boundary",
    )
    assert result.choice_labels == (
        "alignment_global_coherence",
        "fragmentation_societal_spin_glass",
        "collapse_entropy_death",
    )
    assert result.governance.governance_score > result.config_thresholds["governance_threshold"]
    assert result.feedback_loop_delta > 0.0
    assert result.overlap_with_prior_slice == "P0R06251-P0R06272"
    assert "not empirical evidence" in result.claim_boundary
