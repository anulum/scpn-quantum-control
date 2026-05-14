# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 glial slow-control fixture tests
"""Tests for Paper 0 glial-neuronal slow-control simulator fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.glial_slow_control_validation import (
    GlialSlowControlConfig,
    causal_decoupling_score,
    glial_feedback_stability_score,
    protocol_completeness_score,
    validate_glial_slow_control_fixture,
)


def test_glial_feedback_stability_score_penalises_missing_slow_feedback() -> None:
    complete = glial_feedback_stability_score(
        fast_loop_criticality=0.84,
        slow_ca_integration=0.82,
        gliotransmitter_feedback=0.8,
        excitability_control=0.78,
    )
    missing_slow = glial_feedback_stability_score(
        fast_loop_criticality=0.84,
        slow_ca_integration=0.0,
        gliotransmitter_feedback=0.8,
        excitability_control=0.78,
    )

    assert complete > missing_slow
    assert complete > GlialSlowControlConfig().stability_threshold


def test_protocol_completeness_requires_all_four_source_steps() -> None:
    complete = protocol_completeness_score(
        preparation=True,
        simultaneous_recording=True,
        avalanche_analysis=True,
        causal_block=True,
    )
    partial = protocol_completeness_score(
        preparation=True,
        simultaneous_recording=True,
        avalanche_analysis=False,
        causal_block=False,
    )

    assert complete == pytest.approx(1.0)
    assert partial == pytest.approx(0.5)


def test_causal_decoupling_score_requires_pre_post_intervention_change() -> None:
    score = causal_decoupling_score(
        baseline_correlation=0.72,
        post_block_correlation=0.18,
    )

    assert score > GlialSlowControlConfig().decoupling_threshold


def test_glial_slow_control_fixture_preserves_boundaries_and_controls() -> None:
    with pytest.raises(ValueError, match="stability_threshold must be finite and positive"):
        GlialSlowControlConfig(stability_threshold=0.0)
    with pytest.raises(ValueError, match="correlations must be in \\[-1, 1\\]"):
        causal_decoupling_score(baseline_correlation=1.2, post_block_correlation=0.0)

    result = validate_glial_slow_control_fixture()

    assert result.spec_keys == (
        "glial_slow_control.two_timescale_governor",
        "glial_slow_control.homeostatic_feedback_channels",
        "glial_slow_control.experimental_protocol_catalogue",
        "glial_slow_control.falsification_and_causal_decoupling",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06414", "P0R06433")
    assert result.stability_score > result.config_thresholds["stability_threshold"]
    assert result.protocol_completeness == pytest.approx(1.0)
    assert result.decoupling_score > result.config_thresholds["decoupling_threshold"]
    assert result.null_controls["missing_slow_feedback_rejection_label"] == 1.0
    assert result.null_controls["incomplete_protocol_rejection_label"] == 1.0
    assert result.null_controls["unsupported_empirical_evidence_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
