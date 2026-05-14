# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 collective niche construction fixture tests
"""Tests for Paper 0 collective niche construction fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.collective_niche_construction_validation import (
    CollectiveNicheConstructionConfig,
    bidirectional_feedback_score,
    collective_predictability_gain,
    entrainment_coherence_score,
    shared_model_convergence_score,
    validate_collective_niche_construction_fixture,
)


def test_shared_model_convergence_score_requires_all_social_channels() -> None:
    complete = shared_model_convergence_score(
        beliefs=0.82,
        values=0.8,
        language=0.84,
        norms=0.78,
        communication=0.81,
        imitation=0.79,
        artefacts=0.83,
    )
    missing_artefacts = shared_model_convergence_score(
        beliefs=0.82,
        values=0.8,
        language=0.84,
        norms=0.78,
        communication=0.81,
        imitation=0.79,
        artefacts=0.0,
    )

    assert complete > missing_artefacts
    assert complete > CollectiveNicheConstructionConfig().convergence_threshold


def test_entrainment_coherence_score_tracks_institutional_channels() -> None:
    score = entrainment_coherence_score(
        institutions=0.8,
        rituals=0.78,
        language=0.84,
        art=0.76,
    )

    assert score > CollectiveNicheConstructionConfig().entrainment_threshold


def test_bidirectional_feedback_and_predictability_gain_are_finite() -> None:
    feedback = bidirectional_feedback_score(
        collective_to_environment=np.array([0.2, 0.4, 0.8], dtype=np.float64),
        environment_to_collective=np.array([0.1, 0.3, 0.7], dtype=np.float64),
    )
    gain = collective_predictability_gain(
        baseline_surprise=np.array([1.2, 1.0, 0.9], dtype=np.float64),
        modified_surprise=np.array([0.9, 0.7, 0.6], dtype=np.float64),
    )

    assert feedback > 0.98
    assert gain == pytest.approx(0.3)


def test_collective_niche_fixture_preserves_boundaries_and_controls() -> None:
    with pytest.raises(ValueError, match="convergence_threshold must be finite and positive"):
        CollectiveNicheConstructionConfig(convergence_threshold=0.0)
    with pytest.raises(ValueError, match="vectors must have the same shape"):
        bidirectional_feedback_score(
            collective_to_environment=np.array([0.2, 0.4], dtype=np.float64),
            environment_to_collective=np.array([0.1, 0.3, 0.7], dtype=np.float64),
        )

    result = validate_collective_niche_construction_fixture()

    assert result.spec_keys == (
        "collective_niche.shared_generative_model",
        "collective_niche.noosphere_entrainment",
        "collective_niche.biosphere_feedback_loop",
        "collective_niche.gaian_synchrony_boundary",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06519", "P0R06529")
    assert result.shared_model_score > result.config_thresholds["convergence_threshold"]
    assert result.entrainment_score > result.config_thresholds["entrainment_threshold"]
    assert result.predictability_gain > 0.0
    assert result.null_controls["missing_artefacts_rejection_label"] == 1.0
    assert result.null_controls["shape_mismatch_rejection_label"] == 1.0
    assert result.null_controls["unsupported_planetary_evidence_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
