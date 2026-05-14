# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Layer 5 TDA/neurophenomenology fixture tests
"""Tests for Paper 0 Layer 5 TDA/neurophenomenology fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.l5_tda_neurophenomenology_validation import (
    L5TDANeurophenomenologyConfig,
    geometric_qualia_score,
    persistence_lifetimes,
    protocol_completeness_score,
    qualia_topology_correlation,
    validate_l5_tda_neurophenomenology_fixture,
)


def test_persistence_lifetimes_reject_invalid_birth_death_pairs() -> None:
    lifetimes = persistence_lifetimes(
        persistence_pairs=np.array([[0.0, 0.5], [0.2, 0.9]], dtype=np.float64)
    )

    np.testing.assert_allclose(lifetimes, np.array([0.5, 0.7]))
    with pytest.raises(ValueError, match="death must be greater than or equal to birth"):
        persistence_lifetimes(persistence_pairs=np.array([[0.5, 0.2]], dtype=np.float64))


def test_geometric_qualia_score_uses_volume_times_betti_sum() -> None:
    score = geometric_qualia_score(
        manifold_volume=2.0,
        betti_numbers=np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )

    assert score == pytest.approx(12.0)


def test_qualia_topology_correlation_tracks_report_topology_alignment() -> None:
    correlation = qualia_topology_correlation(
        report_scores=np.array([0.2, 0.5, 0.9], dtype=np.float64),
        topology_scores=np.array([1.0, 2.0, 4.0], dtype=np.float64),
    )

    assert correlation > 0.98


def test_protocol_completeness_requires_all_source_steps() -> None:
    complete = protocol_completeness_score(
        high_density_recording=True,
        immediate_interview=True,
        report_scoring=True,
        tda_features=True,
        correlation_test=True,
    )
    partial = protocol_completeness_score(
        high_density_recording=True,
        immediate_interview=True,
        report_scoring=False,
        tda_features=True,
        correlation_test=False,
    )

    assert complete == pytest.approx(1.0)
    assert partial == pytest.approx(0.6)


def test_l5_tda_neurophenomenology_fixture_preserves_boundaries_and_controls() -> None:
    with pytest.raises(ValueError, match="correlation_threshold must be finite and positive"):
        L5TDANeurophenomenologyConfig(correlation_threshold=0.0)
    with pytest.raises(ValueError, match="manifold_volume must be finite and positive"):
        geometric_qualia_score(
            manifold_volume=0.0,
            betti_numbers=np.array([1.0, 2.0], dtype=np.float64),
        )

    result = validate_l5_tda_neurophenomenology_fixture()

    assert result.spec_keys == (
        "l5_tda_neurophenomenology.geometric_qualia_hypothesis",
        "l5_tda_neurophenomenology.neurophenomenology_protocol",
        "l5_tda_neurophenomenology.persistent_homology_features",
        "l5_tda_neurophenomenology.qualia_richness_regression",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06504", "P0R06518")
    assert result.protocol_completeness == pytest.approx(1.0)
    assert result.correlation_score > result.config_thresholds["correlation_threshold"]
    assert result.null_controls["incomplete_protocol_rejection_label"] == 1.0
    assert result.null_controls["constant_report_rejection_label"] == 1.0
    assert result.null_controls["unsupported_empirical_qualia_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
