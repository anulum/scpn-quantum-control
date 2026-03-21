# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for quantum reservoir computing."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.applications.quantum_reservoir import (
    ReservoirResult,
    reservoir_feature_matrix,
    reservoir_features,
    reservoir_ridge_regression,
)
from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27


class TestReservoirFeatures:
    def test_returns_result(self):
        K = build_knm_paper27(L=3)
        x = np.array([0.5, 0.3, 0.7])
        result = reservoir_features(x, K)
        assert isinstance(result, ReservoirResult)

    def test_feature_count(self):
        K = build_knm_paper27(L=3)
        x = np.array([0.5, 0.3, 0.7])
        result = reservoir_features(x, K, max_weight=1)
        # Weight-1: 3 paulis × 3 positions = 9
        assert result.n_features == 9

    def test_features_bounded(self):
        K = build_knm_paper27(L=3)
        x = np.array([0.5, 0.3, 0.7])
        result = reservoir_features(x, K, max_weight=1)
        for f in result.features:
            assert abs(f) <= 1.0 + 1e-6

    def test_different_inputs_different_features(self):
        K = build_knm_paper27(L=3)
        r1 = reservoir_features(np.array([0.1, 0.1, 0.1]), K, max_weight=1)
        r2 = reservoir_features(np.array([0.9, 0.9, 0.9]), K, max_weight=1)
        assert not np.allclose(r1.features, r2.features)

    def test_labels_count_matches(self):
        K = build_knm_paper27(L=3)
        x = np.array([0.5, 0.3, 0.7])
        result = reservoir_features(x, K, max_weight=2)
        assert len(result.feature_labels) == result.n_features


class TestReservoirFeatureMatrix:
    def test_shape(self):
        K = build_knm_paper27(L=3)
        X = np.random.default_rng(42).uniform(size=(5, 3))
        F = reservoir_feature_matrix(X, K, max_weight=1)
        assert F.shape[0] == 5
        assert F.shape[1] == 9  # 3 × 3 weight-1 Pauli features


class TestReservoirRidgeRegression:
    def test_returns_weights_and_preds(self):
        K = build_knm_paper27(L=3)
        X = np.random.default_rng(42).uniform(size=(8, 3))
        y = np.sin(X[:, 0])
        W, preds = reservoir_ridge_regression(X, y, K, max_weight=1)
        assert len(W) > 0
        assert preds.shape == y.shape

    def test_predictions_finite(self):
        K = build_knm_paper27(L=3)
        X = np.random.default_rng(42).uniform(size=(8, 3))
        y = np.sin(X[:, 0])
        _, preds = reservoir_ridge_regression(X, y, K, max_weight=1)
        assert np.all(np.isfinite(preds))
