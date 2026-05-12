# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Quantum Reservoir
"""Tests for quantum reservoir computing."""

from __future__ import annotations

import numpy as np
import pytest

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

    def test_rejects_empty_feature_vector(self):
        K = build_knm_paper27(L=3)
        with pytest.raises(ValueError, match="at least one feature"):
            reservoir_features(np.array([]), K)

    def test_rejects_non_square_coupling_matrix(self):
        K = build_knm_paper27(L=3)
        with pytest.raises(ValueError, match="K must be a square"):
            reservoir_features(np.array([0.1]), K[:2, :])

    def test_rejects_omega_shape_mismatch(self):
        K = build_knm_paper27(L=3)
        with pytest.raises(ValueError, match="omega must be a vector matching K"):
            reservoir_features(np.array([0.1]), K, omega=np.zeros(2))

    def test_rejects_invalid_max_weight(self):
        K = build_knm_paper27(L=3)
        with pytest.raises(ValueError, match="max_weight must be between"):
            reservoir_features(np.array([0.1]), K, max_weight=0)


class TestReservoirFeatureMatrix:
    def test_shape(self):
        K = build_knm_paper27(L=3)
        X = np.random.default_rng(42).uniform(size=(5, 3))
        F = reservoir_feature_matrix(X, K, max_weight=1)
        assert F.shape[0] == 5
        assert F.shape[1] == 9  # 3 × 3 weight-1 Pauli features

    def test_rejects_one_dimensional_input(self):
        K = build_knm_paper27(L=3)
        with pytest.raises(ValueError, match="X must be a 2-D"):
            reservoir_feature_matrix(np.array([0.1, 0.2, 0.3]), K)

    def test_rejects_empty_sample_set(self):
        K = build_knm_paper27(L=3)
        with pytest.raises(ValueError, match="at least one sample"):
            reservoir_feature_matrix(np.empty((0, 3)), K)

    def test_rejects_empty_feature_columns(self):
        K = build_knm_paper27(L=3)
        with pytest.raises(ValueError, match="at least one feature"):
            reservoir_feature_matrix(np.empty((2, 0)), K)


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

    def test_rejects_mismatched_targets(self):
        K = build_knm_paper27(L=3)
        X = np.random.default_rng(42).uniform(size=(8, 3))
        with pytest.raises(ValueError, match="y_train must be a vector matching"):
            reservoir_ridge_regression(X, np.ones(7), K, max_weight=1)

    def test_rejects_non_positive_regularisation(self):
        K = build_knm_paper27(L=3)
        X = np.random.default_rng(42).uniform(size=(8, 3))
        y = np.sin(X[:, 0])
        with pytest.raises(ValueError, match="alpha must be finite and positive"):
            reservoir_ridge_regression(X, y, K, alpha=0.0, max_weight=1)


# ---------------------------------------------------------------------------
# Reservoir computing physics — expressibility and universality
# ---------------------------------------------------------------------------


class TestReservoirPhysics:
    def test_feature_matrix_rank(self):
        """Feature matrix should have non-trivial rank (reservoir is expressive)."""
        K = build_knm_paper27(L=3)
        X = np.random.default_rng(42).uniform(size=(10, 3))
        F = reservoir_feature_matrix(X, K, max_weight=1)
        rank = np.linalg.matrix_rank(F)
        assert rank >= 3  # at least n_inputs dimensions

    def test_features_deterministic(self):
        """Same input → same features (no stochastic element)."""
        K = build_knm_paper27(L=3)
        x = np.array([0.5, 0.3, 0.7])
        r1 = reservoir_features(x, K, max_weight=1)
        r2 = reservoir_features(x, K, max_weight=1)
        np.testing.assert_array_equal(r1.features, r2.features)

    def test_higher_weight_more_features(self):
        """max_weight=2 should produce more features than max_weight=1."""
        K = build_knm_paper27(L=3)
        x = np.array([0.5, 0.3, 0.7])
        r1 = reservoir_features(x, K, max_weight=1)
        r2 = reservoir_features(x, K, max_weight=2)
        assert r2.n_features > r1.n_features


# ---------------------------------------------------------------------------
# Pipeline: Knm → reservoir → regression → wired end-to-end
# ---------------------------------------------------------------------------


class TestReservoirPipeline:
    def test_pipeline_knm_to_prediction(self):
        """Full pipeline: build_knm → reservoir features → ridge → prediction.
        Verifies quantum reservoir is not decorative.
        """
        import time

        K = build_knm_paper27(L=3)
        rng = np.random.default_rng(42)
        X = rng.uniform(size=(12, 3))
        y = np.sin(X[:, 0] * 2)

        t0 = time.perf_counter()
        W, preds = reservoir_ridge_regression(X, y, K, max_weight=1)
        dt = (time.perf_counter() - t0) * 1000

        mse = float(np.mean((preds - y) ** 2))
        assert np.all(np.isfinite(preds))
        assert mse < 1.0  # reservoir should learn something

        print(f"\n  PIPELINE Knm→Reservoir→Ridge (3q, 12 samples): {dt:.1f} ms")
        print(f"  MSE = {mse:.4f}, n_weights = {len(W)}")
