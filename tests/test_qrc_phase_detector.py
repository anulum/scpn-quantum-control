# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Qrc Phase Detector
"""Tests for QRC self-probing phase detector."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.qrc_phase_detector import (
    QRCPhaseResult,
    classify,
    generate_training_data,
    qrc_phase_detection,
    train_linear_readout,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16


def _ring_topology(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestGenerateTrainingData:
    def test_returns_correct_shapes(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        k_range = np.array([0.5, 1.0, 2.0, 3.0])
        X, y = generate_training_data(omega, T, k_range, k_threshold=1.5)
        assert X.shape[0] == 4
        assert X.shape[1] > 0
        assert y.shape == (4,)

    def test_labels_correct(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        k_range = np.array([0.5, 1.0, 2.0, 3.0])
        _, y = generate_training_data(omega, T, k_range, k_threshold=1.5)
        assert y[0] == 0.0  # 0.5 < 1.5
        assert y[1] == 0.0  # 1.0 < 1.5
        assert y[2] == 1.0  # 2.0 >= 1.5
        assert y[3] == 1.0  # 3.0 >= 1.5

    def test_features_vary_with_K(self):
        """Different K values should produce different features."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        k_range = np.array([0.5, 5.0])
        X, _ = generate_training_data(omega, T, k_range, k_threshold=2.0)
        assert not np.allclose(X[0], X[1])


class TestTrainLinearReadout:
    def test_weights_correct_shape(self):
        X = np.random.default_rng(42).standard_normal((10, 5))
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
        W = train_linear_readout(X, y, alpha=1.0)
        assert W.shape == (5,)

    def test_perfect_separation(self):
        """Linearly separable data with bias column should give perfect accuracy."""
        # Add bias column (ones) so ridge regression can learn intercept
        X = np.hstack(
            [
                np.vstack([np.ones((5, 3)), -np.ones((5, 3))]),
                np.ones((10, 1)),
            ]
        )
        y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
        W = train_linear_readout(X, y, alpha=0.01)
        preds = classify(X, W)
        assert np.all(preds == y)


class TestClassify:
    def test_threshold_at_half(self):
        X = np.array([[1.0], [-1.0]])
        W = np.array([1.0])
        preds = classify(X, W)
        assert preds[0] == 1.0
        assert preds[1] == 0.0


class TestQRCPhaseDetection:
    def test_returns_result(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        k_train = np.array([0.3, 0.8, 2.0, 4.0])
        k_test = np.array([0.5, 3.0])
        result = qrc_phase_detection(omega, T, k_train, k_test, k_threshold=1.5)
        assert isinstance(result, QRCPhaseResult)
        assert 0.0 <= result.accuracy <= 1.0
        assert result.n_features > 0

    def test_3qubit_pipeline(self):
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        k_train = np.linspace(0.5, 5.0, 6)
        k_test = np.array([1.0, 4.0])
        result = qrc_phase_detection(omega, T, k_train, k_test, k_threshold=2.5)
        assert isinstance(result, QRCPhaseResult)
        assert result.n_train == 6
        assert result.n_test == 2

    def test_well_separated_high_accuracy(self):
        """With clear separation, accuracy should be reasonable."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        # Train on clearly separated points
        k_train = np.array([0.1, 0.3, 0.5, 3.0, 4.0, 5.0])
        k_test = np.array([0.2, 4.5])
        result = qrc_phase_detection(omega, T, k_train, k_test, k_threshold=1.5)
        # Should get at least 50% (better than random)
        assert result.accuracy >= 0.5


# ---------------------------------------------------------------------------
# QRC physics: self-probing and feature expressiveness
# ---------------------------------------------------------------------------


class TestQRCPhysics:
    def test_features_finite(self):
        """All reservoir features must be finite."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        X, _ = generate_training_data(omega, T, np.array([1.0, 3.0]), k_threshold=2.0)
        assert np.all(np.isfinite(X))

    def test_readout_weights_finite(self):
        X = np.random.default_rng(42).standard_normal((8, 4))
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
        W = train_linear_readout(X, y, alpha=1.0)
        assert np.all(np.isfinite(W))


# ---------------------------------------------------------------------------
# Pipeline: topology → reservoir → readout → wired
# ---------------------------------------------------------------------------


class TestQRCPipeline:
    def test_pipeline_topology_to_phase_detection(self):
        """Full pipeline: ring topology → QRC features → linear readout → phase.
        Verifies QRC phase detector is wired end-to-end.
        """
        import time

        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        k_train = np.linspace(0.5, 5.0, 8)
        k_test = np.array([1.0, 4.0])

        t0 = time.perf_counter()
        result = qrc_phase_detection(omega, T, k_train, k_test, k_threshold=2.0)
        dt = (time.perf_counter() - t0) * 1000

        assert 0 <= result.accuracy <= 1.0
        assert result.n_features > 0

        print(f"\n  PIPELINE QRC phase detector (3q, 8 train, 2 test): {dt:.1f} ms")
        print(f"  Accuracy = {result.accuracy:.2%}, n_features = {result.n_features}")
