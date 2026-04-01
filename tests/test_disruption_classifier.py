# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Disruption Classifier
"""Tests for quantum disruption classifier."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.applications.disruption_classifier import (
    DisruptionClassifierResult,
    generate_synthetic_disruption_data,
    run_disruption_benchmark,
    train_disruption_classifier,
)


class TestSyntheticData:
    def test_shape(self):
        X, y = generate_synthetic_disruption_data(n_samples=20)
        assert X.shape == (20, 5)
        assert y.shape == (20,)

    def test_labels_binary(self):
        _, y = generate_synthetic_disruption_data(n_samples=30)
        assert set(np.unique(y)).issubset({0, 1})

    def test_has_both_classes(self):
        _, y = generate_synthetic_disruption_data(n_samples=30)
        assert 0 in y and 1 in y


class TestTrainClassifier:
    def test_returns_weights(self):
        X, y = generate_synthetic_disruption_data(n_samples=10)
        weights, K = train_disruption_classifier(X, y, n_qubits=3)
        assert len(weights) == 10
        assert K.shape == (10, 10)


class TestRunBenchmark:
    def test_returns_result(self):
        result = run_disruption_benchmark(n_train=8, n_test=4, n_qubits=3)
        assert isinstance(result, DisruptionClassifierResult)

    def test_accuracy_bounded(self):
        result = run_disruption_benchmark(n_train=8, n_test=4, n_qubits=3)
        assert 0 <= result.accuracy <= 1.0

    def test_predictions_shape(self):
        result = run_disruption_benchmark(n_train=8, n_test=4, n_qubits=3)
        assert len(result.predictions) == 4
        assert len(result.labels) == 4

    def test_predictions_binary(self):
        result = run_disruption_benchmark(n_train=8, n_test=4, n_qubits=3)
        assert set(np.unique(result.predictions)).issubset({0, 1})

    def test_scpn_disruption(self):
        """Record disruption classifier benchmark."""
        result = run_disruption_benchmark(n_train=15, n_test=10, n_qubits=3)
        print("\n  Disruption classifier (synthetic):")
        print(f"  Accuracy: {result.accuracy:.2%}")
        print(f"  Train/test: {result.n_train}/{result.n_test}")
        print(f"  Qubits: {result.kernel_n_qubits}")
        assert isinstance(result.accuracy, float)


# ---------------------------------------------------------------------------
# Classifier physics: kernel Gram matrix properties
# ---------------------------------------------------------------------------


class TestClassifierPhysics:
    def test_kernel_matrix_symmetric(self):
        X, y = generate_synthetic_disruption_data(n_samples=6)
        _, K = train_disruption_classifier(X, y, n_qubits=3)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_kernel_matrix_psd(self):
        X, y = generate_synthetic_disruption_data(n_samples=6)
        _, K = train_disruption_classifier(X, y, n_qubits=3)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-6)


# ---------------------------------------------------------------------------
# Pipeline: data → kernel → classifier → prediction → wired
# ---------------------------------------------------------------------------


class TestClassifierPipeline:
    def test_pipeline_end_to_end(self):
        """Full pipeline: synthetic data → quantum kernel → SVM → predictions.
        Verifies disruption classifier is wired end-to-end.
        """
        import time

        t0 = time.perf_counter()
        result = run_disruption_benchmark(n_train=10, n_test=5, n_qubits=3)
        dt = (time.perf_counter() - t0) * 1000

        assert 0 <= result.accuracy <= 1.0
        assert len(result.predictions) == 5

        print(f"\n  PIPELINE DisruptionBenchmark (3q, 10+5): {dt:.1f} ms")
        print(f"  Accuracy = {result.accuracy:.2%}")
