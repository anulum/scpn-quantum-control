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
import pytest

from scpn_quantum_control.applications.disruption_classifier import (
    DisruptionClassifierResult,
    generate_synthetic_disruption_data,
    predict_disruption,
    run_disruption_benchmark,
    train_disruption_classifier,
)


class TestSyntheticData:
    def test_synthetic_data_requires_explicit_opt_in(self):
        with pytest.raises(RuntimeError, match="allow_synthetic"):
            generate_synthetic_disruption_data(n_samples=20)

    def test_shape(self):
        X, y = generate_synthetic_disruption_data(n_samples=20, allow_synthetic=True)
        assert X.shape == (20, 5)
        assert y.shape == (20,)

    def test_labels_binary(self):
        _, y = generate_synthetic_disruption_data(n_samples=30, allow_synthetic=True)
        assert set(np.unique(y)).issubset({0, 1})

    def test_has_both_classes(self):
        _, y = generate_synthetic_disruption_data(n_samples=30, allow_synthetic=True)
        assert 0 in y and 1 in y

    def test_rejects_invalid_generation_contracts(self):
        with pytest.raises(ValueError, match="n_samples"):
            generate_synthetic_disruption_data(n_samples=0, allow_synthetic=True)
        with pytest.raises(ValueError, match="n_features"):
            generate_synthetic_disruption_data(n_features=4, allow_synthetic=True)
        with pytest.raises(ValueError, match="disruption_fraction"):
            generate_synthetic_disruption_data(disruption_fraction=1.0, allow_synthetic=True)


class TestTrainClassifier:
    def test_returns_weights(self):
        X, y = generate_synthetic_disruption_data(n_samples=10, allow_synthetic=True)
        weights, K = train_disruption_classifier(X, y, n_qubits=3)
        assert len(weights) == 10
        assert K.shape == (10, 10)

    def test_rejects_label_count_mismatch(self):
        X, y = generate_synthetic_disruption_data(n_samples=10, allow_synthetic=True)
        with pytest.raises(ValueError, match="y_train must match X_train sample count"):
            train_disruption_classifier(X, y[:-1], n_qubits=3)

    def test_rejects_non_binary_labels(self):
        X, y = generate_synthetic_disruption_data(n_samples=10, allow_synthetic=True)
        y[0] = 2
        with pytest.raises(ValueError, match="y_train labels must be binary"):
            train_disruption_classifier(X, y, n_qubits=3)

    def test_rejects_non_positive_regularisation(self):
        X, y = generate_synthetic_disruption_data(n_samples=10, allow_synthetic=True)
        with pytest.raises(ValueError, match="alpha must be positive"):
            train_disruption_classifier(X, y, n_qubits=3, alpha=0.0)

    def test_rejects_nonfinite_training_features(self):
        X, y = generate_synthetic_disruption_data(n_samples=10, allow_synthetic=True)
        X[0, 0] = np.inf
        with pytest.raises(ValueError, match="X_train must contain only finite"):
            train_disruption_classifier(X, y, n_qubits=3)

    def test_rejects_nonfinite_training_labels(self):
        X, y = generate_synthetic_disruption_data(n_samples=10, allow_synthetic=True)
        y = y.astype(float)
        y[0] = np.nan
        with pytest.raises(ValueError, match="y_train must contain only finite"):
            train_disruption_classifier(X, y, n_qubits=3)

    def test_rejects_non_matrix_training_features(self):
        with pytest.raises(ValueError, match="X_train must be a 2-D"):
            train_disruption_classifier(np.array([0.1, 0.2]), np.array([0, 1]), n_qubits=2)

    def test_rejects_empty_training_matrix_contracts(self):
        with pytest.raises(ValueError, match="at least one sample"):
            train_disruption_classifier(np.empty((0, 5)), np.empty((0,)), n_qubits=2)
        with pytest.raises(ValueError, match="at least one feature"):
            train_disruption_classifier(np.empty((2, 0)), np.array([0, 1]), n_qubits=2)

    def test_rejects_non_vector_labels(self):
        X, _ = generate_synthetic_disruption_data(n_samples=4, allow_synthetic=True)
        with pytest.raises(ValueError, match="1-D label"):
            train_disruption_classifier(X, np.array([[0, 1], [1, 0]]), n_qubits=2)


class TestPredictClassifier:
    def test_rejects_train_weight_count_mismatch(self):
        X, y = generate_synthetic_disruption_data(n_samples=10, allow_synthetic=True)
        weights, _ = train_disruption_classifier(X, y, n_qubits=3)
        with pytest.raises(ValueError, match="weights must match X_train sample count"):
            predict_disruption(X[:2], X, weights[:-1], n_qubits=3)

    def test_rejects_feature_dimension_mismatch(self):
        X, y = generate_synthetic_disruption_data(n_samples=10, allow_synthetic=True)
        weights, _ = train_disruption_classifier(X, y, n_qubits=3)
        with pytest.raises(ValueError, match="X_test feature dimension must match X_train"):
            predict_disruption(X[:2, :4], X, weights, n_qubits=3)

    def test_predictions_are_binary_for_measured_shape_contract(self):
        X, y = generate_synthetic_disruption_data(n_samples=12, allow_synthetic=True)
        weights, _ = train_disruption_classifier(X[:8], y[:8], n_qubits=3)

        predictions = predict_disruption(X[8:], X[:8], weights, n_qubits=3)

        assert predictions.shape == (4,)
        assert set(np.unique(predictions)).issubset({0, 1})

    def test_rejects_invalid_weight_vector_contracts(self):
        X, y = generate_synthetic_disruption_data(n_samples=10, allow_synthetic=True)
        weights, _ = train_disruption_classifier(X, y, n_qubits=3)
        with pytest.raises(ValueError, match="1-D vector"):
            predict_disruption(X[:2], X, weights.reshape(1, -1), n_qubits=3)
        bad_weights = weights.copy()
        bad_weights[0] = np.inf
        with pytest.raises(ValueError, match="weights must contain only finite"):
            predict_disruption(X[:2], X, bad_weights, n_qubits=3)


class TestRunBenchmark:
    def test_benchmark_requires_explicit_synthetic_opt_in(self):
        with pytest.raises(RuntimeError, match="allow_synthetic"):
            run_disruption_benchmark(n_train=8, n_test=4, n_qubits=3)

    def test_returns_result(self):
        result = run_disruption_benchmark(n_train=8, n_test=4, n_qubits=3, allow_synthetic=True)
        assert isinstance(result, DisruptionClassifierResult)

    def test_accuracy_bounded(self):
        result = run_disruption_benchmark(n_train=8, n_test=4, n_qubits=3, allow_synthetic=True)
        assert 0 <= result.accuracy <= 1.0

    def test_predictions_shape(self):
        result = run_disruption_benchmark(n_train=8, n_test=4, n_qubits=3, allow_synthetic=True)
        assert len(result.predictions) == 4
        assert len(result.labels) == 4

    def test_predictions_binary(self):
        result = run_disruption_benchmark(n_train=8, n_test=4, n_qubits=3, allow_synthetic=True)
        assert set(np.unique(result.predictions)).issubset({0, 1})

    def test_result_labels_synthetic_source_mode(self):
        result = run_disruption_benchmark(n_train=8, n_test=4, n_qubits=3, allow_synthetic=True)
        assert result.source_mode == "synthetic"
        assert result.publication_safe is False

    def test_scpn_disruption(self):
        """Record disruption classifier benchmark."""
        result = run_disruption_benchmark(n_train=15, n_test=10, n_qubits=3, allow_synthetic=True)
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
        X, y = generate_synthetic_disruption_data(n_samples=6, allow_synthetic=True)
        _, K = train_disruption_classifier(X, y, n_qubits=3)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_kernel_matrix_psd(self):
        X, y = generate_synthetic_disruption_data(n_samples=6, allow_synthetic=True)
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
        result = run_disruption_benchmark(n_train=10, n_test=5, n_qubits=3, allow_synthetic=True)
        dt = (time.perf_counter() - t0) * 1000

        assert 0 <= result.accuracy <= 1.0
        assert len(result.predictions) == 5

        print(f"\n  PIPELINE DisruptionBenchmark (3q, 10+5): {dt:.1f} ms")
        print(f"  Accuracy = {result.accuracy:.2%}")
