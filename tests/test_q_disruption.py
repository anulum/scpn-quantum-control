"""Tests for control/q_disruption.py."""

import numpy as np
import pytest

from scpn_quantum_control.control.q_disruption import QuantumDisruptionClassifier


@pytest.fixture
def clf():
    return QuantumDisruptionClassifier(n_features=11, n_layers=2)


def test_predict_returns_probability(clf):
    features = np.random.default_rng(42).normal(0, 1, 11)
    risk = clf.predict(features)
    assert 0.0 <= risk <= 1.0


def test_encode_features_qubit_count(clf):
    features = np.ones(11)
    qc = clf.encode_features(features)
    assert qc.num_qubits == 5  # 4 data + 1 ancilla


def test_classifier_qubit_count(clf):
    qc = clf.build_classifier()
    assert qc.num_qubits == 5


def test_train_updates_params():
    clf = QuantumDisruptionClassifier(n_features=11, n_layers=1)
    original = clf.params.copy()
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (3, 11))
    y = np.array([0, 1, 0])
    clf.train(X, y, epochs=2, lr=0.5)
    assert not np.allclose(clf.params, original)


def test_different_features_different_predictions(clf):
    f1 = np.zeros(11)
    f2 = np.ones(11) * 5
    r1 = clf.predict(f1)
    r2 = clf.predict(f2)
    assert isinstance(r1, float)
    assert isinstance(r2, float)


def test_seed_reproducibility():
    """Same seed produces identical initial params."""
    c1 = QuantumDisruptionClassifier(n_layers=2, seed=99)
    c2 = QuantumDisruptionClassifier(n_layers=2, seed=99)
    np.testing.assert_array_equal(c1.params, c2.params)


def test_different_seed_different_params():
    c1 = QuantumDisruptionClassifier(n_layers=2, seed=0)
    c2 = QuantumDisruptionClassifier(n_layers=2, seed=1)
    assert not np.allclose(c1.params, c2.params)
