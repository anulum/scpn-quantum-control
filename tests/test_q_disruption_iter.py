# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for control/q_disruption_iter.py."""

import numpy as np

from scpn_quantum_control.control.q_disruption_iter import (
    DisruptionBenchmark,
    ITERFeatureSpec,
    generate_synthetic_iter_data,
    normalize_iter_features,
)


def test_iter_feature_spec_defaults():
    spec = ITERFeatureSpec()
    assert len(spec.names) == 11
    assert len(spec.mins) == 11
    assert len(spec.maxs) == 11
    assert all(mn < mx for mn, mx in zip(spec.mins, spec.maxs))


def test_normalize_clamps_to_unit():
    spec = ITERFeatureSpec()
    raw = spec.maxs + 100
    normed = normalize_iter_features(raw, spec)
    assert np.all(normed <= 1.0)
    assert np.all(normed >= 0.0)


def test_normalize_preserves_center():
    spec = ITERFeatureSpec()
    mid = (spec.mins + spec.maxs) / 2
    normed = normalize_iter_features(mid, spec)
    np.testing.assert_allclose(normed, 0.5, atol=1e-10)


def test_generate_synthetic_shapes():
    X, y = generate_synthetic_iter_data(200, disruption_fraction=0.3, rng=np.random.default_rng(0))
    assert X.shape == (200, 11)
    assert y.shape == (200,)
    assert set(np.unique(y)) == {0.0, 1.0}


def test_generate_disruption_fraction():
    X, y = generate_synthetic_iter_data(100, disruption_fraction=0.4, rng=np.random.default_rng(0))
    assert int(np.sum(y)) == 40


def test_generate_normalized_range():
    X, y = generate_synthetic_iter_data(500, rng=np.random.default_rng(0))
    assert np.all(X >= 0.0)
    assert np.all(X <= 1.0)


def test_generate_deterministic():
    X1, y1 = generate_synthetic_iter_data(50, rng=np.random.default_rng(42))
    X2, y2 = generate_synthetic_iter_data(50, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_benchmark_init():
    bench = DisruptionBenchmark(n_train=20, n_test=10, seed=0)
    assert bench.X_train.shape == (20, 11)
    assert bench.X_test.shape == (10, 11)


def test_benchmark_run_returns_accuracy():
    bench = DisruptionBenchmark(n_train=10, n_test=5, seed=0)
    result = bench.run(epochs=1, lr=0.01)
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert len(result["predictions"]) == 5


def test_benchmark_classifier_predicts():
    bench = DisruptionBenchmark(n_train=10, n_test=5, seed=0)
    pred = bench.classifier.predict(bench.X_test[0])
    assert 0.0 <= pred <= 1.0
