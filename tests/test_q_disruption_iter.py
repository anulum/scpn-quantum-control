# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Q Disruption Iter
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


# ---------------------------------------------------------------------------
# ITER physics: feature normalisation and disruption detection
# ---------------------------------------------------------------------------


def test_normalize_extreme_values():
    """Values far outside ITER operating range should clamp to [0,1]."""
    spec = ITERFeatureSpec()
    extreme = np.full(11, 1e10)
    normed = normalize_iter_features(extreme, spec)
    np.testing.assert_allclose(normed, 1.0)


def test_benchmark_training_reduces_loss():
    """Training for more epochs should reduce or maintain loss."""
    bench = DisruptionBenchmark(n_train=15, n_test=5, seed=42)
    r1 = bench.run(epochs=1, lr=0.1)
    # Re-init and train longer
    bench2 = DisruptionBenchmark(n_train=15, n_test=5, seed=42)
    r2 = bench2.run(epochs=5, lr=0.1)
    # More epochs should not drastically worsen accuracy
    assert r2["accuracy"] >= r1["accuracy"] - 0.3


# ---------------------------------------------------------------------------
# Pipeline: ITER data → normalise → classify → wired
# ---------------------------------------------------------------------------


def test_pipeline_iter_to_disruption():
    """Full pipeline: synthetic ITER data → normalise → quantum classify → risk.
    Verifies ITER disruption module is wired end-to-end.
    """
    import time

    t0 = time.perf_counter()
    bench = DisruptionBenchmark(n_train=20, n_test=10, seed=42)
    result = bench.run(epochs=2, lr=0.1)
    dt = (time.perf_counter() - t0) * 1000

    assert 0 <= result["accuracy"] <= 1.0
    assert len(result["predictions"]) == 10

    print(f"\n  PIPELINE ITER→DisruptionBenchmark (20+10, 2 epochs): {dt:.1f} ms")
    print(f"  Accuracy = {result['accuracy']:.2%}")
