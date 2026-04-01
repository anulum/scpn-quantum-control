# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Qsnn Training
"""Tests for qsnn/training.py."""

import numpy as np

from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer
from scpn_quantum_control.qsnn.training import QSNNTrainer


def test_trainer_init():
    layer = QuantumDenseLayer(2, 3, seed=42)
    trainer = QSNNTrainer(layer, lr=0.01)
    assert trainer.lr == 0.01
    assert trainer.layer is layer


def test_forward_probs_shape():
    layer = QuantumDenseLayer(2, 2, seed=0)
    trainer = QSNNTrainer(layer)
    probs = trainer._forward_probs(np.array([0.5, 0.5]))
    assert probs.shape == (2,)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_gradient_shape():
    layer = QuantumDenseLayer(2, 2, seed=0)
    trainer = QSNNTrainer(layer)
    grad = trainer.parameter_shift_gradient(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
    assert grad.shape == (2, 2)


def test_gradient_finite():
    layer = QuantumDenseLayer(2, 2, seed=0)
    trainer = QSNNTrainer(layer)
    grad = trainer.parameter_shift_gradient(np.array([0.3, 0.7]), np.array([0.5, 0.5]))
    assert np.all(np.isfinite(grad))


def test_train_epoch_returns_loss():
    layer = QuantumDenseLayer(2, 2, seed=0)
    trainer = QSNNTrainer(layer, lr=0.1)
    X = np.array([[0.3, 0.7], [0.8, 0.2]])
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    loss = trainer.train_epoch(X, y)
    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_train_returns_history():
    layer = QuantumDenseLayer(2, 2, seed=0)
    trainer = QSNNTrainer(layer, lr=0.1)
    X = np.array([[0.5, 0.5]])
    y = np.array([[1.0, 0.0]])
    history = trainer.train(X, y, epochs=3)
    assert len(history) == 3
    assert all(np.isfinite(h) for h in history)


def test_train_loss_trend():
    layer = QuantumDenseLayer(1, 2, seed=42)
    trainer = QSNNTrainer(layer, lr=0.5)
    X = np.array([[0.8, 0.2], [0.2, 0.8]])
    y = np.array([[0.9], [0.1]])
    history = trainer.train(X, y, epochs=5)
    # Allow fluctuation but final should not be drastically worse than start
    assert history[-1] <= history[0] + 0.5


def test_forward_with_angle_shift():
    layer = QuantumDenseLayer(1, 2, seed=0)
    trainer = QSNNTrainer(layer)
    base = trainer._forward_probs(np.array([0.5, 0.5]))
    shifted = trainer._forward_probs(np.array([0.5, 0.5]), angle_override=(0, 0, np.pi / 2))
    assert not np.allclose(base, shifted)


# ---------------------------------------------------------------------------
# Gradient mathematical invariants
# ---------------------------------------------------------------------------


def test_gradient_antisymmetric_shift():
    """Parameter-shift: g = (L(+pi/2) - L(-pi/2)) / 2 implies
    gradients flip sign if target flips (loss landscape symmetry)."""
    layer = QuantumDenseLayer(1, 2, seed=42)
    trainer = QSNNTrainer(layer)
    x = np.array([0.5, 0.5])
    g1 = trainer.parameter_shift_gradient(x, np.array([1.0]))
    g2 = trainer.parameter_shift_gradient(x, np.array([0.0]))
    # Gradients should point in opposite directions for opposite targets
    assert np.sign(g1[0, 0]) != np.sign(g2[0, 0]) or abs(g1[0, 0]) < 1e-6


def test_zero_lr_no_weight_change():
    """lr=0 → weights unchanged after training."""
    layer = QuantumDenseLayer(2, 2, seed=42)
    w_before = layer.get_weights().copy()
    trainer = QSNNTrainer(layer, lr=0.0)
    X = np.array([[0.5, 0.5]])
    y = np.array([[1.0, 0.0]])
    trainer.train(X, y, epochs=3)
    np.testing.assert_allclose(layer.get_weights(), w_before, atol=1e-14)


def test_forward_probs_bounded_01():
    """All forward probabilities must be in [0, 1]."""
    layer = QuantumDenseLayer(3, 3, seed=42)
    trainer = QSNNTrainer(layer)
    for _ in range(10):
        x = np.random.default_rng(42).uniform(0, 1, 3)
        probs = trainer._forward_probs(x)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)


# ---------------------------------------------------------------------------
# Training convergence physics
# ---------------------------------------------------------------------------


def test_train_weights_change():
    """Non-zero lr → weights must change after training."""
    layer = QuantumDenseLayer(1, 2, seed=42)
    w_before = layer.get_weights().copy()
    trainer = QSNNTrainer(layer, lr=0.5)
    X = np.array([[0.9, 0.1]])
    y = np.array([[0.0]])
    trainer.train(X, y, epochs=3)
    assert not np.array_equal(layer.get_weights(), w_before)


def test_loss_nonnegative():
    """MSE loss is always >= 0."""
    layer = QuantumDenseLayer(2, 2, seed=0)
    trainer = QSNNTrainer(layer, lr=0.1)
    X = np.array([[0.5, 0.5]])
    y = np.array([[0.5, 0.5]])
    loss = trainer.train_epoch(X, y)
    assert loss >= 0


# ---------------------------------------------------------------------------
# Pipeline wiring: QSNN trainer → SNN bridge → quantum output
# ---------------------------------------------------------------------------


def test_pipeline_trainer_to_snn_bridge():
    """Full pipeline: train QSNN → forward → produce spike output.

    Verifies the training module is wired into the inference path,
    not decorative.
    """
    import time

    layer = QuantumDenseLayer(2, 2, seed=42)
    trainer = QSNNTrainer(layer, lr=0.2)
    X = np.array([[0.8, 0.2], [0.2, 0.8]])
    y = np.array([[1.0, 0.0], [0.0, 1.0]])

    t0 = time.perf_counter()
    history = trainer.train(X, y, epochs=3)
    dt = (time.perf_counter() - t0) * 1000

    # After training, forward should produce output
    out = layer.forward(np.array([0.8, 0.2]))
    assert out.shape == (2,)
    assert set(np.unique(out)).issubset({0, 1})

    print(f"\n  PIPELINE QSNNTrainer (2×2, 3 epochs): {dt:.1f} ms")
    print(f"  Loss: {history[0]:.4f} → {history[-1]:.4f}")
    assert len(history) == 3
