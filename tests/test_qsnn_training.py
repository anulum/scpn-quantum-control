# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Qsnn Training
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
