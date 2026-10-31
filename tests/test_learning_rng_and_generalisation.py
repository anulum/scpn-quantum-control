# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — learning: random-state custody, resumable training and held-out honesty
"""Training leaves the caller's random state alone, resumes exactly, and scores held-out data honestly.

Oracles: the global NumPy random state before and after, an uninterrupted
run compared with a checkpointed and resumed one, deliberate train/test
overlap, and samples outside the training feature range.
"""

from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer
from scpn_quantum_control.qsnn.training import (
    QSNN_EVALUATION_CLAIM_BOUNDARY,
    QSNN_RANDOMNESS,
    QSNNTrainer,
    QSNNTrainingCheckpoint,
)


def _data(seed: int, count: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return features in ``[0.2, 0.8]^2`` and targets in ``[0, 1]``."""
    rng = np.random.default_rng(seed)
    features = rng.uniform(0.2, 0.8, size=(count, 2))
    targets = (features[:, :1] > 0.5).astype(np.float64)
    return features, targets


def _global_state() -> tuple[str, bytes, int, int, float]:
    """Return the legacy global NumPy random state as comparable values."""
    name, key, position, has_gauss, gauss = cast(
        tuple[str, NDArray[np.uint32], int, int, float], np.random.get_state()
    )
    return name, key.tobytes(), position, has_gauss, gauss


def _trainer() -> QSNNTrainer:
    """Return a seeded one-neuron, two-input trainer."""
    return QSNNTrainer(QuantumDenseLayer(1, 2, seed=11), lr=0.2)


def test_learning_rng_and_generalisation_01_training_does_not_touch_the_global_random_state() -> (
    None
):
    """The global NumPy state is identical before and after training, descent and evaluation."""
    X, y = _data(1, 6)
    np.random.seed(1234)
    before = _global_state()
    trainer = _trainer()
    trainer.train(X, y, epochs=2)
    trainer.train_with_parameter_shift_descent(X, y, max_steps=2)
    trainer.evaluate(*_data(2, 4))
    after = _global_state()
    assert before == after
    np.random.rand()
    assert _global_state() != after  # the oracle sees a single draw
    assert trainer.checkpoint().randomness == QSNN_RANDOMNESS


def test_learning_rng_and_generalisation_02_checkpoint_resume_matches_an_uninterrupted_run() -> (
    None
):
    """Two epochs, a JSON checkpoint, a fresh trainer and two more epochs equal four epochs."""
    X, y = _data(3, 5)
    straight = _trainer()
    straight_history = straight.train(X, y, epochs=4)
    first = _trainer()
    first.train(X, y, epochs=2)
    saved = QSNNTrainingCheckpoint.from_json(first.checkpoint().to_json())
    resumed = QSNNTrainer(QuantumDenseLayer(1, 2, seed=99), lr=0.5)
    resumed.restore(saved)
    resumed.train(X, y, epochs=2)
    final = resumed.checkpoint()
    reference = straight.checkpoint()
    assert final.weights == reference.weights
    assert final.loss_history == tuple(straight_history) == reference.loss_history
    assert final.completed_epochs == 4 and final.learning_rate == 0.2
    assert final == reference


def test_learning_rng_and_generalisation_03_train_test_overlap_refuses_evaluation() -> None:
    """A held-out set containing a training sample gets no score, even after a resume."""
    X, y = _data(4, 6)
    trainer = _trainer()
    trainer.train(X, y, epochs=1)
    # Midpoints of distinct training pairs lie inside the training box but are new samples.
    test_X = 0.5 * (X[:3] + X[3:6])
    test_y = (test_X[:, :1] > 0.5).astype(np.float64)
    mixed_X = np.vstack([test_X, X[2:3]])
    mixed_y = np.vstack([test_y, y[2:3]])
    with pytest.raises(ValueError, match=r"train/test overlap: held-out samples \[3\]"):
        trainer.evaluate(mixed_X, mixed_y)
    resumed = QSNNTrainer(QuantumDenseLayer(1, 2, seed=5), lr=0.2)
    resumed.restore(trainer.checkpoint())
    with pytest.raises(ValueError, match="train/test overlap"):
        resumed.evaluate(mixed_X, mixed_y)
    clean = trainer.evaluate(test_X, test_y)
    assert clean.n_samples == 3 and clean.out_of_distribution_count == 0
    assert clean.in_distribution_loss is not None


def test_learning_rng_and_generalisation_04_out_of_distribution_data_stays_flagged() -> None:
    """Samples outside the training feature box are flagged and excluded from the loss."""
    X, y = _data(6, 8)
    trainer = _trainer()
    trainer.train(X, y, epochs=1)
    inside_X = 0.5 * (X[:3] + X[3:6])
    inside_y = (inside_X[:, :1] > 0.5).astype(np.float64)
    outside_X = np.array([[0.95, 0.5], [0.5, 0.05]])
    outside_y = np.ones((2, 1))
    report = trainer.evaluate(np.vstack([inside_X, outside_X]), np.vstack([inside_y, outside_y]))
    assert report.out_of_distribution == (False, False, False, True, True)
    assert report.out_of_distribution_count == 2
    inside_only = trainer.evaluate(inside_X, inside_y)
    assert report.in_distribution_loss == pytest.approx(inside_only.in_distribution_loss)
    everything_outside = trainer.evaluate(outside_X, outside_y)
    assert everything_outside.in_distribution_loss is None
    assert report.claim_boundary == QSNN_EVALUATION_CLAIM_BOUNDARY
    assert (
        "not scored" in report.claim_boundary and "no physical fidelity" in report.claim_boundary
    )


def test_checkpoint_and_evaluation_refusals() -> None:
    """Unknown schemas, foreign randomness, mismatched shapes and untrained evaluation fail."""
    trainer = _trainer()
    with pytest.raises(ValueError, match="requires recorded training data"):
        trainer.evaluate(*_data(8, 2))
    text = trainer.checkpoint().to_json()
    with pytest.raises(ValueError, match="schema"):
        QSNNTrainingCheckpoint.from_json(text.replace("qsnn_training_checkpoint.v1", "v0"))
    with pytest.raises(ValueError, match="randomness"):
        QSNNTrainingCheckpoint.from_json(text.replace("none consumed", "seeded"))
    with pytest.raises(ValueError, match="layer shape"):
        QSNNTrainer(QuantumDenseLayer(2, 2, seed=1)).restore(trainer.checkpoint())
    stripped = json.loads(text)
    stripped["weights"] = None
    with pytest.raises(ValueError, match="weights and a loss history"):
        QSNNTrainingCheckpoint.from_json(json.dumps(stripped))
    empty = QSNNTrainingCheckpoint.from_json(text)
    assert empty.feature_low is None and empty.completed_epochs == 0
