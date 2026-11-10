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
import math
from dataclasses import replace
from itertools import product
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer
from scpn_quantum_control.qsnn.training import (
    QSNN_CHECKPOINT_SCHEMA,
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
    state = np.random.get_state(legacy=True)
    assert isinstance(state, tuple)
    name, key, position, has_gauss, gauss = state
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


def test_overlap_identity_follows_the_circuit_encoding() -> None:
    """Rows equal after the encoder's clip to ``[0, 1]`` are training samples.

    ``ry(pi * clip(x, 0, 1))`` maps 1.4 and 1.2 to the same angle, and -0.0 and
    0.0 likewise, so both held-out rows below drive circuits identical to
    training rows 0 and 2 and must be refused, also after a JSON resume.
    """
    X = np.array([[1.4, 0.5], [0.2, 0.3], [0.0, 0.7]])
    y = np.array([[1.0], [0.0], [0.0]])
    trainer = _trainer()
    trainer.train(X, y, epochs=1)
    encoded_twins = np.array([[1.2, 0.5], [-0.0, 0.7]])
    assert np.array_equal(np.clip(encoded_twins, 0.0, 1.0), np.clip(X[[0, 2]], 0.0, 1.0))
    with pytest.raises(ValueError, match=r"train/test overlap: held-out samples \[0, 1\]"):
        trainer.evaluate(encoded_twins, np.zeros((2, 1)))
    resumed = QSNNTrainer(QuantumDenseLayer(1, 2, seed=5), lr=0.2)
    resumed.restore(QSNNTrainingCheckpoint.from_json(trainer.checkpoint().to_json()))
    with pytest.raises(ValueError, match=r"held-out samples \[0, 1\]"):
        resumed.evaluate(encoded_twins, np.zeros((2, 1)))


def test_rows_that_differ_after_clipping_stay_distinct_and_raw_range_flags_ood() -> None:
    """A row that differs after clipping is new; the raw training box still flags range."""
    X = np.array([[1.4, 0.5], [0.2, 0.3], [0.0, 0.7]])
    y = np.array([[1.0], [0.0], [0.0]])
    trainer = _trainer()
    trainer.train(X, y, epochs=1)
    # 0.99 clips to 0.99, not 1.0, so it is a new in-range sample.
    # 1.6 clips to 1.0, but with 0.4 the row matches no training row, and the raw
    # value lies above the raw training maximum 1.4, so it is flagged.
    report = trainer.evaluate(np.array([[0.99, 0.5], [1.6, 0.4]]), np.zeros((2, 1)))
    assert report.out_of_distribution == (False, True)
    assert report.in_distribution_loss is not None


def test_version_one_checkpoints_with_raw_digests_are_refused() -> None:
    """Raw-byte ``v1`` digests cannot be converted, so ``v1`` text is refused."""
    X, y = _data(2, 4)
    trainer = _trainer()
    trainer.train(X, y, epochs=1)
    text = trainer.checkpoint().to_json()
    assert QSNN_CHECKPOINT_SCHEMA == "qsnn_training_checkpoint.v2"
    with pytest.raises(ValueError, match="unsupported QSNN checkpoint schema"):
        QSNNTrainingCheckpoint.from_json(
            text.replace(QSNN_CHECKPOINT_SCHEMA, "qsnn_training_checkpoint.v1")
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            {"completed_epochs": 1.5}, "completed_epochs must be a JSON integer", id="epochs-float"
        ),
        pytest.param(
            {"completed_epochs": True}, "completed_epochs must be a JSON integer", id="epochs-bool"
        ),
        pytest.param(
            {"completed_epochs": "1"}, "completed_epochs must be a JSON integer", id="epochs-text"
        ),
        pytest.param({"n_neurons": 1.0}, "n_neurons must be a JSON integer", id="neurons-float"),
        pytest.param({"n_inputs": None}, "n_inputs must be a JSON integer", id="inputs-null"),
        pytest.param({"weights": "0x1p-2"}, "weights must be a list", id="weights-text"),
        pytest.param({"weights": [0.25, 0.25]}, "weights must be a list", id="weights-numbers"),
        pytest.param({"feature_low": [0.2, 0.3]}, "feature_low must be a list", id="low-numbers"),
        pytest.param({"learning_rate": 0.2}, "learning_rate must be a float.hex", id="lr-number"),
        pytest.param({"learning_rate": "fast"}, "invalid hexadecimal", id="lr-not-hex"),
        pytest.param({"training_row_digests": "ab"}, "list of strings", id="digests-text"),
        pytest.param({"training_row_digests": [1]}, "list of strings", id="digests-numbers"),
        pytest.param({"extra": 1}, "exactly its fields", id="extra-field"),
    ],
)
def test_checkpoint_json_refuses_wrong_json_types(change: dict[str, object], message: str) -> None:
    """Every field keeps its JSON type; nothing is truncated or coerced.

    Parameters
    ----------
    change
        Fields replaced in, or added to, a valid checkpoint JSON object.
    message
        Expected refusal text.

    """
    X, y = _data(2, 4)
    trainer = _trainer()
    trainer.train(X, y, epochs=1)
    payload = json.loads(trainer.checkpoint().to_json())
    payload.update(change)
    with pytest.raises(ValueError, match=message):
        QSNNTrainingCheckpoint.from_json(json.dumps(payload))


@pytest.mark.parametrize("text", ["[]", "3", "{}"], ids=["array", "number", "empty-object"])
def test_checkpoint_json_must_be_the_exact_object(text: str) -> None:
    """Non-objects and objects missing fields are refused with ``ValueError``.

    Parameters
    ----------
    text
        JSON text that is not a complete checkpoint object.

    """
    with pytest.raises(ValueError, match="exactly its fields"):
        QSNNTrainingCheckpoint.from_json(text)


def test_checkpoint_and_evaluation_refusals() -> None:
    """Unknown schemas, foreign randomness, mismatched shapes and untrained evaluation fail."""
    trainer = _trainer()
    with pytest.raises(ValueError, match="requires recorded training data"):
        trainer.evaluate(*_data(8, 2))
    text = trainer.checkpoint().to_json()
    with pytest.raises(ValueError, match="schema"):
        QSNNTrainingCheckpoint.from_json(text.replace(QSNN_CHECKPOINT_SCHEMA, "v0"))
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


@pytest.mark.parametrize("encoded", [False, True], ids=["record", "json"])
@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        pytest.param("weights", (), "weight count", id="weights-empty"),
        pytest.param("weights", (0.25,), "weight count", id="weights-short"),
        pytest.param("weights", (0.25,) * 3, "weight count", id="weights-long"),
        pytest.param(
            "weights", (-0.25, 0.25), "weights outside synapse bounds", id="weights-below-bound"
        ),
        pytest.param(
            "weights", (0.25, 1.25), "weights outside synapse bounds", id="weights-above-bound"
        ),
        pytest.param("feature_low", (), "feature bounds", id="low-empty"),
        pytest.param("feature_low", (0.2,), "feature bounds", id="low-short"),
        pytest.param("feature_low", (0.2,) * 3, "feature bounds", id="low-long"),
        pytest.param("feature_high", (), "feature bounds", id="high-empty"),
        pytest.param("feature_high", (0.8,), "feature bounds", id="high-short"),
        pytest.param("feature_high", (0.8,) * 3, "feature bounds", id="high-long"),
        pytest.param("feature_low", None, "feature bounds", id="low-missing"),
        pytest.param("feature_high", None, "feature bounds", id="high-missing"),
        pytest.param(
            "weights", (np.nan, 0.25), "weights must contain only finite values", id="weights-nan"
        ),
        pytest.param(
            "weights", (np.inf, 0.25), "weights must contain only finite values", id="weights-inf"
        ),
        pytest.param("feature_low", (np.nan, 0.2), "feature bounds", id="low-nan"),
        pytest.param("feature_high", (np.inf, 0.8), "feature bounds", id="high-inf"),
        pytest.param("feature_low", (2.0, 2.0), "feature bounds", id="low-above-high"),
        pytest.param("feature_high", (-1.0, -1.0), "feature bounds", id="high-below-low"),
    ],
)
def test_checkpoint_shape_refusal_preserves_the_entire_trainer(
    field: str, values: tuple[float, ...] | None, message: str, encoded: bool
) -> None:
    """Invalid checkpoint dimensions leave weights and all training custody unchanged.

    Parameters
    ----------
    field
        Checkpoint vector with an invalid dimension or missing paired bound.
    values
        Malformed vector or missing bound.
    message
        Required refusal naming the invalid checkpoint contract.
    encoded
        Whether to cross the public JSON codec before restoration.

    """
    source = _trainer()
    source.train(*_data(90, 2), epochs=2)
    checkpoint = source.checkpoint()
    if field == "weights":
        assert values is not None
        candidate = replace(checkpoint, weights=values)
    elif field == "feature_low":
        candidate = replace(checkpoint, feature_low=values)
    else:
        candidate = replace(checkpoint, feature_high=values)
    if encoded:
        candidate = QSNNTrainingCheckpoint.from_json(candidate.to_json())
    target = QSNNTrainer(QuantumDenseLayer(1, 2, seed=37), lr=0.4)
    target.train(*_data(91, 2), epochs=1)
    before = target.checkpoint().to_json()
    optimizer = target.optimizer

    with pytest.raises(ValueError, match=message):
        target.restore(candidate)

    assert target.checkpoint().to_json() == before
    assert target.optimizer is optimizer


@pytest.mark.parametrize("encoded", [False, True], ids=["record", "json"])
@pytest.mark.parametrize("learning_rate", [-0.25, np.nan, np.inf])
def test_checkpoint_optimizer_refusal_precedes_weight_mutation(
    learning_rate: float, encoded: bool
) -> None:
    """An invalid saved optimiser cannot partly overwrite an otherwise valid layer.

    Parameters
    ----------
    learning_rate
        Negative or non-finite saved optimiser learning rate.
    encoded
        Whether to cross the public JSON codec before restoration.

    """
    source = _trainer()
    source.train(*_data(92, 2), epochs=2)
    candidate = replace(source.checkpoint(), learning_rate=learning_rate)
    if encoded:
        candidate = QSNNTrainingCheckpoint.from_json(candidate.to_json())
    target = QSNNTrainer(QuantumDenseLayer(1, 2, seed=37), lr=0.4)
    target.train(*_data(93, 2), epochs=1)
    before = target.checkpoint().to_json()
    optimizer = target.optimizer

    with pytest.raises(ValueError, match="learning_rate"):
        target.restore(candidate)

    assert target.checkpoint().to_json() == before
    assert target.optimizer is optimizer


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("weights", "weights must have shape"),
        ("feature_low", "feature bounds"),
        ("feature_high", "feature bounds"),
        ("loss_history", "loss_history must be a flat vector"),
    ],
)
def test_checkpoint_refuses_nested_vectors_without_mutation(field: str, message: str) -> None:
    """A direct record with an extra vector axis refuses before replacing any state.

    Parameters
    ----------
    field
        Flat checkpoint vector replaced by a nested two-by-two value.
    message
        Expected dimensional refusal.

    """
    source = _trainer()
    source.train(*_data(96, 2), epochs=1)
    nested = cast(tuple[float, ...], ((0.25, 0.25), (0.25, 0.25)))
    checkpoint = source.checkpoint()
    if field == "weights":
        candidate = replace(checkpoint, weights=nested)
    elif field == "feature_low":
        candidate = replace(checkpoint, feature_low=nested)
    elif field == "feature_high":
        candidate = replace(checkpoint, feature_high=nested)
    else:
        candidate = replace(checkpoint, loss_history=nested)
    target = QSNNTrainer(QuantumDenseLayer(1, 2, seed=37), lr=0.4)
    before = target.checkpoint().to_json()

    with pytest.raises(ValueError, match=message):
        target.restore(candidate)

    assert target.checkpoint().to_json() == before


def test_untrained_checkpoint_clears_the_previous_training_custody() -> None:
    """Restoring an initial checkpoint clears epochs, history and the old feature box."""
    initial = _trainer().checkpoint()
    target = QSNNTrainer(QuantumDenseLayer(1, 2, seed=37), lr=0.4)
    target.train(*_data(94, 2), epochs=2)
    assert target.checkpoint().completed_epochs == 2

    target.restore(QSNNTrainingCheckpoint.from_json(initial.to_json()))

    assert target.checkpoint() == initial
    with pytest.raises(ValueError, match="requires recorded training data"):
        target.evaluate(*_data(95, 2))


@pytest.mark.parametrize(
    ("features", "targets", "epochs", "message"),
    [
        (np.zeros(2), np.zeros((2, 1)), 1, "X must be a two-dimensional array"),
        (np.zeros((2, 2)), np.zeros(2), 1, "y must be a two-dimensional array"),
        (np.zeros((2, 3)), np.zeros((2, 1)), 1, "X must have 2 input columns"),
        (np.zeros((2, 2)), np.zeros((2, 2)), 1, "y must have 1 target columns"),
        (np.full((2, 2), np.nan), np.zeros((2, 1)), 1, "X must contain only finite values"),
        (np.zeros((2, 2)), np.full((2, 1), np.inf), 1, "y must contain only finite values"),
        (np.zeros((2, 2)), np.zeros((2, 1)), 0, "epochs must be positive"),
        (np.zeros((2, 2)), np.zeros((2, 1)), -1, "epochs must be positive"),
    ],
)
def test_training_input_refusal_preserves_the_entire_trainer(
    features: NDArray[np.float64], targets: NDArray[np.float64], epochs: int, message: str
) -> None:
    """A rejected training request preserves a previously trained checkpoint and optimiser.

    Parameters
    ----------
    features
        Public training feature array, malformed in the feature cases.
    targets
        Public training target array, malformed in the target cases.
    epochs
        Requested epoch count, non-positive in the epoch cases.
    message
        Required validation refusal from the public training call.

    """
    trainer = _trainer()
    trainer.train(*_data(97, 2), epochs=1)
    before = trainer.checkpoint().to_json()
    optimizer = trainer.optimizer

    with pytest.raises(ValueError, match=message):
        trainer.train(features, targets, epochs=epochs)

    assert trainer.checkpoint().to_json() == before
    assert trainer.optimizer is optimizer


def test_descent_refuses_non_finite_synapses_without_mutating_training_custody() -> None:
    """A non-finite public synapse update prevents descent and preserves saved custody."""
    trainer = _trainer()
    features, targets = _data(98, 2)
    trainer.train(features, targets, epochs=1)
    trainer.layer.synapses[0][0].update_weight(np.nan)
    before = trainer.checkpoint().to_json()

    with pytest.raises(ValueError, match="values must contain only finite angles"):
        trainer.train_with_parameter_shift_descent(features, targets, max_steps=1)

    assert trainer.checkpoint().to_json() == before


def test_non_finite_aggregate_loss_cannot_receive_convergence_diagnostics() -> None:
    """Overflow in accumulated finite per-sample losses cannot become a certificate."""
    trainer = _trainer()
    features, _ = _data(99, 2)
    targets = np.full((2, 1), 1e154, dtype=np.float64)

    with pytest.raises(ValueError, match="loss_history must contain only finite losses"):
        trainer.train_with_diagnostics(features, targets, epochs=1)

    failed = trainer.checkpoint()
    assert failed.completed_epochs == 1
    assert len(failed.loss_history) == 1
    assert np.isinf(failed.loss_history[0])


@pytest.mark.parametrize("encoded", [False, True], ids=["record", "json"])
@pytest.mark.parametrize(
    ("epochs", "history", "message"),
    [
        pytest.param(-1, (0.5,), "completed_epochs must be nonnegative", id="negative-epochs"),
        pytest.param(1, (), "loss_history must contain at least one epoch", id="history-empty"),
        pytest.param(
            2, (0.5,), "loss_history length must match completed_epochs", id="history-short"
        ),
        pytest.param(
            0, (0.5,), "loss_history length must match completed_epochs", id="history-long"
        ),
        pytest.param(
            1, (np.nan,), "loss_history must contain only finite losses", id="history-nan"
        ),
        pytest.param(
            1, (np.inf,), "loss_history must contain only finite losses", id="history-inf"
        ),
        pytest.param(
            1, (-0.5,), "loss_history must contain nonnegative losses", id="history-negative"
        ),
    ],
)
def test_checkpoint_history_refusal_preserves_the_entire_trainer(
    epochs: int, history: tuple[float, ...], message: str, encoded: bool
) -> None:
    """Inconsistent or failed saved training history refuses before replacing trainer state.

    Parameters
    ----------
    epochs
        Saved completed epoch count.
    history
        Saved loss history, malformed or inconsistent with the epoch count.
    message
        Required refusal describing the saved-state inconsistency.
    encoded
        Whether to cross the public JSON codec before restoration.

    """
    source = _trainer()
    source.train(*_data(100, 2), epochs=2)
    candidate = replace(source.checkpoint(), completed_epochs=epochs, loss_history=history)
    if encoded:
        candidate = QSNNTrainingCheckpoint.from_json(candidate.to_json())
    target = QSNNTrainer(QuantumDenseLayer(1, 2, seed=37), lr=0.4)
    target.train(*_data(101, 2), epochs=1)
    before = target.checkpoint().to_json()
    optimizer = target.optimizer

    with pytest.raises(ValueError, match=message):
        target.restore(candidate)

    assert target.checkpoint().to_json() == before
    assert target.optimizer is optimizer


@pytest.mark.parametrize("encoded", [False, True], ids=["record", "json"])
@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("rows", "training rows and feature bounds must be present together"),
        ("bounds", "training rows and feature bounds must be present together"),
        ("both", "completed epochs require training data custody"),
    ],
)
def test_checkpoint_missing_training_custody_refuses_before_mutation(
    missing: str, message: str, encoded: bool
) -> None:
    """A checkpoint cannot resume while forgetting which samples trained its weights.

    Parameters
    ----------
    missing
        Whether saved row digests, feature bounds, or both are missing.
    message
        Required refusal describing the incomplete training custody.
    encoded
        Whether to cross the public JSON codec before restoration.

    """
    source = _trainer()
    source.train(*_data(102, 2), epochs=1)
    candidate = source.checkpoint()
    if missing == "rows":
        candidate = replace(candidate, training_row_digests=())
    elif missing == "bounds":
        candidate = replace(candidate, feature_low=None, feature_high=None)
    else:
        candidate = replace(
            candidate, training_row_digests=(), feature_low=None, feature_high=None
        )
    if encoded:
        candidate = QSNNTrainingCheckpoint.from_json(candidate.to_json())
    target = QSNNTrainer(QuantumDenseLayer(1, 2, seed=37), lr=0.4)
    target.train(*_data(103, 2), epochs=1)
    before = target.checkpoint().to_json()
    optimizer = target.optimizer

    with pytest.raises(ValueError, match=message):
        target.restore(candidate)

    assert target.checkpoint().to_json() == before
    assert target.optimizer is optimizer


@pytest.mark.parametrize("field", ["schema", "randomness"])
def test_direct_checkpoint_identity_refusal_preserves_training_state(field: str) -> None:
    """Direct records enforce the same version and randomness identity as JSON loading.

    Parameters
    ----------
    field
        Checkpoint identity field replaced by an unsupported value.

    """
    candidate = _trainer().checkpoint()
    if field == "schema":
        candidate = replace(candidate, schema="unsupported")
        message = "unsupported QSNN checkpoint schema"
    else:
        candidate = replace(candidate, randomness="foreign random state")
        message = "randomness statement differs"
    target = QSNNTrainer(QuantumDenseLayer(1, 2, seed=37), lr=0.4)
    target.train(*_data(104, 2), epochs=1)
    before = target.checkpoint().to_json()

    with pytest.raises(ValueError, match=message):
        target.restore(candidate)

    assert target.checkpoint().to_json() == before


def test_checkpoint_accepts_exact_synapse_boundary_weights() -> None:
    """Restoring valid boundary weights preserves their exact values."""
    candidate = replace(_trainer().checkpoint(), weights=(0.0, 1.0))
    target = _trainer()

    target.restore(QSNNTrainingCheckpoint.from_json(candidate.to_json()))

    assert target.checkpoint() == candidate


def test_descent_only_checkpoint_preserves_overlap_refusal_without_epoch_history() -> None:
    """A full-batch descent checkpoint keeps data custody with zero SGD epochs."""
    source = _trainer()
    features, targets = _data(105, 2)
    source.train_with_parameter_shift_descent(features, targets, max_steps=1)
    saved = source.checkpoint()
    assert saved.completed_epochs == 0 and saved.loss_history == ()
    assert saved.training_row_digests
    target = _trainer()

    target.restore(QSNNTrainingCheckpoint.from_json(saved.to_json()))

    assert target.checkpoint() == saved
    with pytest.raises(ValueError, match="train/test overlap"):
        target.evaluate(features, targets)


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.75, 1.0])
@pytest.mark.parametrize("feature,target", [(1.0, 0.0), (0.3, 0.6)])
def test_qsnn_public_gradient_matches_the_analytic_cry_mse_derivative(
    weight: float, feature: float, target: float
) -> None:
    """Differentiate the actual one-synapse circuit without clipping shifted angles.

    Parameters
    ----------
    weight
        Physical synapse weight, including both closed interval endpoints.
    feature
        Input whose independently encoded excitation probability multiplies the readout.
    target
        MSE target used by the actual public trainer.

    """
    layer = QuantumDenseLayer(1, 1, weights=np.array([[weight]]), seed=42)
    trainer = QSNNTrainer(layer)
    before = trainer.checkpoint()
    actual = trainer.parameter_shift_gradient(np.array([feature]), np.array([target]))
    theta = math.pi * weight
    input_probability = math.sin(math.pi * feature / 2.0) ** 2
    probability = input_probability * math.sin(theta / 2.0) ** 2
    expected = (probability - target) * input_probability * math.sin(theta)
    np.testing.assert_allclose(actual, [[expected]], rtol=1e-12, atol=1e-14)
    assert trainer.checkpoint() == before


@pytest.mark.parametrize("learning_rate,target", [(0.4, 0.0), (10.0, 1.0)])
def test_qsnn_fullbatch_reports_the_actual_bounded_angles_and_exact_gradient(
    learning_rate: float, target: float
) -> None:
    """Keep full-batch evidence aligned with the trained physical synapse.

    Parameters
    ----------
    learning_rate
        Interior step or a step large enough to reach the synapse bound.
    target
        Requested output probability in the actual one-sample training call.

    """
    layer = QuantumDenseLayer(1, 1, weights=np.array([[0.25]]), seed=42)
    trainer = QSNNTrainer(layer, lr=learning_rate)
    result = trainer.train_with_parameter_shift_descent(
        np.array([[1.0]]), np.array([[target]]), max_steps=1, gradient_tolerance=0.0
    )
    theta = math.pi * trainer.checkpoint().weights[0]
    probability = math.sin(theta / 2.0) ** 2
    np.testing.assert_allclose(result.training.final_params, [theta], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        result.training.final_gradient,
        [(probability - target) * math.sin(theta)],
        rtol=1e-12,
        atol=1e-14,
    )
    assert result.training.final_value == pytest.approx((probability - target) ** 2, abs=1e-14)


def test_qsnn_entangled_gradient_matches_an_independent_input_mixture_oracle() -> None:
    """Differentiate three entangled readouts using a classical conditional-input sum."""
    weights = np.array([[0.0, 0.2], [0.7, 1.0], [0.35, 0.6]])
    features = np.array([0.25, 0.75])
    targets = np.array([0.7, 0.2, 0.9])
    trainer = QSNNTrainer(QuantumDenseLayer(3, 2, weights=weights, seed=42))
    before = trainer.checkpoint()
    actual = trainer.parameter_shift_gradient(features, targets)
    excitation = np.sin(np.pi * features / 2.0) ** 2
    theta = np.pi * weights
    probabilities = np.zeros(3)
    jacobian = np.zeros((3, 3, 2))
    for bits in product((0, 1), repeat=2):
        mass = math.prod(
            float(excitation[i] if bit else 1.0 - excitation[i]) for i, bit in enumerate(bits)
        )
        angles = theta @ np.array(bits)
        # Sequential CNOT readout k is the parity of conditional neurons 0..k.
        for readout in range(3):
            probabilities[readout] += (
                mass * (1.0 - math.prod(math.cos(float(a)) for a in angles[: readout + 1])) / 2.0
            )
            for neuron in range(readout + 1):
                other_cosines = math.prod(
                    math.cos(float(angles[k])) for k in range(readout + 1) if k != neuron
                )
                for input_index, bit in enumerate(bits):
                    jacobian[readout, neuron, input_index] += (
                        mass * bit * math.sin(float(angles[neuron])) * other_cosines / 2.0
                    )
    expected = 2.0 * np.einsum("k,kni->ni", probabilities - targets, jacobian) / 3.0
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    assert trainer.checkpoint() == before


def test_refused_descent_certificate_preserves_weights_and_training_custody() -> None:
    """A rejected certificate cannot commit otherwise successful numerical training."""
    trainer = QSNNTrainer(QuantumDenseLayer(1, 1, weights=np.array([[0.25]]), seed=42))
    before = trainer.checkpoint()
    with pytest.raises(ValueError, match="target_value_tolerance"):
        trainer.train_with_parameter_shift_descent(
            np.array([[1.0]]),
            np.array([[0.0]]),
            max_steps=1,
            target_loss_tolerance=-1.0,
        )
    assert trainer.checkpoint() == before
