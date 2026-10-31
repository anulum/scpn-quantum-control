# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Training
"""Parameter-shift gradient training for QuantumDenseLayer.

Uses (f(w+pi/2) - f(w-pi/2)) / 2 per CRy angle on MSE loss.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ..dense_budget import require_dense_allocation
from ..differentiable import (
    DifferentiableOptimizer,
    GradientResult,
    Parameter,
    value_and_parameter_shift_grad,
)
from ..phase.gradient_descent import (
    ParameterShiftTrainingCertificate,
    ParameterShiftTrainingResult,
    parameter_shift_gradient_descent,
    validate_parameter_shift_training,
)
from .qlayer import QuantumDenseLayer
from .qsynapse import QuantumSynapse


@dataclass(frozen=True)
class QSNNTrainingDiagnostics:
    """Machine-checkable convergence evidence for QSNN parameter-shift training."""

    initial_loss: float
    final_loss: float
    best_loss: float
    loss_decrease: float
    max_loss_increase: float
    monotone_loss: bool
    best_improved: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable training diagnostics."""
        return {
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "loss_decrease": self.loss_decrease,
            "max_loss_increase": self.max_loss_increase,
            "monotone_loss": self.monotone_loss,
            "best_improved": self.best_improved,
        }


@dataclass(frozen=True)
class QSNNTrainingRun:
    """Structured QSNN training result with parameter-shift evaluation accounting."""

    loss_history: tuple[float, ...]
    diagnostics: QSNNTrainingDiagnostics
    epochs: int
    n_samples: int
    learning_rate: float
    parameter_shift_evaluations: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable training evidence."""
        return {
            "loss_history": list(self.loss_history),
            "diagnostics": self.diagnostics.to_dict(),
            "epochs": self.epochs,
            "n_samples": self.n_samples,
            "learning_rate": self.learning_rate,
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
        }


@dataclass(frozen=True)
class QSNNParameterShiftDescentRun:
    """Full-batch QSNN descent result backed by phase parameter-shift training."""

    training: ParameterShiftTrainingResult
    certificate: ParameterShiftTrainingCertificate
    n_samples: int
    n_parameters: int
    backend: str

    @property
    def loss_history(self) -> tuple[float, ...]:
        """Optimiser objective values recorded as QSNN loss history."""
        return self.training.value_history

    @property
    def best_loss(self) -> float:
        """Lowest full-batch QSNN loss observed during descent."""
        return self.training.best_value

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable full-batch training evidence."""
        return {
            "training": self.training.to_dict(),
            "certificate": self.certificate.to_dict(),
            "n_samples": self.n_samples,
            "n_parameters": self.n_parameters,
            "backend": self.backend,
            "loss_history": list(self.loss_history),
            "best_loss": self.best_loss,
        }


def _training_diagnostics(loss_history: tuple[float, ...]) -> QSNNTrainingDiagnostics:
    if not loss_history:
        raise ValueError("loss_history must contain at least one epoch")
    losses = np.asarray(loss_history, dtype=float)
    if not np.all(np.isfinite(losses)):
        raise ValueError("loss_history must contain only finite losses")
    deltas = np.diff(losses)
    max_increase = float(np.max(deltas)) if deltas.size else 0.0
    max_increase = max(0.0, max_increase)
    initial = float(losses[0])
    final = float(losses[-1])
    best = float(np.min(losses))
    return QSNNTrainingDiagnostics(
        initial_loss=initial,
        final_loss=final,
        best_loss=best,
        loss_decrease=initial - best,
        max_loss_increase=max_increase,
        monotone_loss=bool(max_increase <= 1e-12),
        best_improved=bool(best <= initial + 1e-12),
    )


QSNN_CHECKPOINT_SCHEMA: Final[str] = "qsnn_training_checkpoint.v1"
QSNN_RANDOMNESS: Final[str] = (
    "none consumed: forward passes are exact statevectors and parameter-shift gradients are "
    "deterministic; the trainer reads and writes no global or local random state"
)
QSNN_EVALUATION_CLAIM_BOUNDARY: Final[str] = (
    "held-out loss on the given in-distribution samples only; out-of-distribution samples are "
    "flagged, not scored; no physical fidelity, domain generalisation or advantage claim"
)


def _row_digest(row: NDArray[np.float64]) -> str:
    """Return the SHA-256 of one sample's float64 feature bytes."""
    return hashlib.sha256(np.ascontiguousarray(row, dtype=np.float64).tobytes()).hexdigest()


@dataclass(frozen=True)
class QSNNTrainingCheckpoint:
    """Everything needed to resume training exactly.

    The optimiser is plain gradient descent, so its whole state is the learning rate;
    ``randomness`` states that no random state exists to save. Training-row digests and the
    training feature box travel with the checkpoint so that a resumed trainer still refuses
    overlapping test data and flags out-of-distribution samples.
    """

    weights: tuple[float, ...]
    learning_rate: float
    completed_epochs: int
    loss_history: tuple[float, ...]
    n_neurons: int
    n_inputs: int
    training_row_digests: tuple[str, ...]
    feature_low: tuple[float, ...] | None
    feature_high: tuple[float, ...] | None
    randomness: str = QSNN_RANDOMNESS
    schema: str = QSNN_CHECKPOINT_SCHEMA

    def to_json(self) -> str:
        """Serialise exactly; floats are written as ``float.hex``.

        Returns
        -------
        str
            JSON text of the checkpoint.

        """

        def hexes(values: tuple[float, ...] | None) -> list[str] | None:
            return None if values is None else [float(value).hex() for value in values]

        return json.dumps(
            {
                "schema": self.schema,
                "weights": hexes(self.weights),
                "learning_rate": float(self.learning_rate).hex(),
                "completed_epochs": self.completed_epochs,
                "loss_history": hexes(self.loss_history),
                "n_neurons": self.n_neurons,
                "n_inputs": self.n_inputs,
                "training_row_digests": list(self.training_row_digests),
                "feature_low": hexes(self.feature_low),
                "feature_high": hexes(self.feature_high),
                "randomness": self.randomness,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> QSNNTrainingCheckpoint:
        """Parse a checkpoint, refusing another schema or randomness statement.

        Parameters
        ----------
        text
            Output of :meth:`to_json`.

        Returns
        -------
        QSNNTrainingCheckpoint
            The parsed checkpoint.

        Raises
        ------
        ValueError
            If the schema or randomness statement differs.

        """
        payload = json.loads(text)
        if payload.get("schema") != QSNN_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported QSNN checkpoint schema")
        if payload.get("randomness") != QSNN_RANDOMNESS:
            raise ValueError("QSNN checkpoint randomness statement differs from this trainer")

        def floats(values: list[str] | None) -> tuple[float, ...] | None:
            return None if values is None else tuple(float.fromhex(value) for value in values)

        weights = floats(payload["weights"])
        history = floats(payload["loss_history"])
        if weights is None or history is None:
            raise ValueError("QSNN checkpoint must carry weights and a loss history")
        return cls(
            weights=weights,
            learning_rate=float.fromhex(payload["learning_rate"]),
            completed_epochs=int(payload["completed_epochs"]),
            loss_history=history,
            n_neurons=int(payload["n_neurons"]),
            n_inputs=int(payload["n_inputs"]),
            training_row_digests=tuple(payload["training_row_digests"]),
            feature_low=floats(payload["feature_low"]),
            feature_high=floats(payload["feature_high"]),
        )


@dataclass(frozen=True)
class QSNNEvaluation:
    """Held-out evaluation with out-of-distribution samples flagged.

    Attributes
    ----------
    n_samples
        Number of evaluated samples.
    out_of_distribution
        Per sample, whether any feature lies outside the training feature box.
    out_of_distribution_count
        Number of flagged samples.
    in_distribution_loss
        Mean squared error over the unflagged samples, or ``None`` when every sample is flagged.
    claim_boundary
        What the numbers may be cited for.

    """

    n_samples: int
    out_of_distribution: tuple[bool, ...]
    out_of_distribution_count: int
    in_distribution_loss: float | None
    claim_boundary: str = QSNN_EVALUATION_CLAIM_BOUNDARY


class QSNNTrainer:
    """Gradient-based trainer for QuantumDenseLayer via parameter-shift rule."""

    def __init__(
        self,
        layer: QuantumDenseLayer,
        lr: float = 0.01,
        *,
        max_dense_gib: float | None = None,
    ) -> None:
        """Bind a layer to the parameter-shift trainer.

        Every forward pass simulates the layer densely, so the qubit count fixes
        a ``2**n_qubits`` statevector. Admission is checked at construction and
        again before each forward-pass circuit is built, since the layer and
        active process budget may change. The saved explicit budget, when given,
        applies to later passes too. Two complex-vector equivalents account for
        the state and the real absolute-value/square probability buffers.

        Parameters
        ----------
        layer
            Quantum dense layer to train.
        lr
            Learning rate for the descent optimiser.
        max_dense_gib
            Optional dense-allocation budget in GiB. ``None`` uses the active
            process budget.

        Raises
        ------
        DenseAllocationError
            If a forward-pass statevector for the layer exceeds the budget.

        """
        self.layer = layer
        self._max_dense_gib = max_dense_gib
        self._admit_forward()
        self.lr = lr
        self.optimizer = DifferentiableOptimizer(learning_rate=lr)
        self._completed_epochs = 0
        self._loss_history: list[float] = []
        self._training_rows: set[str] = set()
        self._feature_low: NDArray[np.float64] | None = None
        self._feature_high: NDArray[np.float64] | None = None

    def _record_training_data(self, X: NDArray[np.float64]) -> None:
        """Remember which samples and which feature range training has seen."""
        self._training_rows.update(_row_digest(row) for row in X)
        low, high = X.min(axis=0), X.max(axis=0)
        if self._feature_low is None or self._feature_high is None:
            self._feature_low, self._feature_high = low.copy(), high.copy()
        else:
            self._feature_low = np.minimum(self._feature_low, low)
            self._feature_high = np.maximum(self._feature_high, high)

    def checkpoint(self) -> QSNNTrainingCheckpoint:
        """Return the exact training state, including the optimiser and data custody.

        Returns
        -------
        QSNNTrainingCheckpoint
            Synapse weights, learning rate, epochs, loss history and training-data custody.

        """
        return QSNNTrainingCheckpoint(
            weights=tuple(
                float(self.layer.synapses[ni][ii].weight)
                for ni in range(self.layer.n_neurons)
                for ii in range(self.layer.n_inputs)
            ),
            learning_rate=float(self.lr),
            completed_epochs=self._completed_epochs,
            loss_history=tuple(self._loss_history),
            n_neurons=self.layer.n_neurons,
            n_inputs=self.layer.n_inputs,
            training_row_digests=tuple(sorted(self._training_rows)),
            feature_low=None if self._feature_low is None else tuple(self._feature_low.tolist()),
            feature_high=None
            if self._feature_high is None
            else tuple(self._feature_high.tolist()),
        )

    def restore(self, checkpoint: QSNNTrainingCheckpoint) -> None:
        """Resume from a checkpoint so that further training matches an uninterrupted run.

        Parameters
        ----------
        checkpoint
            A checkpoint of a trainer with the same layer shape.

        Raises
        ------
        ValueError
            If the layer shape differs.

        """
        if (checkpoint.n_neurons, checkpoint.n_inputs) != (
            self.layer.n_neurons,
            self.layer.n_inputs,
        ):
            raise ValueError("checkpoint layer shape differs from this trainer's layer")
        flat = iter(checkpoint.weights)
        for ni in range(self.layer.n_neurons):
            for ii in range(self.layer.n_inputs):
                self.layer.synapses[ni][ii].update_weight(next(flat))
        self.lr = checkpoint.learning_rate
        self.optimizer = DifferentiableOptimizer(learning_rate=checkpoint.learning_rate)
        self._completed_epochs = checkpoint.completed_epochs
        self._loss_history = list(checkpoint.loss_history)
        self._training_rows = set(checkpoint.training_row_digests)
        self._feature_low = (
            None if checkpoint.feature_low is None else np.array(checkpoint.feature_low)
        )
        self._feature_high = (
            None if checkpoint.feature_high is None else np.array(checkpoint.feature_high)
        )

    def evaluate(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> QSNNEvaluation:
        """Score held-out data, refusing overlap with training and flagging OOD samples.

        Parameters
        ----------
        X, y
            Held-out features and targets with the layer's shapes.

        Returns
        -------
        QSNNEvaluation
            Out-of-distribution flags and the loss over in-distribution samples.

        Raises
        ------
        ValueError
            If the trainer has seen no training data, or any held-out sample also appears in
            the training data (train/test overlap).

        """
        X, y = self._validate_dataset(X, y)
        if self._feature_low is None or self._feature_high is None:
            raise ValueError("evaluation requires recorded training data")
        overlap = [index for index, row in enumerate(X) if _row_digest(row) in self._training_rows]
        if overlap:
            raise ValueError(
                f"train/test overlap: held-out samples {overlap} were used in training; "
                "no held-out score is issued"
            )
        flags = np.any((self._feature_low > X) | (self._feature_high < X), axis=1)
        inside = [
            float(np.mean((self._forward_probs(row) - target) ** 2))
            for row, target, flagged in zip(X, y, flags, strict=True)
            if not flagged
        ]
        return QSNNEvaluation(
            n_samples=int(X.shape[0]),
            out_of_distribution=tuple(bool(flag) for flag in flags),
            out_of_distribution_count=int(np.count_nonzero(flags)),
            in_distribution_loss=float(np.mean(inside)) if inside else None,
        )

    def _admit_forward(self) -> None:
        """Check the current layer and budget before circuit/state construction."""
        require_dense_allocation(
            self.layer.n_qubits,
            dtype=np.complex128,
            rank=1,
            object_count=2,
            max_gib=self._max_dense_gib,
            label="QSNN forward-pass statevector",
        )

    def _build_circuit(
        self,
        inputs: NDArray[np.float64],
        angle_override: tuple[int, int, float] | None = None,
    ) -> QuantumCircuit:
        """Build layer circuit with optional angle shift on one synapse."""
        qc = QuantumCircuit(self.layer.n_qubits)

        for i, val in enumerate(inputs):
            qc.ry(np.pi * float(np.clip(val, 0.0, 1.0)), i)

        for n in range(self.layer.n_neurons):
            nq = self.layer.n_inputs + n
            for i in range(self.layer.n_inputs):
                angle = self.layer.synapses[n][i].theta
                if angle_override and angle_override[0] == n and angle_override[1] == i:
                    angle += angle_override[2]
                qc.cry(angle, i, nq)

        for n in range(self.layer.n_neurons - 1):
            qc.cx(self.layer.n_inputs + n, self.layer.n_inputs + n + 1)

        return qc

    def _forward_probs(
        self,
        inputs: NDArray[np.float64],
        angle_override: tuple[int, int, float] | None = None,
    ) -> NDArray[np.float64]:
        """Forward pass returning neuron P(|1>) (continuous, not thresholded)."""
        self._admit_forward()
        qc = self._build_circuit(inputs, angle_override)
        sv = Statevector.from_instruction(qc)
        probs: NDArray[np.float64] = np.array(
            [
                float(sv.probabilities([self.layer.n_inputs + n])[1])
                for n in range(self.layer.n_neurons)
            ]
        )
        return probs

    def _validate_dataset(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return finite two-dimensional training arrays with compatible shapes."""
        features = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float)
        if features.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if targets.ndim != 2:
            raise ValueError("y must be a two-dimensional array")
        if features.shape[0] == 0:
            raise ValueError("QSNN training requires at least one sample")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("X and y must have the same sample count")
        if features.shape[1] != self.layer.n_inputs:
            raise ValueError(
                f"X must have {self.layer.n_inputs} input columns, got {features.shape[1]}"
            )
        if targets.shape[1] != self.layer.n_neurons:
            raise ValueError(
                f"y must have {self.layer.n_neurons} target columns, got {targets.shape[1]}"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("X must contain only finite values")
        if not np.all(np.isfinite(targets)):
            raise ValueError("y must contain only finite values")
        return features.astype(np.float64, copy=True), targets.astype(np.float64, copy=True)

    @staticmethod
    def _validate_epochs(epochs: int) -> int:
        epoch_count = int(epochs)
        if epoch_count <= 0:
            raise ValueError("epochs must be positive")
        return epoch_count

    @staticmethod
    def _set_synapse_theta(synapse: QuantumSynapse, theta: float) -> None:
        """Set a synapse through its weight API from a target CRy angle."""
        w_min = float(synapse.w_min)
        w_max = float(synapse.w_max)
        weight = theta / np.pi * (w_max - w_min) + w_min
        synapse.update_weight(weight)

    def _theta_values(self) -> NDArray[np.float64]:
        """Return current synapse CRy angles as a flat parameter vector."""
        return np.array(
            [
                self.layer.synapses[ni][ii].theta
                for ni in range(self.layer.n_neurons)
                for ii in range(self.layer.n_inputs)
            ],
            dtype=np.float64,
        )

    def _set_theta_values(self, values: NDArray[np.float64]) -> None:
        """Set all synapse CRy angles from a flat parameter vector."""
        angles = np.asarray(values, dtype=float)
        expected = self.layer.n_neurons * self.layer.n_inputs
        if angles.shape != (expected,):
            raise ValueError(f"values must have shape ({expected},)")
        if not np.all(np.isfinite(angles)):
            raise ValueError("values must contain only finite angles")
        flat_index = 0
        for ni in range(self.layer.n_neurons):
            for ii in range(self.layer.n_inputs):
                self._set_synapse_theta(
                    self.layer.synapses[ni][ii],
                    float(angles[flat_index]),
                )
                flat_index += 1

    def _batch_loss(
        self, X: NDArray[np.float64], y: NDArray[np.float64], values: NDArray[np.float64]
    ) -> float:
        """Return full-batch MSE loss for a flat synapse-angle vector."""
        self._set_theta_values(values)
        total_loss = 0.0
        for xi, yi in zip(X, y, strict=True):
            prediction = self._forward_probs(xi)
            total_loss += float(np.mean((prediction - yi) ** 2))
        return total_loss / float(X.shape[0])

    def parameter_shift_gradient(
        self,
        inputs: NDArray[np.float64],
        target: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute gradient of MSE loss w.r.t. all synapse angles.

        Returns (n_neurons, n_inputs) gradient array.
        """
        shape = (self.layer.n_neurons, self.layer.n_inputs)
        values = self._theta_values()
        parameters = [
            Parameter(f"synapse_{ni}_{ii}")
            for ni in range(self.layer.n_neurons)
            for ii in range(self.layer.n_inputs)
        ]

        def objective(flat_values: NDArray[np.float64]) -> float:
            flat_index = 0
            for ni in range(self.layer.n_neurons):
                for ii in range(self.layer.n_inputs):
                    self._set_synapse_theta(
                        self.layer.synapses[ni][ii],
                        float(flat_values[flat_index]),
                    )
                    flat_index += 1
            prediction = self._forward_probs(inputs)
            return float(np.mean((prediction - target) ** 2))

        original = values.copy()
        try:
            result = value_and_parameter_shift_grad(
                objective,
                values,
                parameters=parameters,
            )
        finally:
            flat_index = 0
            for ni in range(self.layer.n_neurons):
                for ii in range(self.layer.n_inputs):
                    self._set_synapse_theta(
                        self.layer.synapses[ni][ii],
                        float(original[flat_index]),
                    )
                    flat_index += 1

        return result.gradient.reshape(shape)

    def train_epoch(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        """One epoch over dataset. Returns mean loss."""
        X, y = self._validate_dataset(X, y)
        self._record_training_data(X)
        total_loss = 0.0
        for xi, yi in zip(X, y, strict=True):
            pred = self._forward_probs(xi)
            total_loss += float(np.mean((pred - yi) ** 2))

            grad = self.parameter_shift_gradient(xi, yi)
            theta_values = np.array(
                [
                    self.layer.synapses[ni][ii].theta
                    for ni in range(self.layer.n_neurons)
                    for ii in range(self.layer.n_inputs)
                ],
                dtype=np.float64,
            )
            parameter_names = tuple(
                f"synapse_{ni}_{ii}"
                for ni in range(self.layer.n_neurons)
                for ii in range(self.layer.n_inputs)
            )
            gradient_payload = GradientResult(
                value=float(np.mean((pred - yi) ** 2)),
                gradient=grad.reshape(-1),
                method="parameter_shift",
                shift=np.pi / 2,
                coefficient=0.5,
                evaluations=1 + 2 * grad.size,
                parameter_names=parameter_names,
                trainable=tuple(True for _ in parameter_names),
            )
            updated = self.optimizer.step(theta_values, gradient_payload)
            flat_index = 0
            for ni in range(self.layer.n_neurons):
                for ii in range(self.layer.n_inputs):
                    self._set_synapse_theta(
                        self.layer.synapses[ni][ii],
                        float(updated[flat_index]),
                    )
                    flat_index += 1

        mean_loss = total_loss / len(X)
        self._completed_epochs += 1
        self._loss_history.append(mean_loss)
        return mean_loss

    def train(
        self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int = 10
    ) -> list[float]:
        """Train for multiple epochs. Returns loss history."""
        epoch_count = self._validate_epochs(epochs)
        history: list[float] = []
        for _ in range(epoch_count):
            history.append(self.train_epoch(X, y))
        return history

    def train_with_diagnostics(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        epochs: int = 10,
    ) -> QSNNTrainingRun:
        """Train and return structured convergence plus evaluation evidence."""
        X, y = self._validate_dataset(X, y)
        epoch_count = self._validate_epochs(epochs)
        history = tuple(float(value) for value in self.train(X, y, epochs=epoch_count))
        n_parameters = self.layer.n_neurons * self.layer.n_inputs
        return QSNNTrainingRun(
            loss_history=history,
            diagnostics=_training_diagnostics(history),
            epochs=epoch_count,
            n_samples=int(X.shape[0]),
            learning_rate=float(self.lr),
            parameter_shift_evaluations=epoch_count * int(X.shape[0]) * (1 + 2 * n_parameters),
        )

    def train_with_parameter_shift_descent(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        backend: str = "statevector",
        max_steps: int = 100,
        gradient_tolerance: float = 1e-8,
        value_tolerance: float | None = None,
        target_loss: float | None = None,
        target_loss_tolerance: float = 1e-8,
        min_loss_decrease: float | None = None,
        allow_hardware: bool = False,
    ) -> QSNNParameterShiftDescentRun:
        """Train QSNN synapse angles with full-batch parameter-shift descent.

        This route uses the same auditable optimizer as phase objectives, so
        QSNN training records backend planning, every accepted/rejected line
        search step, total objective evaluations, and a convergence certificate.
        """
        X, y = self._validate_dataset(X, y)
        original = self._theta_values()
        parameters = [
            Parameter(f"synapse_{ni}_{ii}")
            for ni in range(self.layer.n_neurons)
            for ii in range(self.layer.n_inputs)
        ]

        def objective(values: NDArray[np.float64]) -> float:
            return self._batch_loss(X, y, values)

        try:
            training = parameter_shift_gradient_descent(
                objective,
                original,
                parameters=parameters,
                backend=backend,
                learning_rate=self.lr,
                max_steps=max_steps,
                gradient_tolerance=gradient_tolerance,
                value_tolerance=value_tolerance,
                allow_hardware=allow_hardware,
            )
        except Exception:
            self._set_theta_values(original)
            raise

        self._set_theta_values(training.final_params)
        self._record_training_data(X)
        certificate = validate_parameter_shift_training(
            training,
            gradient_tolerance=gradient_tolerance,
            target_value=target_loss,
            target_value_tolerance=target_loss_tolerance,
            min_decrease=min_loss_decrease,
        )
        return QSNNParameterShiftDescentRun(
            training=training,
            certificate=certificate,
            n_samples=int(X.shape[0]),
            n_parameters=int(original.size),
            backend=training.backend_plan.backend,
        )
