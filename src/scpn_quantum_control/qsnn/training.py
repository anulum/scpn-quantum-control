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

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

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
        """Return the optimizer value history as a QSNN loss history."""
        return self.training.value_history

    @property
    def best_loss(self) -> float:
        """Return the best observed full-batch QSNN loss."""
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


class QSNNTrainer:
    """Gradient-based trainer for QuantumDenseLayer via parameter-shift rule."""

    def __init__(self, layer: QuantumDenseLayer, lr: float = 0.01):
        self.layer = layer
        self.lr = lr
        self.optimizer = DifferentiableOptimizer(learning_rate=lr)

    def _build_circuit(
        self,
        inputs: np.ndarray,
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
        inputs: np.ndarray,
        angle_override: tuple[int, int, float] | None = None,
    ) -> np.ndarray:
        """Forward pass returning neuron P(|1>) (continuous, not thresholded)."""
        qc = self._build_circuit(inputs, angle_override)
        sv = Statevector.from_instruction(qc)
        probs: np.ndarray = np.array(
            [
                float(sv.probabilities([self.layer.n_inputs + n])[1])
                for n in range(self.layer.n_neurons)
            ]
        )
        return probs

    def _validate_dataset(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def _theta_values(self) -> np.ndarray:
        """Return current synapse CRy angles as a flat parameter vector."""
        return np.array(
            [
                self.layer.synapses[ni][ii].theta
                for ni in range(self.layer.n_neurons)
                for ii in range(self.layer.n_inputs)
            ],
            dtype=np.float64,
        )

    def _set_theta_values(self, values: np.ndarray) -> None:
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

    def _batch_loss(self, X: np.ndarray, y: np.ndarray, values: np.ndarray) -> float:
        """Return full-batch MSE loss for a flat synapse-angle vector."""
        self._set_theta_values(values)
        total_loss = 0.0
        for xi, yi in zip(X, y, strict=True):
            prediction = self._forward_probs(xi)
            total_loss += float(np.mean((prediction - yi) ** 2))
        return total_loss / float(X.shape[0])

    def parameter_shift_gradient(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
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

        def objective(flat_values: np.ndarray) -> float:
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

    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        """One epoch over dataset. Returns mean loss."""
        X, y = self._validate_dataset(X, y)
        total_loss = 0.0
        for xi, yi in zip(X, y):
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

        return total_loss / len(X)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> list[float]:
        """Train for multiple epochs. Returns loss history."""
        epoch_count = self._validate_epochs(epochs)
        history: list[float] = []
        for _ in range(epoch_count):
            history.append(self.train_epoch(X, y))
        return history

    def train_with_diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
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
        X: np.ndarray,
        y: np.ndarray,
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

        def objective(values: np.ndarray) -> float:
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
