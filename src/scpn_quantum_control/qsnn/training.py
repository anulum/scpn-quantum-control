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

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ..differentiable import (
    DifferentiableOptimizer,
    GradientResult,
    Parameter,
    value_and_parameter_shift_grad,
)
from .qlayer import QuantumDenseLayer
from .qsynapse import QuantumSynapse


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

    @staticmethod
    def _set_synapse_theta(synapse: QuantumSynapse, theta: float) -> None:
        """Set a synapse through its weight API from a target CRy angle."""

        w_min = float(synapse.w_min)
        w_max = float(synapse.w_max)
        weight = theta / np.pi * (w_max - w_min) + w_min
        synapse.update_weight(weight)

    def parameter_shift_gradient(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Compute gradient of MSE loss w.r.t. all synapse angles.

        Returns (n_neurons, n_inputs) gradient array.
        """
        shape = (self.layer.n_neurons, self.layer.n_inputs)
        values = np.array(
            [
                self.layer.synapses[ni][ii].theta
                for ni in range(self.layer.n_neurons)
                for ii in range(self.layer.n_inputs)
            ],
            dtype=np.float64,
        )
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
        history: list[float] = []
        for _ in range(epochs):
            history.append(self.train_epoch(X, y))
        return history
