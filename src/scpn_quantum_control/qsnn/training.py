# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Training
"""Parameter-shift gradient training for QuantumDenseLayer.

Uses (f(w+pi/2) - f(w-pi/2)) / 2 per CRy angle on MSE loss.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .qlayer import QuantumDenseLayer


class QSNNTrainer:
    """Gradient-based trainer for QuantumDenseLayer via parameter-shift rule."""

    def __init__(self, layer: QuantumDenseLayer, lr: float = 0.01):
        self.layer = layer
        self.lr = lr

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

    def parameter_shift_gradient(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Compute gradient of MSE loss w.r.t. all synapse angles.

        Returns (n_neurons, n_inputs) gradient array.
        """
        grad = np.zeros((self.layer.n_neurons, self.layer.n_inputs))

        for ni in range(self.layer.n_neurons):
            for ii in range(self.layer.n_inputs):
                p_plus = self._forward_probs(inputs, (ni, ii, np.pi / 2))
                l_plus = float(np.mean((p_plus - target) ** 2))
                p_minus = self._forward_probs(inputs, (ni, ii, -np.pi / 2))
                l_minus = float(np.mean((p_minus - target) ** 2))
                grad[ni, ii] = (l_plus - l_minus) / 2.0

        result: np.ndarray = grad
        return result

    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        """One epoch over dataset. Returns mean loss."""
        total_loss = 0.0
        for xi, yi in zip(X, y):
            pred = self._forward_probs(xi)
            total_loss += float(np.mean((pred - yi) ** 2))

            grad = self.parameter_shift_gradient(xi, yi)
            for ni in range(self.layer.n_neurons):
                for ii in range(self.layer.n_inputs):
                    syn = self.layer.synapses[ni][ii]
                    new_theta = syn.theta - self.lr * grad[ni, ii]
                    new_w = new_theta / np.pi * (syn.w_max - syn.w_min) + syn.w_min
                    syn.update_weight(new_w)

        return total_loss / len(X)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> list[float]:
        """Train for multiple epochs. Returns loss history."""
        history: list[float] = []
        for _ in range(epochs):
            history.append(self.train_epoch(X, y))
        return history
