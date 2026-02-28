"""Quantum disruption classifier.

Encodes the 11-D disruption feature vector into a 4-qubit amplitude state
(zero-padded to 16-D), applies a parameterized quantum circuit as classifier,
and measures an ancilla qubit for disruption/safe classification.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class QuantumDisruptionClassifier:
    """Parameterized quantum circuit classifier for disruption prediction.

    Feature vector (11-D) -> amplitude encoding (16-D) -> PQC -> ancilla P(|1>).
    """

    N_FEATURES = 11

    def __init__(self, n_features: int = 11, n_layers: int = 3, seed: int = 42):
        self.n_features = n_features
        self.n_data_qubits = 4  # ceil(log2(16)) for 11->16 padding
        self.n_layers = n_layers
        self.n_qubits = self.n_data_qubits + 1  # +1 ancilla
        n_params = n_layers * (self.n_qubits * 2 + (self.n_qubits - 1))
        self.params = np.random.default_rng(seed).uniform(-np.pi, np.pi, n_params)

    def encode_features(self, features: np.ndarray) -> QuantumCircuit:
        """Amplitude-encode 11-D features into 4 qubits (pad to 16-D)."""
        padded = np.zeros(16)
        padded[: len(features)] = features
        norm = np.linalg.norm(padded)
        if norm < 1e-15:
            padded[0] = 1.0  # default to |0...0> for zero input
        else:
            padded /= norm

        qc = QuantumCircuit(self.n_qubits)
        qc.initialize(padded, list(range(self.n_data_qubits)))
        return qc

    def build_classifier(self, params: np.ndarray | None = None) -> QuantumCircuit:
        """Parameterized Ry/Rz + CX layers + ancilla readout."""
        if params is None:
            params = self.params
        qc = QuantumCircuit(self.n_qubits)
        idx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.ry(params[idx], q)
                idx += 1
            for q in range(self.n_qubits):
                qc.rz(params[idx], q)
                idx += 1
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
                idx += 1
        return qc

    def predict(self, features: np.ndarray) -> float:
        """Risk score from P(ancilla=|1>)."""
        enc = self.encode_features(features)
        clf = self.build_classifier()
        qc = enc.compose(clf)
        sv = Statevector.from_instruction(qc)
        ancilla_probs = sv.probabilities([self.n_data_qubits])
        return float(ancilla_probs[1])

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 0.1) -> None:
        """Parameter-shift gradient training on labeled data.

        X: (n_samples, n_features)
        y: (n_samples,) binary labels {0, 1}

        Uses the standard parameter-shift rule for Ry gates (generator Y/2):
        df/dθ = (f(θ+π/2) - f(θ-π/2)) / 2.  Schuld et al., PRA 99, 032331 (2019).
        """
        for _ in range(epochs):
            grad = np.zeros_like(self.params)

            for xi, yi in zip(X, y):
                pred = self._forward_with_params(xi, self.params)

                for p_idx in range(len(self.params)):
                    params_plus = self.params.copy()
                    params_plus[p_idx] += np.pi / 2
                    params_minus = self.params.copy()
                    params_minus[p_idx] -= np.pi / 2

                    f_plus = self._forward_with_params(xi, params_plus)
                    f_minus = self._forward_with_params(xi, params_minus)

                    dpred = (f_plus - f_minus) / 2.0
                    grad[p_idx] += 2.0 * (pred - yi) * dpred

            grad /= len(X)
            self.params -= lr * grad

    def _forward_with_params(self, features: np.ndarray, params: np.ndarray) -> float:
        enc = self.encode_features(features)
        clf = self.build_classifier(params)
        qc = enc.compose(clf)
        sv = Statevector.from_instruction(qc)
        return float(sv.probabilities([self.n_data_qubits])[1])
