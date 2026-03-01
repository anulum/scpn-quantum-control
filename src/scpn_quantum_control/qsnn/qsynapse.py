"""Quantum synapse: controlled-Ry gate.

Classical BitstreamSynapse multiplies via AND gate: P(out) = P(pre)*P(weight).
Quantum analog: CRy(theta_w) on post qubit controlled by pre qubit.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


class QuantumSynapse:
    """Controlled-rotation synapse between two qubits.

    theta_w = pi * (w - w_min) / (w_max - w_min)
    """

    def __init__(self, weight: float, w_min: float = 0.0, w_max: float = 1.0):
        """Weight is clamped to [w_min, w_max]."""
        if w_max <= w_min:
            raise ValueError(f"w_max ({w_max}) must exceed w_min ({w_min})")
        self.w_min = w_min
        self.w_max = w_max
        self.weight = np.clip(weight, w_min, w_max)

    @property
    def theta(self) -> float:
        """CRy rotation angle: pi * (w - w_min) / (w_max - w_min)."""
        return float(np.pi * (self.weight - self.w_min) / (self.w_max - self.w_min))

    def effective_weight(self) -> float:
        """Weight as probability: sin^2(theta/2)."""
        return float(np.sin(self.theta / 2.0) ** 2)

    def apply(self, circuit: QuantumCircuit, pre_qubit: int, post_qubit: int) -> None:
        """Append CRy(theta_w) gate: pre controls rotation on post."""
        circuit.cry(self.theta, pre_qubit, post_qubit)

    def update_weight(self, new_w: float) -> None:
        """Set weight, clamped to [w_min, w_max]."""
        self.weight = np.clip(new_w, self.w_min, self.w_max)
