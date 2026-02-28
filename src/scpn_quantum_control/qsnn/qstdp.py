"""Quantum STDP learning via the parameter-shift rule.

Classical STDP updates weights from spike correlations.  Quantum STDP
computes d<Z>/d(theta) using the parameter-shift rule, then nudges
synapse angle based on pre/post spike correlation (Hebbian).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

if TYPE_CHECKING:
    from .qsynapse import QuantumSynapse


class QuantumSTDP:
    """Parameter-shift STDP for QuantumSynapse objects.

    Gradient:
        d<Z>/d(theta) = [<Z>(theta+s) - <Z>(theta-s)] / (2*sin(s))

    Hebbian update:
        pre=1, post=1: delta = +lr * |gradient|  (LTP)
        pre=1, post=0: delta = -lr * |gradient|  (LTD)
        pre=0:         no update
    """

    def __init__(self, learning_rate: float = 0.01, shift: float = np.pi / 2):
        self.lr = learning_rate
        self.shift = shift

    def _expectation_z(self, theta_w: float) -> float:
        """<Z> of post qubit after CRy(theta_w) with pre qubit in |1>."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cry(theta_w, 0, 1)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities([1])
        return float(probs[0] - probs[1])

    def update(self, synapse: QuantumSynapse, pre_measured: int, post_measured: int) -> None:
        """Apply Hebbian parameter-shift gradient update.

        LTP when both pre and post fire; LTD when pre fires but post doesn't.
        """
        if pre_measured == 0:
            return

        theta = synapse.theta
        exp_plus = self._expectation_z(theta + self.shift)
        exp_minus = self._expectation_z(theta - self.shift)
        gradient = (exp_plus - exp_minus) / (2.0 * np.sin(self.shift))

        sign = 1.0 if post_measured == 1 else -1.0
        delta = sign * self.lr * abs(gradient)
        synapse.update_weight(synapse.weight + delta)
