"""Quantum STDP learning via the parameter-shift rule.

Classical STDP updates weights from spike correlations.  Quantum STDP
computes d<Z>/d(theta) using the parameter-shift rule, then nudges
synapse angle to increase/decrease post-synaptic firing correlation.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class QuantumSTDP:
    """Parameter-shift STDP for QuantumSynapse objects.

    Gradient:
        d<Z>/d(theta) = [<Z>(theta+s) - <Z>(theta-s)] / (2*sin(s))
    Weight update:
        delta_w = lr * pre_spike * gradient
    """

    def __init__(self, learning_rate: float = 0.01, shift: float = np.pi / 2):
        self.lr = learning_rate
        self.shift = shift

    def _expectation_z(self, theta_w: float) -> float:
        """<Z> of post qubit after CRy(theta_w) with pre qubit in |1>."""
        qc = QuantumCircuit(2)
        qc.x(0)  # pre-synaptic spike
        qc.cry(theta_w, 0, 1)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities([1])  # marginal on post qubit
        return float(probs[0] - probs[1])  # <Z> = P(0) - P(1)

    def update(self, synapse: QuantumSynapse, pre_measured: int, post_measured: int) -> None:  # noqa: F821
        """Apply parameter-shift gradient update to synapse weight.

        Only updates when pre-synaptic neuron fired (pre_measured=1).
        """
        if pre_measured == 0:
            return

        theta = synapse.theta
        exp_plus = self._expectation_z(theta + self.shift)
        exp_minus = self._expectation_z(theta - self.shift)
        gradient = (exp_plus - exp_minus) / (2.0 * np.sin(self.shift))

        delta = self.lr * gradient
        synapse.update_weight(synapse.weight + delta)
