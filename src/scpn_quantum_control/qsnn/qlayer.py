"""Quantum dense layer: multi-qubit entangled spiking network.

Maps sc-neurocore SCDenseLayer to a parameterized circuit:
  - Input register: Ry-encoded input values
  - Synapse connections: CRy gates from input to neuron qubits
  - Entanglement: CX chain between neuron qubits
  - Readout: measure neuron register, threshold for spikes
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .qsynapse import QuantumSynapse


class QuantumDenseLayer:
    """Multi-qubit dense layer with entanglement between neurons.

    n_qubits = n_inputs + n_neurons.
    Input qubits: [0, n_inputs)
    Neuron qubits: [n_inputs, n_inputs + n_neurons)
    """

    def __init__(
        self,
        n_neurons: int,
        n_inputs: int,
        weights: np.ndarray | None = None,
        spike_threshold: float = 0.5,
    ):
        """weights: (n_neurons, n_inputs) or None for random init in [0, 1]."""
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.spike_threshold = spike_threshold
        self.n_qubits = n_inputs + n_neurons

        if weights is None:
            weights = np.random.default_rng().uniform(0.0, 1.0, (n_neurons, n_inputs))
        self.synapses = [
            [QuantumSynapse(float(weights[n, i])) for i in range(n_inputs)]
            for n in range(n_neurons)
        ]

    def forward(self, input_values: np.ndarray) -> np.ndarray:
        """Build circuit, measure neuron register, return spike array.

        Args:
            input_values: shape (n_inputs,) with values in [0, 1]

        Returns:
            shape (n_neurons,) int array of 0/1 spikes
        """
        qc = QuantumCircuit(self.n_qubits)

        for i, val in enumerate(input_values):
            theta = np.pi * float(np.clip(val, 0.0, 1.0))
            qc.ry(theta, i)

        for n in range(self.n_neurons):
            neuron_qubit = self.n_inputs + n
            for i in range(self.n_inputs):
                self.synapses[n][i].apply(qc, i, neuron_qubit)

        for n in range(self.n_neurons - 1):
            qc.cx(self.n_inputs + n, self.n_inputs + n + 1)

        sv = Statevector.from_instruction(qc)
        neuron_probs = np.zeros(self.n_neurons)
        for n in range(self.n_neurons):
            marginal = sv.probabilities([self.n_inputs + n])
            neuron_probs[n] = marginal[1]  # P(|1>)

        return (neuron_probs > self.spike_threshold).astype(int)

    def get_weights(self) -> np.ndarray:
        """Return (n_neurons, n_inputs) weight matrix."""
        return np.array(
            [
                [self.synapses[n][i].weight for i in range(self.n_inputs)]
                for n in range(self.n_neurons)
            ]
        )
