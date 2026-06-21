# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Qlayer
"""Quantum dense layer: multi-qubit entangled spiking network.

Maps sc-neurocore SCDenseLayer to a parameterized circuit:
  - Input register: Ry-encoded input values
  - Synapse connections: CRy gates from input to neuron qubits
  - Entanglement: CX chain between neuron qubits
  - Readout: measure neuron register, threshold for spikes
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .qsynapse import QuantumSynapse


def _apply_ry(state: NDArray[np.complex128], qubit: int, theta: float) -> None:
    """Apply an exact Ry rotation to ``qubit`` in-place."""
    cos = np.cos(theta / 2.0)
    sin = np.sin(theta / 2.0)
    stride = 1 << qubit
    step = stride << 1
    for block_start in range(0, state.size, step):
        for offset in range(stride):
            idx0 = block_start + offset
            idx1 = idx0 + stride
            amp0 = state[idx0]
            amp1 = state[idx1]
            state[idx0] = cos * amp0 - sin * amp1
            state[idx1] = sin * amp0 + cos * amp1


def _apply_controlled_ry(
    state: NDArray[np.complex128], control: int, target: int, theta: float
) -> None:
    """Apply controlled Ry exactly using little-endian qubit indexing."""
    cos = np.cos(theta / 2.0)
    sin = np.sin(theta / 2.0)
    control_mask = 1 << control
    target_mask = 1 << target
    for idx0 in range(state.size):
        if (idx0 & control_mask) == 0 or (idx0 & target_mask) != 0:
            continue
        idx1 = idx0 | target_mask
        amp0 = state[idx0]
        amp1 = state[idx1]
        state[idx0] = cos * amp0 - sin * amp1
        state[idx1] = sin * amp0 + cos * amp1


def _apply_cx(state: NDArray[np.complex128], control: int, target: int) -> None:
    """Apply CX exactly using little-endian qubit indexing."""
    control_mask = 1 << control
    target_mask = 1 << target
    for idx0 in range(state.size):
        if (idx0 & control_mask) == 0 or (idx0 & target_mask) != 0:
            continue
        idx1 = idx0 | target_mask
        state[idx0], state[idx1] = state[idx1], state[idx0]


def _probability_one(state: NDArray[np.complex128], qubit: int) -> float:
    """Return marginal P(qubit = 1)."""
    mask = 1 << qubit
    prob = sum(abs(amp) ** 2 for idx, amp in enumerate(state) if idx & mask)
    return float(prob)


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
        weights: NDArray[np.float64] | None = None,
        spike_threshold: float = 0.5,
        seed: int | None = None,
    ):
        """weights: (n_neurons, n_inputs) or None for random init in [0, 1]."""
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.spike_threshold = spike_threshold
        self.n_qubits = n_inputs + n_neurons

        if weights is None:
            weights = np.random.default_rng(seed).uniform(0.0, 1.0, (n_neurons, n_inputs))
        self.synapses = [
            [QuantumSynapse(float(weights[n, i])) for i in range(n_inputs)]
            for n in range(n_neurons)
        ]

    def forward(self, input_values: NDArray[np.float64]) -> NDArray[np.int64]:
        """Build circuit, measure neuron register, return spike array.

        Args:
            input_values: shape (n_inputs,) with values in [0, 1]

        Returns:
            shape (n_neurons,) int array of 0/1 spikes
        """
        state = np.zeros(1 << self.n_qubits, dtype=np.complex128)
        state[0] = 1.0

        for i, val in enumerate(input_values):
            theta = np.pi * float(np.clip(val, 0.0, 1.0))
            _apply_ry(state, i, theta)

        for n in range(self.n_neurons):
            neuron_qubit = self.n_inputs + n
            for i in range(self.n_inputs):
                _apply_controlled_ry(state, i, neuron_qubit, self.synapses[n][i].theta)

        for n in range(self.n_neurons - 1):
            _apply_cx(state, self.n_inputs + n, self.n_inputs + n + 1)

        neuron_probs = np.zeros(self.n_neurons)
        for n in range(self.n_neurons):
            neuron_probs[n] = _probability_one(state, self.n_inputs + n)

        result: NDArray[np.int64] = (neuron_probs > self.spike_threshold).astype(np.int64)
        return result

    def get_weights(self) -> NDArray[np.float64]:
        """Return (n_neurons, n_inputs) weight matrix."""
        result: NDArray[np.float64] = np.array(
            [
                [self.synapses[n][i].weight for i in range(self.n_inputs)]
                for n in range(self.n_neurons)
            ]
        )
        return result
