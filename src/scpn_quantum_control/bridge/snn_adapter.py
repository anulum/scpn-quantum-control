# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""SNN <> quantum bridge: spike trains to rotation angles, measurements to currents.

Supports raw numpy spike arrays and optional sc-neurocore ArcaneNeuron integration.
"""

from __future__ import annotations

import numpy as np

from ..qsnn.qlayer import QuantumDenseLayer


def spike_train_to_rotations(spikes: np.ndarray, window: int = 10) -> np.ndarray:
    """Convert spike history to Ry rotation angles.

    spikes: (timesteps, n_neurons) binary array.
    Returns (n_neurons,) angles = firing_rate * pi, in [0, pi].
    """
    spikes = np.asarray(spikes)
    if spikes.ndim == 1:
        spikes = spikes.reshape(1, -1)
    tail = spikes[-min(window, len(spikes)) :]
    rates = np.mean(tail, axis=0)
    angles: np.ndarray = rates * np.pi
    return angles


def quantum_measurement_to_current(values: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Convert quantum output values to SNN input currents.

    values: (n_neurons,) array — either P(|1>) probabilities in [0, 1]
    or binary spike indicators (0/1). Both are valid inputs.
    Returns (n_neurons,) input currents scaled by ``scale``.
    """
    currents: np.ndarray = np.asarray(values, dtype=np.float64) * scale
    return currents


class SNNQuantumBridge:
    """Bidirectional bridge: spike trains -> quantum circuit -> input currents.

    Orchestrates: firing rate -> Ry angles -> QuantumDenseLayer -> P(|1>) -> current.
    sc-neurocore is optional (pure numpy spike arrays accepted).
    """

    def __init__(
        self,
        n_neurons: int,
        n_inputs: int,
        window: int = 10,
        scale: float = 1.0,
        seed: int | None = None,
    ):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.window = window
        self.scale = scale
        self.layer = QuantumDenseLayer(n_neurons, n_inputs, seed=seed)

    def forward(self, spike_history: np.ndarray) -> np.ndarray:
        """Full forward pass: spike history -> quantum -> output currents.

        spike_history: (timesteps, n_inputs) binary spike array.
        Returns (n_neurons,) input currents for next SNN layer.
        """
        angles = spike_train_to_rotations(spike_history, self.window)
        input_values = angles / np.pi  # QuantumDenseLayer expects [0, 1]
        spikes = self.layer.forward(input_values[: self.n_inputs])
        return quantum_measurement_to_current(spikes.astype(float), self.scale)


class ArcaneNeuronBridge:
    """Bridge between sc-neurocore ArcaneNeuron and quantum layer.

    Runs ArcaneNeuron for n_steps, collects spike history from v_fast
    threshold crossings, passes through quantum layer, feeds output
    currents back as ArcaneNeuron input.

    Requires: pip install sc-neurocore
    """

    def __init__(
        self,
        n_neurons: int,
        n_inputs: int,
        threshold: float = 1.0,
        window: int = 10,
        scale: float = 1.0,
        seed: int | None = None,
    ):
        try:
            from sc_neurocore.neurons.models import ArcaneNeuron
        except ImportError as exc:
            raise ImportError("sc-neurocore required: pip install sc-neurocore") from exc

        self.threshold = threshold
        self.bridge = SNNQuantumBridge(n_neurons, n_inputs, window, scale, seed)
        self.neurons = [ArcaneNeuron() for _ in range(n_inputs)]
        self._spike_history: list[np.ndarray] = []

    def step_neurons(self, currents: np.ndarray) -> np.ndarray:
        """Step all ArcaneNeurons, return binary spike vector."""
        spikes: np.ndarray = np.zeros(len(self.neurons), dtype=np.float64)
        for i, neuron in enumerate(self.neurons):
            spikes[i] = float(neuron.step(float(currents[i])))
        self._spike_history.append(spikes)
        return spikes

    def quantum_forward(self) -> np.ndarray:
        """Pass accumulated spike history through quantum layer.

        Returns (n_neurons,) output currents.
        """
        if not self._spike_history:
            out: np.ndarray = np.zeros(self.bridge.n_neurons)
            return out
        history = np.array(self._spike_history)
        return self.bridge.forward(history)

    def step(self, external_currents: np.ndarray) -> dict:
        """Full cycle: step neurons -> quantum forward -> output.

        Returns dict with spike vector, output currents, and neuron states.
        """
        spikes = self.step_neurons(external_currents)
        output_currents = self.quantum_forward()
        states = [n.get_state() for n in self.neurons]
        return {
            "spikes": spikes,
            "output_currents": output_currents,
            "v_deep": np.array([s["v_deep"] for s in states]),
            "confidence": np.array([s["confidence"] for s in states]),
        }

    def reset(self) -> None:
        """Reset neurons and spike history. v_deep persists (identity)."""
        for n in self.neurons:
            n.reset()
        self._spike_history.clear()
