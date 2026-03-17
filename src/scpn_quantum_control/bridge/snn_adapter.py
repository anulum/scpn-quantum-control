# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""SNN <> quantum bridge: spike trains to rotation angles, measurements to currents."""

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


def quantum_measurement_to_current(probs: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Convert qubit P(|1>) probabilities to SNN input currents.

    probs: (n_neurons,) array of P(|1>) values in [0, 1].
    Returns (n_neurons,) input currents scaled by ``scale``.
    """
    currents: np.ndarray = np.asarray(probs, dtype=np.float64) * scale
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
