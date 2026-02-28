"""Quantum LIF neuron: Ry rotation + Z-basis measurement.

Maps the classical StochasticLIFNeuron membrane dynamics to a parameterized
quantum circuit. Membrane voltage encodes as rotation angle; measurement
produces spike/no-spike with probability matching classical firing rate.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class QuantumLIFNeuron:
    """Single-qubit LIF neuron.

    Membrane equation (Euler):
        v(t+1) = v(t) - (dt/tau)(v(t) - v_rest) + R*I*dt

    Quantum mapping:
        theta = pi * clip((v - v_rest) / (v_threshold - v_rest), 0, 1)
        P(spike) = sin^2(theta/2)
        spike = 1 if P(|1>) > 0.5 (statevector mode)
    """

    def __init__(
        self,
        v_rest: float = 0.0,
        v_threshold: float = 1.0,
        tau_mem: float = 20.0,
        dt: float = 1.0,
        resistance: float = 1.0,
        n_shots: int = 100,
    ):
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        self.tau_mem = tau_mem
        self.dt = dt
        self.resistance = resistance
        self.n_shots = n_shots
        self.v = v_rest
        self._last_circuit: QuantumCircuit | None = None

    def step(self, input_current: float) -> int:
        """Update membrane, build Ry circuit, measure, return spike (0 or 1)."""
        dv_leak = -(self.v - self.v_rest) * (self.dt / self.tau_mem)
        dv_input = self.resistance * input_current * self.dt
        self.v += dv_leak + dv_input

        norm_v = (self.v - self.v_rest) / (self.v_threshold - self.v_rest)
        theta = np.pi * float(np.clip(norm_v, 0.0, 1.0))

        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        self._last_circuit = qc

        sv = Statevector.from_instruction(qc)
        p_spike = float(abs(sv[1]) ** 2)

        if self.n_shots > 0:
            spikes = np.random.binomial(1, p_spike, size=self.n_shots)
            spike = int(np.mean(spikes) > 0.5)
        else:
            spike = int(p_spike > 0.5)

        if spike:
            self.v = self.v_rest
        return spike

    def get_circuit(self) -> QuantumCircuit | None:
        return self._last_circuit

    def reset(self):
        self.v = self.v_rest
        self._last_circuit = None
