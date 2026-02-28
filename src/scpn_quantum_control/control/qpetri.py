"""Quantum Petri net: superposition tokens on places.

Classical SPN tokens are probabilities in [0,1] per place.  Quantum Petri net
encodes tokens as qubit amplitudes -- a transition fires on all branches in
superposition.  Measurement collapses to a single control decision.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ..bridge.sc_to_quantum import probability_to_angle


class QuantumPetriNet:
    """Quantum Petri net with amplitude-encoded token state.

    Each place is one qubit.  Token density p maps to Ry(2*arcsin(sqrt(p))).
    Transitions apply controlled rotations based on arc weights.
    """

    def __init__(
        self,
        n_places: int,
        n_transitions: int,
        W_in: np.ndarray,
        W_out: np.ndarray,
        thresholds: np.ndarray,
    ):
        self.n_places = n_places
        self.n_transitions = n_transitions
        self.W_in = np.asarray(W_in, dtype=np.float64)
        self.W_out = np.asarray(W_out, dtype=np.float64)
        self.thresholds = np.asarray(thresholds, dtype=np.float64)

    def encode_marking(self, marking: np.ndarray) -> QuantumCircuit:
        """Amplitude-encode token state: p_i -> Ry(theta_i)|0>."""
        qc = QuantumCircuit(self.n_places)
        for i, m in enumerate(marking):
            theta = probability_to_angle(float(np.clip(m, 0.0, 1.0)))
            qc.ry(theta, i)
        return qc

    def apply_transition(self, circuit: QuantumCircuit, t_idx: int):
        """Apply transition t_idx as controlled rotations.

        Input arcs: consume tokens (negative rotation on input places).
        Output arcs: produce tokens (positive rotation on output places).
        Threshold check: CRy controlled by input places.
        """
        for p in range(self.n_places):
            w_in = self.W_in[t_idx, p]
            if abs(w_in) > 1e-15:
                theta = probability_to_angle(abs(w_in)) * self.thresholds[t_idx]
                circuit.ry(-theta, p)

        for p in range(self.n_places):
            w_out = self.W_out[p, t_idx]
            if abs(w_out) > 1e-15:
                theta = probability_to_angle(abs(w_out))
                circuit.ry(theta, p)

    def step(self, marking: np.ndarray) -> np.ndarray:
        """Full step: encode marking, apply all transitions, measure -> new marking."""
        qc = self.encode_marking(marking)
        for t in range(self.n_transitions):
            self.apply_transition(qc, t)

        sv = Statevector.from_instruction(qc)
        new_marking = np.zeros(self.n_places)
        for p in range(self.n_places):
            probs = sv.probabilities([p])
            new_marking[p] = probs[1]  # P(|1>) = token density
        return new_marking

    @classmethod
    def from_matrices(
        cls,
        W_in: np.ndarray,
        W_out: np.ndarray,
        thresholds: np.ndarray,
    ) -> QuantumPetriNet:
        n_t, n_p = W_in.shape
        return cls(n_p, n_t, W_in, W_out, thresholds)
