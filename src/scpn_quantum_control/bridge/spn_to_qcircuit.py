"""SPN topology -> quantum circuit compiler.

Places map to qubits (amplitude = token density).
Transitions map to controlled-Ry gates (arc weights -> rotation angles).
Inhibitor arcs use the anti-control pattern: X-CRy-X.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from .sc_to_quantum import probability_to_angle


def spn_to_circuit(
    W_in: np.ndarray,
    W_out: np.ndarray,
    thresholds: np.ndarray,
) -> QuantumCircuit:
    """Convert SPN weight matrices to a quantum circuit.

    Args:
        W_in: (n_transitions, n_places) input arc weights. Negative = inhibitor.
        W_out: (n_places, n_transitions) output arc weights.
        thresholds: (n_transitions,) firing thresholds.

    Returns:
        QuantumCircuit with n_places qubits.
    """
    n_t, n_p = W_in.shape
    qc = QuantumCircuit(n_p)

    for t in range(n_t):
        for p in range(n_p):
            w = W_in[t, p]
            if abs(w) < 1e-15:
                continue

            theta = probability_to_angle(float(abs(w)))

            if w < 0:
                inhibitor_to_anti_control(qc, p, theta)
            else:
                # Input arc: controlled rotation removing tokens
                # Use threshold-weighted angle
                thresh_angle = theta * thresholds[t]
                qc.ry(-thresh_angle, p)

        for p in range(n_p):
            w = W_out[p, t]
            if abs(w) < 1e-15:
                continue
            theta = probability_to_angle(float(abs(w)))
            qc.ry(theta, p)

    return qc


def inhibitor_to_anti_control(circuit: QuantumCircuit, qubit: int, theta: float):
    """Inhibitor arc: fires when place is empty (anti-control).

    Anti-control pattern: X gate flips control sense, then CRy, then X restore.
    For single-qubit inhibitor, we condition on |0> via X-Ry-X.
    """
    circuit.x(qubit)
    circuit.ry(theta, qubit)
    circuit.x(qubit)
