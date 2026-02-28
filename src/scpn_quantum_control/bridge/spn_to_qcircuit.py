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

    Inhibitor arcs (W_in < 0): the place must be empty (|0>) for the
    transition to fire.  Implemented as anti-controlled rotation on output
    places: X on inhibitor place, CRy on output, X restore.

    Returns:
        QuantumCircuit with n_places qubits.
    """
    n_t, n_p = W_in.shape
    qc = QuantumCircuit(n_p)

    for t in range(n_t):
        inhibitor_places = []
        for p in range(n_p):
            w = W_in[t, p]
            if abs(w) < 1e-15:
                continue
            if w < 0:
                inhibitor_places.append(p)
            else:
                thresh_angle = probability_to_angle(float(abs(w))) * thresholds[t]
                qc.ry(-thresh_angle, p)

        for p in range(n_p):
            w = W_out[p, t]
            if abs(w) < 1e-15:
                continue
            theta = probability_to_angle(float(abs(w)))
            if inhibitor_places:
                inhibitor_anti_control(qc, inhibitor_places, p, theta)
            else:
                qc.ry(theta, p)

    return qc


def inhibitor_anti_control(
    circuit: QuantumCircuit, inhibitor_qubits: list[int], target: int, theta: float
):
    """Anti-control: output fires only when inhibitor places are empty (|0>).

    Pattern per inhibitor qubit: X flips control sense so CRy activates on |0>.
    """
    for q in inhibitor_qubits:
        circuit.x(q)
    if len(inhibitor_qubits) == 1 and inhibitor_qubits[0] != target:
        circuit.cry(theta, inhibitor_qubits[0], target)
    elif len(inhibitor_qubits) > 1:
        from qiskit.circuit.library import RYGate

        controls = [q for q in inhibitor_qubits if q != target]
        if controls:
            gate = RYGate(theta).control(len(controls))
            circuit.append(gate, controls + [target])
        else:
            circuit.ry(theta, target)
    else:
        circuit.ry(theta, target)
    for q in inhibitor_qubits:
        circuit.x(q)
