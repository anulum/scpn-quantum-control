# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Spn To Qcircuit
"""SPN topology -> quantum circuit compiler.

Places map to qubits (amplitude = token density).
Transitions map to controlled-Ry gates (arc weights -> rotation angles).
Inhibitor arcs use the anti-control pattern: X-CRy-X.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

from .._constants import WEIGHT_SPARSITY_EPS
from .sc_to_quantum import probability_to_angle


def spn_to_circuit(
    W_in: NDArray[np.float64],
    W_out: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> QuantumCircuit:
    """Convert SPN weight matrices to a quantum circuit.

    Parameters
    ----------
    W_in
        Input arc weights shaped ``(n_transitions, n_places)``. Negative
        entries designate inhibitor arcs.
    W_out
        Output arc weights shaped ``(n_places, n_transitions)``.
    thresholds
        Firing thresholds shaped ``(n_transitions,)``.

    Returns
    -------
    qiskit.QuantumCircuit
        Circuit containing one qubit per place.

    Notes
    -----
    An inhibitor arc requires its place to be empty for the transition to
    fire. The circuit implements this condition by surrounding the controlled
    output rotation with Pauli-X gates on each inhibitor place.

    """
    n_t, n_p = W_in.shape
    qc = QuantumCircuit(n_p)

    for t in range(n_t):
        inhibitor_places = []
        for p in range(n_p):
            w = W_in[t, p]
            if abs(w) < WEIGHT_SPARSITY_EPS:
                continue
            if w < 0:
                inhibitor_places.append(p)
            else:
                thresh_angle = probability_to_angle(float(abs(w))) * thresholds[t]
                qc.ry(-thresh_angle, p)

        for p in range(n_p):
            w = W_out[p, t]
            if abs(w) < WEIGHT_SPARSITY_EPS:
                continue
            theta = probability_to_angle(float(abs(w)))
            if inhibitor_places:
                inhibitor_anti_control(qc, inhibitor_places, p, theta)
            else:
                qc.ry(theta, p)

    return qc


def inhibitor_anti_control(
    circuit: QuantumCircuit, inhibitor_qubits: list[int], target: int, theta: float
) -> None:
    """Anti-control: output fires only when inhibitor places are empty (|0>).

    Pattern per inhibitor qubit: X flips control sense so CRy activates on |0>.

    Parameters
    ----------
    circuit
        Circuit to mutate with the anti-controlled rotation.
    inhibitor_qubits
        Qubits whose zero state enables the output rotation.
    target
        Output qubit that receives the rotation.
    theta
        Rotation angle in radians.

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
