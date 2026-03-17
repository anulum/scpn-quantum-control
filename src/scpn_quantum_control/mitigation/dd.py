# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Dynamical decoupling sequences for idle-qubit decoherence suppression.

Reference: Viola et al., PRL 82 2417 (1999).
"""

from __future__ import annotations

from enum import Enum

from qiskit import QuantumCircuit


class DDSequence(Enum):
    """Supported dynamical decoupling pulse sequences."""

    XY4 = "XY4"
    X2 = "X2"
    CPMG = "CPMG"  # Carr-Purcell-Meiboom-Gill, Meiboom & Gill (1958)


_PULSE_MAP = {
    DDSequence.XY4: ["x", "y", "x", "y"],
    DDSequence.X2: ["x", "x"],
    DDSequence.CPMG: ["y", "x", "y", "x"],
}


def insert_dd_sequence(
    circuit: QuantumCircuit,
    idle_qubits: list[int],
    sequence: DDSequence = DDSequence.XY4,
) -> QuantumCircuit:
    """Insert DD pulses on *idle_qubits* after the existing gates.

    For integration with the transpiler pass manager, prefer
    ``HardwareRunner.transpile_with_dd()`` which uses Qiskit's built-in
    ``PadDynamicalDecoupling`` pass.
    """
    out = circuit.copy()
    for q in idle_qubits:
        if q >= out.num_qubits:
            raise ValueError(f"qubit {q} out of range for {out.num_qubits}-qubit circuit")
    pulses = _PULSE_MAP[sequence]
    for q in idle_qubits:
        for gate in pulses:
            getattr(out, gate)(q)
    return out
