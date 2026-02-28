"""Dynamical decoupling sequences for idle-qubit decoherence suppression.

Reference: Viola et al., PRL 82 2417 (1999).
"""

from __future__ import annotations

from enum import Enum

from qiskit import QuantumCircuit


class DDSequence(Enum):
    XY4 = "XY4"
    X2 = "X2"


_PULSE_MAP = {
    DDSequence.XY4: ["x", "y", "x", "y"],
    DDSequence.X2: ["x", "x"],
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
    pulses = _PULSE_MAP[sequence]
    for q in idle_qubits:
        if q >= out.num_qubits:
            raise ValueError(f"qubit {q} out of range for {out.num_qubits}-qubit circuit")
        for gate in pulses:
            getattr(out, gate)(q)
    return out
