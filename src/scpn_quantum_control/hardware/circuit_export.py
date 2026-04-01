# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multi-Platform Circuit Export
"""Export Kuramoto-XY Trotter circuits to multiple quantum platforms.

Supports: OpenQASM 3.0, Cirq JSON, PyQuil Quil, PennyLane tape.
MISTIQS (USCCACS) inspired the multi-platform approach.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter

from ..bridge.knm_hamiltonian import knm_to_hamiltonian


def build_trotter_circuit(
    K: np.ndarray,
    omega: np.ndarray,
    t: float = 0.1,
    reps: int = 5,
) -> QuantumCircuit:
    """Build a Trotterised evolution circuit for the Kuramoto-XY Hamiltonian.

    Parameters
    ----------
    K : array (n, n)
        Coupling matrix.
    omega : array (n,)
        Natural frequencies.
    t : float
        Evolution time.
    reps : int
        Trotter repetitions.

    Returns
    -------
    QuantumCircuit
        Qiskit circuit ready for export.
    """
    n = K.shape[0]
    H = knm_to_hamiltonian(K, omega)
    synth = LieTrotter(reps=reps)
    evo = PauliEvolutionGate(H, time=t, synthesis=synth)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(float(omega[i]) % (2 * np.pi), i)
    qc.append(evo, range(n))
    qc.measure_all()
    return qc


def to_qasm3(K: np.ndarray, omega: np.ndarray, t: float = 0.1, reps: int = 5) -> str:
    """Export circuit as OpenQASM string.

    Uses qasm2.dumps for Qiskit 2.x compatibility.
    OpenQASM is the standard interchange format accepted by
    IBM, IonQ, Rigetti, Amazon Braket, and most cloud backends.
    """
    from qiskit import qasm2

    qc = build_trotter_circuit(K, omega, t, reps)
    return str(qasm2.dumps(qc))


def to_cirq(K: np.ndarray, omega: np.ndarray, t: float = 0.1, reps: int = 5) -> Any:
    """Export circuit as Cirq Circuit object.

    Requires: pip install cirq-core
    """
    try:
        from mitiq.interface.mitiq_qiskit.conversions import from_qiskit
    except ImportError:
        try:
            from cirq.contrib.qasm_import import circuit_from_qasm

            qasm_str = to_qasm3(K, omega, t, reps)
            return circuit_from_qasm(qasm_str)
        except ImportError as e:
            raise ImportError("cirq not installed: pip install cirq-core") from e

    qc = build_trotter_circuit(K, omega, t, reps)
    result = from_qiskit(qc)
    return result[0] if isinstance(result, tuple) else result


def to_quil(K: np.ndarray, omega: np.ndarray, t: float = 0.1, reps: int = 5) -> str:
    """Export circuit as Quil string (Rigetti PyQuil format).

    Requires: pip install pyquil
    Uses QASM→Quil conversion via qiskit transpilation to basis gates.
    """
    qc = build_trotter_circuit(K, omega, t, reps)
    # Transpile to basis gates that map to Quil
    from qiskit import transpile

    basis = ["rx", "ry", "rz", "cx", "measure"]
    qc_transpiled = transpile(qc, basis_gates=basis, optimization_level=2)

    lines = [f"DECLARE ro BIT[{qc_transpiled.num_qubits}]"]
    for instr in qc_transpiled.data:
        op = instr.operation
        qubits = [qc_transpiled.find_bit(q).index for q in instr.qubits]

        if op.name == "rx":
            lines.append(f"RX({float(op.params[0])}) {qubits[0]}")
        elif op.name == "ry":
            lines.append(f"RY({float(op.params[0])}) {qubits[0]}")
        elif op.name == "rz":
            lines.append(f"RZ({float(op.params[0])}) {qubits[0]}")
        elif op.name == "cx":
            lines.append(f"CNOT {qubits[0]} {qubits[1]}")
        elif op.name == "measure":
            clbits = [qc_transpiled.find_bit(c).index for c in instr.clbits]
            lines.append(f"MEASURE {qubits[0]} ro[{clbits[0]}]")

    return "\n".join(lines)


def export_all(
    K: np.ndarray,
    omega: np.ndarray,
    t: float = 0.1,
    reps: int = 5,
) -> dict:
    """Export circuit in all supported formats.

    Returns dict with keys: qiskit, qasm3, quil.
    Cirq is omitted by default (requires cirq-core).
    """
    qc = build_trotter_circuit(K, omega, t, reps)
    from qiskit import qasm2

    qasm = qasm2.dumps(qc)
    quil = to_quil(K, omega, t, reps)

    result: dict[str, Any] = {
        "qiskit": qc,
        "qasm3": qasm,
        "quil": quil,
        "n_qubits": K.shape[0],
        "depth": qc.depth(),
        "gate_count": qc.size(),
    }

    import contextlib

    with contextlib.suppress(ImportError):
        result["cirq"] = to_cirq(K, omega, t, reps)

    return result
