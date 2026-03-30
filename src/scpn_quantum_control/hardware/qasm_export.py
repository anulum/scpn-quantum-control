# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Qasm Export
"""OpenQASM 3 circuit export for cross-platform portability.

Exports Kuramoto-XY simulation circuits as OpenQASM 3.0 strings
for execution on any QASM-compatible backend (IBM, IonQ, Rigetti,
Quantinuum, etc.).

Supported exports:
    1. Trotter evolution circuit (for given K, omega, t, reps)
    2. VQE ansatz circuit (K_nm-informed)
    3. Measurement circuit (with classical register)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.qasm3 import dumps as qasm3_dumps
from qiskit.synthesis import LieTrotter

from ..bridge.knm_hamiltonian import knm_to_ansatz, knm_to_hamiltonian


@dataclass
class QASMExportResult:
    """QASM export result."""

    qasm_string: str
    n_qubits: int
    gate_count: int
    circuit_depth: int
    format_version: str


def _build_trotter_circuit(
    K: np.ndarray, omega: np.ndarray, t: float, reps: int
) -> QuantumCircuit:
    """Build Trotter evolution circuit from K, omega."""
    n = K.shape[0]
    H = knm_to_hamiltonian(K, omega)
    synth = LieTrotter(reps=reps)
    evo = PauliEvolutionGate(H, time=t, synthesis=synth)
    qc = QuantumCircuit(n)
    qc.append(evo, range(n))
    return qc.decompose()


def export_trotter_qasm(
    K: np.ndarray,
    omega: np.ndarray,
    t: float = 1.0,
    reps: int = 5,
) -> QASMExportResult:
    """Export Trotter evolution circuit as OpenQASM 3."""
    n = K.shape[0]
    qc = _build_trotter_circuit(K, omega, t, reps)
    qasm_str = qasm3_dumps(qc)

    return QASMExportResult(
        qasm_string=qasm_str,
        n_qubits=n,
        gate_count=qc.size(),
        circuit_depth=qc.depth(),
        format_version="OpenQASM 3.0",
    )


def export_ansatz_qasm(
    K: np.ndarray,
    reps: int = 2,
) -> QASMExportResult:
    """Export K_nm-informed VQE ansatz as OpenQASM 3."""
    n = K.shape[0]
    ansatz = knm_to_ansatz(K, reps=reps)

    # Bind parameters to zeros for export
    params = np.zeros(ansatz.num_parameters)
    bound = ansatz.assign_parameters(params)
    qasm_str = qasm3_dumps(bound)

    return QASMExportResult(
        qasm_string=qasm_str,
        n_qubits=n,
        gate_count=bound.size(),
        circuit_depth=bound.depth(),
        format_version="OpenQASM 3.0",
    )


def export_measurement_qasm(
    K: np.ndarray,
    omega: np.ndarray,
    t: float = 1.0,
    reps: int = 5,
) -> QASMExportResult:
    """Export Trotter + measurement circuit as OpenQASM 3."""
    n = K.shape[0]
    qc = _build_trotter_circuit(K, omega, t, reps)
    qc.measure_all()
    qasm_str = qasm3_dumps(qc)

    return QASMExportResult(
        qasm_string=qasm_str,
        n_qubits=n,
        gate_count=qc.size(),
        circuit_depth=qc.depth(),
        format_version="OpenQASM 3.0",
    )
