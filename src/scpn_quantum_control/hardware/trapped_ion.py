# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Trapped Ion
"""Synthetic trapped-ion noise model for cross-platform benchmarking.

Models all-to-all connectivity (no SWAP overhead) with depolarizing +
thermal relaxation on MS gates. Transpiles to {cx, ry, rz} as a proxy
for the native {rxx, ry, rz} basis. Calibration values are representative
order-of-magnitude QCCD benchmarks, not a specific device calibration.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

# QCCD trapped-ion calibration benchmarks
MS_ERROR = 0.005  # Molmer-Sorensen 2-qubit gate error
T1_US = 100_000.0  # T1 coherence, 100 ms
T2_US = 1_000.0  # T2 coherence, 1 ms
SQ_GATE_TIME_US = 10.0  # single-qubit gate time (us)
MS_GATE_TIME_US = 200.0  # MS gate time (us)


def trapped_ion_noise_model(
    ms_error: float = MS_ERROR,
    t1_us: float = T1_US,
    t2_us: float = T2_US,
) -> NoiseModel:
    """Noise model for QCCD trapped-ion hardware.

    Single-qubit gates: thermal relaxation.
    MS gates: depolarizing + thermal relaxation (same pattern as heron_r2).
    """
    model = NoiseModel()

    sq_relax = thermal_relaxation_error(t1_us, t2_us, SQ_GATE_TIME_US)
    model.add_all_qubit_quantum_error(sq_relax, ["ry", "rz"])

    tq_relax = thermal_relaxation_error(t1_us, t2_us, MS_GATE_TIME_US)
    tq_depol = depolarizing_error(ms_error, 2)
    tq_combined = tq_depol.compose(tq_relax.tensor(tq_relax))
    model.add_all_qubit_quantum_error(tq_combined, ["rxx", "cx"])

    return model


def transpile_for_trapped_ion(circuit: QuantumCircuit) -> QuantumCircuit:
    """Transpile with all-to-all connectivity (no SWAP insertion).

    Decomposes to {cx, ry, rz} which maps to native {rxx, ry, rz}.
    """
    n = circuit.num_qubits
    if n < 2:
        return circuit.copy()

    from qiskit.transpiler import CouplingMap

    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    return transpile(
        circuit,
        basis_gates=["cx", "ry", "rz", "sx", "x", "id"],
        coupling_map=CouplingMap(edges),
        optimization_level=2,
    )
