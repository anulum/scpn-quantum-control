# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Trapped Ion
"""Representative trapped-ion noise model for cross-platform benchmarking.

Models all-to-all connectivity (no SWAP overhead) with depolarizing +
thermal relaxation on MS gates. Multiqubit transpilation uses a CX-basis
proxy for native MS/RXX-style entangling operations and must be explicitly
enabled by the caller. Calibration values are representative order-of-
magnitude QCCD benchmarks, not a specific device calibration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit import QuantumCircuit, transpile

if TYPE_CHECKING:
    from qiskit_aer.noise import NoiseModel

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
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

    model = NoiseModel()

    sq_relax = thermal_relaxation_error(t1_us, t2_us, SQ_GATE_TIME_US)
    model.add_all_qubit_quantum_error(sq_relax, ["ry", "rz"])

    tq_relax = thermal_relaxation_error(t1_us, t2_us, MS_GATE_TIME_US)
    tq_depol = depolarizing_error(ms_error, 2)
    tq_combined = tq_depol.compose(tq_relax.tensor(tq_relax))
    model.add_all_qubit_quantum_error(tq_combined, ["rxx", "cx"])

    return model


def _with_trapped_ion_metadata(
    circuit: QuantumCircuit,
    *,
    basis_model: str,
) -> QuantumCircuit:
    """Attach provenance metadata to a trapped-ion representative circuit."""
    circuit.metadata = dict(circuit.metadata or {})
    circuit.metadata["trapped_ion_basis_model"] = basis_model
    circuit.metadata["hardware_claim"] = "representative_noise_model_not_device_calibration"
    circuit.metadata["connectivity_model"] = "all_to_all_no_swap"
    return circuit


def _contains_multiqubit_instruction(circuit: QuantumCircuit) -> bool:
    """Return whether compilation needs an entangling-gate basis proxy."""
    return any(len(instruction.qubits) > 1 for instruction in circuit.data)


def transpile_for_trapped_ion(
    circuit: QuantumCircuit,
    *,
    allow_proxy_basis: bool = False,
) -> QuantumCircuit:
    """Transpile with all-to-all connectivity (no SWAP insertion).

    Multiqubit output is a {cx, ry, rz, sx, x, id} representative proxy for
    native trapped-ion MS/RXX-style gates, not a vendor-native compiler target.
    Callers must pass ``allow_proxy_basis=True`` to make that approximation
    explicit at call sites.
    """
    n = circuit.num_qubits
    has_multiqubit_instruction = _contains_multiqubit_instruction(circuit)
    if n < 2:
        return _with_trapped_ion_metadata(circuit.copy(), basis_model="single_qubit")
    if not has_multiqubit_instruction:
        return _with_trapped_ion_metadata(
            circuit.copy(),
            basis_model="no_entangling_proxy_required",
        )

    if not allow_proxy_basis:
        raise ValueError(
            "transpile_for_trapped_ion uses a CX-basis proxy for native "
            "trapped-ion MS/RXX gates; pass allow_proxy_basis=True when a "
            "representative proxy circuit is intended."
        )

    from qiskit.transpiler import CouplingMap

    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    result = transpile(
        circuit,
        basis_gates=["cx", "ry", "rz", "sx", "x", "id"],
        coupling_map=CouplingMap(edges),
        optimization_level=2,
    )
    return _with_trapped_ion_metadata(result, basis_model="cx_proxy_for_rxx")
