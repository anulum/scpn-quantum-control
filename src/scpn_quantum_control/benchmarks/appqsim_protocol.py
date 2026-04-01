# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Appqsim Protocol
"""AppQSim benchmarking protocol: standardised metrics for publication.

Application-oriented Quantum Simulation benchmarking (AppQSim) defines
metrics that are meaningful for the application, not just the hardware:

    1. Application fidelity: how well does the simulation answer the
       physics question? (not just circuit fidelity)
    2. Time-to-solution: wall clock including compilation, queuing, shots
    3. Quantum resource efficiency: gates per unit of useful information
    4. Classical comparison: honest speedup vs best classical method

For the Kuramoto-XY system, the application metrics are:
    - Order parameter accuracy: |R_quantum - R_exact|
    - Energy accuracy: |E_VQE - E_exact| / |E_exact|
    - Correlation fidelity: ||C_quantum - C_exact||_F / n_pairs
    - Phase recovery fidelity: mean |theta_quantum - theta_exact|

Ref: Lubinski et al., QST 8, 024003 (2023) — benchmarking framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import Statevector

from ..bridge.knm_hamiltonian import knm_to_hamiltonian
from ..bridge.ssgf_adapter import quantum_to_ssgf_state
from ..hardware.classical import classical_exact_diag


@dataclass
class AppQSimMetrics:
    """Application-oriented benchmarking metrics."""

    order_parameter_error: float  # |R_quantum - R_exact|
    energy_relative_error_pct: float  # |E_q - E_ex| / |E_ex| × 100
    correlation_fidelity: float  # 1 - ||C_q - C_ex||_F / norm
    n_qubits: int
    n_gates: int
    circuit_depth: int


def _exact_order_parameter(K: np.ndarray, omega: np.ndarray) -> float:
    """R_global from exact ground state."""
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi = np.ascontiguousarray(exact["ground_state"])
    sv = Statevector(psi)
    state = quantum_to_ssgf_state(sv, n)
    return float(state["R_global"])


def _exact_correlators(K: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Exact <XX+YY> correlator matrix."""
    from ..analysis.hamiltonian_learning import measure_correlators

    return measure_correlators(K, omega)


def appqsim_benchmark(
    K: np.ndarray,
    omega: np.ndarray,
    circuit_sv: Statevector | None = None,
    n_gates: int = 0,
    circuit_depth: int = 0,
) -> AppQSimMetrics:
    """Run AppQSim benchmarking protocol.

    Args:
        K: coupling matrix
        omega: natural frequencies
        circuit_sv: statevector from quantum circuit (default: VQE)
        n_gates: gate count of the circuit
        circuit_depth: depth of the circuit
    """
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)

    # If no circuit SV provided, use VQE
    if circuit_sv is None:
        from ..phase.phase_vqe import PhaseVQE

        vqe = PhaseVQE(K, omega, ansatz_reps=2)
        vqe.solve(maxiter=100, seed=42)
        circuit_sv = Statevector(np.ascontiguousarray(vqe.ground_state()))
        ansatz = vqe.ansatz
        bound = ansatz.assign_parameters(vqe._optimal_params)
        n_gates = bound.size()
        circuit_depth = bound.depth()

    # Order parameter error
    r_exact = _exact_order_parameter(K, omega)
    state_q = quantum_to_ssgf_state(circuit_sv, n)
    r_quantum = state_q["R_global"]
    r_error = abs(r_quantum - r_exact)

    # Energy error
    e_exact = exact["ground_energy"]
    e_quantum = float(circuit_sv.expectation_value(knm_to_hamiltonian(K, omega)).real)
    e_rel_err = abs(e_quantum - e_exact) / max(abs(e_exact), 1e-15) * 100

    # Correlation fidelity
    C_exact = _exact_correlators(K, omega)
    from qiskit.quantum_info import SparsePauliOp

    C_quantum = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            xx = ["I"] * n
            xx[i] = "X"
            xx[j] = "X"
            yy = ["I"] * n
            yy[i] = "Y"
            yy[j] = "Y"
            op = SparsePauliOp(
                ["".join(reversed(xx)), "".join(reversed(yy))],
                coeffs=[1.0, 1.0],
            )
            val = float(circuit_sv.expectation_value(op).real)
            C_quantum[i, j] = C_quantum[j, i] = val

    frob_err = float(np.linalg.norm(C_quantum - C_exact, "fro"))
    frob_norm = float(np.linalg.norm(C_exact, "fro"))
    corr_fid = 1.0 - frob_err / max(frob_norm, 1e-15)

    return AppQSimMetrics(
        order_parameter_error=r_error,
        energy_relative_error_pct=e_rel_err,
        correlation_fidelity=corr_fid,
        n_qubits=n,
        n_gates=n_gates,
        circuit_depth=circuit_depth,
    )
