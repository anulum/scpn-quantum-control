# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Experiment Helpers
"""Shared helper functions used by experiment sub-modules."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from ..bridge.knm_hamiltonian import knm_to_hamiltonian


def _build_evo_base(n, K, omega, t, trotter_reps, trotter_order=1):
    """Build evolution circuit without measurement gates."""
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(float(omega[i]) % (2 * np.pi), i)
    H = knm_to_hamiltonian(K, omega)
    if trotter_order >= 2:
        synthesis = SuzukiTrotter(order=trotter_order, reps=trotter_reps)
    else:
        synthesis = LieTrotter(reps=trotter_reps)
    evo = PauliEvolutionGate(H, time=t, synthesis=synthesis)
    qc.append(evo, range(n))
    return qc


def _build_xyz_circuits(base_circuit, n):
    """Build 3 copies of base_circuit measuring in Z, X, Y bases.

    X-basis: H before measurement.
    Y-basis: Sdg then H before measurement.
    Returns (z_circuit, x_circuit, y_circuit).
    """
    qc_z = base_circuit.copy()
    qc_z.measure_all()

    qc_x = base_circuit.copy()
    for q in range(n):
        qc_x.h(q)
    qc_x.measure_all()

    qc_y = base_circuit.copy()
    for q in range(n):
        qc_y.sdg(q)
        qc_y.h(q)
    qc_y.measure_all()

    return qc_z, qc_x, qc_y


def _expectation_per_qubit(counts, n_qubits):
    """Compute per-qubit <Z> and shot-noise standard deviation.

    Returns:
        (exp_vals, std_vals) where std = sqrt((1 - exp^2) / N_shots).
    """
    total = sum(counts.values())
    exp_vals = np.zeros(n_qubits)
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        for q in range(min(n_qubits, len(bits))):
            bit = int(bits[-(q + 1)])
            exp_vals[q] += (1 - 2 * bit) * count
    exp_vals /= total
    std_vals: np.ndarray = np.sqrt(np.maximum(1.0 - exp_vals**2, 0.0) / total)
    return exp_vals, std_vals


def _R_from_xyz(z_counts, x_counts, y_counts, n_qubits):
    """Compute Kuramoto order parameter R from X, Y, Z basis measurements.

    Returns:
        (R, R_std, exp_x, exp_y, exp_z, std_x, std_y, std_z)
    """
    exp_x, std_x = _expectation_per_qubit(x_counts, n_qubits)
    exp_y, std_y = _expectation_per_qubit(y_counts, n_qubits)
    exp_z, std_z = _expectation_per_qubit(z_counts, n_qubits)
    z_complex = np.mean(exp_x + 1j * exp_y)
    R = float(abs(z_complex))
    # Propagated uncertainty: delta_R ≈ sqrt(sum(std_x^2 + std_y^2)) / N
    R_std = float(np.sqrt(np.mean(std_x**2 + std_y**2)) / np.sqrt(n_qubits))
    return R, R_std, exp_x, exp_y, exp_z, std_x, std_y, std_z


def _qaoa_cost_from_counts(counts: dict, cost_ham: SparsePauliOp, n_qubits: int) -> float:
    """Evaluate QAOA cost Hamiltonian (diagonal in Z) from counts."""
    total = sum(counts.values())
    energy = 0.0
    for pauli, coeff in zip(cost_ham.paulis, cost_ham.coeffs):
        label = str(pauli)
        exp_val = 0.0
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")
            sign = 1
            for q in range(n_qubits):
                p = label[-(q + 1)]
                if p == "Z":
                    bit = int(bits[-(q + 1)])
                    sign *= (-1) ** bit
                elif p in ("X", "Y"):
                    sign = 0
                    break
            exp_val += sign * count
        exp_val /= total
        energy += float(coeff.real) * exp_val
    return energy


def _correlator_from_counts(counts: dict, qubit_a: int, qubit_b: int) -> float:
    """Compute <A B> from 2-qubit marginal of multi-qubit counts.

    E(A,B) = (N_same - N_diff) / N_total where same/diff refers to
    the measurement outcomes of qubits a and b.
    """
    n_same = 0
    n_diff = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        bit_a = int(bits[-(qubit_a + 1)])
        bit_b = int(bits[-(qubit_b + 1)])
        if bit_a == bit_b:
            n_same += count
        else:
            n_diff += count
    total = n_same + n_diff
    if total == 0:
        return 0.0
    return (n_same - n_diff) / total
