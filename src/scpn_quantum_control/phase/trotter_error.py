"""Trotter error analysis for the XY Kuramoto Hamiltonian.

Computes ||U_exact - U_trotter||_F for small systems (n <= 10 qubits)
using scipy.linalg.expm and Qiskit unitary simulation.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..bridge.knm_hamiltonian import knm_to_hamiltonian


def trotter_error_norm(
    K: np.ndarray,
    omega: np.ndarray,
    t: float,
    reps: int,
) -> float:
    """||U_exact - U_trotter||_F for a single (t, reps) pair.

    Raises ValueError for n > 10 (2^10 = 1024, matrices get large fast).
    """
    n = len(omega)
    if n > 10:
        raise ValueError(f"n={n} too large for exact unitary comparison (max 10)")

    H_op = knm_to_hamiltonian(K, omega)
    H_mat = np.array(H_op.to_matrix())

    U_exact = expm(-1j * H_mat * t)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import Operator
    from qiskit.synthesis import LieTrotter

    qc = QuantumCircuit(n)
    evo = PauliEvolutionGate(H_op, time=t, synthesis=LieTrotter(reps=reps))
    qc.append(evo, range(n))
    # Decompose so Operator sees individual gates, not the exact PauliEvolutionGate
    qc_decomposed = qc.decompose(reps=2)
    U_trotter = Operator(qc_decomposed).data

    return float(np.linalg.norm(U_exact - U_trotter, "fro"))


def trotter_error_sweep(
    K: np.ndarray,
    omega: np.ndarray,
    t_values: list[float],
    reps_values: list[int],
) -> dict:
    """2D sweep of Trotter error over (t, reps) pairs.

    Returns dict with keys 't_values', 'reps_values', 'errors' (2D list).
    """
    errors = []
    for t in t_values:
        row = []
        for reps in reps_values:
            row.append(trotter_error_norm(K, omega, t, reps))
        errors.append(row)

    return {
        "t_values": t_values,
        "reps_values": reps_values,
        "errors": errors,
    }
