"""Ansatz benchmark: compare Knm-informed, TwoLocal, and EfficientSU2.

Runs VQE with each ansatz on the XY Kuramoto Hamiltonian and reports
energy, parameter count, and convergence.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from ..bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_hamiltonian,
)


def _vqe_energy(ansatz, hamiltonian, maxiter, seed=42):
    """Run COBYLA VQE, return (energy, n_evals, history)."""
    history = []

    def cost(params):
        bound = ansatz.assign_parameters(params)
        sv = Statevector.from_instruction(bound)
        e = float(sv.expectation_value(hamiltonian).real)
        history.append(e)
        return e

    x0 = np.random.default_rng(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
    res = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter})
    return float(res.fun), res.nfev, history


def benchmark_ansatz(
    K: np.ndarray,
    omega: np.ndarray,
    ansatz_name: str,
    maxiter: int = 200,
    reps: int = 2,
) -> dict:
    """Benchmark a single ansatz by name: 'knm_informed', 'two_local', 'efficient_su2'."""
    n = len(omega)
    H = knm_to_hamiltonian(K, omega)

    if ansatz_name == "knm_informed":
        ansatz = knm_to_ansatz(K, reps=reps)
    elif ansatz_name == "two_local":
        ansatz = TwoLocal(n, ["ry", "rz"], "cz", reps=reps)
    elif ansatz_name == "efficient_su2":
        ansatz = EfficientSU2(n, reps=reps)
    else:
        raise ValueError(f"Unknown ansatz: {ansatz_name}")

    energy, n_evals, history = _vqe_energy(ansatz, H, maxiter)
    return {
        "ansatz": ansatz_name,
        "n_qubits": n,
        "n_params": ansatz.num_parameters,
        "energy": energy,
        "n_evals": n_evals,
        "history": history,
        "reps": reps,
    }


def run_ansatz_benchmark(
    n_qubits: int = 4,
    maxiter: int = 200,
    reps: int = 2,
) -> list[dict]:
    """Benchmark all three ansatze on n_qubits oscillators."""
    K = build_knm_paper27(L=n_qubits)
    omega = OMEGA_N_16[:n_qubits]
    results = []
    for name in ["knm_informed", "two_local", "efficient_su2"]:
        results.append(benchmark_ansatz(K, omega, name, maxiter=maxiter, reps=reps))
    return results
