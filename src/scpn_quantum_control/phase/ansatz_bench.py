# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Ansatz Bench
"""Compare K_nm-informed and generic variational ansatz families.

The module runs local statevector VQE on the Kuramoto-XY Hamiltonian and
returns a small JSON-ready result row for each ansatz family. Rows are local
functional benchmark evidence; production performance claims require the
repository isolated-affinity benchmark gate.
"""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2, n_local
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from ..bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_hamiltonian,
)

FloatArray: TypeAlias = NDArray[np.float64]
AnsatzName: TypeAlias = Literal["knm_informed", "two_local", "efficient_su2"]


class AnsatzBenchmarkRow(TypedDict):
    """JSON-ready result row for one local ansatz VQE benchmark."""

    ansatz: AnsatzName
    n_qubits: int
    n_params: int
    energy: float
    n_evals: int
    history: list[float]
    reps: int


def _normalise_ansatz_name(ansatz_name: str) -> AnsatzName:
    """Return a supported ansatz family name or raise a fail-closed error."""
    if ansatz_name == "knm_informed":
        return "knm_informed"
    if ansatz_name == "two_local":
        return "two_local"
    if ansatz_name == "efficient_su2":
        return "efficient_su2"
    raise ValueError(f"Unknown ansatz: {ansatz_name}")


def _build_benchmark_ansatz(
    K: FloatArray,
    ansatz_name: AnsatzName,
    reps: int,
) -> QuantumCircuit:
    """Return the parameterised circuit for an ansatz benchmark family."""
    n = K.shape[0]
    if ansatz_name == "knm_informed":
        return knm_to_ansatz(K, reps=reps)
    if ansatz_name == "two_local":
        return n_local(n, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=reps)
    return efficient_su2(n, reps=reps)


def _vqe_energy(
    ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    maxiter: int,
    seed: int = 42,
) -> tuple[float, int, list[float]]:
    """Run local COBYLA VQE and return energy, evaluation count, and history."""
    history: list[float] = []

    def cost(params: FloatArray) -> float:
        bound = ansatz.assign_parameters(params)
        sv = Statevector.from_instruction(bound)
        e = float(sv.expectation_value(hamiltonian).real)
        history.append(e)
        return e

    x0 = np.random.default_rng(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
    res = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter})
    return float(res.fun), int(res.nfev), history


def benchmark_ansatz(
    K: FloatArray,
    omega: FloatArray,
    ansatz_name: str,
    maxiter: int = 200,
    reps: int = 2,
) -> AnsatzBenchmarkRow:
    """Benchmark one ansatz family on a Kuramoto-XY Hamiltonian.

    Parameters
    ----------
    K:
        Real ``(n, n)`` K_nm coupling matrix used for the Hamiltonian and, for
        the ``"knm_informed"`` family, the entanglement topology.
    omega:
        Real ``(n,)`` natural-frequency vector in the same qubit order as ``K``.
    ansatz_name:
        Ansatz family to evaluate: ``"knm_informed"``, ``"two_local"``, or
        ``"efficient_su2"``.
    maxiter:
        Maximum COBYLA function evaluations requested from SciPy.
    reps:
        Number of ansatz repetitions/layers.

    Returns
    -------
    AnsatzBenchmarkRow
        JSON-ready row with energy, evaluation count, parameter count, and
        optimisation history.
    """
    family = _normalise_ansatz_name(ansatz_name)
    n = len(omega)
    H = knm_to_hamiltonian(K, omega)
    ansatz = _build_benchmark_ansatz(K, family, reps)

    energy, n_evals, history = _vqe_energy(ansatz, H, maxiter)
    return {
        "ansatz": family,
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
) -> list[AnsatzBenchmarkRow]:
    """Benchmark all supported ansatz families for a K_nm oscillator subset.

    Parameters
    ----------
    n_qubits:
        Number of leading Paper-27 oscillators to include.
    maxiter:
        Maximum COBYLA function evaluations per ansatz family.
    reps:
        Number of circuit repetitions/layers per ansatz family.

    Returns
    -------
    list[AnsatzBenchmarkRow]
        Rows ordered as K_nm-informed, TwoLocal, and EfficientSU2.
    """
    K = build_knm_paper27(L=n_qubits)
    omega = OMEGA_N_16[:n_qubits]
    results: list[AnsatzBenchmarkRow] = []
    for name in ("knm_informed", "two_local", "efficient_su2"):
        results.append(benchmark_ansatz(K, omega, name, maxiter=maxiter, reps=reps))
    return results
