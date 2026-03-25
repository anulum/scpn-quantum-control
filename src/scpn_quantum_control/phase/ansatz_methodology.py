# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coupling-topology-informed ansatz: formal benchmark methodology.

Compares three VQE ansatze on the Kuramoto-XY Hamiltonian:

1. **K_nm-informed**: CZ gates only where K[i,j] > threshold.
   Physics-motivated: entanglement mirrors the coupling topology.
2. **Hardware-efficient (TwoLocal)**: linear CZ chain, no physics.
3. **EfficientSU2**: full SU(2) rotations + linear CZ entanglement.

Metrics:
- Final energy (vs exact diag ground state)
- Energy error |E_VQE - E_exact| / |E_exact|
- Convergence speed (iterations to reach 99% of final energy)
- CNOT/CZ count (circuit cost)
- Parameter count
- Gradient variance (barren plateau indicator)

The key finding: K_nm-informed ansatz converges faster with fewer
entangling gates because it encodes the physical coupling structure
directly. This is the general methodology paper (PRX Quantum target):
"for any Hamiltonian with known coupling structure, place entanglers
only where coupling exists."

Prior art: QIDA (arXiv:2309.15287) does this for molecular QMI.
Nobody has generalized to dynamical systems / physical coupling matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.circuit.library import efficient_su2, n_local
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from ..bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_hamiltonian,
)
from ..hardware.classical import classical_exact_diag


@dataclass
class AnsatzBenchmarkResult:
    """Full benchmark result for one ansatz."""

    ansatz_name: str
    n_qubits: int
    n_params: int
    n_entangling_gates: int
    reps: int
    final_energy: float
    exact_energy: float
    relative_error: float
    n_evals: int
    convergence_iter_99pct: int
    energy_history: list[float]
    gradient_variance: float


def _count_entangling_gates(circuit) -> int:
    """Count two-qubit entangling gates (CX, CZ, ECR, RZZ, etc.)."""
    two_qubit_names = {"cx", "cz", "ecr", "rzz", "rxx", "ryy", "swap", "iswap", "cphase"}
    count = 0
    for inst in circuit.data:
        if (
            inst.operation.name in two_qubit_names
            or len(inst.qubits) >= 2
            and inst.operation.name not in {"barrier", "measure"}
        ):
            count += 1
    return count


def _gradient_variance(ansatz, hamiltonian, n_samples: int = 50, seed: int = 42) -> float:
    """Estimate gradient variance via parameter-shift sampling.

    Large variance → trainable landscape.
    Exponentially small variance → barren plateau.
    """
    rng = np.random.default_rng(seed)
    n_params = ansatz.num_parameters
    shift = np.pi / 2

    gradients = []
    for _ in range(n_samples):
        params = rng.uniform(-np.pi, np.pi, n_params)
        grad_norm_sq = 0.0
        for k in range(n_params):
            params_plus = params.copy()
            params_plus[k] += shift
            params_minus = params.copy()
            params_minus[k] -= shift

            sv_plus = Statevector.from_instruction(ansatz.assign_parameters(params_plus))
            sv_minus = Statevector.from_instruction(ansatz.assign_parameters(params_minus))
            e_plus = float(sv_plus.expectation_value(hamiltonian).real)
            e_minus = float(sv_minus.expectation_value(hamiltonian).real)
            grad_k = (e_plus - e_minus) / 2.0
            grad_norm_sq += grad_k**2
        gradients.append(grad_norm_sq)

    return float(np.var(gradients))


def _vqe_run(ansatz, hamiltonian, maxiter: int = 200, seed: int = 42):
    """COBYLA VQE returning (energy, n_evals, history)."""
    history: list[float] = []

    def cost(params):
        bound = ansatz.assign_parameters(params)
        sv = Statevector.from_instruction(bound)
        e = float(sv.expectation_value(hamiltonian).real)
        history.append(e)
        return e

    x0 = np.random.default_rng(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
    res = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter})
    return float(res.fun), res.nfev, history


def _convergence_99pct(history: list[float]) -> int:
    """Iteration at which energy first reaches 99% of final value."""
    if len(history) < 2:
        return 0
    final = history[-1]
    target = 0.99 * final if final < 0 else 1.01 * final
    for i, e in enumerate(history):
        if final < 0 and e <= target:
            return i
        if final >= 0 and e >= target:
            return i
    return len(history) - 1


def benchmark_single_ansatz(
    K: np.ndarray,
    omega: np.ndarray,
    ansatz_name: str,
    maxiter: int = 200,
    reps: int = 2,
    gradient_samples: int = 30,
    seed: int = 42,
) -> AnsatzBenchmarkResult:
    """Full benchmark for one ansatz on given K, omega."""
    n = len(omega)
    H = knm_to_hamiltonian(K, omega)

    if ansatz_name == "knm_informed":
        ansatz = knm_to_ansatz(K, reps=reps)
    elif ansatz_name == "two_local":
        ansatz = n_local(n, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=reps)
    elif ansatz_name == "efficient_su2":
        ansatz = efficient_su2(n, reps=reps)
    else:
        raise ValueError(f"Unknown ansatz: {ansatz_name}")

    exact = classical_exact_diag(n, K=K, omega=omega)
    e_exact = exact["ground_energy"]

    energy, n_evals, history = _vqe_run(ansatz, H, maxiter, seed)
    conv_iter = _convergence_99pct(history)
    n_ent = _count_entangling_gates(ansatz)

    grad_var = _gradient_variance(ansatz, H, n_samples=gradient_samples, seed=seed)

    rel_err = abs(energy - e_exact) / max(abs(e_exact), 1e-12)

    return AnsatzBenchmarkResult(
        ansatz_name=ansatz_name,
        n_qubits=n,
        n_params=ansatz.num_parameters,
        n_entangling_gates=n_ent,
        reps=reps,
        final_energy=energy,
        exact_energy=e_exact,
        relative_error=rel_err,
        n_evals=n_evals,
        convergence_iter_99pct=conv_iter,
        energy_history=history,
        gradient_variance=grad_var,
    )


def run_full_benchmark(
    system_sizes: list[int] | None = None,
    maxiter: int = 200,
    reps: int = 2,
    gradient_samples: int = 30,
    seed: int = 42,
) -> list[AnsatzBenchmarkResult]:
    """Run benchmark across system sizes and all three ansatze.

    Default sizes: [2, 3, 4, 5, 6] (kept small for statevector sim).
    """
    if system_sizes is None:
        system_sizes = [2, 3, 4, 5, 6]

    results = []
    for n in system_sizes:
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        for name in ["knm_informed", "two_local", "efficient_su2"]:
            r = benchmark_single_ansatz(
                K,
                omega,
                name,
                maxiter=maxiter,
                reps=reps,
                gradient_samples=gradient_samples,
                seed=seed,
            )
            results.append(r)
    return results


def summarize_benchmark(results: list[AnsatzBenchmarkResult]) -> dict:
    """Produce summary table from benchmark results."""
    rows = []
    for r in results:
        rows.append(
            {
                "ansatz": r.ansatz_name,
                "n_qubits": r.n_qubits,
                "n_params": r.n_params,
                "n_entangling": r.n_entangling_gates,
                "energy": r.final_energy,
                "exact": r.exact_energy,
                "rel_error": r.relative_error,
                "conv_iter": r.convergence_iter_99pct,
                "grad_var": r.gradient_variance,
            }
        )
    return {"results": rows}
