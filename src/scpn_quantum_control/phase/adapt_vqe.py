# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Adapt Vqe
"""ADAPT-VQE for automatic ansatz construction.

ADAPT-VQE (Grimsley et al., Nature Comms 10, 3007 (2019)) grows the
ansatz one operator at a time by selecting the operator with the
largest energy gradient. This avoids barren plateaus because the
ansatz is always expressive enough for the current optimization
landscape but never over-parameterised.

For the Kuramoto-XY Hamiltonian, the operator pool consists of:
    - XX+YY exchange generators (from K_nm coupling topology)
    - Single-qubit Y rotations (local field terms)

At each ADAPT step:
    1. Compute gradient of energy w.r.t. each pool operator
    2. Select the operator with largest |gradient|
    3. Append it to the ansatz with a new variational parameter
    4. Re-optimise all parameters
    5. Check convergence (gradient norm < threshold)

Advantage over fixed ansatz: grows O(n) parameters instead of O(n²),
avoids barren plateaus (npj Quantum Information, 2023).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from ..bridge.knm_hamiltonian import knm_to_hamiltonian


@dataclass
class ADAPTResult:
    """ADAPT-VQE result."""

    energy: float
    n_iterations: int
    n_parameters: int
    gradient_norms: list[float]
    energies: list[float]
    selected_operators: list[int]  # indices into pool
    converged: bool


def _build_operator_pool(K: np.ndarray, n: int) -> list[SparsePauliOp]:
    """Build operator pool from K_nm coupling topology.

    Pool elements:
        - For each coupled pair (i,j): i(XX+YY) generator
        - For each qubit i: Y_i generator
    """
    pool: list[SparsePauliOp] = []

    # Exchange generators from coupling topology
    for i in range(n):
        for j in range(i + 1, n):
            if abs(K[i, j]) > 1e-10:
                # i(X_i X_j + Y_i Y_j) is the generator of XX+YY rotation
                xx = ["I"] * n
                xx[i] = "X"
                xx[j] = "X"
                yy = ["I"] * n
                yy[i] = "Y"
                yy[j] = "Y"
                op = SparsePauliOp(
                    ["".join(reversed(xx)), "".join(reversed(yy))],
                    coeffs=[1j, 1j],
                )
                pool.append(op)

    # Single-qubit Y generators
    for i in range(n):
        label = ["I"] * n
        label[i] = "Y"
        pool.append(SparsePauliOp("".join(reversed(label)), coeffs=[1j]))

    return pool


def _compute_gradient(
    sv: Statevector,
    hamiltonian: SparsePauliOp,
    pool: list[SparsePauliOp],
) -> np.ndarray:
    """Compute energy gradient for each pool operator.

    grad_k = 2 Re(<ψ|[H, A_k]|ψ>) where A_k is the pool operator.
    """
    gradients = np.zeros(len(pool))
    for k, op in enumerate(pool):
        commutator = (hamiltonian @ op - op @ hamiltonian).simplify()
        gradients[k] = float(sv.expectation_value(commutator).real)
    result: np.ndarray = gradients
    return result


def adapt_vqe(
    K: np.ndarray,
    omega: np.ndarray,
    max_iterations: int = 20,
    gradient_threshold: float = 1e-3,
    maxiter_opt: int = 100,
    seed: int | None = None,
) -> ADAPTResult:
    """Run ADAPT-VQE for the Kuramoto-XY Hamiltonian.

    Args:
        K: coupling matrix
        omega: natural frequencies
        max_iterations: max ADAPT steps
        gradient_threshold: convergence criterion on gradient norm
        maxiter_opt: max optimizer iterations per ADAPT step
        seed: random seed for optimizer
    """
    n = K.shape[0]
    H = knm_to_hamiltonian(K, omega)
    pool = _build_operator_pool(K, n)

    # Start from |0...0>
    params: list[float] = []
    selected: list[int] = []
    energies: list[float] = []
    grad_norms: list[float] = []
    converged = False

    rng = np.random.default_rng(seed)

    for _iteration in range(max_iterations):
        # Build current ansatz
        qc = _build_ansatz(n, pool, selected, params)
        sv = Statevector.from_instruction(qc)
        energy = float(sv.expectation_value(H).real)
        energies.append(energy)

        # Compute gradients
        gradients = _compute_gradient(sv, H, pool)
        grad_norm = float(np.max(np.abs(gradients)))
        grad_norms.append(grad_norm)

        if grad_norm < gradient_threshold:
            converged = True
            break

        # Select operator with largest gradient
        best_op = int(np.argmax(np.abs(gradients)))
        selected.append(best_op)
        params.append(0.01 * rng.standard_normal())

        # Re-optimise all parameters
        def cost_fn(p: np.ndarray) -> float:
            qc_trial = _build_ansatz(n, pool, selected, list(p))
            sv_trial = Statevector.from_instruction(qc_trial)
            return float(sv_trial.expectation_value(H).real)

        result = minimize(
            cost_fn,
            np.array(params),
            method="COBYLA",
            options={"maxiter": maxiter_opt},
        )
        params = list(result.x)

    # Final energy
    qc_final = _build_ansatz(n, pool, selected, params)
    sv_final = Statevector.from_instruction(qc_final)
    final_energy = float(sv_final.expectation_value(H).real)
    energies.append(final_energy)

    return ADAPTResult(
        energy=final_energy,
        n_iterations=len(selected),
        n_parameters=len(params),
        gradient_norms=grad_norms,
        energies=energies,
        selected_operators=selected,
        converged=converged,
    )


def _build_ansatz(
    n: int,
    pool: list[SparsePauliOp],
    selected: list[int],
    params: list[float],
) -> QuantumCircuit:
    """Build ansatz circuit from selected operators and parameters."""
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter

    qc = QuantumCircuit(n)
    synth = LieTrotter(reps=1)

    for idx, op_idx in enumerate(selected):
        op = pool[op_idx]
        # Remove the i factor for PauliEvolutionGate (it adds its own -i)
        # Our pool has iA, evolution gate does exp(-i * t * H), so
        # exp(-i * t * iA) = exp(t * A)
        real_op = SparsePauliOp(op.paulis, coeffs=op.coeffs / 1j)
        evo = PauliEvolutionGate(real_op, time=params[idx], synthesis=synth)
        qc.append(evo, range(n))

    return qc
