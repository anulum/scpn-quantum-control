# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Varqite
"""Variational Quantum Imaginary Time Evolution (VarQITE).

VarQITE (McArdle et al., npj Quantum Information 5, 75 (2019))
finds the ground state via imaginary time evolution:

    |ψ(τ)> = e^{-Hτ} |ψ(0)> / ||e^{-Hτ} |ψ(0)>||

As τ → ∞, |ψ(τ)> → ground state (for non-orthogonal initial state).

The variational version uses McLachlan's principle:

    min ||d|ψ>/dτ + (H - <H>)|ψ>||²

giving the equation of motion:
    A × dθ/dτ = C

where A_ij = Re(<∂_i ψ|∂_j ψ>) and C_i = -Re(<∂_i ψ|(H - <H>)|ψ>). The state
derivatives ``∂_i ψ`` are evaluated exactly via the π-shift identity (see
:mod:`scpn_quantum_control.phase.variational_metric`), so A is the analytic
quantum geometric tensor of the ansatz rather than a finite-difference estimate.

Advantages over COBYLA VQE:
    - Guaranteed convergence to ground state (no local minima)
    - No optimizer hyperparameters
    - Natural stopping criterion (dθ/dτ → 0)

For the Kuramoto-XY system, VarQITE finds the maximum synchronisation
configuration without the barren plateau problem of VQE.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..bridge.knm_hamiltonian import knm_to_ansatz, knm_to_hamiltonian
from ..dense_budget import require_dense_allocation
from ..hardware.classical import classical_exact_diag
from .variational_metric import (
    analytic_state_derivatives,
    assert_single_parameter_rotations,
    imaginary_time_force,
    mclachlan_metric,
)

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]


@dataclass
class VarQITEResult:
    """VarQITE ground state result."""

    energy: float
    exact_energy: float
    relative_error_pct: float
    n_steps: int
    energy_history: list[float]
    converged: bool
    optimal_params: NDArray[np.float64]


def _qubits_from_state_length(state_length: int) -> int:
    """Infer qubit count from an exact statevector length."""
    if state_length < 1 or state_length & (state_length - 1):
        raise ValueError(f"statevector length must be a positive power of two, got {state_length}")
    return state_length.bit_length() - 1


def _state_evaluator(ansatz: QuantumCircuit) -> Callable[[FloatArray], ComplexArray]:
    """Return a callable mapping parameter values to the ansatz statevector."""

    def state_of(values: FloatArray) -> ComplexArray:
        assigned = ansatz.assign_parameters(values)
        return np.asarray(Statevector.from_instruction(assigned).data, dtype=np.complex128)

    return state_of


def _varqite_matrices(
    ansatz: QuantumCircuit,
    params: NDArray[np.float64],
    H_op: SparsePauliOp,
    *,
    max_dense_gib: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the VarQITE A matrix and C vector with exact state derivatives.

    A_ij = Re(<∂_i ψ|∂_j ψ>)
    C_i = -Re(<∂_i ψ|(H - <H>)|ψ>)

    The state derivatives ``∂_i ψ`` are evaluated exactly through the π-shift
    identity (:func:`...variational_metric.analytic_state_derivatives`); the metric
    A is therefore the analytic quantum geometric tensor of the ansatz, free of the
    finite-difference bias and step-size hyperparameter of a naive estimator.
    """
    ansatz_qubits = getattr(ansatz, "num_qubits", None)
    if ansatz_qubits is not None:
        require_dense_allocation(
            int(ansatz_qubits),
            rank=2,
            max_gib=max_dense_gib,
            label="VarQITE dense Hamiltonian",
        )

    assert_single_parameter_rotations(ansatz)
    state_of = _state_evaluator(ansatz)
    theta = np.asarray(params, dtype=np.float64)
    psi_0 = state_of(theta)
    if ansatz_qubits is None:
        require_dense_allocation(
            _qubits_from_state_length(len(psi_0)),
            rank=2,
            max_gib=max_dense_gib,
            label="VarQITE dense Hamiltonian",
        )

    H_mat = H_op.to_matrix()
    if hasattr(H_mat, "toarray"):
        H_mat = H_mat.toarray()
    e_mean = float(np.real(psi_0.conj() @ H_mat @ psi_0))
    H_shifted = H_mat - e_mean * np.eye(len(psi_0))

    dpsi = analytic_state_derivatives(state_of, theta)
    A = mclachlan_metric(dpsi)
    C = imaginary_time_force(dpsi, H_shifted @ psi_0)
    return A, C


def varqite_ground_state(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    tau_total: float = 3.0,
    n_steps: int = 30,
    ansatz_reps: int = 2,
    convergence_threshold: float = 1e-5,
    seed: int | None = None,
    *,
    max_dense_gib: float | None = None,
) -> VarQITEResult:
    """Find ground state via VarQITE.

    Args:
        K: coupling matrix
        omega: natural frequencies
        tau_total: total imaginary time
        n_steps: number of ITE steps
        ansatz_reps: ansatz repetitions
        convergence_threshold: stop when |ΔE| < threshold
        seed: random seed
        max_dense_gib: dense exact-simulation budget for Hamiltonian,
            statevector, and finite-difference derivative arrays
    """
    n = K.shape[0]
    require_dense_allocation(
        n,
        rank=2,
        max_gib=max_dense_gib,
        label="VarQITE dense Hamiltonian",
    )
    H_op = knm_to_hamiltonian(K, omega)
    ansatz = knm_to_ansatz(K, reps=ansatz_reps)
    n_params = ansatz.num_parameters

    rng = np.random.default_rng(seed)
    params = rng.normal(0, 0.1, size=n_params)
    dtau = tau_total / n_steps

    energy_history: list[float] = []
    converged = False

    for step in range(n_steps):
        sv = Statevector.from_instruction(ansatz.assign_parameters(params))
        energy = float(sv.expectation_value(H_op).real)
        energy_history.append(energy)

        if step > 0 and abs(energy_history[-1] - energy_history[-2]) < convergence_threshold:
            converged = True
            break

        A, C = _varqite_matrices(ansatz, params, H_op, max_dense_gib=max_dense_gib)
        reg = 1e-6 * np.eye(n_params)
        dtheta = np.linalg.solve(A + reg, C) * dtau
        params = params + dtheta

    # Final energy
    sv_final = Statevector.from_instruction(ansatz.assign_parameters(params))
    final_energy = float(sv_final.expectation_value(H_op).real)
    energy_history.append(final_energy)

    exact = classical_exact_diag(n, K=K, omega=omega)
    exact_e = exact["ground_energy"]
    rel_err = abs(final_energy - exact_e) / max(abs(exact_e), 1e-15) * 100

    return VarQITEResult(
        energy=final_energy,
        exact_energy=exact_e,
        relative_error_pct=rel_err,
        n_steps=len(energy_history) - 1,
        energy_history=energy_history,
        converged=converged,
        optimal_params=params,
    )
