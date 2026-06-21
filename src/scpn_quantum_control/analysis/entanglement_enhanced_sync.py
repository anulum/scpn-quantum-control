# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Entanglement Enhanced Sync
"""Entanglement-enhanced synchronization: does initial entanglement
change Kuramoto dynamics?

All prior work treats entanglement as an OUTPUT of synchronization
(Galve 2013, Roulet 2018, Nature Comms 2025). Nobody has tested
entanglement as an INPUT — preparing entangled initial states and
measuring the effect on synchronization dynamics.

Hamaguchi, Zheng & Yunger Halpern (Nature Physics, July 2025) proved
that entangled initial states reduce Trotter simulation error. We test
whether entanglement also changes the PHYSICAL synchronization behaviour
(speed, final R, critical coupling K_c).

Initial states compared:
    1. Product:  ⊗_i Ry(ω_i)|0⟩  (standard Kuramoto initialization)
    2. Bell:     (|00⟩+|11⟩)/√2 ⊗ (|00⟩+|11⟩)/√2  (pairwise entangled)
    3. GHZ:      (|0...0⟩+|1...1⟩)/√2  (globally entangled, even parity)
    4. W:        (|100..0⟩+|010..0⟩+...+|00..01⟩)/√N  (single-excitation)

All evolved under H_XY = -Σ K_ij(X_iX_j + Y_iY_j) - Σ(ω_i/2)Z_i.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian
from ..dense_budget import require_dense_allocation


class InitialState(Enum):
    """Supported initial-state families for entanglement synchronization tests."""

    PRODUCT = "product"
    BELL_PAIRS = "bell_pairs"
    GHZ = "ghz"
    W_STATE = "w_state"


@dataclass
class SyncTrajectory:
    """Time-evolution trajectory of order parameter R."""

    initial_state: str
    times: list[float]
    R_values: list[float]
    final_R: float
    n_qubits: int


def prepare_initial_state(
    n_qubits: int, state_type: InitialState, omega: NDArray[np.float64]
) -> QuantumCircuit:
    """Prepare the specified initial state on n qubits."""
    qc = QuantumCircuit(n_qubits)

    if state_type == InitialState.PRODUCT:
        for i in range(n_qubits):
            qc.ry(float(omega[i]) % (2 * np.pi), i)

    elif state_type == InitialState.BELL_PAIRS:
        for i in range(0, n_qubits - 1, 2):
            qc.h(i)
            qc.cx(i, i + 1)
        if n_qubits % 2 == 1:
            qc.ry(float(omega[-1]) % (2 * np.pi), n_qubits - 1)

    elif state_type == InitialState.GHZ:
        qc.h(0)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    elif state_type == InitialState.W_STATE:
        _prepare_w_state(qc, n_qubits)

    return qc


def _prepare_w_state(qc: QuantumCircuit, n: int) -> None:
    """Prepare |W_n⟩ = (|10..0⟩ + |01..0⟩ + ... + |00..1⟩)/√n.

    Uses the recursive Dicke state preparation (Bärtschi & Eidenbenz, 2019).
    """
    qc.x(0)
    for i in range(n - 1):
        theta = 2 * np.arccos(np.sqrt(1.0 / (n - i)))
        qc.cry(theta, i, i + 1)
        qc.cx(i + 1, i)


def simulate_sync_trajectory(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    state_type: InitialState,
    t_max: float = 2.0,
    n_steps: int = 20,
    *,
    max_dense_gib: float | None = None,
) -> SyncTrajectory:
    """Evolve under H_XY from a given initial state and measure R(t).

    Uses exact matrix exponential (no Trotter error).
    """
    n = K.shape[0]
    require_dense_allocation(
        n,
        dtype=np.complex128,
        rank=2,
        object_count=3,
        max_gib=max_dense_gib,
        label="entanglement-enhanced dense evolution workspace",
    )
    require_dense_allocation(
        n,
        dtype=np.complex128,
        rank=1,
        object_count=3,
        max_gib=max_dense_gib,
        label="entanglement-enhanced dense state workspace",
    )
    knm_to_hamiltonian(K, omega)
    H_mat = knm_to_dense_matrix(K, omega, max_dense_gib=max_dense_gib)

    init_qc = prepare_initial_state(n, state_type, omega)
    psi = np.array(Statevector.from_instruction(init_qc))

    times = np.linspace(0, t_max, n_steps + 1)
    dt = t_max / n_steps
    U_dt = _matrix_exp(-1j * H_mat * dt)
    R_values = []

    for step in range(n_steps + 1):
        if step > 0:
            psi = U_dt @ psi

        R = _state_order_parameter(psi, n)
        R_values.append(R)

    return SyncTrajectory(
        initial_state=state_type.value,
        times=list(times),
        R_values=R_values,
        final_R=R_values[-1],
        n_qubits=n,
    )


def _matrix_exp(A: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Matrix exponential via scipy."""
    from scipy.linalg import expm

    result: NDArray[np.complex128] = expm(A)
    return result


def _state_order_parameter(psi: NDArray[np.complex128], n_qubits: int) -> float:
    """Compute Kuramoto order parameter R from quantum state vector.

    R = |⟨exp(iθ)⟩| where θ_k = arctan2(⟨Y_k⟩, ⟨X_k⟩).
    """
    phases = np.zeros(n_qubits)
    for k in range(n_qubits):
        exp_x = _single_qubit_expectation(psi, n_qubits, k, "X")
        exp_y = _single_qubit_expectation(psi, n_qubits, k, "Y")
        phases[k] = np.arctan2(exp_y, exp_x)

    z = np.mean(np.exp(1j * phases))
    return float(np.abs(z))


def _single_qubit_expectation(
    psi: NDArray[np.complex128], n: int, qubit: int, basis: str
) -> float:
    """Compute ⟨ψ|O_k|ψ⟩ for single-qubit Pauli O on qubit k."""
    pauli_label = "I" * (n - qubit - 1) + basis + "I" * qubit
    op = SparsePauliOp.from_list([(pauli_label, 1.0)])
    mat = np.array(op.to_matrix())
    return float(np.real(psi.conj() @ mat @ psi))


def compare_all_initial_states(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    t_max: float = 2.0,
    n_steps: int = 20,
    *,
    max_dense_gib: float | None = None,
) -> dict[str, SyncTrajectory]:
    """Run synchronization dynamics for all four initial state types."""
    results = {}
    for state_type in InitialState:
        results[state_type.value] = simulate_sync_trajectory(
            K,
            omega,
            state_type,
            t_max,
            n_steps,
            max_dense_gib=max_dense_gib,
        )
    return results


def entanglement_advantage(results: dict[str, SyncTrajectory]) -> dict[str, Any]:
    """Quantify entanglement advantage from comparison results.

    Metrics:
    - Final R difference: R_entangled - R_product
    - Convergence speedup: steps to 90% of final R
    """
    product = results.get("product")
    if product is None:
        return {}

    advantages = {}
    for name, traj in results.items():
        if name == "product":
            continue

        delta_R = traj.final_R - product.final_R

        def _steps_to_90pct(r_vals: list[float]) -> int:
            target = 0.9 * r_vals[-1] if r_vals[-1] > 0.1 else 0.1
            for i, r in enumerate(r_vals):
                if r >= target:
                    return i
            return len(r_vals) - 1

        prod_conv = _steps_to_90pct(product.R_values)
        ent_conv = _steps_to_90pct(traj.R_values)
        speedup = prod_conv / max(ent_conv, 1)

        advantages[name] = {
            "delta_R_final": delta_R,
            "convergence_speedup": speedup,
            "product_steps_90pct": prod_conv,
            "entangled_steps_90pct": ent_conv,
        }

    return advantages
