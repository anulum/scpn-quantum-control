# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Fast Sparse Evolution
"""High-performance sparse statevector engine.

Bypasses Qiskit circuit compilation and decomposition overheads to directly
simulate Trotter or exact evolution using sparse matrix-vector multiplication
(via scipy.sparse.linalg.expm_multiply).

Provides an order-of-magnitude speedup for large classical baselines (N >= 12)
and enables simulation of N=20 systems on standard hardware in seconds.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply

from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_xxz_hamiltonian


def fast_sparse_evolution(
    K: np.ndarray,
    omega: np.ndarray,
    t_total: float,
    n_steps: int,
    initial_state: np.ndarray | None = None,
    delta: float = 0.0,
) -> dict:
    """Evolve a statevector using fast sparse matrix exponentiation.

    Args:
        K: Coupling matrix.
        omega: Natural frequencies.
        t_total: Total evolution time.
        n_steps: Number of intermediate time steps to return.
        initial_state: Initial statevector (default: |0...0>).
        delta: XXZ anisotropy parameter (default: 0.0 = XY model).

    Returns:
        dict: Containing 'times' and 'states' (statevector at each step).
    """
    n = len(omega)
    dim = 1 << n

    if initial_state is None:
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1.0
    else:
        psi = np.ascontiguousarray(initial_state, dtype=complex)

    # Use Qiskit's SparsePauliOp to easily generate the sparse matrix
    H_op = knm_to_xxz_hamiltonian(K, omega, delta)
    H_sparse = H_op.to_matrix(sparse=True).tocsc()

    # The evolution operator is U = exp(-i H t)
    # We want to evolve by dt = t_total / n_steps
    dt = t_total / n_steps if n_steps > 0 else t_total

    A = -1j * dt * H_sparse

    times = np.linspace(0, t_total, n_steps + 1)
    states = [psi.copy()]

    current_psi = psi
    for _ in range(n_steps):
        current_psi = expm_multiply(A, current_psi)
        states.append(current_psi.copy())

    return {
        "times": times,
        "states": states,
        "n_qubits": n,
        "final_state": states[-1],
    }
