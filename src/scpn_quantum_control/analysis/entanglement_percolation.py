# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Entanglement Percolation
"""Entanglement percolation threshold vs synchronization K_c.

Testable conjecture: does the entanglement graph percolate at the
same coupling K_c where the Kuramoto order parameter R jumps?

Method:
1. For each K_base, build H(K), compute ground state exactly
2. Compute pairwise concurrence C(i,j) from reduced density matrices
3. Build entanglement graph: edge (i,j) exists iff C(i,j) > epsilon
4. Check graph connectivity via Fiedler eigenvalue (lambda_2 > 0)
5. K_perc = smallest K_base where entanglement graph is connected

If K_perc ≈ K_c, synchronization onset = entanglement connectivity onset.

Prior art: entanglement percolation (Comms Physics 2025, PRL 2025)
and Kuramoto K_c are separate fields. Nobody has compared them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

from ..bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian


@dataclass
class PercolationScanResult:
    """Result of scanning entanglement percolation across coupling."""

    k_values: np.ndarray
    fiedler_values: np.ndarray  # lambda_2 of entanglement Laplacian at each K
    max_concurrence: np.ndarray  # max pairwise concurrence at each K
    mean_concurrence: np.ndarray  # mean pairwise concurrence at each K
    n_entangled_pairs: np.ndarray  # number of entangled pairs at each K
    R_values: np.ndarray  # Kuramoto order parameter at each K
    k_percolation: float | None  # K where entanglement graph first percolates
    k_sync: float | None  # K where R first exceeds threshold


def _concurrence_2qubit(rho_mat: np.ndarray) -> float:
    """Wootters concurrence for a 2-qubit density matrix (raw numpy)."""
    sigma_y = np.array([[0, -1j], [1j, 0]])
    yy = np.kron(sigma_y, sigma_y)
    rho_tilde = yy @ rho_mat.conj() @ yy
    product = rho_mat @ rho_tilde
    eigvals = np.sort(np.abs(np.linalg.eigvals(product)))[::-1]
    sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0))
    return float(max(0, sqrt_eigvals[0] - sqrt_eigvals[1] - sqrt_eigvals[2] - sqrt_eigvals[3]))


def _order_parameter_from_state(psi: np.ndarray, n: int) -> float:
    """R = |mean(exp(i*theta_k))| from statevector."""
    sv = Statevector(np.ascontiguousarray(psi))
    phases = np.zeros(n)
    for k in range(n):
        x_op = ["I"] * n
        x_op[k] = "X"
        y_op = ["I"] * n
        y_op[k] = "Y"
        from qiskit.quantum_info import SparsePauliOp

        ex = float(sv.expectation_value(SparsePauliOp("".join(reversed(x_op)))).real)
        ey = float(sv.expectation_value(SparsePauliOp("".join(reversed(y_op)))).real)
        phases[k] = np.arctan2(ey, ex)
    return float(abs(np.mean(np.exp(1j * phases))))


def concurrence_map_exact(psi: np.ndarray, n: int, threshold: float = 1e-6) -> np.ndarray:
    """Pairwise concurrence from a pure state (exact, no VQE).

    Returns n×n symmetric matrix.
    """
    sv = Statevector(np.ascontiguousarray(psi))
    rho_full = DensityMatrix(sv)
    conc: np.ndarray = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            trace_out = [q for q in range(n) if q not in (i, j)]
            rho_ij = partial_trace(rho_full, trace_out)
            c = _concurrence_2qubit(np.array(rho_ij.data))
            if c > threshold:
                conc[i, j] = c
                conc[j, i] = c

    return conc


def fiedler_eigenvalue(adj: np.ndarray) -> float:
    """Second-smallest eigenvalue of the graph Laplacian.

    lambda_2 > 0 iff graph is connected (entanglement percolates).
    """
    D = np.diag(adj.sum(axis=1))
    L = D - adj
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1]) if len(eigvals) > 1 else 0.0


def percolation_scan(
    omega: np.ndarray,
    K_topology: np.ndarray,
    k_range: np.ndarray | None = None,
    concurrence_threshold: float = 1e-4,
    R_threshold: float = 0.5,
) -> PercolationScanResult:
    """Scan entanglement percolation and sync order parameter across K.

    K_topology: normalized coupling matrix (max=1), scaled by k_range values.
    concurrence_threshold: minimum concurrence to count a pair as entangled.
    R_threshold: R value to define sync onset.
    """
    if k_range is None:
        k_range = np.linspace(0.1, 5.0, 20)

    n = len(omega)
    n_k = len(k_range)
    fiedler = np.zeros(n_k)
    max_conc = np.zeros(n_k)
    mean_conc = np.zeros(n_k)
    n_pairs = np.zeros(n_k)
    R_vals = np.zeros(n_k)

    k_perc = None
    k_sync = None

    for idx, kb in enumerate(k_range):
        K = float(kb) * K_topology
        knm_to_hamiltonian(K, omega)
        H_mat = knm_to_dense_matrix(K, omega)
        eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
        psi0 = eigenvectors[:, 0]

        # Concurrence map
        cmap = concurrence_map_exact(psi0, n, threshold=concurrence_threshold)
        adj = (cmap > concurrence_threshold).astype(float)

        fiedler[idx] = fiedler_eigenvalue(adj)
        upper = cmap[np.triu_indices(n, k=1)]
        max_conc[idx] = float(np.max(upper)) if len(upper) > 0 else 0.0
        mean_conc[idx] = float(np.mean(upper[upper > 0])) if np.any(upper > 0) else 0.0
        n_pairs[idx] = float(np.sum(upper > concurrence_threshold))

        # Order parameter
        R_vals[idx] = _order_parameter_from_state(psi0, n)

        if k_perc is None and fiedler[idx] > 1e-10:
            k_perc = float(kb)
        if k_sync is None and R_vals[idx] > R_threshold:
            k_sync = float(kb)

    return PercolationScanResult(
        k_values=k_range,
        fiedler_values=fiedler,
        max_concurrence=max_conc,
        mean_concurrence=mean_conc,
        n_entangled_pairs=n_pairs,
        R_values=R_vals,
        k_percolation=k_perc,
        k_sync=k_sync,
    )
