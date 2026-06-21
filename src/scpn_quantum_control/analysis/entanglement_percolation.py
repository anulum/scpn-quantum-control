# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Entanglement Percolation
"""Finite-size entanglement percolation diagnostics vs synchronisation proxies.

Testable finite-size question: does the entanglement graph percolate near the
same coupling region where the Kuramoto order parameter R crosses a selected
threshold?

Method:
1. For each K_base, build H(K), compute ground state exactly
2. Compute pairwise concurrence C(i,j) from reduced density matrices
3. Build entanglement graph: edge (i,j) exists iff C(i,j) > epsilon
4. Check graph connectivity via Fiedler eigenvalue (lambda_2 > 0)
5. K_perc = smallest K_base where entanglement graph is connected

If K_perc is close to the selected synchronisation proxy, the scan provides a
finite-size concordance observation rather than a thermodynamic proof.

Prior art includes entanglement percolation and Kuramoto criticality as
separate diagnostics. This module compares them for exact small-system
Kuramoto-XY ground states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

from ..bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian
from ..dense_budget import require_dense_allocation


@dataclass
class PercolationScanResult:
    """Result of scanning entanglement percolation across coupling."""

    k_values: NDArray[np.float64]
    fiedler_values: NDArray[np.float64]  # lambda_2 of entanglement Laplacian at each K
    max_concurrence: NDArray[np.float64]  # max pairwise concurrence at each K
    mean_concurrence: NDArray[np.float64]  # mean pairwise concurrence at each K
    n_entangled_pairs: NDArray[np.float64]  # number of entangled pairs at each K
    R_values: NDArray[np.float64]  # Kuramoto order parameter at each K
    k_percolation: float | None  # K where entanglement graph first percolates
    k_sync: float | None  # K where R first exceeds threshold


def _concurrence_2qubit(rho_mat: NDArray[np.complex128]) -> float:
    """Wootters concurrence for a 2-qubit density matrix (raw numpy)."""
    sigma_y = np.array([[0, -1j], [1j, 0]])
    yy = np.kron(sigma_y, sigma_y)
    rho_tilde = yy @ rho_mat.conj() @ yy
    product = rho_mat @ rho_tilde
    eigvals = np.sort(np.abs(np.linalg.eigvals(product)))[::-1]
    sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0))
    return float(max(0, sqrt_eigvals[0] - sqrt_eigvals[1] - sqrt_eigvals[2] - sqrt_eigvals[3]))


def _order_parameter_from_state(psi: NDArray[np.complex128], n: int) -> float:
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


def concurrence_map_exact(
    psi: NDArray[np.complex128], n: int, threshold: float = 1e-6
) -> NDArray[np.float64]:
    """Pairwise concurrence from a pure state (exact, no VQE).

    Returns n×n symmetric matrix.
    """
    sv = Statevector(np.ascontiguousarray(psi))
    rho_full = DensityMatrix(sv)
    conc: NDArray[np.float64] = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            trace_out = [q for q in range(n) if q not in (i, j)]
            rho_ij = partial_trace(rho_full, trace_out)
            c = _concurrence_2qubit(np.array(rho_ij.data))
            if c > threshold:
                conc[i, j] = c
                conc[j, i] = c

    return conc


def fiedler_eigenvalue(adj: NDArray[np.float64]) -> float:
    """Second-smallest eigenvalue of the graph Laplacian.

    lambda_2 > 0 iff graph is connected (entanglement percolates).
    """
    D = np.diag(adj.sum(axis=1))
    L = D - adj
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1]) if len(eigvals) > 1 else 0.0


def percolation_scan(
    omega: NDArray[np.float64],
    K_topology: NDArray[np.float64],
    k_range: NDArray[np.float64] | None = None,
    concurrence_threshold: float = 1e-4,
    R_threshold: float = 0.5,
    *,
    max_dense_gib: float | None = None,
) -> PercolationScanResult:
    """Scan entanglement percolation and sync order parameter across K.

    K_topology: normalized coupling matrix (max=1), scaled by k_range values.
    concurrence_threshold: minimum concurrence to count a pair as entangled.
    R_threshold: R value to define sync onset.
    """
    if k_range is None:
        k_range = np.linspace(0.1, 5.0, 20, dtype=np.float64)

    n = len(omega)
    n_k = len(k_range)
    require_dense_allocation(
        n,
        rank=2,
        object_count=2,
        max_gib=max_dense_gib,
        label="entanglement percolation dense eigensolver workspace",
    )
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
        H_mat = knm_to_dense_matrix(K, omega, max_dense_gib=max_dense_gib)
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
