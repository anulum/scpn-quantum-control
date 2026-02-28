"""Entanglement percolation analysis on K_nm coupling graph.

Determines which oscillator pairs are entangled above threshold
and usable as QKD channels. Uses reduced density matrix concurrence
as the entanglement measure.

Refs:
- arXiv:2602.10189 (2026), entanglement percolation in random quantum networks
- Communications Physics 2025, counter-intuitive low percolation threshold
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

from ..phase.phase_vqe import PhaseVQE


def concurrence_map(K: np.ndarray, omega: np.ndarray, maxiter: int = 100) -> np.ndarray:
    """Compute pairwise concurrence from ground state reduced density matrices.

    C(i,j) = max(0, sqrt(e1) - sqrt(e2) - sqrt(e3) - sqrt(e4))
    where e_k are eigenvalues of rho * (Y⊗Y) rho* (Y⊗Y) in decreasing order.

    Returns n×n symmetric matrix with concurrence values.
    """
    n = K.shape[0]
    vqe = PhaseVQE(K, omega, ansatz_reps=2)
    result = vqe.solve(maxiter=maxiter)
    sv = Statevector.from_instruction(vqe.ansatz.assign_parameters(result["optimal_params"]))

    rho_full = DensityMatrix(sv)
    conc = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Trace out all qubits except i and j
            keep = [i, j]
            trace_out = [q for q in range(n) if q not in keep]
            rho_ij = partial_trace(rho_full, trace_out)
            c = _concurrence_2qubit(rho_ij)
            conc[i, j] = c
            conc[j, i] = c

    return conc


def _concurrence_2qubit(rho: DensityMatrix) -> float:
    """Wootters concurrence for a 2-qubit density matrix."""
    rho_mat = np.array(rho.data)
    sigma_y = np.array([[0, -1j], [1j, 0]])
    yy = np.kron(sigma_y, sigma_y)
    rho_tilde = yy @ rho_mat.conj() @ yy
    product = rho_mat @ rho_tilde
    eigvals = np.sort(np.abs(np.linalg.eigvals(product)))[::-1]
    sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0))
    return float(max(0, sqrt_eigvals[0] - sqrt_eigvals[1] - sqrt_eigvals[2] - sqrt_eigvals[3]))


def percolation_threshold(K: np.ndarray) -> float:
    """Minimum K_nm value for end-to-end entanglement.

    Estimated from the Fiedler value of the coupling graph:
    when lambda_1 > 0, the graph is connected and entanglement percolates.
    Returns the smallest nonzero off-diagonal K_nm entry that keeps
    the graph connected.
    """
    off_diag = K[np.triu_indices_from(K, k=1)]
    off_diag = off_diag[off_diag > 0]
    if len(off_diag) == 0:
        return float("inf")

    sorted_vals = np.sort(off_diag)

    # Binary search: smallest threshold that keeps graph connected
    for threshold in sorted_vals:
        adj = (threshold <= K).astype(float)
        np.fill_diagonal(adj, 0)
        D = np.diag(adj.sum(axis=1))
        L = D - adj
        eigvals = np.sort(np.linalg.eigvalsh(L))
        if eigvals[1] > 1e-10:
            return float(threshold)

    return float(sorted_vals[0])


def active_channel_graph(K: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """List of above-threshold entangled pairs usable as QKD channels.

    Returns list of (i, j, K_ij) tuples.
    """
    n = K.shape[0]
    channels = []
    for i in range(n):
        for j in range(i + 1, n):
            if K[i, j] >= threshold:
                channels.append((i, j, float(K[i, j])))
    return channels


def key_rate_per_channel(conc_map: np.ndarray) -> np.ndarray:
    """Devetak-Winter key rate estimate for each link.

    r(i,j) = max(0, 1 - h(e(C))) where e = (1 - sqrt(1 - C^2)) / 2
    and h is binary entropy.
    """
    n = conc_map.shape[0]
    rates = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            C = conc_map[i, j]
            if C < 1e-10:
                continue
            # Concurrence → QBER
            e = (1 - np.sqrt(max(0, 1 - C**2))) / 2
            # Binary entropy
            if e < 1e-15 or e > 1 - 1e-15:
                h_e = 0.0
            else:
                h_e = -e * np.log2(e) - (1 - e) * np.log2(1 - e)
            r = max(0.0, 1.0 - h_e)
            rates[i, j] = r
            rates[j, i] = r
    return rates
