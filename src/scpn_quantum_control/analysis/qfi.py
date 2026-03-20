# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Quantum Fisher Information for coupling parameter estimation.

Computes the QFI matrix F_ij for the ground state |ψ_0(K)⟩ of the
XY Hamiltonian, giving the Cramér-Rao bound on estimation precision:

    Var(K̂_ij) ≥ 1 / (M × F_ij)

where M is the number of measurements.

The QFI is computed via the spectral decomposition:

    F_{ab} = 4 Σ_{m≠0} |⟨ψ_m|V_a|ψ_0⟩|² / (E_m - E_0)²

where V_a = X_iX_j + Y_iY_j is the derivative of H w.r.t. K_ij.

Reference: Braunstein & Caves, PRL 72, 3439 (1994).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from ..bridge.knm_hamiltonian import knm_to_hamiltonian


@dataclass
class QFIResult:
    """Quantum Fisher Information computation result."""

    qfi_matrix: np.ndarray
    coupling_pairs: list[tuple[int, int]]
    precision_bounds: np.ndarray
    spectral_gap: float
    n_qubits: int

    def precision_for(self, i: int, j: int, n_measurements: int = 10000) -> float:
        """Cramér-Rao lower bound on Var(K̂_ij) for given measurement budget."""
        idx = self.coupling_pairs.index((min(i, j), max(i, j)))
        f_ii = self.qfi_matrix[idx, idx]
        if f_ii < 1e-15:
            return float("inf")
        result: float = 1.0 / (n_measurements * f_ii)
        return result


def compute_qfi(
    K: np.ndarray,
    omega: np.ndarray,
    pairs: list[tuple[int, int]] | None = None,
) -> QFIResult:
    """Compute the Quantum Fisher Information matrix for coupling parameters.

    K: coupling matrix (symmetric, n×n)
    omega: natural frequencies (n,)
    pairs: which coupling pairs to compute QFI for (default: all nonzero K_ij)

    Returns QFIResult with the QFI matrix, coupling pairs, and precision bounds.
    """
    n = len(omega)
    H_op = knm_to_hamiltonian(K, omega)
    H_mat = H_op.to_matrix()

    eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
    E0 = eigenvalues[0]
    psi0 = eigenvectors[:, 0]
    gap = float(eigenvalues[1] - eigenvalues[0])

    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if abs(K[i, j]) > 1e-10]

    n_pairs = len(pairs)
    qfi = np.zeros((n_pairs, n_pairs))

    # Build derivative operators V_a = -(X_iX_j + Y_iY_j) for each pair
    V_ops: list[np.ndarray] = []
    for i, j in pairs:
        xx = ["I"] * n
        xx[i] = "X"
        xx[j] = "X"
        yy = ["I"] * n
        yy[i] = "Y"
        yy[j] = "Y"
        V = SparsePauliOp(
            ["".join(reversed(xx)), "".join(reversed(yy))],
            [-1.0, -1.0],
        ).to_matrix()
        V_ops.append(V)

    # Compute matrix elements ⟨ψ_m|V_a|ψ_0⟩ for all m and a
    dim = len(eigenvalues)
    mel: np.ndarray = np.zeros((dim, n_pairs), dtype=complex)
    for a in range(n_pairs):
        for m in range(dim):
            mel[m, a] = eigenvectors[:, m].conj() @ V_ops[a] @ psi0

    # QFI matrix: F_ab = 4 Σ_{m≠0} Re(⟨ψ_0|V_a|ψ_m⟩⟨ψ_m|V_b|ψ_0⟩) / (E_m - E_0)²
    for a in range(n_pairs):
        for b in range(a, n_pairs):
            val = 0.0
            for m in range(1, dim):
                denom = (eigenvalues[m] - E0) ** 2
                if denom < 1e-30:
                    continue
                val += np.real(mel[m, a].conj() * mel[m, b]) / denom
            qfi[a, b] = 4.0 * val
            qfi[b, a] = qfi[a, b]

    # Precision bounds: diagonal elements give per-parameter bounds
    diag = np.diag(qfi)
    precision = np.where(diag > 1e-15, 1.0 / diag, np.inf)

    return QFIResult(
        qfi_matrix=qfi,
        coupling_pairs=pairs,
        precision_bounds=precision,
        spectral_gap=gap,
        n_qubits=n,
    )


def qfi_gap_tradeoff(K: np.ndarray, omega: np.ndarray) -> dict:
    """Analyze the QFI-gap tradeoff for the coupling topology.

    Large spectral gap → robust identity but imprecise estimation.
    Small spectral gap → fragile identity but precise estimation.

    Returns dict with gap, max QFI, min precision bound, and tradeoff ratio.
    """
    result = compute_qfi(K, omega)
    max_qfi = float(np.max(np.diag(result.qfi_matrix)))
    finite_bounds = result.precision_bounds[np.isfinite(result.precision_bounds)]
    min_precision = float(np.min(finite_bounds)) if len(finite_bounds) > 0 else float("inf")

    return {
        "spectral_gap": result.spectral_gap,
        "max_qfi_diagonal": max_qfi,
        "min_precision_bound": min_precision,
        "tradeoff_ratio": result.spectral_gap * max_qfi,
        "n_pairs": len(result.coupling_pairs),
        "gap_squared_over_16": result.spectral_gap**2 / 16.0,
    }
