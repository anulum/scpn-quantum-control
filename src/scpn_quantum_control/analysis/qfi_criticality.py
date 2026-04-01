# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Qfi Criticality
"""QFI divergence at the synchronization critical point.

Scans Quantum Fisher Information across coupling strength K_base.
At the BKT critical point K_c, the spectral gap closes and QFI
should diverge as ~ 1/gap^2 (from the spectral sum rule).

This means the ground state at K_c is an optimal quantum sensor
for coupling strength: the synchronization critical point is also
the metrological sweet spot.

Prior art: Ma et al. (2014) did QFI-BKT on XXZ chain via
renormalization group. Nobody has done QFI for K_ij coupling
parameters of the XY/Kuramoto Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian


@dataclass
class QFICriticalityResult:
    """QFI scan across coupling strength."""

    k_values: np.ndarray
    max_qfi: np.ndarray  # max diagonal QFI element at each K
    spectral_gap: np.ndarray  # gap at each K
    total_qfi: np.ndarray  # trace of QFI matrix at each K
    peak_k: float  # K_base where QFI peaks
    peak_qfi: float  # max QFI value at peak


def qfi_single_coupling(
    K: np.ndarray,
    omega: np.ndarray,
) -> tuple[float, float, float]:
    """Compute max QFI diagonal, spectral gap, and QFI trace for given K.

    Returns (max_qfi_diag, gap, trace_qfi).
    """
    n = len(omega)
    knm_to_hamiltonian(K, omega)
    H_mat = knm_to_dense_matrix(K, omega)

    eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
    E0 = eigenvalues[0]
    psi0 = eigenvectors[:, 0]
    gap = float(eigenvalues[1] - E0)

    # Coupling pairs
    from qiskit.quantum_info import SparsePauliOp

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if abs(K[i, j]) > 1e-10]
    if not pairs:
        return 0.0, gap, 0.0

    # Build dH/dK_ij operators
    V_mats = []
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
        V_mats.append(V)

    # QFI diagonal: F_aa = 4 * sum_{m!=0} |<m|V_a|0>|^2 / (Em - E0)^2
    n_pairs = len(pairs)
    qfi_diag = np.zeros(n_pairs)
    dim = len(eigenvalues)

    for a in range(n_pairs):
        V_psi0 = V_mats[a] @ psi0
        val = 0.0
        for m in range(1, dim):
            denom = (eigenvalues[m] - E0) ** 2
            if denom < 1e-30:
                continue
            overlap = abs(eigenvectors[:, m].conj() @ V_psi0) ** 2
            val += overlap / denom
        qfi_diag[a] = 4.0 * val

    return float(np.max(qfi_diag)), gap, float(np.sum(qfi_diag))


def qfi_vs_coupling(
    omega: np.ndarray,
    K_topology: np.ndarray,
    k_range: np.ndarray | None = None,
) -> QFICriticalityResult:
    """Scan QFI across coupling strength to find the metrological peak.

    K_topology: normalized coupling matrix (max entry = 1). Scaled by k_range values.
    At K_c, the spectral gap minimum and QFI maximum should coincide.
    """
    if k_range is None:
        k_range = np.linspace(0.1, 3.0, 20)

    n_k = len(k_range)
    max_qfi = np.zeros(n_k)
    gaps = np.zeros(n_k)
    total_qfi = np.zeros(n_k)

    for idx, kb in enumerate(k_range):
        K = float(kb) * K_topology
        mq, g, tq = qfi_single_coupling(K, omega)
        max_qfi[idx] = mq
        gaps[idx] = g
        total_qfi[idx] = tq

    peak_idx = int(np.argmax(max_qfi))

    return QFICriticalityResult(
        k_values=k_range,
        max_qfi=max_qfi,
        spectral_gap=gaps,
        total_qfi=total_qfi,
        peak_k=float(k_range[peak_idx]),
        peak_qfi=float(max_qfi[peak_idx]),
    )
