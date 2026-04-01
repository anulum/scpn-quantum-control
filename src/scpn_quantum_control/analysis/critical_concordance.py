# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Critical Concordance
"""Critical point concordance: multiple probes → same K_c.

At the BKT synchronization critical point K_c:
- Order parameter R jumps
- QFI diverges (spectral gap closes)
- Entanglement graph percolates (Fiedler eigenvalue > 0)
- Spectral gap reaches minimum

If all probes agree on K_c, the critical point determination is
robust and independent of the chosen observable.

This module runs a unified K-scan and extracts K_c from each probe,
then computes concordance (agreement) metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian
from .entanglement_percolation import concurrence_map_exact, fiedler_eigenvalue
from .qfi_criticality import qfi_single_coupling


@dataclass
class ConcordanceResult:
    """Unified critical point analysis from multiple probes."""

    k_values: np.ndarray
    R_values: np.ndarray
    qfi_values: np.ndarray
    gap_values: np.ndarray
    fiedler_values: np.ndarray
    n_entangled_pairs: np.ndarray

    k_c_from_gap: float | None  # K where gap is minimum
    k_c_from_qfi: float | None  # K where QFI peaks
    k_c_from_fiedler: float | None  # K where Fiedler first > 0
    k_c_from_R_deriv: float | None  # K where dR/dK is maximum

    concordance_spread: float | None  # std of all K_c estimates


def _R_from_ground_state(psi: np.ndarray, n: int) -> float:
    """Order parameter R from ground state."""
    sv = Statevector(np.ascontiguousarray(psi))
    phases = np.zeros(n)
    for k in range(n):
        x_str = ["I"] * n
        x_str[k] = "X"
        y_str = ["I"] * n
        y_str[k] = "Y"
        ex = float(sv.expectation_value(SparsePauliOp("".join(reversed(x_str)))).real)
        ey = float(sv.expectation_value(SparsePauliOp("".join(reversed(y_str)))).real)
        phases[k] = np.arctan2(ey, ex)
    return float(abs(np.mean(np.exp(1j * phases))))


def critical_concordance(
    omega: np.ndarray,
    K_topology: np.ndarray,
    k_range: np.ndarray | None = None,
    concurrence_threshold: float = 1e-4,
) -> ConcordanceResult:
    """Run all probes across K and extract K_c from each.

    K_topology: normalized coupling matrix (max=1), scaled by k_range values.
    """
    if k_range is None:
        k_range = np.linspace(0.5, 5.0, 15)

    n = len(omega)
    n_k = len(k_range)

    R_vals = np.zeros(n_k)
    qfi_vals = np.zeros(n_k)
    gap_vals = np.zeros(n_k)
    fiedler_vals = np.zeros(n_k)
    n_pairs = np.zeros(n_k)

    for idx, kb in enumerate(k_range):
        K = float(kb) * K_topology
        knm_to_hamiltonian(K, omega)
        H_mat = knm_to_dense_matrix(K, omega)
        eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
        psi0 = eigenvectors[:, 0]

        # R
        R_vals[idx] = _R_from_ground_state(psi0, n)

        # QFI + gap
        mq, gap, _tq = qfi_single_coupling(K, omega)
        qfi_vals[idx] = mq
        gap_vals[idx] = gap

        # Entanglement percolation
        cmap = concurrence_map_exact(psi0, n, threshold=concurrence_threshold)
        adj = (cmap > concurrence_threshold).astype(float)
        fiedler_vals[idx] = fiedler_eigenvalue(adj)
        upper = cmap[np.triu_indices(n, k=1)]
        n_pairs[idx] = float(np.sum(upper > concurrence_threshold))

    # Extract K_c from each probe
    k_c_gap = float(k_range[int(np.argmin(gap_vals))])
    k_c_qfi = float(k_range[int(np.argmax(qfi_vals))]) if np.max(qfi_vals) > 0 else None

    # Fiedler: first K where percolation occurs
    k_c_fiedler = None
    for i, f in enumerate(fiedler_vals):
        if f > 1e-10:
            k_c_fiedler = float(k_range[i])
            break

    # R derivative: where dR/dK is maximum
    k_c_R = None
    if len(k_range) > 2:
        dR = np.gradient(R_vals, k_range)
        k_c_R = float(k_range[int(np.argmax(np.abs(dR)))])

    # Concordance: spread of K_c estimates
    estimates = [v for v in [k_c_gap, k_c_qfi, k_c_fiedler, k_c_R] if v is not None]
    spread = float(np.std(estimates)) if len(estimates) >= 2 else None

    return ConcordanceResult(
        k_values=k_range,
        R_values=R_vals,
        qfi_values=qfi_vals,
        gap_values=gap_vals,
        fiedler_values=fiedler_vals,
        n_entangled_pairs=n_pairs,
        k_c_from_gap=k_c_gap,
        k_c_from_qfi=k_c_qfi,
        k_c_from_fiedler=k_c_fiedler,
        k_c_from_R_deriv=k_c_R,
        concordance_spread=spread,
    )
