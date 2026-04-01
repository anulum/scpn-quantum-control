# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Pairing Correlator
"""Richardson-Gaudin pairing correlators in the synchronized state.

Kouchekian & Teodorescu (arXiv:2601.00113) proved that off-plane
perturbations of the Kuramoto model map to the semiclassical
Gaudin model, connecting oscillator synchronization to the
Richardson spin-pairing mechanism.

The pairing correlator ⟨S_i⁺ S_j⁻⟩ = ⟨(X_i + iY_i)(X_j - iY_j)⟩/4
detects this pairing structure. In the synchronized (paired) phase:
- Nearest-neighbor pairing should be enhanced
- Pairing should correlate with the coupling topology K_nm

This module computes pairing correlators from the XXZ ground state
and compares them across the XY→Heisenberg anisotropy range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..bridge.knm_hamiltonian import knm_to_dense_matrix


@dataclass
class PairingResult:
    """Pairing correlator analysis."""

    pairing_matrix: np.ndarray  # ⟨S_i⁺ S_j⁻⟩ for all pairs
    max_pairing: float
    mean_pairing: float
    pairing_topology_correlation: float  # correlation with K_nm
    n_qubits: int
    delta: float
    K_base: float


def _pairing_correlator(psi: np.ndarray, i: int, j: int, n: int) -> complex:
    """Compute ⟨S_i⁺ S_j⁻⟩ = ⟨(X_i+iY_i)(X_j-iY_j)⟩/4.

    S⁺S⁻ = (XX + YY + i(YX - XY))/4 = (XX + YY)/4 + i(YX - XY)/4
    """
    sv = Statevector(np.ascontiguousarray(psi))

    # Build XX, YY, YX, XY operators
    def _build_op(p1: str, p2: str) -> SparsePauliOp:
        s = ["I"] * n
        s[i] = p1
        s[j] = p2
        return SparsePauliOp("".join(reversed(s)))

    xx = float(sv.expectation_value(_build_op("X", "X")).real)
    yy = float(sv.expectation_value(_build_op("Y", "Y")).real)
    yx = float(sv.expectation_value(_build_op("Y", "X")).real)
    xy = float(sv.expectation_value(_build_op("X", "Y")).real)

    return complex((xx + yy) / 4.0, (yx - xy) / 4.0)


def pairing_map(
    omega: np.ndarray,
    K_topology: np.ndarray,
    K_base: float,
    delta: float = 0.0,
) -> PairingResult:
    """Compute full pairing correlator matrix from XXZ ground state."""
    n = len(omega)
    K = K_base * K_topology
    H_mat = knm_to_dense_matrix(K, omega, delta=delta)
    eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
    psi0 = eigenvectors[:, 0]

    pairing: np.ndarray = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(i + 1, n):
            c = _pairing_correlator(psi0, i, j, n)
            pairing[i, j] = c
            pairing[j, i] = c.conjugate()

    abs_pairing = np.abs(pairing)
    upper = abs_pairing[np.triu_indices(n, k=1)]

    # Correlation with coupling topology
    K_upper = K[np.triu_indices(n, k=1)]
    if np.std(K_upper) > 1e-15 and np.std(upper) > 1e-15:
        corr = float(np.corrcoef(K_upper, upper)[0, 1])
    else:
        corr = 0.0

    return PairingResult(
        pairing_matrix=pairing,
        max_pairing=float(np.max(upper)) if len(upper) > 0 else 0.0,
        mean_pairing=float(np.mean(upper)) if len(upper) > 0 else 0.0,
        pairing_topology_correlation=corr,
        n_qubits=n,
        delta=delta,
        K_base=K_base,
    )


def pairing_vs_anisotropy(
    omega: np.ndarray,
    K_topology: np.ndarray,
    K_base: float,
    delta_range: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Scan pairing strength across anisotropy Δ.

    Shows how off-plane coupling (Δ > 0) activates Richardson pairing.
    """
    if delta_range is None:
        delta_range = np.linspace(0.0, 1.0, 6)

    results: dict[str, list[float]] = {
        "delta": [],
        "max_pairing": [],
        "mean_pairing": [],
        "topology_correlation": [],
    }

    for d in delta_range:
        pr = pairing_map(omega, K_topology, K_base, float(d))
        results["delta"].append(float(d))
        results["max_pairing"].append(pr.max_pairing)
        results["mean_pairing"].append(pr.mean_pairing)
        results["topology_correlation"].append(pr.pairing_topology_correlation)

    return results
