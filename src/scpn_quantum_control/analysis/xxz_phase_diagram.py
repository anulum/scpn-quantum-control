# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Xxz Phase Diagram
"""XXZ anisotropy phase diagram: XY→Heisenberg crossover.

Scans the anisotropy parameter Δ from 0 (XY model, standard
Kuramoto mapping) to 1 (isotropic Heisenberg, full S² dynamics
from Kouchekian-Teodorescu arXiv:2601.00113).

At each Δ, sweeps coupling K to find the critical point K_c(Δ).
The resulting K_c(Δ) curve is the anisotropy phase diagram.

Known limits:
- Δ = 0 (XY): BKT transition at K_c^XY
- Δ = 1 (Heisenberg): different universality class (SU(2) symmetric)
- Δ > 1 (Ising-like): first-order transition possible
- Δ < 0 (frustrated): possible spin liquid phases

The crossover from BKT (Δ=0) to Heisenberg (Δ=1) in the Kuramoto
context with heterogeneous frequencies is unstudied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_dense_matrix


@dataclass
class AnisotropyScanResult:
    """Result of scanning K at fixed Δ."""

    delta: float
    k_values: np.ndarray
    gaps: np.ndarray
    R_values: np.ndarray
    k_c_from_gap: float


@dataclass
class PhaseDiagramResult:
    """Full K_c(Δ) phase diagram."""

    delta_values: np.ndarray
    k_c_values: np.ndarray
    gap_min_values: np.ndarray
    scans: list[AnisotropyScanResult]


def _ground_state_properties(
    K: np.ndarray, omega: np.ndarray, delta: float
) -> tuple[float, float, float]:
    """Compute gap, R, and ground state energy for XXZ Hamiltonian.

    Returns (gap, R, E_gs).
    """
    H_mat = knm_to_dense_matrix(K, omega, delta=delta)
    eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
    gap = float(eigenvalues[1] - eigenvalues[0])
    psi0 = np.ascontiguousarray(eigenvectors[:, 0])

    # R from statevector
    from qiskit.quantum_info import SparsePauliOp, Statevector

    n = len(omega)
    sv = Statevector(psi0)
    phases = np.zeros(n)
    for k in range(n):
        x_str = ["I"] * n
        x_str[k] = "X"
        y_str = ["I"] * n
        y_str[k] = "Y"
        ex = float(sv.expectation_value(SparsePauliOp("".join(reversed(x_str)))).real)
        ey = float(sv.expectation_value(SparsePauliOp("".join(reversed(y_str)))).real)
        phases[k] = np.arctan2(ey, ex)
    R = float(abs(np.mean(np.exp(1j * phases))))

    return gap, R, float(eigenvalues[0])


def scan_coupling_at_delta(
    omega: np.ndarray,
    K_topology: np.ndarray,
    delta: float,
    k_range: np.ndarray | None = None,
) -> AnisotropyScanResult:
    """Sweep K at fixed anisotropy Δ. Find K_c from gap minimum."""
    if k_range is None:
        k_range = np.linspace(0.5, 5.0, 15)

    n_k = len(k_range)
    gaps = np.zeros(n_k)
    R_vals = np.zeros(n_k)

    for idx, kb in enumerate(k_range):
        K = float(kb) * K_topology
        gap, R, _E = _ground_state_properties(K, omega, delta)
        gaps[idx] = gap
        R_vals[idx] = R

    k_c = float(k_range[int(np.argmin(gaps))])

    return AnisotropyScanResult(
        delta=delta,
        k_values=k_range,
        gaps=gaps,
        R_values=R_vals,
        k_c_from_gap=k_c,
    )


def anisotropy_phase_diagram(
    omega: np.ndarray,
    K_topology: np.ndarray,
    delta_range: np.ndarray | None = None,
    k_range: np.ndarray | None = None,
) -> PhaseDiagramResult:
    """Compute K_c(Δ) phase diagram.

    Scans Δ from 0 (XY) to 1 (Heisenberg) and finds K_c at each Δ.
    """
    if delta_range is None:
        delta_range = np.linspace(0.0, 1.0, 6)
    if k_range is None:
        k_range = np.linspace(0.5, 5.0, 12)

    scans = []
    k_c_vals = np.zeros(len(delta_range))
    gap_min_vals = np.zeros(len(delta_range))

    for idx, d in enumerate(delta_range):
        scan = scan_coupling_at_delta(omega, K_topology, float(d), k_range)
        scans.append(scan)
        k_c_vals[idx] = scan.k_c_from_gap
        gap_min_vals[idx] = float(np.min(scan.gaps))

    return PhaseDiagramResult(
        delta_values=delta_range,
        k_c_values=k_c_vals,
        gap_min_values=gap_min_vals,
        scans=scans,
    )
