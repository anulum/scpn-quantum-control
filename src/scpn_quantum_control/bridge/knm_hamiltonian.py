# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Knm coupling matrix -> Pauli Hamiltonian compiler.

Translates the 16x16 Knm coupling matrix + 16 natural frequencies from
Paper 27 into a SparsePauliOp for quantum simulation.

Kuramoto <-> XY mapping:
  K[i,j]*sin(theta_j - theta_i)  <=>  -J_ij*(X_i X_j + Y_i Y_j)
  omega_i                         <=>  -h_i * Z_i
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .._constants import COUPLING_SPARSITY_EPS

KNM_SPARSITY_EPS = COUPLING_SPARSITY_EPS

# Paper 27, Table 1: canonical natural frequencies (rad/s)
OMEGA_N_16 = np.array(
    [
        1.329,
        2.610,
        0.844,
        1.520,
        0.710,
        3.780,
        1.055,
        0.625,
        2.210,
        1.740,
        0.480,
        3.210,
        0.915,
        1.410,
        2.830,
        0.991,
    ],
    dtype=np.float64,
)


def build_knm_paper27(
    L: int = 16,
    K_base: float = 0.45,  # Paper 27, Eq. 3
    K_alpha: float = 0.3,  # Paper 27, Eq. 3
) -> np.ndarray:
    """Build the canonical Knm coupling matrix from Paper 27.

    K[i,j] = K_base * exp(-K_alpha * |i - j|)   (Paper 27, Eq. 3)
    with calibration anchors from Table 2 and cross-hierarchy boosts from S4.3.
    """
    idx = np.arange(L)
    K: np.ndarray = K_base * np.exp(-K_alpha * np.abs(idx[:, None] - idx[None, :]))

    # Paper 27 Table 2 calibration anchors (only apply if indices in range)
    anchors = {(0, 1): 0.302, (1, 2): 0.201, (2, 3): 0.252, (3, 4): 0.154}
    for (i, j), val in anchors.items():
        if i < L and j < L:
            K[i, j] = K[j, i] = val

    # Paper 27 S4.3 cross-hierarchy boosts
    if L > 15:
        K[0, 15] = K[15, 0] = max(K[0, 15], 0.05)  # L1-L16
    if L > 6:
        K[4, 6] = K[6, 4] = max(K[4, 6], 0.15)  # L5-L7

    return K


def build_kuramoto_ring(
    n: int,
    coupling: float = 1.0,
    omega: np.ndarray | None = None,
    rng_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a nearest-neighbour ring coupling matrix for n Kuramoto oscillators.

    Returns (K, omega) ready for QuantumKuramotoSolver or knm_to_hamiltonian.
    If omega is None, draws from N(0,1) with the given seed.
    """
    K: np.ndarray = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        K[i, j] = K[j, i] = coupling
    if omega is None:
        rng = np.random.default_rng(rng_seed)
        omega = rng.standard_normal(n)
    return K, np.asarray(omega, dtype=np.float64)


def knm_to_hamiltonian(K: np.ndarray, omega: np.ndarray) -> SparsePauliOp:
    """Convert Knm coupling matrix + natural frequencies to SparsePauliOp.

    H = -sum_{i<j} K[i,j] * (X_i X_j + Y_i Y_j) - sum_i omega_i * Z_i

    Uses Qiskit little-endian qubit ordering.
    """
    n = len(omega)
    if K.shape[0] != n:
        raise ValueError(f"K has {K.shape[0]} rows but omega has {n} elements")
    pauli_list = []

    for i in range(n):
        if abs(omega[i]) > KNM_SPARSITY_EPS:
            z_str = ["I"] * n
            z_str[i] = "Z"
            pauli_list.append(("".join(reversed(z_str)), -omega[i]))

    for i in range(n):
        for j in range(i + 1, n):
            if abs(K[i, j]) < KNM_SPARSITY_EPS:
                continue
            # XX term
            xx = ["I"] * n
            xx[i] = "X"
            xx[j] = "X"
            pauli_list.append(("".join(reversed(xx)), -K[i, j]))
            # YY term
            yy = ["I"] * n
            yy[i] = "Y"
            yy[j] = "Y"
            pauli_list.append(("".join(reversed(yy)), -K[i, j]))

    labels, coeffs = zip(*pauli_list)
    return SparsePauliOp(list(labels), list(coeffs)).simplify()


def knm_to_xxz_hamiltonian(
    K: np.ndarray,
    omega: np.ndarray,
    delta: float = 0.0,
) -> SparsePauliOp:
    """Convert Knm + frequencies to XXZ Hamiltonian with anisotropy Δ.

    H = -sum_{i<j} K[i,j] * (X_iX_j + Y_iY_j + Δ·Z_iZ_j) - sum_i omega_i * Z_i

    Δ = 0: XY model (standard Kuramoto mapping, in-plane S² dynamics)
    Δ = 1: isotropic Heisenberg (full S² dynamics, Kouchekian-Teodorescu 2025)

    The anisotropy parameter controls the off-plane spin coupling that
    the standard Kuramoto-XY mapping omits. The full Heisenberg model
    (Δ=1) corresponds to the variational S² spin formulation proven in
    arXiv:2601.00113 (Kouchekian & Teodorescu, 2025).

    At Δ=1, perturbations around equilibria connect to the semiclassical
    Gaudin model and the Richardson pairing mechanism.
    """
    n = len(omega)
    if K.shape[0] != n:
        raise ValueError(f"K has {K.shape[0]} rows but omega has {n} elements")
    pauli_list = []

    for i in range(n):
        if abs(omega[i]) > KNM_SPARSITY_EPS:
            z_str = ["I"] * n
            z_str[i] = "Z"
            pauli_list.append(("".join(reversed(z_str)), -omega[i]))

    for i in range(n):
        for j in range(i + 1, n):
            if abs(K[i, j]) < KNM_SPARSITY_EPS:
                continue
            # XX term
            xx = ["I"] * n
            xx[i] = "X"
            xx[j] = "X"
            pauli_list.append(("".join(reversed(xx)), -K[i, j]))
            # YY term
            yy = ["I"] * n
            yy[i] = "Y"
            yy[j] = "Y"
            pauli_list.append(("".join(reversed(yy)), -K[i, j]))
            # ZZ term (off-plane, controlled by delta)
            if abs(delta) > KNM_SPARSITY_EPS:
                zz = ["I"] * n
                zz[i] = "Z"
                zz[j] = "Z"
                pauli_list.append(("".join(reversed(zz)), -K[i, j] * delta))

    labels, coeffs = zip(*pauli_list)
    return SparsePauliOp(list(labels), list(coeffs)).simplify()


def knm_to_ansatz(K: np.ndarray, reps: int = 2, threshold: float = 0.01) -> QuantumCircuit:
    """Build physics-informed ansatz: CZ entanglement only between Knm-connected pairs.

    Pattern from QUANTUM_LAB script 16 (PhysicsInformedAnsatz).
    """
    n = K.shape[0]
    params = ParameterVector("p", n * 2 * reps)
    qc = QuantumCircuit(n)

    idx = 0
    for _ in range(reps):
        for q in range(n):
            qc.ry(params[idx], q)
            idx += 1
        for q in range(n):
            qc.rz(params[idx], q)
            idx += 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(K[i, j]) >= threshold:
                    qc.cz(i, j)

    return qc
