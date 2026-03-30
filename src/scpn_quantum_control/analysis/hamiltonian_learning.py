# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Hamiltonian Learning
"""Hamiltonian learning: recover K_nm from measurement data.

Given correlator measurements <X_i X_j + Y_i Y_j> from the ground
state, reconstruct the coupling matrix K_nm. This is the inverse
problem to Hamiltonian simulation.

Method: maximum likelihood estimation (MLE) assuming the XY model.
For the ground state |ψ_0> of H(K, ω):
    <XX + YY>_ij = -∂E_0/∂K_ij  (Hellmann-Feynman)

So the correlators are gradients of the ground state energy w.r.t.
coupling parameters. We minimise:

    L(K) = Σ_{ij} (C_ij^measured - C_ij^model(K))²

where C_ij^model(K) = <ψ_0(K)|XX_ij + YY_ij|ψ_0(K)>.

This closes the loop: physical measurements → K_nm → quantum simulation.
If the learned K_nm matches the input K_nm, the model is self-consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..hardware.classical import classical_exact_diag


@dataclass
class HamiltonianLearningResult:
    """Result of Hamiltonian learning."""

    K_learned: np.ndarray
    omega_learned: np.ndarray
    loss: float
    n_iterations: int
    correlator_error: float  # mean |C_measured - C_learned|


def measure_correlators(
    K: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """Measure <X_i X_j + Y_i Y_j> from ground state.

    Returns n×n symmetric matrix of correlators.
    """
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi = exact["ground_state"]

    from qiskit.quantum_info import SparsePauliOp, Statevector

    sv = Statevector(np.ascontiguousarray(psi))
    C = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            xx = ["I"] * n
            xx[i] = "X"
            xx[j] = "X"
            yy = ["I"] * n
            yy[i] = "Y"
            yy[j] = "Y"
            op = SparsePauliOp(
                ["".join(reversed(xx)), "".join(reversed(yy))],
                coeffs=[1.0, 1.0],
            )
            val = float(sv.expectation_value(op).real)
            C[i, j] = C[j, i] = val

    result: np.ndarray = C
    return result


def _correlators_from_K(
    K: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """Compute model correlators for given K, omega."""
    return measure_correlators(K, omega)


def _pack_upper_triangle(K: np.ndarray) -> np.ndarray:
    """Pack upper triangle of K into a flat vector."""
    n = K.shape[0]
    params: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            params.append(float(K[i, j]))
    out: np.ndarray = np.array(params)
    return out


def _unpack_upper_triangle(params: np.ndarray, n: int) -> np.ndarray:
    """Unpack flat vector into symmetric K matrix."""
    K = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = K[j, i] = params[idx]
            idx += 1
    result: np.ndarray = K
    return result


def learn_hamiltonian(
    C_measured: np.ndarray,
    omega: np.ndarray,
    K_init: np.ndarray | None = None,
    maxiter: int = 100,
) -> HamiltonianLearningResult:
    """Learn K_nm from measured correlators.

    Args:
        C_measured: n×n correlator matrix <XX+YY>_ij
        omega: known natural frequencies
        K_init: initial guess for K (default: uniform 0.5)
        maxiter: maximum optimizer iterations
    """
    n = len(omega)

    if K_init is None:
        K_init = np.full((n, n), 0.5)
        np.fill_diagonal(K_init, 0.0)

    x0 = _pack_upper_triangle(K_init)

    def loss_fn(params: np.ndarray) -> float:
        K_trial = _unpack_upper_triangle(np.abs(params), n)
        C_model = _correlators_from_K(K_trial, omega)
        return float(np.sum((C_measured - C_model) ** 2))

    result = minimize(loss_fn, x0, method="COBYLA", options={"maxiter": maxiter})

    K_learned = _unpack_upper_triangle(np.abs(result.x), n)
    C_learned = _correlators_from_K(K_learned, omega)
    corr_err = float(np.mean(np.abs(C_measured - C_learned)))

    return HamiltonianLearningResult(
        K_learned=K_learned,
        omega_learned=omega.copy(),
        loss=float(result.fun),
        n_iterations=result.nfev,
        correlator_error=corr_err,
    )
