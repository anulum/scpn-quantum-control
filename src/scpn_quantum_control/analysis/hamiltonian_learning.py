# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hamiltonian Learning
"""Bounded inverse fitting of ``K_nm`` from exact ground-state correlators.

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

The current implementation is a small dense synthetic inverse problem.  It
does not model shot noise, calibration error, model misspecification,
identifiability, or parameter uncertainty, and it has not been validated on
provider or laboratory measurements.  A low training residual therefore shows
only self-consistency on the supplied correlators; it is not evidence that the
generating coupling is unique or experimentally recovered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from ..hardware.classical import classical_exact_diag


@dataclass
class HamiltonianLearningResult:
    """Result of a bounded coupling-matrix fit.

    Attributes
    ----------
    K_learned
        Symmetric non-negative coupling estimate with a zero diagonal.
    omega_learned
        Copy of the caller-supplied frequency vector. Frequencies are not
        optimized by this routine despite the compatibility field name.
    loss
        Sum of squared residuals over the full correlator matrix.
    n_iterations
        Number of COBYLA objective evaluations reported by SciPy.
    correlator_error
        Mean absolute residual over the full correlator matrix.

    """

    K_learned: NDArray[np.float64]
    omega_learned: NDArray[np.float64]
    loss: float
    n_iterations: int
    correlator_error: float  # mean |C_measured - C_learned|


def measure_correlators(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute exact ground-state ``<X_i X_j + Y_i Y_j>`` correlators.

    Parameters
    ----------
    K
        Coupling matrix forwarded to the dense exact diagonalizer.
    omega
        Natural-frequency vector with one entry per oscillator.

    Returns
    -------
    numpy.ndarray
        Symmetric ``n × n`` correlator matrix with a zero diagonal.

    Notes
    -----
    This is an exact-simulator helper, not a hardware measurement routine.

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

    result: NDArray[np.float64] = C
    return result


def _correlators_from_K(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute model correlators for given K, omega."""
    return measure_correlators(K, omega)


def _pack_upper_triangle(K: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pack upper triangle of K into a flat vector."""
    n = K.shape[0]
    params: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            params.append(float(K[i, j]))
    out: NDArray[np.float64] = np.array(params)
    return out


def _unpack_upper_triangle(params: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Unpack flat vector into symmetric K matrix."""
    K = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = K[j, i] = params[idx]
            idx += 1
    result: NDArray[np.float64] = K
    return result


def learn_hamiltonian(
    C_measured: NDArray[np.float64],
    omega: NDArray[np.float64],
    K_init: NDArray[np.float64] | None = None,
    maxiter: int = 100,
) -> HamiltonianLearningResult:
    """Fit a symmetric non-negative ``K_nm`` to supplied correlators.

    Parameters
    ----------
    C_measured
        Target ``n × n`` ``<XX+YY>`` correlator matrix. The name is a
        compatibility convention; the routine does not establish measurement
        provenance.
    omega
        Known natural frequencies. They are copied into the result and are not
        fitted.
    K_init
        Initial symmetric coupling guess. Defaults to off-diagonal ``0.5``.
    maxiter
        Maximum COBYLA objective evaluations.

    Returns
    -------
    HamiltonianLearningResult
        Fitted matrix and in-sample residual summaries.

    Notes
    -----
    The objective repeatedly performs dense exact diagonalization. A small
    residual is not an identifiability, uncertainty, held-out, or experimental
    validation certificate.

    """
    n = len(omega)

    if K_init is None:
        K_init = np.full((n, n), 0.5)
        np.fill_diagonal(K_init, 0.0)

    x0 = _pack_upper_triangle(K_init)

    def loss_fn(params: NDArray[np.float64]) -> float:
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
