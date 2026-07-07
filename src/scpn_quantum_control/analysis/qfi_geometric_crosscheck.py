# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Geometric QFI cross-check
"""Cross-validate the spectral QFI against the quantum-geometric-tensor route.

Two independent physics routes yield the coupling-parameter Quantum Fisher
Information of the XY ground state:

* the exact spectral sum rule (``analysis.qfi.compute_qfi``,
  Braunstein & Caves, PRL 72, 3439 (1994)); and
* four times the Fubini-Study metric — the real part of the quantum geometric
  tensor computed by finite-difference ground states
  (``pgbo.compute_pgbo_tensor``): ``F = 4 Re(Q)`` for pure states.

Agreement between the two is a strong internal-consistency check on both
implementations, and the geometric route additionally exposes the Berry
curvature ``-2 Im(Q)`` (identically zero for the time-reversal-symmetric XY
ground state, which the cross-check asserts as a physics invariant).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..pgbo.quantum_bridge import PGBOResult, compute_pgbo_tensor
from .qfi import QFIResult, compute_qfi

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class QFIGeometricCrosscheck:
    """Agreement report between the spectral and geometric QFI routes."""

    spectral: QFIResult
    geometric: PGBOResult
    qfi_geometric: FloatMatrix
    max_abs_difference: float
    max_rel_difference: float
    max_abs_berry_curvature: float
    coupling_pairs: list[tuple[int, int]]

    @property
    def agrees(self) -> bool:
        """Whether both routes agree within the finite-difference tolerance."""
        return self.max_rel_difference <= 0.05


def crosscheck_qfi_geometric(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    epsilon: float = 0.005,
    max_dense_gib: float | None = None,
) -> QFIGeometricCrosscheck:
    """Compute the QFI by both routes over the full upper-triangle pair set.

    Parameters
    ----------
    K : NDArray[np.float64]
        Symmetric coupling matrix, shape ``(n, n)``.
    omega : NDArray[np.float64]
        Natural frequencies, shape ``(n,)``.
    epsilon : float, optional
        Finite-difference step for the geometric route; must be positive.
    max_dense_gib : float or None, optional
        Dense-allocation budget forwarded to the spectral engine.

    Returns
    -------
    QFIGeometricCrosscheck
        Both results plus elementwise agreement metrics; ``qfi_geometric``
        is ``4 * Re(Q)`` aligned to the same pair ordering as the spectral
        matrix.

    Raises
    ------
    ValueError
        If ``K`` is not square-symmetric, ``omega`` has the wrong shape, or
        ``epsilon`` is not positive.
    """
    K_arr = np.asarray(K, dtype=np.float64)
    omega_arr = np.asarray(omega, dtype=np.float64)
    if K_arr.ndim != 2 or K_arr.shape[0] != K_arr.shape[1]:
        raise ValueError(f"K must be square, got shape {K_arr.shape}")
    if not np.allclose(K_arr, K_arr.T):
        raise ValueError("K must be symmetric")
    if omega_arr.shape != (K_arr.shape[0],):
        raise ValueError(f"omega shape must be ({K_arr.shape[0]},), got {omega_arr.shape}")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    n = K_arr.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    spectral = compute_qfi(K_arr, omega_arr, pairs, max_dense_gib=max_dense_gib)
    geometric = compute_pgbo_tensor(K_arr, omega_arr, epsilon=epsilon)

    qfi_geometric = 4.0 * geometric.metric_tensor
    difference = np.abs(spectral.qfi_matrix - qfi_geometric)
    scale = np.maximum(np.abs(spectral.qfi_matrix), 1e-12)

    return QFIGeometricCrosscheck(
        spectral=spectral,
        geometric=geometric,
        qfi_geometric=qfi_geometric,
        max_abs_difference=float(np.max(difference)),
        max_rel_difference=float(np.max(difference / scale)),
        max_abs_berry_curvature=float(np.max(np.abs(geometric.berry_curvature))),
        coupling_pairs=pairs,
    )


__all__ = ["QFIGeometricCrosscheck", "crosscheck_qfi_geometric"]
