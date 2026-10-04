# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — physical density input admission
"""Admit explicit physical density matrices without repairing their values."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..dense_budget import require_dense_allocation

DENSITY_INPUT_TOLERANCE = 1e-10
"""Absolute trace, Hermiticity and negative-eigenvalue allowance for float64 inputs."""


def validate_density_matrix(
    rho: NDArray[np.complex128] | NDArray[np.float64],
    n_qubits: int,
    *,
    max_dense_gib: float | None = None,
) -> NDArray[np.complex128]:
    """Return an owned complex128 copy of a physical input density matrix.

    Parameters
    ----------
    rho
        Float64 or complex128 array of shape ``(2**n_qubits, 2**n_qubits)``.
        Basis ordering is retained; no normalization or PSD projection occurs.
    n_qubits
        Positive qubit count defining the required Hilbert dimension.
    max_dense_gib
        Optional positive GiB allowance for four validation matrices, checked
        before numeric inspection, copying or eigenvalue materialization.

    Returns
    -------
    numpy.ndarray
        Independent complex128 matrix with the original values and ordering.

    Raises
    ------
    ValueError
        If shape, dtype, finiteness, unit trace, Hermiticity or positive
        semidefiniteness fails. Physical checks use an absolute ``1e-10`` bound.
    DenseAllocationError
        If the declared validation workspace exceeds the active allowance.

    """
    require_dense_allocation(
        n_qubits,
        rank=2,
        object_count=4,
        max_gib=max_dense_gib,
        label="density input validation workspace",
    )
    dimension = 1 << n_qubits
    if not isinstance(rho, np.ndarray) or rho.shape != (dimension, dimension):
        raise ValueError("density matrix shape must match the Hilbert dimension")
    if rho.dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
        raise ValueError("density matrix dtype must be float64 or complex128")
    if not np.all(np.isfinite(rho)):
        raise ValueError("density matrix must contain finite values")
    if np.max(np.abs(rho - rho.conj().T)) > DENSITY_INPUT_TOLERANCE:
        raise ValueError("density matrix must be Hermitian")
    if abs(np.trace(rho) - 1) > DENSITY_INPUT_TOLERANCE:
        raise ValueError("density matrix must have unit trace")
    if np.linalg.eigvalsh(rho).min() < -DENSITY_INPUT_TOLERANCE:
        raise ValueError("density matrix must be positive semidefinite")
    return np.array(rho, dtype=np.complex128, copy=True)
