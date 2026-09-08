# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Readout SVD workspace admission
"""Explicit LAPACK workspace for readout spectral condition numbers.

Accounts for retained/copy matrices, singular values, queried floating workspace,
integer workspace and dummy singular-vector outputs. Python labels, allocator
overhead and vendor-internal threading buffers are not a process-memory bound.
"""

import math

import numpy as np
from numpy.typing import NDArray
from scipy.linalg.lapack import get_lapack_funcs

from ..dense_budget import DenseAllocationError, require_dense_allocation

dgesdd, dgesdd_lwork = get_lapack_funcs(("gesdd", "gesdd_lwork"), dtype=np.float64)


def admit_readout_svd(n_qubits: int, max_dense_gib: float | None) -> int:
    """Return admitted LAPACK work length before matrix or label construction.

    The initial two-matrix guard bounds dimensions before the scalar workspace
    query. Invalid query results fail closed. The returned work length must be
    passed unchanged to the condition-number calculation for the admitted size.

    Parameters
    ----------
    n_qubits
        Positive integer qubit count defining the square matrix dimension.
    max_dense_gib
        Explicit GiB allowance, or None for the current process budget.

    Returns
    -------
    int
        LAPACK floating workspace length in float64 elements.

    Raises
    ------
    DenseAllocationError
        If numeric buffers exceed the budget or LAPACK integer range.
    numpy.linalg.LinAlgError
        If the workspace query fails or returns an invalid length.
    """
    estimate = require_dense_allocation(
        n_qubits,
        dtype=np.float64,
        rank=2,
        object_count=2,
        max_gib=max_dense_gib,
        label="readout confusion matrix",
    )
    n = estimate.dimension
    work, info = dgesdd_lwork(n, n, compute_uv=0)
    if info != 0 or not math.isfinite(work) or work < 1:
        raise np.linalg.LinAlgError("readout SVD workspace query failed")
    lwork = math.ceil(float(work))
    if lwork > np.iinfo(dgesdd.int_dtype).max:
        raise DenseAllocationError("readout SVD workspace exceeds LAPACK integer range")
    # SciPy flapack_gen.pyf.src DGESDD: WORK(lwork), IWORK(8*n), S(n), U/VT(1,1).
    workspace_bytes = (lwork + n + 2) * 8 + 8 * n * np.dtype(dgesdd.int_dtype).itemsize
    required_bytes = estimate.bytes_required + workspace_bytes
    if required_bytes > estimate.budget_bytes:
        raise DenseAllocationError(
            f"readout confusion matrix SVD workspace requires {required_bytes} bytes, "
            f"above the active dense budget {estimate.budget_bytes} bytes"
        )
    return lwork


def readout_svd_condition(matrix: NDArray[np.float64], lwork: int) -> float:
    """Compute the spectral condition without modifying the retained calibration.

    The caller admits this matrix size and work length before constructing it.
    A writable Fortran copy permits LAPACK overwrite without a second input copy.
    Nonconvergence and invalid LAPACK arguments raise LinAlgError; zero smallest
    singular value gives infinity, as for numpy.linalg.cond.

    Parameters
    ----------
    matrix
        Finite square float64 calibration matrix, retained without modification.
    lwork
        Admitted work length from admit_readout_svd for this matrix dimension.

    Returns
    -------
    float
        Ratio of largest to smallest singular value, possibly infinity.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the SVD reports invalid arguments or fails to converge.
    """
    scratch = np.array(matrix, dtype=np.float64, order="F", copy=True)
    _, singular_values, _, info = dgesdd(
        scratch,
        compute_uv=0,
        overwrite_a=1,
        lwork=lwork,
    )
    if info != 0:
        raise np.linalg.LinAlgError(f"readout SVD failed with LAPACK info={info}")
    smallest = float(singular_values[-1])
    if smallest == 0:
        return math.inf
    return float(singular_values[0]) / smallest
