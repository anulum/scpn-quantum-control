# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable batch helpers module
# scpn-quantum-control -- differentiable batch helper contracts
"""Batch and sample tensor validation helpers for native transforms."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array


def _as_parameter_shift_sample_tensor(
    name: str,
    values: ArrayLike,
    *,
    term_count: int,
) -> NDArray[np.float64]:
    """Return a finite parameter-shift sample tensor with term metadata."""
    array = _as_real_numeric_array(name, values)
    if term_count == 1 and array.ndim == 1:
        array = array.reshape(1, array.size)
    elif array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_terms, n_parameters)")
    if array.shape[0] != term_count:
        raise ValueError(f"{name} first dimension must match parameter-shift terms")
    if array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one parameter column")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_batch_parameter_array(
    name: str,
    values: ArrayLike,
    parameter_count: int,
) -> NDArray[np.float64]:
    """Return a finite two-dimensional batch of parameter-space vectors."""
    array = _as_real_numeric_array(name, values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional batch")
    if array.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one row")
    if array.shape[1] != parameter_count:
        raise ValueError(f"{name} row length must match parameter length")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_batch_vector_array(
    name: str,
    values: ArrayLike,
    vector_count: int,
) -> NDArray[np.float64]:
    """Return a finite two-dimensional batch of vector-space cotangents."""
    array = _as_real_numeric_array(name, values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional batch")
    if array.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one row")
    if array.shape[1] != vector_count:
        raise ValueError(f"{name} row length must match vector length")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


__all__ = [
    "_as_batch_parameter_array",
    "_as_batch_vector_array",
    "_as_parameter_shift_sample_tensor",
]
