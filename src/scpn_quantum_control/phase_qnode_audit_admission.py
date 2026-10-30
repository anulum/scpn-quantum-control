# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode audit admission
"""Admit host parameters, batches, scalars and tolerances for capability audits.

Vectors, scalars and tolerances are admitted by the shared differentiable
admission leaf with this module's NumPy, ``float`` and ``ValueError``. Batches
keep the two refusal vocabularies of the PyTorch and JAX audit routes.
"""

from __future__ import annotations

from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .differentiable_tensor_admission import (
    AdmissionPrimitives,
    require_nonnegative_tolerance,
    require_real_parameter_vector,
    require_real_scalar,
)

FloatArray: TypeAlias = NDArray[np.float64]


def _primitives() -> AdmissionPrimitives:
    return AdmissionPrimitives(numpy=lambda: np, scalar=lambda: float, invalid=lambda: ValueError)


def admitted_parameter_vector(
    name: str, values: object, *, width: int | None = None
) -> FloatArray:
    """Return a finite real float64 parameter vector.

    Parameters
    ----------
    name
        Name used in refusal messages.
    values
        Host or native vector-like values.
    width
        Required length, when given.

    Returns
    -------
    FloatArray
        Admitted float64 copy.

    """
    vector = require_real_parameter_vector(name, values, width=width, primitives=_primitives())
    return cast(FloatArray, vector.astype(np.float64, copy=True))


def admitted_scalar(name: str, value: object) -> float:
    """Return a finite real scalar.

    Parameters
    ----------
    name
        Name used in refusal messages.
    value
        Host or native scalar-like value.

    Returns
    -------
    float
        The admitted scalar.

    """
    return require_real_scalar(name, value, primitives=_primitives())


def admitted_torch_tolerance(value: float) -> float:
    """Return a tolerance admitted with the PyTorch audit route's refusal.

    Parameters
    ----------
    value
        Candidate tolerance.

    Returns
    -------
    float
        The finite non-negative tolerance.

    """
    return require_nonnegative_tolerance(
        value,
        error_text="tolerance must be a non-negative finite float",
        negative_first=False,
        primitives=_primitives(),
    )


def admitted_jax_tolerance(value: float) -> float:
    """Return a tolerance admitted with the JAX audit route's refusal.

    Parameters
    ----------
    value
        Candidate tolerance.

    Returns
    -------
    float
        The finite non-negative tolerance.

    """
    return require_nonnegative_tolerance(
        value,
        error_text="tolerance must be finite and non-negative",
        negative_first=True,
        primitives=_primitives(),
    )


def admitted_parameter_matrix(
    name: str, values: object, *, width: int | None = None
) -> FloatArray:
    """Return a finite, non-empty two-dimensional float64 copy (PyTorch route).

    Parameters
    ----------
    name
        Name used in refusal messages.
    values
        Host matrix-like values.
    width
        Required number of columns, when given.

    Returns
    -------
    FloatArray
        Admitted float64 matrix.

    Raises
    ------
    ValueError
        If the values are not a finite, non-empty matrix of the given width.

    """
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if width is not None and matrix.shape[1] != width:
        raise ValueError(f"{name} width must be {width}, got {matrix.shape[1]}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix.astype(np.float64, copy=True)


def admitted_parameter_batch(name: str, values: object, *, width: int) -> FloatArray:
    """Return a finite float64 batch with at least one row of ``width`` values (JAX route).

    Parameters
    ----------
    name
        Name used in refusal messages.
    values
        Host matrix-like values.
    width
        Required number of columns.

    Returns
    -------
    FloatArray
        Admitted float64 batch.

    Raises
    ------
    ValueError
        If the values are not a finite, non-empty batch of the given width.

    """
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one parameter row")
    if matrix.shape[1] != width:
        raise ValueError(f"{name} must have shape (batch, {width}), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix.astype(np.float64, copy=True)


__all__ = [
    "admitted_jax_tolerance",
    "admitted_parameter_batch",
    "admitted_parameter_matrix",
    "admitted_parameter_vector",
    "admitted_scalar",
    "admitted_torch_tolerance",
]
