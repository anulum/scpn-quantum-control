# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Gaussian RBF surrogate fitting
"""Deterministic fitting for differentiable Gaussian-RBF surrogates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import GaussianRBFSurrogate

FloatArray = NDArray[np.float64]
_NUMERIC_CUSTODY_DECIMALS = 6


def _array_digest(values: FloatArray) -> str:
    """Return a shape-bound, cross-runtime SHA-256 identity for one float array."""
    rounded = np.round(np.asarray(values, dtype=np.float64), _NUMERIC_CUSTODY_DECIMALS)
    array = np.ascontiguousarray(np.where(rounded == 0.0, 0.0, rounded), dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def input_row_digests(inputs: FloatArray) -> tuple[str, ...]:
    """Return one shape-bound SHA-256 identity per input row.

    Parameters
    ----------
    inputs:
        Finite two-dimensional input matrix.

    Returns
    -------
    tuple[str, ...]
        Row identities in input order.

    """
    values = np.asarray(inputs, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("inputs must be a non-empty 2-D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("inputs must contain only finite values.")
    return tuple(_array_digest(values[index]) for index in range(values.shape[0]))


@dataclass(frozen=True, slots=True)
class SurrogateFitConfig:
    """Configuration for deterministic Gaussian-RBF ridge fitting."""

    regularisation: float = 1.0e-8
    length_scale: float | None = None

    def __post_init__(self) -> None:
        """Validate the ridge strength and optional kernel length scale."""
        regularisation = float(self.regularisation)
        if not np.isfinite(regularisation) or regularisation <= 0.0:
            raise ValueError("regularisation must be finite and positive.")
        object.__setattr__(self, "regularisation", regularisation)
        if self.length_scale is not None:
            length_scale = float(self.length_scale)
            if not np.isfinite(length_scale) or length_scale <= 0.0:
                raise ValueError("length_scale must be finite and positive when supplied.")
            object.__setattr__(self, "length_scale", length_scale)


def _automatic_length_scale(inputs: FloatArray) -> float:
    """Return the median non-zero pairwise distance between training rows."""
    deltas = inputs[:, None, :] - inputs[None, :, :]
    distances = np.sqrt(np.einsum("ijd,ijd->ij", deltas, deltas))
    positive = distances[distances > 0.0]
    return float(np.median(positive))


def fit_gaussian_rbf_surrogate(
    inputs: FloatArray,
    targets: FloatArray,
    *,
    config: SurrogateFitConfig | None = None,
) -> GaussianRBFSurrogate:
    """Fit a smooth Gaussian-RBF surrogate by regularised linear solve.

    Parameters
    ----------
    inputs:
        Training inputs with shape ``(n_samples, n_parameters)``.
    targets:
        Scalar exact-objective values with one value per input.
    config:
        Optional deterministic fit configuration.

    Returns
    -------
    GaussianRBFSurrogate
        Fitted model carrying training-row and target provenance digests.

    """
    x_train = np.asarray(inputs, dtype=np.float64)
    y_train = np.asarray(targets, dtype=np.float64)
    if x_train.ndim != 2 or x_train.shape[0] < 2 or x_train.shape[1] == 0:
        raise ValueError("inputs must contain at least two rows and one parameter.")
    if not np.all(np.isfinite(x_train)):
        raise ValueError("inputs must contain only finite values.")
    if y_train.ndim != 1 or y_train.shape != (x_train.shape[0],):
        raise ValueError("targets must be a vector matching the input rows.")
    if not np.all(np.isfinite(y_train)):
        raise ValueError("targets must contain only finite values.")

    row_digests = input_row_digests(x_train)
    if len(set(row_digests)) != len(row_digests):
        raise ValueError("training inputs must not contain duplicate rows.")

    fit_config = config or SurrogateFitConfig()
    length_scale = (
        _automatic_length_scale(x_train)
        if fit_config.length_scale is None
        else fit_config.length_scale
    )
    deltas = x_train[:, None, :] - x_train[None, :, :]
    squared_distance = np.einsum("ijd,ijd->ij", deltas, deltas)
    kernel = np.exp(-0.5 * squared_distance / (length_scale**2))
    intercept = float(np.mean(y_train))
    system = kernel + fit_config.regularisation * np.eye(x_train.shape[0])
    weights = np.linalg.solve(system, y_train - intercept).astype(np.float64)
    return GaussianRBFSurrogate(
        centres=x_train,
        weights=weights,
        length_scale=length_scale,
        intercept=intercept,
        training_input_digests=row_digests,
        training_target_digest=_array_digest(y_train),
    )


__all__ = [
    "SurrogateFitConfig",
    "fit_gaussian_rbf_surrogate",
    "input_row_digests",
]
