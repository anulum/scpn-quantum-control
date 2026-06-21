# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- robust residual weighting helpers
"""Robust residual weighting helpers for differentiable least-squares paths."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_parameter_contracts import _as_real_scalar
from .differentiable_transform_helpers import _as_vector_output


def huber_residual_weights(
    residuals: ArrayLike,
    *,
    delta: float = 1.0,
    min_weight: float = 0.0,
) -> NDArray[np.float64]:
    """Return Huber IRLS weights for robust residual-map least squares.

    Parameters
    ----------
    residuals:
        One-dimensional real residual vector.
    delta:
        Positive Huber transition magnitude. Residuals with absolute value at
        or below this threshold keep unit weight.
    min_weight:
        Optional non-negative floor in ``[0, 1]`` applied after Huber
        downweighting.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``float64`` weight vector aligned with ``residuals``.
    """

    residual_arr = _as_vector_output(residuals)
    delta_value = _as_real_scalar("Huber delta", delta)
    if delta_value <= 0.0:
        raise ValueError("Huber delta must be finite and positive")
    min_weight_value = _as_real_scalar("Huber min_weight", min_weight)
    if min_weight_value < 0.0 or min_weight_value > 1.0:
        raise ValueError("Huber min_weight must be finite and in [0, 1]")

    magnitudes = np.abs(residual_arr)
    weights = np.ones_like(residual_arr, dtype=np.float64)
    outliers = magnitudes > delta_value
    weights[outliers] = delta_value / magnitudes[outliers]
    if min_weight_value > 0.0:
        weights = np.maximum(weights, min_weight_value)
    return weights


def soft_l1_residual_weights(
    residuals: ArrayLike,
    *,
    scale: float = 1.0,
    min_weight: float = 0.0,
) -> NDArray[np.float64]:
    """Return smooth Soft-L1 IRLS weights for residual-map least squares.

    Parameters
    ----------
    residuals:
        One-dimensional real residual vector.
    scale:
        Positive residual scale controlling where the Soft-L1 influence curve
        begins to downweight outliers.
    min_weight:
        Optional non-negative floor in ``[0, 1]`` applied after Soft-L1
        downweighting.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``float64`` weight vector aligned with ``residuals``.
    """

    residual_arr = _as_vector_output(residuals)
    scale_value = _as_real_scalar("Soft-L1 scale", scale)
    if scale_value <= 0.0:
        raise ValueError("Soft-L1 scale must be finite and positive")
    min_weight_value = _as_real_scalar("Soft-L1 min_weight", min_weight)
    if min_weight_value < 0.0 or min_weight_value > 1.0:
        raise ValueError("Soft-L1 min_weight must be finite and in [0, 1]")

    scaled = residual_arr / scale_value
    weights = 1.0 / np.sqrt(1.0 + scaled * scaled)
    if min_weight_value > 0.0:
        weights = np.maximum(weights, min_weight_value)
    return weights


__all__ = [
    "huber_residual_weights",
    "soft_l1_residual_weights",
]
