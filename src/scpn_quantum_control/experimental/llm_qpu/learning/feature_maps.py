# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU feature maps
"""Classical feature maps of compressed angles and a train-only standardiser."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

_FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _matrix(values: object, *, name: str) -> _FloatArray:
    """Return a finite two-dimensional float64 copy, or refuse."""
    array = np.array(values, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return array


@dataclass(frozen=True, slots=True)
class Standardiser:
    """Per-feature centre and scale fitted on train rows; constant features keep scale one."""

    centre: _FloatArray
    scale: _FloatArray

    def apply(self, features: object) -> _FloatArray:
        """Return the features centred and scaled with the train statistics."""
        matrix = _matrix(features, name="features")
        if matrix.shape[1] != self.centre.shape[0]:
            raise ValueError("features do not have the fitted width")
        return (matrix - self.centre) / self.scale


def fit_standardiser(train: object) -> Standardiser:
    """Fit a standardiser on train rows only, using the population standard deviation."""
    matrix = _matrix(train, name="train features")
    scale = matrix.std(axis=0)
    return Standardiser(matrix.mean(axis=0), np.where(scale > 1e-12, scale, 1.0))


def sin_cos_basis(angles: object) -> _FloatArray:
    """Return sin and cos of every angle: the single-encoding map without any product terms."""
    matrix = _matrix(angles, name="angles")
    return np.concatenate([np.sin(matrix), np.cos(matrix)], axis=1)


def trig_product_basis(angles: object) -> _FloatArray:
    """Return every product of one factor from {1, sin a_i, cos a_i} per angle: 3**m columns.

    Any linear observable after a fixed, input-independent unitary or channel applied to a single
    RY encoding is a linear combination of these columns, so a readout on them spans every such
    quantum feature map of the same angles.
    """
    matrix = _matrix(angles, name="angles")
    if matrix.shape[1] > 8:
        raise ValueError("the product basis is bounded at eight angles (6561 columns)")
    factors = np.stack([np.ones_like(matrix), np.sin(matrix), np.cos(matrix)], axis=2)
    columns = [
        np.prod(factors[:, np.arange(matrix.shape[1]), list(choice)], axis=1)
        for choice in itertools.product(range(3), repeat=matrix.shape[1])
    ]
    return np.stack(columns, axis=1)


def random_fourier_features(
    inputs: object, *, dimension: int, bandwidth: float, seed: int
) -> _FloatArray:
    """Return seeded random Fourier features of a Gaussian kernel of the given bandwidth."""
    matrix = _matrix(inputs, name="inputs")
    if type(dimension) is not int or not 0 < dimension <= 65_536:
        raise ValueError("feature dimension must be a bounded positive integer")
    if not (type(bandwidth) is float and np.isfinite(bandwidth) and bandwidth > 0.0):
        raise ValueError("bandwidth must be a positive finite float")
    generator = np.random.default_rng(seed)
    weights = generator.normal(0.0, 1.0 / bandwidth, size=(matrix.shape[1], dimension))
    offsets = generator.uniform(0.0, 2.0 * np.pi, size=dimension)
    features: _FloatArray = np.sqrt(2.0 / dimension) * np.cos(matrix @ weights + offsets)
    return features
