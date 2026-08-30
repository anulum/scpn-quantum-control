# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Gaussian RBF surrogate
"""Smooth classical surrogates for bounded quantum-objective studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

CLASSICAL_SURROGATE_CLAIM_BOUNDARY = (
    "Local Gaussian-RBF interpolation with an analytic input gradient only; "
    "not exact quantum differentiation, hardware execution, generalisation, "
    "closed-loop control, or quantum-advantage evidence."
)


def _finite_matrix(values: FloatArray, *, name: str) -> FloatArray:
    """Return an immutable finite two-dimensional float array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty rows and columns.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _finite_vector(values: FloatArray, *, name: str) -> FloatArray:
    """Return an immutable finite one-dimensional float array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class GaussianRBFSurrogate:
    """Gaussian radial-basis surrogate with an analytic input gradient.

    Parameters
    ----------
    centres:
        Training inputs with shape ``(n_centres, n_parameters)``.
    weights:
        Fitted radial-basis weights with one value per centre.
    length_scale:
        Positive Gaussian-kernel length scale.
    intercept:
        Constant target offset fitted with the radial-basis weights.
    training_input_digests:
        SHA-256 identities of every training row, used to reject validation
        leakage.
    training_target_digest:
        SHA-256 identity of the fitted target vector.

    """

    centres: FloatArray
    weights: FloatArray
    length_scale: float
    intercept: float
    training_input_digests: tuple[str, ...]
    training_target_digest: str
    claim_boundary: str = CLASSICAL_SURROGATE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate dimensions, provenance identities, and finite values."""
        centres = _finite_matrix(self.centres, name="centres")
        weights = _finite_vector(self.weights, name="weights")
        if weights.shape != (centres.shape[0],):
            raise ValueError("weights must contain one value per centre.")
        length_scale = float(self.length_scale)
        intercept = float(self.intercept)
        if not np.isfinite(length_scale) or length_scale <= 0.0:
            raise ValueError("length_scale must be finite and positive.")
        if not np.isfinite(intercept):
            raise ValueError("intercept must be finite.")
        if len(self.training_input_digests) != centres.shape[0]:
            raise ValueError("training_input_digests must identify every centre.")
        if len(set(self.training_input_digests)) != len(self.training_input_digests):
            raise ValueError("training_input_digests must be unique.")
        if not all(len(digest) == 64 for digest in self.training_input_digests):
            raise ValueError("training input digests must be SHA-256 hex strings.")
        if len(self.training_target_digest) != 64:
            raise ValueError("training_target_digest must be a SHA-256 hex string.")
        if not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be non-empty.")
        object.__setattr__(self, "centres", centres)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "length_scale", length_scale)
        object.__setattr__(self, "intercept", intercept)

    @property
    def n_parameters(self) -> int:
        """Return the surrogate input dimension."""
        return int(self.centres.shape[1])

    def predict(self, inputs: FloatArray) -> FloatArray:
        """Predict objective values for a two-dimensional input matrix.

        Parameters
        ----------
        inputs:
            Evaluation inputs with shape ``(n_samples, n_parameters)``.

        Returns
        -------
        numpy.ndarray
            Predicted scalar objective for each sample.

        """
        values = _finite_matrix(inputs, name="inputs")
        if values.shape[1] != self.n_parameters:
            raise ValueError("inputs must match the surrogate parameter dimension.")
        deltas = values[:, None, :] - self.centres[None, :, :]
        squared_distance = np.einsum("scd,scd->sc", deltas, deltas)
        kernel = np.exp(-0.5 * squared_distance / (self.length_scale**2))
        result: FloatArray = np.asarray(kernel @ self.weights + self.intercept, dtype=np.float64)
        return result

    def value(self, parameters: FloatArray) -> float:
        """Predict one scalar objective value."""
        point = _finite_vector(parameters, name="parameters")
        if point.shape != (self.n_parameters,):
            raise ValueError("parameters must match the surrogate parameter dimension.")
        return float(self.predict(point[None, :])[0])

    def gradient(self, parameters: FloatArray) -> FloatArray:
        """Return the analytic gradient with respect to one input point."""
        point = _finite_vector(parameters, name="parameters")
        if point.shape != (self.n_parameters,):
            raise ValueError("parameters must match the surrogate parameter dimension.")
        deltas = point[None, :] - self.centres
        squared_distance = np.einsum("cd,cd->c", deltas, deltas)
        kernel = np.exp(-0.5 * squared_distance / (self.length_scale**2))
        coefficients = self.weights * kernel / (self.length_scale**2)
        result: FloatArray = np.asarray(
            np.sum(coefficients[:, None] * (-deltas), axis=0),
            dtype=np.float64,
        )
        return result

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready model summary without duplicating training data."""
        return {
            "model": "gaussian_rbf",
            "n_centres": int(self.centres.shape[0]),
            "n_parameters": self.n_parameters,
            "length_scale": self.length_scale,
            "intercept": self.intercept,
            "training_input_digests": list(self.training_input_digests),
            "training_target_digest": self.training_target_digest,
            "claim_boundary": self.claim_boundary,
        }


__all__ = [
    "CLASSICAL_SURROGATE_CLAIM_BOUNDARY",
    "GaussianRBFSurrogate",
]
