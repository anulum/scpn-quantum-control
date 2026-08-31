# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Surrogate fidelity certificates
"""Held-out value and gradient fidelity gates for classical surrogates."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .models import CLASSICAL_SURROGATE_CLAIM_BOUNDARY, GaussianRBFSurrogate
from .train import input_row_digests

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


def _positive_finite(value: float, *, name: str) -> float:
    """Return a finite positive scalar."""
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


@dataclass(frozen=True, slots=True)
class SurrogateFidelityThresholds:
    """Predeclared held-out value-fidelity acceptance thresholds."""

    max_rmse: float
    max_absolute_error: float
    min_r_squared: float

    def __post_init__(self) -> None:
        """Validate finite thresholds without weakening their meaning."""
        object.__setattr__(self, "max_rmse", _positive_finite(self.max_rmse, name="max_rmse"))
        object.__setattr__(
            self,
            "max_absolute_error",
            _positive_finite(self.max_absolute_error, name="max_absolute_error"),
        )
        minimum = float(self.min_r_squared)
        if not np.isfinite(minimum) or minimum < -1.0 or minimum > 1.0:
            raise ValueError("min_r_squared must be finite and in [-1, 1].")
        object.__setattr__(self, "min_r_squared", minimum)


@dataclass(frozen=True, slots=True)
class SurrogateFidelityCertificate:
    """Held-out value fidelity against an exact local objective."""

    n_training: int
    n_validation: int
    rmse: float
    mean_absolute_error: float
    max_absolute_error: float
    r_squared: float
    passed: bool
    validation_input_digest: str
    exact_target_digest: str
    prediction_digest: str
    thresholds: SurrogateFidelityThresholds
    simulator: str = "local_exact_statevector"
    training_overlap_count: int = 0
    hardware_execution: bool = False
    claim_boundary: str = CLASSICAL_SURROGATE_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready certificate mapping."""
        payload = asdict(self)
        payload["thresholds"] = asdict(self.thresholds)
        return payload


@dataclass(frozen=True, slots=True)
class SurrogateGradientCertificate:
    """Analytic-surrogate gradient fidelity against exact finite differences."""

    n_points: int
    n_parameters: int
    finite_difference_step: float
    rmse: float
    max_absolute_error: float
    max_allowed_error: float
    passed: bool
    validation_input_digest: str
    surrogate_gradient_digest: str
    exact_gradient_digest: str
    simulator: str = "local_exact_statevector_central_difference"
    training_overlap_count: int = 0
    hardware_execution: bool = False
    claim_boundary: str = CLASSICAL_SURROGATE_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready gradient certificate mapping."""
        return asdict(self)


def _validated_validation_data(
    model: GaussianRBFSurrogate,
    inputs: FloatArray,
    targets: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return finite dimension-matched held-out inputs and targets."""
    x_validation = np.asarray(inputs, dtype=np.float64)
    y_exact = np.asarray(targets, dtype=np.float64)
    if x_validation.ndim != 2 or x_validation.shape[0] == 0:
        raise ValueError("validation inputs must be a non-empty 2-D array.")
    if x_validation.shape[1] != model.n_parameters:
        raise ValueError("validation inputs must match the surrogate parameter dimension.")
    if not np.all(np.isfinite(x_validation)):
        raise ValueError("validation inputs must contain only finite values.")
    if y_exact.ndim != 1 or y_exact.shape != (x_validation.shape[0],):
        raise ValueError("exact targets must match the validation rows.")
    if not np.all(np.isfinite(y_exact)):
        raise ValueError("exact targets must contain only finite values.")
    overlaps = set(input_row_digests(x_validation)) & set(model.training_input_digests)
    if overlaps:
        raise ValueError("validation inputs overlap surrogate training rows.")
    return x_validation, y_exact


def certify_surrogate_fidelity(
    model: GaussianRBFSurrogate,
    validation_inputs: FloatArray,
    exact_targets: FloatArray,
    *,
    thresholds: SurrogateFidelityThresholds,
) -> SurrogateFidelityCertificate:
    """Certify surrogate values on disjoint exact-simulator inputs.

    Parameters
    ----------
    model:
        Fitted Gaussian-RBF surrogate.
    validation_inputs:
        Inputs excluded from surrogate fitting.
    exact_targets:
        Exact local objective values at the validation inputs.
    thresholds:
        Acceptance thresholds frozen before certification.

    Returns
    -------
    SurrogateFidelityCertificate
        Digest-bound held-out metrics and pass/fail decision.

    """
    x_validation, y_exact = _validated_validation_data(model, validation_inputs, exact_targets)
    predictions = model.predict(x_validation)
    errors = predictions - y_exact
    rmse = float(np.sqrt(np.mean(errors**2)))
    mean_absolute_error = float(np.mean(np.abs(errors)))
    max_absolute_error = float(np.max(np.abs(errors)))
    target_variation = float(np.sum((y_exact - np.mean(y_exact)) ** 2))
    if target_variation == 0.0:
        raise ValueError("exact validation targets must have non-zero variation.")
    r_squared = float(1.0 - np.sum(errors**2) / target_variation)
    passed = (
        rmse <= thresholds.max_rmse
        and max_absolute_error <= thresholds.max_absolute_error
        and r_squared >= thresholds.min_r_squared
    )
    return SurrogateFidelityCertificate(
        n_training=int(model.centres.shape[0]),
        n_validation=int(x_validation.shape[0]),
        rmse=rmse,
        mean_absolute_error=mean_absolute_error,
        max_absolute_error=max_absolute_error,
        r_squared=r_squared,
        passed=passed,
        validation_input_digest=_array_digest(x_validation),
        exact_target_digest=_array_digest(y_exact),
        prediction_digest=_array_digest(predictions),
        thresholds=thresholds,
    )


def certify_surrogate_gradient(
    model: GaussianRBFSurrogate,
    validation_inputs: FloatArray,
    exact_objective: Callable[[FloatArray], float],
    *,
    finite_difference_step: float,
    max_absolute_error: float,
) -> SurrogateGradientCertificate:
    """Compare analytic surrogate gradients with exact central differences.

    The reference derivative evaluates the caller-supplied exact local
    objective at symmetric perturbations. It is not a hardware gradient or an
    exact analytic quantum derivative.
    """
    points = np.asarray(validation_inputs, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != model.n_parameters:
        raise ValueError("validation_inputs must be a non-empty dimension-matched 2-D array.")
    if not np.all(np.isfinite(points)):
        raise ValueError("validation_inputs must contain only finite values.")
    overlaps = set(input_row_digests(points)) & set(model.training_input_digests)
    if overlaps:
        raise ValueError("gradient validation inputs overlap surrogate training rows.")
    step = _positive_finite(finite_difference_step, name="finite_difference_step")
    maximum = _positive_finite(max_absolute_error, name="max_absolute_error")

    surrogate_gradients = np.vstack([model.gradient(point) for point in points])
    exact_gradients = np.zeros_like(surrogate_gradients)
    for row_index, point in enumerate(points):
        for parameter_index in range(points.shape[1]):
            offset = np.zeros(points.shape[1], dtype=np.float64)
            offset[parameter_index] = step
            plus = float(exact_objective(point + offset))
            minus = float(exact_objective(point - offset))
            if not np.isfinite(plus) or not np.isfinite(minus):
                raise ValueError("exact_objective must return finite values.")
            exact_gradients[row_index, parameter_index] = (plus - minus) / (2.0 * step)

    errors = surrogate_gradients - exact_gradients
    rmse = float(np.sqrt(np.mean(errors**2)))
    observed_maximum = float(np.max(np.abs(errors)))
    return SurrogateGradientCertificate(
        n_points=int(points.shape[0]),
        n_parameters=int(points.shape[1]),
        finite_difference_step=step,
        rmse=rmse,
        max_absolute_error=observed_maximum,
        max_allowed_error=maximum,
        passed=observed_maximum <= maximum,
        validation_input_digest=_array_digest(points),
        surrogate_gradient_digest=_array_digest(surrogate_gradients),
        exact_gradient_digest=_array_digest(exact_gradients),
    )


__all__ = [
    "SurrogateFidelityCertificate",
    "SurrogateFidelityThresholds",
    "SurrogateGradientCertificate",
    "certify_surrogate_fidelity",
    "certify_surrogate_gradient",
]
