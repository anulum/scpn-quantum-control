# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-35 differentiable calibration objective
"""Analytic design-unit coupling-scale objective and drift sensitivity."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
CALIBRATION_BOUNDARY = (
    "Analytic design-unit matrix fit only; gradient is not a calibrated pulse gradient, "
    "device response, closed-loop measurement, or provider execution result"
)


@dataclass(frozen=True, slots=True)
class CalibrationEvaluation:
    """Scalar coupling-scale objective and exact analytic derivative."""

    scale: float
    loss: float
    gradient: float
    boundary: str = CALIBRATION_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready objective evaluation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CalibrationSensitivity:
    """Nominal and symmetric fractional-drift objective evaluations."""

    relative_drift: float
    nominal: CalibrationEvaluation
    minus_drift: CalibrationEvaluation
    plus_drift: CalibrationEvaluation
    worst_case_loss: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready sensitivity record."""
        return {
            "relative_drift": self.relative_drift,
            "nominal": self.nominal.to_dict(),
            "minus_drift": self.minus_drift.to_dict(),
            "plus_drift": self.plus_drift.to_dict(),
            "worst_case_loss": self.worst_case_loss,
        }


def coupling_scale_objective(
    native_couplings: FloatArray,
    target_couplings: FloatArray,
    *,
    scale: float,
) -> CalibrationEvaluation:
    """Evaluate mean-squared upper-triangle coupling mismatch and its derivative."""
    native, target = _validated_pair(native_couplings, target_couplings)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    upper = np.triu_indices(native.shape[0], k=1)
    residual = scale * native[upper] - target[upper]
    loss = float(np.mean(residual**2))
    gradient = float(2.0 * np.mean(residual * native[upper]))
    return CalibrationEvaluation(scale=float(scale), loss=loss, gradient=gradient)


def calibration_sensitivity(
    native_couplings: FloatArray,
    target_couplings: FloatArray,
    *,
    nominal_scale: float,
    relative_drift: float = 0.05,
) -> CalibrationSensitivity:
    """Evaluate the analytic objective at nominal and symmetric scale drift."""
    if not math.isfinite(relative_drift) or not 0.0 < relative_drift <= 1.0:
        raise ValueError("relative_drift must be finite with 0 < value <= 1")
    nominal = coupling_scale_objective(
        native_couplings,
        target_couplings,
        scale=nominal_scale,
    )
    minus = coupling_scale_objective(
        native_couplings,
        target_couplings,
        scale=nominal_scale * (1.0 - relative_drift),
    )
    plus = coupling_scale_objective(
        native_couplings,
        target_couplings,
        scale=nominal_scale * (1.0 + relative_drift),
    )
    return CalibrationSensitivity(
        relative_drift=relative_drift,
        nominal=nominal,
        minus_drift=minus,
        plus_drift=plus,
        worst_case_loss=max(minus.loss, plus.loss),
    )


def _validated_pair(
    native_couplings: FloatArray,
    target_couplings: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    native = np.asarray(native_couplings, dtype=np.float64)
    target = np.asarray(target_couplings, dtype=np.float64)
    if native.ndim != 2 or native.shape[0] != native.shape[1] or native.shape[0] < 2:
        raise ValueError("native_couplings must be a square matrix with at least two nodes")
    if target.shape != native.shape:
        raise ValueError("target_couplings must match native_couplings shape")
    if not np.all(np.isfinite(native)) or not np.all(np.isfinite(target)):
        raise ValueError("coupling matrices must contain finite values")
    if not np.allclose(native, native.T) or not np.allclose(target, target.T):
        raise ValueError("coupling matrices must be symmetric")
    return native, target


__all__ = [
    "CALIBRATION_BOUNDARY",
    "CalibrationEvaluation",
    "CalibrationSensitivity",
    "calibration_sensitivity",
    "coupling_scale_objective",
]
