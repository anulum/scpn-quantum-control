# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- stochastic differentiable evidence policies
"""Fail-closed stochastic-gradient uncertainty policy contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_parameter_contracts import _as_parameter_array, _as_real_scalar

STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY = (
    "materialised parameter-shift finite-shot uncertainty propagation; shifted means, "
    "variances, shot counts, coefficients, trainable masks, confidence policy, and "
    "failure reasons are recorded; no provider callback or hardware execution"
)


@dataclass(frozen=True)
class GradientFailurePolicy:
    """Fail-closed policy for stochastic-gradient uncertainty evidence."""

    max_standard_error: float | None = None
    max_confidence_radius: float | None = None
    require_trainable: bool = True

    def __post_init__(self) -> None:
        max_standard_error = (
            None
            if self.max_standard_error is None
            else _as_real_scalar("failure_policy max_standard_error", self.max_standard_error)
        )
        max_confidence_radius = (
            None
            if self.max_confidence_radius is None
            else _as_real_scalar(
                "failure_policy max_confidence_radius",
                self.max_confidence_radius,
            )
        )
        if max_standard_error is not None and max_standard_error <= 0.0:
            raise ValueError("failure_policy max_standard_error must be finite and positive")
        if max_confidence_radius is not None and max_confidence_radius <= 0.0:
            raise ValueError("failure_policy max_confidence_radius must be finite and positive")
        if not isinstance(self.require_trainable, bool):
            raise ValueError("failure_policy require_trainable must be boolean")
        object.__setattr__(self, "max_standard_error", max_standard_error)
        object.__setattr__(self, "max_confidence_radius", max_confidence_radius)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready policy metadata."""
        return {
            "max_standard_error": self.max_standard_error,
            "max_confidence_radius": self.max_confidence_radius,
            "require_trainable": self.require_trainable,
        }


@dataclass(frozen=True)
class StochasticGradientConfidenceInterval:
    """Per-parameter stochastic-gradient confidence interval and policy status."""

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    confidence_z: float
    confidence_level: float | None
    policy: GradientFailurePolicy
    status: str
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        lower = _as_parameter_array(self.lower)
        upper = _as_parameter_array(self.upper)
        z_value = _as_real_scalar("confidence interval confidence_z", self.confidence_z)
        confidence_level = (
            None
            if self.confidence_level is None
            else _as_real_scalar("confidence interval confidence_level", self.confidence_level)
        )
        if lower.shape != upper.shape:
            raise ValueError("confidence interval lower/upper shapes must match")
        if np.any(lower > upper):
            raise ValueError("confidence interval lower bounds must not exceed upper bounds")
        if z_value <= 0.0:
            raise ValueError("confidence interval confidence_z must be finite and positive")
        if confidence_level is not None and not (0.0 < confidence_level < 1.0):
            raise ValueError("confidence interval confidence_level must be between zero and one")
        if self.status not in {"passed", "failed"}:
            raise ValueError("confidence interval status must be 'passed' or 'failed'")
        reasons = tuple(str(reason) for reason in self.failure_reasons)
        if self.status == "passed" and reasons:
            raise ValueError("passed confidence interval cannot contain failure reasons")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "confidence_z", z_value)
        object.__setattr__(self, "confidence_level", confidence_level)
        object.__setattr__(self, "failure_reasons", reasons)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready interval metadata."""
        return {
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "confidence_z": self.confidence_z,
            "confidence_level": self.confidence_level,
            "policy": self.policy.to_dict(),
            "status": self.status,
            "failure_reasons": list(self.failure_reasons),
        }


def gradient_confidence_interval(
    gradient: ArrayLike,
    standard_error: ArrayLike,
    *,
    confidence_z: float = 1.959963984540054,
    confidence_level: float | None = None,
    trainable: Sequence[bool] | None = None,
    failure_policy: GradientFailurePolicy | None = None,
) -> StochasticGradientConfidenceInterval:
    """Build confidence bounds and fail-closed status for stochastic gradients."""
    gradient_array = _as_parameter_array(gradient)
    standard_error_array = _as_parameter_array(standard_error)
    if standard_error_array.shape != gradient_array.shape:
        raise ValueError("standard_error shape must match gradient shape")
    if np.any(standard_error_array < 0.0):
        raise ValueError("standard_error must be finite and non-negative")
    z_value = _as_real_scalar("confidence_z", confidence_z)
    if z_value <= 0.0:
        raise ValueError("confidence_z must be finite and positive")
    if confidence_level is not None:
        confidence_level = _as_real_scalar("confidence_level", confidence_level)
        if confidence_level <= 0.0 or confidence_level >= 1.0:
            raise ValueError("confidence_level must be between zero and one")
    trainable_mask: NDArray[np.bool_]
    if trainable is None:
        trainable_mask = np.ones(gradient_array.shape, dtype=np.bool_)
    else:
        trainable_mask = np.asarray(tuple(trainable), dtype=np.bool_)
        if trainable_mask.shape != gradient_array.shape:
            raise ValueError("trainable mask length must match gradient length")
    policy = failure_policy or GradientFailurePolicy()
    reasons: list[str] = []
    if policy.require_trainable and not np.any(trainable_mask):
        raise ValueError("trainable mask must include at least one active parameter")
    active_standard_error = standard_error_array[trainable_mask]
    confidence_radius = z_value * standard_error_array
    active_radius = confidence_radius[trainable_mask]
    if policy.max_standard_error is not None and active_standard_error.size:
        observed = float(np.max(active_standard_error))
        if observed > policy.max_standard_error:
            reasons.append(
                "standard_error exceeds policy maximum "
                f"{policy.max_standard_error}: observed {observed}"
            )
    if policy.max_confidence_radius is not None and active_radius.size:
        observed = float(np.max(active_radius))
        if observed > policy.max_confidence_radius:
            reasons.append(
                "confidence_radius exceeds policy maximum "
                f"{policy.max_confidence_radius}: observed {observed}"
            )
    return StochasticGradientConfidenceInterval(
        lower=gradient_array - confidence_radius,
        upper=gradient_array + confidence_radius,
        confidence_z=z_value,
        confidence_level=confidence_level,
        policy=policy,
        status="failed" if reasons else "passed",
        failure_reasons=tuple(reasons),
    )


__all__ = [
    "GradientFailurePolicy",
    "STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY",
    "StochasticGradientConfidenceInterval",
    "gradient_confidence_interval",
]
