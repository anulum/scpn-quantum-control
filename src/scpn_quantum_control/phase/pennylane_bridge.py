# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase PennyLane Bridge
"""Optional PennyLane agreement checks for phase parameter-shift gradients."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import Parameter, ParameterShiftRule, value_and_parameter_shift_grad

FloatArray = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]
GradientCallable = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True)
class PennyLaneGradientAgreementResult:
    """Agreement report between SCPN and PennyLane-style gradient callables."""

    value: float
    scpn_gradient: FloatArray
    pennylane_gradient: FloatArray
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    evaluations: int
    method: str = "parameter_shift"
    shift_terms: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable agreement metadata."""
        return {
            "value": self.value,
            "scpn_gradient": self.scpn_gradient.copy(),
            "pennylane_gradient": self.pennylane_gradient.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "evaluations": self.evaluations,
            "method": self.method,
            "shift_terms": self.shift_terms,
        }


@dataclass(frozen=True)
class PennyLaneRoundTripResult:
    """Round-trip value and gradient agreement report for PennyLane QNode adapters."""

    scpn_value: float
    pennylane_value: float
    value_abs_error: float
    scpn_gradient: FloatArray
    pennylane_gradient: FloatArray
    gradient_max_abs_error: float
    gradient_l2_error: float
    value_tolerance: float
    gradient_tolerance: float
    passed: bool
    evaluations: int
    method: str = "parameter_shift"
    shift_terms: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable round-trip metadata."""
        return {
            "scpn_value": self.scpn_value,
            "pennylane_value": self.pennylane_value,
            "value_abs_error": self.value_abs_error,
            "scpn_gradient": self.scpn_gradient.tolist(),
            "pennylane_gradient": self.pennylane_gradient.tolist(),
            "gradient_max_abs_error": self.gradient_max_abs_error,
            "gradient_l2_error": self.gradient_l2_error,
            "value_tolerance": self.value_tolerance,
            "gradient_tolerance": self.gradient_tolerance,
            "passed": self.passed,
            "evaluations": self.evaluations,
            "method": self.method,
            "shift_terms": self.shift_terms,
        }


def _load_pennylane() -> Any:
    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError(
            "PennyLane is unavailable; install scpn-quantum-control[pennylane]"
        ) from exc
    return qml


def is_phase_pennylane_available() -> bool:
    """Return whether the optional phase PennyLane bridge can import PennyLane."""
    try:
        _load_pennylane()
    except ImportError:
        return False
    return True


def _as_parameter_vector(name: str, values: object, *, width: int | None = None) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _as_non_negative_tolerance(value: float) -> float:
    tolerance = float(value)
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    return tolerance


def _as_finite_scalar(name: str, value: object) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must be a real numeric scalar")
    scalar = float(raw)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def check_pennylane_parameter_shift_agreement(
    objective: ScalarObjective,
    pennylane_gradient: GradientCallable,
    values: ArrayLike,
    *,
    tolerance: float = 1e-6,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PennyLaneGradientAgreementResult:
    """Compare SCPN parameter-shift gradients with a PennyLane gradient callable.

    ``pennylane_gradient`` is intentionally caller-supplied. It can be
    ``qml.grad(qnode)`` or any strict PennyLane-derived gradient function with
    the same one-dimensional parameter vector. This keeps the bridge honest:
    it verifies cross-framework agreement without claiming automatic QNode
    generation for every SCPN ansatz.
    """
    _load_pennylane()
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("values", values)
    scpn = value_and_parameter_shift_grad(
        objective,
        parameter_values,
        parameters=parameters,
        rule=rule,
    )
    shift_terms = len((rule or ParameterShiftRule()).terms)
    external_gradient = _as_parameter_vector(
        "PennyLane gradient",
        pennylane_gradient(parameter_values.copy()),
        width=parameter_values.size,
    )
    delta = scpn.gradient - external_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta, ord=2))
    return PennyLaneGradientAgreementResult(
        value=float(scpn.value),
        scpn_gradient=scpn.gradient.copy(),
        pennylane_gradient=external_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=max_abs_error <= tolerance_value,
        evaluations=scpn.evaluations,
        method=scpn.method,
        shift_terms=shift_terms,
    )


def check_pennylane_qnode_round_trip(
    scpn_objective: ScalarObjective,
    pennylane_objective: ScalarObjective,
    pennylane_gradient: GradientCallable,
    values: ArrayLike,
    *,
    value_tolerance: float = 1e-8,
    gradient_tolerance: float = 1e-6,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PennyLaneRoundTripResult:
    """Verify SCPN value/gradient parity against a caller-supplied PennyLane QNode.

    The PennyLane callables are supplied by the caller so the bridge stays
    explicit about what is actually compared. In a real PennyLane workflow,
    ``pennylane_objective`` should be a QNode and ``pennylane_gradient`` should
    come from PennyLane autodiff, for example ``qml.grad(qnode)``.
    """
    _load_pennylane()
    value_tol = _as_non_negative_tolerance(value_tolerance)
    gradient_tol = _as_non_negative_tolerance(gradient_tolerance)
    parameter_values = _as_parameter_vector("values", values)
    scpn = value_and_parameter_shift_grad(
        scpn_objective,
        parameter_values,
        parameters=parameters,
        rule=rule,
    )
    shift_terms = len((rule or ParameterShiftRule()).terms)
    external_value = _as_finite_scalar(
        "PennyLane objective",
        pennylane_objective(parameter_values.copy()),
    )
    external_gradient = _as_parameter_vector(
        "PennyLane gradient",
        pennylane_gradient(parameter_values.copy()),
        width=parameter_values.size,
    )
    delta = scpn.gradient - external_gradient
    value_abs_error = abs(float(scpn.value) - external_value)
    gradient_max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    gradient_l2_error = float(np.linalg.norm(delta, ord=2))

    return PennyLaneRoundTripResult(
        scpn_value=float(scpn.value),
        pennylane_value=external_value,
        value_abs_error=value_abs_error,
        scpn_gradient=scpn.gradient.copy(),
        pennylane_gradient=external_gradient,
        gradient_max_abs_error=gradient_max_abs_error,
        gradient_l2_error=gradient_l2_error,
        value_tolerance=value_tol,
        gradient_tolerance=gradient_tol,
        passed=bool(value_abs_error <= value_tol and gradient_max_abs_error <= gradient_tol),
        evaluations=scpn.evaluations,
        method=scpn.method,
        shift_terms=shift_terms,
    )


__all__ = [
    "PennyLaneGradientAgreementResult",
    "PennyLaneRoundTripResult",
    "check_pennylane_parameter_shift_agreement",
    "check_pennylane_qnode_round_trip",
    "is_phase_pennylane_available",
]
