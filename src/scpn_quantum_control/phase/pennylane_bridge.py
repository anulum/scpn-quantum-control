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
    )


__all__ = [
    "PennyLaneGradientAgreementResult",
    "check_pennylane_parameter_shift_agreement",
    "is_phase_pennylane_available",
]
