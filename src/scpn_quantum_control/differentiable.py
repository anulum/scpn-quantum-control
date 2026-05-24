# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- native differentiable programming primitives
"""Native differentiable-programming primitives for SCPN quantum objectives.

The base layer is backend-neutral parameter-shift differentiation for scalar
objectives. Optional JAX support is exposed as an adapter without making JAX a
runtime dependency of the core package.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

ScalarObjective = Callable[[NDArray[np.float64]], float | int | np.floating[Any]]
VectorObjective = Callable[[NDArray[np.float64]], ArrayLike]


def _as_real_numeric_array(name: str, values: object) -> NDArray[np.float64]:
    """Return a real numeric vector without implicit string/bool/object coercion."""
    try:
        raw = np.asarray(values)
    except ValueError as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc

    if raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must contain real numeric scalars")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric scalars") from exc
    return cast(NDArray[np.float64], array)


def _as_real_scalar(name: str, value: object) -> float:
    """Return an explicit real numeric scalar without implicit coercion."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real numeric scalar")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must be a real numeric scalar")
    scalar = float(raw)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


@dataclass(frozen=True)
class Parameter:
    """One differentiable scalar parameter in an SCPN objective."""

    name: str
    trainable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("parameter name must be non-empty")
        if not isinstance(self.trainable, bool):
            raise ValueError("parameter trainable flag must be a boolean")


@dataclass(frozen=True)
class ParameterBounds:
    """Closed interval constraint for one differentiable scalar parameter."""

    lower: float | None = None
    upper: float | None = None
    periodic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.periodic, bool):
            raise ValueError("periodic flag must be a boolean")
        lower = None if self.lower is None else _as_real_scalar("lower bound", self.lower)
        upper = None if self.upper is None else _as_real_scalar("upper bound", self.upper)
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower bound must be less than or equal to upper bound")
        if self.periodic:
            if lower is None or upper is None:
                raise ValueError("periodic bounds require finite lower and upper bounds")
            if lower == upper:
                raise ValueError("periodic bounds require lower < upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class ParameterShiftRule:
    """Two-point parameter-shift rule for one-generator rotation parameters."""

    shift: float = float(np.pi / 2.0)
    coefficient: float = 0.5

    def __post_init__(self) -> None:
        shift = _as_real_scalar("shift", self.shift)
        coefficient = _as_real_scalar("coefficient", self.coefficient)
        if shift <= 0.0:
            raise ValueError("shift must be finite and positive")
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "coefficient", coefficient)


@dataclass(frozen=True)
class GradientResult:
    """Value, gradient, and provenance returned by a differentiable backend."""

    value: float
    gradient: NDArray[np.float64]
    method: str
    shift: float | None
    coefficient: float | None
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_scalar("gradient result value", self.value)
        gradient = _as_real_numeric_array("gradient", self.gradient)
        if gradient.ndim != 1:
            raise ValueError("gradient must be a one-dimensional array")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("gradient must contain only finite values")
        if not self.method:
            raise ValueError("gradient method must be non-empty")
        shift = None if self.shift is None else _as_real_scalar("gradient shift", self.shift)
        coefficient = (
            None
            if self.coefficient is None
            else _as_real_scalar("gradient coefficient", self.coefficient)
        )
        if shift is not None and shift <= 0.0:
            raise ValueError("gradient shift must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("gradient evaluations must be non-negative")
        if len(self.parameter_names) != gradient.size:
            raise ValueError("parameter_names length must match gradient length")
        if len(self.trainable) != gradient.size:
            raise ValueError("trainable mask length must match gradient length")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "coefficient", coefficient)


@dataclass(frozen=True)
class OptimizationResult:
    """Bounded gradient-descent result with convergence provenance."""

    values: NDArray[np.float64]
    final_gradient: GradientResult
    value_history: tuple[float, ...]
    steps: int
    converged: bool
    reason: str
    best_values: NDArray[np.float64] | None = None
    best_value: float | None = None

    def __post_init__(self) -> None:
        values = _as_parameter_array(self.values)
        if values.size != self.final_gradient.gradient.size:
            raise ValueError("optimized values length must match gradient length")
        if not self.value_history:
            raise ValueError("value_history must contain at least one value")
        history = tuple(_as_real_scalar("value_history item", item) for item in self.value_history)
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 0:
            raise ValueError("optimization steps must be a non-negative integer")
        if not isinstance(self.converged, bool):
            raise ValueError("optimization converged flag must be a boolean")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("optimization reason must be non-empty")
        best_values = values if self.best_values is None else _as_parameter_array(self.best_values)
        if best_values.size != values.size:
            raise ValueError("best_values length must match optimized values length")
        best_value = (
            min(history)
            if self.best_value is None
            else _as_real_scalar("best_value", self.best_value)
        )
        if best_value > min(history) + 1.0e-12:
            raise ValueError("best_value must not exceed the minimum value_history entry")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "value_history", history)
        object.__setattr__(self, "best_values", best_values)
        object.__setattr__(self, "best_value", best_value)


@dataclass(frozen=True)
class GradientCheckResult:
    """Consistency check between two differentiable gradient estimators."""

    reference: GradientResult
    candidate: GradientResult
    max_abs_error: float
    l2_error: float
    value_delta: float
    tolerance: float
    passed: bool

    def __post_init__(self) -> None:
        if self.reference.gradient.shape != self.candidate.gradient.shape:
            raise ValueError("gradient check operands must have matching shapes")
        max_abs_error = _as_real_scalar("max_abs_error", self.max_abs_error)
        l2_error = _as_real_scalar("l2_error", self.l2_error)
        value_delta = _as_real_scalar("value_delta", self.value_delta)
        tolerance = _as_real_scalar("tolerance", self.tolerance)
        if max_abs_error < 0.0:
            raise ValueError("max_abs_error must be non-negative")
        if l2_error < 0.0:
            raise ValueError("l2_error must be non-negative")
        if value_delta < 0.0:
            raise ValueError("value_delta must be non-negative")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        if not isinstance(self.passed, bool):
            raise ValueError("gradient check passed flag must be a boolean")
        object.__setattr__(self, "max_abs_error", max_abs_error)
        object.__setattr__(self, "l2_error", l2_error)
        object.__setattr__(self, "value_delta", value_delta)
        object.__setattr__(self, "tolerance", tolerance)


@dataclass(frozen=True)
class JacobianResult:
    """Value, Jacobian, and provenance for a vector-valued objective."""

    value: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_numeric_array("jacobian value", self.value)
        jacobian = _as_real_numeric_array("jacobian", self.jacobian)
        if value.ndim != 1:
            raise ValueError("jacobian value must be a one-dimensional array")
        if jacobian.ndim != 2:
            raise ValueError("jacobian must be a two-dimensional array")
        if jacobian.shape[0] != value.size:
            raise ValueError("jacobian row count must match value length")
        if not np.all(np.isfinite(value)):
            raise ValueError("jacobian value must contain only finite values")
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("jacobian must contain only finite values")
        if not self.method:
            raise ValueError("jacobian method must be non-empty")
        step = _as_real_scalar("jacobian step", self.step)
        if step <= 0.0:
            raise ValueError("jacobian step must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("jacobian evaluations must be non-negative")
        if len(self.parameter_names) != jacobian.shape[1]:
            raise ValueError("parameter_names length must match jacobian column count")
        if len(self.trainable) != jacobian.shape[1]:
            raise ValueError("trainable mask length must match jacobian column count")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "jacobian", jacobian)
        object.__setattr__(self, "step", step)


@dataclass(frozen=True)
class JVPResult:
    """Jacobian-vector product with directional finite-difference provenance."""

    value: NDArray[np.float64]
    jvp: NDArray[np.float64]
    tangent: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_numeric_array("JVP value", self.value)
        jvp = _as_real_numeric_array("JVP", self.jvp)
        tangent = _as_real_numeric_array("JVP tangent", self.tangent)
        if value.ndim != 1:
            raise ValueError("JVP value must be a one-dimensional array")
        if jvp.shape != value.shape:
            raise ValueError("JVP shape must match value shape")
        if tangent.ndim != 1:
            raise ValueError("JVP tangent must be one-dimensional")
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(jvp)):
            raise ValueError("JVP value and product must contain only finite values")
        if not np.all(np.isfinite(tangent)):
            raise ValueError("JVP tangent must contain only finite values")
        if not self.method:
            raise ValueError("JVP method must be non-empty")
        step = _as_real_scalar("JVP step", self.step)
        if step <= 0.0:
            raise ValueError("JVP step must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("JVP evaluations must be non-negative")
        if len(self.parameter_names) != tangent.size:
            raise ValueError("parameter_names length must match tangent length")
        if len(self.trainable) != tangent.size:
            raise ValueError("trainable mask length must match tangent length")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "jvp", jvp)
        object.__setattr__(self, "tangent", tangent)
        object.__setattr__(self, "step", step)


@dataclass(frozen=True)
class VJPResult:
    """Vector-Jacobian product with cotangent provenance."""

    value: NDArray[np.float64]
    cotangent: NDArray[np.float64]
    vjp: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_numeric_array("VJP value", self.value)
        cotangent = _as_real_numeric_array("VJP cotangent", self.cotangent)
        vjp = _as_real_numeric_array("VJP", self.vjp)
        if value.ndim != 1:
            raise ValueError("VJP value must be a one-dimensional array")
        if cotangent.shape != value.shape:
            raise ValueError("VJP cotangent shape must match value shape")
        if vjp.ndim != 1:
            raise ValueError("VJP must be one-dimensional")
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(cotangent)):
            raise ValueError("VJP value and cotangent must contain only finite values")
        if not np.all(np.isfinite(vjp)):
            raise ValueError("VJP must contain only finite values")
        if not self.method:
            raise ValueError("VJP method must be non-empty")
        step = _as_real_scalar("VJP step", self.step)
        if step <= 0.0:
            raise ValueError("VJP step must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("VJP evaluations must be non-negative")
        if len(self.parameter_names) != vjp.size:
            raise ValueError("parameter_names length must match VJP length")
        if len(self.trainable) != vjp.size:
            raise ValueError("trainable mask length must match VJP length")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "cotangent", cotangent)
        object.__setattr__(self, "vjp", vjp)
        object.__setattr__(self, "step", step)


@dataclass(frozen=True)
class HessianResult:
    """Value, Hessian, and provenance for a scalar objective."""

    value: float
    hessian: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_scalar("hessian value", self.value)
        hessian = _as_real_numeric_array("hessian", self.hessian)
        if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
            raise ValueError("hessian must be a square two-dimensional array")
        if not np.all(np.isfinite(hessian)):
            raise ValueError("hessian must contain only finite values")
        if not self.method:
            raise ValueError("hessian method must be non-empty")
        step = _as_real_scalar("hessian step", self.step)
        if step <= 0.0:
            raise ValueError("hessian step must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("hessian evaluations must be non-negative")
        if len(self.parameter_names) != hessian.shape[1]:
            raise ValueError("parameter_names length must match hessian dimension")
        if len(self.trainable) != hessian.shape[1]:
            raise ValueError("trainable mask length must match hessian dimension")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        if not np.allclose(hessian, hessian.T, atol=1.0e-8, rtol=1.0e-8):
            raise ValueError("hessian must be symmetric")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "hessian", hessian)
        object.__setattr__(self, "step", step)


@dataclass(frozen=True)
class HVPResult:
    """Hessian-vector product with nested finite-difference provenance."""

    value: float
    hvp: NDArray[np.float64]
    tangent: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_scalar("HVP value", self.value)
        hvp = _as_real_numeric_array("HVP", self.hvp)
        tangent = _as_real_numeric_array("HVP tangent", self.tangent)
        if hvp.ndim != 1:
            raise ValueError("HVP must be one-dimensional")
        if tangent.shape != hvp.shape:
            raise ValueError("HVP tangent shape must match HVP shape")
        if not np.all(np.isfinite(hvp)) or not np.all(np.isfinite(tangent)):
            raise ValueError("HVP and tangent must contain only finite values")
        if not self.method:
            raise ValueError("HVP method must be non-empty")
        step = _as_real_scalar("HVP step", self.step)
        if step <= 0.0:
            raise ValueError("HVP step must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("HVP evaluations must be non-negative")
        if len(self.parameter_names) != hvp.size:
            raise ValueError("parameter_names length must match HVP length")
        if len(self.trainable) != hvp.size:
            raise ValueError("trainable mask length must match HVP length")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "hvp", hvp)
        object.__setattr__(self, "tangent", tangent)
        object.__setattr__(self, "step", step)


@dataclass(frozen=True)
class NaturalGradientResult:
    """Metric-preconditioned gradient with solve provenance."""

    base_gradient: GradientResult
    metric: NDArray[np.float64]
    natural_gradient: NDArray[np.float64]
    damping: float
    condition_number: float

    def __post_init__(self) -> None:
        metric = _as_real_numeric_array("natural-gradient metric", self.metric)
        natural_gradient = _as_real_numeric_array("natural_gradient", self.natural_gradient)
        if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
            raise ValueError("natural-gradient metric must be a square matrix")
        if metric.shape[0] != self.base_gradient.gradient.size:
            raise ValueError("natural-gradient metric dimension must match gradient length")
        if natural_gradient.shape != self.base_gradient.gradient.shape:
            raise ValueError("natural_gradient shape must match gradient shape")
        if not np.all(np.isfinite(metric)):
            raise ValueError("natural-gradient metric must contain only finite values")
        if not np.all(np.isfinite(natural_gradient)):
            raise ValueError("natural_gradient must contain only finite values")
        if not np.allclose(metric, metric.T, atol=1.0e-10, rtol=1.0e-10):
            raise ValueError("natural-gradient metric must be symmetric")
        damping = _as_real_scalar("natural-gradient damping", self.damping)
        if damping < 0.0:
            raise ValueError("natural-gradient damping must be finite and non-negative")
        condition_number = _as_real_scalar(
            "natural-gradient condition_number", self.condition_number
        )
        if condition_number < 1.0:
            raise ValueError("natural-gradient condition_number must be at least 1")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "natural_gradient", natural_gradient)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "condition_number", condition_number)


@dataclass(frozen=True)
class LevenbergMarquardtStep:
    """Bounded Levenberg-Marquardt candidate step with model diagnostics."""

    gauss_newton: NaturalGradientResult
    step: NDArray[np.float64]
    candidate_values: NDArray[np.float64]
    damping: float
    predicted_reduction: float

    def __post_init__(self) -> None:
        step = _as_real_numeric_array("Levenberg-Marquardt step", self.step)
        candidate_values = _as_real_numeric_array(
            "Levenberg-Marquardt candidate_values",
            self.candidate_values,
        )
        if step.ndim != 1:
            raise ValueError("Levenberg-Marquardt step must be one-dimensional")
        if candidate_values.shape != step.shape:
            raise ValueError("candidate_values shape must match step shape")
        if step.shape != self.gauss_newton.base_gradient.gradient.shape:
            raise ValueError("step shape must match Gauss-Newton gradient shape")
        if not np.all(np.isfinite(step)):
            raise ValueError("Levenberg-Marquardt step must contain only finite values")
        if not np.all(np.isfinite(candidate_values)):
            raise ValueError("candidate_values must contain only finite values")
        damping = _as_real_scalar("Levenberg-Marquardt damping", self.damping)
        if damping < 0.0:
            raise ValueError("Levenberg-Marquardt damping must be finite and non-negative")
        predicted_reduction = _as_real_scalar(
            "Levenberg-Marquardt predicted_reduction",
            self.predicted_reduction,
        )
        if predicted_reduction < -1.0e-12:
            raise ValueError("predicted_reduction must be non-negative")
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "candidate_values", candidate_values)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "predicted_reduction", max(0.0, predicted_reduction))


@dataclass(frozen=True)
class LevenbergMarquardtTrial:
    """Actual-vs-predicted Levenberg-Marquardt acceptance diagnostic."""

    step_result: LevenbergMarquardtStep
    candidate_residual: NDArray[np.float64]
    candidate_value: float
    actual_reduction: float
    reduction_ratio: float
    accepted: bool

    def __post_init__(self) -> None:
        candidate_residual = _as_real_numeric_array(
            "Levenberg-Marquardt candidate_residual",
            self.candidate_residual,
        )
        if candidate_residual.ndim != 1:
            raise ValueError("candidate_residual must be one-dimensional")
        if not np.all(np.isfinite(candidate_residual)):
            raise ValueError("candidate_residual must contain only finite values")
        candidate_value = _as_real_scalar(
            "Levenberg-Marquardt candidate_value",
            self.candidate_value,
        )
        actual_reduction = _as_real_scalar(
            "Levenberg-Marquardt actual_reduction",
            self.actual_reduction,
        )
        reduction_ratio = _as_real_scalar(
            "Levenberg-Marquardt reduction_ratio",
            self.reduction_ratio,
        )
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted flag must be a boolean")
        object.__setattr__(self, "candidate_residual", candidate_residual)
        object.__setattr__(self, "candidate_value", candidate_value)
        object.__setattr__(self, "actual_reduction", actual_reduction)
        object.__setattr__(self, "reduction_ratio", reduction_ratio)


@dataclass(frozen=True)
class LevenbergMarquardtDampingUpdate:
    """Deterministic damping update for Levenberg-Marquardt trust regions."""

    trial: LevenbergMarquardtTrial
    next_damping: float
    action: str

    def __post_init__(self) -> None:
        next_damping = _as_real_scalar(
            "Levenberg-Marquardt next_damping",
            self.next_damping,
        )
        if next_damping < 0.0:
            raise ValueError("next_damping must be finite and non-negative")
        if self.action not in {"accept_decrease", "accept_keep", "reject_increase"}:
            raise ValueError("damping action must be a known Levenberg-Marquardt action")
        object.__setattr__(self, "next_damping", next_damping)


@dataclass(frozen=True)
class LevenbergMarquardtResult:
    """Traceable result from a bounded Levenberg-Marquardt optimization run."""

    values: NDArray[np.float64]
    residual: NDArray[np.float64]
    value_history: tuple[float, ...]
    damping_history: tuple[float, ...]
    accepted_history: tuple[bool, ...]
    steps: int
    converged: bool
    reason: str
    best_values: NDArray[np.float64]
    best_value: float

    def __post_init__(self) -> None:
        values = _as_parameter_array(self.values)
        residual = _as_vector_output(self.residual)
        best_values = _as_parameter_array(self.best_values)
        if best_values.shape != values.shape:
            raise ValueError("LM best values must match result values shape")
        if not self.value_history:
            raise ValueError("LM value history must contain the initial objective")
        value_history = tuple(
            _as_real_scalar("LM objective history value", value) for value in self.value_history
        )
        damping_history = tuple(
            _as_real_scalar("LM damping history value", value) for value in self.damping_history
        )
        accepted_history = tuple(bool(value) for value in self.accepted_history)
        steps = int(self.steps)
        if steps < 0:
            raise ValueError("LM result steps must be non-negative")
        if any(value < 0.0 for value in damping_history):
            raise ValueError("LM damping history must contain finite non-negative values")
        if len(accepted_history) != steps:
            raise ValueError("LM accepted history length must match executed steps")
        if len(damping_history) != steps + 1:
            raise ValueError(
                "LM damping history must include initial damping plus one entry per step"
            )
        if len(value_history) != steps + 1:
            raise ValueError("LM value history must include initial value plus one entry per step")
        best_value = _as_real_scalar("LM best objective", self.best_value)
        if best_value > min(value_history) + 1.0e-12:
            raise ValueError("LM best objective must be no larger than the recorded minimum")
        if self.reason not in {
            "residual_tolerance",
            "step_tolerance",
            "value_tolerance",
            "max_steps",
        }:
            raise ValueError("LM result reason must be a known convergence status")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "value_history", value_history)
        object.__setattr__(self, "damping_history", damping_history)
        object.__setattr__(self, "accepted_history", accepted_history)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "best_values", best_values)
        object.__setattr__(self, "best_value", best_value)


@dataclass(frozen=True)
class LeastSquaresCovarianceResult:
    """Parameter uncertainty estimate from a residual-map Fisher metric."""

    covariance: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    residual_variance: float
    degrees_of_freedom: int
    condition_number: float
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        covariance = _as_real_numeric_array("least-squares covariance", self.covariance)
        standard_errors = _as_real_numeric_array(
            "least-squares standard errors",
            self.standard_errors,
        )
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("least-squares covariance must be a square matrix")
        if standard_errors.ndim != 1 or standard_errors.shape[0] != covariance.shape[0]:
            raise ValueError("standard_errors length must match covariance dimension")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("least-squares covariance must contain only finite values")
        if not np.allclose(covariance, covariance.T, atol=1.0e-10):
            raise ValueError("least-squares covariance must be symmetric")
        if not np.all(np.isfinite(standard_errors)) or np.any(standard_errors < 0.0):
            raise ValueError("standard_errors must contain finite non-negative values")
        residual_variance = _as_real_scalar(
            "least-squares residual_variance",
            self.residual_variance,
        )
        if residual_variance < 0.0:
            raise ValueError("residual_variance must be finite and non-negative")
        degrees_of_freedom = int(self.degrees_of_freedom)
        if degrees_of_freedom < 1:
            raise ValueError("degrees_of_freedom must be positive")
        condition_number = _as_real_scalar(
            "least-squares condition_number",
            self.condition_number,
        )
        if condition_number < 1.0:
            raise ValueError("condition_number must be at least one")
        if len(self.parameter_names) != covariance.shape[0]:
            raise ValueError("parameter_names length must match covariance dimension")
        if len(self.trainable) != covariance.shape[0]:
            raise ValueError("trainable mask length must match covariance dimension")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "standard_errors", standard_errors)
        object.__setattr__(self, "residual_variance", residual_variance)
        object.__setattr__(self, "degrees_of_freedom", degrees_of_freedom)
        object.__setattr__(self, "condition_number", condition_number)


@dataclass(frozen=True)
class FisherVectorProductResult:
    """Matrix-free empirical-Fisher vector product with provenance."""

    value: NDArray[np.float64]
    tangent: NDArray[np.float64]
    product: NDArray[np.float64]
    residual_projection: NDArray[np.float64]
    damping: float
    method: str
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_numeric_array("Fisher-vector value", self.value)
        tangent = _as_real_numeric_array("Fisher-vector tangent", self.tangent)
        product = _as_real_numeric_array("Fisher-vector product", self.product)
        projection = _as_real_numeric_array(
            "Fisher-vector residual_projection",
            self.residual_projection,
        )
        if value.ndim != 1:
            raise ValueError("Fisher-vector value must be one-dimensional")
        if tangent.ndim != 1 or product.shape != tangent.shape:
            raise ValueError("Fisher-vector tangent and product must be one-dimensional matches")
        if projection.shape != value.shape:
            raise ValueError("residual_projection shape must match value shape")
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(projection)):
            raise ValueError("Fisher-vector value and projection must contain only finite values")
        if not np.all(np.isfinite(tangent)) or not np.all(np.isfinite(product)):
            raise ValueError("Fisher-vector tangent and product must contain only finite values")
        damping = _as_real_scalar("Fisher-vector damping", self.damping)
        if damping < 0.0:
            raise ValueError("Fisher-vector damping must be finite and non-negative")
        if not self.method:
            raise ValueError("Fisher-vector method must be non-empty")
        if self.evaluations < 0:
            raise ValueError("Fisher-vector evaluations must be non-negative")
        if len(self.parameter_names) != tangent.size:
            raise ValueError("parameter_names length must match Fisher-vector dimension")
        if len(self.trainable) != tangent.size:
            raise ValueError("trainable mask length must match Fisher-vector dimension")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "tangent", tangent)
        object.__setattr__(self, "product", product)
        object.__setattr__(self, "residual_projection", projection)
        object.__setattr__(self, "damping", damping)


@dataclass(frozen=True)
class WeightedGradientResult:
    """Weighted scalarisation of multiple scalar gradient results."""

    value: float
    gradient: NDArray[np.float64]
    components: tuple[GradientResult, ...]
    weights: NDArray[np.float64]
    method: str
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("weighted gradient components must be non-empty")
        value = _as_real_scalar("weighted gradient value", self.value)
        gradient = _as_real_numeric_array("weighted gradient", self.gradient)
        weights = _as_real_numeric_array("weighted gradient weights", self.weights)
        if gradient.ndim != 1:
            raise ValueError("weighted gradient must be a one-dimensional array")
        if weights.ndim != 1 or weights.size != len(self.components):
            raise ValueError("weights length must match weighted gradient components")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("weighted gradient must contain only finite values")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must contain only finite values")
        if not self.method:
            raise ValueError("weighted gradient method must be non-empty")
        if self.evaluations < 0:
            raise ValueError("weighted gradient evaluations must be non-negative")
        if len(self.parameter_names) != gradient.size:
            raise ValueError("parameter_names length must match gradient length")
        if len(self.trainable) != gradient.size:
            raise ValueError("trainable mask length must match gradient length")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "weights", weights)


@dataclass(frozen=True)
class DifferentiableOptimizer:
    """Small native gradient-descent optimizer for differentiable SCPN parameters."""

    learning_rate: float = 0.01

    def __post_init__(self) -> None:
        learning_rate = _as_real_scalar("learning_rate", self.learning_rate)
        if learning_rate < 0.0:
            raise ValueError("learning_rate must be finite and non-negative")
        object.__setattr__(self, "learning_rate", learning_rate)

    def step(
        self,
        values: ArrayLike,
        gradient_result: GradientResult,
        *,
        bounds: Sequence[ParameterBounds] | None = None,
        max_gradient_norm: float | None = None,
    ) -> NDArray[np.float64]:
        """Return one gradient-descent update respecting the trainable mask."""

        parameter_values = _as_parameter_array(values)
        bounds_meta = _normalise_bounds(parameter_values, bounds)
        if parameter_values.size != gradient_result.gradient.size:
            raise ValueError("values length must match gradient length")
        trainable = np.asarray(gradient_result.trainable, dtype=bool)
        if trainable.size != parameter_values.size:
            raise ValueError("trainable mask length must match values length")
        gradient = _clip_gradient(
            gradient_result.gradient,
            trainable,
            max_gradient_norm=max_gradient_norm,
        )
        updated: NDArray[np.float64] = parameter_values.copy()
        updated[trainable] -= self.learning_rate * gradient[trainable]
        return _project_bounds(updated, bounds_meta)

    def minimize(
        self,
        objective: ScalarObjective,
        initial_values: ArrayLike,
        *,
        parameters: Sequence[Parameter] | None = None,
        rule: ParameterShiftRule | None = None,
        gradient_method: str = "parameter_shift",
        finite_difference_step: float = 1.0e-6,
        bounds: Sequence[ParameterBounds] | None = None,
        max_gradient_norm: float | None = None,
        max_steps: int = 100,
        gradient_tolerance: float = 1.0e-8,
        value_tolerance: float | None = None,
    ) -> OptimizationResult:
        """Run bounded gradient descent with parameter-shift gradients."""

        if gradient_method not in {"parameter_shift", "finite_difference"}:
            raise ValueError("gradient_method must be 'parameter_shift' or 'finite_difference'")
        finite_difference_step_value = _as_real_scalar(
            "finite_difference_step", finite_difference_step
        )
        if finite_difference_step_value <= 0.0:
            raise ValueError("finite_difference_step must be finite and positive")
        _validate_max_gradient_norm(max_gradient_norm)
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer")
        gradient_tolerance_value = _as_real_scalar("gradient_tolerance", gradient_tolerance)
        if gradient_tolerance_value < 0.0:
            raise ValueError("gradient_tolerance must be finite and non-negative")
        value_tolerance_value = (
            None
            if value_tolerance is None
            else _as_real_scalar("value_tolerance", value_tolerance)
        )
        if value_tolerance_value is not None and value_tolerance_value < 0.0:
            raise ValueError("value_tolerance must be finite and non-negative")

        values = _as_parameter_array(initial_values).copy()
        bounds_meta = _normalise_bounds(values, bounds)
        values = _project_bounds(values, bounds_meta)
        history: list[float] = []
        best_values = values.copy()
        best_value = float("inf")
        previous_value: float | None = None

        for step_index in range(max_steps + 1):
            if gradient_method == "finite_difference":
                gradient_result = value_and_finite_difference_grad(
                    objective,
                    values,
                    parameters=parameters,
                    step=finite_difference_step_value,
                )
            else:
                gradient_result = value_and_parameter_shift_grad(
                    objective,
                    values,
                    parameters=parameters,
                    rule=rule,
                )
            history.append(gradient_result.value)
            if gradient_result.value < best_value:
                best_value = gradient_result.value
                best_values = values.copy()
            trainable = np.asarray(gradient_result.trainable, dtype=bool)
            gradient_norm = float(np.linalg.norm(gradient_result.gradient[trainable], ord=2))
            if gradient_norm <= gradient_tolerance_value:
                return OptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    value_history=tuple(history),
                    steps=step_index,
                    converged=True,
                    reason="gradient_tolerance",
                    best_values=best_values,
                    best_value=best_value,
                )
            if (
                value_tolerance_value is not None
                and previous_value is not None
                and abs(previous_value - gradient_result.value) <= value_tolerance_value
            ):
                return OptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    value_history=tuple(history),
                    steps=step_index,
                    converged=True,
                    reason="value_tolerance",
                    best_values=best_values,
                    best_value=best_value,
                )
            if step_index == max_steps:
                return OptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    value_history=tuple(history),
                    steps=step_index,
                    converged=False,
                    reason="max_steps",
                    best_values=best_values,
                    best_value=best_value,
                )
            previous_value = gradient_result.value
            values = self.step(
                values,
                gradient_result,
                bounds=bounds_meta,
                max_gradient_norm=max_gradient_norm,
            )

        raise RuntimeError("unreachable optimizer state")


@dataclass(frozen=True)
class LevenbergMarquardtOptimizer:
    """Bounded Levenberg-Marquardt optimizer for residual-map objectives."""

    damping: float = 1.0e-3
    max_steps: int = 100
    residual_tolerance: float = 1.0e-8
    step_tolerance: float = 1.0e-8
    value_tolerance: float | None = None
    acceptance_threshold: float = 1.0e-4
    decrease_factor: float = 1.0 / 3.0
    increase_factor: float = 2.0
    min_damping: float = 1.0e-12
    max_damping: float = 1.0e12
    high_quality_ratio: float = 0.75
    finite_difference_step: float = 1.0e-6
    max_step_norm: float | None = None

    def __post_init__(self) -> None:
        damping = _as_real_scalar("Levenberg-Marquardt damping", self.damping)
        if damping < 0.0:
            raise ValueError("Levenberg-Marquardt damping must be finite and non-negative")
        max_steps = int(self.max_steps)
        if max_steps < 1:
            raise ValueError("Levenberg-Marquardt max_steps must be positive")
        residual_tolerance = _as_real_scalar(
            "Levenberg-Marquardt residual_tolerance",
            self.residual_tolerance,
        )
        step_tolerance = _as_real_scalar(
            "Levenberg-Marquardt step_tolerance",
            self.step_tolerance,
        )
        if residual_tolerance < 0.0 or step_tolerance < 0.0:
            raise ValueError("Levenberg-Marquardt tolerances must be finite and non-negative")
        value_tolerance = (
            None
            if self.value_tolerance is None
            else _as_real_scalar("Levenberg-Marquardt value_tolerance", self.value_tolerance)
        )
        if value_tolerance is not None and value_tolerance < 0.0:
            raise ValueError("Levenberg-Marquardt value_tolerance must be finite and non-negative")
        acceptance_threshold = _as_real_scalar(
            "Levenberg-Marquardt acceptance_threshold",
            self.acceptance_threshold,
        )
        if acceptance_threshold < 0.0:
            raise ValueError("Levenberg-Marquardt acceptance_threshold must be non-negative")
        decrease_factor = _as_real_scalar(
            "Levenberg-Marquardt decrease_factor",
            self.decrease_factor,
        )
        increase_factor = _as_real_scalar(
            "Levenberg-Marquardt increase_factor",
            self.increase_factor,
        )
        min_damping = _as_real_scalar("Levenberg-Marquardt min_damping", self.min_damping)
        max_damping = _as_real_scalar("Levenberg-Marquardt max_damping", self.max_damping)
        high_quality_ratio = _as_real_scalar(
            "Levenberg-Marquardt high_quality_ratio",
            self.high_quality_ratio,
        )
        finite_difference_step = _as_real_scalar(
            "Levenberg-Marquardt finite_difference_step",
            self.finite_difference_step,
        )
        if not 0.0 < decrease_factor < 1.0:
            raise ValueError("decrease_factor must be finite and between 0 and 1")
        if increase_factor <= 1.0:
            raise ValueError("increase_factor must be finite and greater than 1")
        if min_damping < 0.0 or max_damping < min_damping:
            raise ValueError("LM damping bounds must be finite and ordered")
        if high_quality_ratio < 0.0:
            raise ValueError("high_quality_ratio must be finite and non-negative")
        if finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be finite and positive")
        max_step_norm = (
            None
            if self.max_step_norm is None
            else _as_real_scalar("Levenberg-Marquardt max_step_norm", self.max_step_norm)
        )
        if max_step_norm is not None and max_step_norm <= 0.0:
            raise ValueError("max_step_norm must be finite and positive")
        object.__setattr__(self, "damping", min(max_damping, max(min_damping, damping)))
        object.__setattr__(self, "max_steps", max_steps)
        object.__setattr__(self, "residual_tolerance", residual_tolerance)
        object.__setattr__(self, "step_tolerance", step_tolerance)
        object.__setattr__(self, "value_tolerance", value_tolerance)
        object.__setattr__(self, "acceptance_threshold", acceptance_threshold)
        object.__setattr__(self, "decrease_factor", decrease_factor)
        object.__setattr__(self, "increase_factor", increase_factor)
        object.__setattr__(self, "min_damping", min_damping)
        object.__setattr__(self, "max_damping", max_damping)
        object.__setattr__(self, "high_quality_ratio", high_quality_ratio)
        object.__setattr__(self, "finite_difference_step", finite_difference_step)
        object.__setattr__(self, "max_step_norm", max_step_norm)

    def minimize(
        self,
        objective: VectorObjective,
        initial_values: ArrayLike,
        *,
        parameters: Sequence[Parameter] | None = None,
        bounds: Sequence[ParameterBounds] | None = None,
        weight_fn: Callable[[NDArray[np.float64]], ArrayLike] | None = None,
        rcond: float = 1.0e-12,
    ) -> LevenbergMarquardtResult:
        """Minimize a vector residual objective with adaptive bounded LM steps."""

        values = _as_parameter_array(initial_values)
        bounds_meta = _normalise_bounds(values, bounds)
        values = _project_bounds(values, bounds_meta)
        damping = self.damping
        jacobian_result = value_and_finite_difference_jacobian(
            objective,
            values,
            parameters=parameters,
            step=self.finite_difference_step,
        )
        weights = self._weights_for(jacobian_result.value, weight_fn)
        current_value = self._weighted_value(jacobian_result.value, weights)
        current_residual = jacobian_result.value
        best_values = values.copy()
        best_value = current_value
        value_history: list[float] = [current_value]
        damping_history: list[float] = [damping]
        accepted_history: list[bool] = []
        reason = "max_steps"
        converged = False

        if float(np.linalg.norm(current_residual, ord=2)) <= self.residual_tolerance:
            return LevenbergMarquardtResult(
                values=values,
                residual=current_residual,
                value_history=tuple(value_history),
                damping_history=tuple(damping_history),
                accepted_history=(),
                steps=0,
                converged=True,
                reason="residual_tolerance",
                best_values=best_values,
                best_value=best_value,
            )

        for _ in range(self.max_steps):
            step_result = levenberg_marquardt_step(
                jacobian_result,
                values,
                weights=weights,
                damping=damping,
                bounds=bounds_meta,
                max_step_norm=self.max_step_norm,
                rcond=rcond,
            )
            trial = evaluate_levenberg_marquardt_step(
                objective,
                step_result,
                weights=weights,
                acceptance_threshold=self.acceptance_threshold,
            )
            update = update_levenberg_marquardt_damping(
                trial,
                decrease_factor=self.decrease_factor,
                increase_factor=self.increase_factor,
                min_damping=self.min_damping,
                max_damping=self.max_damping,
                high_quality_ratio=self.high_quality_ratio,
            )
            accepted_history.append(trial.accepted)
            trainable = np.asarray(jacobian_result.trainable, dtype=bool)
            step_norm = float(np.linalg.norm(step_result.step[trainable], ord=2))
            if trial.accepted:
                values = step_result.candidate_values
                current_residual = trial.candidate_residual
                current_value = trial.candidate_value
                if current_value < best_value:
                    best_value = current_value
                    best_values = values.copy()
                if float(np.linalg.norm(current_residual, ord=2)) <= self.residual_tolerance:
                    reason = "residual_tolerance"
                    converged = True
                elif step_norm <= self.step_tolerance:
                    reason = "step_tolerance"
                    converged = True
                elif (
                    self.value_tolerance is not None
                    and abs(trial.actual_reduction) <= self.value_tolerance
                ):
                    reason = "value_tolerance"
                    converged = True
            damping = update.next_damping
            value_history.append(current_value)
            damping_history.append(damping)
            if converged:
                break
            if trial.accepted:
                jacobian_result = value_and_finite_difference_jacobian(
                    objective,
                    values,
                    parameters=parameters,
                    step=self.finite_difference_step,
                )
                weights = self._weights_for(jacobian_result.value, weight_fn)

        return LevenbergMarquardtResult(
            values=values,
            residual=current_residual,
            value_history=tuple(value_history),
            damping_history=tuple(damping_history),
            accepted_history=tuple(accepted_history),
            steps=len(accepted_history),
            converged=converged,
            reason=reason,
            best_values=best_values,
            best_value=best_value,
        )

    @staticmethod
    def _weighted_value(
        residual: NDArray[np.float64],
        weights: NDArray[np.float64] | None,
    ) -> float:
        if weights is None:
            return 0.5 * float(residual @ residual)
        return 0.5 * float(residual @ (residual * weights))

    @staticmethod
    def _weights_for(
        residual: NDArray[np.float64],
        weight_fn: Callable[[NDArray[np.float64]], ArrayLike] | None,
    ) -> NDArray[np.float64] | None:
        if weight_fn is None:
            return None
        weights = _as_real_numeric_array("LM weights", weight_fn(residual.copy()))
        if weights.ndim != 1 or weights.shape[0] != residual.size:
            raise ValueError("LM weights must be a one-dimensional array matching residual rows")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("LM weights must contain only finite non-negative values")
        return weights


def _as_parameter_array(values: ArrayLike) -> NDArray[np.float64]:
    array = _as_real_numeric_array("parameters", values)
    if array.ndim != 1:
        raise ValueError("parameters must be a one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError("parameters must contain only finite values")
    return array


def _as_scalar(value: float | int | np.floating[Any] | NDArray[np.float64]) -> float:
    try:
        scalar = _as_real_scalar("differentiable objective", value)
    except ValueError as exc:
        if "scalar" in str(exc):
            raise ValueError("differentiable objective must return a scalar") from exc
        raise
    if not np.isfinite(scalar):
        raise ValueError("differentiable objective returned a non-finite scalar")
    return scalar


def _as_vector_output(value: ArrayLike) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("differentiable vector objective", value)
    if vector.ndim != 1:
        raise ValueError("differentiable vector objective must return a one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError("differentiable vector objective returned non-finite values")
    return vector


def _normalise_parameters(
    values: NDArray[np.float64],
    parameters: Sequence[Parameter] | None,
) -> tuple[Parameter, ...]:
    if parameters is None:
        return tuple(Parameter(f"theta_{index}") for index in range(values.size))
    normalised = tuple(parameters)
    if len(normalised) != values.size:
        raise ValueError("parameters length must match values length")
    if len({parameter.name for parameter in normalised}) != len(normalised):
        raise ValueError("parameter names must be unique")
    return normalised


def _normalise_bounds(
    values: NDArray[np.float64],
    bounds: Sequence[ParameterBounds] | None,
) -> tuple[ParameterBounds, ...]:
    if bounds is None:
        return tuple(ParameterBounds() for _ in range(values.size))
    normalised = tuple(bounds)
    if len(normalised) != values.size:
        raise ValueError("bounds length must match values length")
    if any(not isinstance(item, ParameterBounds) for item in normalised):
        raise ValueError("bounds must contain ParameterBounds instances")
    return normalised


def _project_bounds(
    values: NDArray[np.float64],
    bounds: Sequence[ParameterBounds],
) -> NDArray[np.float64]:
    projected = values.copy()
    for index, bound in enumerate(bounds):
        if bound.periodic:
            lower = cast(float, bound.lower)
            upper = cast(float, bound.upper)
            width = upper - lower
            projected[index] = ((projected[index] - lower) % width) + lower
            continue
        if bound.lower is not None and projected[index] < bound.lower:
            projected[index] = bound.lower
        if bound.upper is not None and projected[index] > bound.upper:
            projected[index] = bound.upper
    return cast(NDArray[np.float64], projected)


def _validate_max_gradient_norm(max_gradient_norm: float | None) -> float | None:
    if max_gradient_norm is None:
        return None
    max_norm = _as_real_scalar("max_gradient_norm", max_gradient_norm)
    if max_norm <= 0.0:
        raise ValueError("max_gradient_norm must be finite and positive")
    return max_norm


def _clip_gradient(
    gradient: NDArray[np.float64],
    trainable: NDArray[np.bool_],
    *,
    max_gradient_norm: float | None,
) -> NDArray[np.float64]:
    max_norm = _validate_max_gradient_norm(max_gradient_norm)
    clipped = gradient.copy()
    if max_norm is None or not np.any(trainable):
        return cast(NDArray[np.float64], clipped)
    trainable_norm = float(np.linalg.norm(clipped[trainable], ord=2))
    if trainable_norm > max_norm:
        clipped[trainable] *= max_norm / trainable_norm
    return cast(NDArray[np.float64], clipped)


def parameter_shift_gradient(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> NDArray[np.float64]:
    """Return the parameter-shift gradient of a scalar objective."""

    result = value_and_parameter_shift_grad(
        objective,
        values,
        parameters=parameters,
        rule=rule,
    )
    return result.gradient


def batch_parameter_shift_gradient(
    objectives: Sequence[ScalarObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> NDArray[np.float64]:
    """Return stacked parameter-shift gradients for multiple scalar objectives."""

    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    rows = [
        parameter_shift_gradient(
            objective,
            values,
            parameters=parameters,
            rule=rule,
        )
        for objective in objectives
    ]
    return np.vstack(rows)


def batch_value_and_parameter_shift_grad(
    objectives: Sequence[ScalarObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> tuple[GradientResult, ...]:
    """Return full parameter-shift results for multiple scalar objectives."""

    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    return tuple(
        value_and_parameter_shift_grad(
            objective,
            values,
            parameters=parameters,
            rule=rule,
        )
        for objective in objectives
    )


def value_and_parameter_shift_grad(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> GradientResult:
    """Evaluate a scalar objective and its native parameter-shift gradient."""

    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    shift_rule = rule or ParameterShiftRule()
    gradient = np.zeros_like(parameter_values)
    base_value = _as_scalar(objective(parameter_values.copy()))
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        plus = parameter_values.copy()
        minus = parameter_values.copy()
        plus[index] += shift_rule.shift
        minus[index] -= shift_rule.shift
        plus_value = _as_scalar(objective(plus))
        minus_value = _as_scalar(objective(minus))
        evaluations += 2
        gradient[index] = shift_rule.coefficient * (plus_value - minus_value)

    return GradientResult(
        value=base_value,
        gradient=gradient,
        method="parameter_shift",
        shift=shift_rule.shift,
        coefficient=shift_rule.coefficient,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def finite_difference_gradient(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a central finite-difference gradient for scalar diagnostics."""

    result = value_and_finite_difference_grad(
        objective,
        values,
        parameters=parameters,
        step=step,
    )
    return result.gradient


def batch_value_and_finite_difference_grad(
    objectives: Sequence[ScalarObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> tuple[GradientResult, ...]:
    """Return full finite-difference results for multiple scalar objectives."""

    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    return tuple(
        value_and_finite_difference_grad(
            objective,
            values,
            parameters=parameters,
            step=step,
        )
        for objective in objectives
    )


def weighted_gradient_sum(
    components: Sequence[GradientResult],
    weights: ArrayLike,
    *,
    method: str = "weighted_sum",
) -> WeightedGradientResult:
    """Combine compatible scalar gradient results by an explicit weight vector."""

    component_tuple = tuple(components)
    if not component_tuple:
        raise ValueError("components must contain at least one GradientResult")
    if any(not isinstance(component, GradientResult) for component in component_tuple):
        raise ValueError("components must contain GradientResult instances")
    weight_arr = _as_real_numeric_array("weights", weights)
    if weight_arr.ndim != 1 or weight_arr.size != len(component_tuple):
        raise ValueError("weights length must match components length")
    if not np.all(np.isfinite(weight_arr)):
        raise ValueError("weights must contain only finite values")
    reference = component_tuple[0]
    for component in component_tuple[1:]:
        if component.gradient.shape != reference.gradient.shape:
            raise ValueError("all component gradients must have matching shapes")
        if component.parameter_names != reference.parameter_names:
            raise ValueError("all component parameter_names must match")
        if component.trainable != reference.trainable:
            raise ValueError("all component trainable masks must match")
    value = float(
        sum(
            float(weight) * component.value
            for weight, component in zip(weight_arr, component_tuple)
        )
    )
    gradient = np.zeros_like(reference.gradient)
    evaluations = 0
    for weight, component in zip(weight_arr, component_tuple):
        gradient += float(weight) * component.gradient
        evaluations += component.evaluations
    return WeightedGradientResult(
        value=value,
        gradient=gradient,
        components=component_tuple,
        weights=weight_arr,
        method=method,
        evaluations=evaluations,
        parameter_names=reference.parameter_names,
        trainable=reference.trainable,
    )


def value_and_finite_difference_grad(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> GradientResult:
    """Evaluate a scalar objective and central finite-difference gradient."""

    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    gradient = np.zeros_like(parameter_values)
    base_value = _as_scalar(objective(parameter_values.copy()))
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        plus = parameter_values.copy()
        minus = parameter_values.copy()
        plus[index] += step_value
        minus[index] -= step_value
        plus_value = _as_scalar(objective(plus))
        minus_value = _as_scalar(objective(minus))
        evaluations += 2
        gradient[index] = (plus_value - minus_value) / (2.0 * step_value)

    return GradientResult(
        value=base_value,
        gradient=gradient,
        method="finite_difference_central",
        shift=step_value,
        coefficient=1.0 / (2.0 * step_value),
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def finite_difference_jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a central finite-difference Jacobian for vector objectives."""

    return value_and_finite_difference_jacobian(
        objective,
        values,
        parameters=parameters,
        step=step,
    ).jacobian


def value_and_finite_difference_jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> JacobianResult:
    """Evaluate a vector objective and its central finite-difference Jacobian."""

    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    base_value = _as_vector_output(objective(parameter_values.copy()))
    jacobian = np.zeros((base_value.size, parameter_values.size), dtype=np.float64)
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        plus = parameter_values.copy()
        minus = parameter_values.copy()
        plus[index] += step_value
        minus[index] -= step_value
        plus_value = _as_vector_output(objective(plus))
        minus_value = _as_vector_output(objective(minus))
        if plus_value.shape != base_value.shape or minus_value.shape != base_value.shape:
            raise ValueError("vector objective output shape must remain stable")
        evaluations += 2
        jacobian[:, index] = (plus_value - minus_value) / (2.0 * step_value)

    return JacobianResult(
        value=base_value,
        jacobian=jacobian,
        method="finite_difference_central",
        step=step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def finite_difference_jvp(
    objective: VectorObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a central finite-difference Jacobian-vector product."""

    return value_and_finite_difference_jvp(
        objective,
        values,
        tangent,
        parameters=parameters,
        step=step,
    ).jvp


def value_and_finite_difference_jvp(
    objective: VectorObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> JVPResult:
    """Evaluate a vector objective and a directional finite-difference JVP."""

    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    tangent_values = _as_parameter_array(tangent)
    if tangent_values.shape != parameter_values.shape:
        raise ValueError("JVP tangent length must match parameter length")
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    masked_tangent = tangent_values.copy()
    masked_tangent[~trainable] = 0.0
    base_value = _as_vector_output(objective(parameter_values.copy()))
    if not np.any(masked_tangent):
        jvp = np.zeros_like(base_value)
        evaluations = 1
    else:
        plus = parameter_values + step_value * masked_tangent
        minus = parameter_values - step_value * masked_tangent
        plus_value = _as_vector_output(objective(plus))
        minus_value = _as_vector_output(objective(minus))
        if plus_value.shape != base_value.shape or minus_value.shape != base_value.shape:
            raise ValueError("vector objective output shape must remain stable")
        jvp = (plus_value - minus_value) / (2.0 * step_value)
        evaluations = 3
    return JVPResult(
        value=base_value,
        jvp=jvp,
        tangent=masked_tangent,
        method="finite_difference_directional",
        step=step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def vector_jacobian_product(
    jacobian: JacobianResult,
    cotangent: ArrayLike,
) -> VJPResult:
    """Contract a validated cotangent with a vector-objective Jacobian."""

    if not isinstance(jacobian, JacobianResult):
        raise ValueError("vector_jacobian_product requires a JacobianResult")
    cotangent_values = _as_vector_output(cotangent)
    if cotangent_values.shape != jacobian.value.shape:
        raise ValueError("VJP cotangent shape must match Jacobian value shape")
    vjp = cast(NDArray[np.float64], jacobian.jacobian.T @ cotangent_values)
    trainable = np.asarray(jacobian.trainable, dtype=bool)
    vjp[~trainable] = 0.0
    return VJPResult(
        value=jacobian.value,
        cotangent=cotangent_values,
        vjp=vjp,
        method=f"vjp:{jacobian.method}",
        step=jacobian.step,
        evaluations=jacobian.evaluations,
        parameter_names=jacobian.parameter_names,
        trainable=jacobian.trainable,
    )


def finite_difference_vjp(
    objective: VectorObjective,
    values: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> VJPResult:
    """Return a finite-difference vector-Jacobian product for a vector objective."""

    jacobian = value_and_finite_difference_jacobian(
        objective,
        values,
        parameters=parameters,
        step=step,
    )
    return vector_jacobian_product(jacobian, cotangent)


def finite_difference_hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-4,
) -> NDArray[np.float64]:
    """Return a central finite-difference Hessian for scalar objectives."""

    return value_and_finite_difference_hessian(
        objective,
        values,
        parameters=parameters,
        step=step,
    ).hessian


def value_and_finite_difference_hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-4,
) -> HessianResult:
    """Evaluate a scalar objective and central finite-difference Hessian."""

    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    base_value = _as_scalar(objective(parameter_values.copy()))
    hessian = np.zeros((parameter_values.size, parameter_values.size), dtype=np.float64)
    evaluations = 1

    for row in range(parameter_values.size):
        if not trainable[row]:
            continue
        plus = parameter_values.copy()
        minus = parameter_values.copy()
        plus[row] += step_value
        minus[row] -= step_value
        plus_value = _as_scalar(objective(plus))
        minus_value = _as_scalar(objective(minus))
        evaluations += 2
        hessian[row, row] = (plus_value - 2.0 * base_value + minus_value) / (step_value**2)

        for column in range(row + 1, parameter_values.size):
            if not trainable[column]:
                continue
            plus_plus = parameter_values.copy()
            plus_minus = parameter_values.copy()
            minus_plus = parameter_values.copy()
            minus_minus = parameter_values.copy()
            plus_plus[row] += step_value
            plus_plus[column] += step_value
            plus_minus[row] += step_value
            plus_minus[column] -= step_value
            minus_plus[row] -= step_value
            minus_plus[column] += step_value
            minus_minus[row] -= step_value
            minus_minus[column] -= step_value
            mixed = (
                _as_scalar(objective(plus_plus))
                - _as_scalar(objective(plus_minus))
                - _as_scalar(objective(minus_plus))
                + _as_scalar(objective(minus_minus))
            ) / (4.0 * step_value**2)
            evaluations += 4
            hessian[row, column] = mixed
            hessian[column, row] = mixed

    return HessianResult(
        value=base_value,
        hessian=hessian,
        method="finite_difference_central",
        step=step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def finite_difference_hvp(
    objective: ScalarObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-5,
) -> NDArray[np.float64]:
    """Return a central finite-difference Hessian-vector product."""

    return value_and_finite_difference_hvp(
        objective,
        values,
        tangent,
        parameters=parameters,
        step=step,
    ).hvp


def value_and_finite_difference_hvp(
    objective: ScalarObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-5,
) -> HVPResult:
    """Evaluate a scalar objective and a directional Hessian-vector product."""

    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    tangent_values = _as_parameter_array(tangent)
    if tangent_values.shape != parameter_values.shape:
        raise ValueError("HVP tangent length must match parameter length")
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    masked_tangent = tangent_values.copy()
    masked_tangent[~trainable] = 0.0
    base_value = _as_scalar(objective(parameter_values.copy()))
    if not np.any(masked_tangent):
        hvp = np.zeros_like(parameter_values)
        evaluations = 1
    else:
        plus = parameter_values + step_value * masked_tangent
        minus = parameter_values - step_value * masked_tangent
        plus_gradient = value_and_finite_difference_grad(
            objective,
            plus,
            parameters=parameter_meta,
            step=step_value,
        )
        minus_gradient = value_and_finite_difference_grad(
            objective,
            minus,
            parameters=parameter_meta,
            step=step_value,
        )
        hvp = (plus_gradient.gradient - minus_gradient.gradient) / (2.0 * step_value)
        hvp[~trainable] = 0.0
        evaluations = 1 + plus_gradient.evaluations + minus_gradient.evaluations
    return HVPResult(
        value=base_value,
        hvp=hvp,
        tangent=masked_tangent,
        method="finite_difference_hvp",
        step=step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def empirical_fisher_metric(
    jacobian: JacobianResult | ArrayLike,
    *,
    weights: ArrayLike | None = None,
    damping: float = 0.0,
) -> NDArray[np.float64]:
    """Return ``J.T @ W @ J + damping * I`` for differentiable residual maps."""

    jacobian_arr = (
        jacobian.jacobian
        if isinstance(jacobian, JacobianResult)
        else _as_real_numeric_array("jacobian", jacobian)
    )
    if jacobian_arr.ndim != 2:
        raise ValueError("jacobian must be a two-dimensional array")
    if not np.all(np.isfinite(jacobian_arr)):
        raise ValueError("jacobian must contain only finite values")
    if weights is None:
        weighted = jacobian_arr
    else:
        weight_arr = _as_real_numeric_array("weights", weights)
        if weight_arr.ndim != 1 or weight_arr.shape[0] != jacobian_arr.shape[0]:
            raise ValueError("weights must be a one-dimensional array matching jacobian rows")
        if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
            raise ValueError("weights must contain only finite non-negative values")
        weighted = jacobian_arr * weight_arr[:, None]
    damping_value = _as_real_scalar("fisher damping", damping)
    if damping_value < 0.0:
        raise ValueError("fisher damping must be finite and non-negative")
    metric = jacobian_arr.T @ weighted
    if damping_value > 0.0:
        metric = metric + damping_value * np.eye(metric.shape[0], dtype=np.float64)
    return cast(NDArray[np.float64], metric)


def empirical_fisher_vector_product(
    jacobian: JacobianResult,
    tangent: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    damping: float = 0.0,
) -> FisherVectorProductResult:
    """Return matrix-free ``(J.T @ W @ J + damping I) @ tangent``."""

    if not isinstance(jacobian, JacobianResult):
        raise ValueError("empirical_fisher_vector_product requires a JacobianResult")
    tangent_values = _as_parameter_array(tangent)
    if tangent_values.shape[0] != jacobian.jacobian.shape[1]:
        raise ValueError("Fisher-vector tangent length must match Jacobian parameter dimension")
    trainable = np.asarray(jacobian.trainable, dtype=bool)
    masked_tangent = tangent_values.copy()
    masked_tangent[~trainable] = 0.0
    damping_value = _as_real_scalar("Fisher-vector damping", damping)
    if damping_value < 0.0:
        raise ValueError("Fisher-vector damping must be finite and non-negative")
    projection = cast(NDArray[np.float64], jacobian.jacobian @ masked_tangent)
    if weights is None:
        weighted_projection = projection
    else:
        weight_arr = _as_real_numeric_array("weights", weights)
        if weight_arr.ndim != 1 or weight_arr.shape[0] != projection.size:
            raise ValueError("weights must be a one-dimensional array matching residual rows")
        if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
            raise ValueError("weights must contain only finite non-negative values")
        weighted_projection = projection * weight_arr
    product = cast(NDArray[np.float64], jacobian.jacobian.T @ weighted_projection)
    if damping_value > 0.0:
        product[trainable] += damping_value * masked_tangent[trainable]
    product[~trainable] = 0.0
    return FisherVectorProductResult(
        value=jacobian.value,
        tangent=masked_tangent,
        product=product,
        residual_projection=projection,
        damping=damping_value,
        method=f"fisher_vector_product:{jacobian.method}",
        evaluations=jacobian.evaluations,
        parameter_names=jacobian.parameter_names,
        trainable=jacobian.trainable,
    )


def least_squares_covariance(
    jacobian: JacobianResult,
    *,
    weights: ArrayLike | None = None,
    residual_variance: float | None = None,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> LeastSquaresCovarianceResult:
    """Estimate parameter covariance from a residual-map Fisher metric."""

    if not isinstance(jacobian, JacobianResult):
        raise ValueError("least_squares_covariance requires a JacobianResult")
    residual = jacobian.value
    trainable = np.asarray(jacobian.trainable, dtype=bool)
    active_count = int(np.count_nonzero(trainable))
    if active_count == 0:
        raise ValueError("least_squares_covariance requires at least one trainable parameter")
    metric = empirical_fisher_metric(jacobian, weights=weights, damping=damping)
    active_metric = metric[np.ix_(trainable, trainable)]
    eigenvalues = np.linalg.eigvalsh(active_metric)
    min_eigenvalue = float(np.min(eigenvalues))
    max_eigenvalue = float(np.max(eigenvalues))
    rcond_value = _as_real_scalar("least-squares rcond", rcond)
    if not 0.0 < rcond_value < 1.0:
        raise ValueError("rcond must be finite and between 0 and 1")
    if min_eigenvalue <= 0.0:
        raise ValueError("least-squares Fisher metric must be positive definite")
    condition_number = max_eigenvalue / min_eigenvalue
    if condition_number > 1.0 / rcond_value:
        raise ValueError("least-squares Fisher metric is ill-conditioned")
    degrees_of_freedom = max(1, residual.size - active_count)
    if residual_variance is None:
        if weights is None:
            weighted_residual = residual
        else:
            weight_arr = _as_real_numeric_array("weights", weights)
            if weight_arr.ndim != 1 or weight_arr.shape[0] != residual.size:
                raise ValueError("weights must be a one-dimensional array matching residual rows")
            if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
                raise ValueError("weights must contain only finite non-negative values")
            weighted_residual = residual * weight_arr
        variance = float(residual @ weighted_residual) / degrees_of_freedom
    else:
        variance = _as_real_scalar("least-squares residual_variance", residual_variance)
        if variance < 0.0:
            raise ValueError("residual_variance must be finite and non-negative")
    active_covariance = np.linalg.inv(active_metric) * variance
    covariance = np.zeros_like(metric)
    covariance[np.ix_(trainable, trainable)] = active_covariance
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return LeastSquaresCovarianceResult(
        covariance=covariance,
        standard_errors=standard_errors,
        residual_variance=variance,
        degrees_of_freedom=degrees_of_freedom,
        condition_number=condition_number,
        parameter_names=jacobian.parameter_names,
        trainable=jacobian.trainable,
    )


def huber_residual_weights(
    residuals: ArrayLike,
    *,
    delta: float = 1.0,
    min_weight: float = 0.0,
) -> NDArray[np.float64]:
    """Return Huber IRLS weights for robust residual-map least squares."""

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
    return cast(NDArray[np.float64], weights)


def soft_l1_residual_weights(
    residuals: ArrayLike,
    *,
    scale: float = 1.0,
    min_weight: float = 0.0,
) -> NDArray[np.float64]:
    """Return smooth Soft-L1 IRLS weights for residual-map least squares."""

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
    return cast(NDArray[np.float64], weights)


def gauss_newton_gradient(
    jacobian: JacobianResult,
    *,
    weights: ArrayLike | None = None,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> NaturalGradientResult:
    """Return the Gauss-Newton-preconditioned least-squares gradient.

    The residual map is read from ``jacobian.value`` and the scalar loss is
    ``0.5 * residual.T @ W @ residual``. The returned ``natural_gradient`` is
    the trainable-subspace solution of ``(J.T @ W @ J + damping * I) @ x =
    J.T @ W @ residual``; subtract it from parameters for a Gauss-Newton
    descent update.
    """

    if not isinstance(jacobian, JacobianResult):
        raise ValueError("gauss-newton gradient requires a JacobianResult")
    jacobian_arr = jacobian.jacobian
    residual = jacobian.value
    if weights is None:
        weighted_residual = residual
    else:
        weight_arr = _as_real_numeric_array("weights", weights)
        if weight_arr.ndim != 1 or weight_arr.shape[0] != residual.size:
            raise ValueError("weights must be a one-dimensional array matching residual rows")
        if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
            raise ValueError("weights must contain only finite non-negative values")
        weighted_residual = residual * weight_arr

    loss_value = 0.5 * float(residual @ weighted_residual)
    gradient = cast(NDArray[np.float64], jacobian_arr.T @ weighted_residual)
    base_gradient = GradientResult(
        value=loss_value,
        gradient=gradient,
        method=f"gauss_newton:{jacobian.method}",
        shift=None,
        coefficient=None,
        evaluations=jacobian.evaluations,
        parameter_names=jacobian.parameter_names,
        trainable=jacobian.trainable,
    )
    metric = empirical_fisher_metric(jacobian, weights=weights, damping=damping)
    return natural_gradient(base_gradient, metric, damping=0.0, rcond=rcond)


def levenberg_marquardt_step(
    jacobian: JacobianResult,
    values: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    damping: float = 1.0e-3,
    bounds: Sequence[ParameterBounds] | None = None,
    max_step_norm: float | None = None,
    rcond: float = 1.0e-12,
) -> LevenbergMarquardtStep:
    """Return a bounded Levenberg-Marquardt candidate for residual objectives."""

    current_values = _as_parameter_array(values)
    if current_values.size != jacobian.jacobian.shape[1]:
        raise ValueError("values length must match Jacobian parameter dimension")
    damping_value = _as_real_scalar("Levenberg-Marquardt damping", damping)
    if damping_value < 0.0:
        raise ValueError("Levenberg-Marquardt damping must be finite and non-negative")
    max_step_norm_value = (
        None
        if max_step_norm is None
        else _as_real_scalar("Levenberg-Marquardt max_step_norm", max_step_norm)
    )
    if max_step_norm_value is not None and max_step_norm_value <= 0.0:
        raise ValueError("Levenberg-Marquardt max_step_norm must be finite and positive")

    gauss_newton = gauss_newton_gradient(
        jacobian,
        weights=weights,
        damping=damping_value,
        rcond=rcond,
    )
    step = -gauss_newton.natural_gradient.copy()
    trainable = np.asarray(jacobian.trainable, dtype=bool)
    if max_step_norm_value is not None and np.any(trainable):
        norm = float(np.linalg.norm(step[trainable], ord=2))
        if norm > max_step_norm_value:
            step[trainable] *= max_step_norm_value / norm

    candidate_values = current_values + step
    if bounds is not None:
        candidate_values = _project_bounds(
            candidate_values, _normalise_bounds(current_values, bounds)
        )
        step = candidate_values - current_values

    model_gradient = gauss_newton.base_gradient.gradient
    predicted_reduction = -float(model_gradient @ step + 0.5 * step @ gauss_newton.metric @ step)
    return LevenbergMarquardtStep(
        gauss_newton=gauss_newton,
        step=step,
        candidate_values=candidate_values,
        damping=damping_value,
        predicted_reduction=predicted_reduction,
    )


def evaluate_levenberg_marquardt_step(
    objective: VectorObjective,
    step_result: LevenbergMarquardtStep,
    *,
    weights: ArrayLike | None = None,
    acceptance_threshold: float = 1.0e-4,
) -> LevenbergMarquardtTrial:
    """Evaluate actual residual reduction for a Levenberg-Marquardt candidate."""

    threshold = _as_real_scalar("Levenberg-Marquardt acceptance_threshold", acceptance_threshold)
    if threshold < 0.0:
        raise ValueError("Levenberg-Marquardt acceptance_threshold must be non-negative")
    candidate_residual = _as_vector_output(objective(step_result.candidate_values.copy()))
    reference_residual = step_result.gauss_newton.base_gradient.value
    if weights is None:
        candidate_value = 0.5 * float(candidate_residual @ candidate_residual)
    else:
        weight_arr = _as_real_numeric_array("weights", weights)
        if weight_arr.ndim != 1 or weight_arr.shape[0] != candidate_residual.size:
            raise ValueError("weights must be a one-dimensional array matching residual rows")
        if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
            raise ValueError("weights must contain only finite non-negative values")
        candidate_value = 0.5 * float(candidate_residual @ (candidate_residual * weight_arr))
    actual_reduction = reference_residual - candidate_value
    predicted = step_result.predicted_reduction
    reduction_ratio = actual_reduction / predicted if predicted > 0.0 else 0.0
    accepted = predicted > 0.0 and reduction_ratio >= threshold
    return LevenbergMarquardtTrial(
        step_result=step_result,
        candidate_residual=candidate_residual,
        candidate_value=candidate_value,
        actual_reduction=actual_reduction,
        reduction_ratio=reduction_ratio,
        accepted=accepted,
    )


def update_levenberg_marquardt_damping(
    trial: LevenbergMarquardtTrial,
    *,
    decrease_factor: float = 1.0 / 3.0,
    increase_factor: float = 2.0,
    min_damping: float = 1.0e-12,
    max_damping: float = 1.0e12,
    high_quality_ratio: float = 0.75,
) -> LevenbergMarquardtDampingUpdate:
    """Return a bounded trust-region damping update for an LM trial."""

    if not isinstance(trial, LevenbergMarquardtTrial):
        raise ValueError("damping update requires a LevenbergMarquardtTrial")
    decrease = _as_real_scalar("Levenberg-Marquardt decrease_factor", decrease_factor)
    increase = _as_real_scalar("Levenberg-Marquardt increase_factor", increase_factor)
    min_value = _as_real_scalar("Levenberg-Marquardt min_damping", min_damping)
    max_value = _as_real_scalar("Levenberg-Marquardt max_damping", max_damping)
    high_quality = _as_real_scalar(
        "Levenberg-Marquardt high_quality_ratio",
        high_quality_ratio,
    )
    if not 0.0 < decrease < 1.0:
        raise ValueError("decrease_factor must be finite and between 0 and 1")
    if increase <= 1.0:
        raise ValueError("increase_factor must be finite and greater than 1")
    if min_value < 0.0:
        raise ValueError("min_damping must be finite and non-negative")
    if max_value < min_value:
        raise ValueError("max_damping must be greater than or equal to min_damping")
    if high_quality < 0.0:
        raise ValueError("high_quality_ratio must be finite and non-negative")

    current = trial.step_result.damping
    if not trial.accepted:
        return LevenbergMarquardtDampingUpdate(
            trial=trial,
            next_damping=min(max_value, max(min_value, current * increase)),
            action="reject_increase",
        )
    if trial.reduction_ratio >= high_quality:
        return LevenbergMarquardtDampingUpdate(
            trial=trial,
            next_damping=min(max_value, max(min_value, current * decrease)),
            action="accept_decrease",
        )
    return LevenbergMarquardtDampingUpdate(
        trial=trial,
        next_damping=min(max_value, max(min_value, current)),
        action="accept_keep",
    )


def natural_gradient(
    gradient_result: GradientResult,
    metric: ArrayLike,
    *,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> NaturalGradientResult:
    """Solve ``metric @ natural_gradient = gradient`` on trainable parameters."""

    metric_arr = _as_real_numeric_array("natural-gradient metric", metric)
    if metric_arr.ndim != 2 or metric_arr.shape != (
        gradient_result.gradient.size,
        gradient_result.gradient.size,
    ):
        raise ValueError("natural-gradient metric must have shape (n_parameters, n_parameters)")
    if not np.all(np.isfinite(metric_arr)):
        raise ValueError("natural-gradient metric must contain only finite values")
    if not np.allclose(metric_arr, metric_arr.T, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("natural-gradient metric must be symmetric")
    damping_value = _as_real_scalar("natural-gradient damping", damping)
    if damping_value < 0.0:
        raise ValueError("natural-gradient damping must be finite and non-negative")
    rcond_value = _as_real_scalar("natural-gradient rcond", rcond)
    if rcond_value <= 0.0:
        raise ValueError("natural-gradient rcond must be finite and positive")

    trainable = np.asarray(gradient_result.trainable, dtype=bool)
    result = np.zeros_like(gradient_result.gradient)
    if not np.any(trainable):
        return NaturalGradientResult(
            base_gradient=gradient_result,
            metric=metric_arr,
            natural_gradient=result,
            damping=damping_value,
            condition_number=1.0,
        )

    active_metric = metric_arr[np.ix_(trainable, trainable)].copy()
    if damping_value > 0.0:
        active_metric += damping_value * np.eye(active_metric.shape[0], dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(active_metric)
    min_eigenvalue = float(np.min(eigenvalues))
    max_eigenvalue = float(np.max(eigenvalues))
    if min_eigenvalue <= 0.0:
        raise ValueError(
            "natural-gradient metric must be positive definite on trainable parameters"
        )
    condition_number = max_eigenvalue / min_eigenvalue
    if condition_number > 1.0 / rcond_value:
        raise ValueError("natural-gradient metric is ill-conditioned")
    result[trainable] = np.linalg.solve(active_metric, gradient_result.gradient[trainable])
    return NaturalGradientResult(
        base_gradient=gradient_result,
        metric=metric_arr,
        natural_gradient=result,
        damping=damping_value,
        condition_number=condition_number,
    )


def check_parameter_shift_consistency(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    finite_difference_step: float = 1.0e-6,
    tolerance: float = 1.0e-5,
) -> GradientCheckResult:
    """Compare parameter-shift gradients against central finite differences."""

    tolerance_value = _as_real_scalar("gradient check tolerance", tolerance)
    if tolerance_value < 0.0:
        raise ValueError("gradient check tolerance must be finite and non-negative")
    candidate = value_and_parameter_shift_grad(
        objective,
        values,
        parameters=parameters,
        rule=rule,
    )
    reference = value_and_finite_difference_grad(
        objective,
        values,
        parameters=parameters,
        step=finite_difference_step,
    )
    delta = candidate.gradient - reference.gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta, ord=2))
    value_delta = float(abs(candidate.value - reference.value))
    return GradientCheckResult(
        reference=reference,
        candidate=candidate,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        value_delta=value_delta,
        tolerance=tolerance_value,
        passed=max_abs_error <= tolerance_value,
    )


def is_jax_autodiff_available() -> bool:
    """Return whether JAX autodiff can be imported in the active environment."""

    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
    except ImportError:
        return False
    return True


def jax_value_and_grad(
    objective: Callable[[Any], Any],
    values: ArrayLike,
) -> tuple[float, NDArray[np.float64]]:
    """Evaluate a JAX scalar objective and return ``(value, gradient)``."""

    parameter_values = _as_parameter_array(values)

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError("JAX autodiff is unavailable; install the [jax] extra") from exc

    def wrapped(raw_values: Any) -> Any:
        return objective(raw_values)

    value, gradient = jax.value_and_grad(wrapped)(jnp.asarray(parameter_values))
    result_value = _as_real_scalar("JAX objective value", value)
    result_gradient = _as_real_numeric_array("JAX gradient", gradient)
    if result_gradient.shape != parameter_values.shape:
        raise ValueError("JAX gradient shape must match parameter shape")
    if not np.all(np.isfinite(result_gradient)):
        raise ValueError("JAX gradient must contain only finite values")
    return result_value, result_gradient


__all__ = [
    "DifferentiableOptimizer",
    "FisherVectorProductResult",
    "GradientCheckResult",
    "GradientResult",
    "HVPResult",
    "HessianResult",
    "JVPResult",
    "JacobianResult",
    "LeastSquaresCovarianceResult",
    "LevenbergMarquardtDampingUpdate",
    "LevenbergMarquardtOptimizer",
    "LevenbergMarquardtResult",
    "LevenbergMarquardtStep",
    "LevenbergMarquardtTrial",
    "NaturalGradientResult",
    "OptimizationResult",
    "Parameter",
    "ParameterBounds",
    "ParameterShiftRule",
    "VJPResult",
    "WeightedGradientResult",
    "batch_parameter_shift_gradient",
    "batch_value_and_finite_difference_grad",
    "batch_value_and_parameter_shift_grad",
    "check_parameter_shift_consistency",
    "empirical_fisher_metric",
    "empirical_fisher_vector_product",
    "evaluate_levenberg_marquardt_step",
    "finite_difference_gradient",
    "finite_difference_hessian",
    "finite_difference_hvp",
    "finite_difference_jacobian",
    "finite_difference_jvp",
    "finite_difference_vjp",
    "gauss_newton_gradient",
    "huber_residual_weights",
    "is_jax_autodiff_available",
    "jax_value_and_grad",
    "least_squares_covariance",
    "levenberg_marquardt_step",
    "natural_gradient",
    "soft_l1_residual_weights",
    "update_levenberg_marquardt_damping",
    "weighted_gradient_sum",
    "parameter_shift_gradient",
    "value_and_finite_difference_grad",
    "value_and_finite_difference_hessian",
    "value_and_finite_difference_hvp",
    "value_and_finite_difference_jacobian",
    "value_and_finite_difference_jvp",
    "value_and_parameter_shift_grad",
    "vector_jacobian_product",
]
