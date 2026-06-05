# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNode Transforms
"""Executable scalar phase-QNode transform evidence."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import Parameter, ParameterShiftRule
from .param_shift import (
    parameter_shift_gradient,
    parameter_shift_hessian,
    value_and_parameter_shift_grad,
)
from .transform_nesting import GradientTransformNestingPlan, plan_gradient_transform_nesting

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]

EVIDENCE_CLASS = "phase_qnode_transform_execution"
CLAIM_BOUNDARY = (
    "supported scalar local phase-QNode transforms only; not hardware execution, "
    "not arbitrary program AD, and not full vector-output framework transform algebra"
)


@dataclass(frozen=True)
class PhaseQNodeTransformResult:
    """Execution or fail-closed evidence for one scalar phase-QNode transform."""

    transform: str
    plan: GradientTransformNestingPlan
    params: FloatArray
    supported: bool
    value: float | None
    gradient: FloatArray | None
    hessian: FloatArray | None
    jvp: float | None
    vjp: FloatArray | None
    jacobian: FloatArray | None
    tangent: FloatArray | None
    cotangent: FloatArray | None
    parameter_shift_evaluations: int
    failure_reason: str
    evidence_class: str = EVIDENCE_CLASS
    claim_boundary: str = CLAIM_BOUNDARY
    hardware_execution: bool = False

    @property
    def fail_closed(self) -> bool:
        """Return true when execution was intentionally refused."""
        return not self.supported

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready transform evidence."""
        return {
            "transform": self.transform,
            "plan": self.plan.to_dict(),
            "params": self.params.tolist(),
            "supported": self.supported,
            "fail_closed": self.fail_closed,
            "value": self.value,
            "gradient": None if self.gradient is None else self.gradient.tolist(),
            "hessian": None if self.hessian is None else self.hessian.tolist(),
            "jvp": self.jvp,
            "vjp": None if self.vjp is None else self.vjp.tolist(),
            "jacobian": None if self.jacobian is None else self.jacobian.tolist(),
            "tangent": None if self.tangent is None else self.tangent.tolist(),
            "cotangent": None if self.cotangent is None else self.cotangent.tolist(),
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
            "failure_reason": self.failure_reason,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
        }


@dataclass(frozen=True)
class PhaseQNodeTransformReadinessSuiteResult:
    """Readiness evidence for supported and blocked scalar QNode transforms."""

    records: tuple[PhaseQNodeTransformResult, ...]
    evidence_class: str = EVIDENCE_CLASS
    claim_boundary: str = CLAIM_BOUNDARY
    hardware_execution: bool = False

    @property
    def record_count(self) -> int:
        """Return total records."""
        return len(self.records)

    @property
    def supported_count(self) -> int:
        """Return supported transform records."""
        return sum(1 for record in self.records if record.supported)

    @property
    def fail_closed_count(self) -> int:
        """Return fail-closed transform records."""
        return sum(1 for record in self.records if record.fail_closed)

    @property
    def total_parameter_shift_evaluations(self) -> int:
        """Return total parameter-shift evaluations across supported records."""
        return sum(record.parameter_shift_evaluations for record in self.records)

    @property
    def passed(self) -> bool:
        """Return true when default supported routes execute and blocked routes refuse."""
        supported = {record.transform for record in self.records if record.supported}
        blocked = [record for record in self.records if record.fail_closed]
        return (
            {
                "grad",
                "value_and_grad",
                "hessian",
                "jvp",
                "vjp",
                "jacfwd",
                "jacrev",
            }.issubset(supported)
            and len(blocked) >= 3
            and not self.hardware_execution
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready suite evidence."""
        return {
            "passed": self.passed,
            "record_count": self.record_count,
            "supported_count": self.supported_count,
            "fail_closed_count": self.fail_closed_count,
            "total_parameter_shift_evaluations": self.total_parameter_shift_evaluations,
            "records": [record.to_dict() for record in self.records],
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
        }


def execute_phase_qnode_transform(
    transform: str | tuple[str, ...],
    objective: ScalarObjective,
    params: ArrayLike,
    *,
    tangent: ArrayLike | None = None,
    cotangent: ArrayLike | float | None = None,
    gate: str = "ry",
    observable: str = "pauli_expectation",
    backend: str = "statevector",
    adapter: str = "native",
    shots: int | None = None,
    shift_terms: int = 1,
    allow_hardware: bool = False,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PhaseQNodeTransformResult:
    """Execute a supported scalar phase-QNode transform or return fail-closed evidence."""
    values = _as_parameter_vector("params", params)
    plan = plan_gradient_transform_nesting(
        transform,
        gate=gate,
        observable=observable,
        backend=backend,
        adapter=adapter,
        n_params=values.size,
        shift_terms=shift_terms if rule is None else len(rule.terms),
        shots=shots,
        allow_hardware=allow_hardware,
    )
    label = ".".join(plan.transforms)
    if plan.fail_closed:
        return _blocked_result(label, plan, values)

    if plan.transforms == ("grad",):
        gradient = parameter_shift_gradient(objective, values, parameters=parameters, rule=rule)
        return _supported_result(label, plan, values, gradient=gradient)

    if plan.transforms == ("value_and_grad",):
        result = value_and_parameter_shift_grad(
            objective, values, parameters=parameters, rule=rule
        )
        return _supported_result(
            label,
            plan,
            values,
            value=float(result.value),
            gradient=cast(FloatArray, result.gradient.astype(np.float64, copy=True)),
        )

    if plan.transforms in {("hessian",), ("grad", "grad")}:
        hessian = parameter_shift_hessian(objective, values, parameters=parameters, rule=rule)
        return _supported_result(
            label,
            plan,
            values,
            hessian=hessian,
            parameter_shift_evaluations=_hessian_evaluations(
                values.size, plan.support_plan.backend_plan.shift_terms
            ),
        )

    if plan.transforms == ("jvp",):
        tangent_vector = _as_parameter_vector("tangent", tangent, width=values.size)
        value = float(objective(values.copy()))
        gradient = parameter_shift_gradient(objective, values, parameters=parameters, rule=rule)
        return _supported_result(
            label,
            plan,
            values,
            value=value,
            gradient=gradient,
            tangent=tangent_vector,
            jvp=float(np.dot(gradient, tangent_vector)),
        )

    if plan.transforms == ("vjp",):
        cotangent_vector = _as_cotangent(cotangent)
        value = float(objective(values.copy()))
        gradient = parameter_shift_gradient(objective, values, parameters=parameters, rule=rule)
        return _supported_result(
            label,
            plan,
            values,
            value=value,
            gradient=gradient,
            cotangent=cotangent_vector,
            vjp=cotangent_vector[0] * gradient,
        )

    if plan.transforms in {("jacfwd",), ("jacrev",)}:
        gradient = parameter_shift_gradient(objective, values, parameters=parameters, rule=rule)
        jacobian = gradient.reshape(1, -1).astype(np.float64, copy=True)
        return _supported_result(
            label,
            plan,
            values,
            gradient=gradient,
            jacobian=jacobian,
        )

    if plan.transforms == ("vmap", "grad"):
        return PhaseQNodeTransformResult(
            transform=label,
            plan=plan,
            params=values.copy(),
            supported=False,
            value=None,
            gradient=None,
            hessian=None,
            jvp=None,
            vjp=None,
            jacobian=None,
            tangent=None,
            cotangent=None,
            parameter_shift_evaluations=0,
            failure_reason=(
                "scalar QNode transform executor does not accept batched parameters; "
                "use execute_phase_qnode_vmap_grad for native manual vmap(grad)"
            ),
        )

    return _blocked_result(label, plan, values)


def run_phase_qnode_transform_readiness_suite() -> PhaseQNodeTransformReadinessSuiteResult:
    """Run scalar QNode transform readiness evidence."""

    def objective(params: FloatArray) -> float:
        return float(np.cos(params[0]) + 0.25 * np.sin(params[1]))

    params = np.array([0.2, -0.4], dtype=np.float64)
    records = [
        execute_phase_qnode_transform("grad", objective, params),
        execute_phase_qnode_transform("value_and_grad", objective, params),
        execute_phase_qnode_transform("hessian", objective, params),
        execute_phase_qnode_transform("jvp", objective, params, tangent=np.array([0.5, -2.0])),
        execute_phase_qnode_transform("vjp", objective, params, cotangent=np.array([3.0])),
        execute_phase_qnode_transform("jacfwd", objective, params),
        execute_phase_qnode_transform("jacrev", objective, params),
        execute_phase_qnode_transform("grad", objective, params, backend="hardware", shots=1024),
        execute_phase_qnode_transform(
            "hessian",
            objective,
            params,
            backend="finite_shot_simulator",
            shots=256,
        ),
        execute_phase_qnode_transform(("vmap", "grad"), objective, params),
    ]
    return PhaseQNodeTransformReadinessSuiteResult(records=tuple(records))


def _supported_result(
    transform: str,
    plan: GradientTransformNestingPlan,
    params: FloatArray,
    *,
    value: float | None = None,
    gradient: FloatArray | None = None,
    hessian: FloatArray | None = None,
    jvp: float | None = None,
    vjp: FloatArray | None = None,
    jacobian: FloatArray | None = None,
    tangent: FloatArray | None = None,
    cotangent: FloatArray | None = None,
    parameter_shift_evaluations: int | None = None,
) -> PhaseQNodeTransformResult:
    return PhaseQNodeTransformResult(
        transform=transform,
        plan=plan,
        params=params.copy(),
        supported=True,
        value=value,
        gradient=None if gradient is None else gradient.astype(np.float64, copy=True),
        hessian=None if hessian is None else hessian.astype(np.float64, copy=True),
        jvp=jvp,
        vjp=None if vjp is None else vjp.astype(np.float64, copy=True),
        jacobian=None if jacobian is None else jacobian.astype(np.float64, copy=True),
        tangent=None if tangent is None else tangent.astype(np.float64, copy=True),
        cotangent=None if cotangent is None else cotangent.astype(np.float64, copy=True),
        parameter_shift_evaluations=(
            plan.support_plan.backend_plan.evaluations
            if parameter_shift_evaluations is None
            else parameter_shift_evaluations
        ),
        failure_reason="",
    )


def _blocked_result(
    transform: str,
    plan: GradientTransformNestingPlan,
    params: FloatArray,
) -> PhaseQNodeTransformResult:
    return PhaseQNodeTransformResult(
        transform=transform,
        plan=plan,
        params=params.copy(),
        supported=False,
        value=None,
        gradient=None,
        hessian=None,
        jvp=None,
        vjp=None,
        jacobian=None,
        tangent=None,
        cotangent=None,
        parameter_shift_evaluations=0,
        failure_reason="; ".join(plan.blocked_reasons),
    )


def _as_parameter_vector(
    name: str,
    values: ArrayLike | None,
    *,
    width: int | None = None,
) -> FloatArray:
    if values is None:
        raise ValueError(f"{name} must be provided")
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, vector.astype(np.float64, copy=True))


def _hessian_evaluations(n_params: int, shift_terms: int) -> int:
    diagonal = 2 * n_params * shift_terms
    mixed_pairs = n_params * (n_params - 1) // 2
    mixed = 4 * mixed_pairs * shift_terms * shift_terms
    return diagonal + mixed


def _as_cotangent(values: ArrayLike | float | None) -> FloatArray:
    if values is None:
        raise ValueError("cotangent must be provided")
    vector = np.asarray(values, dtype=float)
    if vector.ndim == 0:
        vector = vector.reshape(1)
    if vector.ndim != 1 or vector.shape != (1,):
        raise ValueError("cotangent must be a scalar or one-element vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("cotangent must contain only finite values")
    return cast(FloatArray, vector.astype(np.float64, copy=True))


__all__ = [
    "PhaseQNodeTransformReadinessSuiteResult",
    "PhaseQNodeTransformResult",
    "execute_phase_qnode_transform",
    "run_phase_qnode_transform_readiness_suite",
]
