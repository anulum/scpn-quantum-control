# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Vector QNode Transforms
"""Executable vector-output phase-QNode Jacobian and native vmap evidence."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import Parameter, ParameterShiftRule
from .param_shift import parameter_shift_gradient
from .transform_nesting import GradientTransformNestingPlan, plan_gradient_transform_nesting

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]
VectorObjective = Callable[[FloatArray], ArrayLike]

EVIDENCE_CLASS = "phase_qnode_vector_transform_execution"
CLAIM_BOUNDARY = (
    "supported deterministic native local vector-output phase-QNode Jacobian and "
    "manual vmap(grad) evidence only; not provider vectorization, hardware execution, "
    "or arbitrary framework-native transform algebra"
)


@dataclass(frozen=True)
class PhaseQNodeVectorTransformResult:
    """Execution or fail-closed evidence for vector-QNode transforms."""

    transform: str
    plan: GradientTransformNestingPlan
    supported: bool
    params: FloatArray | None
    values: FloatArray | None
    jacobian: FloatArray | None
    batched_params: FloatArray | None
    batched_values: FloatArray | None
    batched_gradients: FloatArray | None
    parameter_shift_evaluations: int
    failure_reason: str
    evidence_class: str = EVIDENCE_CLASS
    claim_boundary: str = CLAIM_BOUNDARY
    hardware_execution: bool = False

    @property
    def fail_closed(self) -> bool:
        """Return true when execution was intentionally refused."""
        return not self.supported

    @property
    def output_dim(self) -> int:
        """Return the vector-output dimension for Jacobian records."""
        return 0 if self.values is None else int(self.values.size)

    @property
    def batch_size(self) -> int:
        """Return the batch size for vmap records."""
        return 0 if self.batched_params is None else int(self.batched_params.shape[0])

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready vector-transform evidence."""
        return {
            "transform": self.transform,
            "plan": self.plan.to_dict(),
            "supported": self.supported,
            "fail_closed": self.fail_closed,
            "params": None if self.params is None else self.params.tolist(),
            "values": None if self.values is None else self.values.tolist(),
            "jacobian": None if self.jacobian is None else self.jacobian.tolist(),
            "batched_params": (
                None if self.batched_params is None else self.batched_params.tolist()
            ),
            "batched_values": (
                None if self.batched_values is None else self.batched_values.tolist()
            ),
            "batched_gradients": (
                None if self.batched_gradients is None else self.batched_gradients.tolist()
            ),
            "output_dim": self.output_dim,
            "batch_size": self.batch_size,
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
            "failure_reason": self.failure_reason,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
        }


@dataclass(frozen=True)
class PhaseQNodeVectorTransformReadinessSuiteResult:
    """Readiness evidence for supported and blocked vector QNode transforms."""

    records: tuple[PhaseQNodeVectorTransformResult, ...]
    evidence_class: str = EVIDENCE_CLASS
    claim_boundary: str = CLAIM_BOUNDARY
    hardware_execution: bool = False

    @property
    def record_count(self) -> int:
        """Return total records."""
        return len(self.records)

    @property
    def supported_count(self) -> int:
        """Return supported records."""
        return sum(1 for record in self.records if record.supported)

    @property
    def fail_closed_count(self) -> int:
        """Return fail-closed records."""
        return sum(1 for record in self.records if record.fail_closed)

    @property
    def total_parameter_shift_evaluations(self) -> int:
        """Return total parameter-shift evaluations across supported records."""
        return sum(record.parameter_shift_evaluations for record in self.records)

    @property
    def passed(self) -> bool:
        """Return true when supported routes execute and unsafe routes refuse."""
        supported = {record.transform for record in self.records if record.supported}
        return (
            {"jacfwd", "jacrev", "vmap.grad"}.issubset(supported)
            and self.fail_closed_count >= 3
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


def execute_phase_qnode_vector_jacobian(
    transform: str,
    objective: VectorObjective,
    params: ArrayLike,
    *,
    gate: str = "ry",
    observable: str = "pauli_expectation",
    backend: str = "statevector",
    adapter: str = "native",
    shots: int | None = None,
    shift_terms: int = 1,
    allow_hardware: bool = False,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PhaseQNodeVectorTransformResult:
    """Execute a deterministic native vector-output QNode Jacobian."""
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
        return _blocked_result(label, plan, params=values)
    if plan.transforms not in {("jacfwd",), ("jacrev",)}:
        return _blocked_result(
            label,
            plan,
            params=values,
            reason="vector QNode execution supports only jacfwd or jacrev",
        )

    vector_value = _as_vector_output("objective(params)", objective(values.copy()))
    rows: list[FloatArray] = []
    for output_index in range(vector_value.size):

        def scalar_component(candidate: FloatArray, *, index: int = output_index) -> float:
            candidate_value = _as_vector_output(
                "objective(shifted_params)",
                objective(candidate.copy()),
                width=vector_value.size,
            )
            return float(candidate_value[index])

        rows.append(
            parameter_shift_gradient(
                scalar_component,
                values,
                parameters=parameters,
                rule=rule,
            )
        )

    jacobian = np.vstack(rows).astype(np.float64, copy=True)
    return _supported_result(
        label,
        plan,
        params=values,
        values=vector_value,
        jacobian=jacobian,
        parameter_shift_evaluations=_parameter_shift_evaluations(
            values.size,
            vector_value.size,
            plan.support_plan.backend_plan.shift_terms,
        ),
    )


def execute_phase_qnode_vmap_grad(
    objective: ScalarObjective,
    batched_params: ArrayLike,
    *,
    gate: str = "ry",
    observable: str = "pauli_expectation",
    backend: str = "statevector",
    adapter: str = "native",
    shots: int | None = None,
    shift_terms: int = 1,
    allow_hardware: bool = False,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PhaseQNodeVectorTransformResult:
    """Execute native host-side manual vmap over scalar parameter-shift gradients."""
    batch = _as_parameter_matrix("batched_params", batched_params)
    plan = plan_gradient_transform_nesting(
        ("vmap", "grad"),
        gate=gate,
        observable=observable,
        backend=backend,
        adapter=adapter,
        n_params=batch.shape[1],
        shift_terms=shift_terms if rule is None else len(rule.terms),
        shots=shots,
        allow_hardware=allow_hardware,
    )
    label = ".".join(plan.transforms)
    if plan.fail_closed:
        return _blocked_result(label, plan, batched_params=batch)

    values: list[float] = []
    gradients: list[FloatArray] = []
    for row in batch:
        row_params = row.astype(np.float64, copy=True)
        values.append(_as_scalar_output("objective(batch_row)", objective(row_params.copy())))
        gradients.append(
            parameter_shift_gradient(
                objective,
                row_params,
                parameters=parameters,
                rule=rule,
            )
        )

    batched_values = np.asarray(values, dtype=np.float64)
    batched_gradients = np.vstack(gradients).astype(np.float64, copy=True)
    return _supported_result(
        label,
        plan,
        batched_params=batch,
        batched_values=batched_values,
        batched_gradients=batched_gradients,
        parameter_shift_evaluations=batch.shape[0]
        * 2
        * batch.shape[1]
        * plan.support_plan.backend_plan.shift_terms,
    )


def run_phase_qnode_vector_transform_readiness_suite() -> (
    PhaseQNodeVectorTransformReadinessSuiteResult
):
    """Run vector QNode transform readiness evidence."""

    def scalar_objective(params: FloatArray) -> float:
        return float(np.cos(params[0]) + 0.25 * np.sin(params[1]))

    def vector_objective(params: FloatArray) -> FloatArray:
        return np.array(
            [
                np.cos(params[0]) + 0.1 * np.sin(params[1]),
                np.sin(params[0]) - 0.25 * np.cos(params[1]),
            ],
            dtype=np.float64,
        )

    params = np.array([0.2, -0.4], dtype=np.float64)
    batched_params = np.array(
        [[0.2, -0.4], [0.7, 0.1], [-0.3, 0.6]],
        dtype=np.float64,
    )
    records = (
        execute_phase_qnode_vector_jacobian("jacfwd", vector_objective, params),
        execute_phase_qnode_vector_jacobian("jacrev", vector_objective, params),
        execute_phase_qnode_vmap_grad(scalar_objective, batched_params),
        execute_phase_qnode_vector_jacobian(
            "jacfwd",
            vector_objective,
            params,
            backend="hardware",
            shots=1024,
        ),
        execute_phase_qnode_vector_jacobian(
            "jacrev",
            vector_objective,
            params,
            adapter="jax",
        ),
        execute_phase_qnode_vmap_grad(
            scalar_objective,
            batched_params,
            backend="qasm_simulator",
            shots=256,
        ),
    )
    return PhaseQNodeVectorTransformReadinessSuiteResult(records=records)


def _supported_result(
    transform: str,
    plan: GradientTransformNestingPlan,
    *,
    params: FloatArray | None = None,
    values: FloatArray | None = None,
    jacobian: FloatArray | None = None,
    batched_params: FloatArray | None = None,
    batched_values: FloatArray | None = None,
    batched_gradients: FloatArray | None = None,
    parameter_shift_evaluations: int,
) -> PhaseQNodeVectorTransformResult:
    return PhaseQNodeVectorTransformResult(
        transform=transform,
        plan=plan,
        supported=True,
        params=None if params is None else params.astype(np.float64, copy=True),
        values=None if values is None else values.astype(np.float64, copy=True),
        jacobian=None if jacobian is None else jacobian.astype(np.float64, copy=True),
        batched_params=(
            None if batched_params is None else batched_params.astype(np.float64, copy=True)
        ),
        batched_values=(
            None if batched_values is None else batched_values.astype(np.float64, copy=True)
        ),
        batched_gradients=(
            None if batched_gradients is None else batched_gradients.astype(np.float64, copy=True)
        ),
        parameter_shift_evaluations=parameter_shift_evaluations,
        failure_reason="",
    )


def _blocked_result(
    transform: str,
    plan: GradientTransformNestingPlan,
    *,
    params: FloatArray | None = None,
    batched_params: FloatArray | None = None,
    reason: str | None = None,
) -> PhaseQNodeVectorTransformResult:
    return PhaseQNodeVectorTransformResult(
        transform=transform,
        plan=plan,
        supported=False,
        params=None if params is None else params.copy(),
        values=None,
        jacobian=None,
        batched_params=None if batched_params is None else batched_params.copy(),
        batched_values=None,
        batched_gradients=None,
        parameter_shift_evaluations=0,
        failure_reason=reason if reason is not None else "; ".join(plan.blocked_reasons),
    )


def _as_parameter_vector(
    name: str,
    values: ArrayLike,
    *,
    width: int | None = None,
) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, vector.astype(np.float64, copy=True))


def _as_parameter_matrix(name: str, values: ArrayLike) -> FloatArray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty batch and parameter axes")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, matrix.astype(np.float64, copy=True))


def _as_vector_output(name: str, values: ArrayLike, *, width: int | None = None) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector output")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must keep output shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, vector.astype(np.float64, copy=True))


def _as_scalar_output(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _parameter_shift_evaluations(n_params: int, output_dim: int, shift_terms: int) -> int:
    return 2 * n_params * output_dim * shift_terms


__all__ = [
    "PhaseQNodeVectorTransformReadinessSuiteResult",
    "PhaseQNodeVectorTransformResult",
    "execute_phase_qnode_vector_jacobian",
    "execute_phase_qnode_vmap_grad",
    "run_phase_qnode_vector_transform_readiness_suite",
]
