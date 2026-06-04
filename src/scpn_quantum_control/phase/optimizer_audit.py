# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Optimizer Audit
"""Multi-start convergence evidence for parameter-shift phase optimizers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import Parameter, ParameterShiftRule
from .gradient_descent import (
    ParameterShiftTrainingCertificate,
    ParameterShiftTrainingResult,
    parameter_shift_gradient_descent,
    validate_parameter_shift_training,
)
from .natural_gradient import (
    MetricTensorInput,
    ParameterShiftNaturalGradientCertificate,
    ParameterShiftNaturalGradientResult,
    parameter_shift_natural_gradient_descent,
    validate_natural_gradient_training,
)

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]


@dataclass(frozen=True)
class OptimizerConvergenceRecord:
    """One optimizer result reduced to serialisable audit evidence."""

    optimizer: str
    start_index: int
    initial_params: FloatArray
    initial_value: float
    final_value: float
    best_value: float
    value_decrease: float
    accepted_steps: int
    rejected_steps: int
    evaluations: int
    final_gradient_norm: float
    final_natural_gradient_norm: float | None
    converged: bool
    reason: str
    certificate_passed: bool
    monotone_accepted_values: bool
    metric_source: str | None
    max_metric_condition_number: float | None
    method: str
    shift_terms: int
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready convergence evidence."""
        return {
            "optimizer": self.optimizer,
            "start_index": self.start_index,
            "initial_params": self.initial_params.tolist(),
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "best_value": self.best_value,
            "value_decrease": self.value_decrease,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "evaluations": self.evaluations,
            "final_gradient_norm": self.final_gradient_norm,
            "final_natural_gradient_norm": self.final_natural_gradient_norm,
            "converged": self.converged,
            "reason": self.reason,
            "certificate_passed": self.certificate_passed,
            "monotone_accepted_values": self.monotone_accepted_values,
            "metric_source": self.metric_source,
            "max_metric_condition_number": self.max_metric_condition_number,
            "method": self.method,
            "shift_terms": self.shift_terms,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class OptimizerComparisonSuiteResult:
    """Multi-start optimizer comparison with explicit claim boundaries."""

    records: tuple[OptimizerConvergenceRecord, ...]
    start_count: int
    passed: bool
    best_optimizer: str
    gradient_descent_best_value: float
    natural_gradient_best_value: float
    natural_gradient_not_worse_count: int
    comparison_tolerance: float
    claim_boundary: str

    @property
    def optimizers(self) -> tuple[str, ...]:
        """Return optimizer names in first-seen order."""
        names: list[str] = []
        for record in self.records:
            if record.optimizer not in names:
                names.append(record.optimizer)
        return tuple(names)

    @property
    def certificate_failures(self) -> tuple[OptimizerConvergenceRecord, ...]:
        """Return records whose convergence certificate failed."""
        return tuple(record for record in self.records if not record.certificate_passed)

    def records_for_start(self, start_index: int) -> tuple[OptimizerConvergenceRecord, ...]:
        """Return all optimizer records for one start index."""
        return tuple(record for record in self.records if record.start_index == start_index)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready suite metadata."""
        return {
            "records": [record.to_dict() for record in self.records],
            "start_count": self.start_count,
            "passed": self.passed,
            "best_optimizer": self.best_optimizer,
            "gradient_descent_best_value": self.gradient_descent_best_value,
            "natural_gradient_best_value": self.natural_gradient_best_value,
            "natural_gradient_not_worse_count": self.natural_gradient_not_worse_count,
            "comparison_tolerance": self.comparison_tolerance,
            "claim_boundary": self.claim_boundary,
        }


def _as_start_matrix(starts: ArrayLike | None) -> FloatArray:
    if starts is None:
        return np.array(
            [
                [0.8, 0.8],
                [-0.7, 1.1],
                [1.2, -0.9],
            ],
            dtype=np.float64,
        )
    matrix = np.asarray(starts, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("starts must be a non-empty one- or two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("starts must contain only finite values")
    return matrix.astype(np.float64, copy=True)


def _positive_float(name: str, value: float) -> float:
    scalar = float(value)
    if scalar <= 0.0 or not np.isfinite(scalar):
        raise ValueError(f"{name} must be a positive finite scalar")
    return scalar


def _non_negative_float(name: str, value: float) -> float:
    scalar = float(value)
    if scalar < 0.0 or not np.isfinite(scalar):
        raise ValueError(f"{name} must be a non-negative finite scalar")
    return scalar


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _default_weights(width: int) -> FloatArray:
    if width == 1:
        return np.ones(1, dtype=np.float64)
    return np.geomspace(1.0, 0.05, num=width).astype(np.float64)


def _default_objective(weights: FloatArray) -> ScalarObjective:
    def objective(params: FloatArray) -> float:
        if params.shape != weights.shape:
            raise ValueError("objective params shape must match default weights")
        return float(np.sum(weights * (1.0 - np.cos(params))))

    return objective


def _default_metric(weights: FloatArray) -> FloatArray:
    return np.diag(weights).astype(np.float64, copy=True)


def _certificate_passed(
    *,
    monotone_accepted_values: bool,
    within_gradient_tolerance: bool | None,
    within_target_value_tolerance: bool | None,
    min_decrease_satisfied: bool | None,
    within_natural_gradient_tolerance: bool | None = None,
) -> bool:
    gates = (
        within_gradient_tolerance,
        within_target_value_tolerance,
        min_decrease_satisfied,
        within_natural_gradient_tolerance,
    )
    return monotone_accepted_values and all(gate is not False for gate in gates)


def _record_gradient_descent(
    *,
    start_index: int,
    result: ParameterShiftTrainingResult,
    certificate: ParameterShiftTrainingCertificate,
) -> OptimizerConvergenceRecord:
    return OptimizerConvergenceRecord(
        optimizer="parameter_shift_gradient_descent",
        start_index=start_index,
        initial_params=result.initial_params.copy(),
        initial_value=result.initial_value,
        final_value=result.final_value,
        best_value=result.best_value,
        value_decrease=result.initial_value - result.best_value,
        accepted_steps=result.accepted_steps,
        rejected_steps=result.rejected_steps,
        evaluations=result.evaluations,
        final_gradient_norm=result.final_gradient_norm,
        final_natural_gradient_norm=None,
        converged=result.converged,
        reason=result.reason,
        certificate_passed=_certificate_passed(
            monotone_accepted_values=certificate.monotone_accepted_values,
            within_gradient_tolerance=certificate.within_gradient_tolerance,
            within_target_value_tolerance=certificate.within_target_value_tolerance,
            min_decrease_satisfied=certificate.min_decrease_satisfied,
        ),
        monotone_accepted_values=certificate.monotone_accepted_values,
        metric_source=None,
        max_metric_condition_number=None,
        method=result.method,
        shift_terms=result.shift_terms,
        claim_boundary=(
            "local parameter-shift gradient descent; no hardware execution or "
            "global optimizer guarantee is implied"
        ),
    )


def _record_natural_gradient(
    *,
    start_index: int,
    result: ParameterShiftNaturalGradientResult,
    certificate: ParameterShiftNaturalGradientCertificate,
) -> OptimizerConvergenceRecord:
    conditions = tuple(
        step.metric_condition_number
        for step in result.steps
        if step.metric_condition_number is not None
    )
    max_condition = max(conditions) if conditions else None
    return OptimizerConvergenceRecord(
        optimizer="parameter_shift_natural_gradient_descent",
        start_index=start_index,
        initial_params=result.initial_params.copy(),
        initial_value=result.initial_value,
        final_value=result.final_value,
        best_value=result.best_value,
        value_decrease=result.initial_value - result.best_value,
        accepted_steps=result.accepted_steps,
        rejected_steps=result.rejected_steps,
        evaluations=result.evaluations,
        final_gradient_norm=result.final_gradient_norm,
        final_natural_gradient_norm=result.final_natural_gradient_norm,
        converged=result.converged,
        reason=result.reason,
        certificate_passed=_certificate_passed(
            monotone_accepted_values=certificate.monotone_accepted_values,
            within_gradient_tolerance=certificate.within_gradient_tolerance,
            within_target_value_tolerance=certificate.within_target_value_tolerance,
            min_decrease_satisfied=certificate.min_decrease_satisfied,
            within_natural_gradient_tolerance=certificate.within_natural_gradient_tolerance,
        ),
        monotone_accepted_values=certificate.monotone_accepted_values,
        metric_source=result.metric_source,
        max_metric_condition_number=max_condition,
        method=result.method,
        shift_terms=result.shift_terms,
        claim_boundary=result.claim_boundary,
    )


def _best_value(records: Sequence[OptimizerConvergenceRecord], optimizer: str) -> float:
    values = [record.best_value for record in records if record.optimizer == optimizer]
    if not values:
        raise ValueError(f"optimizer {optimizer!r} did not produce records")
    return float(min(values))


def _best_optimizer(records: Sequence[OptimizerConvergenceRecord]) -> str:
    best = min(records, key=lambda record: record.best_value)
    return best.optimizer


def _natural_not_worse_count(
    records: Sequence[OptimizerConvergenceRecord],
    *,
    start_count: int,
    tolerance: float,
) -> int:
    count = 0
    for start_index in range(start_count):
        per_start = [record for record in records if record.start_index == start_index]
        gradient = next(
            record
            for record in per_start
            if record.optimizer == "parameter_shift_gradient_descent"
        )
        natural = next(
            record
            for record in per_start
            if record.optimizer == "parameter_shift_natural_gradient_descent"
        )
        if natural.best_value <= gradient.best_value + tolerance:
            count += 1
    return count


def run_parameter_shift_optimizer_comparison(
    objective: ScalarObjective | None = None,
    starts: ArrayLike | None = None,
    *,
    metric_tensor: MetricTensorInput = None,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    backend: str = "statevector",
    learning_rate: float = 0.4,
    max_steps: int = 12,
    gradient_tolerance: float = 1e-8,
    natural_gradient_tolerance: float = 1e-8,
    certificate_gradient_tolerance: float | None = None,
    certificate_natural_gradient_tolerance: float | None = None,
    target_value: float | None = None,
    target_value_tolerance: float | None = None,
    min_decrease: float = 0.0,
    comparison_tolerance: float = 1e-10,
    require_natural_not_worse: bool = True,
    allow_hardware: bool = False,
) -> OptimizerComparisonSuiteResult:
    """Run a bounded multi-start optimizer comparison for phase objectives.

    The default objective is a smooth anisotropic phase-rotation cost. The
    default metric matches that anisotropy so the audit can verify that
    natural-gradient preconditioning helps the slow phase axis. Custom
    objectives are supported, but callers must provide a metric tensor if they
    want natural-gradient semantics beyond the identity baseline.
    """
    start_matrix = _as_start_matrix(starts)
    rate = _positive_float("learning_rate", learning_rate)
    steps_limit = _positive_int("max_steps", max_steps)
    grad_tol = _non_negative_float("gradient_tolerance", gradient_tolerance)
    natural_tol = _non_negative_float("natural_gradient_tolerance", natural_gradient_tolerance)
    cert_grad_tol = None
    if certificate_gradient_tolerance is not None:
        cert_grad_tol = _non_negative_float(
            "certificate_gradient_tolerance",
            certificate_gradient_tolerance,
        )
    cert_natural_tol = None
    if certificate_natural_gradient_tolerance is not None:
        cert_natural_tol = _non_negative_float(
            "certificate_natural_gradient_tolerance",
            certificate_natural_gradient_tolerance,
        )
    min_drop = _non_negative_float("min_decrease", min_decrease)
    compare_tol = _non_negative_float("comparison_tolerance", comparison_tolerance)
    if target_value is None and target_value_tolerance is not None:
        raise ValueError("target_value_tolerance requires target_value")
    target = None if target_value is None else float(target_value)
    target_tol = 0.0
    if target is not None and target_value_tolerance is not None:
        target_tol = _non_negative_float("target_value_tolerance", target_value_tolerance)
    objective_was_default = objective is None
    weights = _default_weights(start_matrix.shape[1])
    objective_fn = _default_objective(weights) if objective_was_default else objective
    if objective_fn is None:
        raise ValueError("objective must be provided when default objective construction fails")
    effective_metric = (
        _default_metric(weights)
        if metric_tensor is None and objective_was_default
        else metric_tensor
    )
    records: list[OptimizerConvergenceRecord] = []

    for start_index, start in enumerate(start_matrix):
        gradient_result = parameter_shift_gradient_descent(
            objective_fn,
            start,
            parameters=parameters,
            rule=rule,
            backend=backend,
            learning_rate=rate,
            max_steps=steps_limit,
            gradient_tolerance=grad_tol,
            allow_hardware=allow_hardware,
        )
        if target is None:
            gradient_certificate = validate_parameter_shift_training(
                gradient_result,
                gradient_tolerance=cert_grad_tol,
                min_decrease=min_drop,
            )
        else:
            gradient_certificate = validate_parameter_shift_training(
                gradient_result,
                gradient_tolerance=cert_grad_tol,
                target_value=target,
                target_value_tolerance=target_tol,
                min_decrease=min_drop,
            )
        records.append(
            _record_gradient_descent(
                start_index=start_index,
                result=gradient_result,
                certificate=gradient_certificate,
            )
        )

        natural_result = parameter_shift_natural_gradient_descent(
            objective_fn,
            start,
            metric_tensor=effective_metric,
            parameters=parameters,
            rule=rule,
            backend=backend,
            learning_rate=rate,
            max_steps=steps_limit,
            gradient_tolerance=grad_tol,
            natural_gradient_tolerance=natural_tol,
            allow_hardware=allow_hardware,
        )
        if target is None:
            natural_certificate = validate_natural_gradient_training(
                natural_result,
                gradient_tolerance=cert_grad_tol,
                natural_gradient_tolerance=cert_natural_tol,
                min_decrease=min_drop,
            )
        else:
            natural_certificate = validate_natural_gradient_training(
                natural_result,
                gradient_tolerance=cert_grad_tol,
                natural_gradient_tolerance=cert_natural_tol,
                target_value=target,
                target_value_tolerance=target_tol,
                min_decrease=min_drop,
            )
        records.append(
            _record_natural_gradient(
                start_index=start_index,
                result=natural_result,
                certificate=natural_certificate,
            )
        )

    natural_not_worse = _natural_not_worse_count(
        records,
        start_count=start_matrix.shape[0],
        tolerance=compare_tol,
    )
    certificates_passed = all(record.certificate_passed for record in records)
    comparison_passed = (
        natural_not_worse == start_matrix.shape[0] if require_natural_not_worse else True
    )
    return OptimizerComparisonSuiteResult(
        records=tuple(records),
        start_count=int(start_matrix.shape[0]),
        passed=certificates_passed and comparison_passed,
        best_optimizer=_best_optimizer(records),
        gradient_descent_best_value=_best_value(records, "parameter_shift_gradient_descent"),
        natural_gradient_best_value=_best_value(
            records,
            "parameter_shift_natural_gradient_descent",
        ),
        natural_gradient_not_worse_count=natural_not_worse,
        comparison_tolerance=compare_tol,
        claim_boundary=(
            "local multi-start convergence evidence for supported smooth "
            "phase objectives; non-isolated functional audit only, not a "
            "hardware, throughput, or global optimality claim"
        ),
    )
