# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Trainability Diagnostics
"""Barren-plateau diagnostics and finite-shot dry-run planning."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import (
    Parameter,
    ParameterShiftRule,
    ShotAllocationResult,
    allocate_parameter_shift_shots,
    value_and_parameter_shift_grad,
)
from .gradient_backend import QuantumGradientPlan, plan_quantum_gradient_backend

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective: TypeAlias = Callable[[FloatArray], float]
TrainabilityStatus: TypeAlias = Literal[
    "trainable",
    "low_gradient_variance",
    "flat_objective",
    "shot_limited",
]
TRAINABILITY_CLAIM_BOUNDARY = (
    "local parameter-shift trainability diagnostic and finite-shot dry-run only; "
    "no hardware execution, provider submission, convergence guarantee, or benchmark "
    "promotion is implied"
)


@dataclass(frozen=True)
class TrainabilityGradientSample:
    """One parameter-shift gradient sample used by a trainability report.

    Parameters
    ----------
    index
        Zero-based sample index from the caller-supplied parameter matrix.
    params
        Parameter vector evaluated for this sample.
    value
        Scalar objective value at ``params``.
    gradient
        Parameter-shift gradient at ``params``.
    gradient_norm
        Euclidean norm of ``gradient``.
    evaluations
        Objective evaluations consumed by the parameter-shift rule.
    method
        Gradient method reported by the differentiable core.
    """

    index: int
    params: FloatArray
    value: float
    gradient: FloatArray
    gradient_norm: float
    evaluations: int
    method: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready gradient-sample metadata.

        Returns
        -------
        dict[str, object]
            Scalar and array metadata converted to built-in Python containers.
        """
        return {
            "index": self.index,
            "params": self.params.tolist(),
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "gradient_norm": self.gradient_norm,
            "evaluations": self.evaluations,
            "method": self.method,
        }


@dataclass(frozen=True)
class AdaptiveShotAllocationDryRun:
    """Finite-shot allocation and cost estimate without backend execution.

    Parameters
    ----------
    allocation
        Variance-aware plus/minus shot allocation returned by the stochastic
        differentiable estimator.
    backend_plan
        Gradient backend plan used only for dry-run evaluation accounting.
    variance_source
        Source of the plus/minus variances, either caller-supplied or derived
        from sampled gradient variance.
    estimated_shift_evaluations
        Number of shifted circuit evaluations required by the backend plan.
    estimated_quantum_shots
        Sum of all planned plus/minus shot counts.
    estimated_cost
        ``estimated_quantum_shots * cost_per_shot`` in ``cost_unit``.
    cost_unit
        Caller-supplied unit label for the dry-run cost estimate.
    capped
        Whether a shot cap prevented the target standard error.
    hardware_execution
        Always ``False`` for this dry-run record.
    """

    allocation: ShotAllocationResult
    backend_plan: QuantumGradientPlan
    variance_source: str
    estimated_shift_evaluations: int
    estimated_quantum_shots: int
    estimated_cost: float
    cost_unit: str
    capped: bool
    hardware_execution: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready dry-run allocation metadata.

        Returns
        -------
        dict[str, object]
            Allocation, backend, cost, and safety metadata converted to
            built-in Python containers.
        """
        return {
            "allocation": {
                "shots": self.allocation.shots.tolist(),
                "predicted_standard_error": self.allocation.predicted_standard_error.tolist(),
                "covariance": self.allocation.covariance.tolist(),
                "target_standard_error": self.allocation.target_standard_error,
                "total_shots": self.allocation.total_shots,
                "method": self.allocation.method,
                "parameter_names": list(self.allocation.parameter_names),
                "trainable": list(self.allocation.trainable),
            },
            "backend": self.backend_plan.backend,
            "backend_method": self.backend_plan.method,
            "variance_source": self.variance_source,
            "estimated_shift_evaluations": self.estimated_shift_evaluations,
            "estimated_quantum_shots": self.estimated_quantum_shots,
            "estimated_cost": self.estimated_cost,
            "cost_unit": self.cost_unit,
            "capped": self.capped,
            "hardware_execution": self.hardware_execution,
        }


@dataclass(frozen=True)
class BarrenPlateauTrainabilityReport:
    """Aggregate trainability diagnostics for a bounded phase objective.

    Parameters
    ----------
    samples
        Parameter-shift gradient samples used to compute the diagnostics.
    gradient_mean
        Per-parameter empirical mean of the sampled gradients.
    gradient_variance
        Per-parameter unbiased empirical gradient variance.
    mean_gradient_norm
        Mean Euclidean gradient norm across samples.
    gradient_norm_variance
        Unbiased empirical variance of gradient norms.
    barren_plateau_detected
        ``True`` when both sampled gradient norm and variance sit below the
        configured thresholds.
    status
        Compact trainability classification derived from the diagnostics.
    warnings
        Machine-readable warning labels for low norm, low variance, and shot
        caps.
    shot_dry_run
        Adaptive shot-allocation dry run for finite-shot parameter-shift use.
    claim_boundary
        Explicit non-hardware and non-promotion boundary for the report.
    """

    samples: tuple[TrainabilityGradientSample, ...]
    gradient_mean: FloatArray
    gradient_variance: FloatArray
    mean_gradient_norm: float
    gradient_norm_variance: float
    barren_plateau_detected: bool
    status: TrainabilityStatus
    warnings: tuple[str, ...]
    shot_dry_run: AdaptiveShotAllocationDryRun
    claim_boundary: str

    @property
    def sample_count(self) -> int:
        """Return the number of gradient samples in the report.

        Returns
        -------
        int
            Number of sample records used by this report.
        """
        return len(self.samples)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready trainability report metadata.

        Returns
        -------
        dict[str, object]
            Report payload suitable for JSON artifact emission.
        """
        return {
            "samples": [sample.to_dict() for sample in self.samples],
            "sample_count": self.sample_count,
            "gradient_mean": self.gradient_mean.tolist(),
            "gradient_variance": self.gradient_variance.tolist(),
            "mean_gradient_norm": self.mean_gradient_norm,
            "gradient_norm_variance": self.gradient_norm_variance,
            "barren_plateau_detected": self.barren_plateau_detected,
            "status": self.status,
            "warnings": list(self.warnings),
            "shot_dry_run": self.shot_dry_run.to_dict(),
            "claim_boundary": self.claim_boundary,
        }


def run_barren_plateau_trainability_report(
    objective: ScalarObjective,
    sample_params: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    plus_variances: ArrayLike | None = None,
    minus_variances: ArrayLike | None = None,
    target_standard_error: float = 0.02,
    gradient_variance_threshold: float = 1.0e-8,
    gradient_norm_threshold: float = 1.0e-6,
    min_shots: int = 16,
    max_shots_per_evaluation: int | None = None,
    backend: str = "finite_shot_simulator",
    cost_per_shot: float = 0.0,
    cost_unit: str = "abstract_shot_cost",
) -> BarrenPlateauTrainabilityReport:
    """Build a trainability report and finite-shot dry-run allocation.

    Parameters
    ----------
    objective
        Scalar objective accepting a one-dimensional ``float64`` parameter
        vector.
    sample_params
        Matrix of parameter vectors. At least two samples are required so the
        empirical gradient variance is defined.
    parameters
        Optional parameter metadata controlling names and trainable masks.
    rule
        Optional single- or multi-frequency parameter-shift rule.
    plus_variances, minus_variances
        Optional caller-supplied finite-shot measurement variances. When absent,
        the report derives a conservative variance tensor from sampled gradient
        variance and ``gradient_variance_threshold``.
    target_standard_error
        Desired standard error for each trainable gradient component.
    gradient_variance_threshold
        Low-variance threshold and variance floor for derived shot allocation.
    gradient_norm_threshold
        Low-gradient-norm threshold for barren-plateau classification.
    min_shots
        Minimum shots for each plus/minus shifted evaluation.
    max_shots_per_evaluation
        Optional cap for each shifted evaluation.
    backend
        Backend name passed to the quantum-gradient planner. Hardware backends
        fail closed because this function never requests hardware approval.
    cost_per_shot
        Non-negative dry-run cost multiplier.
    cost_unit
        Non-empty label for ``estimated_cost``.

    Returns
    -------
    BarrenPlateauTrainabilityReport
        Gradient-variance diagnostics and a zero-execution shot-allocation dry
        run.

    Raises
    ------
    ValueError
        If inputs are malformed or if the backend planner reports a fail-closed
        route.

    The report samples parameter-shift gradients across caller-supplied
    parameter vectors, flags flat low-variance landscapes, and uses the existing
    finite-shot allocator to estimate shot counts before any backend execution.
    """
    starts = _as_sample_matrix(sample_params)
    variance_floor = _non_negative_float(
        "gradient_variance_threshold", gradient_variance_threshold
    )
    norm_threshold = _non_negative_float("gradient_norm_threshold", gradient_norm_threshold)
    target = _positive_float("target_standard_error", target_standard_error)
    shot_floor = _positive_int("min_shots", min_shots)
    shot_cap = _optional_positive_int("max_shots_per_evaluation", max_shots_per_evaluation)
    unit_cost = _non_negative_float("cost_per_shot", cost_per_shot)
    if not cost_unit.strip():
        raise ValueError("cost_unit must be a non-empty string")

    samples = _sample_parameter_shift_gradients(
        objective,
        starts,
        parameters=parameters,
        rule=rule,
    )
    gradients = np.vstack([sample.gradient for sample in samples])
    norms = np.array([sample.gradient_norm for sample in samples], dtype=np.float64)
    gradient_mean = np.mean(gradients, axis=0).astype(np.float64, copy=False)
    gradient_variance = np.var(gradients, axis=0, ddof=1).astype(np.float64, copy=False)
    mean_norm = float(np.mean(norms))
    norm_variance = float(np.var(norms, ddof=1))
    plus_var, minus_var, variance_source = _resolve_shot_variances(
        gradient_variance,
        variance_floor=variance_floor,
        plus_variances=plus_variances,
        minus_variances=minus_variances,
        rule=rule,
    )
    allocation = allocate_parameter_shift_shots(
        plus_var,
        minus_var,
        target_standard_error=target,
        parameters=parameters,
        rule=rule,
        min_shots=shot_floor,
        max_shots_per_evaluation=shot_cap,
    )
    shift_terms = len((rule or ParameterShiftRule()).terms)
    backend_plan = plan_quantum_gradient_backend(
        backend,
        n_params=starts.shape[1],
        shift_terms=shift_terms,
        method="stochastic_parameter_shift",
        shots=int(np.max(allocation.shots)),
        finite_shot=True,
        confidence_level=0.95,
    )
    if backend_plan.fail_closed:
        joined = "; ".join(backend_plan.reasons)
        raise ValueError(f"trainability dry-run backend is unsupported: {joined}")
    dry_run = AdaptiveShotAllocationDryRun(
        allocation=allocation,
        backend_plan=backend_plan,
        variance_source=variance_source,
        estimated_shift_evaluations=backend_plan.evaluations,
        estimated_quantum_shots=allocation.total_shots,
        estimated_cost=float(allocation.total_shots) * unit_cost,
        cost_unit=cost_unit.strip(),
        capped=bool(np.any(allocation.predicted_standard_error > target * (1.0 + 1.0e-12))),
        hardware_execution=False,
    )
    warnings = _trainability_warnings(
        gradient_variance=gradient_variance,
        mean_gradient_norm=mean_norm,
        gradient_variance_threshold=variance_floor,
        gradient_norm_threshold=norm_threshold,
        dry_run=dry_run,
    )
    plateau = (
        mean_norm <= norm_threshold
        and float(np.max(gradient_variance, initial=0.0)) <= variance_floor
    )
    return BarrenPlateauTrainabilityReport(
        samples=samples,
        gradient_mean=gradient_mean,
        gradient_variance=gradient_variance,
        mean_gradient_norm=mean_norm,
        gradient_norm_variance=norm_variance,
        barren_plateau_detected=plateau,
        status=_trainability_status(plateau=plateau, warnings=warnings),
        warnings=warnings,
        shot_dry_run=dry_run,
        claim_boundary=TRAINABILITY_CLAIM_BOUNDARY,
    )


def _as_sample_matrix(values: ArrayLike) -> FloatArray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
        raise ValueError("sample_params must contain at least two parameter vectors")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("sample_params must contain only finite values")
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


def _optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return _positive_int(name, value)


def _sample_parameter_shift_gradients(
    objective: ScalarObjective,
    starts: FloatArray,
    *,
    parameters: Sequence[Parameter] | None,
    rule: ParameterShiftRule | None,
) -> tuple[TrainabilityGradientSample, ...]:
    samples: list[TrainabilityGradientSample] = []
    for index, params in enumerate(starts):
        result = value_and_parameter_shift_grad(
            objective,
            params,
            parameters=parameters,
            rule=rule,
        )
        samples.append(
            TrainabilityGradientSample(
                index=index,
                params=params.copy(),
                value=result.value,
                gradient=result.gradient.copy(),
                gradient_norm=float(np.linalg.norm(result.gradient)),
                evaluations=result.evaluations,
                method=result.method,
            )
        )
    return tuple(samples)


def _resolve_shot_variances(
    gradient_variance: FloatArray,
    *,
    variance_floor: float,
    plus_variances: ArrayLike | None,
    minus_variances: ArrayLike | None,
    rule: ParameterShiftRule | None,
) -> tuple[ArrayLike, ArrayLike, str]:
    if (plus_variances is None) != (minus_variances is None):
        raise ValueError("plus_variances and minus_variances must be provided together")
    if plus_variances is not None and minus_variances is not None:
        return plus_variances, minus_variances, "caller_supplied"
    base = np.maximum(gradient_variance, variance_floor)
    term_count = len((rule or ParameterShiftRule()).terms)
    if term_count == 1:
        return base, base, "gradient_sample_variance_floor"
    tiled = np.tile(base, (term_count, 1))
    return tiled, tiled, "gradient_sample_variance_floor"


def _trainability_warnings(
    *,
    gradient_variance: FloatArray,
    mean_gradient_norm: float,
    gradient_variance_threshold: float,
    gradient_norm_threshold: float,
    dry_run: AdaptiveShotAllocationDryRun,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if mean_gradient_norm <= gradient_norm_threshold:
        warnings.append("mean_gradient_norm_below_threshold")
    if float(np.max(gradient_variance, initial=0.0)) <= gradient_variance_threshold:
        warnings.append("gradient_variance_below_threshold")
    if dry_run.capped:
        warnings.append("shot_allocation_cap_exceeded_target")
    return tuple(warnings)


def _trainability_status(
    *,
    plateau: bool,
    warnings: tuple[str, ...],
) -> TrainabilityStatus:
    if plateau:
        return "flat_objective"
    if "shot_allocation_cap_exceeded_target" in warnings:
        return "shot_limited"
    if "gradient_variance_below_threshold" in warnings:
        return "low_gradient_variance"
    return "trainable"


__all__ = [
    "AdaptiveShotAllocationDryRun",
    "BarrenPlateauTrainabilityReport",
    "TRAINABILITY_CLAIM_BOUNDARY",
    "TrainabilityGradientSample",
    "TrainabilityStatus",
    "run_barren_plateau_trainability_report",
]
