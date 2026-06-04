# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Parameter Shift
"""Parameter-shift gradients for phase and Kuramoto-XY VQE objectives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import (
    GradientResult,
    Parameter,
    ParameterShiftRule,
    ShotAllocationResult,
    StochasticGradientResult,
    allocate_parameter_shift_shots,
    value_and_parameter_shift_grad,
)
from ..differentiable import (
    parameter_shift_gradient as _core_parameter_shift_gradient,
)
from ..differentiable import (
    parameter_shift_gradient_with_uncertainty as _core_parameter_shift_gradient_with_uncertainty,
)
from ..hardware.classical import classical_exact_diag
from .gradient_backend import (
    QuantumGradientPlan,
    plan_quantum_gradient_backend,
)
from .phase_vqe import PhaseVQE

FloatArray = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]


@dataclass(frozen=True)
class ParamShiftConvergenceDiagnostics:
    """Machine-checkable convergence evidence for parameter-shift training."""

    initial_energy: float
    final_energy: float
    best_energy: float
    energy_decrease: float
    max_energy_increase: float
    monotone_energy: bool
    best_improved: bool
    final_gradient_norm: float
    accepted_steps: int
    rejected_steps: int
    line_search_backtracks: tuple[int, ...]
    parameter_shift_evaluations: int
    exact_energy: float | None
    exact_gap: float | None
    within_energy_tolerance: bool | None
    within_gradient_tolerance: bool | None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable convergence evidence."""
        return {
            "initial_energy": self.initial_energy,
            "final_energy": self.final_energy,
            "best_energy": self.best_energy,
            "energy_decrease": self.energy_decrease,
            "max_energy_increase": self.max_energy_increase,
            "monotone_energy": self.monotone_energy,
            "best_improved": self.best_improved,
            "final_gradient_norm": self.final_gradient_norm,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "line_search_backtracks": list(self.line_search_backtracks),
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
            "exact_energy": self.exact_energy,
            "exact_gap": self.exact_gap,
            "within_energy_tolerance": self.within_energy_tolerance,
            "within_gradient_tolerance": self.within_gradient_tolerance,
        }


@dataclass(frozen=True)
class ParamShiftVQEResult:
    """Result of a local simulator-backed parameter-shift VQE descent."""

    initial_energy: float
    final_energy: float
    best_energy: float
    final_params: FloatArray
    best_params: FloatArray
    energies: tuple[float, ...]
    gradient_norms: tuple[float, ...]
    steps: int
    converged: bool
    exact_energy: float | None = None
    energy_gap: float | None = None
    accepted_steps: int = 0
    rejected_steps: int = 0
    line_search_backtracks: tuple[int, ...] = ()
    step_sizes: tuple[float, ...] = ()
    parameter_shift_evaluations: int = 0

    @property
    def optimal_params(self) -> FloatArray:
        """Return the best parameters found during the descent."""
        return cast(FloatArray, self.best_params.copy())

    def to_dict(self) -> dict[str, object]:
        """Return a backwards-compatible mapping for notebooks and scripts."""
        return {
            "initial_energy": self.initial_energy,
            "final_energy": self.final_energy,
            "best_energy": self.best_energy,
            "energy": self.best_energy,
            "final_params": self.final_params.copy(),
            "best_params": self.best_params.copy(),
            "optimal_params": self.best_params.copy(),
            "energy_history": list(self.energies),
            "gradient_norms": list(self.gradient_norms),
            "steps": self.steps,
            "converged": self.converged,
            "exact_energy": self.exact_energy,
            "energy_gap": self.energy_gap,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "line_search_backtracks": list(self.line_search_backtracks),
            "step_sizes": list(self.step_sizes),
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
            "convergence_diagnostics": validate_param_shift_convergence(self).to_dict(),
        }


@dataclass(frozen=True)
class GradientVerificationResult:
    """Finite-difference agreement certificate for parameter-shift gradients."""

    method: str
    passed: bool
    max_abs_error: float
    max_relative_error: float
    absolute_tolerance: float
    relative_tolerance: float
    finite_difference_step: float
    parameters: FloatArray
    analytic_gradient: FloatArray
    finite_difference_gradient: FloatArray
    parameter_shift_evaluations: int
    finite_difference_evaluations: int

    @property
    def total_evaluations(self) -> int:
        """Return objective evaluations spent on verification."""
        return self.parameter_shift_evaluations + self.finite_difference_evaluations

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable gradient verification evidence."""
        return {
            "method": self.method,
            "passed": self.passed,
            "max_abs_error": self.max_abs_error,
            "max_relative_error": self.max_relative_error,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "finite_difference_step": self.finite_difference_step,
            "parameters": self.parameters.tolist(),
            "analytic_gradient": self.analytic_gradient.tolist(),
            "finite_difference_gradient": self.finite_difference_gradient.tolist(),
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
            "finite_difference_evaluations": self.finite_difference_evaluations,
            "total_evaluations": self.total_evaluations,
        }


def _as_finite_vector(name: str, values: ArrayLike, *, width: int | None = None) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _initial_parameters(
    n_params: int,
    *,
    initial_params: ArrayLike | None,
    seed: int | None,
) -> FloatArray:
    if n_params <= 0:
        raise ValueError("n_params must be positive")
    if initial_params is not None:
        return _as_finite_vector("initial_params", initial_params, width=n_params)
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.1, 0.1, n_params).astype(np.float64)


def _normalise_iteration_count(
    *,
    steps: int | None,
    n_iterations: int | None,
) -> int:
    if steps is not None and n_iterations is not None and steps != n_iterations:
        raise ValueError("steps and n_iterations must match when both are provided")
    count = steps if steps is not None else n_iterations
    if count is None:
        count = 100
    if count <= 0:
        raise ValueError("steps must be positive")
    return int(count)


def _shot_vector(shots: int | ArrayLike, width: int) -> FloatArray:
    if isinstance(shots, bool):
        raise ValueError("shots must be a positive integer or one-dimensional shot array")
    if isinstance(shots, int):
        if shots <= 0:
            raise ValueError("shots must be positive")
        return np.full(width, float(shots), dtype=np.float64)
    values = _as_finite_vector("shots", shots, width=width)
    if not np.all(values > 0.0) or not np.allclose(values, np.round(values)):
        raise ValueError("shots must contain positive integers")
    return values


def _validate_non_negative_threshold(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    threshold = float(value)
    if threshold < 0.0 or not np.isfinite(threshold):
        raise ValueError(f"{name} must be finite and non-negative")
    return threshold


def _validate_positive_threshold(name: str, value: float) -> float:
    threshold = float(value)
    if threshold <= 0.0 or not np.isfinite(threshold):
        raise ValueError(f"{name} must be finite and positive")
    return threshold


def _finite_difference_gradient(
    objective: ScalarObjective,
    values: FloatArray,
    *,
    step: float,
) -> FloatArray:
    gradient = np.zeros_like(values, dtype=np.float64)
    for index in range(values.size):
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        plus_value = float(objective(plus))
        minus_value = float(objective(minus))
        if not np.isfinite(plus_value) or not np.isfinite(minus_value):
            raise ValueError("objective must return finite scalars for finite-difference probes")
        gradient[index] = (plus_value - minus_value) / (2.0 * step)
    return gradient


def parameter_shift_gradient(
    objective: ScalarObjective,
    values: ArrayLike,
    shift: float = float(np.pi / 2.0),
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> FloatArray:
    """Return a parameter-shift gradient while preserving the legacy `shift` keyword."""
    shift_value = float(shift)
    if rule is not None and not np.isclose(shift_value, np.pi / 2.0):
        raise ValueError("shift must not be overridden when rule is provided")
    if rule is None:
        if not np.isfinite(shift_value) or shift_value <= 0.0:
            raise ValueError("shift must be finite and positive")
        denominator = 2.0 * np.sin(shift_value)
        if abs(denominator) <= 1e-15:
            raise ValueError("shift must not make the parameter-shift denominator singular")
        rule = ParameterShiftRule(
            shift=shift_value,
            coefficient=float(1.0 / denominator),
        )
    return _core_parameter_shift_gradient(
        objective,
        values,
        parameters=parameters,
        rule=rule,
    )


def verify_parameter_shift_gradient(
    objective: ScalarObjective,
    values: ArrayLike,
    shift: float = float(np.pi / 2.0),
    *,
    finite_difference_step: float = 1e-6,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-5,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> GradientVerificationResult:
    """Verify parameter-shift gradients against central finite differences.

    This helper is designed for notebooks, CI conformance checks, and provider
    adapter smoke tests. It does not claim finite differences are a production
    gradient method; they are used as an independent local diagnostic for small,
    smooth objectives.
    """
    values_vector = _as_finite_vector("values", values)
    fd_step = _validate_positive_threshold(
        "finite_difference_step",
        finite_difference_step,
    )
    abs_tol = _validate_non_negative_threshold(
        "absolute_tolerance",
        absolute_tolerance,
    )
    rel_tol = _validate_non_negative_threshold(
        "relative_tolerance",
        relative_tolerance,
    )
    if abs_tol is None or rel_tol is None:
        raise ValueError("absolute_tolerance and relative_tolerance must be provided")

    analytic = parameter_shift_gradient(
        objective,
        values_vector,
        shift=shift,
        parameters=parameters,
        rule=rule,
    )
    finite_difference = _finite_difference_gradient(
        objective,
        values_vector,
        step=fd_step,
    )
    abs_errors = np.abs(analytic - finite_difference)
    denominator = np.maximum(
        np.maximum(np.abs(analytic), np.abs(finite_difference)),
        np.finfo(np.float64).eps,
    )
    relative_errors = abs_errors / denominator

    return GradientVerificationResult(
        method="parameter_shift_vs_central_finite_difference",
        passed=bool(
            np.allclose(
                analytic,
                finite_difference,
                atol=abs_tol,
                rtol=rel_tol,
            )
        ),
        max_abs_error=float(np.max(abs_errors)) if abs_errors.size else 0.0,
        max_relative_error=float(np.max(relative_errors)) if relative_errors.size else 0.0,
        absolute_tolerance=float(abs_tol),
        relative_tolerance=float(rel_tol),
        finite_difference_step=fd_step,
        parameters=values_vector.copy(),
        analytic_gradient=analytic.astype(np.float64, copy=True),
        finite_difference_gradient=finite_difference.astype(np.float64, copy=True),
        parameter_shift_evaluations=2 * values_vector.size,
        finite_difference_evaluations=2 * values_vector.size,
    )


def verify_vqe_parameter_shift_gradient(
    vqe: PhaseVQE,
    params: ArrayLike,
    *,
    finite_difference_step: float = 1e-6,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-5,
) -> GradientVerificationResult:
    """Verify a `PhaseVQE` parameter-shift gradient against finite differences."""
    values = _as_finite_vector("params", params, width=vqe.n_params)
    return verify_parameter_shift_gradient(
        vqe._cost,
        values,
        finite_difference_step=finite_difference_step,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def parameter_shift_gradient_with_uncertainty(
    plus_values: ArrayLike,
    minus_values: ArrayLike,
    plus_variances: ArrayLike,
    minus_variances: ArrayLike,
    *,
    shots: int | ArrayLike,
    backend: str = "finite_shot_simulator",
    value: float = 0.0,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    confidence_level: float = 0.95,
    confidence_z: float = 1.959963984540054,
) -> StochasticGradientResult:
    """Return finite-shot parameter-shift gradients with propagated uncertainty."""
    plus = _as_finite_vector("plus_values", plus_values)
    shot_counts = _shot_vector(shots, plus.size)
    plan = plan_quantum_gradient_backend(
        backend,
        n_params=plus.size,
        method="stochastic_parameter_shift",
        shots=int(np.max(shot_counts)),
        finite_shot=True,
        confidence_level=confidence_level,
    )
    if plan.fail_closed:
        joined = "; ".join(plan.reasons)
        raise ValueError(f"backend gradient plan is unsupported: {joined}")
    return _core_parameter_shift_gradient_with_uncertainty(
        plus,
        minus_values,
        plus_variances,
        minus_variances,
        shot_counts,
        value=value,
        parameters=parameters,
        rule=rule,
        confidence_level=confidence_level,
        confidence_z=confidence_z,
    )


def plan_parameter_shift_shots(
    plus_variances: ArrayLike,
    minus_variances: ArrayLike,
    *,
    target_standard_error: float,
    min_shots: int = 1,
    max_shots_per_evaluation: int | None = None,
) -> ShotAllocationResult:
    """Plan finite-shot allocations for parameter-shift gradients."""
    return allocate_parameter_shift_shots(
        plus_variances,
        minus_variances,
        target_standard_error=target_standard_error,
        min_shots=min_shots,
        max_shots_per_evaluation=max_shots_per_evaluation,
    )


def validate_param_shift_convergence(
    result: ParamShiftVQEResult,
    *,
    energy_tolerance: float = 1e-10,
    target_gap: float | None = None,
    gradient_tolerance: float | None = None,
) -> ParamShiftConvergenceDiagnostics:
    """Return explicit convergence evidence for a parameter-shift VQE run.

    The validator is intentionally diagnostic, not promotional. It reports
    whether the observed energy history is monotone under the accepted
    backtracking line search, whether the best value improved over the initial
    point, how many parameter-shift objective evaluations were spent, and
    whether optional exact-energy or gradient-norm tolerances were reached.
    """
    energy_tol = _validate_non_negative_threshold("energy_tolerance", energy_tolerance)
    if energy_tol is None:
        energy_tol = 0.0
    gap_tol = _validate_non_negative_threshold("target_gap", target_gap)
    grad_tol = _validate_non_negative_threshold("gradient_tolerance", gradient_tolerance)

    energies = np.asarray(result.energies, dtype=float)
    if energies.ndim != 1 or energies.size == 0:
        raise ValueError("result.energies must contain at least one energy")
    if not np.all(np.isfinite(energies)):
        raise ValueError("result.energies must contain only finite values")

    deltas = np.diff(energies)
    max_increase = float(np.max(deltas)) if deltas.size else 0.0
    max_increase = max(0.0, max_increase)
    final_gradient_norm = (
        float(result.gradient_norms[-1]) if result.gradient_norms else float("nan")
    )
    exact_gap = result.energy_gap
    if exact_gap is None and result.exact_energy is not None:
        exact_gap = abs(result.best_energy - result.exact_energy)

    return ParamShiftConvergenceDiagnostics(
        initial_energy=float(result.initial_energy),
        final_energy=float(result.final_energy),
        best_energy=float(result.best_energy),
        energy_decrease=float(result.initial_energy - result.best_energy),
        max_energy_increase=max_increase,
        monotone_energy=bool(max_increase <= energy_tol),
        best_improved=bool(result.best_energy <= result.initial_energy + energy_tol),
        final_gradient_norm=final_gradient_norm,
        accepted_steps=int(result.accepted_steps),
        rejected_steps=int(result.rejected_steps),
        line_search_backtracks=tuple(int(value) for value in result.line_search_backtracks),
        parameter_shift_evaluations=int(result.parameter_shift_evaluations),
        exact_energy=result.exact_energy,
        exact_gap=exact_gap,
        within_energy_tolerance=None
        if gap_tol is None
        else exact_gap is not None and exact_gap <= gap_tol,
        within_gradient_tolerance=None
        if grad_tol is None
        else np.isfinite(final_gradient_norm) and final_gradient_norm <= grad_tol,
    )


def value_and_vqe_grad(vqe: PhaseVQE, params: ArrayLike) -> GradientResult:
    """Return the VQE energy and parameter-shift gradient for a `PhaseVQE` object."""
    values = _as_finite_vector("params", params, width=vqe.n_params)
    return vqe.value_and_parameter_shift_gradient(values)


def vqe_with_param_shift(
    objective_or_k: ScalarObjective | ArrayLike,
    omega: ArrayLike | None = None,
    *,
    n_params: int | None = None,
    ansatz_reps: int = 2,
    threshold: float = 0.01,
    learning_rate: float = 0.1,
    steps: int | None = None,
    n_iterations: int | None = None,
    tolerance: float = 1e-8,
    seed: int | None = 0,
    initial_params: ArrayLike | None = None,
) -> ParamShiftVQEResult:
    """Run local parameter-shift gradient descent for a callable or Kuramoto-XY VQE.

    Passing a callable preserves the generic scalar-objective route and requires
    `n_params`. Passing `K` as the first argument and `omega` as the second
    argument builds a `PhaseVQE` objective and uses the repository ansatz.
    """
    if learning_rate <= 0.0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")

    iteration_count = _normalise_iteration_count(steps=steps, n_iterations=n_iterations)
    exact_energy: float | None = None

    if callable(objective_or_k):
        if omega is not None:
            raise ValueError("omega must be omitted when objective_or_k is callable")
        if n_params is None:
            raise ValueError("n_params is required when objective_or_k is callable")
        objective = objective_or_k
        width = int(n_params)
    else:
        if omega is None:
            raise ValueError("omega is required when objective_or_k is a coupling matrix")
        k_matrix = np.asarray(objective_or_k, dtype=float)
        omega_values = np.asarray(omega, dtype=float)
        vqe = PhaseVQE(
            k_matrix,
            omega_values,
            ansatz_reps=ansatz_reps,
            threshold=threshold,
        )
        objective = vqe._cost
        width = vqe.n_params
        exact = classical_exact_diag(len(omega_values), K=k_matrix, omega=omega_values)
        exact_energy = float(exact["ground_energy"])

    params = _initial_parameters(width, initial_params=initial_params, seed=seed)
    energy = float(objective(params.copy()))
    if not np.isfinite(energy):
        raise ValueError("objective must return a finite scalar at initial_params")

    energies = [energy]
    gradient_norms: list[float] = []
    line_search_backtracks: list[int] = []
    step_sizes: list[float] = []
    best_energy = energy
    best_params = params.copy()
    converged = False
    accepted_steps = 0
    rejected_steps = 0
    parameter_shift_evaluations = 0

    for _ in range(iteration_count):
        grad_result = value_and_parameter_shift_grad(objective, params)
        parameter_shift_evaluations += grad_result.evaluations
        gradient = grad_result.gradient
        grad_norm = float(np.linalg.norm(gradient))
        gradient_norms.append(grad_norm)
        if grad_norm <= tolerance:
            converged = True
            break

        step_size = float(learning_rate)
        accepted = False
        candidate = params.copy()
        candidate_energy = energy
        backtracks = 0
        for attempt in range(12):
            trial = params - step_size * gradient
            trial_energy = float(objective(trial.copy()))
            if np.isfinite(trial_energy) and trial_energy <= energy:
                candidate = trial
                candidate_energy = trial_energy
                accepted = True
                backtracks = attempt
                break
            step_size *= 0.5

        if not accepted:
            rejected_steps += 1
            break

        accepted_steps += 1
        line_search_backtracks.append(backtracks)
        step_sizes.append(step_size)
        params = candidate.astype(np.float64, copy=False)
        energy = candidate_energy
        energies.append(energy)
        if energy < best_energy:
            best_energy = energy
            best_params = params.copy()

    gap = None if exact_energy is None else abs(best_energy - exact_energy)
    return ParamShiftVQEResult(
        initial_energy=float(energies[0]),
        final_energy=float(energies[-1]),
        best_energy=float(best_energy),
        final_params=params.copy(),
        best_params=best_params.copy(),
        energies=tuple(float(value) for value in energies),
        gradient_norms=tuple(gradient_norms),
        steps=len(gradient_norms),
        converged=converged,
        exact_energy=exact_energy,
        energy_gap=gap,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        line_search_backtracks=tuple(line_search_backtracks),
        step_sizes=tuple(float(value) for value in step_sizes),
        parameter_shift_evaluations=parameter_shift_evaluations,
    )


__all__ = [
    "GradientResult",
    "ParamShiftVQEResult",
    "Parameter",
    "ParameterShiftRule",
    "ParamShiftConvergenceDiagnostics",
    "QuantumGradientPlan",
    "GradientVerificationResult",
    "ShotAllocationResult",
    "StochasticGradientResult",
    "parameter_shift_gradient",
    "parameter_shift_gradient_with_uncertainty",
    "plan_parameter_shift_shots",
    "plan_quantum_gradient_backend",
    "value_and_parameter_shift_grad",
    "value_and_vqe_grad",
    "verify_parameter_shift_gradient",
    "verify_vqe_parameter_shift_gradient",
    "validate_param_shift_convergence",
    "vqe_with_param_shift",
]
