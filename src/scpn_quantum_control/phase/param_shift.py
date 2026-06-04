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
    best_energy = energy
    best_params = params.copy()
    converged = False

    for _ in range(iteration_count):
        grad_result = value_and_parameter_shift_grad(objective, params)
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
        for _ in range(12):
            trial = params - step_size * gradient
            trial_energy = float(objective(trial.copy()))
            if np.isfinite(trial_energy) and trial_energy <= energy:
                candidate = trial
                candidate_energy = trial_energy
                accepted = True
                break
            step_size *= 0.5

        if not accepted:
            break

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
    )


__all__ = [
    "GradientResult",
    "ParamShiftVQEResult",
    "Parameter",
    "ParameterShiftRule",
    "QuantumGradientPlan",
    "ShotAllocationResult",
    "StochasticGradientResult",
    "parameter_shift_gradient",
    "parameter_shift_gradient_with_uncertainty",
    "plan_parameter_shift_shots",
    "plan_quantum_gradient_backend",
    "value_and_parameter_shift_grad",
    "value_and_vqe_grad",
    "vqe_with_param_shift",
]
