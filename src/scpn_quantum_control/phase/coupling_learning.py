# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Coupling Learning
"""Parameter-shift coupling learning for oscillator observation models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import (
    Parameter,
    ParameterShiftRule,
    value_and_finite_difference_grad,
    value_and_parameter_shift_grad,
)
from .gradient_descent import (
    ParameterShiftTrainingCertificate,
    ParameterShiftTrainingResult,
    parameter_shift_gradient_descent,
    validate_parameter_shift_training,
)

FloatArray: TypeAlias = NDArray[np.float64]
Edge = tuple[int, int]
CouplingObservationModel = Callable[[FloatArray], ArrayLike]


def _as_finite_scalar(name: str, value: object) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = float(raw.item())
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


def _as_finite_vector(name: str, values: ArrayLike) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


@dataclass(frozen=True)
class CouplingLearningResult:
    """Auditable parameter-shift coupling-learning result."""

    training: ParameterShiftTrainingResult
    certificate: ParameterShiftTrainingCertificate
    initial_coupling_matrix: FloatArray
    learned_coupling_matrix: FloatArray
    target_observations: FloatArray
    predicted_observations: FloatArray
    residuals: FloatArray
    edges: tuple[Edge, ...]
    backend: str
    claim_boundary: str

    @property
    def best_loss(self) -> float:
        """Return the best full-observation mean-squared loss."""
        return self.training.best_value

    @property
    def max_abs_residual(self) -> float:
        """Return the largest absolute observation residual."""
        return float(np.max(np.abs(self.residuals)))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready coupling-learning evidence."""
        return {
            "training": self.training.to_dict(),
            "certificate": self.certificate.to_dict(),
            "initial_coupling_matrix": self.initial_coupling_matrix.tolist(),
            "learned_coupling_matrix": self.learned_coupling_matrix.tolist(),
            "target_observations": self.target_observations.tolist(),
            "predicted_observations": self.predicted_observations.tolist(),
            "residuals": self.residuals.tolist(),
            "edges": [list(edge) for edge in self.edges],
            "backend": self.backend,
            "best_loss": self.best_loss,
            "max_abs_residual": self.max_abs_residual,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class CouplingGradientVerificationResult:
    """Finite-difference agreement certificate for coupling-learning gradients."""

    parameters: FloatArray
    parameter_shift_gradient: FloatArray
    finite_difference_gradient: FloatArray
    abs_error: FloatArray
    max_abs_error: float
    objective_value: float
    value_delta: float
    passed: bool
    method: str
    finite_difference_step: float
    tolerance: float
    parameter_shift_evaluations: int
    finite_difference_evaluations: int
    edges: tuple[Edge, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        parameters = _as_finite_vector("verification parameters", self.parameters)
        parameter_shift_gradient = _as_finite_vector(
            "parameter_shift_gradient",
            self.parameter_shift_gradient,
        )
        finite_difference_gradient = _as_finite_vector(
            "finite_difference_gradient",
            self.finite_difference_gradient,
        )
        abs_error = _as_finite_vector("abs_error", self.abs_error)
        if parameter_shift_gradient.shape != parameters.shape:
            raise ValueError("parameter_shift_gradient shape must match parameters")
        if finite_difference_gradient.shape != parameters.shape:
            raise ValueError("finite_difference_gradient shape must match parameters")
        if abs_error.shape != parameters.shape:
            raise ValueError("abs_error shape must match parameters")
        if np.any(abs_error < 0.0):
            raise ValueError("abs_error must be non-negative")
        max_abs_error = _as_finite_scalar("max_abs_error", self.max_abs_error)
        objective_value = _as_finite_scalar("objective_value", self.objective_value)
        value_delta = _as_finite_scalar("value_delta", self.value_delta)
        finite_difference_step = _as_finite_scalar(
            "finite_difference_step",
            self.finite_difference_step,
        )
        tolerance = _as_finite_scalar("tolerance", self.tolerance)
        if max_abs_error < 0.0:
            raise ValueError("max_abs_error must be non-negative")
        if finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be finite and positive")
        if tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a boolean")
        if not self.method:
            raise ValueError("method must be non-empty")
        if self.parameter_shift_evaluations < 1:
            raise ValueError("parameter_shift_evaluations must be positive")
        if self.finite_difference_evaluations < 1:
            raise ValueError("finite_difference_evaluations must be positive")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "parameter_shift_gradient", parameter_shift_gradient)
        object.__setattr__(self, "finite_difference_gradient", finite_difference_gradient)
        object.__setattr__(self, "abs_error", abs_error)
        object.__setattr__(self, "max_abs_error", max_abs_error)
        object.__setattr__(self, "objective_value", objective_value)
        object.__setattr__(self, "value_delta", value_delta)
        object.__setattr__(self, "finite_difference_step", finite_difference_step)
        object.__setattr__(self, "tolerance", tolerance)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready gradient-verification evidence."""
        return {
            "parameters": self.parameters.tolist(),
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "finite_difference_gradient": self.finite_difference_gradient.tolist(),
            "abs_error": self.abs_error.tolist(),
            "max_abs_error": self.max_abs_error,
            "objective_value": self.objective_value,
            "value_delta": self.value_delta,
            "passed": self.passed,
            "method": self.method,
            "finite_difference_step": self.finite_difference_step,
            "tolerance": self.tolerance,
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
            "finite_difference_evaluations": self.finite_difference_evaluations,
            "edges": [list(edge) for edge in self.edges],
            "claim_boundary": self.claim_boundary,
        }


def _as_target_observations(values: ArrayLike) -> FloatArray:
    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("target_observations must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(observations)):
        raise ValueError("target_observations must contain only finite values")
    return observations.astype(np.float64, copy=True)


def _as_initial_array(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim not in (1, 2) or array.size == 0:
        raise ValueError("initial_couplings must be a non-empty vector or square matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("initial_couplings must contain only finite values")
    return array.astype(np.float64, copy=True)


def _validate_matrix(matrix: FloatArray) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("initial coupling matrix must be square")
    if matrix.shape[0] < 2:
        raise ValueError("coupling matrix must contain at least two nodes")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12):
        raise ValueError("initial coupling matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("initial coupling matrix diagonal must be zero")


def _complete_graph_edges(n_nodes: int) -> tuple[Edge, ...]:
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least two")
    return tuple((row, col) for row in range(n_nodes) for col in range(row + 1, n_nodes))


def _normalise_edges(edges: Sequence[Sequence[int]] | None, n_nodes: int) -> tuple[Edge, ...]:
    if edges is None:
        return _complete_graph_edges(n_nodes)
    normalised: list[Edge] = []
    seen: set[Edge] = set()
    for raw_edge in edges:
        if len(raw_edge) != 2:
            raise ValueError("each edge must contain exactly two node indices")
        left = int(raw_edge[0])
        right = int(raw_edge[1])
        if left == right:
            raise ValueError("coupling-learning edges must not contain self edges")
        if left < 0 or right < 0 or left >= n_nodes or right >= n_nodes:
            raise ValueError("coupling-learning edge index is out of bounds")
        edge = (left, right) if left < right else (right, left)
        if edge in seen:
            raise ValueError("coupling-learning edges must be unique")
        seen.add(edge)
        normalised.append(edge)
    if not normalised:
        raise ValueError("edges must contain at least one trainable coupling")
    return tuple(normalised)


def coupling_matrix_from_edge_vector(
    values: ArrayLike,
    *,
    n_nodes: int,
    edges: Sequence[Sequence[int]] | None = None,
) -> FloatArray:
    """Build a symmetric zero-diagonal coupling matrix from edge parameters."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError("values must contain only finite couplings")
    edge_tuple = _normalise_edges(edges, int(n_nodes))
    if vector.size != len(edge_tuple):
        raise ValueError("values length must match the number of trainable edges")
    matrix = np.zeros((int(n_nodes), int(n_nodes)), dtype=np.float64)
    for value, (row, col) in zip(vector, edge_tuple, strict=True):
        matrix[row, col] = float(value)
        matrix[col, row] = float(value)
    return matrix


def _initial_matrix_and_vector(
    initial_couplings: ArrayLike,
    *,
    n_nodes: int | None,
    edges: Sequence[Sequence[int]] | None,
) -> tuple[FloatArray, FloatArray, tuple[Edge, ...]]:
    initial = _as_initial_array(initial_couplings)
    if initial.ndim == 2:
        _validate_matrix(initial)
        node_count = int(initial.shape[0])
        if n_nodes is not None and int(n_nodes) != node_count:
            raise ValueError("n_nodes must match initial coupling matrix shape")
        edge_tuple = _normalise_edges(edges, node_count)
        vector = np.array([initial[row, col] for row, col in edge_tuple], dtype=np.float64)
        return initial.copy(), vector, edge_tuple
    if n_nodes is None:
        raise ValueError("n_nodes is required when initial_couplings is a vector")
    node_count = int(n_nodes)
    edge_tuple = _normalise_edges(edges, node_count)
    matrix = coupling_matrix_from_edge_vector(initial, n_nodes=node_count, edges=edge_tuple)
    return matrix, initial.astype(np.float64, copy=True), edge_tuple


def _evaluate_observation_model(
    model: CouplingObservationModel,
    matrix: FloatArray,
    expected_shape: tuple[int, ...],
) -> FloatArray:
    observations = np.asarray(model(matrix.copy()), dtype=float)
    if observations.shape != expected_shape:
        raise ValueError(
            f"observation_model output shape must match target_observations shape {expected_shape}"
        )
    if not np.all(np.isfinite(observations)):
        raise ValueError("observation_model must return only finite values")
    return observations.astype(np.float64, copy=True)


def learn_couplings_from_observations(
    observation_model: CouplingObservationModel,
    target_observations: ArrayLike,
    initial_couplings: ArrayLike,
    *,
    n_nodes: int | None = None,
    edges: Sequence[Sequence[int]] | None = None,
    backend: str = "statevector",
    rule: ParameterShiftRule | None = None,
    learning_rate: float = 0.1,
    max_steps: int = 100,
    gradient_tolerance: float = 1e-8,
    value_tolerance: float | None = None,
    target_loss_tolerance: float = 1e-8,
    min_loss_decrease: float | None = None,
    allow_hardware: bool = False,
) -> CouplingLearningResult:
    """Learn symmetric coupling parameters from differentiable observations.

    The observation model must be a smooth, parameter-shift-compatible quantum
    expectation or sinusoidal surrogate over the supplied couplings. Arbitrary
    classical regressors are intentionally outside this claim boundary.
    """
    target = _as_target_observations(target_observations)
    initial_matrix, initial_vector, edge_tuple = _initial_matrix_and_vector(
        initial_couplings,
        n_nodes=n_nodes,
        edges=edges,
    )
    node_count = int(initial_matrix.shape[0])
    parameters = [Parameter(f"K_{row}_{col}") for row, col in edge_tuple]

    def objective(values: FloatArray) -> float:
        matrix = coupling_matrix_from_edge_vector(
            values,
            n_nodes=node_count,
            edges=edge_tuple,
        )
        prediction = _evaluate_observation_model(
            observation_model,
            matrix,
            target.shape,
        )
        residuals = prediction - target
        return float(np.mean(residuals * residuals))

    training = parameter_shift_gradient_descent(
        objective,
        initial_vector,
        parameters=parameters,
        rule=rule,
        backend=backend,
        learning_rate=learning_rate,
        max_steps=max_steps,
        gradient_tolerance=gradient_tolerance,
        value_tolerance=value_tolerance,
        allow_hardware=allow_hardware,
    )
    learned_matrix = coupling_matrix_from_edge_vector(
        training.best_params,
        n_nodes=node_count,
        edges=edge_tuple,
    )
    predicted = _evaluate_observation_model(
        observation_model,
        learned_matrix,
        target.shape,
    )
    residuals = predicted - target
    certificate = validate_parameter_shift_training(
        training,
        gradient_tolerance=gradient_tolerance,
        target_value=0.0,
        target_value_tolerance=target_loss_tolerance,
        min_decrease=min_loss_decrease,
    )
    return CouplingLearningResult(
        training=training,
        certificate=certificate,
        initial_coupling_matrix=initial_matrix,
        learned_coupling_matrix=learned_matrix,
        target_observations=target,
        predicted_observations=predicted,
        residuals=residuals,
        edges=edge_tuple,
        backend=training.backend_plan.backend,
        claim_boundary=(
            "parameter-shift-compatible sinusoidal or quantum-expectation "
            "observation models only; not arbitrary classical regression"
        ),
    )


def verify_coupling_parameter_shift_gradient(
    observation_model: CouplingObservationModel,
    target_observations: ArrayLike,
    couplings: ArrayLike,
    *,
    n_nodes: int | None = None,
    edges: Sequence[Sequence[int]] | None = None,
    rule: ParameterShiftRule | None = None,
    finite_difference_step: float = 1.0e-6,
    tolerance: float = 1.0e-5,
) -> CouplingGradientVerificationResult:
    """Verify coupling gradients against central finite differences.

    This diagnostic is intended for small smooth observation models where a
    central finite-difference reference is affordable. It does not certify
    discontinuous, shot-noisy, hardware-only, or arbitrary regression models.
    """
    step_value = _as_finite_scalar("finite_difference_step", finite_difference_step)
    if step_value <= 0.0:
        raise ValueError("finite_difference_step must be finite and positive")
    tolerance_value = _as_finite_scalar("tolerance", tolerance)
    if tolerance_value < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    target = _as_target_observations(target_observations)
    initial_matrix, values, edge_tuple = _initial_matrix_and_vector(
        couplings,
        n_nodes=n_nodes,
        edges=edges,
    )
    node_count = int(initial_matrix.shape[0])
    parameters = [Parameter(f"K_{row}_{col}") for row, col in edge_tuple]

    def objective(edge_values: FloatArray) -> float:
        matrix = coupling_matrix_from_edge_vector(
            edge_values,
            n_nodes=node_count,
            edges=edge_tuple,
        )
        prediction = _evaluate_observation_model(
            observation_model,
            matrix,
            target.shape,
        )
        residuals = prediction - target
        return float(np.mean(residuals * residuals))

    parameter_shift = value_and_parameter_shift_grad(
        objective,
        values,
        parameters=parameters,
        rule=rule,
    )
    finite_difference = value_and_finite_difference_grad(
        objective,
        values,
        parameters=parameters,
        step=step_value,
    )
    delta = parameter_shift.gradient - finite_difference.gradient
    abs_error = np.abs(delta)
    max_abs_error = float(np.max(abs_error)) if abs_error.size else 0.0
    value_delta = float(parameter_shift.value - finite_difference.value)
    return CouplingGradientVerificationResult(
        parameters=values,
        parameter_shift_gradient=parameter_shift.gradient,
        finite_difference_gradient=finite_difference.gradient,
        abs_error=abs_error,
        max_abs_error=max_abs_error,
        objective_value=parameter_shift.value,
        value_delta=value_delta,
        passed=max_abs_error <= tolerance_value,
        method="parameter_shift_vs_central_finite_difference",
        finite_difference_step=step_value,
        tolerance=tolerance_value,
        parameter_shift_evaluations=parameter_shift.evaluations,
        finite_difference_evaluations=finite_difference.evaluations,
        edges=edge_tuple,
        claim_boundary=(
            "small smooth parameter-shift-compatible coupling objectives only; "
            "not discontinuous, shot-noisy, hardware-only, or arbitrary "
            "classical regression models"
        ),
    )


__all__ = [
    "CouplingGradientVerificationResult",
    "CouplingLearningResult",
    "CouplingObservationModel",
    "coupling_matrix_from_edge_vector",
    "learn_couplings_from_observations",
    "verify_coupling_parameter_shift_gradient",
]
