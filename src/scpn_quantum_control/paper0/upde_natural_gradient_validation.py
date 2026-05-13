# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 UPDE natural-gradient validation
"""Executable simulator fixture for the Paper 0 FIM natural-gradient term."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import numpy as np

from .spec_loader import load_upde_validation_spec


@dataclass(frozen=True, slots=True)
class NaturalGradientConfig:
    """Configuration for the Paper 0 natural-gradient fixture."""

    eta_L: float = 0.4
    finite_difference_step: float = 1.0e-6
    regularization: float = 1.0e-6

    def __post_init__(self) -> None:
        if not np.isfinite(self.eta_L) or self.eta_L <= 0.0:
            raise ValueError("eta_L must be finite and positive")
        if not np.isfinite(self.finite_difference_step) or self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be finite and positive")
        if not np.isfinite(self.regularization) or self.regularization <= 0.0:
            raise ValueError("regularization must be finite and positive")


@dataclass(frozen=True, slots=True)
class NaturalGradientValidationResult:
    """Result of the Paper 0 natural-gradient fixture."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    free_energy: float
    gradient_error_linf: float
    metric_condition_number: float
    natural_gradient_drift: tuple[float, ...]
    euclidean_gradient_drift: tuple[float, ...]
    null_controls: MappingProxyType[str, float]


def quadratic_free_energy(
    theta: np.ndarray,
    target: np.ndarray,
    precision_matrix: np.ndarray,
) -> float:
    """Return a differentiable quadratic free-energy fixture."""
    theta_arr, target_arr, precision = _validate_quadratic_inputs(theta, target, precision_matrix)
    delta = theta_arr - target_arr
    return 0.5 * float(delta @ precision @ delta)


def finite_difference_quadratic_gradient(
    theta: np.ndarray,
    target: np.ndarray,
    precision_matrix: np.ndarray,
    *,
    step: float = 1.0e-6,
) -> np.ndarray:
    """Estimate the gradient of the quadratic fixture by central difference."""
    theta_arr, target_arr, precision = _validate_quadratic_inputs(theta, target, precision_matrix)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive")
    gradient = np.empty_like(theta_arr)
    for index in range(theta_arr.size):
        direction = np.zeros_like(theta_arr)
        direction[index] = step
        plus = quadratic_free_energy(theta_arr + direction, target_arr, precision)
        minus = quadratic_free_energy(theta_arr - direction, target_arr, precision)
        gradient[index] = (plus - minus) / (2.0 * step)
    return gradient


def natural_gradient_flow(
    free_energy_gradient: np.ndarray,
    fisher_metric: np.ndarray,
    *,
    config: NaturalGradientConfig | None = None,
) -> np.ndarray:
    """Return ``-eta_L * g_F^{-1} grad F`` for a positive-definite FIM."""
    cfg = config or NaturalGradientConfig()
    gradient = _finite_vector(free_energy_gradient, "free_energy_gradient")
    metric = _finite_square_matrix(fisher_metric, "fisher_metric", gradient.size)
    _require_positive_definite(metric, "fisher_metric")
    return -cfg.eta_L * np.linalg.solve(metric, gradient)


def validate_upde_natural_gradient_fixture(
    theta: np.ndarray,
    target: np.ndarray,
    precision_matrix: np.ndarray,
    fisher_metric: np.ndarray,
    *,
    config: NaturalGradientConfig | None = None,
) -> NaturalGradientValidationResult:
    """Run the source-anchored natural-gradient UPDE simulator fixture."""
    cfg = config or NaturalGradientConfig()
    theta_arr, target_arr, precision = _validate_quadratic_inputs(
        theta,
        target,
        precision_matrix,
    )
    metric = _finite_square_matrix(fisher_metric, "fisher_metric", theta_arr.size)
    _require_positive_definite(metric, "fisher_metric")
    spec = load_upde_validation_spec("upde.natural_gradient")
    analytic_gradient = precision @ (theta_arr - target_arr)
    finite_difference_gradient = finite_difference_quadratic_gradient(
        theta_arr,
        target_arr,
        precision,
        step=cfg.finite_difference_step,
    )
    gradient_error = float(np.max(np.abs(analytic_gradient - finite_difference_gradient)))
    natural_drift = natural_gradient_flow(analytic_gradient, metric, config=cfg)
    euclidean_drift = -cfg.eta_L * analytic_gradient
    identity_drift = natural_gradient_flow(analytic_gradient, np.eye(theta_arr.size), config=cfg)
    singular_metric = metric.copy()
    singular_metric[-1, :] = 0.0
    singular_metric[:, -1] = 0.0
    regularised_metric = singular_metric + cfg.regularization * np.eye(theta_arr.size)
    regularised_drift = -cfg.eta_L * np.linalg.solve(regularised_metric, analytic_gradient)
    null_controls = {
        "euclidean_vs_natural_l2": float(np.linalg.norm(euclidean_drift - natural_drift)),
        "identity_fim_matches_euclidean_linf": float(
            np.max(np.abs(identity_drift - euclidean_drift))
        ),
        "regularised_singular_metric_linf": float(np.max(np.abs(regularised_drift))),
    }
    return NaturalGradientValidationResult(
        spec_key="upde.natural_gradient",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        free_energy=quadratic_free_energy(theta_arr, target_arr, precision),
        gradient_error_linf=gradient_error,
        metric_condition_number=float(np.linalg.cond(metric)),
        natural_gradient_drift=tuple(float(item) for item in natural_drift),
        euclidean_gradient_drift=tuple(float(item) for item in euclidean_drift),
        null_controls=MappingProxyType(null_controls),
    )


def _validate_quadratic_inputs(
    theta: np.ndarray,
    target: np.ndarray,
    precision_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_arr = _finite_vector(theta, "theta")
    target_arr = _finite_vector(target, "target")
    if target_arr.shape != theta_arr.shape:
        raise ValueError(f"target must have shape {theta_arr.shape}, got {target_arr.shape}")
    precision = _finite_square_matrix(precision_matrix, "precision_matrix", theta_arr.size)
    _require_positive_definite(precision, "precision_matrix")
    return theta_arr, target_arr, precision


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(np.ndarray, arr.copy())


def _finite_square_matrix(values: np.ndarray, name: str, dimension: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension}), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(arr, arr.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return cast(np.ndarray, arr.copy())


def _require_positive_definite(matrix: np.ndarray, name: str) -> None:
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc


__all__ = [
    "NaturalGradientConfig",
    "NaturalGradientValidationResult",
    "finite_difference_quadratic_gradient",
    "natural_gradient_flow",
    "quadratic_free_energy",
    "validate_upde_natural_gradient_fixture",
]
