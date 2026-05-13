# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 UPDE validation fixtures
"""Executable simulator fixtures for Paper 0 UPDE validation specs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from ..kuramoto_core import build_kuramoto_problem
from .spec_loader import load_upde_validation_spec
from .upde_adaptive_coupling_validation import (
    AdaptiveCouplingConfig,
    AdaptiveCouplingRates,
    AdaptiveCouplingStep,
    AdaptiveCouplingValidationResult,
    adaptive_coupling_rates,
    apply_adaptive_coupling_step,
    validate_upde_adaptive_coupling_fixture,
)
from .upde_field_validation import (
    FieldCouplingConfig,
    FieldValidationResult,
    field_alignment_projection,
    field_coupling_term,
    validate_upde_field_fixture,
)
from .upde_interlayer_validation import (
    InterlayerCouplingConfig,
    InterlayerCouplingTerms,
    InterlayerValidationResult,
    circular_mean_phase,
    interlayer_coupling_terms,
    validate_upde_interlayer_fixture,
)
from .upde_natural_gradient_validation import (
    NaturalGradientConfig,
    NaturalGradientValidationResult,
    finite_difference_quadratic_gradient,
    natural_gradient_flow,
    quadratic_free_energy,
    validate_upde_natural_gradient_fixture,
)


@dataclass(frozen=True, slots=True)
class BasePhaseValidationConfig:
    """Numerical tolerances for the Paper 0 base-phase fixture."""

    finite_difference_step: float = 1.0e-6
    gradient_tolerance: float = 1.0e-7
    onset_dt: float = 0.05
    weak_coupling_scale: float = 0.05
    strong_coupling_scale: float = 1.0
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.finite_difference_step) or self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be finite and positive")
        if not np.isfinite(self.gradient_tolerance) or self.gradient_tolerance <= 0.0:
            raise ValueError("gradient_tolerance must be finite and positive")
        if not np.isfinite(self.onset_dt) or self.onset_dt <= 0.0:
            raise ValueError("onset_dt must be finite and positive")
        if not np.isfinite(self.weak_coupling_scale) or self.weak_coupling_scale < 0.0:
            raise ValueError("weak_coupling_scale must be finite and non-negative")
        if not np.isfinite(self.strong_coupling_scale) or self.strong_coupling_scale < 0.0:
            raise ValueError("strong_coupling_scale must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BasePhaseValidationResult:
    """Result of the Paper 0 base-phase simulator fixture."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    gradient_error_linf: float
    analytic_drift: tuple[float, ...]
    finite_difference_drift: tuple[float, ...]
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


def negative_cosine_potential(theta: np.ndarray, K_nm: np.ndarray, omega: np.ndarray) -> float:
    """Return the source Kuramoto potential whose negative gradient is UPDE drift."""
    K_arr, omega_arr, theta_arr = _validate_base_phase_inputs(K_nm, omega, theta)
    phase_differences = theta_arr[None, :] - theta_arr[:, None]
    coupling_energy = -0.5 * float(np.sum(K_arr * np.cos(phase_differences)))
    frequency_energy = -float(np.dot(omega_arr, theta_arr))
    return frequency_energy + coupling_energy


def kuramoto_phase_drift(theta: np.ndarray, K_nm: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Return ``omega_i + sum_j K_ij sin(theta_j - theta_i)``."""
    K_arr, omega_arr, theta_arr = _validate_base_phase_inputs(K_nm, omega, theta)
    phase_differences = theta_arr[None, :] - theta_arr[:, None]
    return cast(np.ndarray, omega_arr + np.sum(K_arr * np.sin(phase_differences), axis=1))


def finite_difference_negative_gradient(
    potential: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    theta: np.ndarray,
    K_nm: np.ndarray,
    omega: np.ndarray,
    *,
    step: float = 1.0e-6,
) -> np.ndarray:
    """Estimate the negative gradient of a scalar potential by central difference."""
    K_arr, omega_arr, theta_arr = _validate_base_phase_inputs(K_nm, omega, theta)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive")

    gradient = np.empty_like(theta_arr)
    for index in range(theta_arr.size):
        direction = np.zeros_like(theta_arr)
        direction[index] = step
        plus = float(potential(theta_arr + direction, K_arr, omega_arr))
        minus = float(potential(theta_arr - direction, K_arr, omega_arr))
        gradient[index] = (plus - minus) / (2.0 * step)
    return -gradient


def validate_upde_base_phase_fixture(
    K_nm: np.ndarray,
    omega: np.ndarray,
    theta: np.ndarray,
    *,
    config: BasePhaseValidationConfig | None = None,
) -> BasePhaseValidationResult:
    """Run the source-anchored base UPDE/Kuramoto simulator fixture."""
    cfg = config or BasePhaseValidationConfig()
    K_arr, omega_arr, theta_arr = _validate_base_phase_inputs(K_nm, omega, theta)
    spec = load_upde_validation_spec("upde.base_phase", spec_bundle_path=cfg.spec_bundle_path)
    problem = build_kuramoto_problem(
        K_arr,
        omega_arr,
        metadata={
            "paper0_spec_key": "upde.base_phase",
            "paper0_validation_protocol": str(spec["validation_protocol"]),
            "hardware_status": str(spec["hardware_status"]),
        },
    )

    analytic = kuramoto_phase_drift(theta_arr, problem.K_nm, problem.omega)
    finite_difference = finite_difference_negative_gradient(
        negative_cosine_potential,
        theta_arr,
        problem.K_nm,
        problem.omega,
        step=cfg.finite_difference_step,
    )
    gradient_error = float(np.max(np.abs(analytic - finite_difference)))
    if gradient_error > cfg.gradient_tolerance:
        raise ValueError(
            "base-phase gradient check exceeded tolerance: "
            f"{gradient_error:.3e} > {cfg.gradient_tolerance:.3e}"
        )

    null_controls = _base_phase_null_controls(problem.K_nm, problem.omega, theta_arr, cfg)
    metadata = problem.to_metadata()
    metadata.update(
        {
            "paper0_spec_key": "upde.base_phase",
            "paper0_validation_protocol": str(spec["validation_protocol"]),
            "hardware_status": str(spec["hardware_status"]),
        }
    )
    return BasePhaseValidationResult(
        spec_key="upde.base_phase",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        gradient_error_linf=gradient_error,
        analytic_drift=tuple(float(item) for item in analytic),
        finite_difference_drift=tuple(float(item) for item in finite_difference),
        null_controls=MappingProxyType(null_controls),
        problem_metadata=MappingProxyType(metadata),
    )


def _validate_base_phase_inputs(
    K_nm: np.ndarray,
    omega: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K_arr = np.array(K_nm, dtype=np.float64, copy=True)
    omega_arr = np.array(omega, dtype=np.float64, copy=True)
    theta_arr = np.array(theta, dtype=np.float64, copy=True)
    if K_arr.ndim != 2 or K_arr.shape[0] != K_arr.shape[1]:
        raise ValueError(f"K_nm must be a square matrix, got shape {K_arr.shape}")
    n_oscillators = K_arr.shape[0]
    if omega_arr.shape != (n_oscillators,):
        raise ValueError(f"omega must have shape ({n_oscillators},), got {omega_arr.shape}")
    if theta_arr.shape != (n_oscillators,):
        raise ValueError(f"theta must have shape ({n_oscillators},), got {theta_arr.shape}")
    if not np.all(np.isfinite(K_arr)):
        raise ValueError("K_nm must contain only finite values")
    if not np.all(np.isfinite(omega_arr)):
        raise ValueError("omega must contain only finite values")
    if not np.all(np.isfinite(theta_arr)):
        raise ValueError("theta must contain only finite values")
    if not np.allclose(K_arr, K_arr.T, atol=1e-12, rtol=1e-12):
        raise ValueError("K_nm must be symmetric for the Paper 0 base-phase gradient check")
    np.fill_diagonal(K_arr, 0.0)
    return K_arr, omega_arr, theta_arr


def _base_phase_null_controls(
    K_nm: np.ndarray,
    omega: np.ndarray,
    theta: np.ndarray,
    config: BasePhaseValidationConfig,
) -> dict[str, float]:
    zero = np.zeros_like(K_nm)
    zero_drift = kuramoto_phase_drift(theta, zero, omega)
    sign_flip_drift = kuramoto_phase_drift(theta, -K_nm, omega)
    shuffled = _deterministic_shuffled_topology(K_nm)
    shuffled_drift = kuramoto_phase_drift(theta, shuffled, omega)

    weak_step = theta + config.onset_dt * kuramoto_phase_drift(
        theta,
        config.weak_coupling_scale * K_nm,
        omega,
    )
    strong_step = theta + config.onset_dt * kuramoto_phase_drift(
        theta,
        config.strong_coupling_scale * K_nm,
        omega,
    )
    return {
        "zero_coupling_drift_linf": float(np.max(np.abs(zero_drift - omega))),
        "sign_flip_response_l2": float(np.linalg.norm(sign_flip_drift - zero_drift)),
        "shuffled_topology_response_l2": float(np.linalg.norm(shuffled_drift - zero_drift)),
        "off_onset_order_parameter_delta": abs(
            _phase_order_parameter(strong_step) - _phase_order_parameter(weak_step)
        ),
    }


def _deterministic_shuffled_topology(K_nm: np.ndarray) -> np.ndarray:
    if K_nm.shape[0] < 3:
        return cast(np.ndarray, K_nm.copy())
    permutation = np.roll(np.arange(K_nm.shape[0]), 1)
    shuffled = K_nm[np.ix_(permutation, permutation)].copy()
    np.fill_diagonal(shuffled, 0.0)
    return cast(np.ndarray, shuffled)


def _phase_order_parameter(theta: np.ndarray) -> float:
    return float(abs(np.mean(np.exp(1j * theta))))


__all__ = [
    "BasePhaseValidationConfig",
    "BasePhaseValidationResult",
    "AdaptiveCouplingConfig",
    "AdaptiveCouplingRates",
    "AdaptiveCouplingStep",
    "AdaptiveCouplingValidationResult",
    "FieldCouplingConfig",
    "FieldValidationResult",
    "InterlayerCouplingConfig",
    "InterlayerCouplingTerms",
    "InterlayerValidationResult",
    "NaturalGradientConfig",
    "NaturalGradientValidationResult",
    "adaptive_coupling_rates",
    "apply_adaptive_coupling_step",
    "circular_mean_phase",
    "field_alignment_projection",
    "field_coupling_term",
    "finite_difference_negative_gradient",
    "finite_difference_quadratic_gradient",
    "interlayer_coupling_terms",
    "kuramoto_phase_drift",
    "load_upde_validation_spec",
    "natural_gradient_flow",
    "negative_cosine_potential",
    "quadratic_free_energy",
    "validate_upde_adaptive_coupling_fixture",
    "validate_upde_base_phase_fixture",
    "validate_upde_field_fixture",
    "validate_upde_interlayer_fixture",
    "validate_upde_natural_gradient_fixture",
]
