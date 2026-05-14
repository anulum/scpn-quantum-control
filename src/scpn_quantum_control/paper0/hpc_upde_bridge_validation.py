# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 HPC/UPDE bridge validation fixtures
"""Executable simulator fixtures for Paper 0 HPC/UPDE bridge records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .spec_loader import load_hpc_upde_bridge_validation_spec


@dataclass(frozen=True, slots=True)
class HpcHierarchyConfig:
    """Finite predictive-coding hierarchy for downward prediction/upward error."""

    lower_state: np.ndarray | None = None
    higher_state: np.ndarray | None = None
    generative_weights: np.ndarray | None = None
    upward_error_weights: np.ndarray | None = None
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        lower = _validate_vector(
            "lower_state",
            self.lower_state
            if self.lower_state is not None
            else np.array([0.1, 0.4, 0.9], dtype=np.float64),
            min_size=2,
        )
        higher = _validate_vector(
            "higher_state",
            self.higher_state
            if self.higher_state is not None
            else np.array([0.2, 0.6], dtype=np.float64),
            min_size=2,
        )
        generative = _validate_matrix(
            "generative_weights",
            self.generative_weights
            if self.generative_weights is not None
            else np.array([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]], dtype=np.float64),
        )
        upward = _validate_matrix(
            "upward_error_weights",
            self.upward_error_weights
            if self.upward_error_weights is not None
            else np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], dtype=np.float64),
        )
        if generative.shape != (lower.size, higher.size):
            raise ValueError("generative_weights must map higher_state to lower_state")
        if upward.shape != (higher.size, lower.size):
            raise ValueError("upward_error_weights must map prediction error upward")
        object.__setattr__(self, "lower_state", lower)
        object.__setattr__(self, "higher_state", higher)
        object.__setattr__(self, "generative_weights", generative)
        object.__setattr__(self, "upward_error_weights", upward)


@dataclass(frozen=True, slots=True)
class HpcPredictionErrorTerms:
    """Computed predictive-coding residuals for a finite hierarchy."""

    downward_prediction: np.ndarray
    prediction_error: np.ndarray
    upward_error_message: np.ndarray
    upward_state_copy_message: np.ndarray


@dataclass(frozen=True, slots=True)
class PhasePredictionErrorConfig:
    """Finite inter-layer phase-error and precision-weighting fixture."""

    theta_lower: np.ndarray | None = None
    theta_upper: np.ndarray | None = None
    coupling: np.ndarray | None = None
    precision: np.ndarray | None = None
    step_size: float = 0.05
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        lower = _validate_vector(
            "theta_lower",
            self.theta_lower
            if self.theta_lower is not None
            else np.array([0.2, -0.1, 0.45], dtype=np.float64),
            min_size=2,
        )
        upper = _validate_vector(
            "theta_upper",
            self.theta_upper
            if self.theta_upper is not None
            else np.array([0.5, 0.25, 0.1], dtype=np.float64),
            min_size=lower.size,
        )
        coupling = _validate_nonnegative_vector(
            "coupling",
            self.coupling
            if self.coupling is not None
            else np.array([0.7, 0.4, 0.8], dtype=np.float64),
        )
        precision = _validate_nonnegative_vector(
            "precision",
            self.precision
            if self.precision is not None
            else np.array([1.5, 0.6, 2.0], dtype=np.float64),
        )
        if (
            lower.shape != upper.shape
            or lower.shape != coupling.shape
            or lower.shape != precision.shape
        ):
            raise ValueError("theta_lower, theta_upper, coupling, and precision must share shape")
        if not np.isfinite(self.step_size) or self.step_size <= 0.0 or self.step_size >= 0.25:
            raise ValueError("step_size must be finite and in the open interval (0, 0.25)")
        object.__setattr__(self, "theta_lower", lower)
        object.__setattr__(self, "theta_upper", upper)
        object.__setattr__(self, "coupling", coupling)
        object.__setattr__(self, "precision", precision)


@dataclass(frozen=True, slots=True)
class PhasePredictionErrorTerms:
    """Computed finite phase residuals and one-step dissipative update."""

    phase_residual: np.ndarray
    weighted_residual: np.ndarray
    initial_squared_error: float
    updated_squared_error: float
    zero_coupling_squared_error: float


@dataclass(frozen=True, slots=True)
class UpdeGradientBridgeConfig:
    """Finite XY-potential and Kuramoto-gradient bridge settings."""

    theta: np.ndarray | None = None
    coupling: np.ndarray | None = None
    omega: np.ndarray | None = None
    eta: np.ndarray | None = None
    finite_difference_step: float = 1.0e-6
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        theta = _validate_vector(
            "theta",
            self.theta if self.theta is not None else np.array([0.1, 0.35, -0.2]),
            min_size=2,
        )
        coupling = _validate_matrix(
            "coupling",
            self.coupling
            if self.coupling is not None
            else np.array([[0.0, 0.4, 0.2], [0.4, 0.0, 0.3], [0.2, 0.3, 0.0]]),
        )
        omega = _validate_vector(
            "omega",
            self.omega if self.omega is not None else np.array([0.03, -0.02, 0.01]),
            min_size=theta.size,
        )
        eta = _validate_vector(
            "eta",
            self.eta if self.eta is not None else np.array([0.001, -0.002, 0.0]),
            min_size=theta.size,
        )
        if coupling.shape[0] != coupling.shape[1]:
            raise ValueError("coupling must be square")
        if coupling.shape != (theta.size, theta.size):
            raise ValueError("coupling must match theta dimension")
        if omega.shape != theta.shape or eta.shape != theta.shape:
            raise ValueError("omega, eta, and theta must share shape")
        if not np.allclose(coupling, coupling.T, atol=1.0e-12, rtol=0.0):
            raise ValueError("coupling must be symmetric for the undirected XY gradient fixture")
        if np.any(coupling < 0.0):
            raise ValueError("coupling must be non-negative")
        if not np.isfinite(self.finite_difference_step) or self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be finite and positive")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "coupling", coupling)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "eta", eta)


@dataclass(frozen=True, slots=True)
class UpdeGradientBridgeTerms:
    """Computed finite XY-potential gradient and Kuramoto drift."""

    initial_potential: float
    aligned_potential: float
    analytic_gradient: np.ndarray
    finite_difference_gradient: np.ndarray
    kuramoto_drift: np.ndarray
    gradient_bridge_drift: np.ndarray


@dataclass(frozen=True, slots=True)
class HpcBidirectionalFlowValidationResult:
    """Source-anchored HPC bidirectional-flow validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    prediction_error_norm: float
    upward_error_norm: float
    upward_state_copy_residual: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class PhasePredictionErrorValidationResult:
    """Source-anchored UPDE phase-error validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    weighted_residual_norm: float
    squared_error_delta: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class UpdeGradientBridgeValidationResult:
    """Source-anchored UPDE free-energy-gradient validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    initial_potential: float
    aligned_potential: float
    max_gradient_residual: float
    max_drift_residual: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class HpcUpdeBridgeFixtureResult:
    """Combined Paper 0 HPC/UPDE bridge fixture result."""

    hpc: HpcBidirectionalFlowValidationResult
    phase: PhasePredictionErrorValidationResult
    gradient: UpdeGradientBridgeValidationResult


def hpc_prediction_errors(config: HpcHierarchyConfig) -> HpcPredictionErrorTerms:
    """Compute downward prediction and upward error-only message."""
    lower = cast(np.ndarray, config.lower_state)
    higher = cast(np.ndarray, config.higher_state)
    generative = cast(np.ndarray, config.generative_weights)
    upward = cast(np.ndarray, config.upward_error_weights)
    downward_prediction = generative @ higher
    prediction_error = lower - downward_prediction
    return HpcPredictionErrorTerms(
        downward_prediction=downward_prediction,
        prediction_error=prediction_error,
        upward_error_message=upward @ prediction_error,
        upward_state_copy_message=upward @ lower,
    )


def phase_prediction_error_terms(
    config: PhasePredictionErrorConfig,
) -> PhasePredictionErrorTerms:
    """Compute signed phase residuals and a dissipative lower-layer update."""
    lower = cast(np.ndarray, config.theta_lower)
    upper = cast(np.ndarray, config.theta_upper)
    coupling = cast(np.ndarray, config.coupling)
    precision = cast(np.ndarray, config.precision)
    delta = upper - lower
    phase_residual = np.sin(delta)
    weighted_residual = coupling * precision * phase_residual
    updated_lower = lower + config.step_size * weighted_residual
    zero_coupling_lower = lower + config.step_size * np.zeros_like(weighted_residual)
    return PhasePredictionErrorTerms(
        phase_residual=phase_residual,
        weighted_residual=weighted_residual,
        initial_squared_error=float(np.sum(delta * delta)),
        updated_squared_error=float(np.sum((upper - updated_lower) ** 2)),
        zero_coupling_squared_error=float(np.sum((upper - zero_coupling_lower) ** 2)),
    )


def kuramoto_gradient_bridge_terms(
    config: UpdeGradientBridgeConfig,
) -> UpdeGradientBridgeTerms:
    """Compute XY potential, analytic gradient, finite difference, and drift."""
    theta = cast(np.ndarray, config.theta)
    coupling = cast(np.ndarray, config.coupling)
    omega = cast(np.ndarray, config.omega)
    eta = cast(np.ndarray, config.eta)
    analytic_gradient = xy_potential_gradient(theta, coupling)
    finite_difference_gradient = finite_difference_xy_gradient(
        theta, coupling, step=config.finite_difference_step
    )
    kuramoto_drift = (
        omega + np.sum(coupling * np.sin(theta[None, :] - theta[:, None]), axis=1) + eta
    )
    return UpdeGradientBridgeTerms(
        initial_potential=xy_potential(theta, coupling),
        aligned_potential=xy_potential(np.zeros_like(theta), coupling),
        analytic_gradient=analytic_gradient,
        finite_difference_gradient=finite_difference_gradient,
        kuramoto_drift=kuramoto_drift,
        gradient_bridge_drift=omega - analytic_gradient + eta,
    )


def validate_hpc_bidirectional_flow_fixture(
    config: HpcHierarchyConfig | None = None,
) -> HpcBidirectionalFlowValidationResult:
    """Run the source-anchored HPC bidirectional-flow fixture."""
    cfg = config or HpcHierarchyConfig()
    spec = load_hpc_upde_bridge_validation_spec(
        "computational.hpc_bidirectional_flow",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    terms = hpc_prediction_errors(cfg)
    disconnected = cast(np.ndarray, cfg.upward_error_weights) * 0.0
    controls = {
        "disconnected_hierarchy_residual_norm": float(
            np.linalg.norm(disconnected @ terms.prediction_error)
        ),
        "state_copy_rejection_residual": float(
            np.linalg.norm(terms.upward_state_copy_message - terms.upward_error_message)
        ),
        "layer_order_reversal_rejection_label": _layer_order_reversal_rejection_label(),
    }
    return HpcBidirectionalFlowValidationResult(
        spec_key="computational.hpc_bidirectional_flow",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        prediction_error_norm=float(np.linalg.norm(terms.prediction_error)),
        upward_error_norm=float(np.linalg.norm(terms.upward_error_message)),
        upward_state_copy_residual=controls["state_copy_rejection_residual"],
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "simulator_only_mechanism_evidence": True,
                "claim_boundary": "finite_predictive_coding_flow_not_biological_confirmation",
            }
        ),
    )


def validate_upde_phase_prediction_error_fixture(
    config: PhasePredictionErrorConfig | None = None,
) -> PhasePredictionErrorValidationResult:
    """Run the source-anchored UPDE phase prediction-error fixture."""
    cfg = config or PhasePredictionErrorConfig()
    spec = load_hpc_upde_bridge_validation_spec(
        "computational.upde_phase_prediction_error",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    terms = phase_prediction_error_terms(cfg)
    zero_delta = terms.zero_coupling_squared_error - terms.initial_squared_error
    controls = {
        "zero_coupling_error_delta_abs": abs(zero_delta),
        "negative_precision_rejection_label": _negative_precision_rejection_label(),
        "uncoupled_layer_residual_abs": float(
            abs(
                phase_prediction_error_terms(
                    PhasePredictionErrorConfig(coupling=np.zeros(3))
                ).weighted_residual[0]
            )
        ),
    }
    return PhasePredictionErrorValidationResult(
        spec_key="computational.upde_phase_prediction_error",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        weighted_residual_norm=float(np.linalg.norm(terms.weighted_residual)),
        squared_error_delta=terms.updated_squared_error - terms.initial_squared_error,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "simulator_only_mechanism_evidence": True,
                "claim_boundary": "finite_phase_residual_fixture_not_neural_confirmation",
            }
        ),
    )


def validate_upde_free_energy_gradient_bridge_fixture(
    config: UpdeGradientBridgeConfig | None = None,
) -> UpdeGradientBridgeValidationResult:
    """Run the source-anchored UPDE free-energy-gradient bridge fixture."""
    cfg = config or UpdeGradientBridgeConfig()
    spec = load_hpc_upde_bridge_validation_spec(
        "computational.upde_free_energy_gradient_bridge",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    terms = kuramoto_gradient_bridge_terms(cfg)
    controls = {
        "wrong_sign_potential_delta": _wrong_sign_potential_delta(cfg),
        "asymmetric_coupling_rejection_label": _asymmetric_coupling_rejection_label(),
        "non_finite_phase_rejection_label": _non_finite_phase_rejection_label(),
    }
    return UpdeGradientBridgeValidationResult(
        spec_key="computational.upde_free_energy_gradient_bridge",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        initial_potential=terms.initial_potential,
        aligned_potential=terms.aligned_potential,
        max_gradient_residual=float(
            np.max(np.abs(terms.analytic_gradient - terms.finite_difference_gradient))
        ),
        max_drift_residual=float(
            np.max(np.abs(terms.kuramoto_drift - terms.gradient_bridge_drift))
        ),
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "simulator_only_mechanism_evidence": True,
                "claim_boundary": "finite_xy_gradient_identity_not_universal_proof",
            }
        ),
    )


def validate_hpc_upde_bridge_fixture() -> HpcUpdeBridgeFixtureResult:
    """Run all source-anchored HPC/UPDE bridge fixtures."""
    return HpcUpdeBridgeFixtureResult(
        hpc=validate_hpc_bidirectional_flow_fixture(),
        phase=validate_upde_phase_prediction_error_fixture(),
        gradient=validate_upde_free_energy_gradient_bridge_fixture(),
    )


def xy_potential(theta: np.ndarray, coupling: np.ndarray) -> float:
    """Return undirected XY negative-cosine potential over i < j pairs."""
    total = 0.0
    for i in range(theta.size):
        for j in range(i + 1, theta.size):
            total -= float(coupling[i, j] * np.cos(theta[j] - theta[i]))
    return total


def xy_potential_gradient(theta: np.ndarray, coupling: np.ndarray) -> np.ndarray:
    """Return analytic gradient of the undirected XY negative-cosine potential."""
    gradient = -np.sum(coupling * np.sin(theta[None, :] - theta[:, None]), axis=1)
    return cast(np.ndarray, gradient)


def finite_difference_xy_gradient(
    theta: np.ndarray, coupling: np.ndarray, *, step: float
) -> np.ndarray:
    """Return central finite-difference gradient of the XY potential."""
    gradient = np.zeros_like(theta, dtype=np.float64)
    for idx in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[idx] += step
        minus[idx] -= step
        gradient[idx] = (xy_potential(plus, coupling) - xy_potential(minus, coupling)) / (
            2.0 * step
        )
    return gradient


def _layer_order_reversal_rejection_label() -> float:
    try:
        HpcHierarchyConfig(
            lower_state=np.array([0.1, 0.2]),
            higher_state=np.array([0.3, 0.4, 0.5]),
            generative_weights=np.ones((2, 2), dtype=np.float64),
        )
    except ValueError as exc:
        return float("higher_state" in str(exc) or "lower_state" in str(exc))
    return 0.0


def _negative_precision_rejection_label() -> float:
    try:
        PhasePredictionErrorConfig(precision=np.array([1.0, -0.1, 1.0], dtype=np.float64))
    except ValueError as exc:
        return float("non-negative" in str(exc))
    return 0.0


def _wrong_sign_potential_delta(config: UpdeGradientBridgeConfig) -> float:
    theta = cast(np.ndarray, config.theta)
    coupling = cast(np.ndarray, config.coupling)
    initial_wrong_sign = xy_potential(theta, -coupling)
    aligned_wrong_sign = xy_potential(np.zeros_like(theta), -coupling)
    return aligned_wrong_sign - initial_wrong_sign


def _asymmetric_coupling_rejection_label() -> float:
    try:
        UpdeGradientBridgeConfig(
            coupling=np.array([[0.0, 0.2, 0.0], [0.1, 0.0, 0.3], [0.0, 0.3, 0.0]])
        )
    except ValueError as exc:
        return float("symmetric" in str(exc))
    return 0.0


def _non_finite_phase_rejection_label() -> float:
    try:
        UpdeGradientBridgeConfig(theta=np.array([0.0, np.inf, 0.1]))
    except ValueError as exc:
        return float("finite" in str(exc))
    return 0.0


def _validate_vector(name: str, values: np.ndarray, *, min_size: int) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.ndim != 1 or arr.size < min_size:
        raise ValueError(
            f"{name} must be a one-dimensional vector with at least {min_size} entries"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_nonnegative_vector(name: str, values: np.ndarray) -> np.ndarray:
    arr = _validate_vector(name, values, min_size=2)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def _validate_matrix(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


__all__ = [
    "HpcBidirectionalFlowValidationResult",
    "HpcHierarchyConfig",
    "HpcPredictionErrorTerms",
    "HpcUpdeBridgeFixtureResult",
    "PhasePredictionErrorConfig",
    "PhasePredictionErrorTerms",
    "PhasePredictionErrorValidationResult",
    "UpdeGradientBridgeConfig",
    "UpdeGradientBridgeTerms",
    "UpdeGradientBridgeValidationResult",
    "finite_difference_xy_gradient",
    "hpc_prediction_errors",
    "kuramoto_gradient_bridge_terms",
    "phase_prediction_error_terms",
    "validate_hpc_bidirectional_flow_fixture",
    "validate_hpc_upde_bridge_fixture",
    "validate_upde_free_energy_gradient_bridge_fixture",
    "validate_upde_phase_prediction_error_fixture",
    "xy_potential",
    "xy_potential_gradient",
]
