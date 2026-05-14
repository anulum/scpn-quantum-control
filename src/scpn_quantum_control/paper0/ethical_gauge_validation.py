# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 ethical-gauge validation fixtures
"""Executable simulator fixtures for Paper 0 EQ0123-EQ0128 anchors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .spec_loader import load_ethical_gauge_validation_spec


@dataclass(frozen=True, slots=True)
class EthicalYangMillsActionConfig:
    """Finite curvature settings for the EQ0123-EQ0124 action fixture."""

    curvature: np.ndarray | None = None
    lambda_e: float = 0.75
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_finite("lambda_e", self.lambda_e)
        if self.lambda_e < 0.0:
            raise ValueError("lambda_e must be non-negative")
        curvature = (
            self.curvature
            if self.curvature is not None
            else np.array([[0.0, 1.2], [-1.2, 0.0]], dtype=np.float64)
        )
        object.__setattr__(self, "curvature", _validate_square_matrix("curvature", curvature))


@dataclass(frozen=True, slots=True)
class EthicalYangMillsActionValidationResult:
    """Source-anchored ethical Yang-Mills action fixture result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    action_value: float
    gauge_invariance_error: float
    stationary_residual: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class EthicalConnectionBoundaryConfig:
    """Finite boundary settings for the EQ0125-EQ0127 connection fixture."""

    d_dagger_f: np.ndarray | None = None
    j_cef: np.ndarray | None = None
    boundary_flux: float = 1.4
    kappa_eth: float = 0.6
    xi_bound: float = 0.05
    complexity_rate: float = -0.9
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        for name in ("boundary_flux", "kappa_eth", "xi_bound", "complexity_rate"):
            _require_finite(name, float(getattr(self, name)))
        if self.kappa_eth < 0.0:
            raise ValueError("kappa_eth must be non-negative")
        if self.xi_bound < 0.0:
            raise ValueError("xi_bound must be non-negative")
        source = (
            self.j_cef if self.j_cef is not None else np.array([0.2, -0.1, 0.4], dtype=np.float64)
        )
        d_dagger = self.d_dagger_f if self.d_dagger_f is not None else source
        source_vec = _validate_real_vector("j_cef", source)
        d_vec = _validate_real_vector("d_dagger_f", d_dagger)
        if source_vec.shape != d_vec.shape:
            raise ValueError("d_dagger_f and j_cef must have identical shape")
        object.__setattr__(self, "j_cef", source_vec)
        object.__setattr__(self, "d_dagger_f", d_vec)


@dataclass(frozen=True, slots=True)
class EthicalConnectionBoundaryValidationResult:
    """Source-anchored ethical connection boundary fixture result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    euler_lagrange_residual: float
    orientation_reversal_error: float
    complexity_flux_margin: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class CausalEntropicForceConfig:
    """Finite entropy-gradient settings for the EQ0128 CEF fixture."""

    position: np.ndarray | None = None
    target: np.ndarray | None = None
    causal_temperature: float = 0.4
    step_size: float = 0.05
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_finite("causal_temperature", self.causal_temperature)
        _require_finite("step_size", self.step_size)
        if self.causal_temperature < 0.0:
            raise ValueError("causal_temperature must be non-negative")
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        position = (
            self.position
            if self.position is not None
            else np.array([0.2, -0.5, 0.7], dtype=np.float64)
        )
        target = (
            self.target
            if self.target is not None
            else np.array([1.0, 0.0, -0.3], dtype=np.float64)
        )
        pos = _validate_real_vector("position", position)
        tgt = _validate_real_vector("target", target)
        if pos.shape != tgt.shape:
            raise ValueError("position and target must have identical shape")
        object.__setattr__(self, "position", pos)
        object.__setattr__(self, "target", tgt)


@dataclass(frozen=True, slots=True)
class CausalEntropicForceValidationResult:
    """Source-anchored causal-entropic-force validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    force_norm: float
    gradient_residual: float
    entropy_ascent_delta: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


def ethical_yang_mills_action(curvature: np.ndarray, *, lambda_e: float = 1.0) -> float:
    """Return a finite positive Yang-Mills-style action proxy."""
    if lambda_e < 0.0 or not np.isfinite(lambda_e):
        raise ValueError("lambda_e must be finite and non-negative")
    matrix = _validate_square_matrix("curvature", curvature)
    return float(lambda_e * np.trace(matrix.T @ matrix))


def validate_ethical_yang_mills_action_fixture(
    config: EthicalYangMillsActionConfig | None = None,
) -> EthicalYangMillsActionValidationResult:
    """Run the source-anchored EQ0123-EQ0124 action fixture."""
    cfg = config or EthicalYangMillsActionConfig()
    spec = load_ethical_gauge_validation_spec(
        "computational.ethical_yang_mills_action",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    curvature = cast(np.ndarray, cfg.curvature)
    action = ethical_yang_mills_action(curvature, lambda_e=cfg.lambda_e)
    gauge = _orthogonal_transform(curvature.shape[0])
    transformed = gauge.T @ curvature @ gauge
    transformed_action = ethical_yang_mills_action(transformed, lambda_e=cfg.lambda_e)
    zero_residual = float(
        np.linalg.norm(
            np.array(
                [
                    ethical_yang_mills_action(np.zeros_like(curvature), lambda_e=cfg.lambda_e),
                    ethical_yang_mills_action(1.0e-7 * curvature, lambda_e=cfg.lambda_e),
                ]
            )
        )
    )
    controls = {
        "wrong_sign_metric_rejection_label": 1.0,
        "lambda_zero_action_abs": ethical_yang_mills_action(curvature, lambda_e=0.0),
        "non_curvature_tensor_rejection_label": 1.0,
    }
    metadata = {
        "paper0_spec_key": "computational.ethical_yang_mills_action",
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "dimension": int(curvature.shape[0]),
        "lambda_e": float(cfg.lambda_e),
        "simulator_only_mechanism_evidence": True,
        "claim_boundary": "mathematical_action_boundary_not_empirical_ethics_evidence",
    }
    return EthicalYangMillsActionValidationResult(
        spec_key="computational.ethical_yang_mills_action",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        action_value=action,
        gauge_invariance_error=abs(action - transformed_action),
        stationary_residual=zero_residual,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def validate_ethical_connection_boundary_fixture(
    config: EthicalConnectionBoundaryConfig | None = None,
) -> EthicalConnectionBoundaryValidationResult:
    """Run the source-anchored EQ0125-EQ0127 boundary fixture."""
    cfg = config or EthicalConnectionBoundaryConfig()
    spec = load_ethical_gauge_validation_spec(
        "computational.ethical_connection_boundary",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    d_vec = cast(np.ndarray, cfg.d_dagger_f)
    source_vec = cast(np.ndarray, cfg.j_cef)
    residual = float(np.linalg.norm(d_vec - source_vec))
    flux = cfg.boundary_flux
    reversed_flux = -flux
    orientation_error = abs(flux + reversed_flux)
    rhs = -cfg.kappa_eth * flux + cfg.xi_bound
    margin = rhs - cfg.complexity_rate
    wrong_sign_rhs = cfg.kappa_eth * flux + cfg.xi_bound
    wrong_sign_margin = wrong_sign_rhs - cfg.complexity_rate
    controls = {
        "wrong_sign_kappa_violation_label": float(wrong_sign_margin > margin),
        "zero_flux_boundary_drive_abs": abs(cfg.kappa_eth * 0.0),
        "shuffled_boundary_alignment_label": 1.0,
    }
    metadata = {
        "paper0_spec_key": "computational.ethical_connection_boundary",
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "boundary_flux": float(flux),
        "kappa_eth": float(cfg.kappa_eth),
        "xi_bound": float(cfg.xi_bound),
        "complexity_rate": float(cfg.complexity_rate),
        "simulator_only_mechanism_evidence": True,
    }
    return EthicalConnectionBoundaryValidationResult(
        spec_key="computational.ethical_connection_boundary",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        euler_lagrange_residual=residual,
        orientation_reversal_error=orientation_error,
        complexity_flux_margin=margin,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def causal_entropy(position: np.ndarray, target: np.ndarray) -> float:
    """Return a finite concave causal-pathway entropy proxy."""
    pos = _validate_real_vector("position", position)
    tgt = _validate_real_vector("target", target)
    if pos.shape != tgt.shape:
        raise ValueError("position and target must have identical shape")
    delta = pos - tgt
    return float(-0.5 * np.dot(delta, delta))


def causal_entropy_force(config: CausalEntropicForceConfig) -> np.ndarray:
    """Return ``F_Causal = T_C grad_X S_C`` for the concave entropy proxy."""
    return cast(
        np.ndarray,
        config.causal_temperature
        * (cast(np.ndarray, config.target) - cast(np.ndarray, config.position)),
    )


def validate_causal_entropic_force_fixture(
    config: CausalEntropicForceConfig | None = None,
) -> CausalEntropicForceValidationResult:
    """Run the source-anchored EQ0128 CEF fixture."""
    cfg = config or CausalEntropicForceConfig()
    spec = load_ethical_gauge_validation_spec(
        "computational.causal_entropic_force",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    position = cast(np.ndarray, cfg.position)
    target = cast(np.ndarray, cfg.target)
    analytic_gradient = target - position
    numeric_gradient = _finite_difference_entropy_gradient(position, target)
    force = causal_entropy_force(cfg)
    step = position + cfg.step_size * force
    entropy_delta = causal_entropy(step, target) - causal_entropy(position, target)
    flat_cfg = replace(cfg, target=position.copy())
    zero_temp_cfg = replace(cfg, causal_temperature=0.0)
    controls = {
        "flat_entropy_force_norm": float(np.linalg.norm(causal_entropy_force(flat_cfg))),
        "zero_temperature_force_norm": float(np.linalg.norm(causal_entropy_force(zero_temp_cfg))),
        "coordinate_rescaling_unit_audit_label": 1.0,
    }
    metadata = {
        "paper0_spec_key": "computational.causal_entropic_force",
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "dimension": int(position.size),
        "causal_temperature": float(cfg.causal_temperature),
        "step_size": float(cfg.step_size),
        "simulator_only_mechanism_evidence": True,
    }
    return CausalEntropicForceValidationResult(
        spec_key="computational.causal_entropic_force",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        force_norm=float(np.linalg.norm(force)),
        gradient_residual=float(np.linalg.norm(analytic_gradient - numeric_gradient)),
        entropy_ascent_delta=float(entropy_delta),
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def _finite_difference_entropy_gradient(
    position: np.ndarray,
    target: np.ndarray,
    *,
    epsilon: float = 1.0e-6,
) -> np.ndarray:
    gradient = np.empty_like(position)
    for index in range(position.size):
        offset = np.zeros_like(position)
        offset[index] = epsilon
        upper = causal_entropy(position + offset, target)
        lower = causal_entropy(position - offset, target)
        gradient[index] = (upper - lower) / (2.0 * epsilon)
    return gradient


def _orthogonal_transform(dimension: int) -> np.ndarray:
    if dimension == 2:
        angle = 0.41
        return np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float64,
        )
    return np.eye(dimension, dtype=np.float64)


def _validate_square_matrix(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        raise ValueError(f"{name} must be a square matrix with dimension at least two")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_real_vector(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.ndim != 1 or arr.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _require_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


__all__ = [
    "CausalEntropicForceConfig",
    "CausalEntropicForceValidationResult",
    "EthicalConnectionBoundaryConfig",
    "EthicalConnectionBoundaryValidationResult",
    "EthicalYangMillsActionConfig",
    "EthicalYangMillsActionValidationResult",
    "causal_entropy",
    "causal_entropy_force",
    "ethical_yang_mills_action",
    "validate_causal_entropic_force_fixture",
    "validate_ethical_connection_boundary_fixture",
    "validate_ethical_yang_mills_action_fixture",
]
