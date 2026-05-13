# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 macro-transition RG validation fixture
"""Executable simulator fixture for the Paper 0 effective-coupling RG flow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .spec_loader import load_macro_transition_validation_spec

BetaFunction = Callable[[float, float], float]
ScalarBetaFunction = Callable[[float], float]


@dataclass(frozen=True, slots=True)
class RGFlowValidationConfig:
    """Numerical settings for the source-anchored RG-flow fixture."""

    initial_K_eff: float = 0.22
    fixed_point_K_eff: float = 1.25
    beta_rate: float = 0.9
    constant_beta_value: float = 0.23
    scale_min: float = 1.0
    scale_max: float = 16.0
    scale_count: int = 33
    fixed_point_tolerance: float = 1.0e-9
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_K_eff):
            raise ValueError("initial_K_eff must be finite")
        if not np.isfinite(self.fixed_point_K_eff):
            raise ValueError("fixed_point_K_eff must be finite")
        if not np.isfinite(self.beta_rate) or self.beta_rate <= 0.0:
            raise ValueError("beta_rate must be finite and positive")
        if not np.isfinite(self.constant_beta_value):
            raise ValueError("constant_beta_value must be finite")
        if not np.isfinite(self.scale_min) or not np.isfinite(self.scale_max):
            raise ValueError("scale bounds must be finite")
        if self.scale_min <= 0.0 or self.scale_max <= 0.0:
            raise ValueError("scale bounds must be positive")
        if self.scale_max <= self.scale_min:
            raise ValueError("scale_max must exceed scale_min")
        if self.scale_count < 2:
            raise ValueError("scale_count must be at least 2")
        if not np.isfinite(self.fixed_point_tolerance) or self.fixed_point_tolerance <= 0.0:
            raise ValueError("fixed_point_tolerance must be finite and positive")


@dataclass(frozen=True, slots=True)
class RGFlowValidationResult:
    """Result of the Paper 0 effective-coupling RG simulator fixture."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    scale_grid: tuple[float, ...]
    K_eff_trajectory: tuple[float, ...]
    beta_values: tuple[float, ...]
    initial_K_eff: float
    final_K_eff: float
    fixed_point_candidate: float
    fixed_point_stability: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


def zero_beta_flow(initial_K_eff: float, scale_grid: np.ndarray) -> np.ndarray:
    """Return the invariant flow for beta_K = 0."""
    if not np.isfinite(initial_K_eff):
        raise ValueError("initial_K_eff must be finite")
    scales = _validate_scale_grid(scale_grid)
    return cast(np.ndarray, np.full_like(scales, float(initial_K_eff), dtype=np.float64))


def constant_beta_flow(
    initial_K_eff: float,
    scale_grid: np.ndarray,
    beta_value: float,
) -> np.ndarray:
    """Return the analytic log-scale flow for constant beta_K."""
    if not np.isfinite(initial_K_eff):
        raise ValueError("initial_K_eff must be finite")
    if not np.isfinite(beta_value):
        raise ValueError("beta_value must be finite")
    scales = _validate_scale_grid(scale_grid)
    return cast(
        np.ndarray,
        float(initial_K_eff) + float(beta_value) * np.log(scales / scales[0]),
    )


def integrate_rg_flow(
    *,
    initial_K_eff: float,
    scale_grid: np.ndarray,
    beta_function: BetaFunction,
) -> np.ndarray:
    """Integrate ``dK_eff / d log(mu) = beta_K`` over a positive scale grid."""
    if not np.isfinite(initial_K_eff):
        raise ValueError("initial_K_eff must be finite")
    scales = _validate_scale_grid(scale_grid)
    log_scales = np.log(scales)
    trajectory = np.empty_like(scales, dtype=np.float64)
    trajectory[0] = float(initial_K_eff)
    for index in range(1, scales.size):
        previous = float(trajectory[index - 1])
        step = float(log_scales[index] - log_scales[index - 1])
        scale_previous = float(scales[index - 1])
        scale_next = float(scales[index])
        scale_mid = float(np.sqrt(scale_previous * scale_next))
        k1 = _finite_beta(beta_function(previous, scale_previous))
        k2 = _finite_beta(beta_function(previous + 0.5 * step * k1, scale_mid))
        k3 = _finite_beta(beta_function(previous + 0.5 * step * k2, scale_mid))
        k4 = _finite_beta(beta_function(previous + step * k3, scale_next))
        trajectory[index] = previous + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(trajectory[index]):
            raise ValueError("K_eff trajectory produced a non-finite value")
    return trajectory


def classify_fixed_point_stability(
    beta_function: ScalarBetaFunction,
    fixed_point: float,
    *,
    step: float = 1.0e-6,
) -> str:
    """Classify local one-dimensional fixed-point stability from beta prime."""
    if not np.isfinite(fixed_point):
        raise ValueError("fixed_point must be finite")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive")
    derivative = (
        _finite_beta(beta_function(fixed_point + step))
        - _finite_beta(beta_function(fixed_point - step))
    ) / (2.0 * step)
    if derivative < 0.0:
        return "stable"
    if derivative > 0.0:
        return "unstable"
    return "marginal"


def validate_macro_transition_rg_fixture(
    *,
    config: RGFlowValidationConfig | None = None,
) -> RGFlowValidationResult:
    """Run the source-anchored Paper 0 effective-coupling RG fixture."""
    cfg = config or RGFlowValidationConfig()
    spec = load_macro_transition_validation_spec(
        "macro_transition.effective_coupling_rg",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    scales = np.geomspace(cfg.scale_min, cfg.scale_max, cfg.scale_count)

    def beta(value: float, _scale: float) -> float:
        return cfg.beta_rate * (cfg.fixed_point_K_eff - value)

    trajectory = integrate_rg_flow(
        initial_K_eff=cfg.initial_K_eff,
        scale_grid=scales,
        beta_function=beta,
    )
    beta_values = np.array(
        [beta(float(value), float(scale)) for value, scale in zip(trajectory, scales)]
    )
    stability = classify_fixed_point_stability(
        lambda value: cfg.beta_rate * (cfg.fixed_point_K_eff - value),
        cfg.fixed_point_K_eff,
    )
    controls = _rg_null_controls(scales, cfg, trajectory)
    metadata = {
        "paper0_spec_key": "macro_transition.effective_coupling_rg",
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "scale_count": int(scales.size),
        "scale_min": float(scales[0]),
        "scale_max": float(scales[-1]),
        "beta_rate": float(cfg.beta_rate),
        "constant_beta_value": float(cfg.constant_beta_value),
        "integration_variable": "log(mu)",
        "fixed_point_condition": "beta_K(K_star)=0",
    }
    return RGFlowValidationResult(
        spec_key="macro_transition.effective_coupling_rg",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        scale_grid=tuple(float(item) for item in scales),
        K_eff_trajectory=tuple(float(item) for item in trajectory),
        beta_values=tuple(float(item) for item in beta_values),
        initial_K_eff=float(trajectory[0]),
        final_K_eff=float(trajectory[-1]),
        fixed_point_candidate=float(cfg.fixed_point_K_eff),
        fixed_point_stability=stability,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def _rg_null_controls(
    scales: np.ndarray,
    config: RGFlowValidationConfig,
    trajectory: np.ndarray,
) -> dict[str, float]:
    zero = zero_beta_flow(config.initial_K_eff, scales)
    constant = integrate_rg_flow(
        initial_K_eff=config.initial_K_eff,
        scale_grid=scales,
        beta_function=lambda _value, _scale: config.constant_beta_value,
    )
    constant_expected = constant_beta_flow(
        config.initial_K_eff,
        scales,
        config.constant_beta_value,
    )
    reversed_flow = integrate_rg_flow(
        initial_K_eff=config.initial_K_eff,
        scale_grid=scales,
        beta_function=lambda value, _scale: -config.beta_rate * (config.fixed_point_K_eff - value),
    )
    return {
        "zero_beta_invariance_linf": float(np.max(np.abs(zero - config.initial_K_eff))),
        "constant_beta_analytic_error_linf": float(np.max(np.abs(constant - constant_expected))),
        "reverse_beta_final_delta": float(abs(trajectory[-1] - reversed_flow[-1])),
        "fixed_point_beta_abs": abs(
            _finite_beta(config.beta_rate * (config.fixed_point_K_eff - config.fixed_point_K_eff))
        ),
    }


def _validate_scale_grid(scale_grid: np.ndarray) -> np.ndarray:
    scales = np.array(scale_grid, dtype=np.float64, copy=True)
    if scales.ndim != 1:
        raise ValueError("scale_grid must be one-dimensional")
    if scales.size < 2:
        raise ValueError("scale_grid must contain at least two values")
    if not np.all(np.isfinite(scales)):
        raise ValueError("scale_grid must contain only finite values")
    if not np.all(scales > 0.0):
        raise ValueError("scale_grid must contain only positive values")
    if not np.all(np.diff(scales) > 0.0):
        raise ValueError("scale_grid must be strictly increasing")
    return scales


def _finite_beta(value: float) -> float:
    beta = float(value)
    if not np.isfinite(beta):
        raise ValueError("beta_function must return only finite values")
    return beta


__all__ = [
    "RGFlowValidationConfig",
    "RGFlowValidationResult",
    "classify_fixed_point_stability",
    "constant_beta_flow",
    "integrate_rg_flow",
    "validate_macro_transition_rg_fixture",
    "zero_beta_flow",
]
