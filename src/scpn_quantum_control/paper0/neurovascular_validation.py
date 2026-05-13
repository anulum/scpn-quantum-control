# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 neurovascular validation fixture
"""Executable simulator fixture for Paper 0 neurovascular phase coupling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .spec_loader import load_neurovascular_validation_spec


@dataclass(frozen=True, slots=True)
class NeurovascularValidationConfig:
    """Numerical settings for the two-oscillator neurovascular fixture."""

    omega_neural: float = 0.97
    omega_hemo: float = 1.0
    K_NH: float = 0.42
    initial_theta_neural: float = 1.3
    initial_theta_hemo: float = 0.0
    duration: float = 80.0
    dt: float = 0.02
    transient_fraction: float = 0.5
    detuned_omega_hemo: float = 1.55
    impaired_cbf_coupling_scale: float = 0.25
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_finite("omega_neural", self.omega_neural)
        _require_finite("omega_hemo", self.omega_hemo)
        _require_finite("K_NH", self.K_NH)
        _require_finite("initial_theta_neural", self.initial_theta_neural)
        _require_finite("initial_theta_hemo", self.initial_theta_hemo)
        _require_finite("duration", self.duration)
        _require_finite("dt", self.dt)
        _require_finite("transient_fraction", self.transient_fraction)
        _require_finite("detuned_omega_hemo", self.detuned_omega_hemo)
        _require_finite("impaired_cbf_coupling_scale", self.impaired_cbf_coupling_scale)
        if self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self.duration <= self.dt:
            raise ValueError("duration must exceed dt")
        if not 0.0 <= self.transient_fraction < 1.0:
            raise ValueError("transient_fraction must be in [0, 1)")
        if self.K_NH < 0.0:
            raise ValueError("K_NH must be non-negative")
        if self.impaired_cbf_coupling_scale < 0.0:
            raise ValueError("impaired_cbf_coupling_scale must be non-negative")


@dataclass(frozen=True, slots=True)
class NeurovascularTrajectory:
    """Integrated two-oscillator neurovascular phase trajectory."""

    time: np.ndarray
    theta_neural: np.ndarray
    theta_hemo: np.ndarray
    neural_drift: np.ndarray
    analysis_start_index: int


@dataclass(frozen=True, slots=True)
class NeurovascularValidationResult:
    """Result of the Paper 0 neurovascular simulator fixture."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    phase_locking_value: float
    mean_frequency_slip: float
    final_phase_difference: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


def phase_locking_value(phase_difference: np.ndarray) -> float:
    """Return the circular phase-locking value for a phase-difference trace."""
    diff = _validate_vector("phase_difference", phase_difference)
    return float(abs(np.mean(np.exp(1j * diff))))


def mean_frequency_slip(
    phase_difference: np.ndarray,
    dt: float,
    analysis_start_index: int = 0,
) -> float:
    """Return mean derivative of the unwrapped phase difference."""
    diff = _validate_vector("phase_difference", phase_difference)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not 0 <= analysis_start_index < diff.size - 1:
        raise ValueError("analysis_start_index must leave at least two samples")
    unwrapped = np.unwrap(diff)
    derivative = np.diff(unwrapped[analysis_start_index:]) / dt
    return float(np.mean(derivative))


def integrate_neurovascular_phase_coupling(
    config: NeurovascularValidationConfig,
    *,
    hemodynamic_phase: np.ndarray | None = None,
) -> NeurovascularTrajectory:
    """Integrate the Paper 0 two-oscillator hemodynamic-to-neural phase model."""
    steps = int(np.floor(config.duration / config.dt)) + 1
    if steps < 3:
        raise ValueError("duration and dt must produce at least three samples")
    time = np.arange(steps, dtype=np.float64) * config.dt
    if hemodynamic_phase is None:
        theta_hemo = config.initial_theta_hemo + config.omega_hemo * time
    else:
        theta_hemo = _validate_vector("hemodynamic_phase", hemodynamic_phase)
        if theta_hemo.shape != time.shape:
            raise ValueError(
                f"hemodynamic_phase must have shape {time.shape}, got {theta_hemo.shape}"
            )

    theta_neural = np.empty_like(time)
    neural_drift = np.empty_like(time)
    theta_neural[0] = config.initial_theta_neural
    for index in range(time.size - 1):
        drift = _neural_phase_rhs(theta_neural[index], theta_hemo[index], config)
        neural_drift[index] = drift
        theta_neural[index + 1] = theta_neural[index] + config.dt * drift
    neural_drift[-1] = _neural_phase_rhs(theta_neural[-1], theta_hemo[-1], config)
    analysis_start_index = int(np.floor(config.transient_fraction * (time.size - 1)))
    return NeurovascularTrajectory(
        time=time,
        theta_neural=theta_neural,
        theta_hemo=theta_hemo,
        neural_drift=neural_drift,
        analysis_start_index=analysis_start_index,
    )


def validate_neurovascular_phase_coupling_fixture(
    *,
    config: NeurovascularValidationConfig | None = None,
) -> NeurovascularValidationResult:
    """Run the source-anchored Paper 0 neurovascular phase-coupling fixture."""
    cfg = config or NeurovascularValidationConfig()
    spec = load_neurovascular_validation_spec(
        "embodied.neurovascular_phase_coupling",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    trajectory = integrate_neurovascular_phase_coupling(cfg)
    phase_difference = trajectory.theta_hemo - trajectory.theta_neural
    analysis_slice = slice(trajectory.analysis_start_index, None)
    plv = phase_locking_value(phase_difference[analysis_slice])
    slip = mean_frequency_slip(phase_difference, cfg.dt, trajectory.analysis_start_index)
    controls = _neurovascular_null_controls(cfg, trajectory, plv)
    metadata = {
        "paper0_spec_key": "embodied.neurovascular_phase_coupling",
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "duration": float(cfg.duration),
        "dt": float(cfg.dt),
        "sample_count": int(trajectory.time.size),
        "analysis_start_index": int(trajectory.analysis_start_index),
        "omega_neural": float(cfg.omega_neural),
        "omega_hemo": float(cfg.omega_hemo),
        "K_NH": float(cfg.K_NH),
        "impaired_cbf_boundary_is_labelled_control": True,
    }
    return NeurovascularValidationResult(
        spec_key="embodied.neurovascular_phase_coupling",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        phase_locking_value=plv,
        mean_frequency_slip=slip,
        final_phase_difference=float(np.angle(np.exp(1j * phase_difference[-1]))),
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def _neurovascular_null_controls(
    config: NeurovascularValidationConfig,
    baseline: NeurovascularTrajectory,
    baseline_plv: float,
) -> dict[str, float]:
    zero_config = _replace_config(config, K_NH=0.0)
    zero = integrate_neurovascular_phase_coupling(zero_config)
    zero_diff = zero.theta_hemo - zero.theta_neural

    detuned_config = _replace_config(config, omega_hemo=config.detuned_omega_hemo)
    detuned = integrate_neurovascular_phase_coupling(detuned_config)
    detuned_diff = detuned.theta_hemo - detuned.theta_neural
    detuned_plv = phase_locking_value(detuned_diff[detuned.analysis_start_index :])

    shuffled_phase = _deterministic_shuffled_drive(baseline.theta_hemo)
    shuffled = integrate_neurovascular_phase_coupling(config, hemodynamic_phase=shuffled_phase)
    shuffled_diff = shuffled.theta_hemo - shuffled.theta_neural
    shuffled_plv = phase_locking_value(shuffled_diff[shuffled.analysis_start_index :])

    impaired_config = _replace_config(
        config,
        K_NH=config.K_NH * config.impaired_cbf_coupling_scale,
    )
    impaired = integrate_neurovascular_phase_coupling(impaired_config)
    impaired_diff = impaired.theta_hemo - impaired.theta_neural
    impaired_plv = phase_locking_value(impaired_diff[impaired.analysis_start_index :])
    return {
        "zero_K_NH_slip_abs": abs(
            mean_frequency_slip(zero_diff, config.dt, zero.analysis_start_index)
        ),
        "zero_K_NH_phase_locking_value": phase_locking_value(
            zero_diff[zero.analysis_start_index :]
        ),
        "detuned_phase_locking_drop": max(0.0, baseline_plv - detuned_plv),
        "detuned_phase_locking_value": detuned_plv,
        "shuffled_drive_phase_locking_drop": max(0.0, baseline_plv - shuffled_plv),
        "shuffled_drive_phase_locking_value": shuffled_plv,
        "impaired_cbf_phase_locking_value": impaired_plv,
        "impaired_cbf_boundary_label": 1.0,
    }


def _neural_phase_rhs(
    theta_neural: float,
    theta_hemo: float,
    config: NeurovascularValidationConfig,
) -> float:
    return float(config.omega_neural + config.K_NH * np.sin(theta_hemo - theta_neural))


def _deterministic_shuffled_drive(theta_hemo: np.ndarray) -> np.ndarray:
    if theta_hemo.size < 4:
        return cast(np.ndarray, theta_hemo.copy())
    rng = np.random.default_rng(9303)
    permutation = rng.permutation(theta_hemo.size)
    return cast(np.ndarray, theta_hemo[permutation])


def _replace_config(
    config: NeurovascularValidationConfig,
    **updates: float | Path | None,
) -> NeurovascularValidationConfig:
    return replace(config, **cast(Any, updates))


def _validate_vector(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _require_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


__all__ = [
    "NeurovascularTrajectory",
    "NeurovascularValidationConfig",
    "NeurovascularValidationResult",
    "integrate_neurovascular_phase_coupling",
    "mean_frequency_slip",
    "phase_locking_value",
    "validate_neurovascular_phase_coupling_fixture",
]
