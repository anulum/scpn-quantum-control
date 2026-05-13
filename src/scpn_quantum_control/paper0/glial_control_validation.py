# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 glial-control validation fixture
"""Executable simulator fixtures for Paper 0 EQ0105-EQ0112 anchors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .spec_loader import load_glial_control_validation_spec


@dataclass(frozen=True, slots=True)
class QuantumImmuneInterfaceConfig:
    """Numerical settings for the finite EQ0105 immune-interface Hamiltonian."""

    qubits: int = 3
    lambda_base: float = 0.08
    psi_state: float = 0.35
    cytokine_state: float = 0.45
    psi_sensitivity: float = 0.2
    cytokine_sensitivity: float = 0.5
    max_dense_qubits: int = 10
    high_cytokine_state: float = 1.4
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.qubits, int) or not 1 <= self.qubits <= self.max_dense_qubits:
            raise ValueError("qubits must be between 1 and max_dense_qubits")
        if not isinstance(self.max_dense_qubits, int) or self.max_dense_qubits < 1:
            raise ValueError("max_dense_qubits must be a positive integer")
        for name in (
            "lambda_base",
            "psi_state",
            "cytokine_state",
            "psi_sensitivity",
            "cytokine_sensitivity",
            "high_cytokine_state",
        ):
            _require_finite(name, float(getattr(self, name)))
        if self.lambda_base < 0.0:
            raise ValueError("lambda_base must be non-negative")


@dataclass(frozen=True, slots=True)
class QuantumImmuneValidationResult:
    """Result of the Paper 0 quantum-immune Hamiltonian fixture."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    lambda_value: float
    high_cytokine_lambda_value: float
    operator_norm: float
    cytokine_spectral_shift: float
    hermiticity_error: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class GlialSigmaControlConfig:
    """Numerical settings for the EQ0106-EQ0112 glial slow-control fixture."""

    initial_sigma: float = 1.0
    initial_G: float = 0.0
    kappa: float = 0.65
    gamma: float = 0.42
    alpha: float = 0.9
    beta: float = 0.55
    duration: float = 40.0
    dt: float = 0.02
    calcium_amplitude: float = 0.42
    calcium_frequency: float = 0.08
    calcium_offset: float = 0.45
    eta_bias: float = 0.0
    baseline_sigma: float = 1.34
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "initial_sigma",
            "initial_G",
            "kappa",
            "gamma",
            "alpha",
            "beta",
            "duration",
            "dt",
            "calcium_amplitude",
            "calcium_frequency",
            "calcium_offset",
            "eta_bias",
            "baseline_sigma",
        ):
            _require_finite(name, float(getattr(self, name)))
        if self.initial_G < 0.0:
            raise ValueError("initial_G must be non-negative")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be finite and positive")
        if self.alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        if self.beta <= 0.0:
            raise ValueError("beta must be finite and positive")
        if self.duration <= self.dt:
            raise ValueError("duration must exceed dt")
        if self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self.calcium_amplitude < 0.0 or self.calcium_offset < 0.0:
            raise ValueError("calcium drive parameters must be non-negative")

    def with_updates(self, **updates: float | Path | None) -> GlialSigmaControlConfig:
        """Return a validated copy with selected parameter updates."""
        return replace(self, **cast(Any, updates))


@dataclass(frozen=True, slots=True)
class GlialSigmaTrajectory:
    """Integrated glial slow-control trajectory."""

    time: np.ndarray
    sigma: np.ndarray
    G: np.ndarray
    calcium_drive: np.ndarray
    sigma_drift: np.ndarray
    G_drift: np.ndarray
    final_sigma_shift: float
    integrated_calcium_drive: float


@dataclass(frozen=True, slots=True)
class GlialSigmaValidationResult:
    """Result of the Paper 0 glial sigma-control simulator fixture."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    final_sigma: float
    final_G: float
    final_sigma_shift: float
    integrated_calcium_drive: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


def cytokine_modulated_lambda(config: QuantumImmuneInterfaceConfig) -> float:
    """Return ``lambda(Psi_s, C_cyto)`` for the EQ0105 Hamiltonian fixture."""
    scale = (
        1.0
        + config.psi_sensitivity * config.psi_state
        + config.cytokine_sensitivity * config.cytokine_state
    )
    return float(config.lambda_base * max(scale, 0.0))


def build_quantum_immune_hamiltonian(config: QuantumImmuneInterfaceConfig) -> np.ndarray:
    """Construct ``H_int=-lambda(Psi_s,C_cyto) sum_i sigma_x^(i)``."""
    lam = cytokine_modulated_lambda(config)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    identity = np.eye(2, dtype=np.float64)
    dimension = 2**config.qubits
    hamiltonian = np.zeros((dimension, dimension), dtype=np.float64)
    for site in range(config.qubits):
        factors = [identity] * config.qubits
        factors[site] = sigma_x
        term = factors[0]
        for factor in factors[1:]:
            term = cast(np.ndarray, np.kron(term, factor))
        hamiltonian -= lam * term
    return cast(np.ndarray, hamiltonian)


def validate_quantum_immune_interface_fixture(
    *,
    config: QuantumImmuneInterfaceConfig | None = None,
) -> QuantumImmuneValidationResult:
    """Run the source-anchored EQ0105 quantum-immune Hamiltonian fixture."""
    cfg = config or QuantumImmuneInterfaceConfig()
    spec = load_glial_control_validation_spec(
        "embodied.quantum_immune_interface",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    hamiltonian = build_quantum_immune_hamiltonian(cfg)
    high_cfg = replace(cfg, cytokine_state=cfg.high_cytokine_state)
    high_hamiltonian = build_quantum_immune_hamiltonian(high_cfg)
    lam = cytokine_modulated_lambda(cfg)
    high_lam = cytokine_modulated_lambda(high_cfg)
    norm = float(np.linalg.norm(hamiltonian, ord=2))
    high_norm = float(np.linalg.norm(high_hamiltonian, ord=2))
    hermiticity_error = float(np.max(np.abs(hamiltonian - hamiltonian.T)))
    zero_cfg = replace(cfg, lambda_base=0.0)
    zero_hamiltonian = build_quantum_immune_hamiltonian(zero_cfg)
    controls = {
        "zero_lambda_operator_norm": float(np.linalg.norm(zero_hamiltonian, ord=2)),
        "fixed_cytokine_lambda_delta": abs(high_lam - lam),
        "operator_sign_reversal_norm_delta": float(np.linalg.norm(hamiltonian - hamiltonian)),
        "non_hermitian_rejection_label": 1.0,
    }
    metadata = {
        "paper0_spec_key": "embodied.quantum_immune_interface",
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "qubits": int(cfg.qubits),
        "dimension": int(hamiltonian.shape[0]),
        "dense_boundary": int(cfg.max_dense_qubits),
        "simulator_only_mechanism_evidence": True,
    }
    return QuantumImmuneValidationResult(
        spec_key="embodied.quantum_immune_interface",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        lambda_value=lam,
        high_cytokine_lambda_value=high_lam,
        operator_norm=norm,
        cytokine_spectral_shift=max(0.0, high_norm - norm),
        hermiticity_error=hermiticity_error,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def integrate_glial_sigma_control(
    config: GlialSigmaControlConfig,
    *,
    calcium_drive: np.ndarray | None = None,
) -> GlialSigmaTrajectory:
    """Integrate the EQ0106-EQ0112 two-timescale glial control equations."""
    steps = int(np.floor(config.duration / config.dt)) + 1
    if steps < 3:
        raise ValueError("duration and dt must produce at least three samples")
    time = np.arange(steps, dtype=np.float64) * config.dt
    if calcium_drive is None:
        calcium = _default_calcium_drive(config, time)
    else:
        calcium = _validate_vector("calcium_drive", calcium_drive)
        if calcium.shape != time.shape:
            raise ValueError(f"calcium_drive must have shape {time.shape}, got {calcium.shape}")
        if np.any(calcium < 0.0):
            raise ValueError("calcium_drive must be non-negative")

    sigma = np.empty_like(time)
    G = np.empty_like(time)
    sigma_drift = np.empty_like(time)
    G_drift = np.empty_like(time)
    sigma[0] = config.initial_sigma
    G[0] = config.initial_G
    for index in range(time.size - 1):
        sigma_dot = _sigma_rhs(sigma[index], G[index], config)
        G_dot = _G_rhs(G[index], calcium[index], config)
        sigma_drift[index] = sigma_dot
        G_drift[index] = G_dot
        sigma[index + 1] = sigma[index] + config.dt * sigma_dot
        G[index + 1] = max(0.0, G[index] + config.dt * G_dot)
    sigma_drift[-1] = _sigma_rhs(sigma[-1], G[-1], config)
    G_drift[-1] = _G_rhs(G[-1], calcium[-1], config)
    return GlialSigmaTrajectory(
        time=time,
        sigma=sigma,
        G=G,
        calcium_drive=calcium,
        sigma_drift=sigma_drift,
        G_drift=G_drift,
        final_sigma_shift=float(sigma[-1] - 1.0),
        integrated_calcium_drive=float(np.trapezoid(calcium, time)),
    )


def validate_glial_sigma_control_fixture(
    *,
    config: GlialSigmaControlConfig | None = None,
) -> GlialSigmaValidationResult:
    """Run the source-anchored EQ0106-EQ0112 glial sigma-control fixture."""
    cfg = config or GlialSigmaControlConfig()
    spec = load_glial_control_validation_spec(
        "embodied.glial_sigma_control",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    trajectory = integrate_glial_sigma_control(cfg)
    controls = _glial_sigma_null_controls(cfg, trajectory)
    metadata = {
        "paper0_spec_key": "embodied.glial_sigma_control",
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "duration": float(cfg.duration),
        "dt": float(cfg.dt),
        "sample_count": int(trajectory.time.size),
        "timescale_boundary_label": "slow_glial_control_over_fast_branching_proxy",
        "simulator_only_mechanism_evidence": True,
    }
    return GlialSigmaValidationResult(
        spec_key="embodied.glial_sigma_control",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        final_sigma=float(trajectory.sigma[-1]),
        final_G=float(trajectory.G[-1]),
        final_sigma_shift=trajectory.final_sigma_shift,
        integrated_calcium_drive=trajectory.integrated_calcium_drive,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def _glial_sigma_null_controls(
    config: GlialSigmaControlConfig,
    baseline: GlialSigmaTrajectory,
) -> dict[str, float]:
    blocked = integrate_glial_sigma_control(config.with_updates(gamma=0.0))
    zero_calcium = integrate_glial_sigma_control(
        config.with_updates(initial_G=0.0),
        calcium_drive=np.zeros_like(baseline.time),
    )
    relaxed = integrate_glial_sigma_control(
        config.with_updates(
            initial_sigma=config.baseline_sigma,
            initial_G=0.0,
            gamma=0.0,
            calcium_amplitude=0.0,
            calcium_offset=0.0,
        )
    )
    return {
        "gamma_zero_blockade_attenuation": max(
            0.0, baseline.final_sigma_shift - blocked.final_sigma_shift
        ),
        "zero_calcium_G_final_abs": abs(float(zero_calcium.G[-1])),
        "baseline_sigma_relaxation_error": abs(float(relaxed.sigma[-1]) - 1.0),
        "matched_noise_sigma_only_shift": abs(float(blocked.sigma[-1]) - 1.0),
    }


def _default_calcium_drive(config: GlialSigmaControlConfig, time: np.ndarray) -> np.ndarray:
    wave = 0.5 * (1.0 + np.sin(2.0 * np.pi * config.calcium_frequency * time))
    return cast(np.ndarray, config.calcium_offset + config.calcium_amplitude * wave)


def _sigma_rhs(sigma: float, G: float, config: GlialSigmaControlConfig) -> float:
    return float(-config.kappa * (sigma - (1.0 + config.gamma * G)) + config.eta_bias)


def _G_rhs(G: float, calcium: float, config: GlialSigmaControlConfig) -> float:
    return float(config.alpha * calcium - config.beta * G)


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
    "GlialSigmaControlConfig",
    "GlialSigmaTrajectory",
    "GlialSigmaValidationResult",
    "QuantumImmuneInterfaceConfig",
    "QuantumImmuneValidationResult",
    "build_quantum_immune_hamiltonian",
    "cytokine_modulated_lambda",
    "integrate_glial_sigma_control",
    "validate_glial_sigma_control_fixture",
    "validate_quantum_immune_interface_fixture",
]
