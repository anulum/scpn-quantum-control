# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 UPDE field validation
"""Executable simulator fixture for the Paper 0 UPDE global-field term."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from ..bridge.phase_artifact import LayerStateArtifact, UPDEPhaseArtifact
from .spec_loader import load_upde_validation_spec


@dataclass(frozen=True, slots=True)
class FieldCouplingConfig:
    """Configuration for the Paper 0 global-field coupling fixture."""

    zeta_L: float = 0.35
    psi_global: float = 1.2
    theta_psi: float = 0.25
    random_phase_samples: int = 256
    random_seed: int = 13

    def __post_init__(self) -> None:
        _require_non_negative(self.zeta_L, "zeta_L")
        _require_non_negative(self.psi_global, "psi_global")
        if not np.isfinite(self.theta_psi):
            raise ValueError("theta_psi must be finite")
        if not isinstance(self.random_phase_samples, int) or self.random_phase_samples < 2:
            raise ValueError("random_phase_samples must be at least 2")
        if not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")

    @property
    def amplitude(self) -> float:
        """Bounded scalar field amplitude zeta_L * Psi_Global."""
        return float(self.zeta_L * self.psi_global)


@dataclass(frozen=True, slots=True)
class FieldValidationResult:
    """Result of the Paper 0 field-coupling fixture."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    zeta_L: float
    psi_global: float
    theta_psi: float
    field_term: tuple[float, ...]
    field_alignment_projection: float
    null_controls: MappingProxyType[str, float]
    phase_artifact_payload: MappingProxyType[str, Any]


def field_coupling_term(
    theta: np.ndarray,
    *,
    config: FieldCouplingConfig | None = None,
) -> np.ndarray:
    """Return ``zeta_L * Psi_Global * cos(theta_i^L - Theta_Psi)``."""
    cfg = config or FieldCouplingConfig()
    theta_arr = _phase_vector(theta, "theta")
    return cfg.amplitude * np.cos(theta_arr - cfg.theta_psi)


def field_alignment_projection(
    theta: np.ndarray,
    field_term: np.ndarray,
    *,
    theta_psi: float,
) -> float:
    """Project a field term onto the source global-field cosine direction."""
    theta_arr = _phase_vector(theta, "theta")
    term = _phase_vector(field_term, "field_term")
    if theta_arr.shape != term.shape:
        raise ValueError(f"field_term must have shape {theta_arr.shape}, got {term.shape}")
    if not np.isfinite(theta_psi):
        raise ValueError("theta_psi must be finite")
    return float(np.mean(term * np.cos(theta_arr - theta_psi)))


def validate_upde_field_fixture(
    theta: np.ndarray,
    *,
    config: FieldCouplingConfig | None = None,
) -> FieldValidationResult:
    """Run the source-anchored global-field UPDE simulator fixture."""
    cfg = config or FieldCouplingConfig()
    theta_arr = _phase_vector(theta, "theta")
    spec = load_upde_validation_spec("upde.field_coupling")
    term = field_coupling_term(theta_arr, config=cfg)
    projection = field_alignment_projection(theta_arr, term, theta_psi=cfg.theta_psi)
    null_controls = _field_null_controls(theta_arr, cfg)
    artifact = _phase_artifact(theta_arr, term, cfg, spec)
    return FieldValidationResult(
        spec_key="upde.field_coupling",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        zeta_L=float(cfg.zeta_L),
        psi_global=float(cfg.psi_global),
        theta_psi=float(cfg.theta_psi),
        field_term=tuple(float(item) for item in term),
        field_alignment_projection=projection,
        null_controls=MappingProxyType(null_controls),
        phase_artifact_payload=MappingProxyType(artifact.to_dict()),
    )


def _field_null_controls(theta: np.ndarray, config: FieldCouplingConfig) -> dict[str, float]:
    zero_term = field_coupling_term(
        theta,
        config=FieldCouplingConfig(
            zeta_L=0.0,
            psi_global=config.psi_global,
            theta_psi=config.theta_psi,
            random_phase_samples=config.random_phase_samples,
            random_seed=config.random_seed,
        ),
    )
    rng = np.random.default_rng(config.random_seed)
    random_phases = rng.uniform(-np.pi, np.pi, size=config.random_phase_samples)
    random_projections = [
        field_alignment_projection(
            theta,
            field_coupling_term(
                theta,
                config=FieldCouplingConfig(
                    zeta_L=config.zeta_L,
                    psi_global=config.psi_global,
                    theta_psi=float(phase),
                    random_phase_samples=config.random_phase_samples,
                    random_seed=config.random_seed,
                ),
            ),
            theta_psi=config.theta_psi,
        )
        for phase in random_phases
    ]
    return {
        "zero_field_linf": float(np.max(np.abs(zero_term))),
        "randomised_phase_projection_abs_mean": float(abs(np.mean(random_projections))),
        "bounded_amplitude": config.amplitude,
    }


def _phase_artifact(
    theta: np.ndarray,
    field_term: np.ndarray,
    config: FieldCouplingConfig,
    spec: dict[str, Any],
) -> UPDEPhaseArtifact:
    z = np.mean(np.exp(1j * theta))
    layer = LayerStateArtifact(R=float(abs(z)), psi=float(np.angle(z)))
    stability = field_alignment_projection(theta, field_term, theta_psi=config.theta_psi)
    return UPDEPhaseArtifact(
        layers=[layer],
        cross_layer_alignment=np.ones((1, 1), dtype=np.float64),
        stability_proxy=stability,
        regime_id="paper0_upde_field_fixture",
        metadata={
            "paper0_spec_key": "upde.field_coupling",
            "paper0_validation_protocol": str(spec["validation_protocol"]),
            "hardware_status": str(spec["hardware_status"]),
            "zeta_L": float(config.zeta_L),
            "psi_global": float(config.psi_global),
            "theta_psi": float(config.theta_psi),
        },
    )


def _phase_vector(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(np.ndarray, arr.copy())


def _require_non_negative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


__all__ = [
    "FieldCouplingConfig",
    "FieldValidationResult",
    "field_alignment_projection",
    "field_coupling_term",
    "validate_upde_field_fixture",
]
