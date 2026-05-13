# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 UPDE inter-layer validation
"""Executable simulator fixture for the Paper 0 UPDE inter-layer term."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from ..bridge.phase_artifact import LayerStateArtifact, UPDEPhaseArtifact
from ..fep.predictive_coding import hierarchical_prediction_error
from .spec_loader import load_upde_validation_spec


@dataclass(frozen=True, slots=True)
class InterlayerCouplingConfig:
    """Configuration for the inter-layer UPDE coupling fixture."""

    epsilon_lower: float = 0.6
    epsilon_upper: float = 0.35
    perturbation_delta: float = 0.1

    def __post_init__(self) -> None:
        _require_non_negative(self.epsilon_lower, "epsilon_lower")
        _require_non_negative(self.epsilon_upper, "epsilon_upper")
        if not np.isfinite(self.perturbation_delta) or self.perturbation_delta <= 0.0:
            raise ValueError("perturbation_delta must be finite and positive")


@dataclass(frozen=True, slots=True)
class InterlayerCouplingTerms:
    """Downward, upward, and total inter-layer coupling vectors."""

    downward: np.ndarray
    upward: np.ndarray
    total: np.ndarray
    lower_mean_phase: float
    upper_mean_phase: float


@dataclass(frozen=True, slots=True)
class InterlayerValidationResult:
    """Result of the Paper 0 inter-layer coupling fixture."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    lower_mean_phase: float
    upper_mean_phase: float
    coupling_total: tuple[float, ...]
    directional_sensitivity: MappingProxyType[str, float]
    null_controls: MappingProxyType[str, float]
    predictive_error_norm: float
    phase_artifact_payload: MappingProxyType[str, Any]


def circular_mean_phase(phases: np.ndarray) -> float:
    """Return the circular mean angle for a finite 1-D phase vector."""
    values = _phase_vector(phases, "phases")
    resultant = np.mean(np.exp(1j * values))
    if abs(resultant) < 1e-14:
        raise ValueError("phases have near-zero circular resultant")
    return float(np.angle(resultant))


def interlayer_coupling_terms(
    lower_theta: np.ndarray,
    current_theta: np.ndarray,
    upper_theta: np.ndarray,
    *,
    config: InterlayerCouplingConfig | None = None,
) -> InterlayerCouplingTerms:
    """Compute the Paper 0 inter-layer downward and upward channels."""
    cfg = config or InterlayerCouplingConfig()
    lower = _phase_vector(lower_theta, "lower_theta")
    current = _phase_vector(current_theta, "current_theta")
    upper = _phase_vector(upper_theta, "upper_theta")
    lower_mean = circular_mean_phase(lower)
    upper_mean = circular_mean_phase(upper)
    downward = cfg.epsilon_lower * np.sin(lower_mean - current)
    upward = cfg.epsilon_upper * np.sin(upper_mean - current)
    return InterlayerCouplingTerms(
        downward=downward,
        upward=upward,
        total=downward + upward,
        lower_mean_phase=lower_mean,
        upper_mean_phase=upper_mean,
    )


def validate_upde_interlayer_fixture(
    lower_theta: np.ndarray,
    current_theta: np.ndarray,
    upper_theta: np.ndarray,
    *,
    config: InterlayerCouplingConfig | None = None,
) -> InterlayerValidationResult:
    """Run the source-anchored inter-layer UPDE simulator fixture."""
    cfg = config or InterlayerCouplingConfig()
    lower = _phase_vector(lower_theta, "lower_theta")
    current = _phase_vector(current_theta, "current_theta")
    upper = _phase_vector(upper_theta, "upper_theta")
    spec = load_upde_validation_spec("upde.interlayer_coupling")
    terms = interlayer_coupling_terms(lower, current, upper, config=cfg)

    lower_shift = interlayer_coupling_terms(
        lower + cfg.perturbation_delta,
        current,
        upper,
        config=cfg,
    )
    upper_shift = interlayer_coupling_terms(
        lower,
        current,
        upper - cfg.perturbation_delta,
        config=cfg,
    )
    disconnected = interlayer_coupling_terms(
        lower,
        current,
        upper,
        config=InterlayerCouplingConfig(epsilon_lower=0.0, epsilon_upper=0.0),
    )
    directional = {
        "lower_to_downward_l2": float(np.linalg.norm(lower_shift.downward - terms.downward)),
        "lower_to_upward_l2": float(np.linalg.norm(lower_shift.upward - terms.upward)),
        "upper_to_downward_l2": float(np.linalg.norm(upper_shift.downward - terms.downward)),
        "upper_to_upward_l2": float(np.linalg.norm(upper_shift.upward - terms.upward)),
    }
    null_controls = {
        "disconnected_layer_linf": float(np.max(np.abs(disconnected.total))),
        "cross_channel_lower_to_upward_l2": directional["lower_to_upward_l2"],
        "cross_channel_upper_to_downward_l2": directional["upper_to_downward_l2"],
    }
    means = np.asarray(
        [circular_mean_phase(lower), circular_mean_phase(current), circular_mean_phase(upper)],
        dtype=np.float64,
    )
    hierarchy_K = np.array(
        [
            [0.0, cfg.epsilon_lower, 0.0],
            [cfg.epsilon_lower, 0.0, cfg.epsilon_upper],
            [0.0, cfg.epsilon_upper, 0.0],
        ],
        dtype=np.float64,
    )
    predictive_errors = hierarchical_prediction_error(means, means.copy(), hierarchy_K)
    artifact = _phase_artifact(means, spec)
    return InterlayerValidationResult(
        spec_key="upde.interlayer_coupling",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        lower_mean_phase=terms.lower_mean_phase,
        upper_mean_phase=terms.upper_mean_phase,
        coupling_total=tuple(float(item) for item in terms.total),
        directional_sensitivity=MappingProxyType(directional),
        null_controls=MappingProxyType(null_controls),
        predictive_error_norm=float(np.linalg.norm(predictive_errors)),
        phase_artifact_payload=MappingProxyType(artifact.to_dict()),
    )


def _phase_vector(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(np.ndarray, arr.copy())


def _phase_artifact(means: np.ndarray, spec: dict[str, Any]) -> UPDEPhaseArtifact:
    layers = [LayerStateArtifact(R=1.0, psi=float(phase)) for phase in means]
    alignment = np.cos(means[:, None] - means[None, :])
    return UPDEPhaseArtifact(
        layers=layers,
        cross_layer_alignment=alignment,
        stability_proxy=float(np.linalg.norm(alignment, ord="fro")),
        regime_id="paper0_upde_interlayer_fixture",
        metadata={
            "paper0_spec_key": "upde.interlayer_coupling",
            "paper0_validation_protocol": str(spec["validation_protocol"]),
            "hardware_status": str(spec["hardware_status"]),
        },
    )


def _require_non_negative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


__all__ = [
    "InterlayerCouplingConfig",
    "InterlayerCouplingTerms",
    "InterlayerValidationResult",
    "circular_mean_phase",
    "interlayer_coupling_terms",
    "validate_upde_interlayer_fixture",
]
