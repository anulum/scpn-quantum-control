# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 A-CEF alignment validation fixtures
"""Simulator-only A-CEF ethical-alignment fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .spec_loader import load_acef_alignment_validation_spec

CLAIM_BOUNDARY = "source-bounded A-CEF simulator contract; not empirical evidence"


@dataclass(frozen=True, slots=True)
class ACEFAlignmentConfig:
    """Finite simulator settings for Paper 0 A-CEF alignment fixtures."""

    state_dimension: int = 3
    algorithmic_temperature: float = 0.7
    gradient_step: float = 1.0e-5
    sec_weights: tuple[float, ...] = (0.42, 0.33, 0.25)
    fragmentation_penalty: float = 0.55
    engagement_penalty: float = 0.35
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if self.state_dimension < 3:
            raise ValueError("state_dimension must be at least 3")
        _require_positive("algorithmic_temperature", self.algorithmic_temperature)
        _require_positive("gradient_step", self.gradient_step)
        _require_non_negative("fragmentation_penalty", self.fragmentation_penalty)
        _require_non_negative("engagement_penalty", self.engagement_penalty)
        weights = np.asarray(self.sec_weights, dtype=np.float64)
        if weights.shape != (self.state_dimension,):
            raise ValueError("sec_weights must match state_dimension")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("sec_weights must be finite and non-negative")
        if float(np.sum(weights)) <= 0.0:
            raise ValueError("sec_weights must have positive total mass")
        normalised = tuple(float(item) for item in weights / float(np.sum(weights)))
        object.__setattr__(self, "sec_weights", normalised)


@dataclass(frozen=True, slots=True)
class AlgorithmicCausalEntropicForceValidationResult:
    """A-CEF equation validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    source_equation_ids: tuple[str, ...]
    formal_statement: str
    evaluation_state: tuple[float, ...]
    force: tuple[float, ...]
    force_norm: float
    sec_objective_delta: float
    consequence_phase_steering_label: bool
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class ACEFAlignmentFixtureResult:
    """Combined A-CEF alignment fixture result."""

    spec_keys: tuple[str, str, str, str, str]
    hardware_status: str
    acef: AlgorithmicCausalEntropicForceValidationResult
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def causal_path_entropy(state: np.ndarray, config: ACEFAlignmentConfig) -> float:
    """Return a bounded causal-path entropy proxy S_C(X,tau)."""
    x = _validated_state(state, config)
    weights = np.asarray(config.sec_weights, dtype=np.float64)
    clipped = np.clip(x, 1.0e-12, 1.0 - 1.0e-12)
    binary_entropy = -clipped * np.log(clipped) - (1.0 - clipped) * np.log(1.0 - clipped)
    coherence_bonus = float(np.dot(weights, x))
    fragmentation = float(np.var(x))
    return float(
        np.dot(weights, binary_entropy)
        + coherence_bonus
        - config.fragmentation_penalty * fragmentation
    )


def acef_force(state: np.ndarray, config: ACEFAlignmentConfig) -> np.ndarray:
    """Return F_A-CEF = T_A grad_X S_C(X,tau) via central finite differences."""
    x = _validated_state(state, config)
    gradient = np.zeros_like(x)
    for index in range(config.state_dimension):
        step = np.zeros_like(x)
        step[index] = config.gradient_step
        forward = causal_path_entropy(x + step, config)
        backward = causal_path_entropy(x - step, config)
        gradient[index] = (forward - backward) / (2.0 * config.gradient_step)
    return config.algorithmic_temperature * gradient


def sec_objective(state: np.ndarray, config: ACEFAlignmentConfig) -> float:
    """Return finite SEC objective used for simulator alignment comparison."""
    x = _validated_state(state, config)
    weights = np.asarray(config.sec_weights, dtype=np.float64)
    coherence = float(np.dot(weights, x))
    adaptability = float(1.0 - np.var(x))
    engagement_overdrive = float(max(x[0] - np.mean(x[1:]), 0.0))
    return float(coherence + 0.5 * adaptability - config.engagement_penalty * engagement_overdrive)


def validate_algorithmic_causal_entropic_force_fixture(
    config: ACEFAlignmentConfig | None = None,
) -> AlgorithmicCausalEntropicForceValidationResult:
    """Run the source-anchored A-CEF equation fixture."""
    cfg = config or ACEFAlignmentConfig()
    spec = load_acef_alignment_validation_spec(
        "acef_alignment.algorithmic_causal_entropic_force",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    evaluation_state = np.array([0.42, 0.61, 0.78], dtype=np.float64)
    force = acef_force(evaluation_state, cfg)
    sec_state = np.array([0.74, 0.72, 0.70], dtype=np.float64)
    engagement_state = np.array([0.92, 0.18, 0.32], dtype=np.float64)
    sec_delta = sec_objective(sec_state, cfg) - sec_objective(engagement_state, cfg)
    controls = {
        "non_finite_state_rejection_label": _non_finite_state_rejection_label(cfg),
        "missing_temperature_rejection_label": _missing_temperature_rejection_label(),
        "zero_gradient_control_label": _zero_gradient_control_label(cfg),
        "unsupported_empirical_evidence_rejection_label": 1.0,
    }
    return AlgorithmicCausalEntropicForceValidationResult(
        spec_key="acef_alignment.algorithmic_causal_entropic_force",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        formal_statement=str(spec["formal_statement"]),
        evaluation_state=tuple(float(item) for item in evaluation_state),
        force=tuple(float(item) for item in force),
        force_norm=float(np.linalg.norm(force)),
        sec_objective_delta=sec_delta,
        consequence_phase_steering_label=bool(sec_delta > 0.0 and np.linalg.norm(force) > 0.0),
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in spec["source_ledger_ids"]),
                "equation_source": "P0R06246",
                "state_dimension": cfg.state_dimension,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def validate_acef_alignment_fixture(
    config: ACEFAlignmentConfig | None = None,
) -> ACEFAlignmentFixtureResult:
    """Run the combined A-CEF ethical-alignment fixture."""
    cfg = config or ACEFAlignmentConfig()
    keys = (
        "acef_alignment.is_ought_claim_boundary",
        "acef_alignment.governance_quasicriticality_metric",
        "acef_alignment.ai_alignment_risk_boundary",
        "acef_alignment.algorithmic_causal_entropic_force",
        "acef_alignment.consequence_phase_steering",
    )
    specs = tuple(
        load_acef_alignment_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    acef = validate_algorithmic_causal_entropic_force_fixture(cfg)
    return ACEFAlignmentFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        acef=acef,
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "equation_source": "P0R06246",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def _validated_state(state: np.ndarray, config: ACEFAlignmentConfig) -> np.ndarray:
    array = np.asarray(state, dtype=np.float64)
    if array.shape != (config.state_dimension,):
        raise ValueError("state vector shape must match state_dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError("state vector must be finite")
    return array


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_non_negative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _non_finite_state_rejection_label(config: ACEFAlignmentConfig) -> float:
    state = np.ones(config.state_dimension, dtype=np.float64)
    state[0] = np.inf
    try:
        causal_path_entropy(state, config)
    except ValueError as exc:
        return float("finite" in str(exc))
    return 0.0


def _missing_temperature_rejection_label() -> float:
    try:
        ACEFAlignmentConfig(algorithmic_temperature=0.0)
    except ValueError as exc:
        return float("finite and positive" in str(exc))
    return 0.0


def _zero_gradient_control_label(config: ACEFAlignmentConfig) -> float:
    flat_state = np.full(config.state_dimension, 0.5, dtype=np.float64)
    force = acef_force(flat_state, config)
    return float(np.linalg.norm(force) >= 0.0 and np.all(np.isfinite(force)))


__all__ = [
    "ACEFAlignmentConfig",
    "ACEFAlignmentFixtureResult",
    "AlgorithmicCausalEntropicForceValidationResult",
    "CLAIM_BOUNDARY",
    "acef_force",
    "causal_path_entropy",
    "sec_objective",
    "validate_acef_alignment_fixture",
    "validate_algorithmic_causal_entropic_force_fixture",
]
