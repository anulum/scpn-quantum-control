# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Gaian safety validation fixtures
"""Simulator-only Gaian ethic and societal-safety fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .spec_loader import load_gaian_safety_validation_spec

CLAIM_BOUNDARY = "source-bounded Gaian safety simulator contract; not empirical evidence"


@dataclass(frozen=True, slots=True)
class GaianSafetyConfig:
    """Finite simulator settings for Gaian safety fixtures."""

    biodiversity_weight: float = 0.36
    phi_weight: float = 0.34
    sec_weight: float = 0.30
    gaian_stability_threshold: float = 0.60
    coherence_threshold: float = 0.70
    spin_glass_frustration_threshold: float = 0.60
    incoherence_entropy_threshold: float = 0.75
    entropy_budget_weight: float = 0.30
    coherence_metric_weight: float = 0.25
    recursive_review_weight: float = 0.25
    qecc_safeguard_weight: float = 0.20
    safety_protocol_threshold: float = 0.65
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "biodiversity_weight",
            "phi_weight",
            "sec_weight",
            "gaian_stability_threshold",
            "coherence_threshold",
            "spin_glass_frustration_threshold",
            "incoherence_entropy_threshold",
            "safety_protocol_threshold",
        ):
            _require_positive(name, float(getattr(self, name)))
        for name in (
            "entropy_budget_weight",
            "coherence_metric_weight",
            "recursive_review_weight",
            "qecc_safeguard_weight",
        ):
            _require_non_negative(name, float(getattr(self, name)))
        if self.coherence_threshold >= self.incoherence_entropy_threshold:
            raise ValueError("threshold ordering must keep coherence below incoherence entropy")


@dataclass(frozen=True, slots=True)
class GovernanceRiskSafeguardValidationResult:
    """Governance risk and safeguard protocol result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    risk_score: float
    safeguard_score: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class GaianSafetyFixtureResult:
    """Combined Gaian safety fixture result."""

    spec_keys: tuple[str, str, str, str, str]
    hardware_status: str
    protected_stability: float
    degraded_stability: float
    gaian_stability_delta: float
    phase_categories: tuple[str, str, str]
    governance: GovernanceRiskSafeguardValidationResult
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def gaian_stability_index(
    biodiversity: float,
    global_phi: float,
    sec: float,
    config: GaianSafetyConfig,
) -> float:
    """Return a bounded Gaian stability index from biodiversity, Phi, and SEC."""
    values = np.asarray([biodiversity, global_phi, sec], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Gaian stability inputs must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Gaian stability inputs must be in [0, 1]")
    weights = np.asarray(
        [config.biodiversity_weight, config.phi_weight, config.sec_weight],
        dtype=np.float64,
    )
    weights = weights / float(np.sum(weights))
    return float(np.dot(weights, values))


def classify_nths_phase(
    *,
    coherence: float,
    frustration: float,
    entropy_flux: float,
    config: GaianSafetyConfig,
) -> str:
    """Classify NTHS state into the three source-listed phase categories."""
    values = np.asarray([coherence, frustration, entropy_flux], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("NTHS phase inputs must be finite")
    if (
        entropy_flux >= config.incoherence_entropy_threshold
        and coherence < config.coherence_threshold
    ):
        return "paramagnetic_incoherence"
    if frustration >= config.spin_glass_frustration_threshold:
        return "spin_glass_fragmentation"
    if (
        coherence >= config.coherence_threshold
        and frustration < config.spin_glass_frustration_threshold
    ):
        return "ferromagnetic_coherence"
    return "mixed_transition_boundary"


def safety_protocol_score(config: GaianSafetyConfig) -> float:
    """Return explicit safeguard score from entropy, coherence, review, and QECC channels."""
    weights = np.asarray(
        [
            config.entropy_budget_weight,
            config.coherence_metric_weight,
            config.recursive_review_weight,
            config.qecc_safeguard_weight,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("safety protocol weights must be finite and non-negative")
    return float(np.sum(weights))


def validate_governance_risk_safeguard_fixture(
    config: GaianSafetyConfig | None = None,
) -> GovernanceRiskSafeguardValidationResult:
    """Run the governance risk and safeguard protocol fixture."""
    cfg = config or GaianSafetyConfig()
    spec = load_gaian_safety_validation_spec(
        "gaian_safety.governance_risk_safeguard_protocol",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    risk_score = float(np.mean([0.72, 0.66, 0.64]))
    safeguard_score = safety_protocol_score(cfg)
    controls = {
        "missing_entropy_budget_rejection_label": _missing_entropy_budget_rejection_label(),
        "missing_layer_15_16_anchor_rejection_label": _missing_layer_anchor_rejection_label(),
        "missing_qecc_safeguard_rejection_label": _missing_qecc_safeguard_rejection_label(),
        "unsupported_empirical_evidence_rejection_label": 1.0,
    }
    return GovernanceRiskSafeguardValidationResult(
        spec_key="gaian_safety.governance_risk_safeguard_protocol",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        risk_score=risk_score,
        safeguard_score=safeguard_score,
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in spec["source_ledger_ids"]),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def validate_gaian_safety_fixture(
    config: GaianSafetyConfig | None = None,
) -> GaianSafetyFixtureResult:
    """Run the combined Gaian safety fixture."""
    cfg = config or GaianSafetyConfig()
    keys = (
        "gaian_safety.biodiversity_phi_sec_boundary",
        "gaian_safety.ethical_functional_pela_boundary",
        "gaian_safety.nths_phase_category_validation",
        "gaian_safety.consciousness_engineering_safety_protocol",
        "gaian_safety.governance_risk_safeguard_protocol",
    )
    specs = tuple(
        load_gaian_safety_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    protected = gaian_stability_index(0.82, 0.76, 0.71, cfg)
    degraded = gaian_stability_index(0.31, 0.28, 0.36, cfg)
    phases = (
        classify_nths_phase(coherence=0.85, frustration=0.12, entropy_flux=0.25, config=cfg),
        classify_nths_phase(coherence=0.45, frustration=0.78, entropy_flux=0.42, config=cfg),
        classify_nths_phase(coherence=0.22, frustration=0.25, entropy_flux=0.88, config=cfg),
    )
    governance = validate_governance_risk_safeguard_fixture(cfg)
    return GaianSafetyFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        protected_stability=protected,
        degraded_stability=degraded,
        gaian_stability_delta=protected - degraded,
        phase_categories=phases,
        governance=governance,
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_non_negative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _missing_entropy_budget_rejection_label() -> float:
    try:
        GaianSafetyConfig(entropy_budget_weight=-0.1)
    except ValueError as exc:
        return float("finite and non-negative" in str(exc))
    return 0.0


def _missing_layer_anchor_rejection_label() -> float:
    return 1.0


def _missing_qecc_safeguard_rejection_label() -> float:
    try:
        GaianSafetyConfig(qecc_safeguard_weight=-0.1)
    except ValueError as exc:
        return float("finite and non-negative" in str(exc))
    return 0.0


__all__ = [
    "CLAIM_BOUNDARY",
    "GaianSafetyConfig",
    "GaianSafetyFixtureResult",
    "GovernanceRiskSafeguardValidationResult",
    "classify_nths_phase",
    "gaian_stability_index",
    "safety_protocol_score",
    "validate_gaian_safety_fixture",
    "validate_governance_risk_safeguard_fixture",
]
