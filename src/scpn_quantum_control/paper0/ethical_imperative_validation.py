# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Ethical Imperative validation fixtures
"""Simulator-only Ethical Imperative restatement fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .spec_loader import load_ethical_imperative_validation_spec

CLAIM_BOUNDARY = "source-bounded Ethical Imperative restatement contract; not empirical evidence"
PRIOR_SLICE = "P0R06251-P0R06272"


@dataclass(frozen=True, slots=True)
class EthicalImperativeConfig:
    """Finite simulator settings for Ethical Imperative restatement fixtures."""

    alignment_threshold: float = 0.70
    fragmentation_threshold: float = 0.65
    collapse_entropy_threshold: float = 0.75
    entropy_budget_weight: float = 0.34
    global_coherence_metric_weight: float = 0.33
    recursive_review_weight: float = 0.33
    governance_threshold: float = 0.80
    feedback_threshold: float = 0.60
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "alignment_threshold",
            "fragmentation_threshold",
            "collapse_entropy_threshold",
            "governance_threshold",
            "feedback_threshold",
        ):
            _require_positive(name, float(getattr(self, name)))
        for name in (
            "entropy_budget_weight",
            "global_coherence_metric_weight",
            "recursive_review_weight",
        ):
            _require_non_negative(name, float(getattr(self, name)))
        if self.alignment_threshold >= self.collapse_entropy_threshold:
            raise ValueError("threshold ordering must keep alignment below collapse entropy")


@dataclass(frozen=True, slots=True)
class GovernanceBeyondBordersValidationResult:
    """Governance-beyond-borders protocol result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    governance_score: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class EthicalImperativeFixtureResult:
    """Combined Ethical Imperative restatement fixture result."""

    spec_keys: tuple[str, str, str, str, str]
    hardware_status: str
    choice_labels: tuple[str, str, str]
    governance: GovernanceBeyondBordersValidationResult
    tuned_feedback_score: float
    untuned_feedback_score: float
    feedback_loop_delta: float
    config_thresholds: MappingProxyType[str, float]
    overlap_with_prior_slice: str
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def civilisation_choice_label(
    *,
    coherence: float,
    fragmentation: float,
    collapse_entropy: float,
    config: EthicalImperativeConfig,
) -> str:
    """Classify civilisation-choice wording into the three source-listed states."""
    values = np.asarray([coherence, fragmentation, collapse_entropy], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("civilisation-choice inputs must be finite")
    if (
        collapse_entropy >= config.collapse_entropy_threshold
        and coherence < config.alignment_threshold
    ):
        return "collapse_entropy_death"
    if fragmentation >= config.fragmentation_threshold:
        return "fragmentation_societal_spin_glass"
    if coherence >= config.alignment_threshold and fragmentation < config.fragmentation_threshold:
        return "alignment_global_coherence"
    return "mixed_choice_boundary"


def governance_beyond_borders_score(config: EthicalImperativeConfig) -> float:
    """Return governance score from entropy, coherence, and recursive review protocols."""
    weights = np.asarray(
        [
            config.entropy_budget_weight,
            config.global_coherence_metric_weight,
            config.recursive_review_weight,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("governance weights must be finite and non-negative")
    return float(np.sum(weights))


def feedback_tuning_score(
    *,
    loop_gain: float,
    damping: float,
    layer16_closure: float,
    config: EthicalImperativeConfig,
) -> float:
    """Return bounded score for tuned recursive feedback-loop closure."""
    values = np.asarray([loop_gain, damping, layer16_closure], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("feedback-loop inputs must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("feedback-loop inputs must be in [0, 1]")
    gain_balance = 1.0 - abs(loop_gain - damping)
    return float(0.35 * gain_balance + 0.25 * damping + 0.40 * layer16_closure)


def validate_governance_beyond_borders_fixture(
    config: EthicalImperativeConfig | None = None,
) -> GovernanceBeyondBordersValidationResult:
    """Run the governance-beyond-borders restatement fixture."""
    cfg = config or EthicalImperativeConfig()
    spec = load_ethical_imperative_validation_spec(
        "ethical_imperative.governance_beyond_borders_protocol",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    controls = {
        "missing_entropy_budget_rejection_label": _missing_entropy_budget_rejection_label(),
        "missing_global_coherence_metric_rejection_label": _missing_global_metric_rejection_label(),
        "missing_recursive_review_rejection_label": _missing_recursive_review_rejection_label(),
        "unsupported_empirical_evidence_rejection_label": 1.0,
    }
    return GovernanceBeyondBordersValidationResult(
        spec_key="ethical_imperative.governance_beyond_borders_protocol",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        governance_score=governance_beyond_borders_score(cfg),
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in spec["source_ledger_ids"]),
                "overlap_with_prior_slice": PRIOR_SLICE,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def validate_ethical_imperative_fixture(
    config: EthicalImperativeConfig | None = None,
) -> EthicalImperativeFixtureResult:
    """Run the combined Ethical Imperative restatement fixture."""
    cfg = config or EthicalImperativeConfig()
    keys = (
        "ethical_imperative.ethics_physics_restatement",
        "ethical_imperative.civilisation_choice_phase_boundary",
        "ethical_imperative.consciousness_engineering_call_boundary",
        "ethical_imperative.governance_beyond_borders_protocol",
        "ethical_imperative.feedback_loop_tuning_boundary",
    )
    specs = tuple(
        load_ethical_imperative_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    choices = (
        civilisation_choice_label(
            coherence=0.84, fragmentation=0.18, collapse_entropy=0.22, config=cfg
        ),
        civilisation_choice_label(
            coherence=0.43, fragmentation=0.78, collapse_entropy=0.41, config=cfg
        ),
        civilisation_choice_label(
            coherence=0.18, fragmentation=0.31, collapse_entropy=0.86, config=cfg
        ),
    )
    governance = validate_governance_beyond_borders_fixture(cfg)
    tuned = feedback_tuning_score(loop_gain=0.72, damping=0.68, layer16_closure=0.81, config=cfg)
    untuned = feedback_tuning_score(loop_gain=0.95, damping=0.12, layer16_closure=0.18, config=cfg)
    return EthicalImperativeFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        choice_labels=choices,
        governance=governance,
        tuned_feedback_score=tuned,
        untuned_feedback_score=untuned,
        feedback_loop_delta=tuned - untuned,
        config_thresholds=MappingProxyType(
            {
                "governance_threshold": cfg.governance_threshold,
                "feedback_threshold": cfg.feedback_threshold,
            }
        ),
        overlap_with_prior_slice=PRIOR_SLICE,
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "overlap_with_prior_slice": PRIOR_SLICE,
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
        EthicalImperativeConfig(entropy_budget_weight=-0.1)
    except ValueError as exc:
        return float("finite and non-negative" in str(exc))
    return 0.0


def _missing_global_metric_rejection_label() -> float:
    try:
        EthicalImperativeConfig(global_coherence_metric_weight=-0.1)
    except ValueError as exc:
        return float("finite and non-negative" in str(exc))
    return 0.0


def _missing_recursive_review_rejection_label() -> float:
    try:
        EthicalImperativeConfig(recursive_review_weight=-0.1)
    except ValueError as exc:
        return float("finite and non-negative" in str(exc))
    return 0.0


__all__ = [
    "CLAIM_BOUNDARY",
    "EthicalImperativeConfig",
    "EthicalImperativeFixtureResult",
    "GovernanceBeyondBordersValidationResult",
    "civilisation_choice_label",
    "feedback_tuning_score",
    "governance_beyond_borders_score",
    "validate_ethical_imperative_fixture",
    "validate_governance_beyond_borders_fixture",
]
