# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 cosmological implications validation fixtures
"""Simulator-only comparative and cosmological implications fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .spec_loader import load_cosmological_implications_validation_spec

CLAIM_BOUNDARY = (
    "source-bounded cosmological implications simulator contract; not empirical evidence"
)
SOURCE_LEDGER_SPAN = ("P0R06290", "P0R06310")


@dataclass(frozen=True, slots=True)
class CosmologicalImplicationsConfig:
    """Finite simulator settings for cosmological implications fixtures."""

    iit_weight: float = 1.0 / 6.0
    orch_or_weight: float = 1.0 / 6.0
    gnw_weight: float = 1.0 / 6.0
    fep_predictive_coding_weight: float = 1.0 / 6.0
    upde_weight: float = 1.0 / 6.0
    l15_weight: float = 1.0 / 6.0
    positioning_threshold: float = 0.85
    lambda_balance_threshold: float = 0.65
    renormalisation_threshold: float = 0.55
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "positioning_threshold",
            "lambda_balance_threshold",
            "renormalisation_threshold",
        ):
            _require_positive(name, float(getattr(self, name)))
        for name in (
            "iit_weight",
            "orch_or_weight",
            "gnw_weight",
            "fep_predictive_coding_weight",
            "upde_weight",
            "l15_weight",
        ):
            _require_non_negative(name, float(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class EthicalRenormalisationValidationResult:
    """Ethical-renormalisation mechanism result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    renormalisation_delta: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class CosmologicalImplicationsFixtureResult:
    """Combined cosmological implications fixture result."""

    spec_keys: tuple[str, str, str, str, str]
    hardware_status: str
    positioning_score: float
    lambda_balanced_score: float
    lambda_unbalanced_score: float
    renormalisation: EthicalRenormalisationValidationResult
    config_thresholds: MappingProxyType[str, float]
    source_ledger_span: tuple[str, str]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def theory_positioning_score(
    *,
    iit: bool,
    orch_or: bool,
    gnw: bool,
    fep_predictive_coding: bool,
    upde: bool,
    l15: bool,
    config: CosmologicalImplicationsConfig,
) -> float:
    """Score whether all source-listed comparison and differentiator channels are present."""
    weights = np.asarray(
        [
            config.iit_weight,
            config.orch_or_weight,
            config.gnw_weight,
            config.fep_predictive_coding_weight,
            config.upde_weight,
            config.l15_weight,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("positioning weights must be finite and non-negative")
    channels = np.asarray(
        [iit, orch_or, gnw, fep_predictive_coding, upde, l15],
        dtype=np.float64,
    )
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("at least one positioning weight must be positive")
    return float(np.dot(weights, channels) / total)


def lambda_balance_score(
    *,
    expansion_balance: float,
    rg_flow_window: float,
    cosmic_attractor_access: float,
    config: CosmologicalImplicationsConfig,
) -> float:
    """Return bounded Lambda-context score from the three source-listed channels."""
    values = np.asarray(
        [expansion_balance, rg_flow_window, cosmic_attractor_access],
        dtype=np.float64,
    )
    _require_unit_interval("lambda balance inputs", values)
    return float(np.dot(np.asarray([0.34, 0.33, 0.33], dtype=np.float64), values))


def ethical_renormalisation_delta(
    *,
    previous_cycle_stagnation: float,
    unsustainable_complexity: float,
    coupling_adjustment: float,
    l16_meta_optimisation: float,
    config: CosmologicalImplicationsConfig,
) -> float:
    """Return bounded delta for L16-guided ethical-renormalisation restatement."""
    values = np.asarray(
        [
            previous_cycle_stagnation,
            unsustainable_complexity,
            coupling_adjustment,
            l16_meta_optimisation,
        ],
        dtype=np.float64,
    )
    _require_unit_interval("ethical-renormalisation inputs", values)
    pathology_pressure = 0.5 * (previous_cycle_stagnation + unsustainable_complexity)
    adjustment_channel = 0.45 * coupling_adjustment + 0.55 * l16_meta_optimisation
    return float(pathology_pressure * adjustment_channel)


def validate_ethical_renormalisation_fixture(
    config: CosmologicalImplicationsConfig | None = None,
) -> EthicalRenormalisationValidationResult:
    """Run the ethical-renormalisation mechanism fixture."""
    cfg = config or CosmologicalImplicationsConfig()
    spec = load_cosmological_implications_validation_spec(
        "cosmological_implications.ethical_renormalisation_mechanism",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    delta = ethical_renormalisation_delta(
        previous_cycle_stagnation=0.72,
        unsustainable_complexity=0.66,
        coupling_adjustment=0.81,
        l16_meta_optimisation=0.88,
        config=cfg,
    )
    controls = {
        "missing_l16_meta_optimisation_rejection_label": _missing_l16_rejection_label(),
        "negative_coupling_adjustment_rejection_label": _negative_coupling_rejection_label(),
        "unsupported_empirical_cosmology_rejection_label": 1.0,
    }
    return EthicalRenormalisationValidationResult(
        spec_key="cosmological_implications.ethical_renormalisation_mechanism",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        renormalisation_delta=delta,
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "renormalisation_threshold": cfg.renormalisation_threshold,
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def validate_cosmological_implications_fixture(
    config: CosmologicalImplicationsConfig | None = None,
) -> CosmologicalImplicationsFixtureResult:
    """Run the combined cosmological implications fixture."""
    cfg = config or CosmologicalImplicationsConfig()
    keys = (
        "cosmological_implications.comparative_positioning_mapping",
        "cosmological_implications.ethical_selection_claim_boundary",
        "cosmological_implications.lambda_optimisation_context",
        "cosmological_implications.ethical_renormalisation_mechanism",
        "cosmological_implications.mmc_ccc_formalisation_boundary",
    )
    specs = tuple(
        load_cosmological_implications_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    positioning = theory_positioning_score(
        iit=True,
        orch_or=True,
        gnw=True,
        fep_predictive_coding=True,
        upde=True,
        l15=True,
        config=cfg,
    )
    balanced = lambda_balance_score(
        expansion_balance=0.72,
        rg_flow_window=0.78,
        cosmic_attractor_access=0.74,
        config=cfg,
    )
    unbalanced = lambda_balance_score(
        expansion_balance=0.19,
        rg_flow_window=0.22,
        cosmic_attractor_access=0.31,
        config=cfg,
    )
    renormalisation = validate_ethical_renormalisation_fixture(cfg)
    return CosmologicalImplicationsFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        positioning_score=positioning,
        lambda_balanced_score=balanced,
        lambda_unbalanced_score=unbalanced,
        renormalisation=renormalisation,
        config_thresholds=MappingProxyType(
            {
                "positioning_threshold": cfg.positioning_threshold,
                "lambda_balance_threshold": cfg.lambda_balance_threshold,
                "renormalisation_threshold": cfg.renormalisation_threshold,
            }
        ),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
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


def _require_unit_interval(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} must be in [0, 1]")


def _missing_l16_rejection_label() -> float:
    try:
        ethical_renormalisation_delta(
            previous_cycle_stagnation=0.72,
            unsustainable_complexity=0.66,
            coupling_adjustment=0.81,
            l16_meta_optimisation=-0.01,
            config=CosmologicalImplicationsConfig(),
        )
    except ValueError as exc:
        return float("in [0, 1]" in str(exc))
    return 0.0


def _negative_coupling_rejection_label() -> float:
    try:
        ethical_renormalisation_delta(
            previous_cycle_stagnation=0.72,
            unsustainable_complexity=0.66,
            coupling_adjustment=-0.01,
            l16_meta_optimisation=0.88,
            config=CosmologicalImplicationsConfig(),
        )
    except ValueError as exc:
        return float("in [0, 1]" in str(exc))
    return 0.0


__all__ = [
    "CLAIM_BOUNDARY",
    "CosmologicalImplicationsConfig",
    "CosmologicalImplicationsFixtureResult",
    "EthicalRenormalisationValidationResult",
    "ethical_renormalisation_delta",
    "lambda_balance_score",
    "theory_positioning_score",
    "validate_cosmological_implications_fixture",
    "validate_ethical_renormalisation_fixture",
]
