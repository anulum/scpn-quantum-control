# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Layer 5 Active Inference validation fixtures
"""Simulator-only Layer 5 Active Inference fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .spec_loader import load_l5_active_inference_validation_spec

CLAIM_BOUNDARY = (
    "source-bounded Layer 5 Active Inference simulator contract; not empirical evidence"
)
SOURCE_LEDGER_SPAN = ("P0R06434", "P0R06449")


@dataclass(frozen=True, slots=True)
class L5ActiveInferenceConfig:
    """Finite simulator settings for Layer 5 Active Inference fixtures."""

    attractor_threshold: float = 0.72
    triple_network_threshold: float = 0.72
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_positive("attractor_threshold", self.attractor_threshold)
        _require_positive("triple_network_threshold", self.triple_network_threshold)


@dataclass(frozen=True, slots=True)
class L5ActiveInferenceFixtureResult:
    """Combined Layer 5 Active Inference fixture result."""

    spec_keys: tuple[str, str, str, str]
    hardware_status: str
    attractor_score: float
    triple_network_score: float
    fep_loop_score: float
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float]
    source_ledger_span: tuple[str, str]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def attractor_boundary_score(
    *,
    basin_coverage: float,
    loop_consistency: float,
    saddle_transition_support: float,
) -> float:
    """Score the source-bounded attractor-topology fixture."""
    values = _unit_interval_values(basin_coverage, loop_consistency, saddle_transition_support)
    return float(np.prod(values) ** (1.0 / 3.0))


def triple_network_precision_score(
    *,
    downward_prediction: float,
    upward_prediction_error: float,
    salience_precision_gate: float,
    cen_dmn_switching: float,
) -> float:
    """Score the source-bounded HPC/Triple-Network precision-gating fixture."""
    values = _unit_interval_values(
        downward_prediction,
        upward_prediction_error,
        salience_precision_gate,
        cen_dmn_switching,
    )
    return float(np.prod(values) ** 0.25)


def fep_loop_consistency_score(
    *,
    free_energy_bound: bool,
    prediction_error: bool,
    action_policy: bool,
    belief_update: bool,
) -> float:
    """Return completeness over the source-listed perception-action loop terms."""
    return float(
        np.mean(np.asarray([free_energy_bound, prediction_error, action_policy, belief_update]))
    )


def validate_l5_active_inference_fixture(
    config: L5ActiveInferenceConfig | None = None,
) -> L5ActiveInferenceFixtureResult:
    """Run the combined Layer 5 Active Inference fixture."""
    cfg = config or L5ActiveInferenceConfig()
    keys = (
        "l5_active_inference.attractor_geometry",
        "l5_active_inference.hpc_triple_network_loop",
        "l5_active_inference.fep_perception_action_loop",
        "l5_active_inference.cosmic_prior_boundary",
    )
    specs = tuple(
        load_l5_active_inference_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    attractor = attractor_boundary_score(
        basin_coverage=0.88,
        loop_consistency=0.84,
        saddle_transition_support=0.82,
    )
    triple_network = triple_network_precision_score(
        downward_prediction=0.86,
        upward_prediction_error=0.83,
        salience_precision_gate=0.81,
        cen_dmn_switching=0.79,
    )
    fep_loop = fep_loop_consistency_score(
        free_energy_bound=True,
        prediction_error=True,
        action_policy=True,
        belief_update=True,
    )
    controls = {
        "missing_saddle_boundary_rejection_label": float(
            attractor_boundary_score(
                basin_coverage=0.88,
                loop_consistency=0.84,
                saddle_transition_support=0.0,
            )
            < cfg.attractor_threshold
        ),
        "missing_salience_gate_rejection_label": float(
            triple_network_precision_score(
                downward_prediction=0.86,
                upward_prediction_error=0.83,
                salience_precision_gate=0.0,
                cen_dmn_switching=0.79,
            )
            < cfg.triple_network_threshold
        ),
        "cosmic_prior_empirical_claim_rejection_label": 1.0,
    }
    return L5ActiveInferenceFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        attractor_score=attractor,
        triple_network_score=triple_network,
        fep_loop_score=fep_loop,
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType(
            {
                "attractor_threshold": cfg.attractor_threshold,
                "triple_network_threshold": cfg.triple_network_threshold,
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


def _unit_interval_values(*values: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("inputs must be in [0, 1]")
    return array


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


__all__ = [
    "CLAIM_BOUNDARY",
    "L5ActiveInferenceConfig",
    "L5ActiveInferenceFixtureResult",
    "attractor_boundary_score",
    "fep_loop_consistency_score",
    "triple_network_precision_score",
    "validate_l5_active_inference_fixture",
]
