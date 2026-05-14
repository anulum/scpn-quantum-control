# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 glial slow-control validation fixtures
"""Simulator-only glial slow-control fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .spec_loader import load_glial_slow_control_validation_spec

CLAIM_BOUNDARY = "source-bounded glial slow-control simulator contract; not empirical evidence"
SOURCE_LEDGER_SPAN = ("P0R06414", "P0R06433")


@dataclass(frozen=True, slots=True)
class GlialSlowControlConfig:
    """Finite simulator settings for glial slow-control fixtures."""

    stability_threshold: float = 0.70
    decoupling_threshold: float = 0.45
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_positive("stability_threshold", self.stability_threshold)
        _require_positive("decoupling_threshold", self.decoupling_threshold)


@dataclass(frozen=True, slots=True)
class GlialSlowControlFixtureResult:
    """Combined glial slow-control fixture result."""

    spec_keys: tuple[str, str, str, str]
    hardware_status: str
    stability_score: float
    protocol_completeness: float
    decoupling_score: float
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float]
    source_ledger_span: tuple[str, str]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def glial_feedback_stability_score(
    *,
    fast_loop_criticality: float,
    slow_ca_integration: float,
    gliotransmitter_feedback: float,
    excitability_control: float,
) -> float:
    """Score source-bounded two-timescale glial feedback support."""
    values = _unit_interval_values(
        fast_loop_criticality,
        slow_ca_integration,
        gliotransmitter_feedback,
        excitability_control,
    )
    return float(np.prod(values) ** 0.25)


def protocol_completeness_score(
    *,
    preparation: bool,
    simultaneous_recording: bool,
    avalanche_analysis: bool,
    causal_block: bool,
) -> float:
    """Return completeness over the four source-listed experimental protocol steps."""
    return float(
        np.mean(
            np.asarray([preparation, simultaneous_recording, avalanche_analysis, causal_block])
        )
    )


def causal_decoupling_score(
    *, baseline_correlation: float, post_block_correlation: float
) -> float:
    """Return absolute correlation drop after gliotransmission blockade."""
    values = np.asarray([baseline_correlation, post_block_correlation], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("correlations must be in [-1, 1]")
    return float(abs(baseline_correlation - post_block_correlation))


def validate_glial_slow_control_fixture(
    config: GlialSlowControlConfig | None = None,
) -> GlialSlowControlFixtureResult:
    """Run the combined glial slow-control fixture."""
    cfg = config or GlialSlowControlConfig()
    keys = (
        "glial_slow_control.two_timescale_governor",
        "glial_slow_control.homeostatic_feedback_channels",
        "glial_slow_control.experimental_protocol_catalogue",
        "glial_slow_control.falsification_and_causal_decoupling",
    )
    specs = tuple(
        load_glial_slow_control_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    stability = glial_feedback_stability_score(
        fast_loop_criticality=0.84,
        slow_ca_integration=0.82,
        gliotransmitter_feedback=0.8,
        excitability_control=0.78,
    )
    completeness = protocol_completeness_score(
        preparation=True,
        simultaneous_recording=True,
        avalanche_analysis=True,
        causal_block=True,
    )
    decoupling = causal_decoupling_score(
        baseline_correlation=0.72,
        post_block_correlation=0.18,
    )
    controls = {
        "missing_slow_feedback_rejection_label": float(
            glial_feedback_stability_score(
                fast_loop_criticality=0.84,
                slow_ca_integration=0.0,
                gliotransmitter_feedback=0.8,
                excitability_control=0.78,
            )
            < cfg.stability_threshold
        ),
        "incomplete_protocol_rejection_label": float(
            protocol_completeness_score(
                preparation=True,
                simultaneous_recording=True,
                avalanche_analysis=False,
                causal_block=False,
            )
            < 1.0
        ),
        "unsupported_empirical_evidence_rejection_label": 1.0,
    }
    return GlialSlowControlFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        stability_score=stability,
        protocol_completeness=completeness,
        decoupling_score=decoupling,
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType(
            {
                "stability_threshold": cfg.stability_threshold,
                "decoupling_threshold": cfg.decoupling_threshold,
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
        raise ValueError("glial slow-control inputs must be in [0, 1]")
    return array


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


__all__ = [
    "CLAIM_BOUNDARY",
    "GlialSlowControlConfig",
    "GlialSlowControlFixtureResult",
    "causal_decoupling_score",
    "glial_feedback_stability_score",
    "protocol_completeness_score",
    "validate_glial_slow_control_fixture",
]
