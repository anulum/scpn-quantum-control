# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — adaptive FIM feedback module
# scpn-quantum-control -- adaptive FIM feedback protocol
"""Adaptive lambda controller for FIM-feedback follow-up protocols.

This module is deliberately classical and deterministic. It converts observed
leakage/retention witnesses into the next static ``lambda_fim`` value for a
future batch or simulator pass. It does not submit hardware jobs and it does
not imply that adaptive feedback has been validated on IBM hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

FeedbackMode = Literal["leakage_suppression", "retention_recovery"]


@dataclass(frozen=True)
class AdaptiveFIMConfig:
    """Configuration for clipped proportional lambda updates."""

    lambda_min: float = 0.0
    lambda_max: float = 8.0
    step_gain: float = 1.0
    target_leakage: float = 0.0
    target_retention: float = 1.0
    deadband: float = 0.0
    mode: FeedbackMode = "leakage_suppression"

    def __post_init__(self) -> None:
        _require_finite(self.lambda_min, "lambda_min")
        _require_finite(self.lambda_max, "lambda_max")
        _require_finite(self.step_gain, "step_gain")
        _require_finite(self.target_leakage, "target_leakage")
        _require_finite(self.target_retention, "target_retention")
        _require_finite(self.deadband, "deadband")
        if self.lambda_min < 0.0:
            raise ValueError("lambda_min must be non-negative")
        if self.lambda_max < self.lambda_min:
            raise ValueError("lambda_max must be >= lambda_min")
        if self.step_gain < 0.0:
            raise ValueError("step_gain must be non-negative")
        if self.deadband < 0.0:
            raise ValueError("deadband must be non-negative")
        if self.mode not in {"leakage_suppression", "retention_recovery"}:
            raise ValueError("mode must be leakage_suppression or retention_recovery")


@dataclass(frozen=True)
class FIMWitness:
    """Observed hardware or simulator witness for one adaptive update."""

    leakage: float
    retention: float
    depth: int | None = None
    shots: int | None = None

    def __post_init__(self) -> None:
        _require_probability(self.leakage, "leakage")
        _require_probability(self.retention, "retention")
        if self.depth is not None and self.depth < 0:
            raise ValueError("depth must be non-negative when provided")
        if self.shots is not None and self.shots <= 0:
            raise ValueError("shots must be positive when provided")


@dataclass(frozen=True)
class AdaptiveFIMStep:
    """One proposed lambda update and its diagnostic metadata."""

    index: int
    lambda_in: float
    lambda_out: float
    witness: FIMWitness
    error_signal: float
    clipped: bool
    rationale: str


def propose_next_lambda(
    current_lambda: float,
    witness: FIMWitness,
    config: AdaptiveFIMConfig | None = None,
) -> AdaptiveFIMStep:
    """Return the next clipped ``lambda_fim`` value from one witness.

    ``leakage_suppression`` reduces lambda when measured leakage exceeds the
    target, matching the negative IBM FIM result: increasing lambda is not
    rewarded when leakage grows. ``retention_recovery`` reduces lambda when
    exact-state retention falls below the target.
    """
    cfg = config or AdaptiveFIMConfig()
    _require_finite(current_lambda, "current_lambda")
    if current_lambda < 0.0:
        raise ValueError("current_lambda must be non-negative")

    if cfg.mode == "leakage_suppression":
        error_signal = witness.leakage - cfg.target_leakage
        rationale = "reduce lambda when leakage is above target"
    else:
        error_signal = cfg.target_retention - witness.retention
        rationale = "reduce lambda when retention is below target"

    if abs(error_signal) <= cfg.deadband:
        proposed = current_lambda
    else:
        proposed = current_lambda - cfg.step_gain * error_signal
    lambda_out = float(np.clip(proposed, cfg.lambda_min, cfg.lambda_max))
    return AdaptiveFIMStep(
        index=0,
        lambda_in=float(current_lambda),
        lambda_out=lambda_out,
        witness=witness,
        error_signal=float(error_signal),
        clipped=not np.isclose(lambda_out, proposed),
        rationale=rationale,
    )


def adaptive_lambda_schedule(
    initial_lambda: float,
    witnesses: list[FIMWitness],
    config: AdaptiveFIMConfig | None = None,
) -> list[AdaptiveFIMStep]:
    """Generate a deterministic batch-level adaptive lambda schedule."""
    cfg = config or AdaptiveFIMConfig()
    current = float(initial_lambda)
    steps: list[AdaptiveFIMStep] = []
    for index, witness in enumerate(witnesses):
        step = propose_next_lambda(current, witness, cfg)
        steps.append(
            AdaptiveFIMStep(
                index=index,
                lambda_in=step.lambda_in,
                lambda_out=step.lambda_out,
                witness=step.witness,
                error_signal=step.error_signal,
                clipped=step.clipped,
                rationale=step.rationale,
            )
        )
        current = step.lambda_out
    return steps


def _require_finite(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_probability(value: float, name: str) -> None:
    _require_finite(value, name)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
