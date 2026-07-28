# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — co-design latency and safety policies
"""Fail-closed latency and controller-envelope policies for BL-33."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .contracts import (
    ControllerProposal,
    LatencyDecision,
    ObserverInputs,
    QuantumEvaluation,
    SafetyAction,
    SafetyDecision,
    StaleGradientAction,
)


@dataclass(frozen=True, slots=True)
class LatencyPolicy:
    """Maximum gradient age and fail-closed missing/stale actions."""

    max_age_ms: float
    on_stale: StaleGradientAction = StaleGradientAction.HOLD
    on_missing: StaleGradientAction = StaleGradientAction.ABORT

    def __post_init__(self) -> None:
        """Validate a positive finite gradient-age ceiling."""
        if not np.isfinite(self.max_age_ms) or self.max_age_ms <= 0.0:
            raise ValueError("max_age_ms must be finite and positive")

    def assess(
        self,
        evaluation: QuantumEvaluation | None,
        *,
        apply_at_ms: float,
    ) -> LatencyDecision:
        """Classify a gradient as current, stale, missing, or time-invalid."""
        if not np.isfinite(apply_at_ms):
            raise ValueError("apply_at_ms must be finite")
        if evaluation is None:
            return LatencyDecision(
                stale=False,
                missing=True,
                age_ms=None,
                action=_safety_action(self.on_missing),
                reason=f"gradient missing; policy={self.on_missing.value}",
            )
        age_ms = float(apply_at_ms - evaluation.evaluated_at_ms)
        if age_ms < 0.0:
            return LatencyDecision(
                stale=True,
                missing=False,
                age_ms=age_ms,
                action=SafetyAction.ABORT,
                reason="gradient evaluation timestamp is in the logical future",
            )
        if age_ms > self.max_age_ms:
            return LatencyDecision(
                stale=True,
                missing=False,
                age_ms=age_ms,
                action=_safety_action(self.on_stale),
                reason=(
                    f"gradient age {age_ms:.6f} ms exceeds policy ceiling "
                    f"{self.max_age_ms:.6f} ms; policy={self.on_stale.value}"
                ),
            )
        return LatencyDecision(
            stale=False,
            missing=False,
            age_ms=age_ms,
            action=SafetyAction.ALLOW,
            reason="gradient age is within the configured simulator ceiling",
        )


@dataclass(frozen=True, slots=True)
class SafetyEnvelope:
    """Bound parameter, update, gradient, latency, and observer safety inputs."""

    max_abs_parameter: float
    max_update_norm: float
    max_gradient_norm: float

    def __post_init__(self) -> None:
        """Validate strictly positive finite safety ceilings."""
        values = (
            self.max_abs_parameter,
            self.max_update_norm,
            self.max_gradient_norm,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("safety ceilings must be finite and positive")

    def decide(
        self,
        current_parameters: tuple[float, ...],
        evaluation: QuantumEvaluation | None,
        proposal: ControllerProposal | None,
        latency: LatencyDecision,
        observers: ObserverInputs,
    ) -> SafetyDecision:
        """Apply observer, latency, gradient, update, and parameter interlocks."""
        current = _finite_vector("current_parameters", current_parameters)
        observer_decision = _observer_interlock(current, observers)
        if observer_decision is not None:
            return observer_decision
        if latency.action in {SafetyAction.HOLD, SafetyAction.ABORT}:
            return SafetyDecision(
                action=latency.action,
                reason=latency.reason,
                applied_parameters=tuple(float(value) for value in current),
                blockers=("latency_policy",),
            )
        if evaluation is None or proposal is None:
            return SafetyDecision(
                action=SafetyAction.ABORT,
                reason="evaluation or controller proposal absent after latency approval",
                applied_parameters=tuple(float(value) for value in current),
                blockers=("incomplete_loop_step",),
            )

        gradient = _finite_vector("gradient", evaluation.gradient)
        if gradient.shape != current.shape:
            return SafetyDecision(
                action=SafetyAction.ABORT,
                reason="gradient dimension does not match the current parameters",
                applied_parameters=tuple(float(value) for value in current),
                blockers=("gradient_dimension",),
            )
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > self.max_gradient_norm:
            return SafetyDecision(
                action=SafetyAction.ABORT,
                reason=(
                    f"gradient norm {gradient_norm:.6f} exceeds safety ceiling "
                    f"{self.max_gradient_norm:.6f}"
                ),
                applied_parameters=tuple(float(value) for value in current),
                blockers=("gradient_envelope",),
            )

        proposed = _finite_vector("proposal.parameters", proposal.parameters)
        update = _finite_vector("proposal.update", proposal.update)
        if proposed.shape != current.shape or update.shape != current.shape:
            return SafetyDecision(
                action=SafetyAction.ABORT,
                reason="controller proposal dimension does not match current parameters",
                applied_parameters=tuple(float(value) for value in current),
                blockers=("controller_dimension",),
            )

        applied = proposed.copy()
        blockers: list[str] = []
        update_norm = float(np.linalg.norm(update))
        if update_norm > self.max_update_norm:
            applied = current + update * (self.max_update_norm / update_norm)
            blockers.append("update_norm_clamped")
        bounded = np.clip(applied, -self.max_abs_parameter, self.max_abs_parameter)
        if not np.array_equal(bounded, applied):
            applied = bounded
            blockers.append("parameter_envelope_clamped")
        action = SafetyAction.CLAMP if blockers else SafetyAction.ALLOW
        reason = (
            "; ".join(blockers)
            if blockers
            else "proposal is within gradient, update, and parameter envelopes"
        )
        return SafetyDecision(
            action=action,
            reason=reason,
            applied_parameters=tuple(float(value) for value in applied),
            blockers=tuple(blockers),
        )


def _observer_interlock(
    current: NDArray[np.float64],
    observers: ObserverInputs,
) -> SafetyDecision | None:
    if observers.identity_action in {"hold", "abort"}:
        action = SafetyAction.HOLD if observers.identity_action == "hold" else SafetyAction.ABORT
        reason = observers.identity_reason or f"identity observer requested {action.value}"
        return SafetyDecision(
            action=action,
            reason=reason,
            applied_parameters=tuple(float(value) for value in current),
            blockers=("identity_observer",),
        )
    if observers.l16_action in {"adjust", "halt"}:
        action = SafetyAction.HOLD if observers.l16_action == "adjust" else SafetyAction.ABORT
        reason = observers.l16_reason or f"L16 heuristic requested {observers.l16_action}"
        return SafetyDecision(
            action=action,
            reason=reason,
            applied_parameters=tuple(float(value) for value in current),
            blockers=("l16_director",),
        )
    return None


def _finite_vector(name: str, values: tuple[float, ...]) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return vector


def _safety_action(action: StaleGradientAction) -> SafetyAction:
    return SafetyAction.HOLD if action is StaleGradientAction.HOLD else SafetyAction.ABORT


__all__ = ["LatencyPolicy", "SafetyEnvelope"]
