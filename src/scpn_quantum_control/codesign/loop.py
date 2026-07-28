# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum-classical co-design loop
"""Thin deterministic orchestration over BL-33 estimator and policy ports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .contracts import (
    CoDesignMode,
    ControllerPort,
    LoopStepInput,
    LoopStepOutput,
    ObserverInputs,
    QuantumEvaluationPort,
    StateEstimatorPort,
)
from .policies import LatencyPolicy, SafetyEnvelope


@dataclass(slots=True)
class CoDesignLoop:
    """Compose estimator, evaluator, controller, latency, and safety ports."""

    estimator: StateEstimatorPort
    evaluator: QuantumEvaluationPort
    controller: ControllerPort
    latency_policy: LatencyPolicy
    safety_envelope: SafetyEnvelope

    def step(
        self,
        step_input: LoopStepInput,
        *,
        observers: ObserverInputs | None = None,
    ) -> LoopStepOutput:
        """Run one fail-closed local simulator step without applying hardware I/O."""
        observer_inputs = observers if observers is not None else ObserverInputs()
        evaluation = self.evaluator.evaluate(step_input)
        measurement = step_input.measurement
        if step_input.mode is CoDesignMode.QUANTUM_TO_CLASSICAL and evaluation is not None:
            measurement = evaluation.order_parameter
        estimate = self.estimator.update(measurement)
        latency = self.latency_policy.assess(
            evaluation,
            apply_at_ms=step_input.apply_at_ms,
        )
        proposal = None
        if evaluation is not None:
            proposal = self.controller.propose(
                step_input.parameters,
                evaluation.gradient,
                estimate,
                step_input.target_order_parameter,
            )
        safety = self.safety_envelope.decide(
            step_input.parameters,
            evaluation,
            proposal,
            latency,
            observer_inputs,
        )
        return LoopStepOutput(
            step=step_input.step,
            estimate=estimate,
            evaluation=evaluation,
            proposal=proposal,
            latency=latency,
            safety=safety,
            observers=observer_inputs,
            mode=step_input.mode,
        )

    def run(
        self,
        inputs: Sequence[LoopStepInput],
        *,
        observers: Sequence[ObserverInputs] | None = None,
    ) -> tuple[LoopStepOutput, ...]:
        """Run a sequential simulator trace and stop after the first abort."""
        if not inputs:
            raise ValueError("inputs must contain at least one loop step")
        if observers is not None and len(observers) != len(inputs):
            raise ValueError("observer sequence length must match inputs")
        outputs: list[LoopStepOutput] = []
        for index, step_input in enumerate(inputs):
            selected_observers = None if observers is None else observers[index]
            output = self.step(step_input, observers=selected_observers)
            outputs.append(output)
            if output.safety.action.value == "abort":
                break
        return tuple(outputs)


__all__ = ["CoDesignLoop"]
