# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum-classical co-design components
"""Deterministic estimator, evaluator, and controller components for co-design."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from ..control.closed_loop_analysis import (
    ClosedLoopExecutionPolicy,
    ExecutionMode,
    evaluate_closed_loop_policy,
)
from ..phase.gradient_backend import explain_quantum_gradient_method
from ..phase.open_system_objectives import (
    BoundedOpenSystemObjectiveCase,
    run_open_system_objective_suite,
)
from ..phase.synchronisation_objectives import (
    build_synchronisation_objective,
    kuramoto_order_parameter,
)
from .contracts import (
    CODESIGN_CLAIM_BOUNDARY,
    BackendCapabilities,
    ControllerProposal,
    GradientPlanRecord,
    LoopStepInput,
    QuantumEvaluation,
    StateEstimate,
)

OpenSystemBackend = Literal["lindblad_density", "mcwf_ensemble"]


@dataclass(slots=True)
class ExponentialOrderEstimator:
    """Exponentially weighted estimator for scalar order-parameter samples.

    Parameters
    ----------
    alpha
        Weight applied to the newest sample in ``(0, 1]``.
    initial_order_parameter
        Optional prior estimate in ``[0, 1]``. When absent, the first sample
        initialises the estimator without a fabricated prior.

    """

    alpha: float
    initial_order_parameter: float | None = None
    _estimate: float | None = field(init=False, repr=False)
    _sample_count: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        """Validate estimator configuration and initialise private state."""
        if not np.isfinite(self.alpha) or not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be finite and in (0, 1]")
        if self.initial_order_parameter is not None and not (
            np.isfinite(self.initial_order_parameter)
            and 0.0 <= self.initial_order_parameter <= 1.0
        ):
            raise ValueError("initial_order_parameter must be in [0, 1]")
        self._estimate = self.initial_order_parameter
        self._sample_count = 0

    def update(self, measurement: float) -> StateEstimate:
        """Update the estimate from one finite sample in ``[0, 1]``."""
        if not np.isfinite(measurement) or not 0.0 <= measurement <= 1.0:
            raise ValueError("measurement must be finite and in [0, 1]")
        previous = self._estimate
        estimate = (
            float(measurement)
            if previous is None
            else float(self.alpha * measurement + (1.0 - self.alpha) * previous)
        )
        innovation = float(measurement - (measurement if previous is None else previous))
        sample_count = self._sample_count + 1
        self._estimate = estimate
        self._sample_count = sample_count
        return StateEstimate(
            order_parameter=estimate,
            innovation=innovation,
            sample_count=sample_count,
        )


@dataclass(frozen=True, slots=True)
class OpenSystemObjectiveConfig:
    """Optional open-system objective augmentation over a bounded published case."""

    case: BoundedOpenSystemObjectiveCase
    backend: OpenSystemBackend
    weight: float

    def __post_init__(self) -> None:
        """Validate the non-negative finite augmentation weight."""
        if not np.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("open-system weight must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PhaseObjectiveSimulator:
    """Policy-gated local evaluator over existing differentiable phase objectives.

    The primary objective is built by
    :func:`scpn_quantum_control.phase.synchronisation_objectives.build_synchronisation_objective`.
    An optional bounded open-system record may be added with an explicit
    weight. Logical latency is configured rather than measured so replay bytes
    remain stable.
    """

    policy: ClosedLoopExecutionPolicy
    capabilities: BackendCapabilities = BackendCapabilities(
        backend_id="statevector",
        supports_mid_circuit=False,
        supports_adjoint=False,
        max_latency_ms=100.0,
    )
    logical_latency_ms: float = 1.0
    planner_method: str = "auto"
    open_system: OpenSystemObjectiveConfig | None = None
    missing_gradient_steps: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        """Refuse hardware capabilities and invalid logical latency."""
        if self.capabilities.hardware:
            raise ValueError("co-design evaluator capabilities must remain simulator-only")
        if not np.isfinite(self.logical_latency_ms) or self.logical_latency_ms < 0.0:
            raise ValueError("logical_latency_ms must be finite and non-negative")
        if self.logical_latency_ms > self.capabilities.max_latency_ms:
            raise ValueError("logical latency exceeds the declared backend capability")
        if any(step < 0 for step in self.missing_gradient_steps):
            raise ValueError("missing_gradient_steps must be non-negative")

    def evaluate(self, step_input: LoopStepInput) -> QuantumEvaluation | None:
        """Evaluate one exact local phase objective or model a missing update."""
        decision = evaluate_closed_loop_policy(
            self.policy,
            backend=self.capabilities.backend_id,
            requested_rounds=1,
        )
        if decision.mode is not ExecutionMode.SIMULATION:
            raise PermissionError("co-design evaluation refuses hardware execution")
        if step_input.step in self.missing_gradient_steps:
            return None

        objective = build_synchronisation_objective(
            len(step_input.parameters),
            order_parameter_target=step_input.target_order_parameter,
        )
        evaluated = objective.evaluate(np.asarray(step_input.parameters, dtype=np.float64))
        value = evaluated.value
        gradient = np.asarray(evaluated.gradient, dtype=np.float64)
        terms = list(objective.term_names)
        open_system_backend: str | None = None
        if self.open_system is not None and self.open_system.weight > 0.0:
            if self.open_system.case.initial_params.size != gradient.size:
                raise ValueError("open-system objective width does not match loop parameters")
            runtime_case = replace(
                self.open_system.case,
                initial_params=np.asarray(step_input.parameters, dtype=np.float64),
            )
            suite = run_open_system_objective_suite(
                (runtime_case,),
                backends=(self.open_system.backend,),
                include_boundary_rows=False,
            )
            record = suite.records[0]
            value += self.open_system.weight * record.value
            gradient += self.open_system.weight * np.asarray(record.gradient, dtype=np.float64)
            terms.append(f"open_system:{record.backend}")
            open_system_backend = record.backend

        explanation = explain_quantum_gradient_method(
            self.capabilities.backend_id,
            n_params=len(step_input.parameters),
            method=self.planner_method,
            allow_hardware=False,
        )
        plan = explanation.selected_plan
        if not explanation.supported:
            raise RuntimeError("governed gradient planner did not produce a local supported plan")
        plan_record = GradientPlanRecord(
            backend=plan.backend,
            requested_method=explanation.requested_method,
            selected_method=explanation.selected_method,
            supported=explanation.supported,
            evaluations=plan.evaluations,
            requires_hardware_approval=plan.requires_hardware_approval,
            reasons=plan.reasons,
            claim_boundary=explanation.claim_boundary,
        )
        return QuantumEvaluation(
            objective_value=float(value),
            gradient=tuple(float(item) for item in gradient),
            order_parameter=kuramoto_order_parameter(step_input.parameters),
            evaluated_at_ms=step_input.observed_at_ms + self.logical_latency_ms,
            backend_status="local_simulator",
            capabilities=self.capabilities,
            gradient_plan=plan_record,
            objective_terms=tuple(terms),
            open_system_backend=open_system_backend,
        )


@dataclass(frozen=True, slots=True)
class GradientFeedbackController:
    """Baseline gradient controller with estimator-dependent gain scheduling."""

    learning_rate: float
    feedback_gain: float = 0.0

    def __post_init__(self) -> None:
        """Validate finite non-negative controller gains."""
        values = (self.learning_rate, self.feedback_gain)
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("controller gains must be finite and non-negative")
        if self.learning_rate == 0.0:
            raise ValueError("learning_rate must be positive")

    def propose(
        self,
        parameters: tuple[float, ...],
        gradient: tuple[float, ...],
        estimate: StateEstimate,
        target_order_parameter: float,
    ) -> ControllerProposal:
        """Propose a bounded-input gradient update without applying it."""
        current = np.asarray(parameters, dtype=np.float64)
        direction = np.asarray(gradient, dtype=np.float64)
        if current.ndim != 1 or current.size == 0 or direction.shape != current.shape:
            raise ValueError("parameters and gradient must be matching non-empty vectors")
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(direction)):
            raise ValueError("parameters and gradient must be finite")
        if not np.isfinite(target_order_parameter) or not 0.0 <= target_order_parameter <= 1.0:
            raise ValueError("target_order_parameter must be in [0, 1]")
        gain_scale = 1.0 + self.feedback_gain * abs(
            target_order_parameter - estimate.order_parameter
        )
        update = -self.learning_rate * gain_scale * direction
        proposed = current + update
        return ControllerProposal(
            parameters=tuple(float(value) for value in proposed),
            update=tuple(float(value) for value in update),
            gain_scale=float(gain_scale),
        )


def component_claim_boundary() -> str:
    """Return the shared co-design claim boundary for component registries."""
    return CODESIGN_CLAIM_BOUNDARY


__all__ = [
    "ExponentialOrderEstimator",
    "GradientFeedbackController",
    "OpenSystemObjectiveConfig",
    "PhaseObjectiveSimulator",
    "component_claim_boundary",
]
