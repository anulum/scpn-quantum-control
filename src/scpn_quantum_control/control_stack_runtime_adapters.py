# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Policy-gated adapters over the existing control stack
"""Executable control-stack adapters over existing control and co-simulation surfaces.

The adapters in this module are deliberately thin. They accept the ambient
controllers through typed ports, apply the existing closed-loop execution
policy before doing work, and return immutable records. They never submit to
hardware and do not create a second realtime-control implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .control.closed_loop_analysis import (
    ClosedLoopExecutionDecision,
    ClosedLoopExecutionPolicy,
    ExecutionMode,
    evaluate_closed_loop_policy,
)
from .control.realtime_feedback import FeedbackStep
from .cosimulation.knm_partition import KnmPartition
from .cosimulation.quantum_classical import CoSimulationResult, cosimulate

CONTROL_STACK_RUNTIME_ADAPTER_SCHEMA = "control_stack_runtime_adapters.v2"
CONTROL_STACK_RUNTIME_ADAPTER_CLAIM_BOUNDARY = (
    "local simulator adapters over existing control/* and cosimulation modules; "
    "ClosedLoopExecutionPolicy is mandatory; no hardware submission, PCS claim, "
    "realtime_runtime rewrite, or pulse execution"
)


class PolicyGatedAdapterError(RuntimeError):
    """Raised before adapter execution when policy does not authorise local work."""


class RealtimeFeedbackPort(Protocol):
    """Existing realtime-feedback controller surface adapted by this module."""

    def run(self, n_steps: int, seed: int | None = None) -> list[FeedbackStep]:
        """Run existing feedback steps without changing controller ownership."""
        ...


class QaoaMpcPort(Protocol):
    """Existing QAOA-MPC optimisation surface adapted by this module."""

    def optimize(self, seed: int | None = None) -> NDArray[np.int64]:
        """Return the existing controller's binary action schedule."""
        ...


@dataclass(frozen=True, slots=True)
class RealtimeFeedbackAdapterResult:
    """Policy decision and steps from the ambient realtime controller."""

    decision: ClosedLoopExecutionDecision
    steps: tuple[FeedbackStep, ...]
    schema: str = CONTROL_STACK_RUNTIME_ADAPTER_SCHEMA
    claim_boundary: str = CONTROL_STACK_RUNTIME_ADAPTER_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Reject stale schemas and altered claim boundaries."""
        _validate_result_contract(self.schema, self.claim_boundary)


@dataclass(frozen=True, slots=True)
class QaoaMpcAdapterResult:
    """Policy decision and binary schedule from the ambient QAOA-MPC controller."""

    decision: ClosedLoopExecutionDecision
    actions: tuple[int, ...]
    schema: str = CONTROL_STACK_RUNTIME_ADAPTER_SCHEMA
    claim_boundary: str = CONTROL_STACK_RUNTIME_ADAPTER_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Reject stale schemas and altered claim boundaries."""
        _validate_result_contract(self.schema, self.claim_boundary)


@dataclass(frozen=True, slots=True)
class CosimulationPartitionTelemetry:
    """Compact telemetry mapped from the ambient mean-field partition result."""

    n_quantum: int
    n_classical: int
    samples: int
    final_quantum_order: float
    final_classical_order: float
    final_global_order: float
    baseline_deviation: float
    cross_fraction: float
    claim_boundary: str


@dataclass(frozen=True, slots=True)
class CosimulationPartitionAdapterResult:
    """Policy decision, ambient result, and mapped partition telemetry."""

    decision: ClosedLoopExecutionDecision
    result: CoSimulationResult
    telemetry: CosimulationPartitionTelemetry
    schema: str = CONTROL_STACK_RUNTIME_ADAPTER_SCHEMA
    claim_boundary: str = CONTROL_STACK_RUNTIME_ADAPTER_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Reject stale schemas and altered claim boundaries."""
        _validate_result_contract(self.schema, self.claim_boundary)


@dataclass(frozen=True, slots=True)
class PulseComposeBoundaryDecision:
    """Fail-closed hand-off to the optional pulse-execution product."""

    allowed: bool
    reason: str
    dependency: str
    schema: str = CONTROL_STACK_RUNTIME_ADAPTER_SCHEMA
    claim_boundary: str = CONTROL_STACK_RUNTIME_ADAPTER_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Reject stale schemas and altered claim boundaries."""
        _validate_result_contract(self.schema, self.claim_boundary)


def _validate_result_contract(schema: str, claim_boundary: str) -> None:
    """Require the current exact serialized adapter contract."""
    if schema != CONTROL_STACK_RUNTIME_ADAPTER_SCHEMA:
        raise ValueError("unexpected control-stack runtime-adapter schema")
    if claim_boundary != CONTROL_STACK_RUNTIME_ADAPTER_CLAIM_BOUNDARY:
        raise ValueError("unexpected control-stack runtime-adapter claim boundary")


def _authorise_local_adapter(
    policy: ClosedLoopExecutionPolicy | None,
    *,
    requested_rounds: int,
    backend: str | None,
) -> ClosedLoopExecutionDecision:
    """Require an explicit policy and authorise local simulator execution."""
    if policy is None:
        raise PolicyGatedAdapterError(
            "ClosedLoopExecutionPolicy is required before adapter evaluation"
        )
    decision = evaluate_closed_loop_policy(
        policy,
        backend=backend,
        requested_rounds=requested_rounds,
    )
    if not decision.authorised:
        raise PolicyGatedAdapterError(f"execution policy refused adapter: {decision.reason}")
    if decision.mode is not ExecutionMode.SIMULATION:
        raise PolicyGatedAdapterError(
            "control-stack runtime adapters are local-simulator only; hardware mode is refused"
        )
    return decision


def run_realtime_feedback_adapter(
    controller: RealtimeFeedbackPort,
    *,
    policy: ClosedLoopExecutionPolicy | None,
    n_rounds: int,
    seed: int | None = None,
    backend: str | None = None,
) -> RealtimeFeedbackAdapterResult:
    """Run the existing realtime-feedback controller after policy authorisation."""
    decision = _authorise_local_adapter(
        policy,
        requested_rounds=n_rounds,
        backend=backend,
    )
    return RealtimeFeedbackAdapterResult(
        decision=decision,
        steps=tuple(controller.run(n_rounds, seed=seed)),
    )


def run_qaoa_mpc_adapter(
    controller: QaoaMpcPort,
    *,
    policy: ClosedLoopExecutionPolicy | None,
    seed: int | None = None,
    backend: str | None = None,
) -> QaoaMpcAdapterResult:
    """Run the existing abstract QAOA-MPC optimiser under hardware-safe policy."""
    decision = _authorise_local_adapter(policy, requested_rounds=1, backend=backend)
    actions = np.asarray(controller.optimize(seed=seed), dtype=np.int64)
    if actions.ndim != 1 or actions.size == 0:
        raise ValueError("QAOA-MPC adapter requires a non-empty 1-D action schedule")
    if not np.all((actions == 0) | (actions == 1)):
        raise ValueError("QAOA-MPC adapter actions must be binary")
    return QaoaMpcAdapterResult(
        decision=decision,
        actions=tuple(int(action) for action in actions),
    )


def run_cosimulation_partition_adapter(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    policy: ClosedLoopExecutionPolicy | None,
    dt: float,
    n_steps: int,
    partition: KnmPartition | None = None,
    max_quantum_nodes: int = 8,
    coupling_threshold: float = 0.0,
    theta0_classical: NDArray[np.float64] | None = None,
    quantum_state0: NDArray[np.complex128] | None = None,
    seed: int | None = None,
    backend: str | None = None,
) -> CosimulationPartitionAdapterResult:
    """Run and map the existing quantum/classical partition implementation."""
    decision = _authorise_local_adapter(
        policy,
        requested_rounds=n_steps,
        backend=backend,
    )
    result = cosimulate(
        K,
        omega,
        dt=dt,
        n_steps=n_steps,
        partition=partition,
        max_quantum_nodes=max_quantum_nodes,
        coupling_threshold=coupling_threshold,
        theta0_classical=theta0_classical,
        quantum_state0=quantum_state0,
        seed=seed,
    )
    telemetry = CosimulationPartitionTelemetry(
        n_quantum=result.partition.n_quantum,
        n_classical=result.partition.n_classical,
        samples=int(result.times.size),
        final_quantum_order=float(result.quantum_order[-1]),
        final_classical_order=float(result.classical_order[-1]),
        final_global_order=float(result.global_order[-1]),
        baseline_deviation=float(result.baseline_deviation),
        cross_fraction=float(result.partition.conservation.cross_fraction),
        claim_boundary=str(result.provenance["claim_boundary"]),
    )
    return CosimulationPartitionAdapterResult(
        decision=decision,
        result=result,
        telemetry=telemetry,
    )


def decide_pulse_compose_boundary(
    *, policy: ClosedLoopExecutionPolicy | None
) -> PulseComposeBoundaryDecision:
    """Refuse pulse execution and point to the optional execution adapter."""
    _authorise_local_adapter(policy, requested_rounds=1, backend=None)
    return PulseComposeBoundaryDecision(
        allowed=False,
        reason=(
            "pulse execution is outside control-stack adapter ownership; use the optional "
            "pulse-execution adapter after its contracts and export validation are complete"
        ),
        dependency="pulse-boundary/runtime-adapter",
    )


__all__ = [
    "CONTROL_STACK_RUNTIME_ADAPTER_CLAIM_BOUNDARY",
    "CONTROL_STACK_RUNTIME_ADAPTER_SCHEMA",
    "CosimulationPartitionAdapterResult",
    "CosimulationPartitionTelemetry",
    "PolicyGatedAdapterError",
    "PulseComposeBoundaryDecision",
    "QaoaMpcAdapterResult",
    "QaoaMpcPort",
    "RealtimeFeedbackAdapterResult",
    "RealtimeFeedbackPort",
    "decide_pulse_compose_boundary",
    "run_cosimulation_partition_adapter",
    "run_qaoa_mpc_adapter",
    "run_realtime_feedback_adapter",
]
