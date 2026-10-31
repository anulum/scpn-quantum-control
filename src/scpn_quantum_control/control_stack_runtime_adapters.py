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

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

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


# --- Constrained co-design proposals ---------------------------------------------------

ProposalStatus = Literal["accepted", "infeasible", "nonconvergent", "out_of_domain", "over_budget"]
CODESIGN_PROPOSAL_STATUSES: Final[tuple[ProposalStatus, ...]] = (
    "accepted",
    "infeasible",
    "nonconvergent",
    "out_of_domain",
    "over_budget",
)
"""Every outcome of :meth:`ConstrainedCodesignSession.propose`; only ``accepted`` adopts a plan."""
CodesignObjective = Callable[[NDArray[np.float64]], float]
CodesignConstraints = Callable[[NDArray[np.float64], NDArray[np.float64] | None], ArrayLike]


@dataclass(frozen=True, slots=True)
class OptimiserOutcome:
    """What an optimiser port returns: its plan, whether it converged, and its cost."""

    plan: NDArray[np.float64]
    converged: bool
    evaluations: int


class ConstrainedOptimiserPort(Protocol):
    """An existing optimiser adapted to propose a plan from a starting point."""

    def optimize(self, start: NDArray[np.float64]) -> OptimiserOutcome:
        """Return a proposed plan; the session verifies it independently."""
        ...


@dataclass(frozen=True, slots=True)
class ConstrainedProposal:
    """One verified proposal and the plan in force after it.

    ``constraint_values`` holds ``g(plan, d)`` for the nominal case first and then every held-out
    disturbance, recomputed by the session rather than taken from the optimiser; ``disturbances``
    lists the held-out disturbances checked and ``tolerance`` the admitted constraint value.
    ``evaluations`` is the optimiser port's own report. Only an ``accepted`` proposal changes
    ``adopted_plan``; every other status keeps the previous plan. No proposal actuates hardware.
    """

    status: ProposalStatus
    reason: str
    decision: ClosedLoopExecutionDecision
    plan: tuple[float, ...] | None
    objective: float | None
    constraint_values: tuple[tuple[float, ...], ...]
    max_violation: float | None
    evaluations: int
    adopted_plan: tuple[float, ...] | None
    domain: tuple[tuple[float, float], ...]
    tolerance: float
    disturbances: tuple[tuple[float, ...], ...]
    hardware_actuation: bool = False
    schema: str = CONTROL_STACK_RUNTIME_ADAPTER_SCHEMA
    claim_boundary: str = CONTROL_STACK_RUNTIME_ADAPTER_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Reject stale schemas, altered claim boundaries and any actuation."""
        _validate_result_contract(self.schema, self.claim_boundary)
        if self.hardware_actuation:
            raise ValueError("a co-design proposal never actuates hardware")


class ConstrainedCodesignSession:
    """Hold an accepted plan and adopt a new one only after independent verification.

    The plan lives in the box ``[lower, upper]`` (the tested domain). A proposal is accepted
    only when the optimiser converged within the evaluation budget, the plan lies in the
    domain, and ``g(plan, d) <= tolerance`` holds for the nominal case and every held-out
    disturbance, all recomputed here. Infeasibility, non-convergence, an out-of-domain plan, an
    exhausted budget or an optimiser error are outcomes: the previous accepted plan stays in force.
    Every proposal is authorised by the closed-loop execution policy in simulation mode first.
    """

    def __init__(
        self,
        objective: CodesignObjective,
        constraints: CodesignConstraints,
        lower: ArrayLike,
        upper: ArrayLike,
        *,
        policy: ClosedLoopExecutionPolicy | None,
        tolerance: float = 0.0,
        held_out_disturbances: Sequence[ArrayLike] = (),
        evaluation_budget: int = 10_000,
        initial_plan: ArrayLike | None = None,
    ) -> None:
        """Validate the problem and, if given, verify the initial plan.

        Parameters
        ----------
        objective
            ``f(plan)``, recomputed for every proposal.
        constraints
            ``g(plan, disturbance)``; feasible when every entry is ``<= tolerance``. The
            disturbance is ``None`` for the nominal case.
        lower, upper
            Finite bounds of the tested domain, ``lower < upper`` elementwise.
        policy
            Closed-loop execution policy; it must authorise local simulation.
        tolerance
            Largest admitted constraint value, in the constraints' own units; the default ``0``
            admits no violation. Any slack is the caller's choice for its constraint scale.
        held_out_disturbances
            Disturbances the optimiser does not see; every one must also be feasible.
        evaluation_budget
            Largest admitted optimiser evaluation count per proposal, as the port reports it.
        initial_plan
            An already accepted plan; it is verified like a proposal.

        Raises
        ------
        ValueError
            If the bounds, tolerance or budget are malformed, or the initial plan fails.

        """
        self._lower = np.asarray(lower, dtype=np.float64)
        self._upper = np.asarray(upper, dtype=np.float64)
        if (
            self._lower.ndim != 1
            or self._lower.size == 0
            or self._lower.shape != self._upper.shape
            or not np.all(np.isfinite(self._lower))
            or not np.all(np.isfinite(self._upper))
            or np.any(self._lower >= self._upper)
        ):
            raise ValueError("lower and upper must be finite vectors with lower < upper")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        if isinstance(evaluation_budget, bool) or evaluation_budget < 1:
            raise ValueError("evaluation_budget must be a positive integer")
        self._objective = objective
        self._constraints = constraints
        self._policy = policy
        self._tolerance = float(tolerance)
        self._budget = int(evaluation_budget)
        self._disturbances = tuple(
            np.asarray(item, dtype=np.float64) for item in held_out_disturbances
        )
        self._plan: NDArray[np.float64] | None = None
        if initial_plan is not None:
            start = np.asarray(initial_plan, dtype=np.float64)
            status, reason, _, _, _ = self._verify(start)
            if status != "accepted":
                raise ValueError(f"initial plan is not admissible: {reason}")
            self._plan = start.copy()

    @property
    def domain(self) -> tuple[tuple[float, float], ...]:
        """The tested domain as ``(lower, upper)`` per coordinate."""
        return tuple(zip(self._lower.tolist(), self._upper.tolist(), strict=True))

    @property
    def current_plan(self) -> tuple[float, ...] | None:
        """The last accepted plan, or ``None`` before any acceptance."""
        return None if self._plan is None else tuple(self._plan.tolist())

    def _verify(
        self, plan: NDArray[np.float64]
    ) -> tuple[ProposalStatus, str, tuple[tuple[float, ...], ...], float | None, float | None]:
        """Check domain and constraints and recompute the objective."""
        if plan.shape != self._lower.shape or not np.all(np.isfinite(plan)):
            return "out_of_domain", "the plan has the wrong shape or is not finite", (), None, None
        if np.any(plan < self._lower) or np.any(plan > self._upper):
            return "out_of_domain", "the plan leaves the tested domain", (), None, None
        values: list[tuple[float, ...]] = []
        for disturbance in (None, *self._disturbances):
            row = np.asarray(self._constraints(plan.copy(), disturbance), dtype=np.float64)
            if row.ndim != 1 or row.size == 0 or not np.all(np.isfinite(row)):
                raise ValueError(
                    "constraints must return a finite non-empty one-dimensional vector"
                )
            if values and row.size != len(values[0]):
                raise ValueError(
                    "constraints must return the nominal number of values for every disturbance"
                )
            values.append(tuple(row.tolist()))
        violation = max(max(row) for row in values)
        objective = float(self._objective(plan.copy()))
        if not math.isfinite(objective):
            raise ValueError("objective must return a finite value")
        if violation > self._tolerance:
            scope = (
                "a held-out disturbance"
                if max(values[0]) <= self._tolerance
                else "the nominal case"
            )
            return (
                "infeasible",
                f"constraint violation {violation:.3e} exceeds {self._tolerance:.3e} in {scope}",
                tuple(values),
                violation,
                objective,
            )
        return (
            "accepted",
            "every constraint holds in every checked case",
            tuple(values),
            violation,
            objective,
        )

    def propose(self, optimiser: ConstrainedOptimiserPort) -> ConstrainedProposal:
        """Run the optimiser from the current plan and adopt its plan only if it verifies.

        Parameters
        ----------
        optimiser
            The optimiser to run; it starts from the current plan, or from the domain centre.

        Returns
        -------
        ConstrainedProposal
            The outcome, the independently recomputed evidence and the plan now in force.

        Raises
        ------
        PolicyGatedAdapterError
            If the policy does not authorise local simulation; the optimiser is not run.

        """
        decision = _authorise_local_adapter(self._policy, requested_rounds=1, backend=None)
        start = self._plan.copy() if self._plan is not None else 0.5 * (self._lower + self._upper)

        def outcome(
            status: ProposalStatus,
            reason: str,
            plan: NDArray[np.float64] | None,
            values: tuple[tuple[float, ...], ...] = (),
            violation: float | None = None,
            objective: float | None = None,
            evaluations: int = 0,
        ) -> ConstrainedProposal:
            return ConstrainedProposal(
                status=status,
                reason=reason,
                decision=decision,
                plan=None if plan is None else tuple(plan.tolist()),
                objective=objective,
                constraint_values=values,
                max_violation=violation,
                evaluations=evaluations,
                adopted_plan=self.current_plan,
                domain=self.domain,
                tolerance=self._tolerance,
                disturbances=tuple(tuple(item.tolist()) for item in self._disturbances),
            )

        try:
            result = optimiser.optimize(start.copy())
        except (ArithmeticError, ValueError, RuntimeError) as error:
            # An unimplemented port or runaway recursion is a programming error, not an outcome.
            if isinstance(error, NotImplementedError | RecursionError):
                raise
            return outcome("nonconvergent", f"the optimiser failed: {error}", None)
        plan = np.asarray(result.plan, dtype=np.float64)
        if not result.converged:
            return outcome(
                "nonconvergent",
                "the optimiser did not converge",
                plan,
                evaluations=result.evaluations,
            )
        if result.evaluations > self._budget:
            return outcome(
                "over_budget",
                f"{result.evaluations} evaluations exceed the budget {self._budget}",
                plan,
                evaluations=result.evaluations,
            )
        status, reason, values, violation, objective = self._verify(plan)
        if status == "accepted":
            self._plan = plan.copy()
        return outcome(status, reason, plan, values, violation, objective, result.evaluations)


__all__ = [
    "CODESIGN_PROPOSAL_STATUSES",
    "CodesignConstraints",
    "CodesignObjective",
    "CONTROL_STACK_RUNTIME_ADAPTER_CLAIM_BOUNDARY",
    "CONTROL_STACK_RUNTIME_ADAPTER_SCHEMA",
    "ConstrainedCodesignSession",
    "ConstrainedOptimiserPort",
    "ConstrainedProposal",
    "CosimulationPartitionAdapterResult",
    "CosimulationPartitionTelemetry",
    "PolicyGatedAdapterError",
    "PulseComposeBoundaryDecision",
    "QaoaMpcAdapterResult",
    "QaoaMpcPort",
    "RealtimeFeedbackAdapterResult",
    "RealtimeFeedbackPort",
    "OptimiserOutcome",
    "decide_pulse_compose_boundary",
    "run_cosimulation_partition_adapter",
    "run_qaoa_mpc_adapter",
    "run_realtime_feedback_adapter",
]
