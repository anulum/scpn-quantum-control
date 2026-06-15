# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — closed-loop control analysis for measurement feedback
"""Control-theoretic analysis of the measurement-feedback synchronisation loop.

The live-shot controller in :mod:`scpn_quantum_control.control.realtime_feedback`
measures an order-parameter estimate each round and adjusts the coupling scale
toward a set-point. This module turns the resulting response trajectory into a
control-theoretic verdict — settling time, steady-state error, overshoot,
integral absolute error, and a converged / limit-cycle / diverged / unsettled
classification — and gates any non-simulation execution behind an explicit
policy. It produces a deterministic, replayable evidence record.

It is a local software-in-the-loop assessment. It is not provider-prepared
dynamic-circuit evidence and not live closed-loop QPU evidence; hardware
execution is refused fail-closed without an explicit live ticket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .realtime_feedback import RealtimeSyncFeedbackController


class ResponseClass(str, Enum):
    """Closed-loop response verdict."""

    CONVERGED = "converged"
    LIMIT_CYCLE = "limit_cycle"
    DIVERGED = "diverged"
    UNSETTLED = "unsettled"


class ExecutionMode(str, Enum):
    """Where a closed-loop run is authorised to execute."""

    SIMULATION = "simulation"
    HARDWARE = "hardware"


@dataclass(frozen=True)
class ControlPerformance:
    """Control-theoretic metrics of a set-point tracking response."""

    target: float
    final_value: float
    steady_state_error: float
    settling_round: int | None
    overshoot: float
    integral_absolute_error: float
    oscillation_amplitude: float
    error_sign_changes: int


@dataclass(frozen=True)
class ClosedLoopExecutionPolicy:
    """Fail-closed gate for closed-loop execution; simulation-only by default."""

    allow_hardware: bool = False
    live_ticket: str | None = None
    backend_allowlist: tuple[str, ...] = ()
    round_budget: int = 256

    def __post_init__(self) -> None:
        if self.round_budget < 1:
            raise ValueError("round_budget must be a positive integer")


@dataclass(frozen=True)
class ClosedLoopExecutionDecision:
    """Outcome of evaluating a :class:`ClosedLoopExecutionPolicy`."""

    authorised: bool
    mode: ExecutionMode
    reason: str


@dataclass(frozen=True)
class ClosedLoopControlEvidence:
    """Replayable evidence of a closed-loop control run."""

    response: np.ndarray
    feedback_signal: np.ndarray
    target: float
    classification: ResponseClass
    performance: ControlPerformance
    decision: ClosedLoopExecutionDecision
    provenance: dict[str, Any] = field(default_factory=dict)


def evaluate_closed_loop_policy(
    policy: ClosedLoopExecutionPolicy,
    *,
    backend: str | None = None,
    requested_rounds: int,
) -> ClosedLoopExecutionDecision:
    """Authorise a closed-loop run, defaulting to a simulation fallback.

    Hardware execution requires ``allow_hardware``, a non-empty ``live_ticket``,
    and an allow-listed ``backend``; otherwise the run is authorised in
    simulation. The round budget is enforced in both modes.
    """
    if requested_rounds < 1:
        raise ValueError("requested_rounds must be a positive integer")
    if requested_rounds > policy.round_budget:
        return ClosedLoopExecutionDecision(
            authorised=False,
            mode=ExecutionMode.SIMULATION,
            reason=(
                f"requested rounds {requested_rounds} exceed the policy budget "
                f"{policy.round_budget}"
            ),
        )
    if not policy.allow_hardware:
        return ClosedLoopExecutionDecision(
            authorised=True,
            mode=ExecutionMode.SIMULATION,
            reason="hardware not requested; simulation-in-the-loop fallback",
        )
    if not policy.live_ticket:
        return ClosedLoopExecutionDecision(
            authorised=False,
            mode=ExecutionMode.SIMULATION,
            reason="hardware requested without a live execution ticket",
        )
    if backend is None or backend not in policy.backend_allowlist:
        return ClosedLoopExecutionDecision(
            authorised=False,
            mode=ExecutionMode.SIMULATION,
            reason="hardware backend is not on the policy allow-list",
        )
    return ClosedLoopExecutionDecision(
        authorised=True,
        mode=ExecutionMode.HARDWARE,
        reason=f"live ticket {policy.live_ticket} authorised on {backend}",
    )


def _settling_round(error: np.ndarray, tolerance: float) -> int | None:
    """First round after which the response stays inside the tolerance band."""
    within = np.abs(error) <= tolerance
    if not within[-1]:
        return None
    settle = len(error) - 1
    while settle > 0 and within[settle - 1]:
        settle -= 1
    return int(settle)


def analyse_closed_loop_response(
    response: np.ndarray,
    target: float,
    *,
    tolerance: float,
    settle_window: int = 8,
    limit_cycle_sign_changes: int = 4,
) -> tuple[ResponseClass, ControlPerformance]:
    """Classify a set-point tracking response and compute control metrics.

    Args:
        response: the controlled-output trajectory (one value per round).
        target: the set-point.
        tolerance: half-width of the acceptance band around the set-point.
        settle_window: number of trailing rounds used for steady-state metrics.
        limit_cycle_sign_changes: trailing error sign changes that mark a
            sustained oscillation rather than convergence.

    Returns:
        The :class:`ResponseClass` verdict and the :class:`ControlPerformance`.
    """
    response = np.asarray(response, dtype=np.float64)
    if response.ndim != 1 or response.size < 2:
        raise ValueError("response must be a 1-D trajectory of at least two rounds")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if settle_window < 1:
        raise ValueError("settle_window must be a positive integer")

    error = response - target
    window = min(settle_window, response.size)
    # Oscillation evidence needs a longer tail than the steady-state window so a
    # slow limit cycle is not mistaken for a settled response.
    osc_window = min(response.size, max(4 * settle_window, window))

    early_error = float(np.mean(np.abs(error[:window])))
    steady_state_error = float(np.mean(np.abs(error[-window:])))
    settling_round = _settling_round(error, tolerance)
    integral_absolute_error = float(np.sum(np.abs(error)))
    oscillation_amplitude = float(np.max(response[-osc_window:]) - np.min(response[-osc_window:]))

    # Overshoot beyond the set-point in the direction of approach.
    initial_gap = float(abs(error[0]))
    if error[0] < 0:
        peak_excess = float(max(0.0, np.max(response) - target))
    else:
        peak_excess = float(max(0.0, target - np.min(response)))
    overshoot = peak_excess / initial_gap if initial_gap > 1e-12 else 0.0

    sign = np.sign(error[-osc_window:])
    nonzero = sign[sign != 0.0]
    error_sign_changes = int(np.sum(np.abs(np.diff(nonzero)) > 0)) if nonzero.size > 1 else 0

    performance = ControlPerformance(
        target=float(target),
        final_value=float(response[-1]),
        steady_state_error=steady_state_error,
        settling_round=settling_round,
        overshoot=float(overshoot),
        integral_absolute_error=integral_absolute_error,
        oscillation_amplitude=oscillation_amplitude,
        error_sign_changes=error_sign_changes,
    )

    sustained_oscillation = (
        error_sign_changes >= limit_cycle_sign_changes and oscillation_amplitude > tolerance
    )
    if settling_round is not None and steady_state_error <= tolerance:
        classification = ResponseClass.CONVERGED
    elif sustained_oscillation:
        classification = ResponseClass.LIMIT_CYCLE
    elif steady_state_error > early_error + tolerance:
        classification = ResponseClass.DIVERGED
    else:
        classification = ResponseClass.UNSETTLED
    return classification, performance


def run_closed_loop_control(
    controller: RealtimeSyncFeedbackController,
    n_rounds: int,
    *,
    policy: ClosedLoopExecutionPolicy | None = None,
    seed: int | None = None,
    tolerance: float | None = None,
    settle_window: int = 8,
    backend: str | None = None,
) -> ClosedLoopControlEvidence:
    """Run the feedback controller under a policy and analyse the response.

    The exact order parameter is the controlled output; the sampled live order
    parameter is the feedback signal. Hardware is refused fail-closed unless the
    policy authorises it; the controller always runs the local simulation.
    """
    if n_rounds < 2:
        raise ValueError("n_rounds must be at least two for a control verdict")
    policy = policy or ClosedLoopExecutionPolicy()
    decision = evaluate_closed_loop_policy(policy, backend=backend, requested_rounds=n_rounds)
    target = controller.config.target_r
    band = tolerance if tolerance is not None else max(2.0 * controller.config.deadband, 1e-3)

    steps = controller.run(n_rounds, seed=seed)
    response = np.array([step.r_statevector for step in steps], dtype=np.float64)
    feedback_signal = np.array([step.r_live for step in steps], dtype=np.float64)
    classification, performance = analyse_closed_loop_response(
        response, target, tolerance=band, settle_window=settle_window
    )

    provenance = {
        "n_rounds": n_rounds,
        "target_r": target,
        "tolerance": band,
        "settle_window": settle_window,
        "execution_mode": decision.mode.value,
        "execution_authorised": decision.authorised,
        "seed": seed,
        "claim_boundary": (
            "software-in-the-loop closed-loop control assessment; the controlled "
            "output is the exact statevector order parameter and the feedback "
            "signal is the finite-shot estimate; not provider-prepared dynamic "
            "circuit evidence and not live closed-loop QPU evidence"
        ),
    }
    return ClosedLoopControlEvidence(
        response=response,
        feedback_signal=feedback_signal,
        target=float(target),
        classification=classification,
        performance=performance,
        decision=decision,
        provenance=provenance,
    )
