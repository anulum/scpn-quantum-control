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

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

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

    response: NDArray[np.float64]
    feedback_signal: NDArray[np.float64]
    target: float
    classification: ResponseClass
    performance: ControlPerformance
    decision: ClosedLoopExecutionDecision
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClosedLoopLatencyBudget:
    """Wall-clock latency budget for software-in-the-loop feedback rounds."""

    max_round_latency_s: float = 0.050
    p95_round_latency_s: float | None = None
    p99_round_latency_s: float | None = None
    max_total_latency_s: float | None = None

    def __post_init__(self) -> None:
        _require_positive(self.max_round_latency_s, "max_round_latency_s")
        if self.p95_round_latency_s is not None:
            _require_positive(self.p95_round_latency_s, "p95_round_latency_s")
        if self.p99_round_latency_s is not None:
            _require_positive(self.p99_round_latency_s, "p99_round_latency_s")
        if self.max_total_latency_s is not None:
            _require_positive(self.max_total_latency_s, "max_total_latency_s")


@dataclass(frozen=True)
class ClosedLoopLatencyReport:
    """Budget verdict for a no-submit closed-loop latency measurement."""

    budget: ClosedLoopLatencyBudget
    control_evidence: ClosedLoopControlEvidence
    round_latencies_s: tuple[float, ...]
    total_latency_s: float
    max_round_latency_s: float
    p95_round_latency_s: float
    p99_round_latency_s: float
    passes: bool
    blockers: tuple[str, ...]
    classification: str
    clock_source: str
    claim_boundary: str

    @property
    def samples(self) -> int:
        """Number of measured feedback rounds."""

        return len(self.round_latencies_s)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready latency report."""

        return {
            "classification": self.classification,
            "passes": self.passes,
            "blockers": list(self.blockers),
            "samples": self.samples,
            "round_latencies_s": list(self.round_latencies_s),
            "total_latency_s": self.total_latency_s,
            "max_round_latency_s": self.max_round_latency_s,
            "p95_round_latency_s": self.p95_round_latency_s,
            "p99_round_latency_s": self.p99_round_latency_s,
            "budget": {
                "max_round_latency_s": self.budget.max_round_latency_s,
                "p95_round_latency_s": self.budget.p95_round_latency_s,
                "p99_round_latency_s": self.budget.p99_round_latency_s,
                "max_total_latency_s": self.budget.max_total_latency_s,
            },
            "execution": {
                "authorised": self.control_evidence.decision.authorised,
                "mode": self.control_evidence.decision.mode.value,
                "reason": self.control_evidence.decision.reason,
            },
            "response": {
                "classification": self.control_evidence.classification.value,
                "target": self.control_evidence.target,
                "final_value": self.control_evidence.performance.final_value,
                "steady_state_error": self.control_evidence.performance.steady_state_error,
                "settling_round": self.control_evidence.performance.settling_round,
                "integral_absolute_error": (
                    self.control_evidence.performance.integral_absolute_error
                ),
            },
            "clock_source": self.clock_source,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class ClosedLoopPublicationPackage:
    """Publication-control scaffold for closed-loop feedback evidence classes."""

    title: str
    evidence_classes: tuple[dict[str, str], ...]
    methods_sections: tuple[dict[str, str], ...]
    artefact_map: tuple[dict[str, str], ...]
    benchmark_rows: tuple[dict[str, Any], ...]
    claim_ledger_rows: tuple[dict[str, str], ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready publication package scaffold."""

        return {
            "title": self.title,
            "evidence_classes": {
                item["id"]: {key: value for key, value in item.items() if key != "id"}
                for item in self.evidence_classes
            },
            "methods_sections": list(self.methods_sections),
            "artefact_map": list(self.artefact_map),
            "benchmark_rows": list(self.benchmark_rows),
            "claim_ledger_rows": list(self.claim_ledger_rows),
            "claim_boundary": self.claim_boundary,
        }

    def to_markdown(self) -> str:
        """Return a compact Markdown scaffold for campaign notes."""

        lines = [
            f"# {self.title}",
            "",
            self.claim_boundary,
            "",
            "## Evidence Classes",
        ]
        for item in self.evidence_classes:
            lines.append(f"- `{item['id']}`: {item['claim_boundary']}")
        lines.extend(["", "## Benchmark Rows"])
        for row in self.benchmark_rows:
            lines.append(
                f"- `{row['classification']}`: pass={row['passes']}; "
                f"max_round_latency_s={row['max_round_latency_s']:.9f}; "
                f"p95_round_latency_s={row['p95_round_latency_s']:.9f}"
            )
        lines.extend(["", "## Claim Ledger Rows"])
        for row in self.claim_ledger_rows:
            lines.append(f"- `{row['claim_id']}`: {row['promotion_status']} — {row['evidence']}")
        return "\n".join(lines)


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


def _settling_round(error: NDArray[np.float64], tolerance: float) -> int | None:
    """First round after which the response stays inside the tolerance band."""
    within = np.abs(error) <= tolerance
    if not within[-1]:
        return None
    settle = len(error) - 1
    while settle > 0 and within[settle - 1]:
        settle -= 1
    return int(settle)


def analyse_closed_loop_response(
    response: NDArray[np.float64],
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


def measure_closed_loop_latency_budget(
    controller: RealtimeSyncFeedbackController,
    n_rounds: int,
    *,
    budget: ClosedLoopLatencyBudget | None = None,
    policy: ClosedLoopExecutionPolicy | None = None,
    seed: int | None = None,
    tolerance: float | None = None,
    settle_window: int = 8,
    backend: str | None = None,
    observed_round_latencies_s: tuple[float, ...] | None = None,
) -> ClosedLoopLatencyReport:
    """Measure or replay no-submit closed-loop latency against a budget.

    When ``observed_round_latencies_s`` is omitted, each local simulator
    ``step()`` is measured with :func:`time.perf_counter_ns`. Supplying observed
    samples is intended for deterministic replay and CI fixtures; it still runs
    the controller to produce the same response evidence and policy decision.
    """

    if n_rounds < 2:
        raise ValueError("n_rounds must be at least two for a latency verdict")
    budget = budget or ClosedLoopLatencyBudget()
    policy = policy or ClosedLoopExecutionPolicy()
    decision = evaluate_closed_loop_policy(policy, backend=backend, requested_rounds=n_rounds)

    if observed_round_latencies_s is not None:
        if len(observed_round_latencies_s) != n_rounds:
            raise ValueError("observed_round_latencies_s must contain one sample per round")
        if any(sample < 0.0 or not np.isfinite(sample) for sample in observed_round_latencies_s):
            raise ValueError("observed_round_latencies_s must contain finite non-negative samples")
        evidence = run_closed_loop_control(
            controller,
            n_rounds,
            policy=policy,
            seed=seed,
            tolerance=tolerance,
            settle_window=settle_window,
            backend=backend,
        )
        round_latencies = tuple(float(sample) for sample in observed_round_latencies_s)
        clock_source = "replayed_observed_round_latencies_s"
    else:
        evidence, round_latencies = _run_closed_loop_control_with_wall_clock(
            controller,
            n_rounds,
            decision=decision,
            seed=seed,
            tolerance=tolerance,
            settle_window=settle_window,
        )
        clock_source = "time.perf_counter_ns"

    total_latency = float(sum(round_latencies))
    max_latency = float(max(round_latencies))
    p95 = float(np.percentile(np.asarray(round_latencies, dtype=np.float64), 95))
    p99 = float(np.percentile(np.asarray(round_latencies, dtype=np.float64), 99))
    blockers = _latency_budget_blockers(
        budget=budget,
        decision=decision,
        total_latency_s=total_latency,
        max_round_latency_s=max_latency,
        p95_round_latency_s=p95,
        p99_round_latency_s=p99,
    )
    claim_boundary = (
        "software-in-the-loop closed-loop latency measurement; measures local "
        "controller/simulator wall-clock time only; not provider-prepared "
        "dynamic-circuit evidence and not live closed-loop QPU evidence"
    )
    return ClosedLoopLatencyReport(
        budget=budget,
        control_evidence=evidence,
        round_latencies_s=round_latencies,
        total_latency_s=total_latency,
        max_round_latency_s=max_latency,
        p95_round_latency_s=p95,
        p99_round_latency_s=p99,
        passes=not blockers,
        blockers=blockers,
        classification="software_in_loop_latency",
        clock_source=clock_source,
        claim_boundary=claim_boundary,
    )


def build_closed_loop_publication_package(
    *,
    latency_report: ClosedLoopLatencyReport,
) -> ClosedLoopPublicationPackage:
    """Build a no-submit publication scaffold for closed-loop feedback evidence."""

    benchmark_row = {
        "id": "closed_loop_software_latency",
        "classification": latency_report.classification,
        "passes": latency_report.passes,
        "samples": latency_report.samples,
        "max_round_latency_s": latency_report.max_round_latency_s,
        "p95_round_latency_s": latency_report.p95_round_latency_s,
        "p99_round_latency_s": latency_report.p99_round_latency_s,
        "total_latency_s": latency_report.total_latency_s,
        "claim_boundary": latency_report.claim_boundary,
    }
    claim_boundary = (
        "Publication scaffold only: separates software-in-the-loop simulation, "
        "provider-prepared dynamic-circuit readiness, and true live closed-loop "
        "QPU evidence; not live closed-loop QPU evidence until a live-ticket run "
        "adds provider job IDs, raw counts, calibration snapshots, and replay artefacts."
    )
    return ClosedLoopPublicationPackage(
        title="Closed-Loop Quantum Control Evidence Package",
        evidence_classes=(
            {
                "id": "software_in_loop_simulation",
                "status": "available",
                "claim_boundary": latency_report.claim_boundary,
            },
            {
                "id": "provider_prepared_dynamic_circuit",
                "status": "placeholder",
                "claim_boundary": (
                    "requires backend capability proof, transpiled dynamic circuit, "
                    "provider preparation metadata, and no-submit policy approval"
                ),
            },
            {
                "id": "live_closed_loop_qpu",
                "status": "blocked_without_live_ticket",
                "claim_boundary": (
                    "requires explicit live ticket, allow-listed backend, raw counts, "
                    "calibration snapshot, job IDs, and replay harness"
                ),
            },
        ),
        methods_sections=(
            {
                "id": "controller_contract",
                "text": (
                    "Mid-circuit measurement feeds a classical update policy that "
                    "adjusts the next-round coupling parameter under explicit policy gates."
                ),
            },
            {
                "id": "latency_budget",
                "text": (
                    "Software-in-the-loop latency records per-round wall-clock samples "
                    "and fails closed on max, p95, p99, or total-budget breaches."
                ),
            },
            {
                "id": "hardware_boundary",
                "text": (
                    "Provider-prepared and live-QPU claims stay unpromoted until "
                    "the hardware policy authorises a live ticket and artefacts exist."
                ),
            },
        ),
        artefact_map=(
            {
                "id": "software_latency_report",
                "path": "generated_by_measure_closed_loop_latency_budget",
                "required_for": "software-in-the-loop latency claim",
            },
            {
                "id": "provider_dynamic_circuit_pack",
                "path": "pending_live_ticket",
                "required_for": "provider-prepared dynamic-circuit claim",
            },
            {
                "id": "live_qpu_raw_counts",
                "path": "pending_live_ticket",
                "required_for": "true live closed-loop QPU claim",
            },
        ),
        benchmark_rows=(benchmark_row,),
        claim_ledger_rows=(
            {
                "claim_id": "closed_loop_software_latency",
                "promotion_status": "unpromoted",
                "evidence": "software-in-the-loop latency report only",
            },
            {
                "claim_id": "closed_loop_provider_dynamic_circuit",
                "promotion_status": "blocked",
                "evidence": "provider preparation artefacts pending",
            },
            {
                "claim_id": "closed_loop_live_qpu",
                "promotion_status": "blocked",
                "evidence": "live hardware artefacts pending",
            },
        ),
        claim_boundary=claim_boundary,
    )


def _run_closed_loop_control_with_wall_clock(
    controller: RealtimeSyncFeedbackController,
    n_rounds: int,
    *,
    decision: ClosedLoopExecutionDecision,
    seed: int | None,
    tolerance: float | None,
    settle_window: int,
) -> tuple[ClosedLoopControlEvidence, tuple[float, ...]]:
    controller.reset()
    rng = np.random.default_rng(seed)
    steps = []
    latencies: list[float] = []
    for _ in range(n_rounds):
        started = time.perf_counter_ns()
        step = controller.step(seed=int(rng.integers(0, 2**32 - 1)))
        finished = time.perf_counter_ns()
        steps.append(step)
        latencies.append((finished - started) / 1_000_000_000.0)

    response = np.array([step.r_statevector for step in steps], dtype=np.float64)
    feedback_signal = np.array([step.r_live for step in steps], dtype=np.float64)
    target = controller.config.target_r
    band = tolerance if tolerance is not None else max(2.0 * controller.config.deadband, 1e-3)
    classification, performance = analyse_closed_loop_response(
        response, target, tolerance=band, settle_window=settle_window
    )
    evidence = ClosedLoopControlEvidence(
        response=response,
        feedback_signal=feedback_signal,
        target=target,
        classification=classification,
        performance=performance,
        decision=decision,
        provenance={
            "n_rounds": n_rounds,
            "target_r": target,
            "tolerance": band,
            "settle_window": settle_window,
            "execution_mode": decision.mode.value,
            "execution_authorised": decision.authorised,
            "seed": seed,
            "claim_boundary": (
                "software-in-the-loop closed-loop control assessment with local "
                "wall-clock latency measurement; not provider-prepared dynamic "
                "circuit evidence and not live closed-loop QPU evidence"
            ),
        },
    )
    return evidence, tuple(latencies)


def _latency_budget_blockers(
    *,
    budget: ClosedLoopLatencyBudget,
    decision: ClosedLoopExecutionDecision,
    total_latency_s: float,
    max_round_latency_s: float,
    p95_round_latency_s: float,
    p99_round_latency_s: float,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if not decision.authorised:
        blockers.append(f"closed-loop execution policy not authorised: {decision.reason}")
    if max_round_latency_s > budget.max_round_latency_s:
        blockers.append(
            f"max round latency {max_round_latency_s:.9f}s exceeds "
            f"{budget.max_round_latency_s:.9f}s"
        )
    if budget.p95_round_latency_s is not None and p95_round_latency_s > budget.p95_round_latency_s:
        blockers.append(
            f"p95 round latency {p95_round_latency_s:.9f}s exceeds "
            f"{budget.p95_round_latency_s:.9f}s"
        )
    if budget.p99_round_latency_s is not None and p99_round_latency_s > budget.p99_round_latency_s:
        blockers.append(
            f"p99 round latency {p99_round_latency_s:.9f}s exceeds "
            f"{budget.p99_round_latency_s:.9f}s"
        )
    if budget.max_total_latency_s is not None and total_latency_s > budget.max_total_latency_s:
        blockers.append(
            f"total latency {total_latency_s:.9f}s exceeds {budget.max_total_latency_s:.9f}s"
        )
    return tuple(blockers)


def _require_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive")
