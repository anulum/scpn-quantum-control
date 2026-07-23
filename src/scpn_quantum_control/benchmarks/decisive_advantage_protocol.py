# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Decisive advantage-benchmark protocol
"""Single-decision benchmark protocol for the Kuramoto-XY advantage question.

The broad S2 scaling protocol (:mod:`.advantage_protocol`) collects a full
size-by-baseline matrix. This module narrows that to the one comparison the
quantum-advantage gap contract requires to be *decided*: one observable, one
system size, and one accuracy target at which a QPU run either beats the best
available classical path at matched accuracy and budget, or does not.

The decision rule is fail-closed. It returns ``qpu_decides_advantage`` only when
a QPU row strictly beats a *present* best-classical row at the target size within
the accuracy and wall-time budget; every ambiguous or under-populated case
degrades to ``exact_hilbert_space_crossover_only``, ``classical_wins``, or
``inconclusive``. Per the repository's own physics (classical exact/MPS/ODE win
through the accessible NISQ range), the expected outcome of the default protocol
is a crossover or classical-wins label, never a broad-advantage claim.

References
----------
Quantum-advantage gap contract (``docs/internal/audits/contracts/
quantum_advantage_gap_contract_2026-04-30.md``): the promotion gate stays closed
until a task-matched benchmark decides one comparison at matched budget.

"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

from .advantage_protocol import (
    ScalingBaseline,
    ScalingProtocol,
    ScalingRowValidation,
    validate_scaling_rows,
)
from .gpu_baseline import estimate_qpu_time, gate_count_xy_trotter

DecisionLabel = Literal[
    "qpu_decides_advantage",
    "exact_hilbert_space_crossover_only",
    "classical_wins",
    "inconclusive",
]
"""Outcome labels, ordered strongest to weakest claim."""


@dataclass(frozen=True)
class DecisionCriterion:
    """The single observable/size/accuracy point the benchmark decides.

    Parameters
    ----------
    observable
        Metric-payload key whose value the comparison is made on (for example
        ``"order_parameter_R"``).
    target_size
        The one system size, in qubits, at which the decision is made.
    accuracy_target
        Maximum tolerated relative error (``metric_payload["reference_error"]``)
        for a row to qualify; rows above it are excluded from the decision.
    budget_wall_time_ms
        Matched total wall-time budget, in milliseconds, both paths must respect
        for a row to qualify.
    best_classical_baselines
        Baseline labels that count as the best available classical path.
    exact_baselines
        Baseline labels that are the exact reference only (dense statevector
        propagation); beating these but not the best-classical path yields a
        crossover-only label.

    """

    observable: str
    target_size: int
    accuracy_target: float
    budget_wall_time_ms: float
    best_classical_baselines: tuple[str, ...]
    exact_baselines: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the criterion invariants.

        Raises
        ------
        ValueError
            If any field is empty, non-positive, or non-finite.

        """
        if not self.observable:
            raise ValueError("observable must be non-empty")
        if self.target_size < 1:
            raise ValueError("target_size must be positive")
        if not isfinite(self.accuracy_target) or self.accuracy_target < 0.0:
            raise ValueError("accuracy_target must be finite and non-negative")
        if not isfinite(self.budget_wall_time_ms) or self.budget_wall_time_ms <= 0.0:
            raise ValueError("budget_wall_time_ms must be finite and positive")
        if not self.best_classical_baselines:
            raise ValueError("best_classical_baselines must be non-empty")
        if not self.exact_baselines:
            raise ValueError("exact_baselines must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Serialise the criterion.

        Returns
        -------
        dict
            JSON-ready mapping of every field.

        """
        return {
            "observable": self.observable,
            "target_size": self.target_size,
            "accuracy_target": self.accuracy_target,
            "budget_wall_time_ms": self.budget_wall_time_ms,
            "best_classical_baselines": list(self.best_classical_baselines),
            "exact_baselines": list(self.exact_baselines),
        }


@dataclass(frozen=True)
class SubmissionGate:
    """Depth and shot ceilings a preregistered QPU submission must not exceed.

    Parameters
    ----------
    max_circuit_depth
        Transpiled-circuit depth ceiling; the NISQ coherence wall makes deeper
        circuits noise-limited rather than decisive.
    max_total_shots
        Total-shot ceiling across the submitted circuits.

    """

    max_circuit_depth: int
    max_total_shots: int

    def __post_init__(self) -> None:
        """Validate the gate ceilings.

        Raises
        ------
        ValueError
            If either ceiling is not a positive integer.

        """
        if self.max_circuit_depth < 1:
            raise ValueError("max_circuit_depth must be positive")
        if self.max_total_shots < 1:
            raise ValueError("max_total_shots must be positive")

    def check(self, circuit_depth: int, total_shots: int) -> tuple[bool, tuple[str, ...]]:
        """Check a proposed submission against the ceilings (fail-closed).

        Parameters
        ----------
        circuit_depth
            Proposed transpiled-circuit depth.
        total_shots
            Proposed total shot count.

        Returns
        -------
        tuple of (bool, tuple of str)
            ``(passed, reasons)`` where ``passed`` is ``True`` only when both
            ceilings hold; ``reasons`` lists every breached ceiling.

        """
        reasons: list[str] = []
        if circuit_depth > self.max_circuit_depth:
            reasons.append(
                f"circuit_depth {circuit_depth} exceeds ceiling {self.max_circuit_depth}"
            )
        if total_shots > self.max_total_shots:
            reasons.append(f"total_shots {total_shots} exceeds ceiling {self.max_total_shots}")
        return (not reasons, tuple(reasons))

    def to_dict(self) -> dict[str, Any]:
        """Serialise the gate.

        Returns
        -------
        dict
            JSON-ready mapping of both ceilings.

        """
        return {
            "max_circuit_depth": self.max_circuit_depth,
            "max_total_shots": self.max_total_shots,
        }


@dataclass(frozen=True)
class DecisionOutcome:
    """The label a set of measured rows yields against a decisive protocol.

    Parameters
    ----------
    label
        The decision label (see :data:`DecisionLabel`).
    reasons
        Human-readable justifications for the label.

    """

    label: DecisionLabel
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the outcome.

        Returns
        -------
        dict
            JSON-ready mapping of the label and reasons.

        """
        return {"label": self.label, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class DecisiveAdvantageProtocol:
    """A preregistered single-decision advantage benchmark.

    Parameters
    ----------
    protocol
        The underlying :class:`~.advantage_protocol.ScalingProtocol`, restricted
        to the single decision size.
    criterion
        The observable/size/accuracy/budget decision point.
    gate
        Depth and shot ceilings for the QPU submission.
    qpu_time_estimate_s
        Conservative estimated QPU wall-clock, in seconds, for the submission.

    """

    protocol: ScalingProtocol
    criterion: DecisionCriterion
    gate: SubmissionGate
    qpu_time_estimate_s: float

    def __post_init__(self) -> None:
        """Validate cross-field consistency.

        Raises
        ------
        ValueError
            If the underlying protocol is not single-size at the decision size,
            or the QPU time estimate is not finite and positive.

        """
        if self.protocol.sizes != (self.criterion.target_size,):
            raise ValueError(
                "protocol must have exactly the decision size "
                f"{(self.criterion.target_size,)}, got {self.protocol.sizes}"
            )
        if not isfinite(self.qpu_time_estimate_s) or self.qpu_time_estimate_s <= 0.0:
            raise ValueError("qpu_time_estimate_s must be finite and positive")

    def validate_rows(self, rows: list[dict[str, Any]]) -> ScalingRowValidation:
        """Validate measured rows against the underlying scaling protocol.

        Parameters
        ----------
        rows
            Measured benchmark rows in the S2 row schema.

        Returns
        -------
        ScalingRowValidation
            The delegated validation result.

        """
        return validate_scaling_rows(self.protocol, rows)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full decisive protocol.

        Returns
        -------
        dict
            JSON-ready mapping of the protocol, criterion, gate, and estimate.

        """
        return {
            "protocol": self.protocol.to_dict(),
            "criterion": self.criterion.to_dict(),
            "gate": self.gate.to_dict(),
            "qpu_time_estimate_s": self.qpu_time_estimate_s,
        }


def _qualifying_rows(
    rows: list[dict[str, Any]],
    criterion: DecisionCriterion,
) -> list[dict[str, Any]]:
    """Return the ``ok`` rows meeting the accuracy and wall-time budget.

    Assumes ``rows`` already passed :func:`~.advantage_protocol.validate_scaling_rows`
    against the single-size protocol, so every row is at the decision size and
    carries a mapping ``metric_payload`` and a finite non-negative ``wall_time_ms``
    when ``ok``. A row still fails to qualify if it is not ``ok``, lacks a numeric
    ``reference_error``, exceeds the accuracy target, or exceeds the budget.
    """
    qualified: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        error = row["metric_payload"].get("reference_error")
        if (
            not isinstance(error, int | float)
            or not isfinite(error)
            or error > criterion.accuracy_target
        ):
            continue
        if float(row["wall_time_ms"]) > criterion.budget_wall_time_ms:
            continue
        qualified.append(row)
    return qualified


def _best_wall_time(rows: list[dict[str, Any]], labels: tuple[str, ...]) -> float | None:
    """Return the smallest wall time among rows whose baseline is in ``labels``."""
    times = [float(row["wall_time_ms"]) for row in rows if row.get("baseline") in labels]
    return min(times) if times else None


def evaluate_decision(
    protocol: DecisiveAdvantageProtocol,
    rows: list[dict[str, Any]],
) -> DecisionOutcome:
    """Decide the advantage question from measured rows (fail-closed).

    Parameters
    ----------
    protocol
        The preregistered decisive protocol.
    rows
        Measured benchmark rows in the S2 row schema.

    Returns
    -------
    DecisionOutcome
        The label and its justifications. ``qpu_decides_advantage`` is returned
        only when a QPU row strictly beats a present best-classical row at the
        target size within accuracy and budget; every ambiguous case degrades to
        a weaker label.

    """
    validation = protocol.validate_rows(rows)
    if not validation.valid:
        reasons = validation.missing_required + validation.invalid_rows
        return DecisionOutcome("inconclusive", ("rows failed protocol validation",) + reasons)

    criterion = protocol.criterion
    qualified = _qualifying_rows(rows, criterion)
    qpu_wall = _best_wall_time(qualified, ("qpu_hardware",))
    if qpu_wall is None:
        return DecisionOutcome(
            "inconclusive",
            (
                f"no qpu_hardware row at n={criterion.target_size} meets "
                f"accuracy {criterion.accuracy_target} within "
                f"{criterion.budget_wall_time_ms} ms",
            ),
        )

    classical_wall = _best_wall_time(qualified, criterion.best_classical_baselines)
    exact_wall = _best_wall_time(qualified, criterion.exact_baselines)

    if classical_wall is not None and qpu_wall < classical_wall:
        return DecisionOutcome(
            "qpu_decides_advantage",
            (
                f"qpu {qpu_wall} ms beats best classical {classical_wall} ms "
                f"at n={criterion.target_size} within budget and accuracy",
            ),
        )
    if classical_wall is not None:
        if exact_wall is not None and qpu_wall < exact_wall:
            return DecisionOutcome(
                "exact_hilbert_space_crossover_only",
                (
                    f"qpu {qpu_wall} ms beats exact {exact_wall} ms but not best "
                    f"classical {classical_wall} ms",
                ),
            )
        return DecisionOutcome(
            "classical_wins",
            (f"best classical {classical_wall} ms is at least as fast as qpu {qpu_wall} ms",),
        )
    if exact_wall is not None and qpu_wall < exact_wall:
        return DecisionOutcome(
            "exact_hilbert_space_crossover_only",
            (
                f"qpu {qpu_wall} ms beats exact {exact_wall} ms; no qualifying "
                "best-classical row present",
            ),
        )
    return DecisionOutcome(
        "inconclusive",
        ("no qualifying best-classical or exact row to decide against",),
    )


def default_decisive_advantage_protocol() -> DecisiveAdvantageProtocol:
    """Return the preregistered Kuramoto-XY decisive-advantage protocol.

    The decision is the synchronisation order parameter ``R`` at ``n = 12``
    qubits — near the documented exact-Hilbert-space crossover (``n ≈ 11.6``) yet
    inside the accessible NISQ range — at a 1 % relative-accuracy target and a
    matched wall-time budget.

    Every decision baseline produces the same dynamical observable ``R`` at the
    final evolution time, so the comparison is like-for-like: the best-classical
    path is the classical oscillator ODE and the MPS tensor-network evolution;
    the exact-only reference is the dense statevector propagation. Ground-state
    eigensolvers are deliberately excluded — they answer the spectral-gap
    question, not the dynamical order-parameter question a Trotterised QPU run
    decides, and a sparse Krylov propagation earns its keep only above the dense
    memory wall (``n ≳ 13``), which lies outside this single decision size.
    Per the repository physics the expected outcome is a crossover or
    classical-wins label, not broad advantage.

    Returns
    -------
    DecisiveAdvantageProtocol
        A fully populated, serialisable decisive protocol.

    """
    target_size = 12
    common_metrics = ("wall_time_ms", "memory_bytes", "status", "notes", "reference_error")
    baselines = (
        ScalingBaseline(
            kind="classical_ode",
            label="classical_ode",
            required=True,
            max_qubits=target_size,
            metrics=common_metrics + ("order_parameter_R",),
            claim_boundary="Classical oscillator ODE baseline; not a Hilbert-space solver.",
        ),
        ScalingBaseline(
            kind="mps_tensor_network",
            label="mps_tensor_network",
            required=True,
            max_qubits=target_size,
            metrics=common_metrics + ("order_parameter_R", "max_bond"),
            claim_boundary="MPS bounds classical spoofability at the decision size.",
        ),
        ScalingBaseline(
            kind="dense_trotter_expm",
            label="dense_statevector_evolution",
            required=True,
            max_qubits=target_size,
            metrics=common_metrics + ("order_parameter_R", "ground_energy"),
            claim_boundary=(
                "Dense statevector propagation is the exact-only dynamical reference at "
                "n=12; it produces the order parameter R, not merely a spectral quantity."
            ),
        ),
        ScalingBaseline(
            kind="qpu_hardware",
            label="qpu_hardware",
            required=False,
            max_qubits=target_size,
            metrics=common_metrics + ("order_parameter_R", "shots", "job_ids", "backend"),
            claim_boundary=(
                "Hardware column decides the comparison only for the preregistered size; "
                "absence of QPU credits degrades to a classical scaling study."
            ),
        ),
    )
    protocol = ScalingProtocol(
        protocol_id="decisive_advantage_order_parameter_n12_2026-07-15",
        sizes=(target_size,),
        baselines=baselines,
        acceptance=(
            "The qpu_hardware row, when present, is submitted within the depth and shot gate.",
            "Best-classical and exact rows are measured or size-gated skips with notes.",
            "Every timing row records command, machine, dependency versions, and git commit.",
        ),
        falsification=(
            "If the best-classical path is at least as fast as the QPU at matched accuracy, "
            "any advantage framing is rejected.",
            "If the QPU beats only the exact statevector reference, the label is "
            "crossover-only, not advantage.",
        ),
        claim_boundary=(
            "This protocol decides one comparison at one size. A single decisive size does not "
            "establish broad quantum advantage; the outcome is labelled by evaluate_decision."
        ),
        output_schema={
            "row_keys": [
                "protocol_id",
                "n_qubits",
                "baseline",
                "status",
                "wall_time_ms",
                "memory_bytes",
                "metric_payload",
                "command",
                "machine",
                "dependencies",
                "git_commit",
                "notes",
            ]
        },
    )
    criterion = DecisionCriterion(
        observable="order_parameter_R",
        target_size=target_size,
        accuracy_target=0.01,
        budget_wall_time_ms=60_000.0,
        best_classical_baselines=("classical_ode", "mps_tensor_network"),
        exact_baselines=("dense_statevector_evolution",),
    )
    total_shots = 8192
    gate = SubmissionGate(max_circuit_depth=400, max_total_shots=total_shots)
    n_gates = gate_count_xy_trotter(target_size, reps=10)
    qpu_time_estimate_s = estimate_qpu_time(target_size, n_gates) * total_shots
    return DecisiveAdvantageProtocol(
        protocol=protocol,
        criterion=criterion,
        gate=gate,
        qpu_time_estimate_s=qpu_time_estimate_s,
    )
