# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Immutable retained-population comparison contracts
"""Explicit scientific meanings, measured costs and original repetition records.

These records grant no execution authority. Unknown costs stay unknown, failed
attempts stay in the population, and existing estimator intervals retain their
original definition. No solver or confidence method is implemented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from ..analysis.sync_uncertainty import UncertaintyInterval

MAX_COMPARISON_RUNS = 1000


def _text(name: str, value: object) -> None:
    """Reject an empty, excessively long or non-string identity."""
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ValueError(f"{name} requires a nonempty bounded string")


def _number(name: str, value: float | None) -> None:
    """Validate a nonnegative finite measurement while preserving unknown values."""
    if value is None:
        return
    try:
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0.0
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} requires a nonnegative finite number or None")


def _integer(name: str, value: int | None) -> None:
    """Validate an explicitly integral nonnegative count without float coercion."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{name} requires a nonnegative integer or None")


@dataclass(frozen=True)
class ComparisonMeaning:
    """Scientific meanings that must agree before comparing scalar aggregates.

    Parameters
    ----------
    problem, observable, unit, model : str
        Explicit source problem, estimand, unit and generating-model identities.
    precision : str
        Original arithmetic precision, never inferred from rendered digits.
    shots : int or None
        Declared per-repetition shot count; None means no finite-shot protocol.
    shot_protocol, initial_condition, calibration, noise, estimator : str
        Original measurement, initial-state, calibration/noise and estimator
        meanings. Explicit not-applicable strings are allowed; absence is not.

    """

    problem: str
    observable: str
    unit: str
    model: str
    precision: str
    shots: int | None
    shot_protocol: str
    initial_condition: str
    calibration: str
    noise: str
    estimator: str

    def __post_init__(self) -> None:
        """Reject incomplete meanings and malformed shot counts."""
        for name, value in self.__dict__.items():
            if name != "shots":
                _text(name, value)
        _integer("shots", self.shots)


class BudgetRecord(TypedDict):
    """Wire projection of actual cost components and declared resource bounds."""

    compilation_ms: float | None
    transfer_ms: float | None
    execution_ms: float | None
    peak_bytes: int | None
    memory_limit_bytes: int | None
    wall_limit_ms: float | None
    observed_wall_ms: float | None
    total_cost_ms: float | None
    within_budget: bool


@dataclass(frozen=True)
class ComparisonBudget:
    """Full measured time decomposition and original resource ceilings.

    Parameters
    ----------
    compilation_ms, transfer_ms, execution_ms : float or None
        Actual separately declared costs in milliseconds. Unknown is None;
        known absence of a component is explicitly zero.
    peak_bytes, memory_limit_bytes : int or None
        Observed peak and declared memory ceiling, in bytes.
    wall_limit_ms : float or None
        Declared total wall-time ceiling, in milliseconds.
    observed_wall_ms : float or None
        Original advisory timer when component allocation is unavailable. It
        never fills an unknown compilation, transfer or execution component.

    """

    compilation_ms: float | None
    transfer_ms: float | None
    execution_ms: float | None
    peak_bytes: int | None
    memory_limit_bytes: int | None
    wall_limit_ms: float | None
    observed_wall_ms: float | None = None

    def __post_init__(self) -> None:
        """Validate measurements and ceilings without filling missing components."""
        for name in (
            "compilation_ms",
            "transfer_ms",
            "execution_ms",
            "wall_limit_ms",
            "observed_wall_ms",
        ):
            _number(name, getattr(self, name))
        _integer("peak_bytes", self.peak_bytes)
        _integer("memory_limit_bytes", self.memory_limit_bytes)

    @property
    def total_cost_ms(self) -> float | None:
        """Sum all three known time components, retaining unknown or overflow."""
        values = (self.compilation_ms, self.transfer_ms, self.execution_ms)
        if any(value is None for value in values):
            return None
        try:
            return math.fsum(value for value in values if value is not None)
        except OverflowError:
            return None

    @property
    def within_budget(self) -> bool:
        """Whether both original time and memory bounds are known and satisfied."""
        total = self.total_cost_ms
        return (
            total is not None
            and self.wall_limit_ms is not None
            and total <= self.wall_limit_ms
            and self.peak_bytes is not None
            and self.memory_limit_bytes is not None
            and self.peak_bytes <= self.memory_limit_bytes
        )

    def to_dict(self) -> BudgetRecord:
        """Project exact original components alongside the qualified total."""
        return {
            "compilation_ms": self.compilation_ms,
            "transfer_ms": self.transfer_ms,
            "execution_ms": self.execution_ms,
            "peak_bytes": self.peak_bytes,
            "memory_limit_bytes": self.memory_limit_bytes,
            "wall_limit_ms": self.wall_limit_ms,
            "observed_wall_ms": self.observed_wall_ms,
            "total_cost_ms": self.total_cost_ms,
            "within_budget": self.within_budget,
        }


@dataclass(frozen=True)
class MethodRun:
    """One original repetition, including failures and negative observations.

    Parameters
    ----------
    run_id, method, backend : str
        Stable original repetition, algorithm and execution-backend identities.
    seed : int
        Exact integer seed recorded by the producer; never a selected-best flag.
    meaning : ComparisonMeaning
        Complete comparability declaration for this particular row.
    budget : ComparisonBudget
        Original measured cost and declared total resource bounds.
    status : {"completed", "failed", "missing"}
        Original availability state; only completed rows carry a finite value.
    value : float or None
        Original finite scalar, retaining negative results, or explicit absence.
    reason : str or None
        Required failure/missing reason; None for a completed row.
    interval : UncertaintyInterval or None
        Unmodified interval from analysis.sync_uncertainty, or no interval.

    """

    run_id: str
    method: str
    backend: str
    seed: int
    meaning: ComparisonMeaning
    budget: ComparisonBudget
    status: Literal["completed", "failed", "missing"]
    value: float | None
    reason: str | None = None
    interval: UncertaintyInterval | None = None

    def __post_init__(self) -> None:
        """Require honest status/value/interval associations and explicit identity."""
        for name in ("run_id", "method", "backend"):
            _text(name, getattr(self, name))
        _integer("seed", self.seed)
        if self.seed is None:
            raise ValueError("seed must be an explicitly recorded integer")
        if self.status not in ("completed", "failed", "missing"):
            raise ValueError("status must be completed, failed or missing")
        if self.status == "completed":
            try:
                finite = (
                    not isinstance(self.value, bool)
                    and isinstance(self.value, (int, float))
                    and math.isfinite(self.value)
                )
            except OverflowError:
                finite = False
            if not finite or self.reason is not None:
                raise ValueError("completed row requires a finite value and no failure reason")
        else:
            if self.value is not None or self.interval is not None:
                raise ValueError("unavailable row cannot carry a scalar or interval")
            _text("reason", self.reason)
        if self.interval is not None:
            interval = self.interval
            if (
                interval.method not in ("shot-noise-delta", "bootstrap-percentile")
                or interval.method != self.meaning.estimator
                or (
                    interval.method == "shot-noise-delta"
                    and (interval.n_shots < 2 or interval.n_resamples != 0)
                )
                or (interval.method == "bootstrap-percentile" and interval.n_resamples < 1)
                or interval.point != self.value
                or interval.n_shots != self.meaning.shots
                or interval.n_shots < 1
                or not 0 < interval.coverage < 1
                or not all(
                    math.isfinite(v)
                    for v in (interval.point, interval.low, interval.high, interval.standard_error)
                )
                or interval.low > interval.high
                or interval.standard_error < 0
                or interval.n_resamples < 0
            ):
                raise ValueError("interval must retain its original estimator, point and shots")


class IntervalRecord(TypedDict):
    """Original estimator interval; nominal coverage is not a coverage proof."""

    point: float
    standard_error: float
    low: float
    high: float
    coverage: float
    method: str
    n_shots: int
    n_resamples: int


class RunRecord(TypedDict):
    """Complete retained repetition projection with exact producer metadata."""

    run_id: str
    method: str
    backend: str
    seed: int
    meaning: dict[str, str | int | None]
    budget: BudgetRecord
    status: str
    value: float | None
    reason: str | None
    interval: IntervalRecord | None


class SummaryRecord(TypedDict):
    """Descriptive aggregate over the declared population under its missing policy."""

    method: str
    population_size: int
    observed_size: int
    included_run_ids: list[str]
    missing_run_ids: list[str]
    mean: float | None
    sample_variance: float | None
    total_cost_ms: float | None
    difference_vs_reference: float | None


class MethodComparisonRecord(TypedDict):
    """Versioned native projection; selection cannot change an aggregate."""

    schema: str
    baseline_method: str
    missing_data_policy: str
    population_run_ids: list[str]
    selected_run_id: str | None
    ranking_allowed: bool
    cost_comparison_allowed: bool
    blockers: list[str]
    methods: list[SummaryRecord]
    runs: list[RunRecord]
    claim_boundary: str
