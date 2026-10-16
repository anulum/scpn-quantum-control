# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Source-owned method comparison population summaries
"""Read-only summaries of every declared repetition, including failed attempts.

Mean and sample variance are descriptive statistics of the declared observed
population. Existing per-run estimator intervals remain separate; no aggregate
confidence definition, scientific-equivalence or advantage claim is introduced.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, fields
from typing import Literal, cast

from .method_comparison_contracts import (
    MAX_COMPARISON_RUNS,
    IntervalRecord,
    MethodComparisonRecord,
    MethodRun,
    RunRecord,
    SummaryRecord,
)

METHOD_COMPARISON_SCHEMA = "studio.method_comparison.v1"
COMPARISON_CLAIM_BOUNDARY = (
    "Descriptive comparison of the complete declared retained population under "
    "an explicit missing-data policy; original per-run estimator intervals and "
    "nominal coverage are not an aggregate confidence interval or coverage proof. "
    "Matching declarations do not prove producer authenticity, model equivalence, "
    "hardware performance, statistical significance or quantum advantage."
)


def _run_record(run: MethodRun) -> RunRecord:
    """Project original values and interval fields without numerical recomputation."""
    interval: IntervalRecord | None = None
    if run.interval is not None:
        original = run.interval
        interval = {
            "point": original.point,
            "standard_error": original.standard_error,
            "low": original.low,
            "high": original.high,
            "coverage": original.coverage,
            "method": original.method,
            "n_shots": original.n_shots,
            "n_resamples": original.n_resamples,
        }
    return {
        "run_id": run.run_id,
        "method": run.method,
        "backend": run.backend,
        "seed": run.seed,
        "meaning": asdict(run.meaning),
        "budget": run.budget.to_dict(),
        "status": run.status,
        "value": run.value,
        "reason": run.reason,
        "interval": interval,
    }


def _interval_protocol(run: MethodRun) -> tuple[str, float, int] | None:
    """Return existing estimator/nominal coverage/resampling meanings, never infer them."""
    interval = run.interval
    if interval is None:
        return None
    return interval.method, interval.coverage, interval.n_resamples


def _summary(
    method: str,
    runs: Sequence[MethodRun],
    missing_data_policy: Literal["require-complete", "available-case"],
) -> SummaryRecord:
    """Aggregate every observed value with the declared missing-value denominator."""
    observed = [run for run in runs if run.status == "completed"]
    missing = [run.run_id for run in runs if run.status != "completed"]
    values = [run.value for run in observed if run.value is not None]
    allowed = bool(values) and (not missing or missing_data_policy == "available-case")
    mean = float(statistics.mean(values)) if allowed else None
    variance: float | None = None
    if allowed and len(values) > 1:
        try:
            variance = float(statistics.variance(values))
        except OverflowError:
            variance = None
    costs = [run.budget.total_cost_ms for run in runs]
    total: float | None = None
    if all(cost is not None for cost in costs):
        try:
            total = math.fsum(cost for cost in costs if cost is not None)
        except OverflowError:
            total = None
    return {
        "method": method,
        "population_size": len(runs),
        "observed_size": len(observed),
        "included_run_ids": [run.run_id for run in observed],
        "missing_run_ids": missing,
        "mean": mean,
        "sample_variance": variance,
        "total_cost_ms": total,
        "difference_vs_reference": None,
    }


def summarise_method_comparison(
    runs: Sequence[MethodRun],
    *,
    baseline_method: str,
    missing_data_policy: Literal["require-complete", "available-case"] = "require-complete",
    selected_run_id: str | None = None,
) -> MethodComparisonRecord:
    """Project complete retained populations, matched meanings and honest costs.

    Parameters
    ----------
    runs : sequence of MethodRun
        Complete ordered declared population, including explicit missing/failed
        repetitions. Between one and 1000 unique run identities are admitted.
    baseline_method : str
        Explicit existing reference method; never the selected best seed.
    missing_data_policy : {"require-complete", "available-case"}
        Default requires every repetition. Available-case explicitly reports
        observed and missing counts; no imputation or zero filling occurs.
    selected_run_id : str or None
        Display-only selection, retained in output without changing aggregates.

    Returns
    -------
    MethodComparisonRecord
        Versioned original rows, descriptive means/sample variances, original
        intervals and full known costs. Incompatible meanings, seed populations,
        unavailable aggregates or unknown/exceeded budgets refuse comparison.

    Raises
    ------
    ValueError
        If population identities, policy, baseline or display selection is bad.

    Notes
    -----
    Sample variance uses the n-1 denominator; one observation has no estimated
    variance. There is no new confidence interval for a repeated-run mean.
    Every input record is immutable, and this function performs no I/O.

    """
    population = tuple(runs)
    if not 1 <= len(population) <= MAX_COMPARISON_RUNS:
        raise ValueError("comparison requires between 1 and 1000 retained runs")
    identities = [run.run_id for run in population]
    if len(set(identities)) != len(identities):
        raise ValueError("retained run identities must be unique")
    if missing_data_policy not in ("require-complete", "available-case"):
        raise ValueError("unknown missing-data policy")
    if selected_run_id is not None and selected_run_id not in identities:
        raise ValueError("selected run is absent from the retained population")
    groups: dict[str, list[MethodRun]] = {}
    for run in population:
        groups.setdefault(run.method, []).append(run)
    if baseline_method not in groups:
        raise ValueError("baseline method is absent from the retained population")
    reference = groups[baseline_method]
    meanings = fields(reference[0].meaning)
    blockers: list[str] = []
    for run in population:
        for field in meanings:
            if getattr(run.meaning, field.name) != getattr(reference[0].meaning, field.name):
                blockers.append(f"{run.run_id}: {field.name} differs from reference")
        if _interval_protocol(run) != _interval_protocol(reference[0]):
            blockers.append(f"{run.run_id}: uncertainty protocol differs from reference")
    for method, group in groups.items():
        if sorted(run.seed for run in group) != sorted(run.seed for run in reference):
            blockers.append(f"{method}: declared seed population differs from reference")
        if sorted(run.seed for run in group if run.status == "completed") != sorted(
            run.seed for run in reference if run.status == "completed"
        ):
            blockers.append(f"{method}: observed seed population differs from reference")
    summaries = [_summary(method, group, missing_data_policy) for method, group in groups.items()]
    for summary in summaries:
        if summary["mean"] is None:
            blockers.append(
                f"{summary['method']}: aggregate unavailable under {missing_data_policy}"
            )
    cost_allowed = all(run.budget.within_budget for run in population)
    cost_allowed = cost_allowed and all(row["total_cost_ms"] is not None for row in summaries)
    if not cost_allowed:
        blockers.append("Full cost or declared resource budget is unavailable or exceeded")
    if len(groups) < 2:
        blockers.append("At least two declared methods are needed for a comparison")
    if not blockers:
        # The preceding all-summary availability check proves these means finite.
        baseline = cast(
            float, next(row for row in summaries if row["method"] == baseline_method)["mean"]
        )
        for summary in summaries:
            value = cast(float, summary["mean"])
            difference = value - baseline
            if math.isfinite(difference):
                summary["difference_vs_reference"] = difference
            else:
                blockers.append(f"{summary['method']}: difference overflows original precision")
    if blockers:
        for summary in summaries:
            summary["difference_vs_reference"] = None
    return {
        "schema": METHOD_COMPARISON_SCHEMA,
        "baseline_method": baseline_method,
        "missing_data_policy": missing_data_policy,
        "population_run_ids": identities,
        "selected_run_id": selected_run_id,
        "ranking_allowed": not blockers,
        "cost_comparison_allowed": cost_allowed and not blockers,
        "blockers": list(dict.fromkeys(blockers)),
        "methods": summaries,
        "runs": [_run_record(run) for run in population],
        "claim_boundary": COMPARISON_CLAIM_BOUNDARY,
    }
