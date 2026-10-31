# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — competitive qualification on one frozen challenge
"""Aggregates come from every raw run, mismatched precision is not ranked, losses stay visible.

Oracles: hand-written run records with known values and costs, NumPy
recomputation of mean, sample variance and total cost from the returned raw
rows, and host-readiness verdicts built from explicit governor, frequency and
load readings. Nothing is executed on a device and nothing is written.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.benchmarks.isolated_host_readiness import (
    HostReadiness,
    assess_host_readiness,
)
from scpn_quantum_control.benchmarks.method_comparison_contracts import (
    ComparisonBudget,
    ComparisonMeaning,
    MethodRun,
)
from scpn_quantum_control.benchmarks.reproducible_comparison import (
    CAMPAIGN_CLAIM_BOUNDARY,
    IsolationEvidence,
    qualify_competitive_campaign,
)

_MEANING = ComparisonMeaning(
    problem="kuramoto n=8 T=2 dt=0.05; frozen challenge 1",
    observable="absolute final order-parameter error",
    unit="1",
    model="classical phase Kuramoto",
    precision="float64",
    shots=None,
    shot_protocol="deterministic; no shots",
    initial_condition="phases from omega",
    calibration="not-applicable: local simulation",
    noise="none",
    estimator="exact scalar",
)
_SEEDS = (11, 12, 13)
# Per method: (value or None for a failed run, execution ms) for each seed.
_PLAN: dict[str, tuple[tuple[float | None, float], ...]] = {
    "reference": ((0.020, 4.0), (0.024, 4.5), (0.022, 4.2)),
    "candidate": ((0.012, 6.0), (0.018, 6.2), (0.016, 6.5)),
    "competitor": ((0.031, 2.0), (0.029, 2.2), (0.035, 2.1)),
}
# The candidate diverges on seed 12; its measured cost is still part of the record.
_PLAN_WITH_FAILURE = {**_PLAN, "candidate": ((0.012, 6.0), (None, 9.0), (0.016, 6.5))}


def _runs(
    plan: dict[str, tuple[tuple[float | None, float], ...]] = _PLAN,
    meanings: dict[str, ComparisonMeaning] | None = None,
) -> list[MethodRun]:
    """Build one run per method and seed; a failed run keeps its measured cost."""
    runs = []
    for method, rows in plan.items():
        meaning = (meanings or {}).get(method, _MEANING)
        for seed, (value, execution_ms) in zip(_SEEDS, rows, strict=True):
            runs.append(
                MethodRun(
                    run_id=f"{method}/{seed}",
                    method=method,
                    backend="cpu",
                    seed=seed,
                    meaning=meaning,
                    budget=ComparisonBudget(1.0, 0.5, execution_ms, 4096, 1 << 20, 100.0),
                    status="completed" if value is not None else "failed",
                    value=value,
                    reason=None if value is not None else "solver diverged",
                )
            )
    return runs


def _ready_host(core: int = 2) -> HostReadiness:
    """Return a ready host verdict from explicit readings."""
    return assess_host_readiness(
        reserved_core=core,
        governor="performance",
        frequency_mhz=3600.0,
        load_average=(0.2, 0.3, 0.3),
    )


_ISOLATED = IsolationEvidence(cpu_affinity=(2,), host=_ready_host())


def test_competitive_qualification_campaign_01_aggregates_recompute_from_raw_runs() -> None:
    """Mean, spread and total cost recompute from the returned raw rows, failures included."""
    runs = _runs(_PLAN_WITH_FAILURE)
    result = qualify_competitive_campaign(
        runs,
        baseline_method="reference",
        better="lower",
        isolation=_ISOLATED,
        missing_data_policy="available-case",
    )
    comparison = result.comparison
    assert [row["run_id"] for row in comparison["runs"]] == [run.run_id for run in runs]
    for summary in comparison["methods"]:
        rows = [row for row in comparison["runs"] if row["method"] == summary["method"]]
        values = np.array(
            [row["value"] for row in rows if row["status"] == "completed"], dtype=np.float64
        )
        assert summary["population_size"] == len(rows) == 3
        assert summary["mean"] == pytest.approx(float(np.mean(values)), rel=1e-15)
        assert summary["sample_variance"] == pytest.approx(
            float(np.var(values, ddof=1)), rel=1e-12
        )
        assert summary["total_cost_ms"] == pytest.approx(
            math.fsum(cast(float, row["budget"]["total_cost_ms"]) for row in rows), rel=1e-15
        )
    candidate = next(row for row in comparison["methods"] if row["method"] == "candidate")
    assert candidate["missing_run_ids"] == ["candidate/12"]
    assert candidate["total_cost_ms"] == pytest.approx(3 * 1.5 + 6.0 + 9.0 + 6.5)
    # A method that failed on a seed the others completed is not ranked against them.
    assert result.ranking == () and result.performance_class == "functional_non_isolated"
    assert "candidate: observed seed population differs from reference" in (
        result.promotion_blockers
    )
    # The default policy refuses an aggregate with a failed run instead of dropping it.
    strict = qualify_competitive_campaign(
        runs, baseline_method="reference", better="lower", isolation=_ISOLATED
    )
    assert strict.ranking == () and strict.performance_class == "functional_non_isolated"
    assert any("aggregate unavailable" in blocker for blocker in strict.promotion_blockers)


def test_competitive_qualification_campaign_02_changed_competitor_precision_is_not_ranked() -> (
    None
):
    """A competitor measured in float32 makes the whole ranking unavailable."""
    float32 = dataclasses.replace(_MEANING, precision="float32")
    result = qualify_competitive_campaign(
        _runs(meanings={"competitor": float32}),
        baseline_method="reference",
        better="lower",
        isolation=_ISOLATED,
        missing_data_policy="available-case",
    )
    assert result.ranking == () and result.losing_methods == ()
    assert not result.comparison["ranking_allowed"]
    assert "competitor/11: precision differs from reference" in result.comparison["blockers"]
    assert all(row["difference_vs_reference"] is None for row in result.comparison["methods"])
    assert result.performance_class == "functional_non_isolated"
    # The rows themselves are still all there.
    assert len(result.comparison["runs"]) == 9


def test_competitive_qualification_campaign_03_losing_comparison_remains_present() -> None:
    """A method worse than the baseline is ranked last and listed as losing, in either direction."""
    result = qualify_competitive_campaign(
        _runs(),
        baseline_method="reference",
        better="lower",
        isolation=_ISOLATED,
        missing_data_policy="available-case",
    )
    assert result.losing_methods == ("competitor",)
    assert result.ranking[-1] == "competitor"
    competitor = next(row for row in result.comparison["methods"] if row["method"] == "competitor")
    assert competitor["difference_vs_reference"] == pytest.approx(0.095 / 3 - 0.022, rel=1e-12)
    # With "higher is better" the same numbers make the candidate the loser.
    flipped = qualify_competitive_campaign(
        _runs(),
        baseline_method="reference",
        better="higher",
        isolation=_ISOLATED,
        missing_data_policy="available-case",
    )
    assert flipped.ranking == ("competitor", "reference", "candidate")
    assert flipped.losing_methods == ("candidate",)
    assert flipped.claim_boundary == CAMPAIGN_CLAIM_BOUNDARY


def test_competitive_qualification_campaign_04_isolation_needs_affinity_and_load() -> None:
    """Missing affinity or load evidence keeps the ranking but forbids isolated promotion."""
    runs = _runs()

    def qualify(isolation: IsolationEvidence | None) -> tuple[str, tuple[str, ...]]:
        """Return the performance class and blockers for one isolation record."""
        result = qualify_competitive_campaign(
            runs,
            baseline_method="reference",
            better="lower",
            isolation=isolation,
            missing_data_policy="available-case",
        )
        assert result.ranking == ("candidate", "reference", "competitor")
        return result.performance_class, result.promotion_blockers

    assert qualify(_ISOLATED) == ("isolated_affinity", ())
    no_evidence = qualify(None)
    assert no_evidence == (
        "functional_non_isolated",
        ("no isolation evidence was recorded for the runs",),
    )
    no_affinity = qualify(IsolationEvidence(cpu_affinity=None, host=_ready_host()))
    assert no_affinity == ("functional_non_isolated", ("CPU affinity was not recorded",))
    unpinned = qualify(IsolationEvidence(cpu_affinity=(0, 1, 2, 3), host=_ready_host()))
    assert "not only the reserved core 2" in unpinned[1][0]
    no_load = assess_host_readiness(
        reserved_core=2, governor="performance", frequency_mhz=3600.0, load_average=None
    )
    assert qualify(IsolationEvidence(cpu_affinity=(2,), host=no_load)) == (
        "functional_non_isolated",
        ("host load average is unavailable",),
    )
    busy = assess_host_readiness(
        reserved_core=2, governor="powersave", frequency_mhz=3600.0, load_average=(3.0, 2.0, 1.0)
    )
    busy_class, busy_blockers = qualify(IsolationEvidence(cpu_affinity=(2,), host=busy))
    assert busy_class == "functional_non_isolated" and len(busy_blockers) == 2
    assert qualify(IsolationEvidence(cpu_affinity=(2,), host=None)) == (
        "functional_non_isolated",
        ("host governor, frequency and load were not recorded",),
    )
    unexplained = dataclasses.replace(_ready_host(), ready=False)
    assert qualify(IsolationEvidence(cpu_affinity=(2,), host=unexplained))[1] == (
        "host was not ready for isolated benchmarking",
    )
    # An incomplete cost basis also blocks promotion, whatever the host evidence.
    unknown_cost = [
        dataclasses.replace(run, budget=ComparisonBudget(None, 0.5, 4.0, 4096, 1 << 20, 100.0))
        if run.run_id == "competitor/13"
        else run
        for run in runs
    ]
    result = qualify_competitive_campaign(
        unknown_cost,
        baseline_method="reference",
        better="lower",
        isolation=_ISOLATED,
        missing_data_policy="available-case",
    )
    assert result.ranking == () and result.performance_class == "functional_non_isolated"
    assert result.promotion_blockers[0] == "cost comparison is not allowed for this population"


def test_campaign_rejects_malformed_inputs() -> None:
    """Unknown directions and malformed CPU sets are refused before any comparison."""
    with pytest.raises(ValueError, match="better must be"):
        qualify_competitive_campaign(
            _runs(), baseline_method="reference", better=cast(Any, "faster"), isolation=None
        )
    for affinity in ((), (1, 1), (-1,), (True,)):
        with pytest.raises(ValueError, match="distinct non-negative CPU indices"):
            IsolationEvidence(cpu_affinity=affinity, host=None)
    with pytest.raises(ValueError, match="baseline method is absent"):
        qualify_competitive_campaign(
            _runs(), baseline_method="missing", better="lower", isolation=None
        )
