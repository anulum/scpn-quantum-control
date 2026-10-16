# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Retained population method comparison tests
"""Public method-comparison regressions with hand-computable independent oracles."""

from dataclasses import replace

import pytest

from scpn_quantum_control.benchmarks.method_comparison import summarise_method_comparison
from scpn_quantum_control.benchmarks.method_comparison_contracts import (
    ComparisonBudget,
    ComparisonMeaning,
    MethodRun,
)


def run(run_id: str, value: float | None, *, method: str = "reference") -> MethodRun:
    """Declare a single retained repetition without solver or estimator mocks."""
    return MethodRun(
        run_id=run_id,
        method=method,
        backend="native-classical",
        seed=int(run_id[-1]),
        meaning=ComparisonMeaning(
            problem="analytic-difference-v1",
            observable="signed difference",
            unit="1",
            model="declared analytic fixture",
            precision="float64",
            shots=None,
            shot_protocol="deterministic",
            initial_condition="fixed",
            calibration="not-applicable",
            noise="none",
            estimator="exact scalar",
        ),
        budget=ComparisonBudget(2.0, 3.0, 5.0, 256, 1024, 100.0),
        status="completed" if value is not None else "failed",
        value=value,
        reason=None if value is not None else "retained failure",
    )


def test_method_comparison_instrument_01() -> None:
    """Different precision or model refuses ranking while retaining both rows."""
    baseline = run("r0", 1.0)
    for field, value in [("precision", "float32"), ("model", "different physics")]:
        candidate = run("c1", 1.0, method="candidate")
        candidate = replace(
            candidate,
            meaning=(
                replace(candidate.meaning, precision=value)
                if field == "precision"
                else replace(candidate.meaning, model=value)
            ),
        )
        result = summarise_method_comparison((baseline, candidate), baseline_method="reference")
        assert result["ranking_allowed"] is False
        assert field in " ".join(result["blockers"])
        assert len(result["runs"]) == 2
        assert result["methods"][1]["difference_vs_reference"] is None


def test_method_comparison_instrument_02() -> None:
    """All available declared repetitions contribute; missing rows stay visible."""
    population = (run("r0", -1.0), run("r1", 1.0), run("r2", 3.0), run("r3", None))
    result = summarise_method_comparison(
        population, baseline_method="reference", missing_data_policy="available-case"
    )
    summary = result["methods"][0]
    assert summary["mean"] == 1.0
    assert summary["sample_variance"] == 4.0
    assert summary["population_size"] == 4
    assert summary["observed_size"] == 3
    assert summary["missing_run_ids"] == ["r3"]
    assert [row["value"] for row in result["runs"]] == [-1.0, 1.0, 3.0, None]
    complete = summarise_method_comparison(population, baseline_method="reference")
    assert complete["methods"][0]["mean"] is None
    assert complete["ranking_allowed"] is False


def test_method_comparison_instrument_03() -> None:
    """Full cost includes compilation, transfer and execution of failed attempts."""
    original = (run("r0", 1.0), run("r1", None))
    result = summarise_method_comparison(original, baseline_method="reference")
    assert result["methods"][0]["total_cost_ms"] == 20.0
    assert result["runs"][0]["budget"]["total_cost_ms"] == 10.0
    unknown = replace(original[0], budget=replace(original[0].budget, transfer_ms=None))
    result = summarise_method_comparison((unknown,), baseline_method="reference")
    assert result["methods"][0]["total_cost_ms"] is None
    assert result["cost_comparison_allowed"] is False
    exceeded = replace(original[0], budget=replace(original[0].budget, wall_limit_ms=9.0))
    result = summarise_method_comparison((exceeded,), baseline_method="reference")
    assert result["cost_comparison_allowed"] is False
    assert "budget" in " ".join(result["blockers"])


def test_method_comparison_instrument_04() -> None:
    """Selecting the best seed cannot replace the complete retained-population mean."""
    population = (run("r0", -1.0), run("r1", 1.0), run("r2", 3.0))
    original = summarise_method_comparison(population, baseline_method="reference")
    selected = summarise_method_comparison(
        population, baseline_method="reference", selected_run_id="r2"
    )
    assert original["methods"] == selected["methods"]
    assert selected["methods"][0]["mean"] == 1.0
    assert selected["selected_run_id"] == "r2"
    assert selected["runs"] == original["runs"]
    with pytest.raises(ValueError, match="selected"):
        summarise_method_comparison(
            population, baseline_method="reference", selected_run_id="absent"
        )


def test_population_identity_refusal_preserves_input() -> None:
    """Duplicate identity and unknown baseline refuse before producing a projection."""
    population = (run("r0", 1.0),)
    before = repr(population)
    with pytest.raises(ValueError, match="unique"):
        summarise_method_comparison(population * 2, baseline_method="reference")
    with pytest.raises(ValueError, match="baseline"):
        summarise_method_comparison(population, baseline_method="absent")
    assert repr(population) == before


def test_matching_populations_report_all_run_difference() -> None:
    """Same-seed native fixture means use every repetition, not a lucky seed."""
    population = (
        run("r0", -1.0),
        run("r1", 1.0),
        run("c0", 1.0, method="candidate"),
        run("c1", 3.0, method="candidate"),
    )
    result = summarise_method_comparison(population, baseline_method="reference")
    assert result["ranking_allowed"] is True
    assert result["cost_comparison_allowed"] is True
    assert result["methods"][0]["mean"] == 0.0
    assert result["methods"][1]["mean"] == 2.0
    assert result["methods"][1]["sample_variance"] == 2.0
    assert result["methods"][1]["difference_vs_reference"] == 2.0
    assert result["population_run_ids"] == ["r0", "r1", "c0", "c1"]


def test_partial_seed_population_and_all_missing_results_refuse_ranking() -> None:
    """An omitted candidate seed or no observed reference never authorises ranking."""
    result = summarise_method_comparison(
        (run("r0", 1.0), run("r1", 2.0), run("c0", 5.0, method="candidate")),
        baseline_method="reference",
    )
    assert result["ranking_allowed"] is False
    assert "seed population" in " ".join(result["blockers"])
    missing = summarise_method_comparison(
        (run("r0", None),), baseline_method="reference", missing_data_policy="available-case"
    )
    assert missing["methods"][0]["observed_size"] == 0
    assert missing["methods"][0]["mean"] is None


def test_original_interval_is_projected_without_an_aggregate_confidence_claim() -> None:
    """A genuine binomial analytic fixture retains its source estimator interval."""
    import math
    from dataclasses import asdict

    from scpn_quantum_control.analysis.sync_uncertainty import order_parameter_shot_noise

    interval = order_parameter_shot_noise({"0": 75, "1": 25})
    assert interval.point == 0.5
    assert interval.standard_error == pytest.approx(math.sqrt(3 / 396))
    original = run("r0", interval.point)
    original = replace(
        original,
        interval=interval,
        meaning=replace(original.meaning, shots=100, estimator=interval.method),
    )
    result = summarise_method_comparison((original,), baseline_method="reference")
    assert result["runs"][0]["interval"] == asdict(interval)
    assert result["methods"][0]["sample_variance"] is None
    assert "not an aggregate confidence interval" in result["claim_boundary"]


def test_explicit_population_limits_and_unknown_missing_policy_refuse() -> None:
    """Empty/oversize populations and undeclared imputation policies are rejected."""
    from typing import Any

    for population in [(), (run("r0", 1.0),) * 1001]:
        with pytest.raises(ValueError, match="between"):
            summarise_method_comparison(population, baseline_method="reference")
    malformed: dict[str, Any] = {"missing_data_policy": "silently-impute-zero"}
    with pytest.raises(ValueError, match="missing-data"):
        summarise_method_comparison((run("r0", 1.0),), baseline_method="reference", **malformed)


def test_large_finite_values_retain_unavailable_variance_and_refuse_overflowing_difference() -> (
    None
):
    """Finite extreme observations never become an infinite variance or difference."""
    original = summarise_method_comparison(
        (run("r0", -1e308), run("r1", 1e308)), baseline_method="reference"
    )
    assert original["methods"][0]["mean"] == 0.0
    assert original["methods"][0]["sample_variance"] is None
    result = summarise_method_comparison(
        (run("r0", -1e308), run("c0", 1e308, method="candidate")), baseline_method="reference"
    )
    assert result["ranking_allowed"] is False
    assert "overflows" in " ".join(result["blockers"])
    assert all(row["difference_vs_reference"] is None for row in result["methods"])


def test_all_attempt_cost_overflow_remains_unavailable() -> None:
    """Each finite attempt may fit its ceiling while their total exceeds binary64."""
    budget = ComparisonBudget(0.0, 0.0, 1e308, 1, 1, 1e308)
    population = (replace(run("r0", 1.0), budget=budget), replace(run("r1", 2.0), budget=budget))
    result = summarise_method_comparison(population, baseline_method="reference")
    assert result["methods"][0]["total_cost_ms"] is None
    assert result["cost_comparison_allowed"] is False


def test_different_original_interval_confidence_refuses_comparison() -> None:
    """Matching points cannot hide an incompatible nominal confidence protocol."""
    from scpn_quantum_control.analysis.sync_uncertainty import order_parameter_shot_noise

    baseline_interval = order_parameter_shot_noise({"0": 75, "1": 25}, coverage=0.95)
    candidate_interval = order_parameter_shot_noise({"0": 75, "1": 25}, coverage=0.9)
    baseline = run("r0", baseline_interval.point)
    declaration = replace(baseline.meaning, shots=100, estimator=baseline_interval.method)
    baseline = replace(baseline, meaning=declaration, interval=baseline_interval)
    candidate = replace(
        run("c0", candidate_interval.point, method="candidate"),
        meaning=declaration,
        interval=candidate_interval,
    )
    result = summarise_method_comparison((baseline, candidate), baseline_method="reference")
    assert result["ranking_allowed"] is False
    assert "uncertainty" in " ".join(result["blockers"])
    assert result["methods"][1]["difference_vs_reference"] is None


def test_available_case_does_not_rank_different_observed_seed_subsets() -> None:
    """Different failure patterns stay visible and cannot silently create a fair pair."""
    population = (
        run("r0", -1.0),
        run("r1", 1.0),
        run("c0", 9.0, method="candidate"),
        run("c1", None, method="candidate"),
    )
    result = summarise_method_comparison(
        population, baseline_method="reference", missing_data_policy="available-case"
    )
    assert result["methods"][1]["mean"] == 9.0
    assert result["methods"][1]["missing_run_ids"] == ["c1"]
    assert result["ranking_allowed"] is False
    assert "observed seed" in " ".join(result["blockers"])
