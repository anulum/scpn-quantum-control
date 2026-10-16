# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Original native method-comparison artifact boundaries
"""Public native input/export boundary tests; no imported summary is trusted."""

from dataclasses import asdict
from typing import cast

import pytest

from scpn_quantum_control.benchmarks.method_comparison_contracts import (
    ComparisonBudget,
    ComparisonMeaning,
    MethodRun,
)
from scpn_quantum_control.studio.method_comparison_artifact import (
    export_method_comparison,
    read_method_comparison_population,
)
from scpn_quantum_control.studio_workspace.canonical import canonical_digest
from scpn_quantum_control.studio_workspace.json_transport import read_json, write_json


@pytest.fixture
def native_input() -> dict[str, object]:
    """Independent declared population: every seed/reason/measurement is explicit."""
    meaning = ComparisonMeaning(
        "original-problem",
        "signed scalar",
        "1",
        "analytic",
        "float64",
        None,
        "none",
        "fixed",
        "none",
        "none",
        "exact",
    )
    budget = ComparisonBudget(2.0, 3.0, 5.0, 2**53 + 1, 2**54, 10.0)
    runs = [
        MethodRun("r0", "reference", "native", 0, meaning, budget, "completed", -0.0),
        MethodRun("c0", "candidate", "native", 0, meaning, budget, "completed", -1.0),
    ]
    return {
        "schema": "studio.method_comparison_input.v1",
        "runs": [asdict(run) for run in runs],
        "baseline_method": "reference",
        "missing_data_policy": "require-complete",
        "selected_run_id": None,
    }


def test_export_uses_original_all_run_native_summary_and_exact_codec(
    native_input: dict[str, object],
) -> None:
    """Lossless source input and full native output share the original canonical digest."""
    original = write_json(native_input)
    population, baseline, policy, selected = read_method_comparison_population(original)
    text = export_method_comparison(
        population, baseline_method=baseline, missing_data_policy=policy, selected_run_id=selected
    )
    payload = cast(dict[str, object], read_json(text))
    digest = payload.pop("sha256")
    assert canonical_digest("studio.method_comparison.v1", payload) == digest
    body = cast(dict[str, object], payload["body"])
    assert (
        body["producer"]
        == "scpn_quantum_control.benchmarks.method_comparison.summarise_method_comparison"
    )
    comparison = cast(dict[str, object], body["comparison"])
    assert comparison["ranking_allowed"] is True
    summaries = cast(list[dict[str, object]], comparison["methods"])
    assert summaries[1]["mean"] == -1.0
    assert summaries[1]["difference_vs_reference"] == -1.0
    assert "9007199254740993" in text
    assert '"value":-0.0' in text
    assert write_json(native_input) == original


@pytest.mark.parametrize(
    "mutation", ["future", "duplicate", "nonfinite", "unknown-field", "precomputed"]
)
def test_bad_input_refuses_without_modifying_original(
    native_input: dict[str, object],
    mutation: str,
) -> None:
    """Unknown versions, malformed rows and supplied best-run summaries are refused."""
    original = write_json(native_input)
    if mutation == "future":
        native_input["schema"] = "studio.method_comparison_input.v2"
    elif mutation == "duplicate":
        rows = cast(list[dict[str, object]], native_input["runs"])
        rows.append(rows[0])
    elif mutation == "nonfinite":
        with pytest.raises(ValueError):
            read_method_comparison_population(original.replace('"value":-1.0', '"value":NaN'))
        return
    elif mutation == "unknown-field":
        cast(list[dict[str, object]], native_input["runs"])[0]["replaced_seed"] = True
    else:
        native_input["summary"] = {"best_seed": 0}
    changed = write_json(native_input)
    with pytest.raises(ValueError):
        read_method_comparison_population(changed)
    assert write_json(native_input) == changed


def test_unknown_costs_missing_rows_and_display_selection_round_trip(
    native_input: dict[str, object],
) -> None:
    """Unknown resources and failure reasons survive, and selected row is display-only."""
    rows = cast(list[dict[str, object]], native_input["runs"])
    rows[1]["status"], rows[1]["value"], rows[1]["reason"] = "missing", None, "original missing"
    budget = cast(dict[str, object], rows[1]["budget"])
    for key in budget:
        budget[key] = None
    budget["observed_wall_ms"] = 25.0
    native_input["missing_data_policy"] = "available-case"
    native_input["selected_run_id"] = "r0"
    runs, baseline, policy, selected = read_method_comparison_population(write_json(native_input))
    assert selected == "r0"
    assert runs[1].reason == "original missing"
    assert runs[1].budget.observed_wall_ms == 25.0
    assert runs[1].budget.compilation_ms is None
    assert runs[1].budget.memory_limit_bytes is None
    assert '"mean":null' in export_method_comparison(
        runs, baseline_method=baseline, missing_data_policy=policy, selected_run_id=selected
    )


def test_original_count_interval_input_uses_existing_estimator_owner(
    native_input: dict[str, object],
) -> None:
    """All original interval fields and integral shots survive source-bound export."""
    from scpn_quantum_control.analysis.sync_uncertainty import order_parameter_shot_noise

    rows = cast(list[dict[str, object]], native_input["runs"])
    interval = order_parameter_shot_noise({"0": 75, "1": 25})
    for row in rows:
        row["value"], row["interval"] = interval.point, asdict(interval)
        meaning = cast(dict[str, object], row["meaning"])
        meaning["shots"], meaning["estimator"] = 100, interval.method
    runs, baseline, policy, selected = read_method_comparison_population(write_json(native_input))
    assert runs[0].interval == interval
    assert runs[0].meaning.shots == 100
    assert '"n_shots":100' in export_method_comparison(
        runs, baseline_method=baseline, missing_data_policy=policy, selected_run_id=selected
    )


@pytest.mark.parametrize(
    "section,field,bad",
    [
        ("run", "run_id", ""),
        ("run", "seed", True),
        ("run", "value", []),
        ("run", "value", True),
        ("meaning", "shots", 1.5),
        ("meaning", "model", 42),
        ("budget", "execution_ms", "5"),
        ("budget", "compilation_ms", True),
        pytest.param("budget", "transfer_ms", 10**1000, id="cost-out-of-range"),
    ],
)
def test_malformed_native_scalar_shapes_refuse_at_public_input(
    native_input: dict[str, object],
    section: str,
    field: str,
    bad: object,
) -> None:
    """Public JSON admission rejects coercions before any output can be created."""
    first = cast(list[dict[str, object]], native_input["runs"])[0]
    target = first if section == "run" else cast(dict[str, object], first[section])
    target[field] = bad
    with pytest.raises(ValueError):
        read_method_comparison_population(write_json(native_input))


@pytest.mark.parametrize("runs", [None, [], [None]])
def test_missing_or_non_record_population_refuses(
    native_input: dict[str, object],
    runs: object,
) -> None:
    """An empty or malformed population is never repaired with an invented row."""
    native_input["runs"] = runs
    with pytest.raises(ValueError):
        read_method_comparison_population(write_json(native_input))


def test_bounded_native_input_and_export_do_not_truncate_retained_rows(
    native_input: dict[str, object],
) -> None:
    """Over-budget input/export refuses; complete population is never shortened."""
    from dataclasses import replace

    from scpn_quantum_control.studio.method_comparison_artifact import MAX_COMPARISON_BYTES

    with pytest.raises(ValueError, match="four mebibytes"):
        read_method_comparison_population(" " * (MAX_COMPARISON_BYTES + 1))
    original, _, _, _ = read_method_comparison_population(write_json(native_input))
    row = replace(
        original[0], meaning=replace(original[0].meaning, problem="p" * 2000, model="m" * 2000)
    )
    population = tuple(replace(row, run_id=f"retained-{seed}", seed=seed) for seed in range(1000))
    with pytest.raises(ValueError, match="export exceeds"):
        export_method_comparison(population, baseline_method="reference")
    assert len(population) == 1000
    assert population[-1].run_id == "retained-999"


@pytest.mark.parametrize(
    "case",
    [
        "population",
        "model-mismatch",
        "precision-mismatch",
        "missing-population",
        "unknown-cost",
        "budget-exceeded",
        "intervals",
        "original-solver",
        "numeric-boundary",
        "original-ode-only",
        "integer-scalar",
    ],
)
def test_captured_native_inputs_replay_every_original_row_exactly(case: str) -> None:
    """The actual public native producer reproduces retained export bytes, not a best seed."""
    import json
    from pathlib import Path

    pack = json.loads(
        (
            Path(__file__).resolve().parents[1] / "data/studio/method_comparison_cases.json"
        ).read_text()
    )
    original = pack["cases"][case]
    unit = cast(dict[str, object], read_json(original))
    body = cast(dict[str, object], unit["body"])
    comparison = cast(dict[str, object], body["comparison"])
    rows = cast(list[dict[str, object]], comparison["runs"])
    for row in rows:
        budget = cast(dict[str, object], row["budget"])
        del budget["total_cost_ms"], budget["within_budget"]
    source = {
        "schema": "studio.method_comparison_input.v1",
        "runs": rows,
        "baseline_method": comparison["baseline_method"],
        "missing_data_policy": comparison["missing_data_policy"],
        "selected_run_id": comparison["selected_run_id"],
    }
    population, baseline, policy, selected = read_method_comparison_population(write_json(source))
    assert (
        export_method_comparison(
            population,
            baseline_method=baseline,
            missing_data_policy=policy,
            selected_run_id=selected,
        )
        == original
    )


def test_fixed_interval_cohort_retains_negative_empirical_coverage() -> None:
    """Every predetermined seed remains visible when nominal coverage is unsupported."""
    import json
    from pathlib import Path

    pack = json.loads(
        (
            Path(__file__).resolve().parents[1] / "data/studio/method_comparison_cases.json"
        ).read_text()
    )
    protocol = pack["interval_protocol"]
    assert [row["seed"] for row in protocol["samples"]] == list(range(200))
    assert all(sum(row["counts"].values()) == 1000 for row in protocol["samples"])
    original = cast(dict[str, object], read_json(pack["cases"]["intervals"]))
    assert original["sha256"] == protocol["artifact_sha256"]
    from statistics import NormalDist

    from scpn_quantum_control.analysis.adaptive_fim_feedback import wilson_score_interval
    from scpn_quantum_control.analysis.sync_uncertainty import (
        order_parameter_bootstrap,
        order_parameter_shot_noise,
    )

    body = cast(dict[str, object], original["body"])
    comparison = cast(dict[str, object], body["comparison"])
    retained = cast(list[dict[str, object]], comparison["runs"])
    assert len(retained) == 400
    for method, summary in protocol["methods"].items():
        actual = []
        for sample in protocol["samples"]:
            if method == "shot-noise-delta":
                interval = order_parameter_shot_noise(sample["counts"], coverage=0.95)
            else:
                interval = order_parameter_bootstrap(
                    sample["counts"], coverage=0.95, seed=sample["seed"]
                )
            actual.append(interval)
            row = next(
                row
                for row in retained
                if row["method"] == method and row["seed"] == sample["seed"]
            )
            assert row["value"] == interval.point
            saved = cast(dict[str, object], row["interval"])
            assert saved["low"] == pytest.approx(interval.low, abs=1e-12)
            assert saved["high"] == pytest.approx(interval.high, abs=1e-12)
        covered = sum(interval.low <= protocol["truth"] <= interval.high for interval in actual)
        assert covered == summary["covered"]
        observed = wilson_score_interval(covered, 200, z=NormalDist().inv_cdf(0.975))
        assert summary["wilson_interval"]["lower"] == pytest.approx(observed.lower, abs=1e-12)
        assert summary["wilson_interval"]["upper"] == pytest.approx(observed.upper, abs=1e-12)
        assert summary["nominal_coverage"] == 0.95
        assert summary["observed_coverage"] == 0.91
        assert summary["covered"] == 182
        assert summary["replicates"] == 200
        assert summary["nominal_within_observed_binomial_interval"] is False
        assert summary["wilson_interval"]["upper"] < 0.95
