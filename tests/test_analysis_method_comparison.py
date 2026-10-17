# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Analysis comparison public compatibility
"""Exercise shared Analysis ownership without changing the v1 scientific producer."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pytest

from scpn_quantum_control.benchmarks.method_comparison_contracts import (
    ComparisonBudget,
    ComparisonMeaning,
    MethodRun,
)
from scpn_quantum_control.studio.method_comparison_artifact import export_method_comparison
from tools.audit_split_ownership import DEFAULT_MAP, load_domain_map, tracked_files

ROOT = Path(__file__).resolve().parents[1]


def population() -> tuple[MethodRun, ...]:
    """Declare signed zero, a negative result and both retained failed repetitions."""
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
    return (
        MethodRun("r0", "reference", "native", 0, meaning, budget, "completed", -0.0),
        MethodRun(
            "r1", "reference", "native", 1, meaning, budget, "failed", None, "retained failure"
        ),
        MethodRun("c0", "candidate", "native", 0, meaning, budget, "completed", -1.0),
        MethodRun(
            "c1", "candidate", "native", 1, meaning, budget, "failed", None, "retained failure"
        ),
    )


def test_analysis_public_entry_retains_the_complete_original_population() -> None:
    """Compute through the Analysis public API with the original costs and failures."""
    from scpn_quantum_control.analysis.method_comparison import summarise_method_comparison

    original = population()
    result = summarise_method_comparison(
        original,
        baseline_method="reference",
        missing_data_policy="available-case",
        selected_run_id="c0",
    )
    assert result["ranking_allowed"] is True
    assert result["population_run_ids"] == ["r0", "r1", "c0", "c1"]
    assert [row["reason"] for row in result["runs"]] == [
        None,
        "retained failure",
        None,
        "retained failure",
    ]
    assert [row["total_cost_ms"] for row in result["methods"]] == [20.0, 20.0]
    assert [row["mean"] for row in result["methods"]] == [0.0, -1.0]
    assert result["methods"][1]["difference_vs_reference"] == -1.0
    with pytest.raises(ValueError, match="unique"):
        summarise_method_comparison(original * 2, baseline_method="reference")
    assert original == population()


@pytest.mark.parametrize(
    "name",
    [
        "METHOD_COMPARISON_SCHEMA",
        "COMPARISON_CLAIM_BOUNDARY",
        "summarise_method_comparison",
    ],
)
def test_analysis_summary_names_are_the_original_objects(name: str) -> None:
    """Expose one original algorithm, schema and claim boundary through both APIs.

    Parameters
    ----------
    name
        Original public summary binding whose identity must remain stable.

    """
    from scpn_quantum_control.analysis import method_comparison as analysis
    from scpn_quantum_control.benchmarks import method_comparison as original

    assert getattr(analysis, name) is getattr(original, name)


@pytest.mark.parametrize(
    "name",
    [
        "MAX_COMPARISON_RUNS",
        "ComparisonMeaning",
        "ComparisonBudget",
        "MethodRun",
        "BudgetRecord",
        "IntervalRecord",
        "RunRecord",
        "SummaryRecord",
        "MethodComparisonRecord",
    ],
)
def test_analysis_contract_names_preserve_original_type_identity(name: str) -> None:
    """Keep native types at their defining v1 path, including trusted pickle lookup.

    Parameters
    ----------
    name
        Original public immutable or wire-record binding.

    """
    from scpn_quantum_control.analysis import method_comparison_contracts as analysis
    from scpn_quantum_control.benchmarks import method_comparison_contracts as original

    current = getattr(analysis, name)
    assert current is getattr(original, name)
    if isinstance(current, type):
        assert current.__module__ == "scpn_quantum_control.benchmarks.method_comparison_contracts"
        assert pickle.loads(pickle.dumps(current)) is current
    restored = pickle.loads(pickle.dumps(population()))
    assert restored == population()
    assert all(type(run) is MethodRun for run in restored)


def test_studio_export_retains_frozen_original_v1_bytes() -> None:
    """The ownership correction preserves producer identity and every canonical byte."""
    text = export_method_comparison(
        population(),
        baseline_method="reference",
        missing_data_policy="available-case",
        selected_run_id="c0",
    )
    assert (
        hashlib.sha256(text.encode()).hexdigest()
        == "51d9b25d7071454f8871b0208d52cb483f1568b544c85395043ee60b1159b65a"
    )
    assert "scpn_quantum_control.benchmarks.method_comparison.summarise_method_comparison" in text
    assert "9007199254740993" in text and '"value":-0.0' in text


def test_comparison_ownership_is_complete_and_keeps_other_bench_files() -> None:
    """Declare the real two implementation owners and preserve every other Bench file."""
    mapping = load_domain_map(ROOT / DEFAULT_MAP)
    prefix = "src/scpn_quantum_control/"
    tracked = {
        path.removeprefix(prefix)
        for path in tracked_files(ROOT)
        if path.startswith(prefix + "benchmarks/")
    }
    declared = {path for path in mapping.split_files if path.startswith("benchmarks/")}
    assert declared == tracked
    expected_analysis = {
        "benchmarks/method_comparison.py",
        "benchmarks/method_comparison_contracts.py",
    }
    assert {
        path for path in declared if mapping.split_files[path] == "analysis"
    } == expected_analysis
    assert all(mapping.split_files[path] == "benchmarks" for path in declared - expected_analysis)
