# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable isolated benchmark plan edge tests
"""Edge tests for differentiable isolated benchmark batch planning."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import scpn_quantum_control.benchmarks.differentiable_isolated_benchmark_plan as plan_module
from scpn_quantum_control.benchmarks.differentiable_isolated_benchmark_plan import (
    DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_CLAIM_BOUNDARY,
    DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_SCHEMA,
    DifferentiableIsolatedBenchmarkPlan,
    DifferentiableIsolatedBenchmarkPlanRow,
    validate_differentiable_isolated_benchmark_plan,
)
from scpn_quantum_control.benchmarks.isolated_host_readiness import HostReadiness


def _ready_host() -> HostReadiness:
    """Return a host-readiness object with no isolation blockers."""
    return HostReadiness(
        ready=True,
        reserved_core=2,
        governor="performance",
        governor_is_stable=True,
        frequency_mhz=3200.0,
        load_average=(0.1, 0.2, 0.3),
        load_is_low=True,
        blockers=(),
    )


def _plan_row(
    *,
    row_id: str = "row",
    source_path: str = "data/differentiable_phase_qnode/source.json",
    source_classification: plan_module.PlannedBenchmarkClassification = "functional_non_isolated",
    labels: tuple[str, ...] = ("self-hosted", "linux", "isolated-benchmark"),
    command: tuple[str, ...] = ("taskset", "-c", "2", ".venv/bin/python", "bench.py"),
    blockers: tuple[str, ...] = (),
) -> DifferentiableIsolatedBenchmarkPlanRow:
    """Return a valid configurable benchmark-plan row."""
    return DifferentiableIsolatedBenchmarkPlanRow(
        row_id=row_id,
        title="Synthetic row",
        benchmark_family="synthetic",
        source_artifact_paths=(source_path,),
        source_artifact_ids=(f"{row_id}-artifact",),
        source_classifications=(source_classification,),
        rerun_command=command,
        required_runner_labels=labels,
        required_host_context=("reserved_cpu_affinity",),
        expected_output_paths=("data/differentiable_phase_qnode/output.json",),
        blockers=blockers,
        claim_boundary=DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_CLAIM_BOUNDARY,
    )


def _plan(
    rows: tuple[DifferentiableIsolatedBenchmarkPlanRow, ...],
    *,
    schema: str = DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_SCHEMA,
    promotion_ready: bool = False,
    ready_row_count: int = 0,
    total_row_count: int | None = None,
) -> DifferentiableIsolatedBenchmarkPlan:
    """Return a configurable isolated benchmark plan."""
    return DifferentiableIsolatedBenchmarkPlan(
        schema=schema,
        artifact_id="synthetic-plan",
        rows=rows,
        promotion_ready=promotion_ready,
        ready_row_count=ready_row_count,
        total_row_count=len(rows) if total_row_count is None else total_row_count,
        claim_boundary=DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_CLAIM_BOUNDARY,
    )


def _write_plan_evidence_root(tmp_path: Path) -> None:
    """Create the markdown evidence path required by validation."""
    evidence_path = (
        tmp_path
        / "data/differentiable_phase_qnode/differentiable_isolated_benchmark_plan_20260627.md"
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text("# plan\n", encoding="utf-8")


def test_plan_row_constructor_rejects_empty_identity_and_collections() -> None:
    """Plan rows should reject empty required strings and collection members."""
    row = _plan_row()

    with pytest.raises(ValueError, match="row_id"):
        replace(row, row_id="")
    with pytest.raises(ValueError, match="source_artifact_paths"):
        replace(row, source_artifact_paths=())
    with pytest.raises(ValueError, match="expected_output_paths"):
        replace(row, expected_output_paths=("",))


def test_validation_result_to_dict_serialises_failure_fields(tmp_path: Path) -> None:
    """Validation failure evidence should remain JSON-ready."""
    validation = validate_differentiable_isolated_benchmark_plan(
        _plan((_plan_row(command=("python", "bench.py")),)),
        repo_root=tmp_path,
    )
    payload = validation.to_dict()

    assert not validation.passed
    assert payload["passed"] is False
    assert payload["errors"] == list(validation.errors)
    assert payload["checked_row_ids"] == ["row"]
    assert "promoting" in cast(str, payload["claim_boundary"])


def test_validation_reports_schema_counts_duplicates_and_label_errors(tmp_path: Path) -> None:
    """Plan validation should detect stale schema, counts, duplicate IDs, and labels."""
    _write_plan_evidence_root(tmp_path)
    artifact = tmp_path / "data/differentiable_phase_qnode/source.json"
    artifact.write_text(
        json.dumps({"classification": "functional_non_isolated"}), encoding="utf-8"
    )
    row = _plan_row(labels=("self-hosted", "linux"))
    plan = _plan(
        (row, row),
        schema="wrong-schema",
        promotion_ready=True,
        ready_row_count=2,
        total_row_count=9,
    )

    validation = validate_differentiable_isolated_benchmark_plan(plan, repo_root=tmp_path)

    assert not validation.passed
    assert any("unexpected isolated-benchmark-plan schema" in error for error in validation.errors)
    assert any("total_row_count does not match row count" in error for error in validation.errors)
    assert any("ready_row_count does not match ready rows" in error for error in validation.errors)
    assert any(
        "promotion_ready does not match row readiness" in error for error in validation.errors
    )
    assert any("duplicate isolated benchmark plan row_id" in error for error in validation.errors)
    assert any("required runner labels are incomplete" in error for error in validation.errors)


def test_validation_reports_classification_and_output_path_mismatches(tmp_path: Path) -> None:
    """Validation should reject stale source classifications and output locations."""
    _write_plan_evidence_root(tmp_path)
    artifact = tmp_path / "data/differentiable_phase_qnode/source.json"
    artifact.write_text(json.dumps({"classification": "hard_gap"}), encoding="utf-8")
    row = replace(
        _plan_row(source_classification="functional_non_isolated"),
        expected_output_paths=("tmp/output.json",),
    )

    validation = validate_differentiable_isolated_benchmark_plan(_plan((row,)), repo_root=tmp_path)

    assert not validation.passed
    assert any("expected functional_non_isolated" in error for error in validation.errors)
    assert any(
        "expected output must stay under evidence data path" in error
        for error in validation.errors
    )


def test_validation_raises_on_source_path_classification_misalignment(tmp_path: Path) -> None:
    """Validation should not silently zip uneven source path and classification lists."""
    _write_plan_evidence_root(tmp_path)
    row = replace(
        _plan_row(),
        source_artifact_paths=(
            "data/differentiable_phase_qnode/source.json",
            "data/differentiable_phase_qnode/source-two.json",
        ),
    )
    for path in row.source_artifact_paths:
        artifact = tmp_path / path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps({"classification": "functional_non_isolated"}),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="zip"):
        validate_differentiable_isolated_benchmark_plan(_plan((row,)), repo_root=tmp_path)


def test_ready_host_plan_omits_host_blockers() -> None:
    """Ready hosts should not add reserved-host blockers to plan rows."""
    plan = plan_module.run_differentiable_isolated_benchmark_plan(host_readiness=_ready_host())

    assert all(
        not blocker.startswith("reserved host readiness blocker:")
        for row in plan.rows
        for blocker in row.blockers
    )


def test_artifact_classification_fallbacks_and_json_validation(tmp_path: Path) -> None:
    """Artifact classification should expose metadata fallback, hard-gap default, and JSON guards."""
    missing_path = tmp_path / "missing.json"
    fallback_path = tmp_path / "fallback.json"
    default_path = tmp_path / "default.json"
    scalar_metadata_path = tmp_path / "scalar-metadata.json"
    bad_path = tmp_path / "bad.json"
    fallback_path.write_text(
        json.dumps({"metadata": {"classification": "isolated_affinity"}}),
        encoding="utf-8",
    )
    default_path.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
    scalar_metadata_path.write_text(json.dumps({"metadata": "classification"}), encoding="utf-8")
    bad_path.write_text(json.dumps(["not", "object"]), encoding="utf-8")

    assert plan_module._classification_for_path(missing_path) == "hard_gap"
    assert plan_module._artifact_classification(fallback_path) == "isolated_affinity"
    assert plan_module._artifact_classification(default_path) == "hard_gap"
    assert plan_module._artifact_classification(scalar_metadata_path) == "hard_gap"
    with pytest.raises(ValueError, match="JSON object"):
        plan_module._artifact_classification(bad_path)


def test_duplicate_helper_reports_sorted_duplicates() -> None:
    """Duplicate helper output should be stable for validation evidence."""
    assert plan_module._duplicates(("b", "a", "b", "c", "a")) == ("a", "b")
