# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable isolated benchmark plan tests
"""Tests for differentiable isolated benchmark batch planning."""

from __future__ import annotations

from pathlib import Path

import scpn_quantum_control as scpn
from scpn_quantum_control import (
    DifferentiableIsolatedBenchmarkPlanRow,
    differentiable_api,
    differentiable_module_hardening_registry,
    render_differentiable_isolated_benchmark_plan_markdown,
    run_differentiable_isolated_benchmark_plan,
    validate_differentiable_isolated_benchmark_plan,
)


def test_isolated_benchmark_plan_covers_current_non_isolated_artifacts() -> None:
    """The plan must enumerate current differentiable benchmark promotion blockers."""
    plan = run_differentiable_isolated_benchmark_plan()

    assert plan.schema == "scpn_qc_differentiable_isolated_benchmark_plan_v1"
    assert plan.promotion_ready is False
    assert plan.total_row_count == len(plan.rows)
    assert plan.ready_row_count == 0
    assert {row.row_id for row in plan.rows} == {
        "ci_external_comparison_bundle",
        "phase_qnode_affinity",
        "identical_circuit_gradient_comparison",
        "domain_benchmark_dataset_closure",
        "torch_maturity_audit",
        "enzyme_mlir_maturity_audit",
        "compiler_promotion_batch",
    }
    assert "no isolated_affinity benchmark evidence" in plan.claim_boundary


def test_isolated_benchmark_plan_rows_are_artifact_and_command_backed() -> None:
    """Each plan row must cite real source artifacts and rerun commands."""
    plan = run_differentiable_isolated_benchmark_plan()
    rows = {row.row_id: row for row in plan.rows}

    ci_row = rows["ci_external_comparison_bundle"]
    assert ci_row.source_artifact_paths == (
        "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
        "diff-qnode-ci-evidence-schema-v1.json",
        "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
        "diff-qnode-external-comparison.json",
    )
    assert ci_row.source_classifications == ("functional_non_isolated", "functional_non_isolated")
    assert "scripts/run_differentiable_benchmark_evidence.py" in " ".join(ci_row.rerun_command)
    assert "taskset" in ci_row.rerun_command
    assert "chrt" in ci_row.rerun_command

    enzyme_row = rows["enzyme_mlir_maturity_audit"]
    assert enzyme_row.source_classifications == ("hard_gap",)
    assert any("Enzyme/MLIR" in blocker for blocker in enzyme_row.blockers)

    compiler_row = rows["compiler_promotion_batch"]
    assert compiler_row.source_artifact_paths == (
        "data/differentiable_phase_qnode/compiler_promotion_batch_20260706.json",
        "data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.json",
    )
    assert compiler_row.source_artifact_ids == (
        "compiler-promotion-batch-20260706",
        "compiler-evidence-boundary-20260705",
    )
    assert compiler_row.source_classifications == (
        "functional_non_isolated",
        "functional_non_isolated",
    )
    compiler_command = " ".join(compiler_row.rerun_command)
    assert "scripts/run_native_whole_program_ad_execution_evidence.py" in compiler_command
    assert "taskset" in compiler_row.rerun_command
    assert "chrt" in compiler_row.rerun_command
    assert any(
        "isolated compiler benchmark artifact IDs" in blocker for blocker in compiler_row.blockers
    )

    validation = validate_differentiable_isolated_benchmark_plan(plan)
    assert validation.passed, validation.errors
    assert (
        "data/differentiable_phase_qnode/differentiable_isolated_benchmark_plan_20260627.md"
        in (validation.checked_paths)
    )
    assert "data/differentiable_phase_qnode/compiler_promotion_batch_20260706.json" in (
        validation.checked_paths
    )


def test_isolated_benchmark_plan_validation_rejects_stale_paths_and_promotion(
    tmp_path: Path,
) -> None:
    """Validation must fail closed on stale plan paths and over-promoted rows."""
    invalid_row = DifferentiableIsolatedBenchmarkPlanRow(
        row_id="invalid_row",
        title="Invalid row",
        benchmark_family="invalid",
        source_artifact_paths=("missing.json",),
        source_artifact_ids=("missing",),
        source_classifications=("isolated_affinity",),
        rerun_command=("python", "missing.py"),
        required_runner_labels=("self-hosted", "linux", "isolated-benchmark"),
        required_host_context=("cpu_affinity", "host_load", "governor"),
        expected_output_paths=("missing-output.json",),
        blockers=("unexpected blocker",),
        claim_boundary="test-only invalid row",
    )
    plan = type(run_differentiable_isolated_benchmark_plan())(
        schema="scpn_qc_differentiable_isolated_benchmark_plan_v1",
        artifact_id="test-isolated-benchmark-plan",
        rows=(invalid_row,),
        promotion_ready=True,
        ready_row_count=1,
        total_row_count=1,
        claim_boundary="test-only invalid plan",
    )

    validation = validate_differentiable_isolated_benchmark_plan(plan, repo_root=tmp_path)

    assert not validation.passed
    assert any("source artifact path does not exist" in error for error in validation.errors)
    assert any("ready plan rows must not carry blockers" in error for error in validation.errors)
    assert any(
        "rerun command must include taskset or chrt" in error for error in validation.errors
    )


def test_isolated_benchmark_plan_markdown_unified_api_and_exports() -> None:
    """The plan must render, dispatch, and export through public package surfaces."""
    plan = run_differentiable_isolated_benchmark_plan()
    markdown = render_differentiable_isolated_benchmark_plan_markdown(plan)
    result = differentiable_api("isolated_benchmark_plan")

    assert "# Differentiable Isolated Benchmark Batch Plan" in markdown
    assert "phase_qnode_affinity" in markdown
    assert result.operation == "isolated_benchmark_plan"
    assert result.supported is False
    assert result.payload["total_row_count"] == plan.total_row_count
    assert "run_differentiable_isolated_benchmark_plan" in scpn.__all__
    assert (
        scpn.run_differentiable_isolated_benchmark_plan
        is run_differentiable_isolated_benchmark_plan
    )
    registry_paths = {record.module_path for record in differentiable_module_hardening_registry()}
    assert "src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py" in (
        registry_paths
    )
