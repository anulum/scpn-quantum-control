# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable isolated benchmark plan.
"""Isolated benchmark batch plan for differentiable promotion evidence."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from .isolated_host_readiness import HostReadiness, capture_host_readiness

REPO_ROOT = Path(__file__).resolve().parents[3]
DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_SCHEMA = "scpn_qc_differentiable_isolated_benchmark_plan_v1"
DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_ARTIFACT_ID = "diff-isolated-benchmark-plan-20260627"
DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_CLAIM_BOUNDARY = (
    "Differentiable isolated benchmark batch plan only; no isolated_affinity "
    "benchmark evidence, production-performance claim, provider execution, QPU "
    "execution, GPU execution, Enzyme promotion, or claim-ledger promotion is implied."
)
PlannedBenchmarkClassification = Literal[
    "functional_non_isolated",
    "hard_gap",
    "isolated_affinity",
]


@dataclass(frozen=True)
class DifferentiableIsolatedBenchmarkPlanRow:
    """One current evidence artifact and its isolated rerun requirements."""

    row_id: str
    title: str
    benchmark_family: str
    source_artifact_paths: tuple[str, ...]
    source_artifact_ids: tuple[str, ...]
    source_classifications: tuple[PlannedBenchmarkClassification, ...]
    rerun_command: tuple[str, ...]
    required_runner_labels: tuple[str, ...]
    required_host_context: tuple[str, ...]
    expected_output_paths: tuple[str, ...]
    blockers: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        """Validate row fields before emitting benchmark planning evidence."""
        for field_name in ("row_id", "title", "benchmark_family", "claim_boundary"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in (
            "source_artifact_paths",
            "source_artifact_ids",
            "source_classifications",
            "rerun_command",
            "required_runner_labels",
            "required_host_context",
            "expected_output_paths",
        ):
            value = getattr(self, field_name)
            if not value or any(not str(item).strip() for item in value):
                raise ValueError(f"{field_name} must contain non-empty entries")

    @property
    def promotion_ready(self) -> bool:
        """Return whether this row already has isolated evidence and no blockers."""
        return not self.blockers and all(
            classification == "isolated_affinity" for classification in self.source_classifications
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready benchmark plan row."""
        return {
            "row_id": self.row_id,
            "title": self.title,
            "benchmark_family": self.benchmark_family,
            "source_artifact_paths": list(self.source_artifact_paths),
            "source_artifact_ids": list(self.source_artifact_ids),
            "source_classifications": list(self.source_classifications),
            "rerun_command": list(self.rerun_command),
            "required_runner_labels": list(self.required_runner_labels),
            "required_host_context": list(self.required_host_context),
            "expected_output_paths": list(self.expected_output_paths),
            "blockers": list(self.blockers),
            "promotion_ready": self.promotion_ready,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableIsolatedBenchmarkPlan:
    """Validated batch plan for reproducing differentiable benchmarks in isolation."""

    schema: str
    artifact_id: str
    rows: tuple[DifferentiableIsolatedBenchmarkPlanRow, ...]
    promotion_ready: bool
    ready_row_count: int
    total_row_count: int
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready isolated benchmark plan."""
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "promotion_ready": self.promotion_ready,
            "ready_row_count": self.ready_row_count,
            "total_row_count": self.total_row_count,
            "claim_boundary": self.claim_boundary,
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class DifferentiableIsolatedBenchmarkPlanValidation:
    """Validation result for an isolated benchmark batch plan."""

    passed: bool
    errors: tuple[str, ...]
    checked_row_ids: tuple[str, ...]
    checked_paths: tuple[str, ...]
    checked_source_classifications: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready plan validation evidence."""
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "checked_row_ids": list(self.checked_row_ids),
            "checked_paths": list(self.checked_paths),
            "checked_source_classifications": list(self.checked_source_classifications),
            "claim_boundary": self.claim_boundary,
        }


def run_differentiable_isolated_benchmark_plan(
    *,
    repo_root: Path = REPO_ROOT,
    host_readiness: HostReadiness | None = None,
) -> DifferentiableIsolatedBenchmarkPlan:
    """Build the isolated benchmark batch plan from committed evidence artifacts."""
    readiness = capture_host_readiness(0) if host_readiness is None else host_readiness
    rows = _default_plan_rows(repo_root=repo_root, host_readiness=readiness)
    ready_count = sum(1 for row in rows if row.promotion_ready)
    return DifferentiableIsolatedBenchmarkPlan(
        schema=DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_SCHEMA,
        artifact_id=DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_ARTIFACT_ID,
        rows=rows,
        promotion_ready=ready_count == len(rows),
        ready_row_count=ready_count,
        total_row_count=len(rows),
        claim_boundary=DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_CLAIM_BOUNDARY,
    )


def validate_differentiable_isolated_benchmark_plan(
    plan: DifferentiableIsolatedBenchmarkPlan,
    *,
    repo_root: Path = REPO_ROOT,
) -> DifferentiableIsolatedBenchmarkPlanValidation:
    """Validate plan rows, source artifacts, commands, and promotion boundaries."""
    errors: list[str] = []
    checked_paths: set[str] = {
        "data/differentiable_phase_qnode/differentiable_isolated_benchmark_plan_20260627.md"
    }
    checked_classifications: list[str] = []
    row_ids = tuple(row.row_id for row in plan.rows)

    if plan.schema != DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_SCHEMA:
        errors.append(f"unexpected isolated-benchmark-plan schema: {plan.schema}")
    if plan.total_row_count != len(plan.rows):
        errors.append("total_row_count does not match row count")
    ready_count = sum(1 for row in plan.rows if row.promotion_ready)
    if plan.ready_row_count != ready_count:
        errors.append("ready_row_count does not match ready rows")
    if plan.promotion_ready != (ready_count == len(plan.rows)):
        errors.append("promotion_ready does not match row readiness")
    for row_id in _duplicates(row_ids):
        errors.append(f"duplicate isolated benchmark plan row_id: {row_id}")

    for row in plan.rows:
        if len(row.source_artifact_paths) != len(row.source_classifications):
            errors.append(f"{row.row_id}: source paths and classifications must align")
        if plan.promotion_ready and row.blockers:
            errors.append(f"{row.row_id}: ready plan rows must not carry blockers")
        if "taskset" not in row.rerun_command and "chrt" not in row.rerun_command:
            errors.append(f"{row.row_id}: rerun command must include taskset or chrt")
        if not {"self-hosted", "linux", "isolated-benchmark"} <= set(row.required_runner_labels):
            errors.append(f"{row.row_id}: required runner labels are incomplete")
        for path, expected_classification in zip(
            row.source_artifact_paths,
            row.source_classifications,
            strict=True,
        ):
            checked_paths.add(path)
            artifact_path = repo_root / path
            if not artifact_path.exists():
                errors.append(f"{row.row_id}: source artifact path does not exist: {path}")
                continue
            actual_classification = _artifact_classification(artifact_path)
            checked_classifications.append(actual_classification)
            if actual_classification != expected_classification:
                errors.append(
                    f"{row.row_id}: expected {expected_classification} for {path}, "
                    f"found {actual_classification}"
                )
        for output_path in row.expected_output_paths:
            checked_paths.add(output_path)
            if not output_path.startswith("data/differentiable_phase_qnode/"):
                errors.append(f"{row.row_id}: expected output must stay under evidence data path")
    for path in tuple(checked_paths):
        if (
            path.endswith(".md")
            and "differentiable_isolated_benchmark_plan" in path
            and not (repo_root / path).exists()
        ):
            errors.append(f"isolated benchmark plan evidence path does not exist: {path}")

    return DifferentiableIsolatedBenchmarkPlanValidation(
        passed=not errors,
        errors=tuple(errors),
        checked_row_ids=row_ids,
        checked_paths=tuple(sorted(checked_paths)),
        checked_source_classifications=tuple(checked_classifications),
        claim_boundary=(
            "Isolated benchmark plan validation only; checks source artifacts, "
            "rerun commands, runner labels, and output paths without promoting "
            "functional_non_isolated or hard_gap evidence."
        ),
    )


def render_differentiable_isolated_benchmark_plan_markdown(
    plan: DifferentiableIsolatedBenchmarkPlan,
) -> str:
    """Render a reviewer-facing Markdown summary of the isolated benchmark plan."""
    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "SCPN Quantum Control — Differentiable Isolated Benchmark Batch Plan",
        "-->",
        "",
        "# Differentiable Isolated Benchmark Batch Plan",
        "",
        f"- Schema: `{plan.schema}`",
        f"- Artifact ID: `{plan.artifact_id}`",
        f"- Promotion ready: `{plan.promotion_ready}`",
        f"- Ready rows: `{plan.ready_row_count}/{plan.total_row_count}`",
        f"- Claim boundary: {plan.claim_boundary}",
        "",
        "| Row | Source classifications | Command | Blockers |",
        "|---|---|---|---|",
    ]
    for row in plan.rows:
        lines.append(
            "| `{row}` | {classifications} | `{command}` | {blockers} |".format(
                row=row.row_id,
                classifications=_markdown_cell("<br>".join(row.source_classifications)),
                command=_markdown_cell(" ".join(row.rerun_command)),
                blockers=_markdown_cell("<br>".join(row.blockers) or "none"),
            )
        )
    lines.append("")
    lines.append(
        "This plan is a reserved-host execution queue. It does not promote any "
        "performance, provider, QPU, GPU, Enzyme, or claim-ledger row until the "
        "listed commands produce validated isolated_affinity artifacts."
    )
    lines.append("")
    return "\n".join(lines)


def _default_plan_rows(
    *,
    repo_root: Path,
    host_readiness: HostReadiness,
) -> tuple[DifferentiableIsolatedBenchmarkPlanRow, ...]:
    host_blockers = _host_blockers(host_readiness)
    return (
        _row(
            row_id="ci_external_comparison_bundle",
            title="CI external comparison bundle",
            benchmark_family="external_comparison",
            source_paths=(
                "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
                "diff-qnode-ci-evidence-schema-v1.json",
                "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
                "diff-qnode-external-comparison.json",
            ),
            source_artifact_ids=(
                "diff-qnode-ci-evidence-schema-v1",
                "diff-qnode-external-comparison-local",
            ),
            rerun_command=(
                "taskset",
                "-c",
                "2",
                "chrt",
                "-f",
                "1",
                ".venv/bin/python",
                "scripts/run_differentiable_benchmark_evidence.py",
                "--output-dir",
                "data/differentiable_phase_qnode/isolated_benchmark_batch_20260627",
                "--cpu-affinity",
                "2",
                "--isolation-method",
                "taskset+chrt",
            ),
            expected_outputs=(
                "data/differentiable_phase_qnode/isolated_benchmark_batch_20260627/"
                "diff-qnode-ci-evidence-schema-v1.json",
                "data/differentiable_phase_qnode/isolated_benchmark_batch_20260627/"
                "diff-qnode-ci-evidence-schema-v1.csv",
                "data/differentiable_phase_qnode/isolated_benchmark_batch_20260627/"
                "diff-qnode-external-comparison.json",
            ),
            repo_root=repo_root,
            blockers=host_blockers,
        ),
        _row(
            row_id="phase_qnode_affinity",
            title="Phase-QNode affinity timing",
            benchmark_family="phase_qnode_affinity",
            source_paths=(
                "data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/"
                "phase_qnode_affinity.json",
            ),
            source_artifact_ids=("phase-qnode-affinity-local-functional",),
            rerun_command=(
                "taskset",
                "-c",
                "2",
                "chrt",
                "-f",
                "1",
                ".venv/bin/python",
                "tools/run_phase_qnode_affinity_benchmark.py",
                "--reserved-cpus",
                "2",
                "--output",
                "data/differentiable_phase_qnode/isolated_benchmark_batch_20260627/"
                "phase_qnode_affinity.json",
            ),
            expected_outputs=(
                "data/differentiable_phase_qnode/isolated_benchmark_batch_20260627/"
                "phase_qnode_affinity.json",
            ),
            repo_root=repo_root,
            blockers=host_blockers,
        ),
        _row(
            row_id="identical_circuit_gradient_comparison",
            title="Identical-circuit gradient comparison",
            benchmark_family="gradient_comparison",
            source_paths=(
                "data/differentiable_phase_qnode/identical_circuit_gradient_comparison_20260616.json",
            ),
            source_artifact_ids=("identical-circuit-gradient-comparison-20260616",),
            rerun_command=(
                "taskset",
                "-c",
                "2",
                "chrt",
                "-f",
                "1",
                ".venv/bin/python",
                "scripts/run_differentiable_benchmark_evidence.py",
                "--output-dir",
                "data/differentiable_phase_qnode/isolated_benchmark_batch_20260627",
            ),
            expected_outputs=(
                "data/differentiable_phase_qnode/isolated_benchmark_batch_20260627/"
                "identical_circuit_gradient_comparison_20260616.json",
            ),
            repo_root=repo_root,
            blockers=host_blockers,
        ),
        _row(
            row_id="domain_benchmark_dataset_closure",
            title="Domain benchmark dataset closure",
            benchmark_family="domain_dataset",
            source_paths=(
                "data/differentiable_phase_qnode/domain_benchmark_dataset_closure_20260616.json",
            ),
            source_artifact_ids=("differentiable-domain-benchmark-closure-20260616",),
            rerun_command=(
                "taskset",
                "-c",
                "2",
                "chrt",
                "-f",
                "1",
                ".venv/bin/python",
                "scripts/run_differentiable_benchmark_evidence.py",
                "--output-dir",
                "data/differentiable_phase_qnode/isolated_benchmark_batch_20260627",
            ),
            expected_outputs=(
                "data/differentiable_phase_qnode/isolated_benchmark_batch_20260627/"
                "domain_benchmark_dataset_closure_20260616.json",
            ),
            repo_root=repo_root,
            blockers=host_blockers,
        ),
        _row(
            row_id="torch_maturity_audit",
            title="PyTorch maturity audit",
            benchmark_family="torch_maturity",
            source_paths=("data/differentiable_phase_qnode/torch_maturity_audit_20260616.json",),
            source_artifact_ids=("torch-maturity-audit-20260616",),
            rerun_command=(
                "taskset",
                "-c",
                "2",
                "chrt",
                "-f",
                "1",
                ".venv/bin/python",
                "scripts/run_differentiable_benchmark_evidence.py",
                "--output-dir",
                "data/differentiable_phase_qnode/isolated_benchmark_batch_20260627",
            ),
            expected_outputs=(
                "data/differentiable_phase_qnode/isolated_benchmark_batch_20260627/"
                "torch_maturity_audit_20260616.json",
            ),
            repo_root=repo_root,
            blockers=host_blockers,
        ),
        _row(
            row_id="enzyme_mlir_maturity_audit",
            title="Enzyme/MLIR maturity audit",
            benchmark_family="enzyme_mlir",
            source_paths=(
                "data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.json",
            ),
            source_artifact_ids=("enzyme-mlir-maturity-audit-20260616",),
            rerun_command=(
                "taskset",
                "-c",
                "2",
                "chrt",
                "-f",
                "1",
                ".venv/bin/python",
                "scripts/run_differentiable_benchmark_evidence.py",
                "--output-dir",
                "data/differentiable_phase_qnode/isolated_benchmark_batch_20260627",
            ),
            expected_outputs=(
                "data/differentiable_phase_qnode/isolated_benchmark_batch_20260627/"
                "enzyme_mlir_maturity_audit_20260616.json",
            ),
            repo_root=repo_root,
            blockers=(
                *host_blockers,
                "Enzyme/MLIR compiler-native breadth remains hard_gap until raw 11-case "
                "breadth, native execution, and isolated benchmark attachments pass.",
            ),
        ),
    )


def _row(
    *,
    row_id: str,
    title: str,
    benchmark_family: str,
    source_paths: tuple[str, ...],
    source_artifact_ids: tuple[str, ...],
    rerun_command: tuple[str, ...],
    expected_outputs: tuple[str, ...],
    repo_root: Path,
    blockers: tuple[str, ...],
) -> DifferentiableIsolatedBenchmarkPlanRow:
    classifications = tuple(
        _classification_for_path(repo_root / source_path) for source_path in source_paths
    )
    row_blockers = tuple(
        dict.fromkeys(
            (
                *(blockers or ()),
                *(
                    f"{source_path} is {classification}, not isolated_affinity"
                    for source_path, classification in zip(
                        source_paths,
                        classifications,
                        strict=True,
                    )
                    if classification != "isolated_affinity"
                ),
            )
        )
    )
    return DifferentiableIsolatedBenchmarkPlanRow(
        row_id=row_id,
        title=title,
        benchmark_family=benchmark_family,
        source_artifact_paths=source_paths,
        source_artifact_ids=source_artifact_ids,
        source_classifications=classifications,
        rerun_command=rerun_command,
        required_runner_labels=("self-hosted", "linux", "isolated-benchmark"),
        required_host_context=(
            "reserved_cpu_affinity",
            "observed_process_affinity",
            "host_load_before_after",
            "performance_governor_or_frequency",
            "no_heavy_concurrent_jobs",
            "raw_timing_rows",
        ),
        expected_output_paths=expected_outputs,
        blockers=row_blockers,
        claim_boundary=DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_CLAIM_BOUNDARY,
    )


def _host_blockers(readiness: HostReadiness) -> tuple[str, ...]:
    if readiness.ready:
        return ()
    return tuple(f"reserved host readiness blocker: {blocker}" for blocker in readiness.blockers)


def _classification_for_path(path: Path) -> PlannedBenchmarkClassification:
    if not path.exists():
        return "hard_gap"
    return cast(PlannedBenchmarkClassification, _artifact_classification(path))


def _artifact_classification(path: Path) -> str:
    payload = _json_object(path)
    direct = payload.get("classification")
    if isinstance(direct, str) and direct:
        return direct
    evidence_label = payload.get("evidence_label")
    if isinstance(evidence_label, str) and evidence_label:
        return evidence_label
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        nested = metadata.get("classification")
        if isinstance(nested, str) and nested:
            return nested
    return "hard_gap"


def _json_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(Mapping[str, object], payload)


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _markdown_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|")


__all__ = [
    "DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_ARTIFACT_ID",
    "DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_CLAIM_BOUNDARY",
    "DIFFERENTIABLE_ISOLATED_BENCHMARK_PLAN_SCHEMA",
    "DifferentiableIsolatedBenchmarkPlan",
    "DifferentiableIsolatedBenchmarkPlanRow",
    "DifferentiableIsolatedBenchmarkPlanValidation",
    "render_differentiable_isolated_benchmark_plan_markdown",
    "run_differentiable_isolated_benchmark_plan",
    "validate_differentiable_isolated_benchmark_plan",
]
