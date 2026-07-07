# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- compiler isolated benchmark evidence
"""Attachment-grade isolated benchmark evidence for compiler promotion gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from scpn_quantum_control.compiler import NativeWholeProgramADExecutionEvidence

from .differentiable_evidence import BenchmarkIsolationMetadata

CompilerBenchmarkClassification = Literal[
    "functional_non_isolated",
    "hard_gap",
    "isolated_affinity",
]

COMPILER_ISOLATED_BENCHMARK_EVIDENCE_SCHEMA = "scpn_qc_compiler_isolated_benchmark_evidence_v1"
COMPILER_ISOLATED_BENCHMARK_EVIDENCE_PREFIX = "compiler-isolated-benchmark-evidence-"
COMPILER_ISOLATED_BENCHMARK_SOURCE_PATHS = (
    "data/differentiable_phase_qnode/compiler_promotion_batch_20260706.json",
    "data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.json",
)
COMPILER_ISOLATED_BENCHMARK_SOURCE_IDS = (
    "compiler-promotion-batch-20260706",
    "compiler-evidence-boundary-20260705",
)
COMPILER_ISOLATED_BENCHMARK_CLAIM_BOUNDARY = (
    "Compiler isolated benchmark attachment evidence only: native whole-program "
    "AD value/gradient correctness and host isolation metadata may produce an "
    "attachable compiler benchmark artifact ID, but this record does not promote "
    "the compiler batch, baseline scorecard, provider execution, hardware execution, "
    "QPU execution, GPU execution, or production-performance claims. It carries "
    "no provider, hardware, QPU, GPU, Enzyme, or public promotion claim."
)


@dataclass(frozen=True)
class CompilerIsolatedBenchmarkEvidenceFiles:
    """File paths written for one compiler isolated benchmark evidence artifact."""

    artifact_id: str
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class CompilerIsolatedBenchmarkEvidence:
    """Validated compiler isolated benchmark attachment evidence.

    The record is attachable only when the benchmark metadata is classified as
    ``isolated_affinity`` and the native whole-program AD evidence executed a
    beyond-scalar compiler path within its declared gradient tolerance. The
    artifact itself never promotes the compiler batch; it only provides a
    candidate benchmark artifact ID that a later claim-ledger update may attach.
    """

    artifact_id: str
    native_execution_artifact_id: str
    native_execution_evidence: NativeWholeProgramADExecutionEvidence
    benchmark_metadata: BenchmarkIsolationMetadata
    source_artifact_ids: tuple[str, ...]
    source_artifact_paths: tuple[str, ...]
    benchmark_artifact_ids: tuple[str, ...]
    classification: CompilerBenchmarkClassification
    status: str
    ready_for_compiler_promotion_attachment: bool
    blocking_reasons: tuple[str, ...]
    executed_operation_families: tuple[str, ...]
    executed_case_count: int
    fail_closed_case_count: int
    max_value_error: float
    max_gradient_error: float
    total_runtime_seconds: float
    gradient_parity_tolerance: float
    schema: str = COMPILER_ISOLATED_BENCHMARK_EVIDENCE_SCHEMA
    promotion_ready: bool = False
    claim_boundary: str = COMPILER_ISOLATED_BENCHMARK_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate the evidence record cannot overstate promotion readiness."""
        if self.schema != COMPILER_ISOLATED_BENCHMARK_EVIDENCE_SCHEMA:
            raise ValueError("compiler isolated benchmark evidence schema drifted")
        if not self.artifact_id.strip():
            raise ValueError("compiler isolated benchmark evidence artifact_id must be non-empty")
        if self.native_execution_artifact_id != self.native_execution_evidence.artifact_id:
            raise ValueError("native execution artifact_id must match embedded evidence")
        if not self.source_artifact_ids or not self.source_artifact_paths:
            raise ValueError("compiler isolated benchmark evidence requires source artifacts")
        if len(self.source_artifact_ids) != len(self.source_artifact_paths):
            raise ValueError("source artifact ids and paths must align")
        if len(set(self.source_artifact_ids)) != len(self.source_artifact_ids):
            raise ValueError("source artifact ids must be unique")
        if len(set(self.source_artifact_paths)) != len(self.source_artifact_paths):
            raise ValueError("source artifact paths must be unique")
        if self.artifact_id not in self.benchmark_artifact_ids:
            raise ValueError("benchmark_artifact_ids must include this artifact")
        if self.native_execution_artifact_id not in self.benchmark_artifact_ids:
            raise ValueError("benchmark_artifact_ids must include native execution evidence")
        if self.promotion_ready:
            raise ValueError("compiler isolated benchmark evidence cannot promote the batch")
        expected_ready = (
            self.classification == "isolated_affinity"
            and self.benchmark_metadata.production_eligible
            and not self.blocking_reasons
        )
        if self.ready_for_compiler_promotion_attachment != expected_ready:
            raise ValueError("attachment readiness does not match classification and blockers")
        if self.status == "ready_for_compiler_promotion_attachment" and not expected_ready:
            raise ValueError("ready status requires attachment-ready evidence")
        if not self.executed_operation_families:
            raise ValueError("executed operation families must be non-empty")
        observed_executed = tuple(
            case for case in self.native_execution_evidence.cases if case.status == "executed"
        )
        observed_fail_closed = tuple(
            case for case in self.native_execution_evidence.cases if case.status == "fail_closed"
        )
        if self.executed_case_count != len(observed_executed):
            raise ValueError("executed_case_count must match embedded native evidence")
        if self.fail_closed_case_count != len(observed_fail_closed):
            raise ValueError("fail_closed_case_count must match embedded native evidence")
        if (
            self.executed_operation_families
            != self.native_execution_evidence.executed_operation_families
        ):
            raise ValueError("executed operation families must match embedded native evidence")
        if self.max_value_error != self.native_execution_evidence.max_value_error:
            raise ValueError("max_value_error must match embedded native evidence")
        if self.max_gradient_error != self.native_execution_evidence.max_gradient_error:
            raise ValueError("max_gradient_error must match embedded native evidence")
        if self.total_runtime_seconds != self.native_execution_evidence.total_runtime_seconds:
            raise ValueError("total_runtime_seconds must match embedded native evidence")
        if (
            self.gradient_parity_tolerance
            != self.native_execution_evidence.gradient_parity_tolerance
        ):
            raise ValueError("gradient_parity_tolerance must match embedded native evidence")
        for phrase in ("no provider", "hardware", "QPU", "GPU", "production-performance"):
            if phrase not in self.claim_boundary:
                raise ValueError("compiler isolated benchmark claim boundary is incomplete")

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready compiler benchmark evidence payload."""
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "classification": self.classification,
            "status": self.status,
            "promotion_ready": self.promotion_ready,
            "ready_for_compiler_promotion_attachment": (
                self.ready_for_compiler_promotion_attachment
            ),
            "blocking_reasons": list(self.blocking_reasons),
            "source_artifact_ids": list(self.source_artifact_ids),
            "source_artifact_paths": list(self.source_artifact_paths),
            "benchmark_artifact_ids": list(self.benchmark_artifact_ids),
            "native_execution_artifact_id": self.native_execution_artifact_id,
            "native_execution_evidence": self.native_execution_evidence.to_dict(),
            "benchmark_metadata": self.benchmark_metadata.to_dict(),
            "executed_operation_families": list(self.executed_operation_families),
            "executed_case_count": self.executed_case_count,
            "fail_closed_case_count": self.fail_closed_case_count,
            "max_value_error": self.max_value_error,
            "max_gradient_error": self.max_gradient_error,
            "total_runtime_seconds": self.total_runtime_seconds,
            "gradient_parity_tolerance": self.gradient_parity_tolerance,
            "claim_boundary": self.claim_boundary,
        }


def build_compiler_isolated_benchmark_evidence(
    *,
    native_execution_evidence: NativeWholeProgramADExecutionEvidence,
    benchmark_metadata: BenchmarkIsolationMetadata,
    stamp: str,
    source_artifact_ids: tuple[str, ...] = COMPILER_ISOLATED_BENCHMARK_SOURCE_IDS,
    source_artifact_paths: tuple[str, ...] = COMPILER_ISOLATED_BENCHMARK_SOURCE_PATHS,
) -> CompilerIsolatedBenchmarkEvidence:
    """Build fail-closed compiler isolated benchmark attachment evidence."""
    if not stamp.strip():
        raise ValueError("compiler isolated benchmark evidence stamp must be non-empty")
    artifact_id = f"{COMPILER_ISOLATED_BENCHMARK_EVIDENCE_PREFIX}{stamp}"
    blockers = _blocking_reasons(
        native_execution_evidence=native_execution_evidence,
        benchmark_metadata=benchmark_metadata,
    )
    classification = _classify_evidence(benchmark_metadata, blockers)
    ready = classification == "isolated_affinity" and not blockers
    status = "ready_for_compiler_promotion_attachment" if ready else f"blocked_{classification}"
    executed = tuple(case for case in native_execution_evidence.cases if case.status == "executed")
    fail_closed = tuple(
        case for case in native_execution_evidence.cases if case.status == "fail_closed"
    )
    return CompilerIsolatedBenchmarkEvidence(
        artifact_id=artifact_id,
        native_execution_artifact_id=native_execution_evidence.artifact_id,
        native_execution_evidence=native_execution_evidence,
        benchmark_metadata=benchmark_metadata,
        source_artifact_ids=source_artifact_ids,
        source_artifact_paths=source_artifact_paths,
        benchmark_artifact_ids=(artifact_id, native_execution_evidence.artifact_id),
        classification=classification,
        status=status,
        ready_for_compiler_promotion_attachment=ready,
        blocking_reasons=blockers,
        executed_operation_families=native_execution_evidence.executed_operation_families,
        executed_case_count=len(executed),
        fail_closed_case_count=len(fail_closed),
        max_value_error=native_execution_evidence.max_value_error,
        max_gradient_error=native_execution_evidence.max_gradient_error,
        total_runtime_seconds=native_execution_evidence.total_runtime_seconds,
        gradient_parity_tolerance=native_execution_evidence.gradient_parity_tolerance,
    )


def render_compiler_isolated_benchmark_evidence_markdown(
    evidence: CompilerIsolatedBenchmarkEvidence,
) -> str:
    """Render a reviewer-facing Markdown summary for the compiler benchmark artifact."""
    blockers = evidence.blocking_reasons or ("none",)
    lines = [
        "# Compiler Isolated Benchmark Evidence",
        "",
        f"- artifact_id: `{evidence.artifact_id}`",
        f"- classification: `{evidence.classification}`",
        f"- status: `{evidence.status}`",
        f"- promotion_ready: `{evidence.promotion_ready}`",
        (
            "- ready_for_compiler_promotion_attachment: "
            f"`{evidence.ready_for_compiler_promotion_attachment}`"
        ),
        f"- native_execution_artifact_id: `{evidence.native_execution_artifact_id}`",
        f"- max_value_error: `{evidence.max_value_error:.3e}`",
        f"- max_gradient_error: `{evidence.max_gradient_error:.3e}`",
        f"- gradient_parity_tolerance: `{evidence.gradient_parity_tolerance:.3e}`",
        f"- executed families: `{', '.join(evidence.executed_operation_families)}`",
        "",
        "Blocking reasons:",
        "",
        *[f"- {reason}" for reason in blockers],
        "",
        "| Benchmark artifact ID |",
        "|---|",
        *[f"| `{artifact_id}` |" for artifact_id in evidence.benchmark_artifact_ids],
        "",
        f"Claim boundary: {evidence.claim_boundary}",
        "",
    ]
    return "\n".join(lines)


def write_compiler_isolated_benchmark_evidence(
    output_dir: Path,
    evidence: CompilerIsolatedBenchmarkEvidence,
) -> CompilerIsolatedBenchmarkEvidenceFiles:
    """Write compiler isolated benchmark evidence JSON and Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_file_stem(evidence.artifact_id)
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_compiler_isolated_benchmark_evidence_markdown(evidence),
        encoding="utf-8",
    )
    return CompilerIsolatedBenchmarkEvidenceFiles(
        artifact_id=evidence.artifact_id,
        json_path=json_path,
        markdown_path=markdown_path,
    )


def _blocking_reasons(
    *,
    native_execution_evidence: NativeWholeProgramADExecutionEvidence,
    benchmark_metadata: BenchmarkIsolationMetadata,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if benchmark_metadata.classification != "isolated_affinity":
        blockers.append(
            "benchmark metadata classification is "
            f"{benchmark_metadata.classification}, not isolated_affinity"
        )
    if not benchmark_metadata.production_eligible:
        blockers.append("benchmark metadata is not production eligible")
    if not native_execution_evidence.beyond_scalar_executed:
        blockers.append("native whole-program AD evidence did not execute beyond scalar replay")
    if (
        native_execution_evidence.max_gradient_error
        > native_execution_evidence.gradient_parity_tolerance
    ):
        blockers.append("native whole-program AD gradient error exceeds declared tolerance")
    if not any(case.status == "fail_closed" for case in native_execution_evidence.cases):
        blockers.append("native whole-program AD evidence did not record fail-closed boundaries")
    return tuple(dict.fromkeys(blockers))


def _classify_evidence(
    benchmark_metadata: BenchmarkIsolationMetadata,
    blockers: tuple[str, ...],
) -> CompilerBenchmarkClassification:
    if not blockers and benchmark_metadata.classification == "isolated_affinity":
        return "isolated_affinity"
    if benchmark_metadata.classification == "functional_non_isolated":
        return "functional_non_isolated"
    return "hard_gap"


def _artifact_file_stem(artifact_id: str) -> str:
    if artifact_id.startswith(COMPILER_ISOLATED_BENCHMARK_EVIDENCE_PREFIX):
        suffix = artifact_id.removeprefix(COMPILER_ISOLATED_BENCHMARK_EVIDENCE_PREFIX)
        return f"compiler_isolated_benchmark_evidence_{suffix.replace('-', '_')}"
    return artifact_id.replace("-", "_")


__all__ = [
    "COMPILER_ISOLATED_BENCHMARK_CLAIM_BOUNDARY",
    "COMPILER_ISOLATED_BENCHMARK_EVIDENCE_PREFIX",
    "COMPILER_ISOLATED_BENCHMARK_EVIDENCE_SCHEMA",
    "COMPILER_ISOLATED_BENCHMARK_SOURCE_IDS",
    "COMPILER_ISOLATED_BENCHMARK_SOURCE_PATHS",
    "CompilerBenchmarkClassification",
    "CompilerIsolatedBenchmarkEvidence",
    "CompilerIsolatedBenchmarkEvidenceFiles",
    "build_compiler_isolated_benchmark_evidence",
    "render_compiler_isolated_benchmark_evidence_markdown",
    "write_compiler_isolated_benchmark_evidence",
]
