# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — compiler isolated benchmark evidence tests
# scpn-quantum-control -- compiler isolated benchmark evidence tests
"""Tests for compiler isolated benchmark attachment evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control.benchmarks import (
    AcceleratorEvidenceMetadata,
    BenchmarkIsolationMetadata,
    CompilerIsolatedBenchmarkEvidence,
    build_compiler_isolated_benchmark_evidence,
    render_compiler_isolated_benchmark_evidence_markdown,
    write_compiler_isolated_benchmark_evidence,
)
from scpn_quantum_control.compiler import (
    NativeWholeProgramADExecutionCase,
    NativeWholeProgramADExecutionEvidence,
    build_native_whole_program_ad_execution_evidence,
)


def test_compiler_isolated_benchmark_evidence_becomes_attachable_only_when_isolated() -> None:
    """A reserved-host run with correct native AD evidence may produce attachable IDs."""
    native_evidence = _native_execution_evidence()
    metadata = _isolation_metadata()

    evidence = build_compiler_isolated_benchmark_evidence(
        native_execution_evidence=native_evidence,
        benchmark_metadata=metadata,
        stamp="isolated_compiler_20260706",
    )

    payload = evidence.to_dict()
    assert payload["schema"] == "scpn_qc_compiler_isolated_benchmark_evidence_v1"
    assert payload["artifact_id"] == (
        "compiler-isolated-benchmark-evidence-isolated_compiler_20260706"
    )
    assert payload["classification"] == "isolated_affinity"
    assert payload["ready_for_compiler_promotion_attachment"] is True
    assert payload["promotion_ready"] is False
    assert payload["blocking_reasons"] == []
    assert payload["native_execution_artifact_id"] == native_evidence.artifact_id
    assert payload["max_gradient_error"] == native_evidence.max_gradient_error
    assert payload["benchmark_artifact_ids"] == [
        "compiler-isolated-benchmark-evidence-isolated_compiler_20260706",
        "native-whole-program-ad-execution-test",
    ]
    assert "no provider, hardware, QPU, GPU" in evidence.claim_boundary


def test_compiler_isolated_benchmark_evidence_fails_closed_without_isolation() -> None:
    """Local or incomplete host metadata must not produce attachable compiler IDs."""
    native_evidence = _native_execution_evidence()
    metadata = _isolation_metadata(
        env={
            "RUNNER_ENVIRONMENT": "github-hosted",
            "RUNNER_LABELS": "ubuntu-latest",
            "RUNNER_NAME": "github-runner",
        },
        cpu_affinity=None,
        isolation_method=None,
        load_before=None,
        load_after=None,
        governor=None,
        frequency_mhz=None,
    )

    evidence = build_compiler_isolated_benchmark_evidence(
        native_execution_evidence=native_evidence,
        benchmark_metadata=metadata,
        stamp="local",
    )

    assert evidence.classification == "functional_non_isolated"
    assert evidence.ready_for_compiler_promotion_attachment is False
    assert evidence.promotion_ready is False
    assert any(
        "benchmark metadata classification is functional_non_isolated" in reason
        for reason in evidence.blocking_reasons
    )
    assert any(
        "benchmark metadata is not production eligible" in reason
        for reason in evidence.blocking_reasons
    )


def test_compiler_isolated_benchmark_evidence_blocks_scalar_only_native_evidence() -> None:
    """Compiler benchmark attachments require native AD evidence beyond scalar replay."""
    metadata = _isolation_metadata()
    scalar_only = build_native_whole_program_ad_execution_evidence(
        artifact_id="native-whole-program-ad-scalar-only",
        cases=(
            NativeWholeProgramADExecutionCase(
                case_id="scalar_poly_3",
                operation_family="scalar",
                operand_dimension=3,
                status="executed",
                value_error=0.0,
                gradient_error=0.0,
                runtime_seconds=0.001,
                native_symbol="whole_program_ad_scalar_value",
                failure_class=None,
                claim_boundary="test scalar-only native evidence",
            ),
        ),
        gradient_parity_tolerance=1e-6,
        fail_closed_boundaries={"determinant": 20},
        claim_boundary="test scalar-only native evidence",
    )

    evidence = build_compiler_isolated_benchmark_evidence(
        native_execution_evidence=scalar_only,
        benchmark_metadata=metadata,
        stamp="scalar_only",
    )

    assert evidence.classification == "hard_gap"
    assert evidence.ready_for_compiler_promotion_attachment is False
    assert any("beyond scalar" in reason for reason in evidence.blocking_reasons)


def test_compiler_isolated_benchmark_evidence_markdown_and_writer(tmp_path: Path) -> None:
    """JSON and Markdown writers must preserve the validated attachment boundary."""
    evidence = build_compiler_isolated_benchmark_evidence(
        native_execution_evidence=_native_execution_evidence(),
        benchmark_metadata=_isolation_metadata(),
        stamp="writer",
    )

    markdown = render_compiler_isolated_benchmark_evidence_markdown(evidence)
    files = write_compiler_isolated_benchmark_evidence(tmp_path, evidence)

    assert "# Compiler Isolated Benchmark Evidence" in markdown
    assert "ready_for_compiler_promotion_attachment: `True`" in markdown
    assert files.artifact_id == evidence.artifact_id
    assert files.json_path.name == "compiler_isolated_benchmark_evidence_writer.json"
    assert files.markdown_path.name == "compiler_isolated_benchmark_evidence_writer.md"
    written = json.loads(files.json_path.read_text(encoding="utf-8"))
    assert written == evidence.to_dict()
    assert "Claim boundary:" in files.markdown_path.read_text(encoding="utf-8")


def test_compiler_isolated_benchmark_evidence_exports_are_public() -> None:
    """The benchmark package must expose the attachment evidence contract."""
    assert CompilerIsolatedBenchmarkEvidence.__name__ in {
        "CompilerIsolatedBenchmarkEvidence",
    }
    assert "CompilerIsolatedBenchmarkEvidence" in scpn.__all__
    assert scpn.CompilerIsolatedBenchmarkEvidence is CompilerIsolatedBenchmarkEvidence


def _native_execution_evidence() -> NativeWholeProgramADExecutionEvidence:
    """Build a compact native execution evidence record for attachment tests."""
    cases = (
        NativeWholeProgramADExecutionCase(
            case_id="scalar_poly_3",
            operation_family="scalar",
            operand_dimension=3,
            status="executed",
            value_error=0.0,
            gradient_error=0.0,
            runtime_seconds=0.001,
            native_symbol="whole_program_ad_scalar_value",
            failure_class=None,
            claim_boundary="test native evidence",
        ),
        NativeWholeProgramADExecutionCase(
            case_id="determinant_2x2",
            operation_family="determinant",
            operand_dimension=2,
            status="executed",
            value_error=1e-12,
            gradient_error=1e-12,
            runtime_seconds=0.002,
            native_symbol="whole_program_ad_determinant_value",
            failure_class=None,
            claim_boundary="test native evidence",
        ),
        NativeWholeProgramADExecutionCase(
            case_id="determinant_20x20_fail_closed",
            operation_family="determinant",
            operand_dimension=20,
            status="fail_closed",
            value_error=None,
            gradient_error=None,
            runtime_seconds=None,
            native_symbol=None,
            failure_class="determinant native lowering supports dimension <= 4",
            claim_boundary="test native evidence",
        ),
    )
    return build_native_whole_program_ad_execution_evidence(
        artifact_id="native-whole-program-ad-execution-test",
        cases=cases,
        gradient_parity_tolerance=1e-6,
        fail_closed_boundaries={"determinant": 20},
        claim_boundary="test native evidence",
    )


def _isolation_metadata(
    *,
    env: dict[str, str] | None = None,
    cpu_affinity: str | None = "2",
    isolation_method: str | None = "taskset+chrt",
    load_before: tuple[float, float, float] | None = (0.05, 0.04, 0.03),
    load_after: tuple[float, float, float] | None = (0.06, 0.05, 0.04),
    governor: str | None = "performance",
    frequency_mhz: float | None = 3200.0,
) -> BenchmarkIsolationMetadata:
    """Build benchmark isolation metadata with controllable runner fields."""
    runner_env = {
        "RUNNER_ENVIRONMENT": "self-hosted",
        "RUNNER_LABELS": "self-hosted,linux,isolated-benchmark,ml350",
        "RUNNER_NAME": "ml350",
        "GITHUB_RUN_ID": "123456",
        "GITHUB_SHA": "abcdef123456",
        "RUNNER_OS": "Linux",
    }
    if env is not None:
        runner_env = env
    return BenchmarkIsolationMetadata.from_ci_environment(
        runner_env,
        command=(
            "python",
            "scripts/run_compiler_isolated_benchmark_evidence.py",
            "--stamp",
            "isolated_compiler_20260706",
        ),
        cpu_affinity=cpu_affinity,
        isolation_method=isolation_method,
        load_before=load_before,
        load_after=load_after,
        governor=governor,
        frequency_mhz=frequency_mhz,
        heavy_jobs_running=False,
        accelerator_metadata=AcceleratorEvidenceMetadata.cpu_only(),
    )


def _ready_evidence() -> CompilerIsolatedBenchmarkEvidence:
    """Build attachment-ready evidence from the compact native record and a reserved host."""
    return build_compiler_isolated_benchmark_evidence(
        native_execution_evidence=_native_execution_evidence(),
        benchmark_metadata=_isolation_metadata(),
        stamp="refusals",
    )


def test_compiler_isolated_benchmark_evidence_refuses_inconsistent_records() -> None:
    """Every field that restates the native evidence or the source list must agree with it."""
    evidence = _ready_evidence()
    native_id = evidence.native_execution_artifact_id
    ids = evidence.source_artifact_ids
    paths = evidence.source_artifact_paths
    cases: list[tuple[dict[str, Any], str]] = [
        ({"schema": "other"}, "^compiler isolated benchmark evidence schema drifted$"),
        ({"artifact_id": "  "}, "artifact_id must be non-empty$"),
        ({"native_execution_artifact_id": "other"}, "^native execution artifact_id must match"),
        ({"source_artifact_ids": ()}, "requires source artifacts$"),
        ({"source_artifact_paths": ()}, "requires source artifacts$"),
        ({"source_artifact_ids": ids[:1]}, "^source artifact ids and paths must align$"),
        ({"source_artifact_ids": (ids[0], ids[0])}, "^source artifact ids must be unique$"),
        (
            {"source_artifact_paths": (paths[0], paths[0])},
            "^source artifact paths must be unique$",
        ),
        ({"benchmark_artifact_ids": (native_id,)}, "must include this artifact$"),
        ({"benchmark_artifact_ids": (evidence.artifact_id,)}, "must include native execution"),
        ({"promotion_ready": True}, "cannot promote the batch$"),
        ({"ready_for_compiler_promotion_attachment": False}, "^attachment readiness does not"),
        ({"executed_operation_families": ()}, "^executed operation families must be non-empty$"),
        ({"executed_case_count": evidence.executed_case_count + 1}, "^executed_case_count must"),
        ({"fail_closed_case_count": 0}, "^fail_closed_case_count must match"),
        ({"executed_operation_families": ("scalar",)}, "^executed operation families must match"),
        ({"max_value_error": 1.0}, "^max_value_error must match"),
        ({"max_gradient_error": 1.0}, "^max_gradient_error must match"),
        ({"total_runtime_seconds": 9.0}, "^total_runtime_seconds must match"),
        ({"gradient_parity_tolerance": 1.0}, "^gradient_parity_tolerance must match"),
        ({"claim_boundary": "no provider, hardware, QPU, GPU"}, "claim boundary is incomplete$"),
    ]
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(evidence, **changes)


def test_compiler_isolated_benchmark_evidence_refuses_a_ready_status_on_blocked_evidence() -> None:
    """A blocked record cannot carry the ready status, and the builder needs a stamp."""
    blocked = build_compiler_isolated_benchmark_evidence(
        native_execution_evidence=_native_execution_evidence(),
        benchmark_metadata=_isolation_metadata(cpu_affinity=None),
        stamp="blocked",
    )
    assert blocked.status == "blocked_hard_gap"
    with pytest.raises(ValueError, match="^ready status requires attachment-ready evidence$"):
        replace(blocked, status="ready_for_compiler_promotion_attachment")
    with pytest.raises(ValueError, match="^attachment readiness does not match"):
        replace(blocked, ready_for_compiler_promotion_attachment=True)
    with pytest.raises(ValueError, match="stamp must be non-empty$"):
        build_compiler_isolated_benchmark_evidence(
            native_execution_evidence=_native_execution_evidence(),
            benchmark_metadata=_isolation_metadata(),
            stamp=" ",
        )


def test_compiler_isolated_benchmark_evidence_blocks_an_aggregate_gradient_above_tolerance() -> (
    None
):
    """An error at the tolerance is attachable; an aggregate the native record admits above it is not."""
    tolerance = 1e-6
    at_tolerance = replace(_native_execution_evidence().cases[1], gradient_error=tolerance)
    native = build_native_whole_program_ad_execution_evidence(
        artifact_id="native-whole-program-ad-at-tolerance",
        cases=(
            _native_execution_evidence().cases[0],
            at_tolerance,
            _native_execution_evidence().cases[2],
        ),
        gradient_parity_tolerance=tolerance,
        fail_closed_boundaries={"determinant": 20},
        claim_boundary="test native evidence",
    )
    at_limit = build_compiler_isolated_benchmark_evidence(
        native_execution_evidence=native,
        benchmark_metadata=_isolation_metadata(),
        stamp="at_tolerance",
    )
    assert at_limit.ready_for_compiler_promotion_attachment is True
    above = replace(native, max_gradient_error=tolerance + 5e-19)
    assert above.max_gradient_error > above.gradient_parity_tolerance

    evidence = build_compiler_isolated_benchmark_evidence(
        native_execution_evidence=above,
        benchmark_metadata=_isolation_metadata(),
        stamp="above_tolerance",
    )

    assert evidence.classification == "hard_gap"
    assert evidence.ready_for_compiler_promotion_attachment is False
    assert evidence.blocking_reasons == (
        "native whole-program AD gradient error exceeds declared tolerance",
    )


def test_compiler_isolated_benchmark_evidence_blocks_native_evidence_without_fail_closed_cases() -> (
    None
):
    """Attachments need the native record to show where its lowering stops."""
    native = build_native_whole_program_ad_execution_evidence(
        artifact_id="native-whole-program-ad-no-boundary",
        cases=_native_execution_evidence().cases[:2],
        gradient_parity_tolerance=1e-6,
        fail_closed_boundaries={},
        claim_boundary="test native evidence",
    )

    evidence = build_compiler_isolated_benchmark_evidence(
        native_execution_evidence=native,
        benchmark_metadata=_isolation_metadata(),
        stamp="no_boundary",
    )

    assert evidence.status == "blocked_hard_gap"
    assert evidence.blocking_reasons == (
        "native whole-program AD evidence did not record fail-closed boundaries",
    )
    assert "- native whole-program AD evidence did not record" in (
        render_compiler_isolated_benchmark_evidence_markdown(evidence)
    )


def test_compiler_isolated_benchmark_writer_names_an_unprefixed_artifact_by_its_id(
    tmp_path: Path,
) -> None:
    """An artifact ID without the evidence prefix is written under its own underscored name."""
    evidence = _ready_evidence()
    external = replace(
        evidence,
        artifact_id="external-benchmark-7",
        benchmark_artifact_ids=("external-benchmark-7", evidence.native_execution_artifact_id),
    )

    files = write_compiler_isolated_benchmark_evidence(tmp_path, external)

    assert files.json_path.name == "external_benchmark_7.json"
    assert files.markdown_path.name == "external_benchmark_7.md"
    assert json.loads(files.json_path.read_text(encoding="utf-8")) == external.to_dict()
