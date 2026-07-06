# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- compiler isolated benchmark evidence tests
"""Tests for compiler isolated benchmark attachment evidence."""

from __future__ import annotations

import json
from pathlib import Path

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
