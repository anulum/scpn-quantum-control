# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNode Compiler Lowering
"""Tests for compiler/mlir.py phase-QNode lowering reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control as scpn
import scpn_quantum_control.compiler as compiler
from scpn_quantum_control.compiler.mlir import (
    EnzymeMLIRBenchmarkAttachment,
    EnzymeMLIRCompilerADBreadthArtifact,
    EnzymeMLIRCompilerADBreadthCaseEvidence,
    EnzymeMLIRCompilerADBreadthEvidence,
    EnzymeMLIRMaturityAuditResult,
    EnzymeNativeExecutionEvidence,
    MLIRLLVMCorrectnessEvidence,
    PhaseQNodeMLIRRuntimeExecutable,
    build_enzyme_mlir_benchmark_attachment,
    build_enzyme_mlir_compiler_ad_breadth_artifact,
    build_enzyme_mlir_compiler_ad_breadth_evidence,
    build_enzyme_mlir_compiler_ad_breadth_gap_artifact,
    compile_phase_qnode_circuit_to_mlir_runtime,
    lower_phase_qnode_circuit_to_mlir,
    render_enzyme_mlir_compiler_ad_breadth_artifact_markdown,
    run_enzyme_mlir_maturity_audit,
    write_enzyme_mlir_compiler_ad_breadth_artifact,
)
from scpn_quantum_control.phase.qnode_affinity_benchmark import (
    PhaseQNodeAffinityArtifactValidation,
)
from scpn_quantum_control.phase.qnode_circuit import PauliTerm, PhaseQNodeCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
ENZYME_MLIR_AUDIT_PATH = (
    REPO_ROOT / "data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.json"
)
ENZYME_MLIR_BREADTH_ARTIFACT_PATH = (
    REPO_ROOT
    / "data/differentiable_phase_qnode/enzyme_mlir_compiler_ad_breadth_artifact_20260706.json"
)


def test_phase_qnode_compiler_lowering_reports_registered_subset() -> None:
    """Lower the registered Phase-QNode subset with bounded claim metadata."""
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("ry", (0,), 0), ("cnot", (0, 1)), ("rzz", (0, 1), 1)),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )

    module = lower_phase_qnode_circuit_to_mlir(circuit, np.array([0.2, -0.3], dtype=float))

    assert module.dialect == "scpn_phase_qnode"
    assert module.metadata["supported"] is True
    assert module.metadata["primitive_support"]["gates"] == ["ry", "cnot", "rzz"]
    assert module.metadata["shape_limits"]["max_qubits"] >= 2
    assert module.metadata["rust_pyo3_parity"] == "blocked: no Rust phase-QNode lowering backend"
    assert "scpn_phase_qnode.ry" in module.text
    assert "scpn_phase_qnode.expectation" in module.text
    assert module.metadata["interpreter_fallback"] == (
        "blocked: cannot report interpreter fallback as compiled success"
    )
    assert module.metadata["dialect_operations"][0] == {
        "op": "scpn_phase_qnode.ry",
        "gate": "ry",
        "qubits": [0],
        "parameter_index": 0,
        "operand_type": "f64",
        "result_type": "statevector",
    }


def test_phase_qnode_mlir_runtime_executes_value_and_gradient() -> None:
    """Execute public value and gradient kernels from the verified runtime adapter."""
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0), ("rx", (0,), 1)),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )
    params = np.array([0.2, -0.3], dtype=float)

    executable = compile_phase_qnode_circuit_to_mlir_runtime(circuit, params)
    value = executable.value(params)
    gradient = executable.gradient(params)
    payload = executable.to_dict()

    assert isinstance(executable, PhaseQNodeMLIRRuntimeExecutable)
    assert executable.runtime_backend == "scpn_mlir_runtime_adapter"
    assert executable.parameter_shape == (2,)
    assert executable.parameter_dtype == "float64"
    assert executable.verification["value_close"] is True
    assert executable.verification["gradient_close"] is True
    assert payload["interpreter_fallback"] == (
        "blocked: cannot report interpreter fallback as compiled success"
    )
    np.testing.assert_allclose(value, np.cos(params[0]) * np.cos(params[1]), atol=1e-12)
    np.testing.assert_allclose(
        gradient,
        np.array(
            [-np.sin(params[0]) * np.cos(params[1]), -np.cos(params[0]) * np.sin(params[1])],
            dtype=float,
        ),
        atol=1e-12,
    )


def test_phase_qnode_mlir_runtime_verifies_shape_and_dtype() -> None:
    """Reject runtime parameters whose shape or dtype violates the compiled contract."""
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )
    executable = compile_phase_qnode_circuit_to_mlir_runtime(
        circuit,
        np.array([0.2], dtype=float),
    )

    with pytest.raises(ValueError, match="runtime parameter shape"):
        executable.value(np.array([0.2, 0.3], dtype=float))
    with pytest.raises(ValueError, match="runtime parameters must contain finite real"):
        executable.gradient(np.array([1.0 + 0.0j]))


def test_phase_qnode_mlir_runtime_exports_are_public() -> None:
    """Keep Phase-QNode runtime records and compilers on public facades."""
    assert (
        compiler.compile_phase_qnode_circuit_to_mlir_runtime
        is compile_phase_qnode_circuit_to_mlir_runtime
    )
    assert (
        scpn.compile_phase_qnode_circuit_to_mlir_runtime
        is compile_phase_qnode_circuit_to_mlir_runtime
    )
    assert compiler.PhaseQNodeMLIRRuntimeExecutable is PhaseQNodeMLIRRuntimeExecutable
    assert scpn.PhaseQNodeMLIRRuntimeExecutable is PhaseQNodeMLIRRuntimeExecutable


def test_phase_qnode_compiler_lowering_fails_closed_for_unsupported_circuit() -> None:
    """Reject unsupported Phase-QNode gates before emitting MLIR metadata."""
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("u3", (0,), 0),),
        observable="pauli_z",
    )

    with pytest.raises(ValueError, match="phase-QNode lowering failed closed"):
        lower_phase_qnode_circuit_to_mlir(circuit, np.array([0.2], dtype=float))


def test_enzyme_mlir_maturity_audit_records_runtime_evidence_and_toolchain_gaps() -> None:
    """Record verified runtime checks alongside absent native toolchain gaps."""
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0), ("rx", (0,), 1)),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )

    result = run_enzyme_mlir_maturity_audit(
        circuit,
        np.array([0.2, -0.3], dtype=float),
        toolchain_probe=lambda command: None,
    )
    payload = cast(dict[str, Any], result.to_dict())

    assert isinstance(result, EnzymeMLIRMaturityAuditResult)
    assert result.scpn_mlir_runtime_verified is True
    assert result.native_llvm_jit_surface == "available: bounded in-process native LLVM/JIT"
    assert result.ready_for_provider_exceedance is False
    assert payload["claim_boundary"] == "bounded_enzyme_mlir_compiler_maturity_audit"
    assert payload["toolchain"]["enzyme"]["available"] is False
    assert payload["toolchain"]["mlir-opt"]["failure_class"] == "toolchain_missing"
    assert "isolated benchmark artefact missing" in payload["hard_gaps"]
    assert "MLIR/LLVM correctness artefact missing" in payload["hard_gaps"]
    assert payload["correctness_checks"]["phase_qnode_value_close"] is True
    assert payload["correctness_checks"]["phase_qnode_gradient_close"] is True


def test_enzyme_mlir_maturity_audit_records_versions_but_requires_execution_artifacts() -> None:
    """Keep detected toolchain versions non-promotional without execution artefacts."""
    result = run_enzyme_mlir_maturity_audit(
        toolchain_probe=lambda command: f"/opt/toolchain/bin/{command}",
        version_probe=lambda executable: f"{executable} version 1.2.3",
        isolated_benchmark_artifact_id="iso-bench-001",
        mlir_llvm_correctness_artifact_id="mlir-correctness-001",
    )
    payload = cast(dict[str, Any], result.to_dict())

    assert payload["toolchain"]["enzyme"]["available"] is True
    assert payload["toolchain"]["opt"]["version"] == "/opt/toolchain/bin/opt version 1.2.3"
    assert payload["mlir_llvm_correctness_evidence"]["artifact_id"] == "mlir-correctness-001"
    assert result.ready_for_provider_exceedance is False
    assert "native Enzyme execution artefact missing" in result.hard_gaps
    assert "MLIR/LLVM correctness check missing" not in result.hard_gaps


def test_enzyme_mlir_maturity_audit_keeps_runtime_gap_non_promotional() -> None:
    """Retain named native execution failures as promotion-blocking hard gaps."""
    evidence = EnzymeNativeExecutionEvidence(
        artifact_id="enzyme-runtime-gap-001",
        status="hard_gap",
        failure_class="runtime_error",
        value_error=None,
        gradient_error=None,
        runtime_seconds=None,
        toolchain={"enzyme_ad": "0.0.6", "runner": "enzyme_jax_runner.py"},
        setup_instructions="Configured Enzyme runner failed during MHLO lowering.",
        claim_boundary="Runtime comparison gap only; no hidden success or promoted claim.",
    )

    result = run_enzyme_mlir_maturity_audit(
        toolchain_probe=lambda command: f"/opt/toolchain/bin/{command}",
        version_probe=lambda executable: f"{executable} version 1.2.3",
        isolated_benchmark_artifact_id="iso-bench-001",
        native_enzyme_execution_evidence=evidence,
        mlir_llvm_correctness_artifact_id="mlir-correctness-001",
    )
    payload = cast(dict[str, Any], result.to_dict())

    assert result.native_enzyme_execution_artifact_id == "enzyme-runtime-gap-001"
    assert payload["native_enzyme_execution_evidence"]["failure_class"] == "runtime_error"
    assert payload["native_enzyme_execution_evidence"]["passed"] is False
    assert "native Enzyme execution hard gap: runtime_error" in result.hard_gaps
    assert result.ready_for_provider_exceedance is False


def test_enzyme_mlir_maturity_audit_separates_installed_toolchain_from_runtime_gap() -> None:
    """Distinguish installed compiler commands from failed native execution."""
    evidence = EnzymeNativeExecutionEvidence(
        artifact_id="enzyme-runtime-gap-installed-stack-001",
        status="hard_gap",
        failure_class="runtime_error",
        value_error=None,
        gradient_error=None,
        runtime_seconds=None,
        toolchain={
            "enzyme": "Enzyme LLVM plugin 0.0.79",
            "opt": "Ubuntu LLVM version 18.1.3",
            "mlir-opt": "Ubuntu LLVM version 18.1.3",
            "clang": "Ubuntu clang version 18.1.3",
        },
        setup_instructions=(
            "The compiler commands are installed; this row remains blocked by "
            "native Enzyme execution correctness, not by PATH discovery."
        ),
        claim_boundary="Installed toolchain with runtime hard gap only.",
    )

    result = run_enzyme_mlir_maturity_audit(
        toolchain_probe=lambda command: f"/usr/bin/{command}",
        version_probe=lambda executable: f"{executable} 18.1.3",
        native_enzyme_execution_evidence=evidence,
        mlir_llvm_correctness_artifact_id="mlir-correctness-installed-stack-001",
    )

    assert all(status.available for status in result.toolchain.values())
    assert not any("toolchain unavailable" in gap for gap in result.hard_gaps)
    assert "isolated benchmark artefact missing" in result.hard_gaps
    assert "native Enzyme execution hard gap: runtime_error" in result.hard_gaps
    assert result.ready_for_provider_exceedance is False


def test_enzyme_mlir_maturity_audit_requires_all_artifacts_for_promotion() -> None:
    """Require every evidence attachment before provider-exceedance promotion."""
    evidence = EnzymeNativeExecutionEvidence(
        artifact_id="enzyme-success-001",
        status="success",
        failure_class=None,
        value_error=0.0,
        gradient_error=0.0,
        runtime_seconds=0.01,
        toolchain={"enzyme": "1.0", "llvm": "18.1.3"},
        setup_instructions=None,
        claim_boundary="Bounded native Enzyme execution evidence.",
    )

    result = run_enzyme_mlir_maturity_audit(
        toolchain_probe=lambda command: f"/opt/toolchain/bin/{command}",
        version_probe=lambda executable: f"{executable} version 1.2.3",
        isolated_benchmark_artifact_id="iso-bench-001",
        native_enzyme_execution_evidence=evidence,
        mlir_llvm_correctness_artifact_id="mlir-correctness-001",
    )

    assert result.ready_for_provider_exceedance is False
    assert "compiler AD breadth artifact missing" in result.hard_gaps
    assert "compiler AD breadth evidence missing" in result.hard_gaps


def _compiler_ad_breadth_evidence(
    *,
    isolated_benchmark_artifact_id: str = "iso-bench-001",
    case_overrides: dict[str, bool] | None = None,
) -> EnzymeMLIRCompilerADBreadthEvidence:
    cases = {
        "scalar_forward_mode": True,
        "scalar_reverse_mode": True,
        "vector_jvp": True,
        "vector_vjp": True,
        "matrix_jvp": True,
        "matrix_vjp": True,
        "loop_activity": True,
        "alias_activity": True,
        "mlir_lowering": True,
        "llvm_ir_generation": True,
        "native_enzyme_execution": True,
    }
    if case_overrides is not None:
        cases.update(case_overrides)
    return build_enzyme_mlir_compiler_ad_breadth_evidence(
        artifact_id="enzyme-mlir-breadth-001",
        cases=cases,
        transform_modes=("forward", "reverse", "jvp", "vjp"),
        frontend_languages=("c", "cxx", "rust", "julia", "mlir", "llvm_ir"),
        isolated_benchmark_artifact_id=isolated_benchmark_artifact_id,
        max_abs_error=0.0,
        runtime_seconds=0.12,
        claim_boundary="captured Enzyme/MLIR compiler AD breadth evidence.",
    )


def test_enzyme_mlir_maturity_audit_requires_complete_compiler_ad_breadth() -> None:
    """Accept a complete linked compiler-AD breadth evidence chain."""
    evidence = EnzymeNativeExecutionEvidence(
        artifact_id="enzyme-success-001",
        status="success",
        failure_class=None,
        value_error=0.0,
        gradient_error=0.0,
        runtime_seconds=0.01,
        toolchain={"enzyme": "1.0", "llvm": "18.1.3"},
        setup_instructions=None,
        claim_boundary="Bounded native Enzyme execution evidence.",
    )

    artifact = _compiler_ad_breadth_artifact()
    result = run_enzyme_mlir_maturity_audit(
        toolchain_probe=lambda command: f"/opt/toolchain/bin/{command}",
        version_probe=lambda executable: f"{executable} version 1.2.3",
        isolated_benchmark_artifact_id="iso-bench-001",
        isolated_benchmark_evidence=_benchmark_attachment(),
        native_enzyme_execution_evidence=evidence,
        mlir_llvm_correctness_artifact_id="mlir-correctness-001",
        compiler_ad_breadth_evidence=artifact.to_breadth_evidence(),
        compiler_ad_breadth_artifact=artifact,
    )

    assert result.hard_gaps == ()
    assert result.ready_for_provider_exceedance is True
    payload = cast(dict[str, Any], result.to_dict())
    assert payload["compiler_ad_breadth_evidence"]["passed"] is True
    assert payload["compiler_ad_breadth_evidence"]["case_count"] == 11
    assert payload["compiler_ad_breadth_evidence"]["transform_modes"] == [
        "forward",
        "jvp",
        "reverse",
        "vjp",
    ]
    assert payload["compiler_ad_breadth_artifact"]["promotion_ready"] is True


def _benchmark_attachment(
    *,
    benchmark_artifact_id: str = "iso-bench-001",
    promotion_ready: bool = True,
    evidence_label: str = "isolated_affinity",
    production_benchmark: bool = True,
    raw_timing_row_count: int = 5,
    missing_requirements: tuple[str, ...] = (),
) -> EnzymeMLIRBenchmarkAttachment:
    validation = PhaseQNodeAffinityArtifactValidation(
        artifact_path="data/benchmarks/enzyme_mlir_iso.json",
        artifact_sha256="a" * 64,
        benchmark_artifact_id=benchmark_artifact_id,
        evidence_label=evidence_label,
        production_benchmark=production_benchmark,
        promotion_ready=promotion_ready,
        raw_timing_row_count=raw_timing_row_count,
        missing_requirements=missing_requirements,
        claim_boundary="isolated benchmark validation for Enzyme/MLIR compiler AD.",
    )
    return build_enzyme_mlir_benchmark_attachment(
        validation=validation,
        required_breadth_cases=tuple(sorted(_compiler_ad_breadth_evidence().cases)),
        claim_boundary="Enzyme/MLIR compiler-AD benchmark attachment.",
    )


def _compiler_ad_breadth_artifact(
    *,
    isolated_benchmark_evidence: EnzymeMLIRBenchmarkAttachment | None = None,
    case_overrides: dict[str, bool] | None = None,
) -> EnzymeMLIRCompilerADBreadthArtifact:
    rows: list[EnzymeMLIRCompilerADBreadthCaseEvidence] = []
    for case_id in sorted(_compiler_ad_breadth_evidence().cases):
        passed = case_overrides.get(case_id, True) if case_overrides is not None else True
        rows.append(
            EnzymeMLIRCompilerADBreadthCaseEvidence(
                case_id=case_id,
                status="success" if passed else "hard_gap",
                transform_modes=("forward", "reverse", "jvp", "vjp"),
                frontend_language="llvm_ir" if case_id == "native_enzyme_execution" else "mlir",
                value_error=0.0 if passed else None,
                gradient_error=0.0 if passed else None,
                runtime_seconds=0.01 if passed else None,
                artifact_refs={"raw_case": f"data/differentiable_phase_qnode/{case_id}.json"},
                failure_class=None if passed else "missing_case_evidence",
                setup_instructions=None if passed else "Attach passing raw case evidence.",
                claim_boundary="bounded Enzyme/MLIR compiler-AD breadth case evidence.",
            )
        )
    return build_enzyme_mlir_compiler_ad_breadth_artifact(
        artifact_id="enzyme-mlir-breadth-artifact-001",
        cases=tuple(rows),
        isolated_benchmark_evidence=isolated_benchmark_evidence or _benchmark_attachment(),
        claim_boundary="bounded Enzyme/MLIR compiler-AD breadth artifact.",
    )


def test_enzyme_mlir_maturity_audit_requires_validated_isolated_benchmark() -> None:
    """Reject an isolated benchmark identifier without validated attachment evidence."""
    evidence = EnzymeNativeExecutionEvidence(
        artifact_id="enzyme-success-001",
        status="success",
        failure_class=None,
        value_error=0.0,
        gradient_error=0.0,
        runtime_seconds=0.01,
        toolchain={"enzyme": "1.0", "llvm": "18.1.3"},
        setup_instructions=None,
        claim_boundary="Bounded native Enzyme execution evidence.",
    )

    result = run_enzyme_mlir_maturity_audit(
        toolchain_probe=lambda command: f"/opt/toolchain/bin/{command}",
        version_probe=lambda executable: f"{executable} version 1.2.3",
        isolated_benchmark_artifact_id="iso-bench-001",
        isolated_benchmark_evidence=_benchmark_attachment(
            promotion_ready=False,
            evidence_label="functional_non_isolated",
            production_benchmark=False,
            missing_requirements=("isolated_affinity evidence label",),
        ),
        native_enzyme_execution_evidence=evidence,
        mlir_llvm_correctness_artifact_id="mlir-correctness-001",
        compiler_ad_breadth_evidence=_compiler_ad_breadth_evidence(),
    )

    assert result.ready_for_provider_exceedance is False
    assert "validated isolated benchmark evidence missing" in result.hard_gaps
    payload = cast(dict[str, Any], result.to_dict())
    assert payload["isolated_benchmark_evidence"]["promotion_ready"] is False
    assert payload["isolated_benchmark_evidence"]["evidence_label"] == "functional_non_isolated"


def test_enzyme_mlir_maturity_audit_accepts_validated_isolated_benchmark() -> None:
    """Accept promotion-ready isolated benchmark evidence with matching identifiers."""
    evidence = EnzymeNativeExecutionEvidence(
        artifact_id="enzyme-success-001",
        status="success",
        failure_class=None,
        value_error=0.0,
        gradient_error=0.0,
        runtime_seconds=0.01,
        toolchain={"enzyme": "1.0", "llvm": "18.1.3"},
        setup_instructions=None,
        claim_boundary="Bounded native Enzyme execution evidence.",
    )

    artifact = _compiler_ad_breadth_artifact()
    result = run_enzyme_mlir_maturity_audit(
        toolchain_probe=lambda command: f"/opt/toolchain/bin/{command}",
        version_probe=lambda executable: f"{executable} version 1.2.3",
        isolated_benchmark_artifact_id="iso-bench-001",
        isolated_benchmark_evidence=_benchmark_attachment(),
        native_enzyme_execution_evidence=evidence,
        mlir_llvm_correctness_artifact_id="mlir-correctness-001",
        compiler_ad_breadth_evidence=artifact.to_breadth_evidence(),
        compiler_ad_breadth_artifact=artifact,
    )

    assert result.hard_gaps == ()
    assert result.ready_for_provider_exceedance is True
    payload = cast(dict[str, Any], result.to_dict())
    assert payload["isolated_benchmark_evidence"]["required_breadth_case_count"] == 11


def test_enzyme_mlir_breadth_artifact_builds_promotion_evidence() -> None:
    """Derive promotion evidence from a complete passing breadth artefact."""
    artifact = _compiler_ad_breadth_artifact()
    evidence = artifact.to_breadth_evidence()
    payload = artifact.to_dict()

    assert artifact.promotion_ready is True
    assert evidence.passed is True
    assert evidence.isolated_benchmark_artifact_id == "iso-bench-001"
    assert evidence.max_abs_error == 0.0
    assert payload["schema"] == "scpn_qc_enzyme_mlir_compiler_ad_breadth_artifact_v1"
    assert payload["case_count"] == 11
    benchmark_payload = cast(dict[str, Any], payload["isolated_benchmark_evidence"])
    assert benchmark_payload["promotion_ready"] is True


def test_enzyme_mlir_breadth_artifact_blocks_failed_case() -> None:
    """Prevent breadth promotion when any required compiler-AD case fails."""
    artifact = _compiler_ad_breadth_artifact(case_overrides={"loop_activity": False})

    assert artifact.promotion_ready is False
    assert artifact.failed_case_ids == ("loop_activity",)
    assert "loop_activity" not in artifact.passed_case_ids
    with pytest.raises(ValueError, match="promotion-ready"):
        artifact.to_breadth_evidence()


def test_enzyme_mlir_breadth_gap_artifact_records_missing_cases() -> None:
    """Materialise explicit hard-gap rows for every unobserved breadth case."""
    observed_case = EnzymeMLIRCompilerADBreadthCaseEvidence(
        case_id="scalar_forward_mode",
        status="success",
        transform_modes=("forward",),
        frontend_language="llvm_ir",
        value_error=0.0,
        gradient_error=0.0,
        runtime_seconds=0.01,
        artifact_refs={"raw_case": "data/differentiable_phase_qnode/scalar_forward.json"},
        failure_class=None,
        setup_instructions=None,
        claim_boundary="bounded observed scalar forward-mode evidence.",
    )

    artifact = build_enzyme_mlir_compiler_ad_breadth_gap_artifact(
        artifact_id="enzyme-mlir-gap-artifact-001",
        observed_cases=(observed_case,),
        isolated_benchmark_evidence=_benchmark_attachment(),
    )
    payload = artifact.to_dict()

    assert artifact.promotion_ready is False
    assert artifact.passed_case_ids == ("scalar_forward_mode",)
    assert len(artifact.failed_case_ids) == 10
    assert "matrix_vjp" in artifact.failed_case_ids
    assert payload["failed_case_ids"] == list(artifact.failed_case_ids)
    hard_gap_rows = [
        row for row in cast(list[dict[str, Any]], payload["cases"]) if row["status"] == "hard_gap"
    ]
    assert len(hard_gap_rows) == 10
    assert hard_gap_rows[0]["failure_class"] == "missing_case_evidence"


def test_enzyme_mlir_breadth_gap_artifact_rejects_duplicate_observations() -> None:
    """Reject duplicate raw observations before building a breadth gap artefact."""
    observed_case = EnzymeMLIRCompilerADBreadthCaseEvidence(
        case_id="scalar_forward_mode",
        status="success",
        transform_modes=("forward",),
        frontend_language="llvm_ir",
        value_error=0.0,
        gradient_error=0.0,
        runtime_seconds=0.01,
        artifact_refs={"raw_case": "case.json"},
        failure_class=None,
        setup_instructions=None,
        claim_boundary="bounded observed scalar forward-mode evidence.",
    )

    with pytest.raises(ValueError, match="duplicate case identifiers"):
        build_enzyme_mlir_compiler_ad_breadth_gap_artifact(
            artifact_id="enzyme-mlir-gap-artifact-duplicate",
            observed_cases=(observed_case, observed_case),
            isolated_benchmark_evidence=_benchmark_attachment(),
        )


def test_enzyme_mlir_breadth_artifact_writer_preserves_gap_boundary(tmp_path: Path) -> None:
    """The raw breadth writer must persist complete case-level hard-gap evidence."""
    observed_case = EnzymeMLIRCompilerADBreadthCaseEvidence(
        case_id="scalar_reverse_mode",
        status="success",
        transform_modes=("reverse",),
        frontend_language="llvm_ir",
        value_error=0.0,
        gradient_error=0.0,
        runtime_seconds=0.01,
        artifact_refs={
            "raw_case": "data/differentiable_phase_qnode/enzyme_toolchain_ad_execution_evidence_20260622.json"
        },
        failure_class=None,
        setup_instructions=None,
        claim_boundary="bounded observed scalar reverse-mode evidence.",
    )
    artifact = build_enzyme_mlir_compiler_ad_breadth_gap_artifact(
        artifact_id="enzyme-mlir-breadth-artifact-writer",
        observed_cases=(observed_case,),
        isolated_benchmark_evidence=_benchmark_attachment(promotion_ready=False),
    )

    markdown = render_enzyme_mlir_compiler_ad_breadth_artifact_markdown(artifact)
    files = write_enzyme_mlir_compiler_ad_breadth_artifact(tmp_path, artifact)

    assert files.artifact_id == artifact.artifact_id
    assert files.json_path.name == "enzyme_mlir_compiler_ad_breadth_artifact_writer.json"
    assert files.markdown_path.name == "enzyme_mlir_compiler_ad_breadth_artifact_writer.md"
    assert json.loads(files.json_path.read_text(encoding="utf-8")) == artifact.to_dict()
    assert "promotion_ready: `False`" in markdown
    assert "Failed cases" in files.markdown_path.read_text(encoding="utf-8")


def test_enzyme_mlir_maturity_audit_derives_breadth_from_artifact() -> None:
    """Derive breadth promotion evidence from a linked raw artefact when omitted."""
    evidence = EnzymeNativeExecutionEvidence(
        artifact_id="enzyme-success-001",
        status="success",
        failure_class=None,
        value_error=0.0,
        gradient_error=0.0,
        runtime_seconds=0.01,
        toolchain={"enzyme": "1.0", "llvm": "18.1.3"},
        setup_instructions=None,
        claim_boundary="Bounded native Enzyme execution evidence.",
    )

    result = run_enzyme_mlir_maturity_audit(
        toolchain_probe=lambda command: f"/opt/toolchain/bin/{command}",
        version_probe=lambda executable: f"{executable} version 1.2.3",
        isolated_benchmark_artifact_id="iso-bench-001",
        isolated_benchmark_evidence=_benchmark_attachment(),
        native_enzyme_execution_evidence=evidence,
        mlir_llvm_correctness_artifact_id="mlir-correctness-001",
        compiler_ad_breadth_artifact=_compiler_ad_breadth_artifact(),
    )

    assert result.ready_for_provider_exceedance is True
    assert result.compiler_ad_breadth_evidence is not None
    assert result.compiler_ad_breadth_artifact is not None
    result_payload = cast(dict[str, Any], result.to_dict())
    artifact_payload = cast(dict[str, Any], result_payload["compiler_ad_breadth_artifact"])
    assert artifact_payload["promotion_ready"] is True


def test_enzyme_mlir_maturity_audit_blocks_nonpromotional_breadth_artifact() -> None:
    """Report failed case identifiers from a non-promotional breadth artefact."""
    result = run_enzyme_mlir_maturity_audit(
        toolchain_probe=lambda command: f"/opt/toolchain/bin/{command}",
        version_probe=lambda executable: f"{executable} version 1.2.3",
        isolated_benchmark_artifact_id="iso-bench-001",
        isolated_benchmark_evidence=_benchmark_attachment(),
        native_enzyme_execution_artifact_id="enzyme-success-001",
        mlir_llvm_correctness_artifact_id="mlir-correctness-001",
        compiler_ad_breadth_artifact=_compiler_ad_breadth_artifact(
            case_overrides={"loop_activity": False}
        ),
    )

    assert result.ready_for_provider_exceedance is False
    assert "compiler AD breadth artifact not promotion-ready" in result.hard_gaps
    assert "compiler AD breadth case hard gaps: loop_activity" in result.hard_gaps


def test_enzyme_mlir_maturity_audit_separates_benchmark_and_case_gaps() -> None:
    """Do not invent failed breadth cases for an unready benchmark attachment."""
    attachment = _benchmark_attachment(promotion_ready=False)
    artifact = _compiler_ad_breadth_artifact(isolated_benchmark_evidence=attachment)

    result = run_enzyme_mlir_maturity_audit(
        toolchain_probe=lambda command: f"/opt/toolchain/bin/{command}",
        version_probe=lambda executable: f"{executable} version 1.2.3",
        isolated_benchmark_artifact_id="iso-bench-001",
        isolated_benchmark_evidence=attachment,
        native_enzyme_execution_artifact_id="enzyme-success-001",
        mlir_llvm_correctness_artifact_id="mlir-correctness-001",
        compiler_ad_breadth_artifact=artifact,
    )

    assert artifact.failed_case_ids == ()
    assert "validated isolated benchmark evidence missing" in result.hard_gaps
    assert "compiler AD breadth artifact not promotion-ready" in result.hard_gaps
    assert not any("compiler AD breadth case hard gaps" in gap for gap in result.hard_gaps)


def test_enzyme_mlir_maturity_audit_reports_gap_artifact_case_ids() -> None:
    """Expose every missing case identifier carried by a raw gap artefact."""
    observed_case = EnzymeMLIRCompilerADBreadthCaseEvidence(
        case_id="scalar_reverse_mode",
        status="success",
        transform_modes=("reverse",),
        frontend_language="llvm_ir",
        value_error=0.0,
        gradient_error=0.0,
        runtime_seconds=0.01,
        artifact_refs={"raw_case": "data/differentiable_phase_qnode/scalar_reverse.json"},
        failure_class=None,
        setup_instructions=None,
        claim_boundary="bounded observed scalar reverse-mode evidence.",
    )
    artifact = build_enzyme_mlir_compiler_ad_breadth_gap_artifact(
        artifact_id="enzyme-mlir-gap-artifact-audit",
        observed_cases=(observed_case,),
        isolated_benchmark_evidence=_benchmark_attachment(),
    )

    result = run_enzyme_mlir_maturity_audit(
        toolchain_probe=lambda command: f"/opt/toolchain/bin/{command}",
        version_probe=lambda executable: f"{executable} version 1.2.3",
        isolated_benchmark_artifact_id="iso-bench-001",
        isolated_benchmark_evidence=_benchmark_attachment(),
        native_enzyme_execution_artifact_id="enzyme-success-001",
        mlir_llvm_correctness_artifact_id="mlir-correctness-001",
        compiler_ad_breadth_artifact=artifact,
    )

    expected_failed = ", ".join(artifact.failed_case_ids)
    assert result.ready_for_provider_exceedance is False
    assert f"compiler AD breadth case hard gaps: {expected_failed}" in result.hard_gaps


def test_enzyme_mlir_maturity_audit_rejects_benchmark_attachment_mismatch() -> None:
    """Reject mismatched isolated benchmark identifiers across audit attachments."""
    with pytest.raises(ValueError, match="isolated_benchmark_evidence.benchmark_artifact_id"):
        run_enzyme_mlir_maturity_audit(
            toolchain_probe=lambda command: f"/opt/toolchain/bin/{command}",
            version_probe=lambda executable: f"{executable} version 1.2.3",
            isolated_benchmark_artifact_id="iso-bench-001",
            isolated_benchmark_evidence=_benchmark_attachment(
                benchmark_artifact_id="iso-bench-002"
            ),
        )


def test_enzyme_mlir_benchmark_attachment_rejects_incomplete_case_set() -> None:
    """Require benchmark attachments to name the complete breadth case set."""
    validation = PhaseQNodeAffinityArtifactValidation(
        artifact_path="data/benchmarks/enzyme_mlir_iso.json",
        artifact_sha256="b" * 64,
        benchmark_artifact_id="iso-bench-001",
        evidence_label="isolated_affinity",
        production_benchmark=True,
        promotion_ready=True,
        raw_timing_row_count=5,
        missing_requirements=(),
        claim_boundary="isolated benchmark validation for Enzyme/MLIR compiler AD.",
    )

    with pytest.raises(ValueError, match="required_breadth_cases"):
        build_enzyme_mlir_benchmark_attachment(
            validation=validation,
            required_breadth_cases=("scalar_forward_mode",),
            claim_boundary="Enzyme/MLIR compiler-AD benchmark attachment.",
        )


def test_enzyme_mlir_breadth_case_rejects_malformed_success() -> None:
    """Reject successful breadth rows that omit finite execution evidence."""
    with pytest.raises(ValueError, match="success case rows require finite"):
        EnzymeMLIRCompilerADBreadthCaseEvidence(
            case_id="scalar_forward_mode",
            status="success",
            transform_modes=("forward",),
            frontend_language="mlir",
            value_error=0.0,
            gradient_error=None,
            runtime_seconds=0.01,
            artifact_refs={"raw": "case.json"},
            failure_class=None,
            setup_instructions=None,
            claim_boundary="bounded case",
        )


def test_enzyme_mlir_compiler_ad_breadth_rejects_missing_case() -> None:
    """Reject derived breadth evidence with a non-passing required case."""
    with pytest.raises(ValueError, match="compiler AD breadth cases"):
        _compiler_ad_breadth_evidence(case_overrides={"matrix_vjp": False})


def test_enzyme_mlir_maturity_audit_rejects_breadth_benchmark_mismatch() -> None:
    """Reject breadth evidence linked to a different isolated benchmark."""
    evidence = EnzymeNativeExecutionEvidence(
        artifact_id="enzyme-success-001",
        status="success",
        failure_class=None,
        value_error=0.0,
        gradient_error=0.0,
        runtime_seconds=0.01,
        toolchain={"enzyme": "1.0", "llvm": "18.1.3"},
        setup_instructions=None,
        claim_boundary="Bounded native Enzyme execution evidence.",
    )

    with pytest.raises(ValueError, match="compiler_ad_breadth_evidence.isolated_benchmark"):
        run_enzyme_mlir_maturity_audit(
            toolchain_probe=lambda command: f"/opt/toolchain/bin/{command}",
            version_probe=lambda executable: f"{executable} version 1.2.3",
            isolated_benchmark_artifact_id="iso-bench-001",
            native_enzyme_execution_evidence=evidence,
            mlir_llvm_correctness_artifact_id="mlir-correctness-001",
            compiler_ad_breadth_evidence=_compiler_ad_breadth_evidence(
                isolated_benchmark_artifact_id="other-bench"
            ),
        )


def test_committed_enzyme_mlir_audit_records_installed_native_probe() -> None:
    """Validate the committed audit's installed native probe classification."""
    payload = json.loads(ENZYME_MLIR_AUDIT_PATH.read_text(encoding="utf-8"))

    assert payload["classification"] == "hard_gap"
    assert payload["ready_for_provider_exceedance"] is False
    assert payload["hard_gaps"] == [
        "validated isolated benchmark evidence missing",
        ("compiler AD breadth artifact not promotion-ready"),
        (
            "compiler AD breadth case hard gaps: alias_activity, matrix_jvp, "
            "mlir_lowering, scalar_forward_mode, vector_jvp"
        ),
        "compiler AD breadth evidence missing",
    ]
    assert payload["compiler_ad_breadth_artifact"]["artifact_id"] == (
        "enzyme-mlir-compiler-ad-breadth-artifact-20260706"
    )
    assert payload["compiler_ad_breadth_artifact"]["case_count"] == 11
    assert payload["compiler_ad_breadth_artifact"]["promotion_ready"] is False
    assert payload["compiler_ad_breadth_evidence"] is None
    assert payload["isolated_benchmark_evidence"]["promotion_ready"] is False
    assert all(status["available"] for status in payload["toolchain"].values())
    assert payload["native_enzyme_execution_evidence"]["status"] == "success"
    assert payload["native_enzyme_execution_evidence"]["value_error"] == 0.0
    assert payload["native_enzyme_execution_evidence"]["gradient_error"] == 0.0
    assert "arbitrary-program AD" in payload["native_enzyme_execution_evidence"]["claim_boundary"]


def test_committed_enzyme_mlir_breadth_artifact_records_case_level_gaps() -> None:
    """Committed raw breadth artifact must expose every Enzyme/MLIR case outcome."""
    payload = json.loads(ENZYME_MLIR_BREADTH_ARTIFACT_PATH.read_text(encoding="utf-8"))
    cases = {row["case_id"]: row for row in payload["cases"]}

    assert payload["schema"] == "scpn_qc_enzyme_mlir_compiler_ad_breadth_artifact_v1"
    assert payload["artifact_id"] == "enzyme-mlir-compiler-ad-breadth-artifact-20260706"
    assert payload["case_count"] == 11
    assert payload["promotion_ready"] is False
    assert payload["passed_case_ids"] == [
        "llvm_ir_generation",
        "loop_activity",
        "matrix_vjp",
        "native_enzyme_execution",
        "scalar_reverse_mode",
        "vector_vjp",
    ]
    assert payload["failed_case_ids"] == [
        "alias_activity",
        "matrix_jvp",
        "mlir_lowering",
        "scalar_forward_mode",
        "vector_jvp",
    ]
    assert cases["scalar_reverse_mode"]["gradient_error"] == 0.0
    assert cases["native_enzyme_execution"]["artifact_refs"] == {
        "native_enzyme_execution": (
            "data/differentiable_phase_qnode/enzyme_toolchain_ad_execution_evidence_20260622.json"
        )
    }
    assert cases["alias_activity"]["failure_class"] == "program_ad_alias_not_enzyme_mlir_raw_case"
    assert payload["isolated_benchmark_evidence"]["evidence_label"] == "functional_non_isolated"


def test_enzyme_mlir_evidence_validation_paths() -> None:
    """Exercise fail-closed validation for native execution evidence records."""
    with pytest.raises(ValueError, match="hard-gap Enzyme evidence"):
        EnzymeNativeExecutionEvidence(
            artifact_id="gap",
            status="hard_gap",
            failure_class=None,
            value_error=None,
            gradient_error=None,
            runtime_seconds=None,
            toolchain={"enzyme": "1.0"},
            setup_instructions=None,
            claim_boundary="gap",
        )
    with pytest.raises(ValueError, match="successful Enzyme evidence requires finite"):
        EnzymeNativeExecutionEvidence(
            artifact_id="success",
            status="success",
            failure_class=None,
            value_error=0.0,
            gradient_error=None,
            runtime_seconds=0.01,
            toolchain={"enzyme": "1.0"},
            setup_instructions=None,
            claim_boundary="success",
        )
    with pytest.raises(ValueError, match="checks"):
        MLIRLLVMCorrectnessEvidence(
            artifact_id="mlir",
            checks={},
            toolchain_versions={"clang": "18.1.3"},
            claim_boundary="bounded correctness",
        )


def test_enzyme_mlir_maturity_audit_exports_are_public() -> None:
    """Keep maturity-audit records and builders identical across public facades."""
    assert compiler.run_enzyme_mlir_maturity_audit is run_enzyme_mlir_maturity_audit
    assert scpn.run_enzyme_mlir_maturity_audit is run_enzyme_mlir_maturity_audit
    assert compiler.EnzymeMLIRMaturityAuditResult is EnzymeMLIRMaturityAuditResult
    assert scpn.EnzymeMLIRMaturityAuditResult is EnzymeMLIRMaturityAuditResult
    assert compiler.EnzymeNativeExecutionEvidence is EnzymeNativeExecutionEvidence
    assert scpn.EnzymeNativeExecutionEvidence is EnzymeNativeExecutionEvidence
    assert compiler.MLIRLLVMCorrectnessEvidence is MLIRLLVMCorrectnessEvidence
    assert scpn.MLIRLLVMCorrectnessEvidence is MLIRLLVMCorrectnessEvidence
    assert compiler.EnzymeMLIRBenchmarkAttachment is EnzymeMLIRBenchmarkAttachment
    assert scpn.EnzymeMLIRBenchmarkAttachment is EnzymeMLIRBenchmarkAttachment
    assert compiler.EnzymeMLIRCompilerADBreadthArtifact is EnzymeMLIRCompilerADBreadthArtifact
    assert scpn.EnzymeMLIRCompilerADBreadthArtifact is EnzymeMLIRCompilerADBreadthArtifact
    assert (
        compiler.EnzymeMLIRCompilerADBreadthCaseEvidence is EnzymeMLIRCompilerADBreadthCaseEvidence
    )
    assert scpn.EnzymeMLIRCompilerADBreadthCaseEvidence is EnzymeMLIRCompilerADBreadthCaseEvidence
    assert (
        compiler.build_enzyme_mlir_benchmark_attachment is build_enzyme_mlir_benchmark_attachment
    )
    assert scpn.build_enzyme_mlir_benchmark_attachment is build_enzyme_mlir_benchmark_attachment
    assert (
        compiler.build_enzyme_mlir_compiler_ad_breadth_artifact
        is build_enzyme_mlir_compiler_ad_breadth_artifact
    )
    assert (
        scpn.build_enzyme_mlir_compiler_ad_breadth_artifact
        is build_enzyme_mlir_compiler_ad_breadth_artifact
    )
    assert (
        compiler.build_enzyme_mlir_compiler_ad_breadth_gap_artifact
        is build_enzyme_mlir_compiler_ad_breadth_gap_artifact
    )
    assert (
        scpn.build_enzyme_mlir_compiler_ad_breadth_gap_artifact
        is build_enzyme_mlir_compiler_ad_breadth_gap_artifact
    )
