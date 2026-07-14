# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR enzyme audit module
# scpn-quantum-control -- Enzyme MLIR maturity audit
"""Enzyme/MLIR toolchain probing and bounded maturity aggregation."""

from __future__ import annotations

import os
import shutil

# Subprocess is used only for admitted local toolchain version probes.
import subprocess  # nosec B404
from collections.abc import Callable, Sequence
from os.path import realpath
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .mlir_enzyme_evidence import (
    EnzymeMLIRBenchmarkAttachment,
    EnzymeMLIRCompilerADBreadthArtifact,
    EnzymeMLIRCompilerADBreadthEvidence,
    EnzymeMLIRMaturityAuditResult,
    EnzymeMLIRToolchainStatus,
    EnzymeNativeExecutionEvidence,
    MLIRLLVMCorrectnessEvidence,
)
from .mlir_phase_qnode_runtime import compile_phase_qnode_circuit_to_mlir_runtime
from .mlir_whole_program_native import native_whole_program_ad_linalg_support

FloatArray: TypeAlias = NDArray[np.float64]


def run_enzyme_mlir_maturity_audit(
    circuit: Any | None = None,
    parameters: Sequence[float] | FloatArray | None = None,
    *,
    toolchain_probe: Callable[[str], str | None] | None = None,
    version_probe: Callable[[str], str | None] | None = None,
    isolated_benchmark_artifact_id: str | None = None,
    isolated_benchmark_evidence: EnzymeMLIRBenchmarkAttachment | None = None,
    native_enzyme_execution_artifact_id: str | None = None,
    native_enzyme_execution_evidence: EnzymeNativeExecutionEvidence | None = None,
    mlir_llvm_correctness_artifact_id: str | None = None,
    compiler_ad_breadth_evidence: EnzymeMLIRCompilerADBreadthEvidence | None = None,
    compiler_ad_breadth_artifact: EnzymeMLIRCompilerADBreadthArtifact | None = None,
) -> EnzymeMLIRMaturityAuditResult:
    """Audit Enzyme/MLIR maturity without promoting unsupported compiler-AD claims."""
    executable = compile_phase_qnode_circuit_to_mlir_runtime(
        _default_enzyme_mlir_audit_circuit() if circuit is None else circuit,
        np.array([0.2, -0.3], dtype=np.float64) if parameters is None else parameters,
    )
    verification = dict(executable.verification)
    correctness_checks = {
        "phase_qnode_value_close": verification.get("value_close") is True,
        "phase_qnode_gradient_close": verification.get("gradient_close") is True,
        "mlir_runtime_backend_verified": executable.runtime_backend == "scpn_mlir_runtime_adapter",
        "native_llvm_jit_support_matrix_declared": bool(native_whole_program_ad_linalg_support()),
    }
    toolchain = {
        command: _enzyme_mlir_toolchain_status(command, toolchain_probe, version_probe)
        for command in ("enzyme", "opt", "mlir-opt", "clang")
    }
    hard_gaps: list[str] = []
    for command, status in toolchain.items():
        if not status.available:
            hard_gaps.append(f"{command} toolchain unavailable")
    if isolated_benchmark_artifact_id is None:
        hard_gaps.append("isolated benchmark artefact missing")
    elif isolated_benchmark_evidence is None or not isolated_benchmark_evidence.promotion_ready:
        hard_gaps.append("validated isolated benchmark evidence missing")
    if native_enzyme_execution_evidence is not None:
        native_enzyme_execution_artifact_id = native_enzyme_execution_evidence.artifact_id
        if not native_enzyme_execution_evidence.passed:
            failure = native_enzyme_execution_evidence.failure_class or "unknown"
            hard_gaps.append(f"native Enzyme execution hard gap: {failure}")
    if native_enzyme_execution_artifact_id is None:
        hard_gaps.append("native Enzyme execution artefact missing")
    toolchain_versions = {
        command: status.version
        for command, status in toolchain.items()
        if status.available and status.version is not None
    }
    mlir_llvm_correctness_evidence = (
        MLIRLLVMCorrectnessEvidence(
            artifact_id=mlir_llvm_correctness_artifact_id,
            checks=correctness_checks,
            toolchain_versions=toolchain_versions,
            claim_boundary=(
                "Bounded SCPN MLIR-runtime and native LLVM/JIT support snapshot; "
                "not native Enzyme execution, provider, hardware, or performance evidence."
            ),
        )
        if mlir_llvm_correctness_artifact_id is not None
        else None
    )
    if mlir_llvm_correctness_evidence is None:
        hard_gaps.append("MLIR/LLVM correctness artefact missing")
    if compiler_ad_breadth_artifact is None:
        hard_gaps.append("compiler AD breadth artifact missing")
    elif compiler_ad_breadth_evidence is None:
        if compiler_ad_breadth_artifact.promotion_ready:
            compiler_ad_breadth_evidence = compiler_ad_breadth_artifact.to_breadth_evidence()
        else:
            hard_gaps.append("compiler AD breadth artifact not promotion-ready")
            if compiler_ad_breadth_artifact.failed_case_ids:
                hard_gaps.append(
                    "compiler AD breadth case hard gaps: "
                    + ", ".join(compiler_ad_breadth_artifact.failed_case_ids)
                )
    if compiler_ad_breadth_evidence is None:
        hard_gaps.append("compiler AD breadth evidence missing")
    return EnzymeMLIRMaturityAuditResult(
        scpn_mlir_runtime_verified=bool(
            correctness_checks["phase_qnode_value_close"]
            and correctness_checks["phase_qnode_gradient_close"]
            and correctness_checks["mlir_runtime_backend_verified"]
        ),
        native_llvm_jit_surface="available: bounded in-process native LLVM/JIT",
        toolchain=toolchain,
        correctness_checks=correctness_checks,
        hard_gaps=tuple(dict.fromkeys(hard_gaps)),
        isolated_benchmark_artifact_id=isolated_benchmark_artifact_id,
        isolated_benchmark_evidence=isolated_benchmark_evidence,
        native_enzyme_execution_artifact_id=native_enzyme_execution_artifact_id,
        native_enzyme_execution_evidence=native_enzyme_execution_evidence,
        mlir_llvm_correctness_evidence=mlir_llvm_correctness_evidence,
        compiler_ad_breadth_evidence=compiler_ad_breadth_evidence,
        compiler_ad_breadth_artifact=compiler_ad_breadth_artifact,
    )


def _enzyme_mlir_toolchain_status(
    command: str,
    toolchain_probe: Callable[[str], str | None] | None,
    version_probe: Callable[[str], str | None] | None,
) -> EnzymeMLIRToolchainStatus:
    executable = (
        _resolve_toolchain_executable(command)
        if toolchain_probe is None
        else toolchain_probe(command)
    )
    if executable is None:
        return EnzymeMLIRToolchainStatus(
            command=command,
            executable=None,
            available=False,
            version=None,
            failure_class="toolchain_missing",
            setup_instructions=(
                f"Install and expose {command} on PATH before promoting Enzyme/MLIR "
                "compiler-AD maturity evidence."
            ),
        )
    version = (
        _probe_toolchain_version(executable)
        if version_probe is None
        else version_probe(executable)
    )
    if version is None:
        return EnzymeMLIRToolchainStatus(
            command=command,
            executable=None,
            available=False,
            version=None,
            failure_class="version_probe_failed",
            setup_instructions=(
                f"{command} was found at {executable}, but version metadata could not "
                "be captured reproducibly."
            ),
        )
    return EnzymeMLIRToolchainStatus(
        command=command,
        executable=executable,
        available=True,
        version=version,
        failure_class=None,
        setup_instructions=None,
    )


def _resolve_toolchain_executable(command: str) -> str | None:
    """Return an absolute executable path for a PATH-discovered toolchain command."""
    resolved = shutil.which(command)
    if not resolved:
        return None
    return realpath(resolved)


def _probe_toolchain_version(executable: str) -> str | None:
    executable_path = Path(executable)
    if not executable_path.is_absolute():
        return None
    executable_path = Path(realpath(executable_path))
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        return None
    for flag in ("--version", "-version"):
        try:
            completed = subprocess.run(  # nosec B603
                (str(executable_path), flag),
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        output = (completed.stdout or completed.stderr).strip().splitlines()
        if output:
            return output[0][:240]
    return None


def _default_enzyme_mlir_audit_circuit() -> Any:
    from scpn_quantum_control.phase.qnode_circuit import PauliTerm, PhaseQNodeCircuit

    return PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0), ("rx", (0,), 1)),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )
