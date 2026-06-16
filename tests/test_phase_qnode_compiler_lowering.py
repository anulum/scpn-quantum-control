# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNode Compiler Lowering
"""Tests for compiler/mlir.py phase-QNode lowering reports."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control as scpn
import scpn_quantum_control.compiler as compiler
from scpn_quantum_control.compiler.mlir import (
    EnzymeMLIRMaturityAuditResult,
    EnzymeNativeExecutionEvidence,
    MLIRLLVMCorrectnessEvidence,
    PhaseQNodeMLIRRuntimeExecutable,
    compile_phase_qnode_circuit_to_mlir_runtime,
    lower_phase_qnode_circuit_to_mlir,
    run_enzyme_mlir_maturity_audit,
)
from scpn_quantum_control.phase.qnode_circuit import PauliTerm, PhaseQNodeCircuit


def test_phase_qnode_compiler_lowering_reports_registered_subset() -> None:
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
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("u3", (0,), 0),),
        observable="pauli_z",
    )

    with pytest.raises(ValueError, match="phase-QNode lowering failed closed"):
        lower_phase_qnode_circuit_to_mlir(circuit, np.array([0.2], dtype=float))


def test_enzyme_mlir_maturity_audit_records_runtime_evidence_and_toolchain_gaps() -> None:
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


def test_enzyme_mlir_maturity_audit_requires_all_artifacts_for_promotion() -> None:
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

    assert result.hard_gaps == ()
    assert result.ready_for_provider_exceedance is True


def test_enzyme_mlir_evidence_validation_paths() -> None:
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
    assert compiler.run_enzyme_mlir_maturity_audit is run_enzyme_mlir_maturity_audit
    assert scpn.run_enzyme_mlir_maturity_audit is run_enzyme_mlir_maturity_audit
    assert compiler.EnzymeMLIRMaturityAuditResult is EnzymeMLIRMaturityAuditResult
    assert scpn.EnzymeMLIRMaturityAuditResult is EnzymeMLIRMaturityAuditResult
    assert compiler.EnzymeNativeExecutionEvidence is EnzymeNativeExecutionEvidence
    assert scpn.EnzymeNativeExecutionEvidence is EnzymeNativeExecutionEvidence
    assert compiler.MLIRLLVMCorrectnessEvidence is MLIRLLVMCorrectnessEvidence
    assert scpn.MLIRLLVMCorrectnessEvidence is MLIRLLVMCorrectnessEvidence
