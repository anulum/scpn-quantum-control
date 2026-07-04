# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- LLVM/JIT claim gate tests
"""Tests for the LLVM/JIT promotion claim gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.compiler import (
    LLVMJITClaimGate,
    NativeWholeProgramADExecutionCase,
    NativeWholeProgramADExecutionEvidence,
    build_llvm_jit_claim_gate,
    build_native_whole_program_ad_execution_evidence,
    llvm_jit_claim_gate_from_dict,
    render_llvm_jit_claim_gate_markdown,
)

_EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "differentiable_phase_qnode"


def _case(
    case_id: str,
    operation_family: str,
    operand_dimension: int,
) -> NativeWholeProgramADExecutionCase:
    """Build a verified native LLVM/JIT execution row for gate tests."""

    return NativeWholeProgramADExecutionCase(
        case_id=case_id,
        operation_family=operation_family,
        operand_dimension=operand_dimension,
        status="executed",
        value_error=0.0,
        gradient_error=0.0,
        runtime_seconds=1e-3,
        native_symbol=f"whole_program_ad_{case_id}",
        failure_class=None,
        claim_boundary="bounded native test evidence",
    )


def _native_evidence(*, beyond_scalar: bool = True) -> NativeWholeProgramADExecutionEvidence:
    """Build minimal validated native execution evidence."""

    cases = [_case("scalar_poly_3", "scalar", 3)]
    if beyond_scalar:
        cases.append(_case("determinant_2x2", "determinant", 2))
    return build_native_whole_program_ad_execution_evidence(
        artifact_id="native-whole-program-ad-execution-test",
        cases=cases,
        gradient_parity_tolerance=1e-9,
        fail_closed_boundaries={"determinant": 20},
        claim_boundary="bounded native test evidence",
    )


def test_native_execution_alone_keeps_jit_promotion_blocked() -> None:
    """Executable lowering plus correctness still blocks without all promotion artefacts."""

    gate = build_llvm_jit_claim_gate(
        artifact_id="probe",
        native_execution_evidence=_native_evidence(),
        correctness_test_ids=(
            "tests/test_native_whole_program_ad_execution_evidence.py::"
            "test_runner_executes_beyond_scalar_with_reference_parity",
        ),
    )

    assert gate.executable_lowering_verified is True
    assert gate.executable_lowering_evidence_id == "native-whole-program-ad-execution-test"
    assert gate.promotion_ready is False
    assert gate.missing_requirements == (
        "crash_safety_tests",
        "benchmark_artifact_ids",
        "rollback_policy",
        "fallback_policy",
    )
    assert "no LLVM/JIT promotion" in gate.claim_boundary


def test_gate_promotes_only_when_every_required_evidence_class_is_attached() -> None:
    """The promotion flag is derived from all required evidence classes."""

    gate = build_llvm_jit_claim_gate(
        artifact_id="probe",
        native_execution_evidence=_native_evidence(),
        correctness_test_ids=("tests/native_correctness.py::test_parity",),
        crash_safety_test_ids=("tests/native_crash_safety.py::test_fail_closed",),
        benchmark_artifact_ids=("diff-isolated-native-jit-20260704",),
        rollback_policy="Disable native LLVM/JIT promotion and route through Program AD.",
        fallback_policy="Use interpreted Program AD when the native lowering gate is not ready.",
    )

    assert gate.promotion_ready is True
    assert gate.missing_requirements == ()
    assert gate.to_dict()["promotion_ready"] is True
    assert "diff-isolated-native-jit-20260704" in render_llvm_jit_claim_gate_markdown(gate)


def test_scalar_only_native_evidence_does_not_satisfy_executable_lowering() -> None:
    """Scalar-only evidence cannot unlock a wider LLVM/JIT claim."""

    gate = build_llvm_jit_claim_gate(
        artifact_id="probe",
        native_execution_evidence=_native_evidence(beyond_scalar=False),
        correctness_test_ids=("tests/native_correctness.py::test_parity",),
        crash_safety_test_ids=("tests/native_crash_safety.py::test_fail_closed",),
        benchmark_artifact_ids=("diff-isolated-native-jit-20260704",),
        rollback_policy="Disable native LLVM/JIT promotion and route through Program AD.",
        fallback_policy="Use interpreted Program AD when the native lowering gate is not ready.",
    )

    assert gate.executable_lowering_verified is False
    assert gate.missing_requirements == ("executable_lowering",)
    assert gate.promotion_ready is False


def test_gate_rejects_blank_ids_and_unbacked_executable_success() -> None:
    """The frozen record fails closed on blank evidence and unsupported success claims."""

    with pytest.raises(ValueError, match="executable_lowering_evidence_id"):
        LLVMJITClaimGate(
            artifact_id="probe",
            executable_lowering_evidence_id=None,
            executable_lowering_verified=True,
            correctness_test_ids=("tests/native_correctness.py::test_parity",),
            crash_safety_test_ids=("tests/native_crash_safety.py::test_fail_closed",),
            benchmark_artifact_ids=("diff-isolated-native-jit-20260704",),
            rollback_policy="Disable native LLVM/JIT promotion.",
            fallback_policy="Use interpreted Program AD.",
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="correctness_test_ids"):
        build_llvm_jit_claim_gate(
            artifact_id="probe",
            native_execution_evidence=_native_evidence(),
            correctness_test_ids=("",),
        )
    with pytest.raises(ValueError, match="rollback_policy"):
        build_llvm_jit_claim_gate(
            artifact_id="probe",
            native_execution_evidence=_native_evidence(),
            correctness_test_ids=("tests/native_correctness.py::test_parity",),
            rollback_policy=" ",
        )


def test_committed_llvm_jit_claim_gate_artifact_is_blocked_and_consistent() -> None:
    """Committed claim-gate JSON must match the derived blockers."""

    path = _EVIDENCE_DIR / "llvm_jit_claim_gate_20260704.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = llvm_jit_claim_gate_from_dict(payload)

    assert gate.artifact_id == "llvm-jit-claim-gate-20260704"
    assert gate.promotion_ready is False
    assert gate.missing_requirements == (
        "crash_safety_tests",
        "benchmark_artifact_ids",
        "rollback_policy",
        "fallback_policy",
    )
    assert payload["missing_requirements"] == list(gate.missing_requirements)
    assert payload["promotion_ready"] is gate.promotion_ready
