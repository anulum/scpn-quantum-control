# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- real Enzyme/LLVM toolchain AD execution evidence tests
"""Tests for the real Enzyme/LLVM reverse-mode AD execution evidence surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.compiler import (
    EnzymeToolchainADCase,
    EnzymeToolchainADExecutionEvidence,
    run_enzyme_toolchain_execution_evidence,
)
from scpn_quantum_control.compiler.mlir_enzyme_execution_runner import resolve_enzyme_toolchain

_EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "differentiable_phase_qnode"


def _rebuild(payload: dict[str, object]) -> EnzymeToolchainADExecutionEvidence:
    cases = tuple(
        EnzymeToolchainADCase(
            case_id=row["case_id"],
            operation_family=row["operation_family"],
            operand_dimension=row["operand_dimension"],
            status=row["status"],
            gradient_error=row["gradient_error"],
            runtime_seconds=row["runtime_seconds"],
            failure_class=row["failure_class"],
            claim_boundary=row["claim_boundary"],
        )
        for row in payload["cases"]
    )
    return EnzymeToolchainADExecutionEvidence(
        artifact_id=payload["artifact_id"],
        toolchain_available=payload["toolchain_available"],
        toolchain=payload["toolchain"],
        cases=cases,
        beyond_scalar_executed=payload["beyond_scalar_executed"],
        executed_operation_families=tuple(payload["executed_operation_families"]),
        max_gradient_error=payload["max_gradient_error"],
        gradient_parity_tolerance=payload["gradient_parity_tolerance"],
        claim_boundary=payload["claim_boundary"],
    )


def test_runner_is_gated_or_executes_beyond_scalar() -> None:
    """The runner either executes the battery beyond scalar or fails closed as gated."""

    evidence = run_enzyme_toolchain_execution_evidence()
    if evidence.toolchain_available:
        assert evidence.beyond_scalar_executed is True
        assert set(evidence.executed_operation_families) >= {"scalar", "vector", "matrix"}
        assert evidence.max_gradient_error <= evidence.gradient_parity_tolerance
        for case in evidence.cases:
            if case.status == "executed":
                assert case.gradient_error is not None
                assert case.gradient_error <= evidence.gradient_parity_tolerance
    else:
        assert evidence.beyond_scalar_executed is False
        assert evidence.executed_operation_families == ()
        assert evidence.max_gradient_error == 0.0
        assert all(case.status == "hard_gap" for case in evidence.cases)
        assert all(case.failure_class for case in evidence.cases)


def test_runner_matches_toolchain_detection() -> None:
    """The evidence toolchain flag agrees with the toolchain resolver."""

    evidence = run_enzyme_toolchain_execution_evidence()
    assert evidence.toolchain_available is (resolve_enzyme_toolchain() is not None)


def test_runner_evidence_round_trips_through_json() -> None:
    """The captured evidence serialises and rebuilds into an equivalent record."""

    evidence = run_enzyme_toolchain_execution_evidence()
    payload = json.loads(json.dumps(evidence.to_dict(), sort_keys=True))
    rebuilt = _rebuild(payload)
    assert rebuilt.toolchain_available == evidence.toolchain_available
    assert rebuilt.beyond_scalar_executed == evidence.beyond_scalar_executed
    assert len(rebuilt.cases) == len(evidence.cases)


def test_executed_case_requires_finite_metrics() -> None:
    """An executed case must carry finite non-negative metrics and no failure_class."""

    with pytest.raises(ValueError, match="finite non-negative"):
        EnzymeToolchainADCase(
            case_id="x",
            operation_family="vector",
            operand_dimension=4,
            status="executed",
            gradient_error=None,
            runtime_seconds=1e-3,
            failure_class=None,
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="must not carry a failure_class"):
        EnzymeToolchainADCase(
            case_id="x",
            operation_family="vector",
            operand_dimension=4,
            status="executed",
            gradient_error=0.0,
            runtime_seconds=1e-3,
            failure_class="should-not-be-here",
            claim_boundary="bounded",
        )


def test_hard_gap_case_requires_reason_and_no_metrics() -> None:
    """A hard-gap case must carry a reason and no execution metrics."""

    with pytest.raises(ValueError, match="failure_class"):
        EnzymeToolchainADCase(
            case_id="x",
            operation_family="vector",
            operand_dimension=4,
            status="hard_gap",
            gradient_error=None,
            runtime_seconds=None,
            failure_class=None,
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="must not carry execution metrics"):
        EnzymeToolchainADCase(
            case_id="x",
            operation_family="vector",
            operand_dimension=4,
            status="hard_gap",
            gradient_error=0.0,
            runtime_seconds=None,
            failure_class="declined",
            claim_boundary="bounded",
        )


def test_unavailable_toolchain_cannot_record_executed_cases() -> None:
    """An evidence marked toolchain-unavailable cannot hold executed rows."""

    executed = EnzymeToolchainADCase(
        case_id="vec",
        operation_family="vector",
        operand_dimension=4,
        status="executed",
        gradient_error=0.0,
        runtime_seconds=1e-3,
        failure_class=None,
        claim_boundary="bounded",
    )
    with pytest.raises(ValueError, match="unavailable toolchain cannot record executed"):
        EnzymeToolchainADExecutionEvidence(
            artifact_id="probe",
            toolchain_available=False,
            toolchain={"status": "unavailable"},
            cases=(executed,),
            beyond_scalar_executed=True,
            executed_operation_families=("vector",),
            max_gradient_error=0.0,
            gradient_parity_tolerance=1e-9,
            claim_boundary="bounded",
        )


def test_aggregate_rejects_gradient_error_over_tolerance() -> None:
    """Evidence cannot be built when an executed gradient error exceeds tolerance."""

    over = EnzymeToolchainADCase(
        case_id="vec",
        operation_family="vector",
        operand_dimension=4,
        status="executed",
        gradient_error=1e-3,
        runtime_seconds=1e-3,
        failure_class=None,
        claim_boundary="bounded",
    )
    with pytest.raises(ValueError, match="exceeds the declared parity tolerance"):
        EnzymeToolchainADExecutionEvidence(
            artifact_id="probe",
            toolchain_available=True,
            toolchain={"clang": "x"},
            cases=(over,),
            beyond_scalar_executed=True,
            executed_operation_families=("vector",),
            max_gradient_error=1e-3,
            gradient_parity_tolerance=1e-9,
            claim_boundary="bounded",
        )


def test_committed_evidence_artifacts_are_valid() -> None:
    """Every committed Enzyme-execution artefact reloads into valid evidence."""

    artifacts = sorted(_EVIDENCE_DIR.glob("enzyme_toolchain_ad_execution_evidence_*.json"))
    assert artifacts, "expected at least one committed Enzyme-execution evidence artefact"
    for path in artifacts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rebuilt = _rebuild(payload)
        assert isinstance(rebuilt, EnzymeToolchainADExecutionEvidence)
        if rebuilt.toolchain_available:
            assert rebuilt.beyond_scalar_executed is True
            assert rebuilt.max_gradient_error <= rebuilt.gradient_parity_tolerance
