# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- native LLVM/JIT whole-program AD execution evidence tests
"""Tests for the native LLVM/JIT whole-program AD execution evidence surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.compiler import (
    NativeWholeProgramADExecutionCase,
    NativeWholeProgramADExecutionEvidence,
    build_native_whole_program_ad_execution_evidence,
    run_native_whole_program_ad_execution_evidence,
)

_EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "differentiable_phase_qnode"


def _executed_case(
    case_id: str = "scalar_poly_3",
    operation_family: str = "scalar",
    operand_dimension: int = 3,
    *,
    value_error: float = 0.0,
    gradient_error: float = 0.0,
) -> NativeWholeProgramADExecutionCase:
    return NativeWholeProgramADExecutionCase(
        case_id=case_id,
        operation_family=operation_family,
        operand_dimension=operand_dimension,
        status="executed",
        value_error=value_error,
        gradient_error=gradient_error,
        runtime_seconds=1e-3,
        native_symbol="whole_program_ad_3_2_whole_program_ad",
        failure_class=None,
        claim_boundary="bounded",
    )


def test_runner_executes_beyond_scalar_with_reference_parity() -> None:
    """The runner compiles, executes and reference-checks beyond-scalar families."""

    evidence = run_native_whole_program_ad_execution_evidence()

    assert evidence.beyond_scalar_executed is True
    assert set(evidence.executed_operation_families) >= {
        "scalar",
        "determinant",
        "inverse",
        "solve",
        "trace",
    }
    assert evidence.max_gradient_error <= evidence.gradient_parity_tolerance
    assert evidence.max_value_error <= 1e-3
    assert evidence.fail_closed_boundaries["determinant"] == 20
    assert evidence.fail_closed_boundaries["inverse"] == 8

    executed = [case for case in evidence.cases if case.status == "executed"]
    fail_closed = [case for case in evidence.cases if case.status == "fail_closed"]
    assert len(executed) >= 10
    assert {case.operation_family for case in fail_closed} == {"determinant", "inverse"}
    for case in executed:
        assert case.native_symbol and case.native_symbol.startswith("whole_program_ad_")
        assert case.gradient_error is not None
        assert case.gradient_error <= evidence.gradient_parity_tolerance
    for case in fail_closed:
        assert case.failure_class
        assert case.value_error is None and case.gradient_error is None


def test_runner_evidence_is_json_serialisable_and_round_trips() -> None:
    """The captured evidence serialises to JSON with all rows preserved."""

    evidence = run_native_whole_program_ad_execution_evidence()
    payload = evidence.to_dict()
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded["beyond_scalar_executed"] is True
    assert len(decoded["cases"]) == len(evidence.cases)
    assert decoded["executed_operation_families"] == list(evidence.executed_operation_families)


def test_builder_derives_aggregates_from_cases() -> None:
    """The builder derives the family list, worst errors and beyond-scalar flag."""

    evidence = build_native_whole_program_ad_execution_evidence(
        artifact_id="probe",
        cases=[
            _executed_case(gradient_error=0.0),
            _executed_case(
                "determinant_2x2",
                "determinant",
                2,
                value_error=2e-13,
                gradient_error=4e-13,
            ),
        ],
        gradient_parity_tolerance=1e-9,
        fail_closed_boundaries={"determinant": 20},
        claim_boundary="bounded",
    )

    assert evidence.beyond_scalar_executed is True
    assert evidence.executed_operation_families == ("scalar", "determinant")
    assert evidence.max_gradient_error == pytest.approx(4e-13)


def test_executed_case_requires_symbol_and_finite_metrics() -> None:
    """An executed case without a native symbol or with non-finite metrics is rejected."""

    with pytest.raises(ValueError, match="native_symbol"):
        NativeWholeProgramADExecutionCase(
            case_id="x",
            operation_family="determinant",
            operand_dimension=2,
            status="executed",
            value_error=0.0,
            gradient_error=0.0,
            runtime_seconds=1e-3,
            native_symbol=None,
            failure_class=None,
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="finite non-negative"):
        NativeWholeProgramADExecutionCase(
            case_id="x",
            operation_family="determinant",
            operand_dimension=2,
            status="executed",
            value_error=float("nan"),
            gradient_error=0.0,
            runtime_seconds=1e-3,
            native_symbol="whole_program_ad_4_2",
            failure_class=None,
            claim_boundary="bounded",
        )


def test_fail_closed_case_requires_failure_class_and_no_metrics() -> None:
    """A fail-closed case must carry a reason and no execution metrics."""

    with pytest.raises(ValueError, match="failure_class"):
        NativeWholeProgramADExecutionCase(
            case_id="x",
            operation_family="determinant",
            operand_dimension=20,
            status="fail_closed",
            value_error=None,
            gradient_error=None,
            runtime_seconds=None,
            native_symbol=None,
            failure_class=None,
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="must not carry execution metrics"):
        NativeWholeProgramADExecutionCase(
            case_id="x",
            operation_family="determinant",
            operand_dimension=20,
            status="fail_closed",
            value_error=0.0,
            gradient_error=None,
            runtime_seconds=None,
            native_symbol=None,
            failure_class="declined",
            claim_boundary="bounded",
        )


def test_aggregate_rejects_gradient_error_over_tolerance() -> None:
    """Evidence cannot be built when an executed gradient error exceeds tolerance."""

    with pytest.raises(ValueError, match="exceeds the declared parity tolerance"):
        build_native_whole_program_ad_execution_evidence(
            artifact_id="probe",
            cases=[_executed_case(gradient_error=1e-3)],
            gradient_parity_tolerance=1e-9,
            fail_closed_boundaries={},
            claim_boundary="bounded",
        )


def test_aggregate_requires_at_least_one_executed_case() -> None:
    """Evidence with only fail-closed rows is rejected."""

    fail_closed = NativeWholeProgramADExecutionCase(
        case_id="det_20",
        operation_family="determinant",
        operand_dimension=20,
        status="fail_closed",
        value_error=None,
        gradient_error=None,
        runtime_seconds=None,
        native_symbol=None,
        failure_class="declined",
        claim_boundary="bounded",
    )
    with pytest.raises(ValueError, match="at least one executed case"):
        build_native_whole_program_ad_execution_evidence(
            artifact_id="probe",
            cases=[fail_closed],
            gradient_parity_tolerance=1e-9,
            fail_closed_boundaries={"determinant": 20},
            claim_boundary="bounded",
        )


def test_committed_evidence_artifact_is_valid() -> None:
    """Every committed native-execution artefact reloads into valid evidence."""

    artifacts = sorted(_EVIDENCE_DIR.glob("native_whole_program_ad_execution_evidence_*.json"))
    assert artifacts, "expected at least one committed native-execution evidence artefact"
    for path in artifacts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = tuple(
            NativeWholeProgramADExecutionCase(
                case_id=row["case_id"],
                operation_family=row["operation_family"],
                operand_dimension=row["operand_dimension"],
                status=row["status"],
                value_error=row["value_error"],
                gradient_error=row["gradient_error"],
                runtime_seconds=row["runtime_seconds"],
                native_symbol=row["native_symbol"],
                failure_class=row["failure_class"],
                claim_boundary=row["claim_boundary"],
            )
            for row in payload["cases"]
        )
        rebuilt = build_native_whole_program_ad_execution_evidence(
            artifact_id=payload["artifact_id"],
            cases=cases,
            gradient_parity_tolerance=payload["gradient_parity_tolerance"],
            fail_closed_boundaries=payload["fail_closed_boundaries"],
            claim_boundary=payload["claim_boundary"],
        )
        assert isinstance(rebuilt, NativeWholeProgramADExecutionEvidence)
        assert rebuilt.beyond_scalar_executed is payload["beyond_scalar_executed"]
        assert rebuilt.max_gradient_error <= rebuilt.gradient_parity_tolerance
