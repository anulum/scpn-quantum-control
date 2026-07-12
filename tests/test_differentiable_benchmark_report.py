# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable benchmark report tests
# scpn-quantum-control -- tests for differentiable benchmark report builders
"""Tests for scpn_quantum_control.differentiable_benchmark_report."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control as scpn
from scpn_quantum_control.differentiable_api import differentiable_benchmark_report
from scpn_quantum_control.differentiable_benchmark_report import (
    DIFFERENTIABLE_BENCHMARK_REPORT_CLAIM_BOUNDARY,
    DIFFERENTIABLE_BENCHMARK_REPORT_METHOD,
    DifferentiableBenchmarkReport,
    build_differentiable_benchmark_report,
)

benchmark_report_module = importlib.import_module(
    "scpn_quantum_control.differentiable_benchmark_report",
)


@dataclass(frozen=True)
class _ControlledBenchmarkRow:
    """Controlled benchmark row for exercising report serialization branches."""

    case_id: str
    values: NDArray[np.float64]
    tags: list[tuple[str, int]]
    metadata: dict[str, tuple[NDArray[np.float64], ...]]
    passed: object


def _require_torch_backend() -> None:
    pytest.importorskip("torch", reason="native Torch differentiable rows require PyTorch")


def test_differentiable_benchmark_report_validates_required_payload_keys() -> None:
    """Report rows fail closed when a benchmark evidence key is missing."""

    with pytest.raises(ValueError, match="payload missing"):
        DifferentiableBenchmarkReport(
            supported=True,
            method=DIFFERENTIABLE_BENCHMARK_REPORT_METHOD,
            payload={"program_ad_case_count": 1},
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"supported": "yes"}, "supported must be boolean"),
        ({"method": ""}, "method must be non-empty"),
        ({"claim_boundary": ""}, "claim_boundary must be non-empty"),
    ),
)
def test_differentiable_benchmark_report_rejects_invalid_claim_metadata(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Report construction should reject malformed claim-boundary metadata."""

    params: dict[str, object] = {
        "supported": True,
        "method": DIFFERENTIABLE_BENCHMARK_REPORT_METHOD,
        "payload": {
            "program_ad_case_count": 0,
            "quantum_gradient_case_count": 0,
            "support_audit_passed": True,
            "program_ad_cases": [],
            "quantum_gradient_cases": [],
        },
        "claim_boundary": DIFFERENTIABLE_BENCHMARK_REPORT_CLAIM_BOUNDARY,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        DifferentiableBenchmarkReport(**params)  # type: ignore[arg-type]  # malformed metadata guard


def test_differentiable_benchmark_report_to_dict_preserves_payload() -> None:
    """Report dictionaries should preserve the bounded evidence envelope."""

    report = DifferentiableBenchmarkReport(
        supported=True,
        method=DIFFERENTIABLE_BENCHMARK_REPORT_METHOD,
        payload={
            "program_ad_case_count": 0,
            "quantum_gradient_case_count": 0,
            "support_audit_passed": True,
            "program_ad_cases": [],
            "quantum_gradient_cases": [],
        },
    )

    payload = report.to_dict()

    assert payload["supported"] is True
    assert payload["method"] == DIFFERENTIABLE_BENCHMARK_REPORT_METHOD
    assert payload["claim_boundary"] == DIFFERENTIABLE_BENCHMARK_REPORT_CLAIM_BOUNDARY
    assert payload["payload"] == report.payload


def test_build_differentiable_benchmark_report_is_non_performance_evidence() -> None:
    """The extracted builder preserves local conformance semantics and boundaries."""

    _require_torch_backend()

    report = build_differentiable_benchmark_report()

    assert report.method == DIFFERENTIABLE_BENCHMARK_REPORT_METHOD
    assert report.claim_boundary == DIFFERENTIABLE_BENCHMARK_REPORT_CLAIM_BOUNDARY
    program_ad_case_count = report.payload["program_ad_case_count"]
    quantum_gradient_case_count = report.payload["quantum_gradient_case_count"]
    assert isinstance(program_ad_case_count, int)
    assert isinstance(quantum_gradient_case_count, int)
    assert program_ad_case_count > 0
    assert quantum_gradient_case_count > 0
    assert report.payload["support_audit_passed"] is True
    assert "not isolated performance" in report.claim_boundary
    program_ad_cases = report.payload["program_ad_cases"]
    assert isinstance(program_ad_cases, list)
    first_program_case = program_ad_cases[0]
    assert isinstance(first_program_case, dict)
    assert first_program_case["passed"] is True
    blocked_program_cases = [
        case for case in program_ad_cases if isinstance(case, dict) and case.get("blocked_reasons")
    ]
    assert report.supported is (not blocked_program_cases)
    if blocked_program_cases:
        assert {
            case["case_id"]
            for case in blocked_program_cases
            if isinstance(case.get("case_id"), str)
        } == {"program_ad_rust_scalar_interpreter_contracts"}


def test_unified_api_wraps_extracted_benchmark_report() -> None:
    """The compatibility facade delegates to the extracted report builder."""

    _require_torch_backend()

    report = differentiable_benchmark_report()

    assert report.operation == "benchmark_report"
    assert report.method == DIFFERENTIABLE_BENCHMARK_REPORT_METHOD
    assert report.claim_boundary == DIFFERENTIABLE_BENCHMARK_REPORT_CLAIM_BOUNDARY
    assert report.payload["support_audit_passed"] is True
    program_ad_cases = report.payload["program_ad_cases"]
    assert isinstance(program_ad_cases, list)
    blocked_program_cases = [
        case for case in program_ad_cases if isinstance(case, dict) and case.get("blocked_reasons")
    ]
    assert report.supported is (not blocked_program_cases)
    assert scpn.DifferentiableBenchmarkReport is DifferentiableBenchmarkReport
    assert scpn.build_differentiable_benchmark_report is build_differentiable_benchmark_report


def test_builder_serializes_nested_dataclass_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public builder should serialize arrays, tuples, lists, and mappings."""

    program_row = _ControlledBenchmarkRow(
        case_id="controlled_program_row",
        values=np.array([1.0, 2.0], dtype=np.float64),
        tags=[("alpha", 1)],
        metadata={"nested": (np.array([3.0], dtype=np.float64),)},
        passed="metadata-only",
    )
    quantum_row = _ControlledBenchmarkRow(
        case_id="controlled_quantum_row",
        values=np.array([4.0], dtype=np.float64),
        tags=[("beta", 2)],
        metadata={"nested": (np.array([5.0], dtype=np.float64),)},
        passed=True,
    )
    monkeypatch.setattr(
        benchmark_report_module,
        "run_differentiable_programming_benchmark_suite",
        lambda: (program_row,),
    )
    monkeypatch.setattr(
        benchmark_report_module,
        "run_quantum_gradient_benchmark_suite",
        lambda: (quantum_row,),
    )
    monkeypatch.setattr(
        benchmark_report_module,
        "run_gradient_support_matrix_audit",
        lambda: SimpleNamespace(passed=True),
    )

    report = build_differentiable_benchmark_report()

    program_cases = report.payload["program_ad_cases"]
    quantum_cases = report.payload["quantum_gradient_cases"]
    assert isinstance(program_cases, list)
    assert isinstance(quantum_cases, list)
    assert program_cases == [
        {
            "case_id": "controlled_program_row",
            "values": [1.0, 2.0],
            "tags": [["alpha", 1]],
            "metadata": {"nested": [[3.0]]},
            "passed": "metadata-only",
        },
    ]
    assert quantum_cases[0]["passed"] is True


def test_builder_rejects_non_dataclass_benchmark_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public builder should fail closed if a suite returns non-dataclass rows."""

    monkeypatch.setattr(
        benchmark_report_module,
        "run_differentiable_programming_benchmark_suite",
        lambda: (SimpleNamespace(passed=True),),
    )
    monkeypatch.setattr(
        benchmark_report_module,
        "run_quantum_gradient_benchmark_suite",
        lambda: (),
    )
    monkeypatch.setattr(
        benchmark_report_module,
        "run_gradient_support_matrix_audit",
        lambda: SimpleNamespace(passed=True),
    )

    with pytest.raises(TypeError, match="dataclass"):
        build_differentiable_benchmark_report()
