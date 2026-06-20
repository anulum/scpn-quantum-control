# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for differentiable benchmark report builders
"""Tests for scpn_quantum_control.differentiable_benchmark_report."""

from __future__ import annotations

import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control.differentiable_api import differentiable_benchmark_report
from scpn_quantum_control.differentiable_benchmark_report import (
    DIFFERENTIABLE_BENCHMARK_REPORT_CLAIM_BOUNDARY,
    DIFFERENTIABLE_BENCHMARK_REPORT_METHOD,
    DifferentiableBenchmarkReport,
    build_differentiable_benchmark_report,
)


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


def test_build_differentiable_benchmark_report_is_non_performance_evidence() -> None:
    """The extracted builder preserves local conformance semantics and boundaries."""

    _require_torch_backend()

    report = build_differentiable_benchmark_report()

    assert report.supported
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


def test_unified_api_wraps_extracted_benchmark_report() -> None:
    """The compatibility facade delegates to the extracted report builder."""

    _require_torch_backend()

    report = differentiable_benchmark_report()

    assert report.operation == "benchmark_report"
    assert report.supported
    assert report.method == DIFFERENTIABLE_BENCHMARK_REPORT_METHOD
    assert report.claim_boundary == DIFFERENTIABLE_BENCHMARK_REPORT_CLAIM_BOUNDARY
    assert report.payload["support_audit_passed"] is True
    assert scpn.DifferentiableBenchmarkReport is DifferentiableBenchmarkReport
    assert scpn.build_differentiable_benchmark_report is build_differentiable_benchmark_report
