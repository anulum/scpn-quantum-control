# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable benchmark report builders
"""Claim-bounded local differentiable benchmark report builders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, cast

import numpy as np

from .benchmarks.differentiable_programming import (
    run_differentiable_programming_benchmark_suite,
    run_quantum_gradient_benchmark_suite,
)
from .phase.gradient_support_matrix import run_gradient_support_matrix_audit

DIFFERENTIABLE_BENCHMARK_REPORT_METHOD = "local_conformance_bundle"
DIFFERENTIABLE_BENCHMARK_REPORT_CLAIM_BOUNDARY = (
    "local deterministic conformance benchmark bundle; not isolated "
    "performance, hardware, or provider execution evidence"
)


@dataclass(frozen=True)
class DifferentiableBenchmarkReport:
    """Local conformance benchmark evidence for the unified differentiable API."""

    supported: bool
    method: str
    payload: Mapping[str, object]
    claim_boundary: str = DIFFERENTIABLE_BENCHMARK_REPORT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate that the report preserves claim-bounded benchmark semantics."""
        if not isinstance(self.supported, bool):
            raise ValueError("differentiable benchmark report supported must be boolean")
        if not self.method:
            raise ValueError("differentiable benchmark report method must be non-empty")
        if not self.claim_boundary:
            raise ValueError("differentiable benchmark report claim_boundary must be non-empty")
        required_keys = {
            "program_ad_case_count",
            "quantum_gradient_case_count",
            "support_audit_passed",
            "program_ad_cases",
            "quantum_gradient_cases",
        }
        missing = required_keys.difference(self.payload)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"differentiable benchmark report payload missing: {missing_text}")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready benchmark report payload."""
        return {
            "supported": self.supported,
            "method": self.method,
            "payload": dict(self.payload),
            "claim_boundary": self.claim_boundary,
        }


def build_differentiable_benchmark_report() -> DifferentiableBenchmarkReport:
    """Run local conformance suites and return bounded benchmark evidence."""
    program_rows = run_differentiable_programming_benchmark_suite()
    quantum_rows = run_quantum_gradient_benchmark_suite()
    support_audit = run_gradient_support_matrix_audit()
    passed = (
        all(row.passed for row in program_rows)
        and all(row.passed for row in quantum_rows)
        and support_audit.passed
    )
    return DifferentiableBenchmarkReport(
        supported=passed,
        method=DIFFERENTIABLE_BENCHMARK_REPORT_METHOD,
        payload={
            "program_ad_case_count": len(program_rows),
            "quantum_gradient_case_count": len(quantum_rows),
            "support_audit_passed": support_audit.passed,
            "program_ad_cases": [_dataclass_payload(row) for row in program_rows],
            "quantum_gradient_cases": [_dataclass_payload(row) for row in quantum_rows],
        },
    )


def _dataclass_payload(value: object) -> dict[str, object]:
    """Convert benchmark dataclass rows into deterministic JSON-ready payloads."""
    if not is_dataclass(value):
        raise TypeError("benchmark payload values must be dataclass instances")
    payload: dict[str, object] = {}
    for field in fields(cast(Any, value)):
        payload[field.name] = _json_ready(getattr(value, field.name))
    passed = getattr(value, "passed", None)
    if isinstance(passed, bool):
        payload["passed"] = passed
    return payload


def _json_ready(value: object) -> object:
    """Return a JSON-compatible representation of benchmark row values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


__all__ = [
    "DIFFERENTIABLE_BENCHMARK_REPORT_CLAIM_BOUNDARY",
    "DIFFERENTIABLE_BENCHMARK_REPORT_METHOD",
    "DifferentiableBenchmarkReport",
    "build_differentiable_benchmark_report",
]
