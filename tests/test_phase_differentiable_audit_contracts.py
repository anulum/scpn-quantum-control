# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Gradient Audit Contract Tests
"""Tests for immutable differentiable gradient audit contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar

import numpy as np
import pytest

import scpn_quantum_control.phase.differentiable_audit as audit_module
import scpn_quantum_control.phase.differentiable_audit_contracts as contract_module
from scpn_quantum_control.phase import (
    run_differentiable_workflow_audit_suite,
    run_finite_shot_gradient_uncertainty_audit,
    run_known_phase_gradient_audit,
)

_RecordT = TypeVar("_RecordT")


def _replace(record: _RecordT, changes: dict[str, object]) -> _RecordT:
    """Replace validation fields while retaining the concrete record type."""
    return replace(record, **changes)  # type: ignore[type-var]


@pytest.mark.parametrize(
    "name",
    (
        "DifferentiableQuantumAuditReport",
        "DifferentiableWorkflowAuditSuiteResult",
        "FiniteShotGradientAuditResult",
        "MLFrameworkGradientAuditRecord",
        "MLFrameworkGradientAuditSuiteResult",
        "ParameterShiftAnalyticAgreement",
        "PhaseGradientBenchmarkSuiteResult",
    ),
)
def test_differentiable_audit_facade_reexports_contract_identity(name: str) -> None:
    """Verify that every moved public contract keeps exact facade identity."""
    assert getattr(audit_module, name) is getattr(contract_module, name)


def test_parameter_shift_analytic_agreement_rejects_non_finite_gradient() -> None:
    """Verify that extracted agreement contracts retain fail-closed validation."""
    with pytest.raises(ValueError, match="parameter_shift_gradient"):
        contract_module.ParameterShiftAnalyticAgreement(
            parameters=np.array([0.1], dtype=float),
            parameter_shift_gradient=np.array([np.nan], dtype=float),
            analytic_gradient=np.array([0.2], dtype=float),
            abs_error=np.array([0.1], dtype=float),
            max_abs_error=0.1,
            tolerance=1.0e-8,
            passed=False,
            method="parameter_shift_vs_analytic_gradient",
            evaluations=2,
            claim_boundary="bounded unit-test contract",
        )


def _analytic_agreement() -> contract_module.ParameterShiftAnalyticAgreement:
    """Build a minimal valid analytic-agreement record."""
    return contract_module.ParameterShiftAnalyticAgreement(
        parameters=np.array([0.1], dtype=float),
        parameter_shift_gradient=np.array([0.2], dtype=float),
        analytic_gradient=np.array([0.2], dtype=float),
        abs_error=np.array([0.0], dtype=float),
        max_abs_error=0.0,
        tolerance=1.0e-8,
        passed=True,
        method="parameter_shift_vs_analytic_gradient",
        evaluations=2,
        claim_boundary="bounded unit-test contract",
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"parameters": np.array([True])}, "finite real numeric"),
        ({"parameters": np.array([[0.1]])}, "one-dimensional"),
        ({"parameters": np.array([np.inf])}, "finite real numeric"),
        ({"parameter_shift_gradient": np.array([0.1, 0.2])}, "shape must match"),
        ({"analytic_gradient": np.array([0.1, 0.2])}, "shape must match"),
        ({"abs_error": np.array([0.1, 0.2])}, "shape must match"),
        ({"abs_error": np.array([-0.1])}, "must be non-negative"),
        ({"max_abs_error": np.array([0.0])}, "finite real scalar"),
        ({"max_abs_error": True}, "finite real scalar"),
        ({"max_abs_error": np.inf}, "finite real scalar"),
        ({"max_abs_error": -0.1}, "must be non-negative"),
        ({"tolerance": -0.1}, "finite and non-negative"),
        ({"passed": 1}, "must be a boolean"),
        ({"method": ""}, "must be non-empty"),
        ({"evaluations": True}, "positive integer"),
        ({"evaluations": 0}, "positive integer"),
    ),
)
def test_parameter_shift_analytic_agreement_rejects_invalid_evidence(
    changes: dict[str, object],
    message: str,
) -> None:
    """Reject every malformed analytic-agreement field."""
    with pytest.raises(ValueError, match=message):
        _replace(_analytic_agreement(), changes)


def test_parameter_shift_analytic_agreement_copies_arrays() -> None:
    """Copy caller-owned arrays and serialize the complete public schema."""
    parameters = np.array([0.1], dtype=float)
    record = replace(_analytic_agreement(), parameters=parameters)
    parameters[0] = 9.0

    assert record.parameters.tolist() == [0.1]
    assert record.to_dict() == {
        "parameters": [0.1],
        "parameter_shift_gradient": [0.2],
        "analytic_gradient": [0.2],
        "abs_error": [0.0],
        "max_abs_error": 0.0,
        "tolerance": 1.0e-8,
        "passed": True,
        "method": "parameter_shift_vs_analytic_gradient",
        "evaluations": 2,
        "claim_boundary": "bounded unit-test contract",
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"benchmark_names": ()}, "benchmark_names must be non-empty"),
        ({"reports": ()}, "length must match"),
        ({"benchmark_names": ("",)}, "contain non-empty names"),
        ({"unsupported_scenarios": ()}, "unsupported_scenarios must be non-empty"),
        ({"unsupported_scenarios": ("",)}, "contain non-empty items"),
        ({"passed": 1}, "passed must be a boolean"),
        ({"claim_boundary": ""}, "claim_boundary must be non-empty"),
    ),
)
def test_phase_benchmark_suite_rejects_invalid_metadata(
    changes: dict[str, object],
    message: str,
) -> None:
    """Reject every malformed benchmark-suite metadata field."""
    report = run_known_phase_gradient_audit(np.array([0.2], dtype=float))
    suite = contract_module.PhaseGradientBenchmarkSuiteResult(
        benchmark_names=("one",),
        reports=(report,),
        unsupported_scenarios=("hardware",),
        passed=True,
        claim_boundary="bounded unit-test contract",
    )

    with pytest.raises(ValueError, match=message):
        _replace(suite, changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"workflow_names": ()}, "workflow_names must be non-empty"),
        ({"workflow_names": ("",)}, "contain non-empty names"),
        ({"unsupported_scenarios": ()}, "unsupported_scenarios must be non-empty"),
        ({"unsupported_scenarios": ("",)}, "contain non-empty items"),
        ({"passed": 1}, "passed must be a boolean"),
        ({"claim_boundary": ""}, "claim_boundary must be non-empty"),
    ),
)
def test_workflow_suite_rejects_invalid_metadata(
    changes: dict[str, object],
    message: str,
) -> None:
    """Reject every malformed workflow-suite metadata field."""
    with pytest.raises(ValueError, match=message):
        _replace(run_differentiable_workflow_audit_suite(), changes)


def _finite_shot_audit() -> contract_module.FiniteShotGradientAuditResult:
    """Build a deterministic finite-shot audit result."""

    def objective(values: contract_module.FloatArray) -> float:
        """Return a bounded periodic objective."""
        return float(np.mean(1.0 - np.cos(values)))

    return run_finite_shot_gradient_uncertainty_audit(
        objective,
        np.array([0.2], dtype=float),
        target_standard_error=0.02,
        plus_variances=np.array([0.04], dtype=float),
        minus_variances=np.array([0.04], dtype=float),
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"deterministic_gradient": np.array([0.1, 0.2])}, "shape must match"),
        ({"abs_error": np.array([0.1, 0.2])}, "shape must match"),
        ({"within_confidence": ()}, "length must match"),
        ({"abs_error": np.array([-0.1])}, "must be non-negative"),
        ({"target_standard_error": 0.0}, "finite and positive"),
        ({"max_abs_error": -0.1}, "maxima must be non-negative"),
        ({"max_confidence_radius": -0.1}, "maxima must be non-negative"),
        ({"max_standard_error": -0.1}, "maxima must be non-negative"),
        ({"executed_total_shots": True}, "positive integer"),
        ({"executed_total_shots": 0}, "positive integer"),
        ({"passed": 1}, "passed must be a boolean"),
        ({"method": ""}, "method must be non-empty"),
        ({"claim_boundary": ""}, "claim_boundary must be non-empty"),
    ),
)
def test_finite_shot_audit_rejects_invalid_evidence(
    changes: dict[str, object],
    message: str,
) -> None:
    """Reject every inconsistent finite-shot evidence field."""
    with pytest.raises(ValueError, match=message):
        _replace(_finite_shot_audit(), changes)


def _framework_record() -> contract_module.MLFrameworkGradientAuditRecord:
    """Build a minimal executed framework record."""
    return contract_module.MLFrameworkGradientAuditRecord(
        framework="jax",
        available=True,
        executed=True,
        status="passed",
        reason="adapter executed",
        value=0.5,
        gradient=np.array([0.2], dtype=float),
        reference_gradient=np.array([0.2], dtype=float),
        abs_error=np.array([0.0], dtype=float),
        max_abs_error=0.0,
        tolerance=1.0e-8,
        claim_boundary="bounded unit-test contract",
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"framework": ""}, "framework must be non-empty"),
        ({"available": 1}, "available and executed must be booleans"),
        ({"executed": 1}, "available and executed must be booleans"),
        ({"status": "unknown"}, "status must be"),
        ({"reason": ""}, "reason must be non-empty"),
        ({"tolerance": -0.1}, "finite and non-negative"),
        ({"gradient": np.array([0.1, 0.2])}, "shape must match"),
        ({"abs_error": np.array([0.1, 0.2])}, "shape must match"),
        ({"max_abs_error": -0.1}, "must be non-negative"),
        ({"claim_boundary": ""}, "claim_boundary must be non-empty"),
    ),
)
def test_framework_record_rejects_invalid_evidence(
    changes: dict[str, object],
    message: str,
) -> None:
    """Reject every malformed per-framework evidence field."""
    with pytest.raises(ValueError, match=message):
        _replace(_framework_record(), changes)


def test_framework_record_supports_unexecuted_optional_values() -> None:
    """Serialize an unavailable framework without fabricated numerical evidence."""
    record = replace(
        _framework_record(),
        available=False,
        executed=False,
        status="unavailable",
        reason="dependency absent",
        value=None,
        gradient=None,
        abs_error=None,
        max_abs_error=None,
    )

    payload = record.to_dict()
    assert payload["value"] is None
    assert payload["gradient"] is None
    assert payload["abs_error"] is None
    assert payload["max_abs_error"] is None


@pytest.mark.parametrize(
    ("records", "audit_passed", "claim_boundary", "message"),
    (
        ((), True, "bounded", "records must be non-empty"),
        ((_framework_record(), _framework_record()), True, "bounded", "must be unique"),
        ((_framework_record(),), 1, "bounded", "audit_passed must be a boolean"),
        ((_framework_record(),), True, "", "claim_boundary must be non-empty"),
    ),
)
def test_framework_suite_rejects_invalid_aggregate_metadata(
    records: tuple[contract_module.MLFrameworkGradientAuditRecord, ...],
    audit_passed: object,
    claim_boundary: str,
    message: str,
) -> None:
    """Reject empty, duplicate, and malformed framework-suite evidence."""
    with pytest.raises(ValueError, match=message):
        contract_module.MLFrameworkGradientAuditSuiteResult(
            records=records,
            audit_passed=audit_passed,  # type: ignore[arg-type]
            claim_boundary=claim_boundary,
        )


def test_framework_suite_projects_each_status_and_error() -> None:
    """Project executed, unavailable, blocked, and failed framework statuses."""
    base = _framework_record()
    records = (
        base,
        replace(
            base,
            framework="torch",
            available=False,
            executed=False,
            status="unavailable",
            value=None,
            gradient=None,
            abs_error=None,
            max_abs_error=None,
        ),
        replace(base, framework="tensorflow", executed=False, status="blocked"),
        replace(base, framework="pennylane", status="failed", max_abs_error=0.25),
    )
    suite = contract_module.MLFrameworkGradientAuditSuiteResult(
        records=records,
        audit_passed=False,
        claim_boundary="bounded unit-test contract",
    )

    assert suite.executed_frameworks == ("jax", "pennylane")
    assert suite.unavailable_frameworks == ("torch",)
    assert suite.blocked_frameworks == ("tensorflow",)
    assert suite.failed_frameworks == ("pennylane",)
    assert suite.worst_executed_error == 0.25
    assert suite.to_dict()["worst_executed_error"] == 0.25
