# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — transform algebra tests.
"""Tests for differentiable transform-algebra metamorphic coverage."""

from __future__ import annotations

import subprocess
import sys
from typing import cast

import pytest

from scpn_quantum_control.differentiable_transform_algebra import (
    REQUIRED_TRANSFORM_ALGEBRA_CATEGORIES,
    TransformAlgebraAudit,
    TransformAlgebraCase,
    assert_transform_algebra_audit_passes,
    run_transform_algebra_audit,
)


def test_transform_algebra_audit_covers_required_categories() -> None:
    """The audit covers every TODO category with pass or fail-closed evidence."""
    audit = run_transform_algebra_audit()

    assert isinstance(audit, TransformAlgebraAudit)
    assert audit.passed
    assert audit.missing_categories == ()
    assert set(audit.categories) == set(REQUIRED_TRANSFORM_ALGEBRA_CATEGORIES)
    assert len(audit.passed_cases) >= 14
    assert len(audit.blocked_cases) >= 4
    assert audit.failed_cases == ()
    assert assert_transform_algebra_audit_passes(audit) is audit


def test_transform_algebra_cases_preserve_diagnostic_boundaries() -> None:
    """Finite-difference and unsupported rows stay explicitly bounded."""
    audit = run_transform_algebra_audit()
    cases = {case.case_id: case for case in audit.cases}

    finite = cases["finite_difference_result_keeps_diagnostic_claim_boundary"]
    periodic = cases["parameter_shift_gradient_is_phase_periodic"]
    nondiff = cases["nondifferentiable_abs_zero_boundary"]
    custom = cases["custom_jvp_vjp_unregistered_boundary"]
    container = cases["structured_parameter_container_boundary"]

    assert finite.status == "passed"
    assert finite.residual == pytest.approx(0.0)
    assert "finite_difference_diagnostic_only" in finite.evidence
    assert periodic.status == "passed"
    assert periodic.category == "periodicity"
    assert periodic.residual == pytest.approx(0.0, abs=1.0e-12)
    assert "parameter_shift" in periodic.evidence
    assert "phase_wraparound" in periodic.evidence
    assert nondiff.status == "blocked"
    assert "cannot promote differentiability" in nondiff.blocked_reasons[0]
    assert custom.status == "blocked"
    assert "registered exact rules" in custom.blocked_reasons[0]
    assert container.status == "blocked"
    assert "PyTree/container metadata" in container.blocked_reasons[0]


def test_transform_algebra_payload_is_stable_and_json_ready() -> None:
    """The audit payload is deterministic enough for reviewer-facing gates."""
    first = run_transform_algebra_audit().to_dict()
    second = run_transform_algebra_audit().to_dict()

    assert first == second
    assert first["passed"] is True
    assert first["case_count"] == len(REQUIRED_TRANSFORM_ALGEBRA_CATEGORIES) + 1
    assert "hardware" in cast(str, first["claim_boundary"])
    case_payload = cast(list[dict[str, object]], first["cases"])[0]
    assert sorted(case_payload) == [
        "blocked_reasons",
        "case_id",
        "category",
        "claim_boundary",
        "evidence",
        "lhs",
        "residual",
        "rhs",
        "status",
        "tolerance",
    ]


def test_transform_algebra_case_rejects_invalid_pass_state() -> None:
    """Case validation rejects inconsistent pass/fail/block metadata."""
    with pytest.raises(ValueError, match="exceeds tolerance"):
        TransformAlgebraCase(
            case_id="bad",
            category="linearity",
            status="passed",
            lhs=(1.0,),
            rhs=(2.0,),
            residual=1.0,
            tolerance=0.1,
            evidence=("unit",),
            blocked_reasons=(),
        )


def test_transform_algebra_cli_uses_real_gate() -> None:
    """The CLI checker executes the production transform-algebra audit."""
    completed = subprocess.run(
        [sys.executable, "tools/check_differentiable_transform_algebra.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "differentiable transform-algebra gate: PASS" in completed.stdout
