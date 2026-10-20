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
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.differentiable_transform_algebra as algebra
from scpn_quantum_control.differentiable_transform_algebra import (
    REQUIRED_TRANSFORM_ALGEBRA_CATEGORIES,
    REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS,
    TransformAlgebraAudit,
    TransformAlgebraCase,
    TransformAlgebraSupportMatrixRow,
    assert_transform_algebra_audit_passes,
    run_transform_algebra_audit,
)


@pytest.fixture(scope="module")
def audit_pair() -> tuple[TransformAlgebraAudit, TransformAlgebraAudit]:
    """Compute two independent immutable audits for numeric and replay assertions.

    Returns
    -------
    tuple[TransformAlgebraAudit, TransformAlgebraAudit]
        Two actual executions, reused read-only across the projection contracts.

    """
    return run_transform_algebra_audit(), run_transform_algebra_audit()


def test_transform_algebra_audit_covers_required_categories(
    audit_pair: tuple[TransformAlgebraAudit, TransformAlgebraAudit],
) -> None:
    """The audit covers every TODO category with pass or fail-closed evidence.

    Parameters
    ----------
    audit_pair:
        Two independent actual audits retained as immutable evidence fixtures.

    """
    audit = audit_pair[0]

    assert isinstance(audit, TransformAlgebraAudit)
    assert audit.passed
    assert audit.missing_categories == ()
    assert set(audit.categories) == set(REQUIRED_TRANSFORM_ALGEBRA_CATEGORIES)
    assert len(audit.passed_cases) >= 14
    assert len(audit.blocked_cases) >= 4
    assert audit.failed_cases == ()
    assert assert_transform_algebra_audit_passes(audit) is audit


def test_transform_algebra_support_matrix_is_generated_from_cases(
    audit_pair: tuple[TransformAlgebraAudit, TransformAlgebraAudit],
) -> None:
    """The reviewer-facing matrix is derived from executed or blocked audit rows.

    Parameters
    ----------
    audit_pair:
        Two independent actual audits retained as immutable evidence fixtures.

    """
    audit = audit_pair[0]
    matrix = audit.support_matrix
    rows = {row.row_id: row for row in matrix}
    case_ids = {case.case_id for case in audit.cases}

    assert set(rows) == set(REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS)
    assert all(isinstance(row, TransformAlgebraSupportMatrixRow) for row in matrix)
    assert all(set(row.case_ids).issubset(case_ids) for row in matrix)
    assert all(row.supported for row in matrix if row.status == "passed")
    assert all(not row.supported for row in matrix if row.status == "blocked")
    assert rows["native_grad_vmap"].transform_stack == ("grad", "vmap")
    assert rows["native_vmap_grad"].transform_stack == ("vmap", "grad")
    assert rows["native_jacfwd_jacrev"].transform_stack == ("jacfwd", "jacrev")
    assert rows["native_hessian"].transform_stack == ("hessian",)
    assert rows["native_jvp_vjp"].transform_stack == ("jvp", "vjp")
    assert rows["registered_custom_rules"].lane == "custom_rules"
    assert rows["program_ad_jvp_vjp"].lane == "program_ad"
    assert rows["quantum_gradient_native_nesting"].lane == "quantum_gradients"
    assert rows["unsupported_structured_container"].status == "blocked"
    assert (
        "structured parameter containers"
        in rows["unsupported_structured_container"].blocked_reasons[0]
    )


def test_transform_algebra_support_matrix_payload_is_stable(
    audit_pair: tuple[TransformAlgebraAudit, TransformAlgebraAudit],
) -> None:
    """The matrix serializes deterministically for docs and CI guards.

    Parameters
    ----------
    audit_pair:
        Two independent actual audits retained as immutable evidence fixtures.

    """
    first = audit_pair[0].to_dict()
    second = audit_pair[1].to_dict()

    assert first["support_matrix"] == second["support_matrix"]
    matrix = cast(list[dict[str, object]], first["support_matrix"])
    assert len(matrix) == len(REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS)
    assert matrix[0]["row_id"] == REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS[0]
    assert sorted(matrix[0]) == [
        "blocked_reasons",
        "case_ids",
        "claim_boundary",
        "evidence",
        "lane",
        "notes",
        "residual",
        "row_id",
        "status",
        "supported",
        "tolerance",
        "transform_stack",
    ]


def test_transform_algebra_cases_preserve_diagnostic_boundaries(
    audit_pair: tuple[TransformAlgebraAudit, TransformAlgebraAudit],
) -> None:
    """Finite-difference and unsupported rows stay explicitly bounded.

    Parameters
    ----------
    audit_pair:
        Two independent actual audits retained as immutable evidence fixtures.

    """
    audit = audit_pair[0]
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


def test_transform_algebra_payload_is_stable_and_json_ready(
    audit_pair: tuple[TransformAlgebraAudit, TransformAlgebraAudit],
) -> None:
    """The audit payload is deterministic enough for reviewer-facing gates.

    Parameters
    ----------
    audit_pair:
        Two independent actual audits retained as immutable evidence fixtures.

    """
    first = audit_pair[0].to_dict()
    second = audit_pair[1].to_dict()

    assert first == second
    assert first["passed"] is True
    assert first["case_count"] == len(REQUIRED_TRANSFORM_ALGEBRA_CATEGORIES) + 1
    assert "support_matrix" in first
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


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"case_id": ""}, "case_id"),
        ({"category": ""}, "category"),
        ({"status": "unknown"}, "status"),
        ({"status": "blocked"}, "must not carry"),
        ({"status": "blocked", "residual": None}, "require blocked_reasons"),
        ({"residual": None}, "require a residual"),
        ({"residual": -1.0}, "non-negative"),
        ({"status": "failed"}, "within tolerance"),
        ({"tolerance": -1.0}, "exceeds tolerance"),
        ({"status": "failed", "residual": 1.0, "tolerance": -1.0}, "tolerance must be"),
        ({"evidence": ("",)}, "evidence"),
        ({"blocked_reasons": ("",)}, "blocked reasons"),
        ({"claim_boundary": ""}, "claim_boundary"),
    ],
)
def test_case_boundary_rejects_inconsistent_evidence(changes: dict[str, Any], match: str) -> None:
    """Reject corrupt public case metadata before serializing support claims.

    Parameters
    ----------
    changes:
        Invalid case fields applied to a valid executed result.
    match:
        Required observable refusal diagnostic.

    """
    valid = TransformAlgebraCase(
        "sample", "gradient", "passed", (1.0,), (1.0,), 0.0, 1e-10, ("runtime",), ()
    )
    with pytest.raises(ValueError, match=match):
        replace(valid, **changes)


def test_audit_refuses_invalid_envelopes_and_reports_incomplete_evidence() -> None:
    """Incomplete categories and missing support rows fail the public assertion."""
    case = TransformAlgebraCase(
        "sample", "gradient", "failed", (1.0,), (0.0,), 1.0, 1e-10, ("runtime",), ()
    )
    with pytest.raises(ValueError, match="contain cases"):
        TransformAlgebraAudit(())
    with pytest.raises(ValueError, match="required_categories"):
        TransformAlgebraAudit((case,), required_categories=("",))
    with pytest.raises(ValueError, match="claim_boundary"):
        TransformAlgebraAudit((case,), claim_boundary="")
    audit = TransformAlgebraAudit((case,), required_categories=("hessian",))
    assert not audit.passed
    assert audit.missing_categories == ("hessian",)
    assert audit.failed_support_rows
    assert audit.failed_cases == (case,)
    with pytest.raises(AssertionError, match="missing categories: hessian.*sample residual=1.0"):
        assert_transform_algebra_audit_passes(audit)


def test_audit_rejects_nonpositive_diagnostic_tolerance() -> None:
    """A zero residual budget is not an admitted legacy diagnostic request."""
    with pytest.raises(ValueError, match="tolerance must be positive"):
        run_transform_algebra_audit(tolerance=0.0)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--batch-axis", "1"],
        ["--values", "[]"],
        ["--batch-axis", "0"],
        ["--values", "[0.25, -0.5, 0.75]"],
    ],
)
def test_cli_refuses_native_input_options_without_framework(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse ambiguous CLI options instead of running the default diagnostic.

    Parameters
    ----------
    arguments:
        Native-only options with no runtime selector.
    monkeypatch:
        Scoped process argument substitution for the public CLI entrypoint.

    """
    from tools.check_differentiable_transform_algebra import main

    monkeypatch.setattr(sys, "argv", ["transform-check", *arguments])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_public_audit_refuses_corrupt_empty_or_misshaped_engine_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fault injection cannot promote empty transforms or wrong-shape Jacobians.

    Parameters
    ----------
    monkeypatch:
        Scoped public engine-output faults supplementing real positive audits.

    """
    from scpn_quantum_control.phase.qnode_vector_transforms import execute_phase_qnode_vmap_grad

    def no_objective_call(values: object) -> float:
        """Refuse evaluation because the actual unsupported planner must stop first."""
        raise AssertionError("unsupported quantum bridge executed objective")

    blocked = execute_phase_qnode_vmap_grad(no_objective_call, [[0.2, -0.3]], adapter="jax")
    assert not blocked.supported
    monkeypatch.setattr(algebra, "execute_phase_qnode_vmap_grad", lambda *args, **kwargs: blocked)
    monkeypatch.setattr(algebra, "jacfwd", lambda *args, **kwargs: np.array([]))
    monkeypatch.setattr(algebra, "jacrev", lambda *args, **kwargs: np.array([]))
    monkeypatch.setattr(algebra, "jacobian", lambda *args, **kwargs: np.ones((1, 1)))
    audit = run_transform_algebra_audit()
    cases = {case.case_id: case for case in audit.cases}
    empty = cases["jacfwd_matches_jacrev_for_local_vector_objective"]
    wrong_shape = cases["batched_observable_jacobian_matches_analytic_reference"]
    assert empty.status == "failed"
    assert empty.residual == float("inf")
    assert wrong_shape.status == "failed"
    assert wrong_shape.residual == float("inf")
    assert cases["phase_qnode_native_vmap_grad_is_fail_closed"].status == "blocked"
    assert not audit.passed
    with pytest.raises(AssertionError, match="jacfwd_matches_jacrev"):
        assert_transform_algebra_audit_passes(audit)


def test_public_audit_rejects_nonfinite_engine_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-finite numerical-engine result is an observable audit failure.

    Parameters
    ----------
    monkeypatch:
        Scoped public forward-Jacobian failure; removed after the refusal.

    """
    monkeypatch.setattr(algebra, "jacfwd", lambda *args, **kwargs: np.array([np.inf]))
    with pytest.raises(ValueError, match="values must be finite"):
        run_transform_algebra_audit()


def test_incomplete_native_case_ids_refuse_support_promotion() -> None:
    """Unknown native case IDs cannot satisfy a declared category or support row."""
    from scpn_quantum_control.differentiable_transform_support_matrix import (
        NATIVE_TRANSFORM_ALGEBRA_CATEGORIES,
    )

    case = TransformAlgebraCase(
        "unknown",
        "framework_grad",
        "passed",
        (1.0,),
        (1.0,),
        0.0,
        1e-10,
        ("unregistered row",),
        (),
    )
    audit = TransformAlgebraAudit((case,), required_categories=NATIVE_TRANSFORM_ALGEBRA_CATEGORIES)
    assert audit.missing_support_rows == ("unknown",)
    assert not audit.passed


def test_public_audit_detects_support_generator_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """A generator that drops its required rows remains visibly incomplete.

    Parameters
    ----------
    monkeypatch:
        Scoped source-to-matrix projection fault with no numerical execution.

    """
    case = TransformAlgebraCase(
        "sample", "gradient", "passed", (1.0,), (1.0,), 0.0, 1e-10, ("projection fault",), ()
    )
    monkeypatch.setattr(algebra, "build_transform_algebra_support_matrix", lambda cases: ())
    audit = TransformAlgebraAudit((case,))
    assert audit.missing_support_rows == REQUIRED_TRANSFORM_ALGEBRA_SUPPORT_ROWS
    assert not audit.passed
    with pytest.raises(AssertionError, match="missing support rows"):
        assert_transform_algebra_audit_passes(audit)
