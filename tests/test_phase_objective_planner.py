# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Objective Planner
"""Tests for phase/objective_planner.py execution support decisions."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    ComposedObjectivePlannerAuditResult,
    assert_composed_objective_execution_supported,
    build_phase_control_objective,
    plan_composed_objective_execution,
    run_composed_objective_planner_audit,
)


def test_objective_planner_routes_pure_periodic_objective_to_parameter_shift() -> None:
    objective = build_phase_control_objective(
        2,
        energy_weight=1.0,
        fidelity_target=np.zeros(2, dtype=float),
        fidelity_weight=0.2,
    )

    plan = plan_composed_objective_execution(objective, require_parameter_shift=True)

    assert plan.supported
    assert plan.mode == "pure_parameter_shift"
    assert plan.recommended_entrypoint == "parameter_shift_gradient_descent"
    assert plan.parameter_shift_compatible
    assert plan.analytic_terms == ()
    assert plan.parameter_shift_terms == objective.term_names
    assert "parameter-shift" in plan.reason
    supported = assert_composed_objective_execution_supported(plan)
    assert supported is plan


def test_objective_planner_routes_hybrid_objective_to_term_gradient() -> None:
    objective = build_phase_control_objective(
        2,
        energy_weight=1.0,
        safety_bounds=(-1.0, 1.0),
        safety_weight=0.2,
    )

    plan = plan_composed_objective_execution(objective)

    assert plan.supported
    assert plan.mode == "hybrid_term_gradient"
    assert plan.recommended_entrypoint == "train_composed_phase_objective"
    assert not plan.parameter_shift_compatible
    assert plan.analytic_terms == ("smooth_box_safety_penalty",)
    assert "exact term-gradient" in plan.claim_boundary


def test_objective_planner_fails_closed_for_forced_parameter_shift_and_hardware() -> None:
    objective = build_phase_control_objective(
        2,
        energy_weight=1.0,
        safety_bounds=(-1.0, 1.0),
        safety_weight=0.2,
    )

    forced = plan_composed_objective_execution(objective, require_parameter_shift=True)
    hardware = plan_composed_objective_execution(objective, backend="hardware")

    assert not forced.supported
    assert "analytic terms" in forced.reason
    assert forced.blocked_reasons
    assert not hardware.supported
    assert "provider-gradient policy" in hardware.reason
    with pytest.raises(ValueError, match="analytic terms"):
        assert_composed_objective_execution_supported(forced)


def test_objective_planner_audit_records_supported_and_blocked_routes() -> None:
    audit = run_composed_objective_planner_audit()
    payload = audit.to_dict()

    assert isinstance(audit, ComposedObjectivePlannerAuditResult)
    assert audit.passed
    assert audit.pure_plan.supported
    assert audit.hybrid_plan.supported
    assert len(audit.blocked_plans) == 3
    assert payload["passed"] is True
    assert payload["hardware_plan"]["supported"] is False
    assert "planning audit" in audit.claim_boundary


def test_objective_planner_rejects_empty_backend() -> None:
    objective = build_phase_control_objective(1)

    with pytest.raises(ValueError, match="backend"):
        plan_composed_objective_execution(objective, backend="  ")
