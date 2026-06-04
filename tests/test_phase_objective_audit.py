# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Objective Audit
"""Tests for phase/objective_audit.py objective evidence reports."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    ComposedObjectiveAuditSuiteResult,
    build_phase_control_objective,
    run_composed_objective_audit_suite,
    verify_composed_objective_gradient,
)


def test_verify_composed_objective_gradient_matches_finite_difference() -> None:
    objective = build_phase_control_objective(
        2,
        energy_weight=1.0,
        fidelity_target=np.zeros(2, dtype=float),
        fidelity_weight=0.2,
        symmetry_pairs=((0, 1),),
        symmetry_weight=0.1,
    )

    agreement = verify_composed_objective_gradient(
        objective,
        np.array([0.7, -0.5], dtype=float),
    )

    assert agreement.passed
    assert agreement.parameter_shift_compatible
    assert agreement.finite_difference_evaluations == 4
    assert agreement.max_abs_error < 1e-5
    assert agreement.term_names == objective.term_names
    assert "finite-difference diagnostic" in agreement.claim_boundary


def test_composed_objective_audit_suite_reports_supported_and_blocked_routes() -> None:
    suite = run_composed_objective_audit_suite()
    payload = suite.to_dict()

    assert isinstance(suite, ComposedObjectiveAuditSuiteResult)
    assert suite.passed
    assert suite.pure_gradient.parameter_shift_compatible
    assert not suite.hybrid_gradient.parameter_shift_compatible
    assert suite.pure_parameter_shift_gate_passed
    assert suite.hybrid_parameter_shift_gate_failed
    assert "non-parameter-shift" in suite.hybrid_parameter_shift_error
    assert suite.pure_certificate.monotone_accepted_values
    assert suite.hybrid_certificate.monotone_accepted_values
    assert suite.pure_training.best_value < suite.pure_training.initial_value
    assert suite.hybrid_training.best_value < suite.hybrid_training.initial_value
    assert len(suite.gradient_records) == 2
    assert payload["passed"] is True
    assert payload["unsupported_scenarios"]


def test_verify_composed_objective_gradient_rejects_invalid_controls() -> None:
    objective = build_phase_control_objective(1)

    with pytest.raises(ValueError, match="finite_difference_step"):
        verify_composed_objective_gradient(objective, np.array([0.2]), finite_difference_step=0.0)

    with pytest.raises(ValueError, match="absolute_tolerance"):
        verify_composed_objective_gradient(objective, np.array([0.2]), absolute_tolerance=0.0)

    with pytest.raises(ValueError, match="params"):
        verify_composed_objective_gradient(objective, np.array([[0.2]], dtype=float))
