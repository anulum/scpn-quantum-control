# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Gradient Audit Contract Tests
"""Tests for immutable differentiable gradient audit contracts."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control.phase.differentiable_audit as audit_module
import scpn_quantum_control.phase.differentiable_audit_contracts as contract_module


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
