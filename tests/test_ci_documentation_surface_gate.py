# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- CI documentation surface gate contract
"""Static contract for the CI documentation-surface gate."""

from __future__ import annotations

from pathlib import Path


def test_ci_lint_job_gates_documentation_surface() -> None:
    """CI must fail if repository documentation-surface findings reappear."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Audit documentation surface" in workflow
    assert "python tools/audit_documentation_surface.py" in workflow
    assert "--allowlist tools/documentation_surface_allowlist.json" in workflow
    assert "--fail-on-findings" in workflow


def test_ci_gates_differentiable_strict_mypy_ratchet() -> None:
    """CI must enforce strict mypy on promoted differentiable modules."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "mypy --strict" in workflow
    assert "src/scpn_quantum_control/differentiable.py" in workflow
    assert "src/scpn_quantum_control/differentiable_claim_ledger.py" in workflow
    assert "src/scpn_quantum_control/differentiable_api.py" in workflow
    assert "src/scpn_quantum_control/benchmarks/differentiable_programming.py" in workflow
    assert "src/scpn_quantum_control/differentiable_external_validation.py" in workflow
    assert "src/scpn_quantum_control/differentiable_framework_overlay.py" in workflow
    assert "src/scpn_quantum_control/differentiable_module_hardening_audit.py" in workflow
    assert "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py" in workflow
    assert "src/scpn_quantum_control/benchmarks/differentiable_evidence.py" in workflow
    assert "src/scpn_quantum_control/phase/differentiable_readiness.py" in workflow
    assert "src/scpn_quantum_control/phase/differentiable_audit.py" in workflow
    assert "src/scpn_quantum_control/phase/gradient_support_matrix.py" in workflow
    assert "src/scpn_quantum_control/phase/provider_gradient.py" in workflow
    assert "src/scpn_quantum_control/phase/hardware_gradient_policy.py" in workflow
    assert "src/scpn_quantum_control/phase/provider_gradient_audit.py" in workflow
    assert "src/scpn_quantum_control/phase/hardware_gradient_publication.py" in workflow
    assert "src/scpn_quantum_control/phase/provider_hardware_gradient_audit.py" in workflow
