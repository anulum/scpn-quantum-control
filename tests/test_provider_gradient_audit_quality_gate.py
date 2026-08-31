# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — provider-gradient audit quality-gate tests
"""Lock the provider-gradient audit owner into preflight and required CI."""

from pathlib import Path

from tools import preflight
from tools import provider_gradient_audit_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and isolated complete NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-provider-gradient-audit-quality"][5:]
        == quality_gates.PROVIDER_GRADIENT_AUDIT_TYPING_RATCHET
    )
    docs = gates["ruff D provider-gradient-audit quality ratchet"]
    assert "D,D413,D417,D420" in docs
    assert "--preview" in docs
    assert "lint.explicit-preview-rules = true" in docs
    assert docs[-len(quality_gates.PROVIDER_GRADIENT_AUDIT_DOCSTRING_RATCHET) :] == (
        quality_gates.PROVIDER_GRADIENT_AUDIT_DOCSTRING_RATCHET
    )


def test_coverage_gate_runs_real_audit_suite_and_is_exact() -> None:
    """Require public audit execution and exact source branch coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["provider-gradient-audit focused coverage"]
    report = gates["provider-gradient-audit exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.PROVIDER_GRADIENT_AUDIT_COVERAGE_COHORT) :] == (
        quality_gates.PROVIDER_GRADIENT_AUDIT_COVERAGE_COHORT
    )
    assert quality_gates.PROVIDER_GRADIENT_AUDIT_COVERAGE_DATA_FILE.startswith("/tmp/")
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.PROVIDER_GRADIENT_AUDIT_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper-defined static and coverage commands verbatim in preflight."""
    assert dict(preflight.PROVIDER_GRADIENT_AUDIT_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(PROVIDER_GRADIENT_AUDIT_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_provider_gradient_audit_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  provider-gradient-audit-quality:")
    end = workflow.index("\n\n  entropy-randomness-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.PROVIDER_GRADIENT_AUDIT_TYPING_RATCHET)
    assert all(path in block for path in quality_gates.PROVIDER_GRADIENT_AUDIT_DOCSTRING_RATCHET)
    assert all(path in block for path in quality_gates.PROVIDER_GRADIENT_AUDIT_TESTS)
    assert quality_gates.PROVIDER_GRADIENT_AUDIT_COVERAGE_INCLUDE in block
    assert "provider-gradient-audit-quality" in workflow[workflow.index("  ci-gate:") :]
