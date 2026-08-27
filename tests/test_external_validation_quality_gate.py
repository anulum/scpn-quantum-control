# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable external-validation quality-gate tests
"""Lock differentiable external-validation gates into preflight and CI."""

from pathlib import Path

from tools import external_validation_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert gates["mypy-strict-external-validation-quality"][5:] == (
        quality_gates.EXTERNAL_VALIDATION_QUALITY_RATCHET
    )
    assert "D,D413" in gates["ruff D external-validation quality ratchet"]


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["external-validation focused coverage"]
    report = gates["external-validation exact coverage threshold"]
    assert "--branch" in run
    assert run[-1:] == quality_gates.EXTERNAL_VALIDATION_COVERAGE_COHORT
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/differentiable_external_validation.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.EXTERNAL_VALIDATION_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text()
    start = workflow.index("  external-validation-manifest-quality:")
    end = workflow.index("\n\n  gradient-tape-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.EXTERNAL_VALIDATION_QUALITY_RATCHET)
    assert quality_gates.EXTERNAL_VALIDATION_COVERAGE_DATA_FILE in block
    assert "external-validation-manifest-quality" in workflow[workflow.index("  ci-gate:") :]
