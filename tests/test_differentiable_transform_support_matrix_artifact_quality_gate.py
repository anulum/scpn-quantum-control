# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — transform support-matrix artefact quality-gate tests
"""Lock the transform support-matrix artefact gate into preflight and CI."""

from pathlib import Path

from tools import differentiable_transform_support_matrix_artifact_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_documented_and_drift_checked() -> None:
    """Require strict typing, complete docstrings, and committed drift checks."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert gates["mypy-strict-transform-support-matrix-artifact-quality"][5:] == (
        quality_gates.TRANSFORM_SUPPORT_MATRIX_ARTIFACT_TYPING_RATCHET
    )
    ruff = gates["ruff D transform-support-matrix-artifact quality ratchet"]
    assert ruff[-len(quality_gates.TRANSFORM_SUPPORT_MATRIX_ARTIFACT_DOCSTRING_RATCHET) :] == (
        quality_gates.TRANSFORM_SUPPORT_MATRIX_ARTIFACT_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "--preview" in ruff
    assert "D,D413,D417,D420" in ruff
    assert gates["transform-support-matrix committed artefact drift"][-1] == "--check"


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require offline artefact execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["transform-support-matrix-artifact focused coverage"]
    report = gates["transform-support-matrix-artifact exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.TRANSFORM_SUPPORT_MATRIX_ARTIFACT_COVERAGE_COHORT) :] == (
        quality_gates.TRANSFORM_SUPPORT_MATRIX_ARTIFACT_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert (
        "--include=*/differentiable_transform_support_matrix_artifact.py,*/studio/support_matrix_bundle.py"
        in report
    )


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.TRANSFORM_SUPPORT_MATRIX_ARTIFACT_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(TRANSFORM_SUPPORT_MATRIX_ARTIFACT_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_transform_support_matrix_artifact_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  transform-support-matrix-artifact-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.TRANSFORM_SUPPORT_MATRIX_ARTIFACT_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.TRANSFORM_SUPPORT_MATRIX_ARTIFACT_COVERAGE_COHORT:
        assert path in block
    assert "--check" in block
    assert "--fail-under=100" in block
    assert "differentiable_transform_support_matrix_artifact.py" in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "transform-support-matrix-artifact-quality" in aggregate
