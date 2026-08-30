# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — generalised parameter-shift quality-gate tests
"""Lock finite-spectrum parameter-shift quality commands into preflight and CI."""

from pathlib import Path

from tools import generalised_parameter_shift_quality_gates as quality_gates
from tools import preflight


def test_helper_builds_strict_preview_and_exact_gates() -> None:
    """Require strict preview docs, the real suite, and exact source coverage."""
    static = dict(quality_gates.build_static_quality_gates("/python"))
    docs = static["ruff D generalised-parameter-shift quality ratchet"]
    coverage = dict(quality_gates.build_coverage_gates("/python"))
    run = coverage["generalised-parameter-shift focused coverage"]
    report = coverage["generalised-parameter-shift exact coverage threshold"]
    assert "--preview" in docs
    assert "D,D413,D417,D420" in docs
    assert "lint.explicit-preview-rules = true" in docs
    assert run[-1:] == quality_gates.GENERALISED_PARAMETER_SHIFT_COVERAGE_COHORT
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.GENERALISED_PARAMETER_SHIFT_COVERAGE_INCLUDE}" in report
    assert quality_gates.GENERALISED_PARAMETER_SHIFT_COVERAGE_DATA_FILE.startswith("/tmp/")


def test_preflight_uses_helper_commands_verbatim() -> None:
    """Keep helper-defined commands exact in preflight."""
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert dict(preflight.GENERALISED_PARAMETER_SHIFT_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )


def test_ci_runs_and_aggregates_generalised_parameter_shift() -> None:
    """Keep the dedicated job and transitive aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  generalised-parameter-shift-quality:")
    end = workflow.index("\n\n  differentiable-parameter-shift-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.GENERALISED_PARAMETER_SHIFT_QUALITY_RATCHET:
        assert path in block
    for path in quality_gates.GENERALISED_PARAMETER_SHIFT_COVERAGE_COHORT:
        assert path in block
    assert quality_gates.GENERALISED_PARAMETER_SHIFT_COVERAGE_INCLUDE in block
    parameter_start = workflow.index("  differentiable-parameter-shift-quality:", end)
    parameter_block = workflow[
        parameter_start : workflow.index("\n\n  varqite-quality:", parameter_start)
    ]
    assert "needs: [lint, generalised-parameter-shift-quality]" in parameter_block
