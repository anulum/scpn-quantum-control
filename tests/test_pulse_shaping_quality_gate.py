# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — pulse-shaping quality-gate tests
"""Lock the pulse-shaping quality gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import pulse_shaping_quality_gates as quality_gates


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete connected docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-pulse-shaping-quality"][5:]
        == quality_gates.PULSE_SHAPING_TYPING_RATCHET
    )
    ruff = gates["ruff D pulse-shaping quality ratchet"]
    assert (
        ruff[-len(quality_gates.PULSE_SHAPING_DOCSTRING_RATCHET) :]
        == quality_gates.PULSE_SHAPING_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require offline pulse execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["pulse-shaping focused coverage"]
    report = gates["pulse-shaping exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.PULSE_SHAPING_COVERAGE_COHORT) :]
        == quality_gates.PULSE_SHAPING_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/phase/pulse_shaping.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.PULSE_SHAPING_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(PULSE_SHAPING_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_pulse_shaping_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  pulse-shaping-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.PULSE_SHAPING_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert "phase/pulse_shaping.py" in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "pulse-shaping-quality" in aggregate
