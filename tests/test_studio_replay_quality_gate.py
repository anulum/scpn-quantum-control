# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio replay quality-gate tests
"""Lock the Studio replay quality gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import studio_replay_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete direct-owner documentation."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-studio-replay-quality"][5:]
        == quality_gates.STUDIO_REPLAY_TYPING_RATCHET
    )
    ruff = gates["ruff D studio-replay quality ratchet"]
    assert (
        ruff[-len(quality_gates.STUDIO_REPLAY_DOCSTRING_RATCHET) :]
        == quality_gates.STUDIO_REPLAY_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_connected_and_exact() -> None:
    """Require offline verifier/handler/CLI execution and exact coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["studio-replay focused coverage"]
    report = gates["studio-replay exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.STUDIO_REPLAY_COVERAGE_COHORT) :]
        == quality_gates.STUDIO_REPLAY_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert any("executive_replay.py" in argument for argument in report)


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.STUDIO_REPLAY_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(STUDIO_REPLAY_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_studio_replay_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  studio-replay-quality:")
    end = workflow.index("\n\n  studio-simulation-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.STUDIO_REPLAY_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.STUDIO_REPLAY_COVERAGE_COHORT:
        assert path in block
    assert "--fail-under=100" in block
    assert "studio-replay-quality" in workflow[workflow.index("  ci-gate:") :]
