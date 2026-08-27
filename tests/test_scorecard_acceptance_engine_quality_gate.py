# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — scorecard acceptance engine quality-gate tests
"""Lock the scorecard-acceptance-engine gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import scorecard_acceptance_engine_quality_gates as quality_gates


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-scorecard-acceptance-engine-quality"][5:]
        == quality_gates.SCORECARD_ACCEPTANCE_ENGINE_QUALITY_RATCHET
    )
    assert "D,D413" in gates["ruff D scorecard-acceptance-engine quality ratchet"]


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["scorecard-acceptance-engine focused coverage"]
    assert "--branch" in run
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in gates["scorecard-acceptance-engine exact coverage threshold"]


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.SCORECARD_ACCEPTANCE_ENGINE_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text()
    start = workflow.index("  scorecard-acceptance-engine-quality:")
    end = workflow.index("\n\n  unsuitable-scenario-registry-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.SCORECARD_ACCEPTANCE_ENGINE_QUALITY_RATCHET)
    assert "scorecard-acceptance-engine-quality" in workflow[workflow.index("  ci-gate:") :]
