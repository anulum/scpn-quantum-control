# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto core quality-gate tests
"""Lock the Kuramoto core quality owner into preflight and CI."""

from pathlib import Path

from tools import kuramoto_core_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated complete NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-kuramoto-core-quality"][5:]
        == quality_gates.KURAMOTO_CORE_TYPING_RATCHET
    )
    docs = gates["ruff D Kuramoto-core quality ratchet"]
    assert "D,D413,D417,D420" in docs
    assert "--preview" in docs
    assert "lint.explicit-preview-rules = true" in docs
    assert quality_gates.KURAMOTO_CORE_SOURCE in quality_gates.KURAMOTO_CORE_DOCSTRING_RATCHET


def test_coverage_gate_is_connected_and_exact() -> None:
    """Require connected branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["Kuramoto-core focused coverage"]
    report = gates["Kuramoto-core exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.KURAMOTO_CORE_COVERAGE_COHORT) :] == (
        quality_gates.KURAMOTO_CORE_COVERAGE_COHORT
    )
    assert "tests/test_kuramoto_variants.py" in run
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.KURAMOTO_CORE_COVERAGE_INCLUDE}" in report
    assert quality_gates.KURAMOTO_CORE_COVERAGE_DATA_FILE.startswith("/tmp/")


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.KURAMOTO_CORE_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text()
    assert "gates.extend(KURAMOTO_CORE_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  kuramoto-core-quality:")
    end = workflow.index("\n\n  mps-evolution-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.KURAMOTO_CORE_TYPING_RATCHET)
    assert all(path in block for path in quality_gates.KURAMOTO_CORE_DOCSTRING_RATCHET)
    assert all(path in block for path in quality_gates.KURAMOTO_CORE_COVERAGE_COHORT)
    assert quality_gates.KURAMOTO_CORE_COVERAGE_INCLUDE in block
    assert "kuramoto-core-quality" in workflow[workflow.index("  ci-gate:") :]
