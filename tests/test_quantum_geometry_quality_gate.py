# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum-geometry quality-gate tests
"""Lock the quantum-geometry quality gate into preflight and CI."""

from __future__ import annotations

from pathlib import Path

from tools import preflight
from tools import quantum_geometry_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete NumPy docstrings on every owner."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-quantum-geometry-quality"][5:]
        == quality_gates.QUANTUM_GEOMETRY_TYPING_RATCHET
    )
    ruff = gates["ruff D quantum-geometry quality ratchet"]
    assert (
        ruff[-len(quality_gates.QUANTUM_GEOMETRY_DOCSTRING_RATCHET) :]
        == quality_gates.QUANTUM_GEOMETRY_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413,D417,D420" in ruff
    for source in quality_gates.QUANTUM_GEOMETRY_SOURCES:
        assert source in quality_gates.QUANTUM_GEOMETRY_TYPING_RATCHET


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require a separate data file and 100 % on the three owners."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["quantum-geometry focused coverage"]
    report = gates["quantum-geometry exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.QUANTUM_GEOMETRY_COVERAGE_COHORT) :]
        == quality_gates.QUANTUM_GEOMETRY_COVERAGE_COHORT
    )
    assert quality_gates.QUANTUM_GEOMETRY_COVERAGE_DATA_FILE.startswith("/tmp/")
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.QUANTUM_GEOMETRY_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.QUANTUM_GEOMETRY_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(QUANTUM_GEOMETRY_COVERAGE_GATES)" in source


def test_ci_runs_the_quantum_geometry_gate() -> None:
    """Keep the focused CI job with every owner, test and the exact threshold."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  quantum-geometry-quality:")
    end = workflow.index("\n\n  ", start)
    block = workflow[start:end]
    for path in [
        *quality_gates.QUANTUM_GEOMETRY_TYPING_RATCHET,
        *quality_gates.QUANTUM_GEOMETRY_COVERAGE_COHORT,
    ]:
        assert path in block
    assert "--fail-under=100" in block
    assert quality_gates.QUANTUM_GEOMETRY_COVERAGE_INCLUDE in block
