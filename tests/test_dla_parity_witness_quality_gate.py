# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA parity-witness quality-gate tests
"""Lock the DLA parity-witness quality gate into preflight and CI."""

from pathlib import Path

from tools import dla_parity_witness_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete mixed-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-dla-parity-witness-quality"][5:]
        == quality_gates.DLA_PARITY_WITNESS_TYPING_RATCHET
    )
    ruff = gates["ruff D DLA-parity-witness quality ratchet"]
    assert (
        ruff[-len(quality_gates.DLA_PARITY_WITNESS_DOCSTRING_RATCHET) :]
        == quality_gates.DLA_PARITY_WITNESS_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D107,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_connected_and_exact() -> None:
    """Require real observable execution and exact parity-witness coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["DLA-parity-witness focused coverage"]
    report = gates["DLA-parity-witness exact coverage threshold"]
    assert "--branch" in run
    assert run[-1:] == quality_gates.DLA_PARITY_WITNESS_COVERAGE_COHORT
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/analysis/dla_parity_witness.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.DLA_PARITY_WITNESS_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(DLA_PARITY_WITNESS_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_dla_parity_witness_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  dla-parity-witness-quality:")
    end = workflow.index("\n\n  phase-results-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.DLA_PARITY_WITNESS_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert "analysis/dla_parity_witness.py" in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "dla-parity-witness-quality" in aggregate
