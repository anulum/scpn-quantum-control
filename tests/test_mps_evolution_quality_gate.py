# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MPS-evolution quality-gate tests
"""Lock the MPS-evolution quality gate into preflight and CI."""

from pathlib import Path

from tools import mps_evolution_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete direct-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-mps-evolution-quality"][5:]
        == quality_gates.MPS_EVOLUTION_TYPING_RATCHET
    )
    ruff = gates["ruff D MPS-evolution quality ratchet"]
    assert (
        ruff[-len(quality_gates.MPS_EVOLUTION_DOCSTRING_RATCHET) :]
        == quality_gates.MPS_EVOLUTION_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_connected_and_exact() -> None:
    """Require connected real tensor-network execution and exact coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["MPS-evolution focused coverage"]
    report = gates["MPS-evolution exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.MPS_EVOLUTION_COVERAGE_COHORT) :]
        == quality_gates.MPS_EVOLUTION_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/phase/mps_evolution.py,*/phase/contraction_optimiser.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.MPS_EVOLUTION_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(MPS_EVOLUTION_COVERAGE_GATES)" in source


def test_ci_runs_real_quimb_and_aggregates_mps_gate() -> None:
    """Keep a real pinned-quimb CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  mps-evolution-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.MPS_EVOLUTION_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.MPS_EVOLUTION_COVERAGE_COHORT:
        assert path in block
    assert "python -m pip install --require-hashes" in block
    assert "requirements-ci-quimb-py312-linux.txt" in block
    assert 'python -m pip install "quimb==1.13.0"' not in block
    assert "python -c \"import quimb; assert quimb.__version__ == '1.13.0'\"" in block
    assert "--fail-under=100" in block
    assert "phase/mps_evolution.py" in block
    assert "phase/contraction_optimiser.py" in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "mps-evolution-quality" in aggregate
