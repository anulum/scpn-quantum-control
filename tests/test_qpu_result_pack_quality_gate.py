# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QPU result-pack quality-gate tests
"""Lock the QPU result-pack quality gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import qpu_result_pack_quality_gates as quality_gates


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete owning-cohort docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-qpu-result-pack-quality"][5:]
        == quality_gates.QPU_RESULT_PACK_TYPING_RATCHET
    )
    ruff = gates["ruff D QPU result-pack quality ratchet"]
    assert (
        ruff[-len(quality_gates.QPU_RESULT_PACK_DOCSTRING_RATCHET) :]
        == quality_gates.QPU_RESULT_PACK_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417" in ruff


def test_coverage_gate_is_isolated_connected_and_exact() -> None:
    """Require connected offline execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["QPU result-pack focused coverage"]
    report = gates["QPU result-pack exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.QPU_RESULT_PACK_COVERAGE_COHORT) :]
        == quality_gates.QPU_RESULT_PACK_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/studio/qpu_result_pack.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.QPU_RESULT_PACK_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(QPU_RESULT_PACK_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_qpu_result_pack_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  qpu-result-pack-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.QPU_RESULT_PACK_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.QPU_RESULT_PACK_COVERAGE_COHORT:
        assert path in block
    assert "PyO3/maturin-action@" in block
    assert "--fail-under=100" in block
    assert "studio/qpu_result_pack.py" in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "qpu-result-pack-quality" in aggregate
