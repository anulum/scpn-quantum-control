# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — benchmark CLI quality-gate tests
"""Lock the benchmark CLI quality gate into preflight and CI."""

from pathlib import Path

from tools import bench_cli_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert gates["mypy-strict-bench-cli-quality"][5:] == quality_gates.BENCH_CLI_QUALITY_RATCHET
    docstring_gate = gates["ruff D bench-cli quality ratchet"]
    assert "D,D413" in docstring_gate
    assert docstring_gate[-4:] == quality_gates.BENCH_CLI_DOCSTRING_RATCHET


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["bench-cli focused coverage"]
    report = gates["bench-cli exact coverage threshold"]
    assert "--branch" in run
    assert run[-3:] == quality_gates.BENCH_CLI_COVERAGE_COHORT
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/bench_cli.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.BENCH_CLI_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(BENCH_CLI_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_benchmark_cli_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  bench-cli-quality:")
    end = workflow.index("\n\n  governed-route-matrix-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.BENCH_CLI_QUALITY_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert "bench-cli-quality" in workflow[workflow.index("  ci-gate:") :]
