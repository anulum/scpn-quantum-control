# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR whole-program native quality-gate tests
"""Lock the MLIR whole-program native quality gate into preflight and CI."""

from pathlib import Path

from tools import mlir_whole_program_native_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete direct-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-mlir-whole-program-native-quality"][5:]
        == quality_gates.MLIR_WHOLE_PROGRAM_NATIVE_TYPING_RATCHET
    )
    ruff = gates["ruff D MLIR whole-program native quality ratchet"]
    assert (
        ruff[-len(quality_gates.MLIR_WHOLE_PROGRAM_NATIVE_DOCSTRING_RATCHET) :]
        == quality_gates.MLIR_WHOLE_PROGRAM_NATIVE_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_connected_and_exact() -> None:
    """Require real native/JIT execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["MLIR whole-program native focused coverage"]
    report = gates["MLIR whole-program native exact coverage threshold"]
    assert "--branch" in run
    assert run[-2:] == quality_gates.MLIR_WHOLE_PROGRAM_NATIVE_COVERAGE_COHORT
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/compiler/mlir_whole_program_native.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.MLIR_WHOLE_PROGRAM_NATIVE_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert "gates.extend(MLIR_WHOLE_PROGRAM_NATIVE_COVERAGE_GATES)" in Path(
        "tools/preflight.py"
    ).read_text(encoding="utf-8")


def test_ci_runs_and_aggregates_mlir_whole_program_native_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  mlir-whole-program-native-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.MLIR_WHOLE_PROGRAM_NATIVE_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert "compiler/mlir_whole_program_native.py" in block
    assert "mlir-whole-program-native-quality" in workflow[workflow.index("  ci-gate:") :]
