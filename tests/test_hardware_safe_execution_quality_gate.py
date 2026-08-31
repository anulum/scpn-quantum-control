# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware-safe execution quality-gate tests
"""Lock the hardware-safe execution gate into preflight and CI."""

from pathlib import Path

from tools import hardware_safe_execution_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    capability_source = "src/scpn_quantum_control/hardware/feedback_capability_probe.py"
    capability_tests = {
        "tests/test_feedback_capability_probe.py",
        "tests/test_feedback_capability_probe_branch.py",
    }
    assert capability_source in quality_gates.HARDWARE_SAFE_EXECUTION_TYPING_RATCHET
    assert capability_source in quality_gates.HARDWARE_SAFE_EXECUTION_QUALITY_RATCHET
    assert capability_tests.issubset(quality_gates.HARDWARE_SAFE_EXECUTION_QUALITY_RATCHET)
    assert capability_tests.issubset(quality_gates.HARDWARE_SAFE_EXECUTION_COVERAGE_COHORT)
    assert "*/hardware/feedback_capability_probe.py" in (
        quality_gates.HARDWARE_SAFE_EXECUTION_COVERAGE_INCLUDE
    )
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-hardware-safe-execution-quality"][5:]
        == quality_gates.HARDWARE_SAFE_EXECUTION_TYPING_RATCHET
    )
    docs = gates["ruff D hardware-safe-execution quality ratchet"]
    assert "D,D413,D417,D420" in docs
    assert "--preview" in docs
    assert "lint.explicit-preview-rules = true" in docs


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["hardware-safe-execution focused coverage"]
    report = gates["hardware-safe-execution exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.HARDWARE_SAFE_EXECUTION_COVERAGE_COHORT) :] == (
        quality_gates.HARDWARE_SAFE_EXECUTION_COVERAGE_COHORT
    )
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.HARDWARE_SAFE_EXECUTION_COVERAGE_INCLUDE}" in report
    assert quality_gates.HARDWARE_SAFE_EXECUTION_COVERAGE_DATA_FILE.startswith("/tmp/")


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.HARDWARE_SAFE_EXECUTION_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text()
    start = workflow.index("  hardware-safe-execution-quality:")
    end = workflow.index("\n\n  decisive-advantage-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.HARDWARE_SAFE_EXECUTION_QUALITY_RATCHET)
    assert quality_gates.HARDWARE_SAFE_EXECUTION_COVERAGE_INCLUDE in block
    assert "hardware-safe-execution-quality" in workflow[workflow.index("  ci-gate:") :]
