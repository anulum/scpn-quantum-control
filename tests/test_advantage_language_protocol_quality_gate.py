# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — advantage-language quality-gate tests
"""Lock the advantage-language protocol gate into preflight and CI."""

from tools import advantage_language_protocol_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    scaling_source = "src/scpn_quantum_control/benchmarks/advantage_protocol.py"
    scaling_tests = {
        "tests/test_advantage_protocol.py",
        "tests/test_advantage_protocol_guards.py",
    }
    assert scaling_source in quality_gates.ADVANTAGE_LANGUAGE_PROTOCOL_QUALITY_RATCHET
    assert scaling_tests.issubset(quality_gates.ADVANTAGE_LANGUAGE_PROTOCOL_QUALITY_RATCHET)
    assert scaling_tests.issubset(quality_gates.ADVANTAGE_LANGUAGE_PROTOCOL_COVERAGE_COHORT)
    assert "*/benchmarks/advantage_protocol.py" in (
        quality_gates.ADVANTAGE_LANGUAGE_PROTOCOL_COVERAGE_INCLUDE
    )
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-advantage-language-protocol-quality"][5:]
        == quality_gates.ADVANTAGE_LANGUAGE_PROTOCOL_QUALITY_RATCHET
    )
    ruff = gates["ruff D advantage-language-protocol quality ratchet"]
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff
    assert "lint.explicit-preview-rules = true" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    assert "--branch" in gates["advantage-language-protocol focused coverage"]
    report = gates["advantage-language-protocol exact coverage threshold"]
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.ADVANTAGE_LANGUAGE_PROTOCOL_COVERAGE_INCLUDE}" in report
    assert quality_gates.ADVANTAGE_LANGUAGE_PROTOCOL_COVERAGE_DATA_FILE.startswith("/tmp/")


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.ADVANTAGE_LANGUAGE_PROTOCOL_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  advantage-language-protocol-quality:")
    end = workflow.index("\n\n  decisive-advantage-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.ADVANTAGE_LANGUAGE_PROTOCOL_QUALITY_RATCHET)
    assert "advantage-language-protocol-quality" in workflow[workflow.index("  ci-gate:") :]
