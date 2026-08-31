# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QRC-baseline quality-gate tests
"""Lock the QRC-baseline quality gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import qrc_baseline_quality_gates as quality_gates


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete direct-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-qrc-baseline-quality"][5:] == quality_gates.QRC_BASELINE_TYPING_RATCHET
    )
    ruff = gates["ruff D QRC-baseline quality ratchet"]
    assert (
        ruff[-len(quality_gates.QRC_BASELINE_DOCSTRING_RATCHET) :]
        == quality_gates.QRC_BASELINE_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff
    assert "lint.explicit-preview-rules = true" in ruff


def test_coverage_gate_is_isolated_connected_and_exact() -> None:
    """Require real offline QRC/surrogate execution and exact coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["QRC-baseline focused coverage"]
    report = gates["QRC-baseline exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.QRC_BASELINE_COVERAGE_COHORT) :] == (
        quality_gates.QRC_BASELINE_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    include = next(argument for argument in report if argument.startswith("--include="))
    assert "applications/qrc_baseline.py" in include
    assert "applications/quantum_reservoir.py" in include
    assert "surrogates/models.py" in include
    assert "surrogates/train.py" in include
    assert "surrogates/fidelity.py" in include


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.QRC_BASELINE_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert "gates.extend(QRC_BASELINE_COVERAGE_GATES)" in Path("tools/preflight.py").read_text(
        encoding="utf-8"
    )


def test_ci_runs_and_aggregates_qrc_baseline_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  qrc-baseline-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.QRC_BASELINE_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert "applications/qrc_baseline.py" in block
    assert "applications/quantum_reservoir.py" in block
    assert "tests/test_quantum_reservoir.py" in block
    assert "surrogates/models.py" in block
    assert "surrogates/train.py" in block
    assert "tests/test_surrogate_models.py" in block
    assert "tests/test_surrogate_train.py" in block
    assert "src/scpn_quantum_control/surrogates/fidelity.py" in block
    assert "tests/test_surrogate_fidelity.py" in block
    assert "qrc-baseline-quality" in workflow[workflow.index("  ci-gate:") :]
