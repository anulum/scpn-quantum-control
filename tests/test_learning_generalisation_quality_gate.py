# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — learning-generalisation quality-gate tests
"""Lock the learning-generalisation quality gate into preflight and CI."""

from __future__ import annotations

from pathlib import Path

from tools import learning_generalisation_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete NumPy docstrings on every owner."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-learning-generalisation-quality"][5:]
        == quality_gates.LEARNING_GENERALISATION_TYPING_RATCHET
    )
    ruff = gates["ruff D learning-generalisation quality ratchet"]
    assert (
        ruff[-len(quality_gates.LEARNING_GENERALISATION_DOCSTRING_RATCHET) :]
        == quality_gates.LEARNING_GENERALISATION_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413,D417,D420" in ruff
    assert (
        "src/scpn_quantum_control/phase/gradient_descent.py"
        in quality_gates.LEARNING_GENERALISATION_SOURCES
    )
    assert (
        "tests/test_phase_gradient_descent.py"
        in quality_gates.LEARNING_GENERALISATION_COVERAGE_COHORT
    )
    for source in quality_gates.LEARNING_GENERALISATION_SOURCES:
        assert source in quality_gates.LEARNING_GENERALISATION_TYPING_RATCHET
    for name in (
        "contracts",
        "decoding",
        "evidence",
        "qnn_convergence",
        "qgnn_convergence",
        "qsnn_convergence",
        "suite",
    ):
        assert (
            f"src/scpn_quantum_control/ml_examples/{name}.py"
            in quality_gates.LEARNING_GENERALISATION_SOURCES
        )
    for name in ("contracts", "decoding", "evidence", "qnn", "qgnn", "qsnn"):
        assert (
            f"tests/test_ml_convergence_{name}.py"
            in quality_gates.LEARNING_GENERALISATION_COVERAGE_COHORT
        )
    for name in ("_identity", "models", "train", "fidelity"):
        assert (
            f"src/scpn_quantum_control/surrogates/{name}.py"
            in quality_gates.LEARNING_GENERALISATION_SOURCES
        )
        assert (
            f"tests/test_surrogate_{name.lstrip('_')}.py"
            in quality_gates.LEARNING_GENERALISATION_COVERAGE_COHORT
        )
    assert (
        "src/scpn_quantum_control/forecasting/multimodal_forecaster.py"
        in quality_gates.LEARNING_GENERALISATION_SOURCES
    )
    assert (
        "tests/test_multimodal_forecaster.py"
        in quality_gates.LEARNING_GENERALISATION_COVERAGE_COHORT
    )
    assert (
        "tests/test_multimodal_forecaster.py"
        in quality_gates.LEARNING_GENERALISATION_TYPING_RATCHET
    )


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require a separate data file and 100 % on the owner."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["learning-generalisation focused coverage"]
    report = gates["learning-generalisation exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.LEARNING_GENERALISATION_COVERAGE_COHORT) :]
        == quality_gates.LEARNING_GENERALISATION_COVERAGE_COHORT
    )
    assert quality_gates.LEARNING_GENERALISATION_COVERAGE_DATA_FILE.startswith("/tmp/")
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.LEARNING_GENERALISATION_COVERAGE_INCLUDE}" in report
    absent = gates["learning-generalisation SDK-absent coverage"]
    assert "run_sdk_absent_coverage" in absent[2]
    assert absent[-1] == quality_gates.LEARNING_GENERALISATION_COVERAGE_DATA_FILE


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.LEARNING_GENERALISATION_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(LEARNING_GENERALISATION_COVERAGE_GATES)" in source


def test_ci_runs_the_learning_generalisation_gate() -> None:
    """Keep the focused CI job with every owner, test and the exact threshold."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  learning-generalisation-quality:")
    end = workflow.index("\n\n  ", start)
    block = workflow[start:end]
    for path in [
        *quality_gates.LEARNING_GENERALISATION_TYPING_RATCHET,
        *quality_gates.LEARNING_GENERALISATION_COVERAGE_COHORT,
    ]:
        assert path in block
    assert "--fail-under=100" in block
    assert quality_gates.LEARNING_GENERALISATION_COVERAGE_INCLUDE in block
    assert "run_sdk_absent_coverage" in block
    assert "install-differentiable-framework-overlay" in block


def test_sdk_absent_gate_runs_real_training_and_records_its_trace(tmp_path: Path) -> None:
    """Qualify the actual isolated runtime controller through its public entry.

    Parameters
    ----------
    tmp_path
        Separate coverage destination so the child never overwrites its parent.

    """
    from coverage import CoverageData

    destination = tmp_path / "actual-absence.coverage"

    assert quality_gates.run_sdk_absent_coverage(str(destination)) == 0
    data = CoverageData(basename=str(destination))
    data.read()
    assert data.has_arcs()
    assert any(path.endswith("/ml_examples/qnn_convergence.py") for path in data.measured_files())
    assert any(
        path.endswith("/tests/ml_convergence_qnn_absence.py") for path in data.measured_files()
    )
