# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — typed phase-result quality-gate tests
"""Lock the typed phase-result quality gate into preflight and CI."""

from pathlib import Path

from scpn_quantum_control.ci_workflow_ownership import read_ci_job_blocks
from tools import phase_results_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import (
    REPOSITORY_ROOT,
    load_ci_workflow_policy,
    read_ci_workflow_source,
    workflow_path_for_job,
)


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete direct-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-phase-results-quality"][5:]
        == quality_gates.PHASE_RESULTS_TYPING_RATCHET
    )
    ruff = gates["ruff D typed phase-result quality ratchet"]
    assert (
        ruff[-len(quality_gates.PHASE_RESULTS_DOCSTRING_RATCHET) :]
        == quality_gates.PHASE_RESULTS_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_connected_and_exact() -> None:
    """Require connected trajectory production and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["typed phase-result focused coverage"]
    report = gates["typed phase-result exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.PHASE_RESULTS_COVERAGE_COHORT) :]
        == quality_gates.PHASE_RESULTS_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.PHASE_RESULTS_COVERAGE_INCLUDE}" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.PHASE_RESULTS_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert "gates.extend(PHASE_RESULTS_COVERAGE_GATES)" in Path("tools/preflight.py").read_text(
        encoding="utf-8"
    )


def test_ci_runs_and_aggregates_phase_results_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  phase-results-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in (
        quality_gates.PHASE_RESULTS_DOCSTRING_RATCHET + quality_gates.PHASE_RESULTS_COVERAGE_COHORT
    ):
        assert path in block
    assert "--fail-under=100" in block
    assert quality_gates.PHASE_RESULTS_COVERAGE_INCLUDE in block
    assert "phase-results-quality" in workflow[workflow.index("  ci-gate:") :]


def test_ci_provisions_the_native_reference_wheel_before_execution() -> None:
    """Build and install the current ABI wheel before real FFI reference tests."""
    policy = load_ci_workflow_policy()
    category = next(row for row in policy["categories"] if "phase-results-quality" in row["jobs"])
    assert "native-build" in category["caller_needs"]
    coordinator = (REPOSITORY_ROOT / policy["coordinator"]).read_text(encoding="utf-8")
    caller = read_ci_job_blocks(coordinator)[category["id"]]
    assert "needs: [static-analysis, native-build]" in caller

    source = workflow_path_for_job("phase-results-quality").read_text(encoding="utf-8")
    job = read_ci_job_blocks(source)["phase-results-quality"]
    assert 'python-version: "3.12"' in job
    assert "name: scpn-quantum-engine-3.12" in job
    download = job.index("actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c")
    install = job.index(
        "python -m pip install --force-reinstall --no-deps dist/scpn_quantum_engine-*.whl"
    )
    assert download < install < job.index("python -m coverage run")
    assert "tests/test_quantum_reference_evolution.py" in job

    producer = workflow_path_for_job("native-wheels").read_text(encoding="utf-8")
    assert 'python-version: ["3.11", "3.12", "3.13"]' in producer
    assert "name: scpn-quantum-engine-${{ matrix.python-version }}" in producer
    assert "path: dist/scpn_quantum_engine-*.whl" in producer
