# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Program AD adjoint quality-gate tests
"""Lock the Program AD adjoint quality gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import program_ad_adjoint_quality_gates as quality_gates
from tools.audit_test_typing_policy import load_policy
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete connected docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-program-ad-adjoint-quality"][5:]
        == quality_gates.PROGRAM_AD_ADJOINT_TYPING_RATCHET
    )
    ruff = gates["ruff D Program AD adjoint quality ratchet"]
    assert (
        ruff[-len(quality_gates.PROGRAM_AD_ADJOINT_DOCSTRING_RATCHET) :]
        == quality_gates.PROGRAM_AD_ADJOINT_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "--preview" in ruff
    assert "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require offline Program AD adjoint execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["Program AD adjoint focused coverage"]
    report = gates["Program AD adjoint exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.PROGRAM_AD_ADJOINT_COVERAGE_COHORT) :]
        == quality_gates.PROGRAM_AD_ADJOINT_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/program_ad_adjoint.py,*/program_ad_adjoint_generation.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.PROGRAM_AD_ADJOINT_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(PROGRAM_AD_ADJOINT_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_program_ad_adjoint_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  program-ad-adjoint-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.PROGRAM_AD_ADJOINT_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.PROGRAM_AD_ADJOINT_COVERAGE_COHORT:
        assert path in block
    assert "--fail-under=100" in block
    assert "program_ad_adjoint.py" in block
    assert "program_ad_adjoint_generation.py" in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "program-ad-adjoint-quality" in aggregate
    native_build = block.index("Build native Program AD replay wheel")
    native_install = block.index("Install native Program AD replay wheel")
    execution = block.index("Run Program AD adjoint focused coverage")
    assert native_build < native_install < execution
    assert "--features extension-module" in block[native_build:native_install]
    assert "--locked" in block[native_build:native_install]
    assert "--no-deps dist/scpn_quantum_engine-*.whl" in block[native_install:execution]


def test_captured_owner_coverage_is_required_by_preflight_and_ci() -> None:
    """Require the same new-owner threshold in actual local and hosted wiring."""
    name = "Program AD captured state exact coverage threshold"
    report = dict(quality_gates.build_coverage_gates(preflight._PY))[name]
    assert dict(preflight.PROGRAM_AD_ADJOINT_COVERAGE_GATES)[name] == report
    assert "--fail-under=100" in report
    assert len(quality_gates.PROGRAM_AD_CAPTURED_SOURCES) == 10
    workflow = read_ci_workflow_source()
    block = workflow[workflow.index("  program-ad-adjoint-quality:") :]
    block = block.split("\n\n  tn-mps-baseline-design-quality:", 1)[0]
    start = block.index("Enforce captured program state exact coverage")
    assert "--fail-under=100" in block[start:]
    for source in quality_gates.PROGRAM_AD_CAPTURED_SOURCES:
        assert source in block
        assert source.rsplit("/", 1)[-1] in block[start:]
    for test in quality_gates.PROGRAM_AD_CAPTURED_TESTS:
        assert test in block[block.index("Run Program AD adjoint focused coverage") : start]


def test_effect_companions_are_required_by_execution_and_typing_cohorts() -> None:
    """Require extracted owners to keep execution and enforced typing coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    execution = gates["Program AD adjoint focused coverage"]
    report = gates["Program AD captured state exact coverage threshold"]
    static = dict(quality_gates.build_static_quality_gates("/python"))
    policy = load_policy(Path("tools/test_typing_policy.json"))
    scientific = next(
        cohort
        for cohort in policy.enforced_cohorts
        if cohort.cohort_id == "scientific_runtime_contracts"
    )
    effectful = "tests/test_effectful_program_semantics.py"
    assert effectful in execution and effectful in scientific.files
    assert effectful in static["mypy-strict-program-ad-adjoint-quality"]
    assert effectful in static["ruff D Program AD adjoint quality ratchet"]
    for module in (
        "program_ad_tape_binding",
        "program_ad_effect_dispatch",
        "program_ad_effect_values",
        "program_ad_effect_analysis",
        "program_ad_effect_source_binding",
        "program_ad_effect_call_binding",
    ):
        source = f"src/scpn_quantum_control/{module}.py"
        test = f"tests/test_{module}.py"
        assert test in execution and test in scientific.files
        assert source in static["mypy-strict-program-ad-adjoint-quality"]
        assert test in static["mypy-strict-program-ad-adjoint-quality"]
        assert source in static["ruff D Program AD adjoint quality ratchet"]
        assert test in static["ruff D Program AD adjoint quality ratchet"]
        assert any(f"*/{module}.py" in argument for argument in report)
