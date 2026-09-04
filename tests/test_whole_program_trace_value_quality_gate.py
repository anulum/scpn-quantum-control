# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole-program trace-value quality-gate tests
"""Lock the extracted trace-value and linalg owner into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import whole_program_trace_value_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source, workflow_path_for_job


def _step_paths(workflow: str, step_name: str) -> list[str]:
    """Return ordered repository paths from one named workflow step."""
    start = workflow.index(f"      - name: {step_name}")
    end = workflow.index("\n      - name:", start + 1)
    return [
        line.strip()
        for line in workflow[start:end].splitlines()
        if line.strip().startswith(("src/", "tests/", "tools/"))
    ]


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete preview documentation."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert gates["mypy-strict-whole-program-trace-values"][5:] == (
        quality_gates.WHOLE_PROGRAM_TRACE_VALUE_QUALITY_RATCHET
    )
    ruff = gates["ruff D whole-program trace-value quality ratchet"]
    assert ruff[-len(quality_gates.WHOLE_PROGRAM_TRACE_VALUE_QUALITY_RATCHET) :] == (
        quality_gates.WHOLE_PROGRAM_TRACE_VALUE_QUALITY_RATCHET
    )
    assert "--isolated" in ruff and "--preview" in ruff
    assert "D,D413,D417,D420" in ruff
    assert "lint.explicit-preview-rules = true" in ruff


def test_coverage_gate_reuses_execution_and_covers_program_ad_exactly() -> None:
    """Require exact trace, linalg, cumulative, effect-IR, and selection coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["whole-program trace-value focused coverage"]
    report = gates["whole-program trace-value exact coverage threshold"]
    assert "--branch" in run
    assert run[-len(quality_gates.WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_COHORT) :] == (
        quality_gates.WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_COHORT
    )
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_INCLUDE}" in report
    assert any("*/program_ad_linalg_primitives.py" in argument for argument in report)
    assert any("*/program_ad_product_primitives.py" in argument for argument in report)
    assert any("*/program_ad_reduction_primitives.py" in argument for argument in report)
    assert any("*/program_ad_cumulative_primitives.py" in argument for argument in report)
    assert any("*/program_ad_effect_ir.py" in argument for argument in report)
    assert any("*/program_ad_selection_primitives.py" in argument for argument in report)


def test_preflight_reexports_helper_defined_gates() -> None:
    """Keep extracted helper commands verbatim in preflight."""
    assert dict(preflight.WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_GATES)" in source


def test_ci_preserves_extracted_static_and_runtime_owner() -> None:
    """Keep both distributed jobs identical to the extracted helper."""
    workflow = read_ci_workflow_source()
    for step_name in (
        "Type-check whole-program trace-value quality cohort",
        "Ruff NumPy docstrings for whole-program trace-value quality cohort",
    ):
        assert _step_paths(workflow, step_name) == (
            quality_gates.WHOLE_PROGRAM_TRACE_VALUE_QUALITY_RATCHET
        )
    assert _step_paths(workflow, "Run whole-program trace-value focused coverage") == (
        quality_gates.WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_COHORT
    )
    assert quality_gates.WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_INCLUDE in workflow
    assert workflow_path_for_job("whole-program-trace-value-quality").name == (
        "ci-whole-program-trace.yml"
    )
