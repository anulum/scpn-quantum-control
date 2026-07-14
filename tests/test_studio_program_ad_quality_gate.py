# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio Program-AD quality-gate contract tests
"""Verify local and CI Studio Program-AD quality-gate parity."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from tools import program_ad_quality_gates


def _load_preflight_module() -> ModuleType:
    """Load the executable preflight module with its sibling imports available.

    Returns
    -------
    ModuleType
        Isolated preflight module used for command-contract inspection.

    Raises
    ------
    ImportError
        If Python cannot construct a loader for the preflight module.

    """
    module_name = "preflight_for_studio_program_ad_quality_tests"
    module_path = Path(__file__).resolve().parents[1] / "tools" / "preflight.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    tool_dir = str(module_path.parent)
    inserted_tool_dir = tool_dir not in sys.path
    if inserted_tool_dir:
        sys.path.insert(0, tool_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted_tool_dir:
            sys.path.remove(tool_dir)
    return module


_preflight = _load_preflight_module()


def _workflow_step_paths(workflow: str, step_name: str) -> list[str]:
    """Extract repository paths from one CI workflow step.

    Parameters
    ----------
    workflow
        Complete CI workflow text.
    step_name
        Exact GitHub Actions step name delimiting the command block.

    Returns
    -------
    list[str]
        Repository paths in their declared command order.

    """
    block_start = workflow.index(f"      - name: {step_name}")
    block_end = workflow.index("\n      - name:", block_start + 1)
    return [
        line.strip()
        for line in workflow[block_start:block_end].splitlines()
        if line.strip().startswith(("src/", "tools/", "tests/"))
    ]


def test_static_gates_include_studio_program_ad_quality_ratchets() -> None:
    """Local static gates must type and document every owner-policy file."""
    gate_map = dict(_preflight.STATIC_GATES)
    strict_cmd = gate_map["mypy-strict-studio-program-ad"]
    docstring_cmd = gate_map["ruff D studio-program-ad quality ratchet"]
    cohort = _preflight.STUDIO_PROGRAM_AD_QUALITY_RATCHET

    assert strict_cmd[:4] == [_preflight._PY, "-m", "mypy", "--strict"]
    assert "--explicit-package-bases" in strict_cmd
    assert strict_cmd[-len(cohort) :] == cohort
    assert "--isolated" in docstring_cmd
    assert "D,D413" in docstring_cmd
    assert 'lint.pydocstyle.convention = "numpy"' in docstring_cmd
    assert docstring_cmd[-len(cohort) :] == cohort


def test_default_preflight_has_exact_studio_program_ad_coverage() -> None:
    """Default local coverage must enforce the Python owner at exactly 100%."""
    gate_map = dict(_preflight.STUDIO_PROGRAM_AD_COVERAGE_GATES)
    run_cmd = gate_map["studio Program-AD focused coverage"]
    report_cmd = gate_map["studio Program-AD exact coverage threshold"]
    cohort = _preflight.STUDIO_PROGRAM_AD_COVERAGE_COHORT
    data_file = _preflight.STUDIO_PROGRAM_AD_COVERAGE_DATA_FILE

    assert run_cmd[:4] == [_preflight._PY, "-m", "coverage", "run"]
    assert "--rcfile=/dev/null" in run_cmd
    assert "--branch" in run_cmd
    assert run_cmd[-len(cohort) :] == cohort
    assert f"--data-file={data_file}" in run_cmd
    assert report_cmd[:4] == [_preflight._PY, "-m", "coverage", "report"]
    assert "--rcfile=/dev/null" in report_cmd
    assert "--precision=2" in report_cmd
    assert "--fail-under=100" in report_cmd
    assert "--include=*/program_ad_replay_artifact.py" in report_cmd
    assert f"--data-file={data_file}" in report_cmd


def test_ci_and_preflight_share_studio_program_ad_cohorts() -> None:
    """CI and local gates must preserve the exact replay-owner contracts."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    quality_steps = (
        "Type-check Studio Program-AD replay quality cohort",
        "Ruff NumPy docstrings for Studio Program-AD replay quality cohort",
    )

    for step_name in quality_steps:
        assert _workflow_step_paths(workflow, step_name) == (
            _preflight.STUDIO_PROGRAM_AD_QUALITY_RATCHET
        )

    ci_coverage_paths = _workflow_step_paths(
        workflow,
        "Run Studio Program-AD focused coverage",
    )
    assert ci_coverage_paths == _preflight.STUDIO_PROGRAM_AD_COVERAGE_COHORT
    assert "Enforce Studio Program-AD exact coverage" in workflow
    assert "--include=*/program_ad_replay_artifact.py" in workflow
    assert "needs['studio-program-ad-quality'].result" in workflow
    coverage_job = workflow[
        workflow.index("  studio-program-ad-quality:") : workflow.index(
            "\n  whole-program-trace-value-quality:"
        )
    ]
    native_env = program_ad_quality_gates.STUDIO_PROGRAM_AD_REQUIRE_NATIVE_ENV
    assert f'{native_env}: "1"' in coverage_job


def test_aggregate_owner_cannot_restore_the_native_collection_skip() -> None:
    """The aggregate owner must reach its seam before optional engine selection."""
    owner = Path(program_ad_quality_gates.STUDIO_PROGRAM_AD_COVERAGE_COHORT[0])
    source = owner.read_text(encoding="utf-8")

    assert 'pytest.importorskip("scpn_quantum_engine"' not in source


def test_local_studio_program_ad_runtime_gates_match_ci() -> None:
    """Local gates must retain Rust, WASM, TypeScript, and browser coverage."""
    runtime = dict(_preflight.STUDIO_PROGRAM_AD_RUNTIME_GATES)
    rust_test = runtime["studio Program-AD Rust kernel tests"]
    wasm_build = runtime["studio Program-AD WASM release build"]
    typecheck = runtime["studio Program-AD browser strict typecheck"]
    browser_test = _preflight.STUDIO_PROGRAM_AD_BROWSER_TEST_GATE[1]
    browser_coverage = _preflight.STUDIO_PROGRAM_AD_BROWSER_COVERAGE_GATE[1]
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert rust_test[0] == _preflight._CARGO
    assert rust_test[1:3] == ["test", "--locked"]
    assert "scpn_quantum_engine/studio_program_ad_wasm/Cargo.toml" in rust_test
    assert wasm_build[0] == _preflight._CARGO
    assert "wasm32-unknown-unknown" in wasm_build
    assert typecheck == [_preflight._PNPM, "--dir", "studio-web", "typecheck"]
    browser_prefix = [
        _preflight._PNPM,
        "--dir",
        "studio-web",
        "exec",
        "vitest",
        "run",
    ]
    assert browser_test[:6] == browser_prefix
    assert browser_coverage[:6] == browser_prefix
    test_count = len(_preflight.STUDIO_PROGRAM_AD_BROWSER_TESTS)
    assert browser_test[6 : 6 + test_count] == _preflight.STUDIO_PROGRAM_AD_BROWSER_TESTS
    assert browser_coverage[6 : 6 + test_count] == (_preflight.STUDIO_PROGRAM_AD_BROWSER_TESTS)
    assert "--coverage.thresholds.statements=100" in browser_coverage
    assert "--coverage.thresholds.branches=100" in browser_coverage
    assert "--coverage.thresholds.functions=100" in browser_coverage
    assert "--coverage.thresholds.lines=100" in browser_coverage
    assert "Enforce Program-AD browser owner exact coverage (ST-12)" in workflow
    assert "--coverage.include=src/panel/programAd.ts" in workflow
    assert "--coverage.include=src/panel/ProgramADReplayCard.tsx" in workflow
