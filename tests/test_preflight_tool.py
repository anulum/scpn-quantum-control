# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for local preflight tool
"""Tests for the local CI preflight helper."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from shutil import which
from types import ModuleType

import pytest


def _load_tool_module(module_name: str, filename: str) -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "tools" / filename
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


_preflight = _load_tool_module("preflight_for_tests", "preflight.py")


def test_static_gates_include_documentation_surface_gate() -> None:
    """Preflight must run the documentation-surface audit."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}

    assert "documentation-surface" in gate_map
    assert "tools/audit_documentation_surface.py" in gate_map["documentation-surface"]
    assert "--allowlist" in gate_map["documentation-surface"]
    assert "tools/documentation_surface_allowlist.json" in gate_map["documentation-surface"]
    assert "--fail-on-findings" in gate_map["documentation-surface"]


def test_static_gates_include_module_size_policy_and_typing() -> None:
    """Preflight must keep the oversized-code registry current and typed."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}

    assert "tools/audit_module_size_policy.py" in gate_map["module-size-policy"]
    strict_cmd = gate_map["mypy-strict-module-size-policy"]
    assert "--strict" in strict_cmd
    assert "tools/audit_module_size_policy.py" in strict_cmd


def test_static_gates_include_test_typing_policy_and_tool_typing() -> None:
    """Preflight must execute the test cohort and keep its audit tool strict."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}

    assert gate_map["test-typing-policy"][-1] == "tools/audit_test_typing_policy.py"
    strict_cmd = gate_map["mypy-strict-test-typing-policy"]
    assert "--strict" in strict_cmd
    assert strict_cmd[-1] == "tools/audit_test_typing_policy.py"


def test_static_gates_include_coverage_policy_and_tool_typing() -> None:
    """Preflight must validate the branch policy and keep its audit strict."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}

    policy_cmd = gate_map["coverage-policy"]
    assert "tools/audit_coverage_policy.py" in policy_cmd
    assert "--validate-policy" in policy_cmd
    strict_cmd = gate_map["mypy-strict-coverage-policy"]
    assert "--strict" in strict_cmd
    assert strict_cmd[-1] == "tools/audit_coverage_policy.py"


def test_static_gates_include_coverage_debt_register_and_tool_typing() -> None:
    """Preflight must audit the debt register and keep its generator strict."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}

    assert gate_map["coverage-debt"][-1] == "tools/audit_coverage_debt.py"
    strict_cmd = gate_map["mypy-strict-coverage-debt"]
    assert "--strict" in strict_cmd
    assert strict_cmd[-1] == "tools/audit_coverage_debt.py"


def test_static_gates_include_manifest_scoped_rustfmt() -> None:
    """Preflight must reject formatting drift across the Rust engine crate."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}
    command = gate_map["rustfmt"]

    # ``_CARGO`` resolves to an absolute path only when a Rust toolchain is on
    # PATH; the reproduction image ships no cargo, so preflight falls back to the
    # bare ``"cargo"`` name. Assert the exact resolved path when cargo exists and
    # the bare fallback otherwise.
    cargo = which("cargo")
    if cargo is None:
        assert command[0] == "cargo"
    else:
        assert command[0] == cargo
        assert Path(command[0]).is_absolute()
    assert command[1:] == [
        "fmt",
        "--manifest-path",
        "scpn_quantum_engine/Cargo.toml",
        "--all",
        "--",
        "--check",
    ]


def test_static_gates_include_external_validation_manifest_audit() -> None:
    """Preflight must check both external-validation manifest pairs strictly."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}

    assert gate_map["differentiable-external-validation"][-1] == (
        "tools/check_differentiable_external_validation.py"
    )
    strict_cmd = gate_map["mypy-strict-differentiable-external-validation"]
    assert "--strict" in strict_cmd
    assert strict_cmd[-1] == "tools/check_differentiable_external_validation.py"


def test_static_gates_include_generated_differentiable_support_matrix() -> None:
    """The local static gate must mirror page drift and strict typing checks."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}

    assert gate_map["differentiable-support-matrix-page"][-2:] == [
        "tools/differentiable_support_matrix_page.py",
        "--check",
    ]
    strict_cmd = gate_map["mypy-strict-differentiable-support-matrix-page"]
    assert "--strict" in strict_cmd
    assert "--explicit-package-bases" in strict_cmd
    assert "tools/differentiable_support_matrix_page.py" in strict_cmd
    assert "tests/test_differentiable_support_matrix_page.py" in strict_cmd


def test_static_gates_include_generated_differentiable_reviewer_evidence() -> None:
    """The local static gate must enforce reviewer-page drift and strict typing."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}

    assert gate_map["differentiable-reviewer-evidence-page"][-2:] == [
        "tools/differentiable_reviewer_evidence_page.py",
        "--check",
    ]
    strict_cmd = gate_map["mypy-strict-differentiable-reviewer-evidence-page"]
    assert "--strict" in strict_cmd
    assert "--explicit-package-bases" in strict_cmd
    assert "tools/differentiable_reviewer_evidence_catalog.py" in strict_cmd
    assert "tools/differentiable_reviewer_evidence_page.py" in strict_cmd
    assert "tests/test_differentiable_reviewer_evidence_page.py" in strict_cmd


def test_static_gates_include_differentiable_docstring_ratchet() -> None:
    """Differentiable docstring-clean modules must stay under Ruff D."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}
    docstring_cmd = gate_map["ruff D differentiable module-hardening ratchet"]

    assert "--isolated" in docstring_cmd
    assert "--select" in docstring_cmd
    assert "D,D413" in docstring_cmd
    assert "--config" in docstring_cmd
    assert 'lint.pydocstyle.convention = "numpy"' in docstring_cmd
    assert "src/scpn_quantum_control/differentiable_architecture_map.py" in docstring_cmd
    assert "src/scpn_quantum_control/differentiable_claim_ledger.py" in docstring_cmd
    assert "src/scpn_quantum_control/differentiable_claim_rendering.py" in docstring_cmd
    assert "src/scpn_quantum_control/differentiable_dependency_environment_map.py" in (
        docstring_cmd
    )
    assert "src/scpn_quantum_control/differentiable_baseline_scorecard.py" in docstring_cmd
    assert "src/scpn_quantum_control/differentiable_external_validation.py" in (docstring_cmd)
    assert "src/scpn_quantum_control/differentiable_finite_difference.py" in docstring_cmd
    assert "src/scpn_quantum_control/differentiable_module_hardening_audit.py" in (docstring_cmd)
    assert "src/scpn_quantum_control/program_ad_alias_contracts.py" in docstring_cmd
    assert "src/scpn_quantum_control/program_ad_registry.py" in docstring_cmd
    assert "src/scpn_quantum_control/studio/evidence_bundle.py" in docstring_cmd
    assert "src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py" in (
        docstring_cmd
    )
    assert "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py" in (
        docstring_cmd
    )
    assert "tests/test_differentiable_external_validation.py" in docstring_cmd
    assert "tests/test_differentiable_finite_difference.py" in docstring_cmd
    assert "tests/test_differentiable_module_hardening_audit.py" in docstring_cmd
    assert "tests/test_program_ad_alias_contracts.py" in docstring_cmd
    assert "tests/test_program_ad_registry.py" in docstring_cmd
    assert "tests/test_differentiable_hardening_gate.py" in docstring_cmd
    assert "tools/differentiable_support_matrix_page.py" in docstring_cmd
    assert "tests/test_differentiable_support_matrix_page.py" in docstring_cmd
    assert "tools/differentiable_reviewer_evidence_catalog.py" in docstring_cmd
    assert "tools/differentiable_reviewer_evidence_page.py" in docstring_cmd
    assert "tests/test_differentiable_reviewer_evidence_page.py" in docstring_cmd


def test_static_gates_include_decisive_advantage_quality_ratchets() -> None:
    """The decisive benchmark source and owner test must stay typed and documented."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}
    strict_cmd = gate_map["mypy-strict-decisive-advantage-quality"]
    docstring_cmd = gate_map["ruff D decisive-advantage quality ratchet"]
    cohort = _preflight._decisive_advantage_quality_gates.DECISIVE_ADVANTAGE_QUALITY_RATCHET
    assert "--strict" in strict_cmd
    assert "--explicit-package-bases" in strict_cmd
    assert strict_cmd[-len(cohort) :] == cohort
    assert "--isolated" in docstring_cmd
    assert "D,D413" in docstring_cmd
    assert 'lint.pydocstyle.convention = "numpy"' in docstring_cmd
    assert docstring_cmd[-len(cohort) :] == cohort
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    for step_name in (
        "Type-check decisive-advantage quality cohort",
        "Ruff NumPy docstrings for decisive-advantage quality cohort",
    ):
        block_start = workflow.index(f"      - name: {step_name}")
        block_end = workflow.index("\n      - name:", block_start + 1)
        ci_paths = [
            line.strip()
            for line in workflow[block_start:block_end].splitlines()
            if line.strip().startswith(("src/", "tests/", "tools/"))
        ]
        assert ci_paths == cohort
    assert "needs['decisive-advantage-quality'].result" in workflow


def test_decisive_advantage_coverage_gate_is_exact_and_focused() -> None:
    """The decisive benchmark must retain exact statement and branch coverage."""
    focused_name, focused_cmd = _preflight.DECISIVE_ADVANTAGE_COVERAGE_GATES[0]
    threshold_name, threshold_cmd = _preflight.DECISIVE_ADVANTAGE_COVERAGE_GATES[1]
    assert focused_name == "decisive-advantage focused coverage"
    assert "--branch" in focused_cmd
    assert focused_cmd[-1:] == (
        _preflight._decisive_advantage_quality_gates.DECISIVE_ADVANTAGE_COVERAGE_COHORT
    )
    assert threshold_name == "decisive-advantage exact coverage threshold"
    assert "--fail-under=100" in threshold_cmd
    assert "--include=*/decisive_advantage_protocol.py" in threshold_cmd
    gate_names = {name for name, _cmd in _preflight.STATIC_GATES}
    assert "mypy-strict-program-ad-array-indexing-quality" in gate_names
    array_gates = _preflight.PROGRAM_AD_ARRAY_INDEXING_COVERAGE_GATES
    assert "--branch" in array_gates[0][1]
    assert "--fail-under=100" in array_gates[1][1]
    assert "--include=*/program_ad_array_indexing.py" in array_gates[1][1]
    assert "mypy-strict-differentiable-scalar-kernels-quality" in gate_names
    assert "ruff D differentiable-scalar-kernels quality ratchet" in gate_names


def test_static_gates_include_realtime_runtime_quality_ratchets() -> None:
    """Realtime runtime source and focused tests must stay typed and documented."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}
    strict_cmd = gate_map["mypy-strict-realtime-runtime"]
    docstring_cmd = gate_map["ruff D realtime-runtime quality ratchet"]
    assert "--strict" in strict_cmd
    assert "--explicit-package-bases" in strict_cmd
    assert strict_cmd[-3:] == _preflight.REALTIME_RUNTIME_QUALITY_RATCHET
    assert "--isolated" in docstring_cmd
    assert "D,D413" in docstring_cmd
    assert 'lint.pydocstyle.convention = "numpy"' in docstring_cmd
    assert docstring_cmd[-3:] == _preflight.REALTIME_RUNTIME_QUALITY_RATCHET


def test_ci_and_preflight_share_realtime_runtime_quality_cohort() -> None:
    """CI and local static gates must enforce the same realtime file order."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    step_names = (
        "Type-check realtime runtime quality cohort",
        "Ruff NumPy docstrings for realtime runtime quality cohort",
    )

    for step_name in step_names:
        block_start = workflow.index(f"      - name: {step_name}")
        block_end = workflow.index("\n      - name:", block_start + 1)
        ci_paths = [
            line.strip()
            for line in workflow[block_start:block_end].splitlines()
            if line.strip().startswith(("src/", "tests/"))
        ]
        assert ci_paths == _preflight.REALTIME_RUNTIME_QUALITY_RATCHET


def test_static_gates_include_whole_program_trace_value_quality_ratchets() -> None:
    """Trace-value runtime and focused owners must stay typed and documented."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}
    strict_cmd = gate_map["mypy-strict-whole-program-trace-values"]
    docstring_cmd = gate_map["ruff D whole-program trace-value quality ratchet"]
    cohort = _preflight.WHOLE_PROGRAM_TRACE_VALUE_QUALITY_RATCHET

    assert "--strict" in strict_cmd
    assert "--explicit-package-bases" in strict_cmd
    assert strict_cmd[-len(cohort) :] == cohort
    assert "--isolated" in docstring_cmd
    assert "D,D413" in docstring_cmd
    assert 'lint.pydocstyle.convention = "numpy"' in docstring_cmd
    assert docstring_cmd[-len(cohort) :] == cohort


def test_static_gates_include_phase_qnode_affinity_quality_ratchets() -> None:
    """Affinity source, CLI tools, and tests must stay typed and documented."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}
    strict_cmd = gate_map["mypy-strict-phase-qnode-affinity"]
    docstring_cmd = gate_map["ruff D phase-qnode-affinity quality ratchet"]
    cohort = _preflight.PHASE_QNODE_AFFINITY_QUALITY_RATCHET

    assert "--strict" in strict_cmd
    assert "--explicit-package-bases" in strict_cmd
    assert strict_cmd[-len(cohort) :] == cohort
    assert "--isolated" in docstring_cmd
    assert "D,D413" in docstring_cmd
    assert 'lint.pydocstyle.convention = "numpy"' in docstring_cmd
    assert docstring_cmd[-len(cohort) :] == cohort


def test_static_gates_include_phase_qnode_vector_quality_ratchets() -> None:
    """Vector QNode runtime and tests must stay strictly typed and documented."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}
    strict_cmd = gate_map["mypy-strict-phase-qnode-vector"]
    docstring_cmd = gate_map["ruff D phase-qnode-vector quality ratchet"]
    cohort = _preflight.PHASE_QNODE_VECTOR_QUALITY_RATCHET

    assert "--strict" in strict_cmd
    assert "--explicit-package-bases" in strict_cmd
    assert strict_cmd[-len(cohort) :] == cohort
    assert "--isolated" in docstring_cmd
    assert "D,D413" in docstring_cmd
    assert 'lint.pydocstyle.convention = "numpy"' in docstring_cmd
    assert docstring_cmd[-len(cohort) :] == cohort


def test_static_gates_include_mlir_leaf_quality_ratchets() -> None:
    """MLIR leaves and their real integration cohort must stay fully typed and documented."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}
    strict_cmd = gate_map["mypy-strict-mlir-leaf-quality"]
    docstring_cmd = gate_map["ruff D MLIR-leaf quality ratchet"]
    cohort = _preflight.MLIR_LEAF_QUALITY_RATCHET

    assert "--strict" in strict_cmd
    assert strict_cmd[-len(cohort) :] == cohort
    assert "--isolated" in docstring_cmd
    assert "D,D413" in docstring_cmd
    assert 'lint.pydocstyle.convention = "numpy"' in docstring_cmd
    assert docstring_cmd[-len(cohort) :] == cohort


def test_default_preflight_has_exact_mlir_leaf_coverage() -> None:
    """Default local coverage must enforce all post-baseline MLIR leaves at 100%."""
    gate_map = dict(_preflight.MLIR_LEAF_COVERAGE_GATES)
    run_cmd = gate_map["MLIR leaf focused coverage"]
    report_cmd = gate_map["MLIR leaf exact coverage threshold"]
    cohort = _preflight.MLIR_LEAF_COVERAGE_COHORT
    data_file = _preflight.MLIR_LEAF_COVERAGE_DATA_FILE
    source = _preflight.MLIR_LEAF_COVERAGE_SOURCE
    include = _preflight.MLIR_LEAF_COVERAGE_INCLUDE

    assert run_cmd[:4] == [_preflight._PY, "-m", "coverage", "run"]
    assert "--branch" in run_cmd
    assert f"--source={source}" in run_cmd
    assert f"--include={include}" not in run_cmd
    assert run_cmd[-len(cohort) :] == cohort
    assert f"--data-file={data_file}" in run_cmd
    assert report_cmd[:4] == [_preflight._PY, "-m", "coverage", "report"]
    assert "--precision=2" in report_cmd
    assert "--fail-under=100" in report_cmd
    assert f"--include={include}" in report_cmd
    assert f"--data-file={data_file}" in report_cmd


def test_ci_and_preflight_share_mlir_leaf_cohorts() -> None:
    """CI and local MLIR gates must preserve identical quality and coverage order."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    quality_steps = (
        "Type-check MLIR leaf quality cohort",
        "Ruff NumPy docstrings for MLIR leaf quality cohort",
    )

    for step_name in quality_steps:
        block_start = workflow.index(f"      - name: {step_name}")
        block_end = workflow.index("\n      - name:", block_start + 1)
        ci_paths = [
            line.strip()
            for line in workflow[block_start:block_end].splitlines()
            if line.strip().startswith(("src/", "tests/"))
        ]
        assert ci_paths == _preflight.MLIR_LEAF_QUALITY_RATCHET

    coverage_start = workflow.index("      - name: Run MLIR leaf focused coverage")
    coverage_end = workflow.index("\n      - name:", coverage_start + 1)
    ci_coverage_paths = [
        line.strip()
        for line in workflow[coverage_start:coverage_end].splitlines()
        if line.strip().startswith("tests/")
    ]
    assert ci_coverage_paths == _preflight.MLIR_LEAF_COVERAGE_COHORT
    assert "Enforce MLIR leaf exact coverage" in workflow
    assert f"--source={_preflight.MLIR_LEAF_COVERAGE_SOURCE}" in workflow
    assert f"--include={_preflight.MLIR_LEAF_COVERAGE_INCLUDE}" in workflow
    assert "needs['mlir-leaf-quality'].result" in workflow


def test_default_preflight_has_exact_phase_qnode_affinity_coverage() -> None:
    """Default local coverage must enforce the affinity evidence owner at 100%."""
    gate_map = dict(_preflight.PHASE_QNODE_AFFINITY_COVERAGE_GATES)
    run_cmd = gate_map["phase-qnode affinity focused coverage"]
    report_cmd = gate_map["phase-qnode affinity exact coverage threshold"]
    cohort = _preflight.PHASE_QNODE_AFFINITY_COVERAGE_COHORT
    data_file = _preflight.PHASE_QNODE_AFFINITY_COVERAGE_DATA_FILE

    assert run_cmd[:4] == [_preflight._PY, "-m", "coverage", "run"]
    assert "--branch" in run_cmd
    assert run_cmd[-len(cohort) :] == cohort
    assert f"--data-file={data_file}" in run_cmd
    assert report_cmd[:4] == [_preflight._PY, "-m", "coverage", "report"]
    assert "--precision=2" in report_cmd
    assert "--fail-under=100" in report_cmd
    assert "--include=*/qnode_affinity_benchmark.py" in report_cmd
    assert f"--data-file={data_file}" in report_cmd


def test_ci_and_preflight_share_phase_qnode_affinity_cohorts() -> None:
    """CI and local gates must preserve identical affinity-owner file order."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    quality_steps = (
        "Type-check Phase-QNode affinity quality cohort",
        "Ruff NumPy docstrings for Phase-QNode affinity quality cohort",
    )

    for step_name in quality_steps:
        block_start = workflow.index(f"      - name: {step_name}")
        block_end = workflow.index("\n      - name:", block_start + 1)
        ci_paths = [
            line.strip()
            for line in workflow[block_start:block_end].splitlines()
            if line.strip().startswith(("src/", "tools/", "tests/"))
        ]
        assert ci_paths == _preflight.PHASE_QNODE_AFFINITY_QUALITY_RATCHET

    coverage_start = workflow.index("      - name: Run Phase-QNode affinity focused coverage")
    coverage_end = workflow.index("\n      - name:", coverage_start + 1)
    ci_coverage_paths = [
        line.strip()
        for line in workflow[coverage_start:coverage_end].splitlines()
        if line.strip().startswith("tests/")
    ]
    assert ci_coverage_paths == _preflight.PHASE_QNODE_AFFINITY_COVERAGE_COHORT
    assert "Enforce Phase-QNode affinity exact coverage" in workflow
    assert "--include=*/qnode_affinity_benchmark.py" in workflow
    assert "needs['phase-qnode-affinity-quality'].result" in workflow


def test_default_preflight_has_exact_phase_qnode_vector_coverage() -> None:
    """Default local coverage must enforce the public vector owner at 100%."""
    gate_map = dict(_preflight.PHASE_QNODE_VECTOR_COVERAGE_GATES)
    run_cmd = gate_map["phase-qnode vector focused coverage"]
    report_cmd = gate_map["phase-qnode vector exact coverage threshold"]
    cohort = _preflight.PHASE_QNODE_VECTOR_COVERAGE_COHORT
    data_file = _preflight.PHASE_QNODE_VECTOR_COVERAGE_DATA_FILE

    assert run_cmd[:4] == [_preflight._PY, "-m", "coverage", "run"]
    assert "--branch" in run_cmd
    assert run_cmd[-len(cohort) :] == cohort
    assert f"--data-file={data_file}" in run_cmd
    assert report_cmd[:4] == [_preflight._PY, "-m", "coverage", "report"]
    assert "--precision=2" in report_cmd
    assert "--fail-under=100" in report_cmd
    assert "--include=*/qnode_vector_transforms.py" in report_cmd
    assert f"--data-file={data_file}" in report_cmd


def test_ci_and_preflight_share_phase_qnode_vector_cohorts() -> None:
    """CI and local gates must preserve identical vector-owner file order."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    quality_steps = (
        "Type-check Phase-QNode vector quality cohort",
        "Ruff NumPy docstrings for Phase-QNode vector quality cohort",
    )

    for step_name in quality_steps:
        block_start = workflow.index(f"      - name: {step_name}")
        block_end = workflow.index("\n      - name:", block_start + 1)
        ci_paths = [
            line.strip()
            for line in workflow[block_start:block_end].splitlines()
            if line.strip().startswith(("src/", "tests/"))
        ]
        assert ci_paths == _preflight.PHASE_QNODE_VECTOR_QUALITY_RATCHET

    coverage_start = workflow.index("      - name: Run Phase-QNode vector focused coverage")
    coverage_end = workflow.index("\n      - name:", coverage_start + 1)
    ci_coverage_paths = [
        line.strip()
        for line in workflow[coverage_start:coverage_end].splitlines()
        if line.strip().startswith("tests/")
    ]
    assert ci_coverage_paths == _preflight.PHASE_QNODE_VECTOR_COVERAGE_COHORT
    assert "Enforce Phase-QNode vector exact coverage" in workflow
    assert "--include=*/qnode_vector_transforms.py" in workflow
    assert "needs['phase-qnode-vector-quality'].result" in workflow


def test_default_preflight_has_exact_whole_program_trace_value_coverage() -> None:
    """Default local coverage must enforce the explicit trace cohort at 100%."""
    gate_map = dict(_preflight.WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_GATES)
    run_cmd = gate_map["whole-program trace-value focused coverage"]
    report_cmd = gate_map["whole-program trace-value exact coverage threshold"]
    cohort = _preflight.WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_COHORT
    data_file = _preflight.WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_DATA_FILE

    assert run_cmd[:4] == [_preflight._PY, "-m", "coverage", "run"]
    assert "--branch" in run_cmd
    assert run_cmd[-len(cohort) :] == cohort
    assert f"--data-file={data_file}" in run_cmd
    assert report_cmd[:4] == [_preflight._PY, "-m", "coverage", "report"]
    assert "--precision=2" in report_cmd
    assert "--fail-under=100" in report_cmd
    assert "--include=*/whole_program_trace_values.py" in report_cmd
    assert f"--data-file={data_file}" in report_cmd
    alias_report_cmd = gate_map["program AD alias-contract exact coverage threshold"]
    assert alias_report_cmd[:4] == [_preflight._PY, "-m", "coverage", "report"]
    assert "--precision=2" in alias_report_cmd
    assert "--fail-under=100" in alias_report_cmd
    assert "--include=*/program_ad_alias_contracts.py" in alias_report_cmd
    assert f"--data-file={data_file}" in alias_report_cmd
    shape_report_cmd = gate_map["program AD shape-transform exact coverage threshold"]
    assert shape_report_cmd[:4] == [_preflight._PY, "-m", "coverage", "report"]
    assert "--precision=2" in shape_report_cmd
    assert "--fail-under=100" in shape_report_cmd
    assert "--include=*/program_ad_shape_transforms.py" in shape_report_cmd
    assert f"--data-file={data_file}" in shape_report_cmd


def test_ci_and_preflight_share_whole_program_trace_value_cohorts() -> None:
    """CI and local gates must preserve identical quality and coverage file order."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    quality_steps = (
        "Type-check whole-program trace-value quality cohort",
        "Ruff NumPy docstrings for whole-program trace-value quality cohort",
    )

    for step_name in quality_steps:
        block_start = workflow.index(f"      - name: {step_name}")
        block_end = workflow.index("\n      - name:", block_start + 1)
        ci_paths = [
            line.strip()
            for line in workflow[block_start:block_end].splitlines()
            if line.strip().startswith(("src/", "tests/"))
        ]
        assert ci_paths == _preflight.WHOLE_PROGRAM_TRACE_VALUE_QUALITY_RATCHET

    coverage_start = workflow.index("      - name: Run whole-program trace-value focused coverage")
    coverage_end = workflow.index("\n      - name:", coverage_start + 1)
    ci_coverage_paths = [
        line.strip()
        for line in workflow[coverage_start:coverage_end].splitlines()
        if line.strip().startswith("tests/")
    ]
    assert ci_coverage_paths == _preflight.WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_COHORT
    assert "Enforce whole-program trace-value exact coverage" in workflow
    assert "--include=*/whole_program_trace_values.py" in workflow
    assert "Enforce Program-AD alias-contract exact coverage" in workflow
    assert "--include=*/program_ad_alias_contracts.py" in workflow
    assert "Enforce Program-AD shape-transform exact coverage" in workflow
    assert "--include=*/program_ad_shape_transforms.py" in workflow
    assert "--fail-under=100" in workflow
    assert "needs['whole-program-trace-value-quality'].result" in workflow


def test_ci_and_preflight_share_the_docstring_ratchet_cohort() -> None:
    """CI and the local static gate must enforce the same ordered file cohort."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    block_start = workflow.index("      - name: Ruff docstring ratchet")
    block_end = workflow.index("\n\n  decisive-advantage-quality:", block_start)
    ci_paths = [
        line.strip()
        for line in workflow[block_start:block_end].splitlines()
        if line.strip().startswith(("src/", "tests/", "tools/"))
    ]

    assert ci_paths == _preflight.DIFFERENTIABLE_DOCSTRING_RATCHET


def test_static_gates_include_differentiable_strict_mypy_ratchet() -> None:
    """Differentiable promotion modules must stay under explicit strict mypy."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}
    strict_cmd = gate_map["mypy-strict-differentiable"]
    language_gate = gate_map["differentiable-promotion-language"]

    assert "--strict" in strict_cmd
    assert "tools/check_differentiable_promotion_language.py" in language_gate
    assert "src/scpn_quantum_control/differentiable.py" in strict_cmd
    assert "src/scpn_quantum_control/differentiable_claim_ledger.py" in strict_cmd
    assert "src/scpn_quantum_control/differentiable_architecture_map.py" in strict_cmd
    assert "src/scpn_quantum_control/differentiable_dependency_environment_map.py" in strict_cmd
    assert "src/scpn_quantum_control/differentiable_baseline_scorecard.py" in strict_cmd
    assert "src/scpn_quantum_control/differentiable_api.py" in strict_cmd
    assert "src/scpn_quantum_control/benchmarks/differentiable_programming.py" in strict_cmd
    assert "src/scpn_quantum_control/differentiable_external_validation.py" in strict_cmd
    assert "src/scpn_quantum_control/differentiable_framework_overlay.py" in strict_cmd
    assert "src/scpn_quantum_control/differentiable_module_hardening_audit.py" in strict_cmd
    assert "src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py" in (
        strict_cmd
    )
    assert "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py" in strict_cmd
    assert "src/scpn_quantum_control/benchmarks/differentiable_evidence.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/differentiable_readiness.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/differentiable_audit.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/gradient_support_matrix.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/provider_gradient.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/hardware_gradient_policy.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/provider_gradient_audit.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/hardware_gradient_publication.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/provider_hardware_gradient_audit.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/hardware_gradient_campaign.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/gradient_backend.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/gradient_tape.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/natural_gradient.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/gradient_descent.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qnode_affinity_benchmark.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qnode_tape.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qnode_provider_transforms.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qnode_transforms.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qnode_vector_transforms.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qnode_framework_parity.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qnode_circuit.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/pennylane_bridge.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/jax_bridge.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/torch_bridge.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/tensorflow_bridge.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/tensorflow_maintenance.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qiskit_bridge.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qnn_framework_bridge_matrix.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/transform_nesting.py" in strict_cmd
    assert (
        "src/scpn_quantum_control/benchmarks/differentiable_external_comparison.py" in strict_cmd
    )
    assert "src/scpn_quantum_control/phase/xy_compiler.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/pennylane_import.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qnn_optimizer_benchmark.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qnn_training.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qnn_conformance.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qnn_finite_shot.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qnn_convergence.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qnn_loss_landscape.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qgnn.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/qnn_framework_agreement.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/model_training_evidence.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/domain_benchmark_datasets.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/objectives.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/objective_planner.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/objective_audit.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/optimizer_audit.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/param_shift.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/general_unitary.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/phase_vqe.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/structured_ansatz.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/xy_kuramoto.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/kuramoto_variants.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/adapt_vqe.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/trotter_error.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/ansatz_methodology.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/results.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/provider_hardware_safety_audit.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/backend_selector.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/ansatz_bench.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/trotter_upde.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/adiabatic_preparation.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/ancilla_lindblad.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/avqds.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/coupling_learning.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/contraction_optimiser.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/cross_domain_transfer.py" in strict_cmd
    assert "src/scpn_quantum_control/phase/floquet_kuramoto.py" in strict_cmd


def test_preflight_coverage_gate_collects_branches_with_local_smoke_threshold() -> None:
    """Local full preflight must collect arcs without impersonating CI's line gate."""
    assert "--cov=src/scpn_quantum_control" in _preflight._PYTEST_COV
    assert "--cov=scpn_quantum_control" not in _preflight._PYTEST_COV
    assert "--cov-branch" in _preflight._PYTEST_COV
    assert "--cov-fail-under=70" in _preflight._PYTEST_COV


def test_gate_environment_prepends_local_source_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate subprocesses must resolve local package source trees."""
    monkeypatch.setenv("PYTHONPATH", f"/external{os.pathsep}/external")

    env = _preflight._gate_environment()

    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries[:2] == [
        str(_preflight.ROOT / "src"),
        str(_preflight.ROOT / "oscillatools" / "src"),
    ]
    assert entries.count("/external") == 1


def test_gate_environment_leaves_pythonpath_absent_without_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Gate environments should not invent an empty PYTHONPATH entry."""
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(_preflight, "_RUNTIME_SOURCE_ROOTS", (tmp_path / "missing",))

    env = _preflight._gate_environment()

    assert "PYTHONPATH" not in env


def test_run_gate_reports_pass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Passing gates should print a compact pass summary."""
    observed: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["cmd"] = cmd
        observed["env"] = kwargs.get("env")
        observed["shell"] = kwargs.get("shell")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(_preflight.subprocess, "run", fake_run)

    assert _preflight.run_gate("unit", [sys.executable, "-c", "pass"]) is True
    assert "PASS  unit" in capsys.readouterr().out
    assert observed["cmd"] == [sys.executable, "-c", "pass"]
    assert isinstance(observed["env"], dict)
    assert observed["shell"] is False


def test_run_gate_reports_failure_tail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Failing gates should print only the tail of captured output."""

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            cmd,
            2,
            stdout="\n".join(f"out-{idx}" for idx in range(12)),
            stderr="\n".join(f"err-{idx}" for idx in range(12)),
        )

    monkeypatch.setattr(_preflight.subprocess, "run", fake_run)

    assert _preflight.run_gate("unit", [sys.executable, "-c", "fail"]) is False
    output = capsys.readouterr().out
    assert "FAIL  unit" in output
    assert "out-2" in output
    assert "out-11" in output
    assert "err-2" in output
    assert "err-11" in output
    assert "        out-1\n" not in output


def test_run_gate_rejects_empty_gate_commands(capsys: pytest.CaptureFixture[str]) -> None:
    """Empty gate commands should fail before subprocess execution."""
    assert _preflight.run_gate("unit", []) is False

    output = capsys.readouterr().out
    assert "FAIL  unit" in output
    assert "gate command is empty" in output


def test_run_gate_rejects_missing_gate_executables(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing executable paths should fail before subprocess execution."""
    missing = tmp_path / "missing-python"

    assert _preflight.run_gate("unit", [str(missing), "-m", "pytest"]) is False

    output = capsys.readouterr().out
    assert "FAIL  unit" in output
    assert "gate executable is not resolvable" in output


def test_run_gate_rejects_unstatable_gate_executables(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Filesystem errors during executable admission should fail the gate."""

    def raising_exists(_path: Path) -> bool:
        raise OSError("stat failed")

    monkeypatch.setattr(_preflight.Path, "exists", raising_exists)

    assert _preflight.run_gate("unit", [sys.executable]) is False

    output = capsys.readouterr().out
    assert "FAIL  unit" in output
    assert "gate executable is not resolvable" in output


def test_run_gate_rejects_relative_gate_executables(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Relative executable paths should fail admission before subprocess execution."""
    assert _preflight.run_gate("unit", ["python", "-m", "pytest"]) is False

    output = capsys.readouterr().out
    assert "FAIL  unit" in output
    assert "gate executable is not absolute" in output


def test_run_gate_rejects_directory_gate_executables(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Directory paths should fail executable admission before subprocess execution."""
    assert _preflight.run_gate("unit", [str(tmp_path)]) is False

    output = capsys.readouterr().out
    assert "FAIL  unit" in output
    assert "gate executable is not a file" in output


def test_run_gate_rejects_non_executable_gate_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-executable files should fail admission before subprocess execution."""
    candidate = tmp_path / "python"
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")
    candidate.chmod(0o644)

    assert _preflight.run_gate("unit", [str(candidate)]) is False

    output = capsys.readouterr().out
    assert "FAIL  unit" in output
    assert "gate executable is not executable" in output


def test_run_gate_reports_failure_without_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Failing gates without captured output should still report the gate name."""

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(_preflight.subprocess, "run", fake_run)

    assert _preflight.run_gate("unit", [sys.executable, "-c", "fail"]) is False
    output = capsys.readouterr().out
    assert "FAIL  unit" in output


def test_main_skips_tests_with_no_tests_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The no-tests option should run static gates and bandit only."""
    calls: list[str] = []
    monkeypatch.setattr(_preflight, "STATIC_GATES", [("lint", ["lint"])])
    monkeypatch.setattr(_preflight, "BANDIT_GATE", ("bandit", ["bandit"]))
    monkeypatch.setattr(sys, "argv", ["preflight.py", "--no-tests"])

    def fake_run_gate(name: str, cmd: list[str]) -> bool:
        calls.append(name)
        return True

    monkeypatch.setattr(_preflight, "run_gate", fake_run_gate)

    assert _preflight.main() == 0
    assert calls == ["lint", "bandit"]
    assert "ALL CLEAR" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_main_prints_help_without_running_gates(
    flag: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Help requests should return immediately without executing gates."""
    calls: list[str] = []
    monkeypatch.setattr(sys, "argv", ["preflight.py", flag])

    def fake_run_gate(name: str, cmd: list[str]) -> bool:
        calls.append(name)
        return False

    monkeypatch.setattr(_preflight, "run_gate", fake_run_gate)

    assert _preflight.main() == 0
    output = capsys.readouterr().out
    assert "Usage:" in output
    assert "python tools/preflight.py --no-tests" in output
    assert calls == []


def test_main_uses_plain_pytest_when_coverage_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-coverage option should keep pytest but drop coverage flags."""
    calls: list[str] = []
    monkeypatch.setattr(_preflight, "STATIC_GATES", [("lint", ["lint"])])
    monkeypatch.setattr(_preflight, "BANDIT_GATE", ("bandit", ["bandit"]))
    monkeypatch.setattr(sys, "argv", ["preflight.py", "--no-coverage"])

    def fake_run_gate(name: str, cmd: list[str]) -> bool:
        calls.append(name)
        return True

    monkeypatch.setattr(_preflight, "run_gate", fake_run_gate)

    assert _preflight.main() == 0
    assert calls == [
        "lint",
        "studio Program-AD Rust kernel tests",
        "studio Program-AD WASM release build",
        "studio Program-AD browser strict typecheck",
        "studio Program-AD focused browser tests",
        "pytest",
        "bandit",
    ]


def test_main_uses_coverage_pytest_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default preflight should use the coverage-enforced pytest gate."""
    calls: list[str] = []
    monkeypatch.setattr(_preflight, "STATIC_GATES", [("lint", ["lint"])])
    monkeypatch.setattr(_preflight, "BANDIT_GATE", ("bandit", ["bandit"]))
    monkeypatch.setattr(sys, "argv", ["preflight.py"])

    def fake_run_gate(name: str, cmd: list[str]) -> bool:
        calls.append(name)
        return True

    monkeypatch.setattr(_preflight, "run_gate", fake_run_gate)
    assert _preflight.main() == 0
    assert calls == [
        "lint",
        "studio Program-AD Rust kernel tests",
        "studio Program-AD WASM release build",
        "studio Program-AD browser strict typecheck",
        "decisive-advantage focused coverage",
        "decisive-advantage exact coverage threshold",
        "differentiable-scalar-kernels focused coverage",
        "differentiable-scalar-kernels exact coverage threshold",
        "program-ad-array-indexing focused coverage",
        "program-ad-array-indexing exact coverage threshold",
        "MLIR leaf focused coverage",
        "MLIR leaf exact coverage threshold",
        "phase-qnode affinity focused coverage",
        "phase-qnode affinity exact coverage threshold",
        "studio Program-AD focused coverage",
        "studio Program-AD exact coverage threshold",
        "phase-qnode vector focused coverage",
        "phase-qnode vector exact coverage threshold",
        "phase-jax-qnode focused coverage",
        "phase-jax-qnode exact coverage threshold",
        "whole-program trace-value focused coverage",
        "whole-program trace-value exact coverage threshold",
        "program AD alias-contract exact coverage threshold",
        "program AD shape-transform exact coverage threshold",
        "studio Program-AD exact browser coverage",
        "pytest + coverage",
        "bandit",
    ]


def test_main_stops_on_first_failed_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The preflight runner should stop after the first failed gate."""
    calls: list[str] = []
    monkeypatch.setattr(_preflight, "STATIC_GATES", [("lint", ["lint"]), ("type", ["type"])])
    monkeypatch.setattr(_preflight, "BANDIT_GATE", ("bandit", ["bandit"]))
    monkeypatch.setattr(sys, "argv", ["preflight.py", "--no-tests"])

    def fake_run_gate(name: str, cmd: list[str]) -> bool:
        calls.append(name)
        return name != "lint"

    monkeypatch.setattr(_preflight, "run_gate", fake_run_gate)

    assert _preflight.main() == 1
    assert calls == ["lint"]
    assert "BLOCKED: lint" in capsys.readouterr().out
