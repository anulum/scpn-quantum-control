# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode JAX quality-gate contract tests
"""Verify local and CI Phase-QNode JAX quality-gate parity."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from tools import phase_jax_qnode_quality_gates as _quality_gates


def _load_preflight_module() -> ModuleType:
    """Load preflight with sibling policy imports available.

    Returns
    -------
    ModuleType
        Isolated preflight module used for command-contract inspection.

    Raises
    ------
    ImportError
        If Python cannot construct a loader for the preflight module.

    """
    module_name = "preflight_for_phase_jax_qnode_quality_tests"
    module_path = Path(__file__).resolve().parents[1] / "tools" / "preflight.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_preflight = _load_preflight_module()


def _workflow_step_paths(workflow: str, step_name: str) -> list[str]:
    """Extract ordered repository paths from one workflow step.

    Parameters
    ----------
    workflow
        Complete CI workflow text.
    step_name
        Exact GitHub Actions step name delimiting the command block.

    Returns
    -------
    list[str]
        Repository paths in declared command order.

    """
    block_start = workflow.index(f"      - name: {step_name}")
    block_end = workflow.index("\n      - name:", block_start + 1)
    return [
        line.strip()
        for line in workflow[block_start:block_end].splitlines()
        if line.strip().startswith(("src/", "tools/", "tests/"))
    ]


def test_static_gates_cover_the_exact_quality_ratchet() -> None:
    """Local static gates should type and document every owner-policy file."""
    gate_map = dict(_preflight.STATIC_GATES)
    strict_cmd = gate_map["mypy-strict-phase-jax-qnode"]
    docstring_cmd = gate_map["ruff D phase-jax-qnode quality ratchet"]
    cohort = _quality_gates.PHASE_JAX_QNODE_QUALITY_RATCHET

    assert strict_cmd[:4] == [_preflight._PY, "-m", "mypy", "--strict"]
    assert "--explicit-package-bases" in strict_cmd
    assert strict_cmd[-len(cohort) :] == cohort
    assert "--isolated" in docstring_cmd
    assert "D,D413" in docstring_cmd
    assert 'lint.pydocstyle.convention = "numpy"' in docstring_cmd
    assert docstring_cmd[-len(cohort) :] == cohort


def test_default_local_gate_has_exact_owner_coverage() -> None:
    """Default local coverage should enforce the transform owner at 100%."""
    gate_map = dict(_preflight.PHASE_JAX_QNODE_COVERAGE_GATES)
    run_cmd = gate_map["phase-jax-qnode focused coverage"]
    report_cmd = gate_map["phase-jax-qnode exact coverage threshold"]
    cohort = _quality_gates.PHASE_JAX_QNODE_COVERAGE_COHORT
    data_file = _quality_gates.PHASE_JAX_QNODE_COVERAGE_DATA_FILE

    assert run_cmd[:4] == [_preflight._PY, "-m", "coverage", "run"]
    assert "--rcfile=/dev/null" in run_cmd
    assert "--branch" in run_cmd
    assert "--source=src/scpn_quantum_control/phase" in run_cmd
    assert run_cmd[-len(cohort) :] == cohort
    assert f"--data-file={data_file}" in run_cmd
    assert report_cmd[:4] == [_preflight._PY, "-m", "coverage", "report"]
    assert "--rcfile=/dev/null" in report_cmd
    assert "--precision=2" in report_cmd
    assert "--fail-under=100" in report_cmd
    assert "--include=*/jax_qnode_transforms.py" in report_cmd
    assert f"--data-file={data_file}" in report_cmd


def test_preflight_reexports_the_policy_contract() -> None:
    """Preflight should expose the canonical policy constants unchanged."""
    assert _preflight.PHASE_JAX_QNODE_QUALITY_RATCHET == (
        _quality_gates.PHASE_JAX_QNODE_QUALITY_RATCHET
    )
    assert _preflight.PHASE_JAX_QNODE_COVERAGE_COHORT == (
        _quality_gates.PHASE_JAX_QNODE_COVERAGE_COHORT
    )
    assert _preflight.PHASE_JAX_QNODE_COVERAGE_DATA_FILE == (
        _quality_gates.PHASE_JAX_QNODE_COVERAGE_DATA_FILE
    )


def test_ci_and_local_gates_share_exact_owner_order() -> None:
    """CI and local gates should preserve identical owner-file ordering."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    quality_steps = (
        "Type-check Phase-QNode JAX quality cohort",
        "Ruff NumPy docstrings for Phase-QNode JAX quality cohort",
    )

    for step_name in quality_steps:
        assert _workflow_step_paths(workflow, step_name) == (
            _quality_gates.PHASE_JAX_QNODE_QUALITY_RATCHET
        )

    assert (
        _workflow_step_paths(
            workflow,
            "Run Phase-QNode JAX focused coverage",
        )
        == _quality_gates.PHASE_JAX_QNODE_COVERAGE_COHORT
    )
    assert "Enforce Phase-QNode JAX exact coverage" in workflow
    assert "--source=src/scpn_quantum_control/phase" in workflow
    assert "--include=*/jax_qnode_transforms.py" in workflow
    overlay_position = workflow.index("Build CPU-only differentiable framework overlay")
    runtime_probe_position = workflow.index("Verify real JAX runtime for Phase-QNode coverage")
    coverage_position = workflow.index("Run Phase-QNode JAX focused coverage")
    parity_job_position = workflow.index("  differentiable-parity:")
    next_job_position = workflow.index("\n  security:", parity_job_position)
    assert (
        parity_job_position
        < overlay_position
        < runtime_probe_position
        < coverage_position
        < next_job_position
    )
    assert "import jax; devices=jax.devices()" in workflow
    assert "primary={devices[0]}" in workflow
    assert "SCPN_FRAMEWORK_OVERLAY:$PYTHONPATH" in workflow
    assert "differentiable-parity" in workflow[workflow.index("  ci-gate:") :]
