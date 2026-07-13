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
    spec.loader.exec_module(module)
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


def test_static_gates_include_differentiable_docstring_ratchet() -> None:
    """Differentiable docstring-clean modules must stay under Ruff D."""
    gate_map = {name: cmd for name, cmd in _preflight.STATIC_GATES}
    docstring_cmd = gate_map["ruff D differentiable module-hardening ratchet"]

    assert "--select" in docstring_cmd
    assert "D" in docstring_cmd
    assert "src/scpn_quantum_control/differentiable_architecture_map.py" in docstring_cmd
    assert "src/scpn_quantum_control/differentiable_dependency_environment_map.py" in (
        docstring_cmd
    )
    assert "src/scpn_quantum_control/differentiable_baseline_scorecard.py" in docstring_cmd
    assert "src/scpn_quantum_control/differentiable_external_validation.py" in (docstring_cmd)
    assert "src/scpn_quantum_control/differentiable_module_hardening_audit.py" in (docstring_cmd)
    assert "src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py" in (
        docstring_cmd
    )
    assert "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py" in (
        docstring_cmd
    )
    assert "tests/test_differentiable_external_validation.py" in docstring_cmd
    assert "tests/test_differentiable_module_hardening_audit.py" in docstring_cmd
    assert "tests/test_differentiable_hardening_gate.py" in docstring_cmd


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
    assert calls == ["lint", "pytest", "bandit"]


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
    assert calls == ["lint", "pytest + coverage", "bandit"]


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
