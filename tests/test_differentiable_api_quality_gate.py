# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — API quality gate enforcement tests
"""Prove result-test inventory coverage and executable documentation refusal."""

from __future__ import annotations

import runpy
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tools import differentiable_api_quality_gates as gates

ROOT = Path(__file__).resolve().parents[1]
"""Actual checkout whose test inventory and hosted commands must agree."""


def _workflow_command(path: str, step: str) -> list[str]:
    """Read one existing folded run step; fail if its literal structure drifts."""
    source = (ROOT / path).read_text(encoding="utf-8")
    marker = f"      - name: {step}\n        run: >-\n"
    assert source.count(marker) == 1
    body = source.split(marker, 1)[1].split("      - name:", 1)[0]
    return shlex.split(body)


def test_all_result_tests_belong_to_static_and_runtime_cohorts() -> None:
    """A new result test must not inherit the blanket test documentation omission."""
    owned = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").glob("test_differentiable_result*.py")
    }
    owned.add("tests/test_differentiable_api_quality_gate.py")
    assert "tests/test_differentiable_result_provenance.py" in owned
    assert owned <= set(gates.DIFFERENTIABLE_API_QUALITY_RATCHET)
    assert owned <= set(gates.DIFFERENTIABLE_API_COVERAGE_COHORT)


def test_hosted_static_commands_match_canonical_gate_arguments() -> None:
    """Hosted typing and docs must enforce the same explicit source/test cohort."""
    workflow = ".github/workflows/ci-static-analysis.yml"
    assert _workflow_command(workflow, "Type-check unified differentiable API quality cohort") == [
        "python",
        "tools/differentiable_api_quality_gates.py",
        "typing",
    ]
    assert _workflow_command(
        workflow, "Ruff NumPy docstrings for unified differentiable API quality cohort"
    ) == ["python", "tools/differentiable_api_quality_gates.py", "docstrings"]


def test_native_axis_compilation_uses_required_static_runtime_and_combined_coverage() -> None:
    """Require the axis compiler bridge in the real SDK producer and final report."""
    source = "src/scpn_quantum_control/differentiable_axis_compilation.py"
    test = "tests/test_differentiable_axis_compilation.py"
    assert {source, test} <= set(gates.DIFFERENTIABLE_API_QUALITY_RATCHET)
    execution, report = [
        command for _name, command in gates.build_native_tensor_coverage_gates("python")
    ]
    assert test in execution
    assert "--fail-under=100" in report
    assert any("*/differentiable_axis_compilation.py" in argument for argument in report)
    workflow = ".github/workflows/ci-framework-parity.yml"
    assert _workflow_command(workflow, "Run real native tensor construction coverage") == execution
    assert (
        _workflow_command(workflow, "Enforce native tensor construction exact coverage") == report
    )
    combined = (ROOT / ".github/workflows/ci-native-integration.yml").read_text()
    download = combined.index("Download exact-commit native tensor construction coverage")
    combine = combined.index(
        'coverage combine --append --keep "$RUNNER_TEMP/native-tensor-coverage/'
    )
    exact = combined.index("*/differentiable_axis_compilation.py", combine)
    xml = combined.index("coverage xml", exact)
    assert download < combine < exact < xml


@pytest.mark.parametrize("check", ["typing", "docstrings"])
def test_static_gate_cli_executes_the_real_canonical_check(check: str) -> None:
    """Exercise the actual strict and documentation tools through the CI entry.

    Parameters
    ----------
    check
        Canonical static check selected by the hosted step.

    """
    assert gates.main([check]) == 0


def test_static_gate_cli_refuses_an_unknown_check() -> None:
    """Reject an unsupported selector before any command can execute."""
    with pytest.raises(SystemExit) as exc:
        gates.main(["unregistered"])
    assert exc.value.code == 2


def test_static_gate_script_entrypoint_executes_the_documentation_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the same script bootstrap used by the hosted documentation step.

    Parameters
    ----------
    monkeypatch
        Restore process arguments after the actual script entrypoint.

    """
    monkeypatch.setattr(sys, "argv", [gates.__file__, "docstrings"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(gates.__file__, run_name="__main__")
    assert exc.value.code == 0


def test_all_numerical_core_owners_and_tests_are_enforced() -> None:
    """Keep every actual numerical owner and dedicated runtime test in both gates."""
    sources = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src/scpn_quantum_control").glob("differentiable_api_core*.py")
    }
    tests = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").glob("test_differentiable_api_core*.py")
    }
    assert len(sources) == len(tests) == 3
    assert sources | tests <= set(gates.DIFFERENTIABLE_API_QUALITY_RATCHET)
    assert tests <= set(gates.DIFFERENTIABLE_API_COVERAGE_COHORT)


def test_hosted_coverage_executes_every_result_test() -> None:
    """The exact-coverage job must execute the new tests, not just lint them."""
    command = _workflow_command(
        ".github/workflows/ci-whole-program-trace.yml",
        "Run unified differentiable API focused coverage",
    )
    assert command == gates.build_coverage_gates("python")[0][1]
    report = _workflow_command(
        ".github/workflows/ci-whole-program-trace.yml",
        "Enforce unified differentiable API exact coverage",
    )
    assert report == gates.build_coverage_gates("python")[1][1]


@pytest.mark.parametrize("documented", [False, True])
def test_actual_doc_gate_rejects_missing_docs_and_accepts_valid_test(
    documented: bool,
) -> None:
    """Execute Ruff with the governed profile despite repository test ignores.

    Parameters
    ----------
    documented
        Include real module/function docstrings for the positive candidate.

    """
    command = dict(gates.build_static_quality_gates(sys.executable))[
        "ruff D differentiable-api quality ratchet"
    ]
    cohort = gates.DIFFERENTIABLE_API_QUALITY_RATCHET
    assert command[-len(cohort) :] == cohort
    candidate = (
        '"""Verify a deterministic arithmetic invariant."""\n'
        'def test_sum():\n    """Integer addition preserves the expected sum."""\n'
        "    assert 1 + 1 == 2\n"
        if documented
        else "def test_sum():\n    assert 1 + 1 == 2\n"
    )
    result = subprocess.run(
        [
            *command[: -len(cohort)],
            "--stdin-filename",
            "tests/test_differentiable_result_provenance.py",
            "-",
        ],
        input=candidate,
        text=True,
        capture_output=True,
        cwd=ROOT,
        timeout=15,
        check=False,
    )
    assert result.returncode == (0 if documented else 1), result.stdout + result.stderr
    if not documented:
        assert "Missing docstring in public module" in result.stdout
        assert "Missing docstring in public function" in result.stdout
