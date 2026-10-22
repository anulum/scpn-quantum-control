# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native transform quality gate tests
"""Guard the required native cohort, hosted wiring and executable documentation refusal."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import differentiable_transform_algebra_quality_gates as gates
from tools.ci_workflow_inventory import load_ci_workflow_policy


def test_owning_workflow_uses_canonical_gates_and_required_framework_profile() -> None:
    """Keep hosted execution on the canonical commands and require actual dependencies."""
    workflow = Path(".github/workflows/ci-differentiable-control.yml").read_text()
    native_job = workflow.split("  native-transform-algebra-quality:", 1)[1]
    assert "requirements-ci-native-ad-py312-linux.txt" in native_job
    assert "requirements-ci-torch-cpu-py312-linux.txt" in native_job
    assert "--require-hashes" in native_job
    assert "pip check" in native_job
    assert 'echo "TMPDIR=$RUNNER_TEMP" >> "$GITHUB_ENV"' in native_job
    assert "python -m pip_audit -r requirements-ci-native-ad-py312-linux.txt" in native_job
    assert "--no-deps --disable-pip --strict --desc on" in native_job
    assert "continue-on-error" not in native_job
    assert "|| true" not in native_job
    artifact = "native-transform-algebra-coverage-${{ github.sha }}"
    assert artifact in native_job
    integration = Path(".github/workflows/ci-native-integration.yml").read_text()
    assert artifact in integration
    assert "coverage combine --append --keep" in integration
    assert "coverage xml --include='*/scpn_quantum_control/*'" in integration
    assert integration.index("coverage combine --append --keep") < integration.index(
        "tools/audit_coverage_policy.py --coverage-xml"
    )
    policy = load_ci_workflow_policy()
    native_category = next(
        row for row in policy["categories"] if row["id"] == "native-integration"
    )
    assert native_category["caller_needs"] == [
        "static-analysis",
        "native-build",
        "differentiable-control",
        "simulation-artifact",
    ]
    assert "gates.build_static_quality_gates" in native_job
    assert "gates.build_coverage_gates" in native_job
    commands = dict(gates.build_coverage_gates(sys.executable))
    for framework in ("jax", "pytorch", "tensorflow"):
        assert "-m" in commands[f"native-transform required {framework}"]
        assert gates.RUNTIME_TESTS[framework] in commands[f"native-transform required {framework}"]
        assert "--framework" in commands[f"native-transform CLI {framework}"]
    report = commands["native-transform exact coverage"]
    assert "--fail-under=100" in report
    assert "--branch" in commands["native-transform required jax"]


def test_native_gate_executes_the_live_hardening_registry_and_covers_its_owner() -> None:
    """Catch unregistered source additions through the actual owning runtime gate."""
    owner = "src/scpn_quantum_control/differentiable_module_hardening_audit.py"
    test = "tests/test_differentiable_module_hardening_audit.py"
    commands = dict(gates.build_coverage_gates(sys.executable))
    assert owner in gates.COVERAGE_OWNERS
    assert owner in gates.STATIC_COHORT and test in gates.STATIC_COHORT
    assert test in commands["native-transform local boundaries"]
    assert "*/differentiable_module_hardening_audit.py" in next(
        argument
        for argument in commands["native-transform exact coverage"]
        if argument.startswith("--include=")
    )


@pytest.mark.parametrize("documented", [False, True])
def test_native_doc_gate_executes_missing_docstring_refusal(documented: bool) -> None:
    """Verify actual Ruff refusal independently from gate command construction.

    Parameters
    ----------
    documented:
        Whether the candidate includes native module and function documentation.

    """
    command = gates.build_static_quality_gates(sys.executable)[1][1]
    prefix = command[: -len(gates.STATIC_COHORT)]
    candidate = (
        '"""A deterministic sample."""\ndef test_value():\n    """Assert the independent result."""\n    assert 2 + 3 == 5\n'
        if documented
        else "def test_value():\n    assert 2 + 3 == 5\n"
    )
    result = subprocess.run(
        [*prefix, "--stdin-filename", "tests/test_native_transform_composition.py", "-"],
        input=candidate,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == (0 if documented else 1), result.stdout + result.stderr
    if not documented:
        assert "Missing docstring" in result.stdout


def test_example_refuses_failed_runtime_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Preserve a failed report as a nonzero example exit; real CLI execution is separate.

    Parameters
    ----------
    monkeypatch:
        Scoped failure at the public facade dependency.
    capsys:
        Captured example output retaining failed evidence.

    """
    spec = importlib.util.spec_from_file_location(
        "native_transform_example", "examples/native_transform_algebra.py"
    )
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    failed = SimpleNamespace(supported=False, to_dict=lambda: {"supported": False})
    monkeypatch.setattr(example, "differentiable_api", lambda *args, **kwargs: failed)
    monkeypatch.setattr(sys, "argv", ["native-transform-example", "jax"])
    with pytest.raises(SystemExit) as exc:
        example.main()
    assert exc.value.code == 1
    assert '"supported": false' in capsys.readouterr().out
