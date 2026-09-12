# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — API quality gate enforcement tests
"""Prove result-test inventory coverage and executable documentation refusal."""

from __future__ import annotations

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
    static = dict(gates.build_static_quality_gates("python"))
    workflow = ".github/workflows/ci-static-analysis.yml"
    assert (
        _workflow_command(workflow, "Type-check unified differentiable API quality cohort")
        == static["mypy-strict-differentiable-api-quality"]
    )
    assert (
        _workflow_command(
            workflow, "Ruff NumPy docstrings for unified differentiable API quality cohort"
        )
        == static["ruff D differentiable-api quality ratchet"]
    )


def test_hosted_coverage_executes_every_result_test() -> None:
    """The exact-coverage job must execute the new tests, not just lint them."""
    command = _workflow_command(
        ".github/workflows/ci-whole-program-trace.yml",
        "Run unified differentiable API focused coverage",
    )
    assert command == gates.build_coverage_gates("python")[0][1]


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
