# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for test-quality policy audit
"""Tests for repository test-quality policy enforcement."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "audit_test_quality.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("audit_test_quality_for_tests", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_forbidden_test_names_are_reported_with_policy_reasons() -> None:
    tool = _load_tool_module()

    findings = tool.audit_test_paths(
        [
            Path("tests/test_coverage_100_remaining.py"),
            Path("tests/test_runner_coverage.py"),
            Path("tests/test_e2e_new_modules.py"),
            Path("tests/test_phase/results_contract.py"),
        ]
    )

    assert [finding.path.as_posix() for finding in findings] == [
        "tests/test_coverage_100_remaining.py",
        "tests/test_e2e_new_modules.py",
        "tests/test_runner_coverage.py",
    ]
    assert all("non-specific bucket" in finding.reason for finding in findings)


def test_repository_tests_do_not_use_forbidden_coverage_bucket_names() -> None:
    tool = _load_tool_module()
    repository_paths = subprocess.run(
        ["git", "ls-files", "tests/*.py"],
        check=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    repository_paths.extend(
        path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").glob("*.py")
    )

    findings = tool.audit_test_paths(Path(path) for path in repository_paths)

    assert findings == []
