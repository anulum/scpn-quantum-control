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
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "audit_test_quality.py"


def _load_tool_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_test_quality_for_tests", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load audit_test_quality from {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str = "# test fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@example.test"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)


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


def test_exceptions_and_module_backed_tests_are_not_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _load_tool_module()
    _write(tmp_path / "src" / "domain_runner.py")
    monkeypatch.setattr(tool, "ROOT", tmp_path)

    findings = tool.audit_test_paths(
        [
            Path("tests/test_audit_coverage_gaps.py"),
            Path("tests/test_domain_runner.py"),
        ]
    )

    assert findings == []


def test_repository_tests_do_not_use_forbidden_coverage_bucket_names() -> None:
    tool = _load_tool_module()

    findings = tool.audit_test_paths(tool.repository_test_paths(ROOT))

    assert findings == []


def test_repository_test_paths_reads_tracked_and_worktree_tests(tmp_path: Path) -> None:
    tool = _load_tool_module()
    _init_repo(tmp_path)
    tests_dir = tmp_path / "tests"
    _write(tests_dir / "test_tracked_contract.py")
    _write(tests_dir / "test_worktree_contract.py")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "tests/test_tracked_contract.py"],
        check=True,
    )

    assert tool.repository_test_paths(tmp_path) == [
        Path("tests/test_tracked_contract.py"),
        Path("tests/test_worktree_contract.py"),
    ]


def test_repository_test_paths_falls_back_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool_module()
    _write(tmp_path / "tests" / "test_domain_contract.py")

    monkeypatch.setattr(tool, "_resolve_git_executable", lambda: None)

    assert tool.repository_test_paths(tmp_path) == [Path("tests/test_domain_contract.py")]


def test_repository_test_paths_falls_back_when_git_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool_module()
    _write(tmp_path / "tests" / "test_domain_contract.py")
    monkeypatch.setattr(tool, "_resolve_git_executable", lambda: sys.executable)

    def failing_git(
        _root: Path, _git_executable: str, *args: str
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(128, ["git", *args], stderr="not a repository")

    monkeypatch.setattr(tool, "_run_git", failing_git)

    assert tool.repository_test_paths(tmp_path) == [Path("tests/test_domain_contract.py")]


def test_resolve_git_executable_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool_module()
    monkeypatch.setattr(tool.shutil, "which", lambda _name: None)

    assert tool._resolve_git_executable() is None


def test_resolve_git_executable_rejects_stale_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _load_tool_module()
    missing = tmp_path / "missing-git"
    monkeypatch.setattr(tool.shutil, "which", lambda _name: str(missing))

    assert tool._resolve_git_executable() is None


def test_resolve_git_executable_rejects_non_executable_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _load_tool_module()
    candidate = tmp_path / "git"
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")
    candidate.chmod(0o644)
    monkeypatch.setattr(tool.shutil, "which", lambda _name: str(candidate))

    assert tool._resolve_git_executable() is None


def test_resolve_git_executable_and_run_git_admit_current_git() -> None:
    tool = _load_tool_module()
    git_executable = tool._resolve_git_executable()

    assert git_executable is not None
    result = tool._run_git(ROOT, git_executable, "--version")
    assert result.stdout.startswith("git version ")


def test_format_findings_pass_and_fail_messages() -> None:
    tool = _load_tool_module()
    finding = tool.TestQualityFinding(
        path=Path("tests/test_final_bucket.py"),
        reason="non-specific bucket pytest modules are forbidden",
    )

    assert "passed" in tool.format_findings([])
    rendered = tool.format_findings([finding])
    assert rendered.startswith("test-quality audit failed:")
    assert "tests/test_final_bucket.py" in rendered


def test_main_returns_zero_for_clean_repository(capsys: pytest.CaptureFixture[str]) -> None:
    tool = _load_tool_module()

    assert tool.main(["--root", str(ROOT)]) == 0
    assert "test-quality audit passed" in capsys.readouterr().out


def test_main_returns_one_for_forbidden_worktree_test(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _load_tool_module()
    _write(tmp_path / "tests" / "test_coverage_100_remaining.py")

    assert tool.main(["--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "test-quality audit failed" in out
    assert "tests/test_coverage_100_remaining.py" in out
