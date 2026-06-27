# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for commit authorship checker
"""Tests for the commit-message authorship checker."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
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


_check_commit_trailers = _load_tool_module(
    "check_commit_trailers_for_tests",
    "check_commit_trailers.py",
)

REQUIRED_AUTHORSHIP_LINE = "Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)"
LEGACY_COAUTHOR_TRAILER = "Co-Authored-By: " + "Arcane Sapience <protoscience@anulum.li>"


def _message_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(text, encoding="utf-8")
    return path


def test_commit_message_hook_accepts_required_authorship_line(tmp_path: Path) -> None:
    path = _message_file(
        tmp_path,
        "\n".join(
            [
                "Add release audit coverage",
                "",
                REQUIRED_AUTHORSHIP_LINE,
            ]
        ),
    )

    assert _check_commit_trailers.main(["check_commit_trailers.py", str(path)]) == 0


def test_commit_message_hook_rejects_missing_authorship_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _message_file(tmp_path, "Add release audit coverage\n")

    assert _check_commit_trailers.main(["check_commit_trailers.py", str(path)]) == 1
    assert f"missing `{REQUIRED_AUTHORSHIP_LINE}` authorship line" in capsys.readouterr().err


def test_commit_message_hook_rejects_banned_subject_word(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _message_file(
        tmp_path,
        "\n".join(
            [
                "Add comprehensive release audit",
                "",
                REQUIRED_AUTHORSHIP_LINE,
            ]
        ),
    )

    assert _check_commit_trailers.main(["check_commit_trailers.py", str(path)]) == 1
    assert "banned word(s) in subject: comprehensive" in capsys.readouterr().err


def test_message_violations_allow_banned_words_in_body_by_default() -> None:
    message = "\n".join(
        [
            "Add release audit coverage",
            "",
            "This removes a prior comprehensive wording from docs.",
            "",
            REQUIRED_AUTHORSHIP_LINE,
        ]
    )

    assert _check_commit_trailers._message_violations(message) == []
    assert _check_commit_trailers._message_violations(message, check_body_banned=True) == [
        "banned word(s) in message: comprehensive"
    ]


def test_message_violations_deduplicates_repeated_banned_words() -> None:
    message = "\n".join(
        [
            "Add robust robust release audit",
            "",
            REQUIRED_AUTHORSHIP_LINE,
        ]
    )

    assert _check_commit_trailers._message_violations(message) == [
        "banned word(s) in subject: robust"
    ]


def test_legacy_coauthor_trailer_is_transition_only() -> None:
    message = "\n".join(
        [
            "Add release audit coverage",
            "",
            LEGACY_COAUTHOR_TRAILER,
        ]
    )

    assert (
        _check_commit_trailers._message_violations(
            message,
            allow_legacy_trailer=True,
        )
        == []
    )
    assert _check_commit_trailers._message_violations(message) == [
        f"missing `{REQUIRED_AUTHORSHIP_LINE}` authorship line"
    ]


def test_commit_trailer_checker_help_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert _check_commit_trailers.main(["check_commit_trailers.py", "--help"]) == 0
    assert "Verify commit-message hygiene" in capsys.readouterr().out


def test_ci_audit_default_range_starts_at_clean_public_tag() -> None:
    assert _check_commit_trailers.DEFAULT_AUDIT_RANGE == "v0.9.6..HEAD"


def test_ci_audit_fails_closed_when_git_executable_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CI audit refuses to launch a partial git command."""
    monkeypatch.setenv("PATH", "")

    assert _check_commit_trailers._ci_audit("HEAD") == 2
    assert "git executable unavailable" in capsys.readouterr().err


def test_resolve_git_executable_rejects_unresolvable_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale path from PATH lookup is not admitted as git."""
    missing = tmp_path / "missing-git"
    monkeypatch.setattr(_check_commit_trailers.shutil, "which", lambda _name: str(missing))

    assert _check_commit_trailers._resolve_git_executable() is None


def test_resolve_git_executable_rejects_non_executable_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-executable file is not admitted as git."""
    candidate = tmp_path / "git"
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")
    candidate.chmod(0o644)
    monkeypatch.setattr(_check_commit_trailers.shutil, "which", lambda _name: str(candidate))

    assert _check_commit_trailers._resolve_git_executable() is None


def test_resolve_git_executable_and_run_git_admit_current_git() -> None:
    """The resolver returns an executable path that can run a harmless git command."""
    git_executable = _check_commit_trailers._resolve_git_executable()

    assert git_executable is not None
    result = _check_commit_trailers._run_git(git_executable, "--version")
    assert result.stdout.startswith("git version ")


def test_ci_audit_reports_rev_list_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failing rev-list call returns the CI-audit infrastructure error code."""
    monkeypatch.setattr(_check_commit_trailers, "_resolve_git_executable", lambda: sys.executable)

    def fake_run_git(_git_executable: str, *args: str) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(128, ["git", *args], stderr="bad revision")

    monkeypatch.setattr(_check_commit_trailers, "_run_git", fake_run_git)

    assert _check_commit_trailers._ci_audit("bad..range") == 2
    assert "git rev-list failed: bad revision" in capsys.readouterr().err


def test_ci_audit_reports_exempt_and_failing_commits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CI audit counts historical exemptions and reports current violations."""
    monkeypatch.setattr(_check_commit_trailers, "_resolve_git_executable", lambda: sys.executable)
    clean_sha = "abc1234567890"
    exempt_sha = "2a7d604000000"
    failing_sha = "bad9999000000"

    def fake_run_git(_git_executable: str, *args: str) -> subprocess.CompletedProcess[str]:
        if args == ("rev-list", "HEAD"):
            return subprocess.CompletedProcess(
                args, 0, stdout=f"{clean_sha}\n{exempt_sha}\n{failing_sha}\n"
            )
        if args == ("log", "-1", "--format=%B", clean_sha):
            return subprocess.CompletedProcess(
                args, 0, stdout=f"Add audit\n\n{REQUIRED_AUTHORSHIP_LINE}\n"
            )
        if args == ("log", "-1", "--format=%B", failing_sha):
            return subprocess.CompletedProcess(args, 0, stdout="Add audit\n")
        if args[0:3] == ("log", "-1", "--format=%cI"):
            return subprocess.CompletedProcess(args, 0, stdout="2026-06-01T00:00:00+00:00\n")
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(_check_commit_trailers, "_run_git", fake_run_git)

    assert _check_commit_trailers._ci_audit("HEAD") == 1
    out = capsys.readouterr().out
    assert "Audited 3 commits in HEAD" in out
    assert "Exempt (historical debt): 1" in out
    assert "Violations: 1" in out
    assert f"{failing_sha[:7]}: missing `{REQUIRED_AUTHORSHIP_LINE}` authorship line" in out


def test_main_dispatches_audit_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI dispatcher passes the requested audit range through."""
    seen: list[str] = []

    def fake_ci_audit(range_spec: str = _check_commit_trailers.DEFAULT_AUDIT_RANGE) -> int:
        seen.append(range_spec)
        return 0

    monkeypatch.setattr(_check_commit_trailers, "_ci_audit", fake_ci_audit)

    assert _check_commit_trailers.main(["check_commit_trailers.py", "--audit"]) == 0
    assert _check_commit_trailers.main(["check_commit_trailers.py", "--range", "HEAD"]) == 0
    assert _check_commit_trailers.main(["check_commit_trailers.py", "--range"]) == 0
    assert _check_commit_trailers.main(["check_commit_trailers.py"]) == 0
    assert seen == [
        _check_commit_trailers.DEFAULT_AUDIT_RANGE,
        "HEAD",
        _check_commit_trailers.DEFAULT_AUDIT_RANGE,
        _check_commit_trailers.DEFAULT_AUDIT_RANGE,
    ]
