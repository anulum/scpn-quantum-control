# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the shell script lint gate
"""Exercise the shell script lint gate with real Git and the real ShellCheck."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools import audit_shell_scripts as gate

LINTER = gate.environment_executable(Path(sys.executable))

pytestmark = pytest.mark.skipif(
    shutil.which(LINTER) is None, reason="ShellCheck executable unavailable"
)

CLEAN = "#!/bin/sh\nprintf '%s\\n' \"$1\"\n"
UNQUOTED = "#!/bin/sh\necho $1\n"


def _git(repository: Path, *arguments: str) -> str:
    """Run Git in ``repository`` and return its standard output."""
    completed = subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _write(repository: Path, name: str, text: str) -> None:
    """Write ``text`` to ``name`` inside ``repository``, creating parents, and stage it."""
    path = repository / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _git(repository, "add", "--", name)


def _run(repository: Path, *arguments: str) -> int:
    """Run the gate's command line against ``repository``."""
    return gate.main(["--repo", str(repository), *arguments])


@pytest.fixture
def installed(tmp_path: Path) -> str:
    """Return the release of the ShellCheck executable the gate uses by default."""
    return gate.installed_release(tmp_path, LINTER)


@pytest.fixture
def repository(tmp_path: Path, installed: str) -> Path:
    """Return a repository with one clean script and a pin of the installed release."""
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "--quiet")
    _write(
        root, str(gate.PIN_FILE), f"{gate.DISTRIBUTION}=={installed}.1 \\\n    --hash=sha256:00\n"
    )
    _write(root, "scripts/run.sh", CLEAN)
    _write(root, "README.md", "# Title\n")
    return root


def test_fixture_repository_is_separate_from_the_checkout(repository: Path) -> None:
    """The fixture repository's Git directory is its own."""
    assert (
        Path(_git(repository, "rev-parse", "--absolute-git-dir"))
        == (repository / ".git").resolve()
    )


def test_clean_scripts_pass(
    repository: Path, installed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A repository whose scripts have no finding passes."""
    assert _run(repository) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == f"Shell script lint: 1 scripts; ShellCheck {installed}; 0 findings\n"


def test_finding_fails(repository: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A script with an unquoted expansion fails the gate with the linter's message."""
    _write(repository, "tools/helper.sh", UNQUOTED)

    assert _run(repository) == 1

    captured = capsys.readouterr()
    assert captured.err.startswith("tools/helper.sh:2:6: info: SC2086: Double quote")
    assert "2 scripts" in captured.out
    assert captured.out.endswith("1 findings\n")


def test_script_without_suffix_is_found_by_its_interpreter(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hook script without a suffix is linted when its first line names a shell."""
    _write(repository, "hooks/pre-push", "#!/usr/bin/env bash\necho $1\n")
    _write(repository, "hooks/describe", "#!/usr/bin/env python3\nprint(1)\n")
    _write(repository, "LICENSE", "Text without an interpreter line.\n")
    _write(repository, "notes/run.txt", UNQUOTED)

    assert gate.shell_scripts(repository) == ["hooks/pre-push", "scripts/run.sh"]
    assert _run(repository) == 1
    assert "hooks/pre-push:2:6: info: SC2086" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("first_line", "expected"),
    [
        ("#!/bin/sh", True),
        ("#!/bin/bash -e", True),
        ("#!/usr/bin/env dash", True),
        ("#!/bin/ksh", True),
        ("#! /usr/bin/env bash", True),
        ("#!/usr/bin/zsh", False),
        ("#!/usr/bin/fish", False),
        ("#!/usr/bin/env python3", False),
        ("echo sh", False),
    ],
)
def test_interpreter_line_decides(repository: Path, first_line: str, expected: bool) -> None:
    """Only a first line that names a supported shell marks a suffix-less file."""
    (repository / "tool").write_text(f"{first_line}\ntrue\n", encoding="utf-8")

    assert gate.is_shell_script(repository, "tool") is expected


def test_suffix_decides_without_reading(repository: Path) -> None:
    """Shell suffixes are scripts, and tracked paths that are no files are not."""
    assert gate.is_shell_script(repository, "absent/Install.SH") is True
    assert gate.is_shell_script(repository, "absent/setup.bash") is True
    assert gate.is_shell_script(repository, "absent/tool") is False
    assert gate.is_shell_script(repository, "scripts") is False


def test_repository_without_scripts_passes(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no shell script the linter is not run and the gate passes."""
    _git(repository, "rm", "--quiet", "--force", "scripts/run.sh")

    assert gate.lint(repository, [], "shellcheck-that-is-not-installed") == []
    assert _run(repository) == 0
    assert "0 scripts" in capsys.readouterr().out


def test_environment_linter_is_preferred_over_the_path(tmp_path: Path) -> None:
    """The linter beside the interpreter is used; without one the name is left to ``PATH``."""
    with_linter = tmp_path / "with" / "bin"
    without_linter = tmp_path / "without" / "bin"
    with_linter.mkdir(parents=True)
    without_linter.mkdir(parents=True)
    (with_linter / "shellcheck").write_text("", encoding="utf-8")
    (without_linter / "shellcheck").mkdir()

    assert gate.environment_executable(with_linter / "python") == str(with_linter / "shellcheck")
    assert gate.environment_executable(without_linter / "python") == "shellcheck"
    assert gate.environment_executable(tmp_path / "absent" / "python") == "shellcheck"


def test_other_release_is_refused(
    repository: Path, installed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A linter release other than the pinned one fails instead of reporting."""
    _write(repository, str(gate.PIN_FILE), f"{gate.DISTRIBUTION}==99.98.97.1\n")

    assert gate.pinned_release(repository) == "99.98.97"
    assert _run(repository) == 1
    assert (
        f"shell script lint failed: ShellCheck {installed} is not the pinned release 99.98.97; "
        f"install {gate.PIN_FILE}" in capsys.readouterr().err
    )


def test_missing_or_empty_pin_is_refused(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pin file without the linter, or no pin file, fails the gate."""
    _write(repository, str(gate.PIN_FILE), "ruff==0.16.4\n")

    assert _run(repository) == 1
    assert f"{gate.PIN_FILE} does not pin {gate.DISTRIBUTION}" in capsys.readouterr().err

    (repository / gate.PIN_FILE).unlink()

    assert _run(repository) == 1
    assert "shell script lint failed: cannot read the linter pin" in capsys.readouterr().err


def test_linter_that_cannot_run_is_refused(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing linter, or a program that prints no release, fails the gate."""
    assert _run(repository, "--shellcheck", "shellcheck-that-is-not-installed") == 1
    assert (
        "shell script lint failed: cannot run shellcheck-that-is-not-installed"
        in capsys.readouterr().err
    )

    assert _run(repository, "--shellcheck", "true") == 1
    assert (
        "shell script lint failed: true --version reported no release" in capsys.readouterr().err
    )


def test_unreadable_script_is_an_error_not_a_pass(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tracked script missing from the work tree fails the gate."""
    (repository / "scripts" / "run.sh").unlink()

    assert _run(repository) == 1
    assert f"shell script lint failed: {LINTER} failed: scripts/run.sh" in capsys.readouterr().err


def test_directory_without_a_repository_fails(
    tmp_path: Path,
    installed: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Outside a Git repository the tracked files cannot be listed.

    The ceiling keeps Git from finding a repository above the temporary
    directory, wherever the test session places it.
    """
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    (tmp_path / gate.PIN_FILE).write_text(
        f"{gate.DISTRIBUTION}=={installed}.1\n", encoding="utf-8"
    )

    assert _run(tmp_path) == 1
    assert "shell script lint failed: git ls-files failed" in capsys.readouterr().err
