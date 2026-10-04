# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the test suite's Git location isolation
"""Exercise the Git location isolation with real Git and a real nested session."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import _git_location_isolation as isolation
import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
PROBE = "tests/test_git_location_isolation.py::test_temporary_repository_owns_its_git_directory"
LOCATION_VARIABLES = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_COMMON_DIR",
)


def _executable(path: Path, script: str) -> None:
    """Write a POSIX shell ``script`` to ``path`` and make it executable."""
    path.write_text(f"#!/bin/sh\n{script}\n", encoding="utf-8")
    path.chmod(0o755)


def test_git_names_the_location_variables() -> None:
    """Git's own list contains every variable that redirects a repository."""
    reported = isolation.repository_local_variables()

    assert set(LOCATION_VARIABLES) <= set(reported)
    assert len(reported) == len(set(reported))


def test_session_holds_no_location_variable() -> None:
    """The running session carries none of Git's repository-local variables."""
    inherited = [name for name in isolation.repository_local_variables() if name in os.environ]

    assert inherited == []


def test_drop_removes_location_variables_and_keeps_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dropping returns the removed pairs in Git's order and spares other variables."""
    elsewhere = str(tmp_path / "elsewhere.git")
    index = str(tmp_path / "index")
    monkeypatch.setenv("GIT_INDEX_FILE", index)
    monkeypatch.setenv("GIT_DIR", elsewhere)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Kept Author")

    removed = isolation.drop_inherited_git_location()

    assert removed == {"GIT_DIR": elsewhere, "GIT_INDEX_FILE": index}
    assert list(removed) == ["GIT_DIR", "GIT_INDEX_FILE"]
    assert "GIT_DIR" not in os.environ
    assert "GIT_INDEX_FILE" not in os.environ
    assert os.environ["GIT_AUTHOR_NAME"] == "Kept Author"
    assert isolation.drop_inherited_git_location() == {}


def test_missing_git_leaves_nothing_to_drop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without a ``git`` executable on ``PATH`` the list is empty."""
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere.git"))

    assert isolation.repository_local_variables() == ()
    assert isolation.drop_inherited_git_location() == {}


def test_git_that_cannot_start_leaves_nothing_to_drop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ``git`` file without execute permission counts as no Git."""
    (tmp_path / "git").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (tmp_path / "git").chmod(0o644)
    monkeypatch.setenv("PATH", str(tmp_path))

    assert isolation.repository_local_variables() == ()


def test_git_that_cannot_report_the_list_stops_the_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Git that starts but fails raises instead of running without isolation."""
    _executable(tmp_path / "git", "exit 3")
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(subprocess.CalledProcessError) as failure:
        isolation.repository_local_variables()

    assert failure.value.returncode == 3


def test_temporary_repository_owns_its_git_directory(tmp_path: Path) -> None:
    """A repository created by a test lives in the test's own directory."""
    inherited = [name for name in isolation.repository_local_variables() if name in os.environ]
    assert inherited == []
    repository = tmp_path / "repository"
    repository.mkdir()

    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True, capture_output=True)
    reported = subprocess.run(
        ["git", "rev-parse", "--absolute-git-dir"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert Path(reported) == (repository / ".git").resolve()


def test_session_start_drops_an_inherited_location(tmp_path: Path) -> None:
    """A session started with a foreign ``GIT_DIR`` reports the drop and spares it.

    The nested session runs the repository probe above with ``GIT_DIR`` and
    ``GIT_WORK_TREE`` pointing at a decoy directory. Without the isolation the
    probe's ``git init`` would create the decoy's Git directory.
    """
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    environment = {
        **os.environ,
        "GIT_DIR": str(decoy / ".git"),
        "GIT_WORK_TREE": str(decoy),
    }

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", PROBE],
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "git location isolation: dropped GIT_DIR, GIT_WORK_TREE" in completed.stdout
    assert "1 passed" in completed.stdout
    assert list(decoy.iterdir()) == []
