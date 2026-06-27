# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the quantum-advantage provenance helpers
"""Fallback tests for the quantum-advantage benchmark provenance helpers.

Covers the git-commit failure fallback, the empty-argv command default and the
absent-qiskit dependency-version fallback.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

import pytest

from scpn_quantum_control.benchmarks.quantum_advantage import (
    _default_command,
    _default_dependency_versions,
    _git_commit,
    _resolve_git_executable,
)


def test_git_commit_uses_resolved_absolute_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The benchmark provenance git probe should launch an admitted executable."""
    seen: list[Sequence[str]] = []

    def fake_run(command: Sequence[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout="deadbeef\n",
            stderr="",
        )

    monkeypatch.setattr(
        "scpn_quantum_control.benchmarks.quantum_advantage.subprocess.run",
        fake_run,
    )

    assert _git_commit() == "deadbeef"
    assert seen
    executable = Path(seen[0][0])
    assert executable.is_absolute()
    assert executable.name == "git"


def test_git_commit_falls_back_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing git invocation yields the 'unknown' commit marker."""

    def _raise(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr("scpn_quantum_control.benchmarks.quantum_advantage.subprocess.run", _raise)
    assert _git_commit() == "unknown"


def test_git_commit_falls_back_when_git_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing git executable yields the unknown commit marker."""
    monkeypatch.setattr(
        "scpn_quantum_control.benchmarks.quantum_advantage.shutil.which",
        lambda _: None,
    )

    assert _git_commit() == "unknown"


def test_git_resolution_rejects_unresolvable_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken git lookup result is rejected before subprocess launch."""
    monkeypatch.setattr(
        "scpn_quantum_control.benchmarks.quantum_advantage.shutil.which",
        lambda _: "\0bad",
    )

    assert _resolve_git_executable() is None


def test_git_resolution_rejects_non_executable_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-executable git path is rejected before subprocess launch."""
    fake_git = tmp_path / "git"
    fake_git.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_git.chmod(0o600)
    monkeypatch.setattr(
        "scpn_quantum_control.benchmarks.quantum_advantage.shutil.which",
        lambda _: str(fake_git),
    )

    assert _resolve_git_executable() is None


def test_git_commit_falls_back_after_subprocess_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subprocess error after admission yields the unknown commit marker."""

    def _raise(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise OSError("cannot execute git")

    monkeypatch.setattr("scpn_quantum_control.benchmarks.quantum_advantage.subprocess.run", _raise)

    assert _git_commit() == "unknown"


def test_default_command_for_empty_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no argv the default command falls back to 'python'."""
    monkeypatch.setattr(sys, "argv", [])
    assert _default_command() == "python"


def test_dependency_versions_without_qiskit(monkeypatch: pytest.MonkeyPatch) -> None:
    """When qiskit cannot be imported its version is reported as not installed."""
    monkeypatch.setitem(sys.modules, "qiskit", None)
    versions = _default_dependency_versions()
    assert versions["qiskit"] == "not installed"
