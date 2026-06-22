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
from typing import Any

import pytest

from scpn_quantum_control.benchmarks.quantum_advantage import (
    _default_command,
    _default_dependency_versions,
    _git_commit,
)


def test_git_commit_falls_back_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing git invocation yields the 'unknown' commit marker."""

    def _raise(*_args: Any, **_kwargs: Any) -> str:
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(
        "scpn_quantum_control.benchmarks.quantum_advantage.subprocess.check_output", _raise
    )
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
