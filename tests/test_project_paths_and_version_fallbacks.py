# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for project-path and version fallbacks
"""Tests for the defensive fallbacks in project-path and version resolution."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

import scpn_quantum_control
from scpn_quantum_control import _paths


def test_project_data_root_resolves_when_marker_present() -> None:
    """A marker that exists in the tree resolves to a containing root."""
    root = _paths.project_data_root("pyproject.toml")

    assert (root / "pyproject.toml").exists()


def test_project_data_path_joins_relative_resource() -> None:
    """A relative resource is joined onto the resolved project root."""
    resolved = _paths.project_data_path("pyproject.toml")

    assert resolved == _paths.project_data_root("pyproject.toml") / "pyproject.toml"
    assert resolved.exists()


def test_project_data_root_honours_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit data-root environment variable is used when set."""
    (tmp_path / "marker.txt").write_text("present", encoding="utf-8")
    monkeypatch.setenv(_paths.DATA_ROOT_ENV, str(tmp_path))

    assert _paths.project_data_root("marker.txt") == tmp_path.resolve()


def test_project_data_root_returns_fallback_when_marker_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No candidate matching the required marker falls back to the source root.

    This exercises the final ``return fallback`` guard: with the environment
    override cleared and a marker that exists nowhere up the tree, every
    candidate is rejected and the source-derived fallback is returned.
    """
    monkeypatch.delenv(_paths.DATA_ROOT_ENV, raising=False)
    fallback = Path(_paths.__file__).resolve().parents[2]

    result = _paths.project_data_root("__scpn_marker_that_exists_nowhere_9f3a__")

    assert result == fallback


def test_resolve_version_returns_distribution_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The installed distribution version is returned when metadata is present."""
    monkeypatch.setattr(scpn_quantum_control, "version", lambda _name: "9.9.9-test")

    assert scpn_quantum_control._resolve_version() == "9.9.9-test"


def test_resolve_version_falls_back_when_distribution_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local fallback is returned when distribution metadata is absent."""

    def _raise(_name: str) -> str:
        raise PackageNotFoundError(_name)

    monkeypatch.setattr(scpn_quantum_control, "version", _raise)

    assert scpn_quantum_control._resolve_version() == "0.0.0+local"
