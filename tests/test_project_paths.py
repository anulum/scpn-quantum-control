# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project path tests
"""Tests for repository data-resource path resolution."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control._paths import DATA_ROOT_ENV, project_data_path, project_data_root


def test_project_data_root_prefers_explicit_environment_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    marker = Path("data/public_application_benchmarks")
    (tmp_path / marker).mkdir(parents=True)
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))

    assert project_data_root(str(marker)) == tmp_path
    assert project_data_path(str(marker)) == tmp_path / marker


def test_project_data_root_falls_back_to_current_working_tree(
    monkeypatch,
    tmp_path: Path,
) -> None:
    marker = Path("docker-only-fixture/marker.txt")
    (tmp_path / marker.parent).mkdir(parents=True)
    (tmp_path / marker).write_text("fixture\n", encoding="utf-8")
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    assert project_data_root(str(marker)) == tmp_path
