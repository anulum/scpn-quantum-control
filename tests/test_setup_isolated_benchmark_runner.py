# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Isolated Benchmark Runner Setup Tests
"""Tests for the isolated benchmark runner setup helper."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

import tools.setup_isolated_benchmark_runner as runner_setup


def test_runner_setup_plan_requires_isolated_labels(tmp_path: Path) -> None:
    plan = runner_setup.build_runner_setup_plan(
        repo="anulum/scpn-quantum-control",
        runner_dir=tmp_path / "runner",
        runner_name="isolated-test",
    )

    assert plan.labels == ("self-hosted", "linux", "isolated-benchmark")
    assert plan.download_url.endswith(".tar.gz")
    assert plan.to_dict()["claim_boundary"]


def test_runner_setup_plan_rejects_missing_required_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="labels must include"):
        runner_setup.build_runner_setup_plan(
            repo="anulum/scpn-quantum-control",
            runner_dir=tmp_path / "runner",
            runner_name="bad",
            labels=("self-hosted", "linux"),
        )


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.tar.gz"
    outside = tmp_path / "outside.txt"
    outside.write_text("bad", encoding="utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(outside, arcname="../outside.txt")

    with (
        tarfile.open(archive_path, "r:gz") as archive,
        pytest.raises(ValueError, match="unsafe runner archive member"),
    ):
        runner_setup._safe_extract(archive, tmp_path / "target")
