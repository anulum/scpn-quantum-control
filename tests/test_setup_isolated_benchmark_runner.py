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


class _NonGitHubRunnerPlan(runner_setup.IsolatedRunnerSetupPlan):
    @property
    def download_url(self) -> str:
        """Return a non-GitHub URL for install-path validation tests."""
        return "https://example.invalid/actions-runner-linux-x64-2.330.0.tar.gz"


def test_runner_setup_plan_requires_isolated_labels(tmp_path: Path) -> None:
    """Build the default runner plan with the required isolation labels."""
    plan = runner_setup.build_runner_setup_plan(
        repo="anulum/scpn-quantum-control",
        runner_dir=tmp_path / "runner",
        runner_name="isolated-test",
    )

    assert plan.labels == ("self-hosted", "linux", "isolated-benchmark")
    assert plan.download_url.endswith(".tar.gz")
    assert plan.validated_download_url == plan.download_url
    assert plan.to_dict()["claim_boundary"]


def test_runner_setup_plan_rejects_missing_required_labels(tmp_path: Path) -> None:
    """Reject runner plans that would not match the isolated CI workflow."""
    with pytest.raises(ValueError, match="labels must include"):
        runner_setup.build_runner_setup_plan(
            repo="anulum/scpn-quantum-control",
            runner_dir=tmp_path / "runner",
            runner_name="bad",
            labels=("self-hosted", "linux"),
        )


def test_runner_setup_plan_rejects_unsafe_repo_slug(tmp_path: Path) -> None:
    """Reject repository slugs that would not be one GitHub API path segment."""
    with pytest.raises(ValueError, match="owner/repository"):
        runner_setup.IsolatedRunnerSetupPlan(
            repo="anulum/scpn-quantum-control/actions/runs",
            runner_dir=tmp_path / "runner",
            runner_name="bad",
            labels=runner_setup.DEFAULT_LABELS,
        )


def test_runner_setup_plan_rejects_unsafe_label(tmp_path: Path) -> None:
    """Reject labels that would change the comma-delimited runner label list."""
    with pytest.raises(ValueError, match="labels must contain"):
        runner_setup.IsolatedRunnerSetupPlan(
            repo="anulum/scpn-quantum-control",
            runner_dir=tmp_path / "runner",
            runner_name="bad",
            labels=("self-hosted", "linux", "isolated-benchmark,extra"),
        )


def test_runner_setup_plan_rejects_unsafe_runner_version(tmp_path: Path) -> None:
    """Reject release versions that could alter the expected archive path."""
    with pytest.raises(ValueError, match="runner_version"):
        runner_setup.IsolatedRunnerSetupPlan(
            repo="anulum/scpn-quantum-control",
            runner_dir=tmp_path / "runner",
            runner_name="bad",
            labels=runner_setup.DEFAULT_LABELS,
            runner_version="../2.330.0",
        )


def test_install_runner_rejects_non_github_download_url(tmp_path: Path) -> None:
    """Reject non-GitHub archives before any runner download starts."""
    plan = _NonGitHubRunnerPlan(
        repo="anulum/scpn-quantum-control",
        runner_dir=tmp_path / "runner",
        runner_name="bad-url",
        labels=runner_setup.DEFAULT_LABELS,
    )

    with pytest.raises(ValueError, match="hosted on github.com"):
        runner_setup.install_runner(plan)

    archive_path = tmp_path / "runner" / "actions-runner-linux-x64-2.330.0.tar.gz"
    assert not archive_path.exists()


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    """Reject archive members that would escape the runner directory."""
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
