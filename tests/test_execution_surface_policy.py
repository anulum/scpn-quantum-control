# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — execution-surface policy tests
"""Tests for non-executing notebook and CLI execution-surface policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_quantum_control import bench_cli
from scpn_quantum_control.execution_surface import (
    EXECUTION_SURFACE_CLASSIFICATIONS,
    ExecutionSurfaceFinding,
    evaluate_execution_surface_manifest,
    find_unmanifested_high_risk_surfaces,
    iter_execution_surface_paths,
    load_execution_surface_manifest,
    scan_execution_surface_path,
)


def test_every_bench_harness_has_explicit_execution_policy() -> None:
    for harness in bench_cli.HARNESS_REGISTRY:
        assert harness.policy.classification == "trusted_offline_executable"
        assert harness.policy.network_allowed is False
        assert harness.policy.credential_allowed is False
        assert harness.policy.hardware_submission_allowed is False
        assert harness.policy.subprocess_allowed is True
        assert harness.policy.ci_blocking is True
        assert harness.policy.allowed_write_roots


def test_bench_harness_policy_rejects_missing_or_out_of_tree_script() -> None:
    missing = bench_cli.Harness("missing", "scripts/does_not_exist.py", frozenset({"unit"}))
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        bench_cli._validate_harness_policy(missing)

    escaped = bench_cli.Harness("escaped", "../outside.py", frozenset({"unit"}))
    with pytest.raises(ValueError, match="inside repository"):
        bench_cli._validate_harness_policy(escaped)


def test_bench_harness_policy_rejects_untrusted_execution() -> None:
    policy = bench_cli.ExecutionSurfacePolicy(
        classification="untrusted_user",
        network_allowed=False,
        credential_allowed=False,
        hardware_submission_allowed=False,
        allowed_write_roots=("data/rust_vqe_methods",),
        subprocess_allowed=False,
        ci_blocking=True,
    )
    harness = bench_cli.Harness(
        "untrusted",
        "scripts/benchmark_rust_core_methods.py",
        frozenset({"unit"}),
        policy=policy,
    )

    with pytest.raises(PermissionError, match="not executable"):
        bench_cli._validate_harness_policy(harness)


def test_notebook_scanner_reports_known_ibm_submission_surface() -> None:
    findings = scan_execution_surface_path(Path("notebooks/39_ibm_hardware_v2.ipynb"))
    rules = {finding.rule for finding in findings}

    assert "credential_read" in rules
    assert "hardware_submission" in rules


def test_notebook_scanner_reports_colab_shell_install_surface() -> None:
    findings = scan_execution_surface_path(
        Path("notebooks/colab/kaggle_protein_folding_kuramoto.ipynb")
    )

    assert any(finding.rule == "shell_magic" for finding in findings)


def test_notebook_scanner_reports_kaggle_public_push_surface() -> None:
    findings = scan_execution_surface_path(Path("notebooks/kaggle_push/push_all.sh"))

    assert any(finding.rule == "external_publication" for finding in findings)
    assert any("kaggle kernels push" in finding.evidence for finding in findings)


def test_execution_surface_finding_is_stable_machine_readable_record() -> None:
    finding = ExecutionSurfaceFinding(
        path="notebooks/example.ipynb",
        rule="network_access",
        line=12,
        evidence="requests.get",
    )

    assert finding.to_dict() == {
        "path": "notebooks/example.ipynb",
        "rule": "network_access",
        "line": 12,
        "evidence": "requests.get",
    }


def test_default_execution_surface_manifest_accepts_classified_surfaces() -> None:
    violations = evaluate_execution_surface_manifest(Path.cwd())

    assert violations == ()


def test_execution_surface_manifest_uses_declared_classifications() -> None:
    entries = load_execution_surface_manifest()

    assert {entry.path for entry in entries} >= {
        "notebooks/39_ibm_hardware_v2.ipynb",
        "notebooks/colab/kaggle_protein_folding_kuramoto.ipynb",
        "notebooks/kaggle_push/push_all.sh",
    }
    assert all(entry.classification in EXECUTION_SURFACE_CLASSIFICATIONS for entry in entries)
    assert all(entry.ci_blocking for entry in entries)


def test_execution_surface_manifest_covers_high_risk_inventory() -> None:
    paths = {
        path.relative_to(Path.cwd()).as_posix()
        for path in iter_execution_surface_paths(Path.cwd())
    }

    assert "notebooks/14_dla_parity_ibm_hardware.ipynb" in paths
    assert "notebooks/kaggle_push/kaggle_protein_folding_kuramoto.py" in paths
    assert find_unmanifested_high_risk_surfaces(Path.cwd()) == ()


def test_execution_surface_manifest_blocks_unapproved_findings(tmp_path: Path) -> None:
    notebook = tmp_path / "network.ipynb"
    notebook.write_text(
        """
{
  "cells": [
    {"cell_type": "code", "source": ["import requests\\n", "requests.get('https://example.org')\\n"]}
  ]
}
""".strip(),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        f"""
[[surface]]
path = "{notebook.name}"
classification = "trusted_static"
allowed_rules = []
ci_blocking = true
""".strip(),
        encoding="utf-8",
    )

    violations = evaluate_execution_surface_manifest(tmp_path, manifest)

    assert len(violations) == 1
    assert violations[0].reason == "unapproved execution-surface finding"
    assert violations[0].rule == "network_access"
    assert violations[0].line is not None


def test_execution_surface_manifest_rejects_paths_outside_repo(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        """
[[surface]]
path = "../outside.ipynb"
classification = "trusted_static"
allowed_rules = []
ci_blocking = true
""".strip(),
        encoding="utf-8",
    )

    violations = evaluate_execution_surface_manifest(tmp_path, manifest)

    assert len(violations) == 1
    assert violations[0].reason == "path escapes repository"
