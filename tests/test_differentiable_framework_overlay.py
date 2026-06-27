# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Framework Overlay Tests
"""Tests for reproducible optional framework overlay manifests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scpn_quantum_control.differentiable_framework_overlay import (
    CPU_FRAMEWORK_WHEELS,
    FrameworkOverlayManifest,
    build_framework_overlay_manifest,
    framework_overlay_pythonpath,
    install_framework_overlay,
    verify_framework_overlay_manifest,
)


def test_framework_overlay_manifest_uses_explicit_ext4_backed_target(tmp_path: Path) -> None:
    """The manifest should record an explicit CPU-only overlay target."""
    overlay = tmp_path / "scpn-qc-framework-site-py312"
    manifest = build_framework_overlay_manifest(overlay_path=overlay, python_version="3.12")

    assert isinstance(manifest, FrameworkOverlayManifest)
    assert manifest.overlay_path == overlay
    assert manifest.python_version == "3.12"
    assert manifest.install_command[:5] == (
        "python",
        "-m",
        "pip",
        "install",
        "--target",
    )
    assert str(overlay) in manifest.install_command
    assert manifest.cpu_wheels == CPU_FRAMEWORK_WHEELS
    assert "jax[cuda12]" not in " ".join(manifest.install_command)
    assert "tensorflow-cpu" in manifest.install_command
    assert manifest.wheel_sources["torch"].endswith("/cpu")
    assert manifest.verification_status == "not_verified"
    assert manifest.pythonpath == str(overlay)


def test_framework_overlay_manifest_round_trips_and_reports_missing_state(tmp_path: Path) -> None:
    """The manifest should round-trip and report missing overlay state."""
    overlay = tmp_path / "overlay"
    manifest = build_framework_overlay_manifest(overlay_path=overlay, python_version="3.12")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    loaded = FrameworkOverlayManifest.from_json(manifest_path)
    assert loaded == manifest

    missing = verify_framework_overlay_manifest(manifest_path)
    assert not missing.ready
    assert missing.status == "missing_overlay_path"
    assert missing.pythonpath == str(overlay)
    assert "PYTHONPATH" in missing.message

    overlay.mkdir()
    for package in ("jax", "torch", "tensorflow", "pennylane"):
        (overlay / package).mkdir()
    ready = verify_framework_overlay_manifest(manifest_path)

    assert ready.ready
    assert ready.status == "ready"
    assert ready.pythonpath == str(overlay)


def test_framework_overlay_install_records_versions_without_hidden_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installer should record versions discovered from the target overlay."""
    overlay = tmp_path / "overlay"
    calls: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], *, check: bool) -> None:
        assert check
        calls.append(command)
        overlay.mkdir(parents=True, exist_ok=True)
        for package in ("jax", "torch", "tensorflow", "pennylane"):
            (overlay / package).mkdir(exist_ok=True)
            dist = overlay / f"{package}-1.2.3.dist-info"
            dist.mkdir(exist_ok=True)
            (dist / "METADATA").write_text("Name: test\nVersion: 1.2.3\n", encoding="utf-8")

    monkeypatch.setattr(subprocess, "run", fake_run)

    manifest = install_framework_overlay(overlay)

    assert len(calls) == 2
    assert any("jax[cpu]" in command for command in calls)
    assert any("tensorflow-cpu" in command for command in calls)
    assert any("pennylane" in command for command in calls)
    assert any("torch" in command for command in calls)
    assert not any("jax[cuda12]" in " ".join(command) for command in calls)
    assert manifest.verification_status == "ready"
    assert manifest.package_versions == {
        "jax": "1.2.3",
        "torch": "1.2.3",
        "tensorflow": "1.2.3",
        "pennylane": "1.2.3",
    }


def test_framework_overlay_install_rejects_relative_path_before_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The installer should reject relative targets before invoking pip."""

    def fail_run(command: tuple[str, ...], *, check: bool) -> None:
        raise AssertionError(f"unexpected subprocess call: {command!r}, check={check!r}")

    monkeypatch.setattr(subprocess, "run", fail_run)

    with pytest.raises(ValueError, match="absolute directory path"):
        install_framework_overlay(Path("relative-overlay"))


def test_framework_overlay_install_rejects_existing_file_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installer should reject file targets before invoking pip."""
    target = tmp_path / "overlay-file"
    target.write_text("not a directory\n", encoding="utf-8")

    def fail_run(command: tuple[str, ...], *, check: bool) -> None:
        raise AssertionError(f"unexpected subprocess call: {command!r}, check={check!r}")

    monkeypatch.setattr(subprocess, "run", fail_run)

    with pytest.raises(ValueError, match="must be a directory"):
        install_framework_overlay(target)


def test_framework_overlay_pythonpath_requires_existing_manifest(tmp_path: Path) -> None:
    """The PYTHONPATH helper should require an existing manifest file."""
    with pytest.raises(FileNotFoundError):
        framework_overlay_pythonpath(tmp_path / "missing.json")
