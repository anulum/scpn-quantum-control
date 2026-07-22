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
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import scpn_quantum_control.differentiable_framework_overlay as overlay_module
from scpn_quantum_control.differentiable_framework_overlay import (
    CPU_FRAMEWORK_WHEELS,
    FrameworkOverlayManifest,
    build_framework_overlay_manifest,
    default_framework_overlay_path,
    framework_overlay_pythonpath,
    install_framework_overlay,
    verify_framework_overlay_manifest,
    verify_framework_overlay_path,
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


def test_framework_overlay_manifest_rejects_coercive_json(tmp_path: Path) -> None:
    """The claim manifest loader preserves schema and exact runtime types."""
    manifest = build_framework_overlay_manifest(overlay_path=tmp_path / "overlay")
    path = tmp_path / "manifest.json"

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        FrameworkOverlayManifest.from_json(path)

    payload = manifest.to_dict()
    path.write_text(json.dumps({**payload, "schema": "old"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema is unknown"):
        FrameworkOverlayManifest.from_json(path)

    path.write_text(json.dumps({**payload, "claim_boundary": "vague"}), encoding="utf-8")
    with pytest.raises(ValueError, match="claim boundary is not canonical"):
        FrameworkOverlayManifest.from_json(path)

    path.write_text(json.dumps({**payload, "overlay_path": 7}), encoding="utf-8")
    with pytest.raises(ValueError, match="overlay_path must be a non-empty string"):
        FrameworkOverlayManifest.from_json(path)

    path.write_text(json.dumps({**payload, "cpu_wheels": ["jax[cpu]", 7]}), encoding="utf-8")
    with pytest.raises(ValueError, match="cpu_wheels must contain non-empty strings"):
        FrameworkOverlayManifest.from_json(path)

    path.write_text(json.dumps({**payload, "wheel_sources": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="wheel_sources must map non-empty strings"):
        FrameworkOverlayManifest.from_json(path)


def test_framework_overlay_manifest_rejects_ambiguous_contract_fields(tmp_path: Path) -> None:
    """Direct construction cannot weaken CPU profile identity or provenance."""
    manifest = build_framework_overlay_manifest(overlay_path=tmp_path / "overlay")

    with pytest.raises(ValueError, match="overlay_path must be a Path"):
        replace(manifest, overlay_path=cast(Path, "bad"))
    with pytest.raises(ValueError, match="python_version must be a non-empty string"):
        replace(manifest, python_version=cast(str, 7))
    with pytest.raises(ValueError, match="install_command must contain non-empty strings"):
        replace(manifest, install_command=())
    with pytest.raises(ValueError, match="cpu_wheels must contain non-empty strings"):
        replace(manifest, cpu_wheels=cast(tuple[str, ...], ["jax[cpu]"]))
    with pytest.raises(ValueError, match="cpu_wheels must match"):
        replace(manifest, cpu_wheels=("jax[cpu]",))
    with pytest.raises(ValueError, match="wheel_sources must map non-empty strings"):
        replace(manifest, wheel_sources={"": "source"})
    with pytest.raises(ValueError, match="wheel_sources must cover"):
        replace(manifest, wheel_sources={"jax[cpu]": "PyPI"})
    with pytest.raises(ValueError, match="package_versions must map non-empty strings"):
        replace(manifest, package_versions={"jax": ""})
    with pytest.raises(ValueError, match="pythonpath must match overlay_path"):
        replace(manifest, pythonpath=str(tmp_path / "other"))
    with pytest.raises(ValueError, match="artifact_id must match"):
        replace(manifest, artifact_id="other")


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


def test_framework_overlay_defaults_follow_cache_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default manifests use the declared cache root and running Python minor."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    default_path = default_framework_overlay_path()
    manifest = build_framework_overlay_manifest()

    assert default_path == tmp_path / overlay_module.DEFAULT_OVERLAY_BASENAME
    assert manifest.overlay_path == default_path
    assert manifest.python_version == (f"{sys.version_info.major}.{sys.version_info.minor}")


def test_framework_overlay_rejects_null_and_root_paths() -> None:
    """Destructive or OS-invalid installation targets fail before filesystem use."""
    with pytest.raises(ValueError, match="null byte"):
        build_framework_overlay_manifest(overlay_path=Path("/tmp/invalid\x00overlay"))
    with pytest.raises(ValueError, match="filesystem root"):
        build_framework_overlay_manifest(overlay_path=Path(Path.cwd().anchor))


def test_framework_overlay_reports_partial_packages_and_unknown_version(tmp_path: Path) -> None:
    """Partial overlays retain missing roots and fail-closed version metadata."""
    overlay = tmp_path / "overlay"
    (overlay / "jax").mkdir(parents=True)
    (overlay / "jax-unknown.dist-info").mkdir()

    result = verify_framework_overlay_path(overlay, pythonpath="explicit-pythonpath")

    assert not result.ready
    assert result.status == "missing_packages"
    assert result.pythonpath == "explicit-pythonpath"
    assert result.missing_packages == ("torch", "tensorflow", "pennylane")
    assert result.package_versions == {}
    assert result.to_dict()["missing_packages"] == ["torch", "tensorflow", "pennylane"]


def test_framework_overlay_discovers_underscore_distribution_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Underscore package roots can use their matching dist-info spelling."""
    monkeypatch.setattr(
        overlay_module,
        "CPU_FRAMEWORK_PACKAGE_ROOTS",
        ("tensorflow_cpu",),
    )
    overlay = tmp_path / "overlay"
    (overlay / "tensorflow_cpu").mkdir(parents=True)
    dist_info = overlay / "tensorflow_cpu-2.3.4.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Name: tensorflow-cpu\nVersion: 2.3.4\n", encoding="utf-8")

    result = verify_framework_overlay_path(overlay)

    assert result.ready
    assert result.package_versions == {"tensorflow_cpu": "2.3.4"}


def test_framework_overlay_metadata_without_version_is_unknown(tmp_path: Path) -> None:
    """Present metadata without a Version field is reported as unknown."""
    overlay = tmp_path / "overlay"
    for package in ("jax", "torch", "tensorflow", "pennylane"):
        (overlay / package).mkdir(parents=True)
    dist_info = overlay / "jax-unknown.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Name: jax\n", encoding="utf-8")

    result = verify_framework_overlay_path(overlay)

    assert result.ready
    assert result.package_versions == {"jax": "unknown"}


def test_framework_overlay_cli_emits_and_verifies_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI emits its default manifest and returns fail-closed verify codes."""
    overlay = tmp_path / "overlay"
    manifest_path = overlay / "framework_overlay_manifest.json"

    assert overlay_module.main(["--overlay-path", str(overlay)]) == 0
    emitted = capsys.readouterr().out
    assert str(manifest_path) in emitted
    assert f"PYTHONPATH={overlay}" in emitted
    assert framework_overlay_pythonpath(manifest_path) == str(overlay)

    assert overlay_module.main(["--manifest-path", str(manifest_path), "--verify"]) == 2
    missing_payload = json.loads(capsys.readouterr().out)
    assert missing_payload["status"] == "missing_packages"

    for package in ("jax", "torch", "tensorflow", "pennylane"):
        (overlay / package).mkdir(parents=True)
    assert overlay_module.main(["--manifest-path", str(manifest_path), "--verify"]) == 0
    ready_payload = json.loads(capsys.readouterr().out)
    assert ready_payload["ready"] is True


def test_framework_overlay_cli_install_delegates_to_installer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The install flag delegates once and writes the returned manifest."""
    overlay = tmp_path / "overlay"
    manifest_path = tmp_path / "installed.json"
    installed_paths: list[Path] = []

    def fake_install(path: Path) -> FrameworkOverlayManifest:
        installed_paths.append(path)
        return build_framework_overlay_manifest(
            overlay_path=path,
            package_versions={"jax": "test"},
            verification_status="ready",
        )

    monkeypatch.setattr(overlay_module, "install_framework_overlay", fake_install)

    assert (
        overlay_module.main(
            [
                "--overlay-path",
                str(overlay),
                "--manifest-path",
                str(manifest_path),
                "--install",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert installed_paths == [overlay]
    assert FrameworkOverlayManifest.from_json(manifest_path).verification_status == "ready"
