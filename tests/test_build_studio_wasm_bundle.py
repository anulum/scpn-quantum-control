# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — studio WASM bundle tool tests
"""Tests for the studio WASM bundle build + deploy manifest tool (ST-08)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import build_studio_wasm_bundle as bundle_tool  # noqa: E402


def _fake_bundle(tmp_path: Path) -> Path:
    """Create a minimal fake portal bundle with all tracked artefacts."""
    dist = tmp_path / "dist"
    (dist / "wasm").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (dist / "remoteEntry.js").write_text("export{};", encoding="utf-8")
    (dist / "wasm" / bundle_tool.KERNEL_WASM_NAME).write_bytes(b"\0asm-fake")
    (dist / "wasm" / bundle_tool.PROGRAM_AD_WASM_NAME).write_bytes(b"\0asm-program-ad")
    return dist


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    """The digest helper content-addresses exact bytes with a sha256 prefix."""
    payload = tmp_path / "artefact.bin"
    payload.write_bytes(b"studio-wasm")
    expected = hashlib.sha256(b"studio-wasm").hexdigest()
    assert bundle_tool.sha256_file(payload) == f"sha256:{expected}"


def test_kernel_crate_version_reads_the_committed_manifest() -> None:
    """The committed kernel crate carries a non-empty package version."""
    if not (bundle_tool.KERNEL_CRATE_DIR / "Cargo.toml").is_file():
        pytest.skip("kernel crate manifest is not present in this environment")
    version = bundle_tool.kernel_crate_version()
    assert version
    assert version.count(".") == 2


def test_kernel_crate_version_fails_closed_on_missing_version(tmp_path: Path) -> None:
    """A crate manifest without a version raises instead of defaulting."""
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "x"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no package version"):
        bundle_tool.kernel_crate_version(tmp_path)


def test_build_wasm_kernel_invokes_cargo_with_the_locked_wasm_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The build shells out to cargo with --locked and the wasm32 target."""
    recorded: dict[str, object] = {}
    artefact_dir = tmp_path / "target" / bundle_tool.WASM_TARGET / "release"
    artefact_dir.mkdir(parents=True)
    (artefact_dir / bundle_tool.KERNEL_WASM_NAME).write_bytes(b"\0asm")

    def fake_run(args: list[str], **kwargs: object) -> None:
        recorded["args"] = args
        recorded["cwd"] = kwargs.get("cwd")
        recorded["check"] = kwargs.get("check")

    monkeypatch.setattr(subprocess, "run", fake_run)
    artefact = bundle_tool.build_wasm_kernel(tmp_path)
    assert artefact.name == bundle_tool.KERNEL_WASM_NAME
    assert recorded["args"] == [
        "cargo",
        "build",
        "--release",
        "--locked",
        "--target",
        bundle_tool.WASM_TARGET,
    ]
    assert recorded["cwd"] == tmp_path
    assert recorded["check"] is True


def test_build_wasm_kernel_fails_closed_on_missing_artefact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A cargo run that yields no artefact raises instead of continuing."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    with pytest.raises(ValueError, match="no artefact"):
        bundle_tool.build_wasm_kernel(tmp_path)


def test_ship_wasm_requires_an_existing_bundle(tmp_path: Path) -> None:
    """Shipping into a missing portal bundle fails closed."""
    artefact = tmp_path / bundle_tool.KERNEL_WASM_NAME
    artefact.write_bytes(b"\0asm")
    with pytest.raises(ValueError, match="portal bundle does not exist"):
        bundle_tool.ship_wasm_into_bundle(artefact, tmp_path / "missing-dist")


def test_ship_wasm_copies_bytes_into_the_bundle(tmp_path: Path) -> None:
    """The kernel bytes land verbatim under dist/wasm/."""
    dist = _fake_bundle(tmp_path)
    artefact = tmp_path / bundle_tool.KERNEL_WASM_NAME
    artefact.write_bytes(b"\0asm-built")
    shipped = bundle_tool.ship_wasm_into_bundle(artefact, dist)
    assert shipped.read_bytes() == b"\0asm-built"
    assert shipped.parent.name == "wasm"


def test_deploy_manifest_digests_every_tracked_artefact(tmp_path: Path) -> None:
    """The manifest carries one digest row per shipped artefact."""
    dist = _fake_bundle(tmp_path)
    manifest = bundle_tool.build_deploy_manifest(
        dist,
        toolchain="rustc 1.96.0",
        crate_version="0.1.0",
    )
    assert manifest["schema"] == bundle_tool.DEPLOY_MANIFEST_SCHEMA
    assert manifest["studio"] == "scpn-quantum-control"
    artefacts = manifest["artifacts"]
    assert isinstance(artefacts, list)
    paths = [row["path"] for row in artefacts]
    assert paths == [
        "index.html",
        "remoteEntry.js",
        f"wasm/{bundle_tool.KERNEL_WASM_NAME}",
        f"wasm/{bundle_tool.PROGRAM_AD_WASM_NAME}",
    ]
    for row in artefacts:
        assert str(row["sha256"]).startswith("sha256:")
        assert isinstance(row["bytes"], int) and row["bytes"] > 0
    assert "timestamp" not in json.dumps(manifest)


def test_deploy_manifest_fails_closed_on_a_partial_bundle(tmp_path: Path) -> None:
    """A bundle missing a tracked artefact is never manifest-signed."""
    dist = _fake_bundle(tmp_path)
    (dist / "remoteEntry.js").unlink()
    with pytest.raises(ValueError, match="bundle artefact missing: remoteEntry.js"):
        bundle_tool.build_deploy_manifest(dist, toolchain="rustc", crate_version="0.1.0")


def test_write_deploy_manifest_emits_sorted_json(tmp_path: Path) -> None:
    """The manifest file is deterministic sorted JSON with a trailing newline."""
    dist = _fake_bundle(tmp_path)
    manifest = bundle_tool.build_deploy_manifest(
        dist,
        toolchain="rustc 1.96.0",
        crate_version="0.1.0",
    )
    path = bundle_tool.write_deploy_manifest(manifest, dist)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == manifest
    assert path.name == bundle_tool.DEPLOY_MANIFEST_NAME


def test_rustc_version_reports_the_local_toolchain() -> None:
    """The toolchain probe returns the exact rustc version string."""
    if shutil.which("rustc") is None:
        pytest.skip("rustc is not installed in this environment")
    version = bundle_tool.rustc_version()
    assert version.startswith("rustc ")


def test_main_builds_ships_and_manifests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI wires build, ship, and manifest together for both kernels."""
    dist = _fake_bundle(tmp_path)

    build_dir = tmp_path / "built"
    build_dir.mkdir()

    def fake_build(
        crate_dir: Path = bundle_tool.KERNEL_CRATE_DIR,
        wasm_name: str = bundle_tool.KERNEL_WASM_NAME,
    ) -> Path:
        source = build_dir / wasm_name
        source.write_bytes(b"\0asm-" + wasm_name.encode())
        return source

    monkeypatch.setattr(bundle_tool, "build_wasm_kernel", fake_build)
    monkeypatch.setattr(bundle_tool, "rustc_version", lambda: "rustc 1.96.0")
    monkeypatch.setattr(bundle_tool, "kernel_crate_version", lambda: "0.1.0")
    exit_code = bundle_tool.main(["--dist-dir", str(dist)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "deploy-manifest.json" in captured.out
    manifest = json.loads((dist / bundle_tool.DEPLOY_MANIFEST_NAME).read_text(encoding="utf-8"))
    rows = {row["path"]: row["sha256"] for row in manifest["artifacts"]}
    for name in (bundle_tool.KERNEL_WASM_NAME, bundle_tool.PROGRAM_AD_WASM_NAME):
        assert (
            rows[f"wasm/{name}"]
            == "sha256:" + hashlib.sha256(b"\0asm-" + name.encode()).hexdigest()
        )


def test_real_kernel_builds_when_the_wasm_target_is_installed() -> None:
    """End-to-end: the committed crate builds for wasm32 when available."""
    if shutil.which("rustup") is None:
        pytest.skip("rustup is not installed in this environment")
    probe = subprocess.run(
        ["rustup", "target", "list", "--installed"],
        capture_output=True,
        text=True,
        check=False,
    )
    if bundle_tool.WASM_TARGET not in probe.stdout:
        pytest.skip("wasm32-unknown-unknown target not installed")
    artefact = bundle_tool.build_wasm_kernel()
    assert artefact.stat().st_size > 0
