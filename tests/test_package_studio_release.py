# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — studio release packaging tests (ST-36)
"""Tests for the pull-deploy Release packaging tool (ST-36).

The packer publishes the credential-free pull-deploy pair: a deterministic
release tarball plus a standalone manifest carrying ``studio_version`` and
the tarball's content digest. Every fail-closed path (missing bundle, stale
digests, tag/version mismatch) is exercised here.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import package_studio_release as release_tool  # noqa: E402


def _fake_bundle(tmp_path: Path) -> Path:
    """Create a fake built bundle with a consistent deploy manifest."""
    dist = tmp_path / "dist"
    (dist / "wasm").mkdir(parents=True)
    (dist / "assets").mkdir()
    (dist / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (dist / "remoteEntry.js").write_text("export{};", encoding="utf-8")
    (dist / "assets" / "panel.js").write_text("render();", encoding="utf-8")
    (dist / "wasm" / "kernel.wasm").write_bytes(b"\0asm-kernel")
    manifest = {
        "schema": "scpn_qc_studio_deploy_manifest_v1",
        "studio": "scpn-quantum-control",
        "kernel_crate": "scpn_quantum_engine/studio_wasm_kernel",
        "kernel_crate_version": "0.1.0",
        "wasm_target": "wasm32-unknown-unknown",
        "toolchain": "rustc 1.96.0",
        "artifacts": [
            {
                "path": path,
                "sha256": "sha256:" + hashlib.sha256((dist / path).read_bytes()).hexdigest(),
                "bytes": (dist / path).stat().st_size,
            }
            for path in ("index.html", "remoteEntry.js", "wasm/kernel.wasm")
        ],
    }
    (dist / release_tool.DEPLOY_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dist


def test_sha256_bytes_matches_hashlib() -> None:
    """The digest helper content-addresses exact bytes with a sha256 prefix."""
    expected = hashlib.sha256(b"studio-release").hexdigest()
    assert release_tool.sha256_bytes(b"studio-release") == f"sha256:{expected}"


def test_studio_version_reads_the_committed_project_metadata() -> None:
    """The canonical version comes from the committed pyproject."""
    version = release_tool.studio_version()
    assert version
    assert version.count(".") == 2


def test_studio_version_fails_closed_on_missing_version(tmp_path: Path) -> None:
    """Project metadata without a version raises instead of defaulting."""
    metadata = tmp_path / "pyproject.toml"
    metadata.write_text('[project]\nname = "x"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no version"):
        release_tool.studio_version(metadata)


def test_load_deploy_manifest_verifies_every_row(tmp_path: Path) -> None:
    """A consistent bundle loads with its artefact rows re-verified."""
    dist = _fake_bundle(tmp_path)
    manifest = release_tool.load_deploy_manifest(dist)
    assert manifest["studio"] == "scpn-quantum-control"
    assert len(manifest["artifacts"]) == 3


def test_load_deploy_manifest_fails_closed_on_missing_manifest(tmp_path: Path) -> None:
    """A bundle without its deploy manifest is never packaged."""
    with pytest.raises(ValueError, match="deploy manifest missing"):
        release_tool.load_deploy_manifest(tmp_path / "missing-dist")


def test_load_deploy_manifest_fails_closed_on_empty_rows(tmp_path: Path) -> None:
    """A manifest without artefact rows is refused."""
    dist = _fake_bundle(tmp_path)
    (dist / release_tool.DEPLOY_MANIFEST_NAME).write_text(
        json.dumps({"artifacts": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no artefact rows"):
        release_tool.load_deploy_manifest(dist)


def test_load_deploy_manifest_fails_closed_on_missing_artefact(tmp_path: Path) -> None:
    """A manifested file absent from the bundle is refused."""
    dist = _fake_bundle(tmp_path)
    (dist / "remoteEntry.js").unlink()
    with pytest.raises(ValueError, match="missing from the bundle: remoteEntry.js"):
        release_tool.load_deploy_manifest(dist)


def test_load_deploy_manifest_fails_closed_on_digest_drift(tmp_path: Path) -> None:
    """A stale bundle (bytes changed after manifesting) is refused."""
    dist = _fake_bundle(tmp_path)
    (dist / "remoteEntry.js").write_text("export{}; // drifted", encoding="utf-8")
    with pytest.raises(ValueError, match="stale bundle: digest drift on remoteEntry.js"):
        release_tool.load_deploy_manifest(dist)


def test_bundle_file_table_walks_the_tree_and_excludes_the_manifest(tmp_path: Path) -> None:
    """The file table covers every deployable file, sorted, manifest excluded."""
    dist = _fake_bundle(tmp_path)
    rows = release_tool.bundle_file_table(dist)
    paths = [row["path"] for row in rows]
    assert paths == ["assets/panel.js", "index.html", "remoteEntry.js", "wasm/kernel.wasm"]
    assert release_tool.DEPLOY_MANIFEST_NAME not in paths
    for row in rows:
        assert str(row["sha256"]).startswith("sha256:")
        assert isinstance(row["bytes"], int) and row["bytes"] > 0


def test_bundle_file_table_fails_closed_on_an_empty_tree(tmp_path: Path) -> None:
    """An empty bundle tree is never packaged."""
    empty = tmp_path / "empty-dist"
    empty.mkdir()
    with pytest.raises(ValueError, match="no deployable files"):
        release_tool.bundle_file_table(empty)


def test_pack_release_tarball_is_deterministic_and_complete(tmp_path: Path) -> None:
    """Packing the same tree twice yields identical bytes; contents verbatim."""
    dist = _fake_bundle(tmp_path)
    rows = release_tool.bundle_file_table(dist)
    first = release_tool.pack_release_tarball(dist, rows, tmp_path / "out-a")
    second = release_tool.pack_release_tarball(dist, rows, tmp_path / "out-b")
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, mode="r:gz") as archive:
        names = archive.getnames()
        assert names == [str(row["path"]) for row in rows]
        for member in archive.getmembers():
            assert member.uid == 0 and member.gid == 0
            assert member.uname == "" and member.gname == ""
            assert member.mtime == 0
        extracted = archive.extractfile("remoteEntry.js")
        assert extracted is not None
        assert extracted.read() == (dist / "remoteEntry.js").read_bytes()


def test_build_release_manifest_carries_the_contract_fields(tmp_path: Path) -> None:
    """The manifest carries version, tag, bundle digest, files, provenance."""
    dist = _fake_bundle(tmp_path)
    deploy_manifest = release_tool.load_deploy_manifest(dist)
    rows = release_tool.bundle_file_table(dist)
    manifest = release_tool.build_release_manifest(
        version="0.10.0",
        bundle_digest="sha256:" + "0" * 64,
        bundle_bytes=1234,
        files=rows,
        deploy_manifest=deploy_manifest,
    )
    assert manifest["schema"] == release_tool.RELEASE_MANIFEST_SCHEMA
    assert manifest["studio"] == release_tool.STUDIO_ID
    assert manifest["studio_version"] == "0.10.0"
    assert manifest["release_tag"] == "studio-remote-v0.10.0"
    bundle = manifest["bundle"]
    assert isinstance(bundle, dict)
    assert bundle["name"] == release_tool.RELEASE_TARBALL_NAME
    assert bundle["sha256"] == "sha256:" + "0" * 64
    assert bundle["bytes"] == 1234
    assert manifest["toolchain"] == "rustc 1.96.0"
    assert manifest["kernel_crate_version"] == "0.1.0"
    assert manifest["files"] == rows
    assert "timestamp" not in json.dumps(manifest)


def test_write_release_manifest_emits_sorted_json(tmp_path: Path) -> None:
    """The manifest file is deterministic sorted JSON with a trailing newline."""
    manifest: dict[str, object] = {"schema": release_tool.RELEASE_MANIFEST_SCHEMA, "b": 1, "a": 2}
    path = release_tool.write_release_manifest(manifest, tmp_path / "out")
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == manifest
    assert text.index('"a"') < text.index('"b"')
    assert path.name == release_tool.DEPLOY_MANIFEST_NAME


def test_main_packages_the_release_pair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI packs the tarball + manifest and the digests agree."""
    dist = _fake_bundle(tmp_path)
    out = tmp_path / "release"
    exit_code = release_tool.main(["--dist-dir", str(dist), "--out-dir", str(out)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "packed" in captured.out
    tarball = out / release_tool.RELEASE_TARBALL_NAME
    manifest = json.loads((out / release_tool.DEPLOY_MANIFEST_NAME).read_text(encoding="utf-8"))
    bundle = manifest["bundle"]
    assert bundle["sha256"] == "sha256:" + hashlib.sha256(tarball.read_bytes()).hexdigest()
    assert bundle["bytes"] == tarball.stat().st_size
    assert manifest["studio_version"] == release_tool.studio_version()
    # the tarball unpacks to exactly the manifested file set
    with tarfile.open(tarball, mode="r:gz") as archive:
        assert archive.getnames() == [row["path"] for row in manifest["files"]]


def test_main_accepts_a_matching_expected_version(tmp_path: Path) -> None:
    """The Release tag gate passes when the versions agree."""
    dist = _fake_bundle(tmp_path)
    out = tmp_path / "release"
    exit_code = release_tool.main(
        [
            "--dist-dir",
            str(dist),
            "--out-dir",
            str(out),
            "--expect-version",
            release_tool.studio_version(),
        ]
    )
    assert exit_code == 0


def test_main_fails_closed_on_a_version_mismatch(tmp_path: Path) -> None:
    """A Release tag that disagrees with the studio version is refused."""
    dist = _fake_bundle(tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        release_tool.main(["--dist-dir", str(dist), "--expect-version", "99.99.99"])
