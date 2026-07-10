#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — studio remote Release packaging (pull-deploy contract).
"""Package the built studio remote into the public pull-deploy Release triple.

The studio hosting contract is a credential-free PULL: this repository holds
zero deploy secrets. Instead it publishes three public GitHub Release assets
that the SCPN-STUDIO platform discovers, verifies fail-closed, and deploys:

* ``studio-deploy.json`` — the fleet-normative ``studio.deploy-bundle.v1``
  discovery descriptor (hosting contract §3, identical schema for every
  studio): studio id, ``studio_version``, the bundle asset name, and the
  tarball's BARE lowercase-hex ``bundle_sha256``. The reflector scans recent
  releases for exactly this asset name.
* ``scpn-quantum-control-studio-remote.tar.gz`` — the deployable bundle:
  the full ``vite build`` tree including the shipped WASM kernels, plus the
  committed schema-A ``manifest.json`` staged at the ARCHIVE ROOT (the
  stage gate requires it to name this studio; the box aggregation composes
  ``federation.json`` from it).
* ``deploy-manifest.json`` — this repo's richer release manifest carrying
  the ``sha256:``-prefixed bundle digest, one digest row per bundled file,
  and the kernel toolchain provenance.

The studio version comes from the Release tag (``--studio-version``, passed
by CI); the ``pyproject.toml`` project version is only the local default —
the studio remote iterates independently of the Python package.

The tarball is packed deterministically (sorted members, zeroed owner and
timestamps) so re-packing an identical tree yields an identical digest. The
packer fails closed on a missing or stale bundle: every artefact row of the
in-bundle deploy manifest written by ``build_studio_wasm_bundle.py`` is
re-hashed and must match before anything is packaged.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Final

import tomllib

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_DIST_DIR: Final[Path] = REPO_ROOT / "studio-web" / "dist"
DEFAULT_OUT_DIR: Final[Path] = REPO_ROOT / "studio-web" / "release"
DEFAULT_CAPABILITY_MANIFEST: Final[Path] = (
    REPO_ROOT / "docs" / "_generated" / "studio_manifest.json"
)
BUNDLE_MANIFEST_NAME: Final[str] = "manifest.json"
DEPLOY_MANIFEST_NAME: Final[str] = "deploy-manifest.json"
DEPLOY_DESCRIPTOR_NAME: Final[str] = "studio-deploy.json"
DEPLOY_BUNDLE_SCHEMA: Final[str] = "studio.deploy-bundle.v1"
RELEASE_TARBALL_NAME: Final[str] = "scpn-quantum-control-studio-remote.tar.gz"
RELEASE_MANIFEST_SCHEMA: Final[str] = "scpn_qc_studio_release_manifest_v1"
RELEASE_TAG_PREFIX: Final[str] = "studio-remote-v"
STUDIO_ID: Final[str] = "scpn-quantum-control"


def sha256_bytes(payload: bytes) -> str:
    """Return the ``sha256:<hex>`` digest of a byte payload.

    Parameters
    ----------
    payload
        Bytes to content-address.

    Returns
    -------
    str
        The prefixed SHA-256 hex digest.
    """
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def studio_version(pyproject_path: Path = REPO_ROOT / "pyproject.toml") -> str:
    """Return the canonical studio version from the project metadata.

    Parameters
    ----------
    pyproject_path
        The repository ``pyproject.toml``.

    Returns
    -------
    str
        The ``[project].version`` value.

    Raises
    ------
    ValueError
        If the project metadata carries no version.
    """
    metadata = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    version = metadata.get("project", {}).get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("project metadata carries no version")
    return version


def load_deploy_manifest(dist_dir: Path) -> dict[str, Any]:
    """Load the in-bundle deploy manifest and re-verify its artefact rows.

    Parameters
    ----------
    dist_dir
        The built portal bundle (``vite build`` output plus shipped kernels).

    Returns
    -------
    dict[str, Any]
        The parsed deploy manifest.

    Raises
    ------
    ValueError
        If the bundle or its manifest is missing, or if any manifest row's
        digest no longer matches the bytes on disk — a stale or tampered
        bundle is never packaged.
    """
    manifest_path = dist_dir / DEPLOY_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(
            f"deploy manifest missing: {manifest_path.as_posix()}"
            " (run vite build + build_studio_wasm_bundle.py first)"
        )
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("deploy manifest carries no artefact rows")
    for row in rows:
        path = dist_dir / str(row["path"])
        if not path.is_file():
            raise ValueError(f"manifested artefact missing from the bundle: {row['path']}")
        digest = sha256_bytes(path.read_bytes())
        if digest != row["sha256"]:
            raise ValueError(f"stale bundle: digest drift on {row['path']}")
    return manifest


def bundle_file_table(dist_dir: Path) -> list[dict[str, object]]:
    """Digest every deployable file in the bundle tree, sorted by path.

    Parameters
    ----------
    dist_dir
        The built portal bundle.

    Returns
    -------
    list[dict[str, object]]
        One ``{path, sha256, bytes}`` row per file. The in-bundle deploy
        manifest itself is excluded — the standalone release manifest is the
        verification document, so it never rides inside its own claim.

    Raises
    ------
    ValueError
        If the bundle tree contains no deployable files.
    """
    rows: list[dict[str, object]] = []
    for path in sorted(dist_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(dist_dir).as_posix()
        if relative == DEPLOY_MANIFEST_NAME:
            continue
        rows.append(
            {
                "path": relative,
                "sha256": sha256_bytes(path.read_bytes()),
                "bytes": path.stat().st_size,
            }
        )
    if not rows:
        raise ValueError(f"bundle tree carries no deployable files: {dist_dir.as_posix()}")
    return rows


def stage_capability_manifest(dist_dir: Path, source: Path = DEFAULT_CAPABILITY_MANIFEST) -> Path:
    """Stage the committed schema-A manifest as ``manifest.json`` at the root.

    The hosting contract's stage gate is fail-closed: the tarball root MUST
    carry a ``manifest.json`` naming this studio, and the box aggregation
    composes ``federation.json`` from it.

    Parameters
    ----------
    dist_dir
        The built portal bundle to stage into.
    source
        The committed schema-A studio manifest.

    Returns
    -------
    Path
        The staged ``manifest.json`` inside the bundle.

    Raises
    ------
    ValueError
        If the committed manifest is missing or does not name this studio.
    """
    if not source.is_file():
        raise ValueError(f"capability manifest missing: {source.as_posix()}")
    payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    named = payload.get("schema_a", {}).get("studio")
    if named != STUDIO_ID:
        raise ValueError(f"capability manifest names {named!r}, expected {STUDIO_ID!r}")
    staged = dist_dir / BUNDLE_MANIFEST_NAME
    staged.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return staged


def _deterministic_member(member: tarfile.TarInfo) -> tarfile.TarInfo:
    """Zero the owner and timestamp of a tar member for reproducible packing."""
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mtime = 0
    return member


def pack_release_tarball(dist_dir: Path, files: list[dict[str, object]], out_dir: Path) -> Path:
    """Pack the bundle tree into a deterministic gzip tarball.

    Parameters
    ----------
    dist_dir
        The built portal bundle.
    files
        The digest table from :func:`bundle_file_table`; exactly these files
        (and nothing else) ride in the tarball, in table order.
    out_dir
        Output directory for the tarball (created if absent).

    Returns
    -------
    Path
        The written tarball.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = out_dir / RELEASE_TARBALL_NAME
    with (
        tarball_path.open("wb") as raw,
        # gzip mtime is zeroed so identical trees produce identical bytes.
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for row in files:
            relative = str(row["path"])
            archive.add(
                dist_dir / relative,
                arcname=relative,
                recursive=False,
                filter=_deterministic_member,
            )
    return tarball_path


def build_release_manifest(
    *,
    version: str,
    bundle_digest: str,
    bundle_bytes: int,
    files: list[dict[str, object]],
    deploy_manifest: dict[str, Any],
) -> dict[str, object]:
    """Assemble the standalone release manifest for the pull-deploy reflector.

    Parameters
    ----------
    version
        The canonical studio version (must match the Release tag).
    bundle_digest
        The packed tarball's ``sha256:<hex>`` content digest.
    bundle_bytes
        The packed tarball's byte size.
    files
        The per-file digest table of the bundle tree.
    deploy_manifest
        The verified in-bundle deploy manifest; kernel provenance rides
        through verbatim.

    Returns
    -------
    dict[str, object]
        JSON-ready release manifest. Deterministic — no timestamps.
    """
    return {
        "schema": RELEASE_MANIFEST_SCHEMA,
        "studio": STUDIO_ID,
        "studio_version": version,
        "release_tag": f"{RELEASE_TAG_PREFIX}{version}",
        "bundle": {
            "name": RELEASE_TARBALL_NAME,
            "sha256": bundle_digest,
            "bytes": bundle_bytes,
        },
        "kernel_crate": deploy_manifest["kernel_crate"],
        "kernel_crate_version": deploy_manifest["kernel_crate_version"],
        "wasm_target": deploy_manifest["wasm_target"],
        "toolchain": deploy_manifest["toolchain"],
        "files": files,
    }


def build_deploy_descriptor(
    *, version: str, bundle_digest: str, bundle_bytes: int
) -> dict[str, object]:
    """Assemble the ``studio.deploy-bundle.v1`` discovery descriptor.

    This is the fleet-normative asset the platform reflector scans releases
    for; the schema is identical across every federated studio (hosting
    contract §3) and MUST NOT be forked. ``bundle_sha256`` is the BARE
    lowercase hex digest — no ``sha256:`` prefix.

    Parameters
    ----------
    version
        The studio version (equal to the Release tag's semver).
    bundle_digest
        The tarball digest in this repo's ``sha256:<hex>`` form; the prefix
        is stripped for the descriptor.
    bundle_bytes
        The tarball byte size (permitted extra field).

    Returns
    -------
    dict[str, object]
        JSON-ready deploy descriptor.
    """
    return {
        "schema": DEPLOY_BUNDLE_SCHEMA,
        "studio": STUDIO_ID,
        "studio_version": version,
        "bundle_asset": RELEASE_TARBALL_NAME,
        "bundle_sha256": bundle_digest.removeprefix("sha256:"),
        "bundle_bytes": bundle_bytes,
        "release_tag": f"{RELEASE_TAG_PREFIX}{version}",
    }


def write_deploy_descriptor(descriptor: dict[str, object], out_dir: Path) -> Path:
    """Write the deploy descriptor as deterministic sorted JSON.

    Parameters
    ----------
    descriptor
        The ``studio.deploy-bundle.v1`` payload.
    out_dir
        Output directory (created if absent).

    Returns
    -------
    Path
        The written ``studio-deploy.json`` path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / DEPLOY_DESCRIPTOR_NAME
    path.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_release_manifest(manifest: dict[str, object], out_dir: Path) -> Path:
    """Write the release manifest as deterministic sorted JSON.

    Parameters
    ----------
    manifest
        The release manifest payload.
    out_dir
        Output directory (created if absent).

    Returns
    -------
    Path
        The written manifest path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / DEPLOY_MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: package the release tarball + standalone manifest.

    Parameters
    ----------
    argv
        Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        ``0`` on success.

    Raises
    ------
    ValueError
        If the bundle is missing or stale, or if the capability manifest
        does not name this studio.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=DEFAULT_DIST_DIR,
        help="built portal bundle directory (vite build + WASM bundle output)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output directory for the release tarball + manifest + descriptor",
    )
    parser.add_argument(
        "--studio-version",
        default=None,
        help="studio version for this release (CI passes the tag's semver;"
        " defaults to the pyproject project version)",
    )
    parser.add_argument(
        "--capability-manifest",
        type=Path,
        default=DEFAULT_CAPABILITY_MANIFEST,
        help="committed schema-A studio manifest staged into the tarball root",
    )
    args = parser.parse_args(argv)
    version = args.studio_version if args.studio_version is not None else studio_version()
    deploy_manifest = load_deploy_manifest(args.dist_dir)
    stage_capability_manifest(args.dist_dir, args.capability_manifest)
    files = bundle_file_table(args.dist_dir)
    tarball_path = pack_release_tarball(args.dist_dir, files, args.out_dir)
    payload = tarball_path.read_bytes()
    manifest = build_release_manifest(
        version=version,
        bundle_digest=sha256_bytes(payload),
        bundle_bytes=len(payload),
        files=files,
        deploy_manifest=deploy_manifest,
    )
    manifest_path = write_release_manifest(manifest, args.out_dir)
    descriptor_path = write_deploy_descriptor(
        build_deploy_descriptor(
            version=version,
            bundle_digest=sha256_bytes(payload),
            bundle_bytes=len(payload),
        ),
        args.out_dir,
    )
    print(f"packed {tarball_path} ({len(payload)} bytes, {sha256_bytes(payload)})")
    print(f"wrote {manifest_path}")
    print(f"wrote {descriptor_path}")
    return 0


__all__ = [
    "BUNDLE_MANIFEST_NAME",
    "DEPLOY_BUNDLE_SCHEMA",
    "DEPLOY_DESCRIPTOR_NAME",
    "DEPLOY_MANIFEST_NAME",
    "RELEASE_MANIFEST_SCHEMA",
    "RELEASE_TAG_PREFIX",
    "RELEASE_TARBALL_NAME",
    "STUDIO_ID",
    "build_deploy_descriptor",
    "build_release_manifest",
    "bundle_file_table",
    "load_deploy_manifest",
    "main",
    "pack_release_tarball",
    "sha256_bytes",
    "stage_capability_manifest",
    "studio_version",
    "write_deploy_descriptor",
    "write_release_manifest",
]


if __name__ == "__main__":
    sys.exit(main())
