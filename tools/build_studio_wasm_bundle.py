#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — studio WASM bundle build + deploy manifest.
"""Build the studio WASM kernel and record the deploy manifest.

Builds ``scpn_quantum_engine/studio_wasm_kernel`` to
``wasm32-unknown-unknown`` with the locked dependency set, ships the ``.wasm``
inside the built portal bundle (``studio-web/dist/wasm/``), and writes
``deploy-manifest.json`` recording the SHA-256 digest and byte size of every
shipped artefact plus the exact toolchain that produced the kernel.

No binaries are committed: this tool runs in CI (and locally for
verification) after ``vite build``, and the manifest is the content-addressed
record the keeper's federation aggregation can verify against. The manifest
is deterministic — no timestamps ride in it; provenance beyond the digest
belongs to the CI run metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import tomllib

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
KERNEL_CRATE_DIR: Final[Path] = REPO_ROOT / "scpn_quantum_engine" / "studio_wasm_kernel"
KERNEL_WASM_NAME: Final[str] = "scpn_quantum_studio_wasm_kernel.wasm"
WASM_TARGET: Final[str] = "wasm32-unknown-unknown"
DEFAULT_DIST_DIR: Final[Path] = REPO_ROOT / "studio-web" / "dist"
DEPLOY_MANIFEST_NAME: Final[str] = "deploy-manifest.json"
DEPLOY_MANIFEST_SCHEMA: Final[str] = "scpn_qc_studio_deploy_manifest_v1"
_TRACKED_BUNDLE_FILES: Final[tuple[str, ...]] = ("index.html", "remoteEntry.js")


def sha256_file(path: Path) -> str:
    """Return the ``sha256:<hex>`` digest of a file's bytes.

    Parameters
    ----------
    path
        File to content-address.

    Returns
    -------
    str
        The prefixed SHA-256 hex digest.
    """
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def kernel_crate_version(crate_dir: Path = KERNEL_CRATE_DIR) -> str:
    """Return the kernel crate version from its committed manifest.

    Parameters
    ----------
    crate_dir
        The kernel crate directory.

    Returns
    -------
    str
        The ``[package].version`` value.

    Raises
    ------
    ValueError
        If the crate manifest carries no version.
    """
    manifest = tomllib.loads((crate_dir / "Cargo.toml").read_text(encoding="utf-8"))
    version = manifest.get("package", {}).get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("kernel crate manifest carries no package version")
    return version


def build_wasm_kernel(crate_dir: Path = KERNEL_CRATE_DIR) -> Path:
    """Build the kernel crate for the WASM target and return the artefact path.

    Parameters
    ----------
    crate_dir
        The kernel crate directory.

    Returns
    -------
    Path
        The built ``.wasm`` artefact.

    Raises
    ------
    ValueError
        If the build succeeds but the expected artefact is absent.
    """
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "--target", WASM_TARGET],
        cwd=crate_dir,
        check=True,
    )
    artefact = crate_dir / "target" / WASM_TARGET / "release" / KERNEL_WASM_NAME
    if not artefact.exists():
        raise ValueError(f"wasm build produced no artefact at {artefact.as_posix()}")
    return artefact


def rustc_version() -> str:
    """Return the exact ``rustc --version`` string of the building toolchain."""
    result = subprocess.run(
        ["rustc", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def ship_wasm_into_bundle(artefact: Path, dist_dir: Path) -> Path:
    """Copy the built kernel into the portal bundle's ``wasm/`` directory.

    Parameters
    ----------
    artefact
        The built ``.wasm`` file.
    dist_dir
        The built portal bundle (``vite build`` output).

    Returns
    -------
    Path
        The shipped artefact path inside the bundle.

    Raises
    ------
    ValueError
        If the portal bundle does not exist — the WASM tier ships inside the
        portal bundle, never on its own.
    """
    if not dist_dir.is_dir():
        raise ValueError(f"portal bundle does not exist: {dist_dir.as_posix()} (run vite build)")
    wasm_dir = dist_dir / "wasm"
    wasm_dir.mkdir(exist_ok=True)
    shipped = wasm_dir / KERNEL_WASM_NAME
    shutil.copyfile(artefact, shipped)
    return shipped


def build_deploy_manifest(
    dist_dir: Path,
    *,
    toolchain: str,
    crate_version: str,
) -> dict[str, object]:
    """Build the deploy-manifest payload over the shipped bundle.

    Parameters
    ----------
    dist_dir
        The portal bundle directory containing the shipped artefacts.
    toolchain
        The exact ``rustc --version`` string that built the kernel.
    crate_version
        The kernel crate version.

    Returns
    -------
    dict[str, object]
        JSON-ready manifest: one digest row per shipped artefact.

    Raises
    ------
    ValueError
        If a tracked bundle artefact is missing — a partial bundle is never
        manifest-signed.
    """
    artefacts: list[dict[str, object]] = []
    tracked = [*_TRACKED_BUNDLE_FILES, f"wasm/{KERNEL_WASM_NAME}"]
    for relative in tracked:
        path = dist_dir / relative
        if not path.is_file():
            raise ValueError(f"bundle artefact missing: {relative}")
        artefacts.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema": DEPLOY_MANIFEST_SCHEMA,
        "studio": "scpn-quantum-control",
        "kernel_crate": "scpn_quantum_engine/studio_wasm_kernel",
        "kernel_crate_version": crate_version,
        "wasm_target": WASM_TARGET,
        "toolchain": toolchain,
        "artifacts": artefacts,
    }


def write_deploy_manifest(manifest: dict[str, object], dist_dir: Path) -> Path:
    """Write the deploy manifest into the bundle and return its path."""
    path = dist_dir / DEPLOY_MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: build, ship, and manifest the WASM tier.

    Parameters
    ----------
    argv
        Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        ``0`` on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=DEFAULT_DIST_DIR,
        help="built portal bundle directory (vite build output)",
    )
    args = parser.parse_args(argv)
    artefact = build_wasm_kernel()
    shipped = ship_wasm_into_bundle(artefact, args.dist_dir)
    manifest = build_deploy_manifest(
        args.dist_dir,
        toolchain=rustc_version(),
        crate_version=kernel_crate_version(),
    )
    manifest_path = write_deploy_manifest(manifest, args.dist_dir)
    print(f"shipped {shipped}")
    print(f"wrote {manifest_path}")
    return 0


__all__ = [
    "DEPLOY_MANIFEST_NAME",
    "DEPLOY_MANIFEST_SCHEMA",
    "KERNEL_WASM_NAME",
    "WASM_TARGET",
    "build_deploy_manifest",
    "build_wasm_kernel",
    "kernel_crate_version",
    "main",
    "rustc_version",
    "sha256_file",
    "ship_wasm_into_bundle",
    "write_deploy_manifest",
]


if __name__ == "__main__":
    sys.exit(main())
