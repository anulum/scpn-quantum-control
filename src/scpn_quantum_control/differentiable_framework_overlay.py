# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable framework overlay profile.
"""Reproducible CPU-only overlay profile for optional AD frameworks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CPU_FRAMEWORK_WHEELS: tuple[str, ...] = (
    "jax[cpu]",
    "torch",
    "tensorflow-cpu",
    "pennylane",
)
CPU_FRAMEWORK_PACKAGE_ROOTS: tuple[str, ...] = ("jax", "torch", "tensorflow", "pennylane")
PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
FRAMEWORK_OVERLAY_SCHEMA = "scpn_qc_framework_overlay_manifest_v2"

DEFAULT_OVERLAY_BASENAME = "scpn-qc-framework-site-py312"


@dataclass(frozen=True)
class FrameworkOverlayManifest:
    """CPU-only optional-framework overlay installation contract."""

    overlay_path: Path
    python_version: str
    cpu_wheels: tuple[str, ...]
    install_command: tuple[str, ...]
    pythonpath: str
    platform: str
    wheel_sources: dict[str, str]
    package_versions: dict[str, str]
    verification_status: str
    artifact_id: str = "diff-qnode-framework-overlay-profile-v1"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready manifest payload."""

        return {
            "schema": FRAMEWORK_OVERLAY_SCHEMA,
            "artifact_id": self.artifact_id,
            "overlay_path": str(self.overlay_path),
            "python_version": self.python_version,
            "platform": self.platform,
            "cpu_wheels": list(self.cpu_wheels),
            "install_command": list(self.install_command),
            "pythonpath": self.pythonpath,
            "wheel_sources": self.wheel_sources,
            "package_versions": self.package_versions,
            "verification_status": self.verification_status,
            "claim_boundary": (
                "CPU-only optional framework parity overlay; no CUDA, provider, QPU, "
                "or performance promotion claim."
            ),
        }

    @classmethod
    def from_json(cls, path: Path) -> FrameworkOverlayManifest:
        """Load a manifest from disk."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            overlay_path=Path(str(payload["overlay_path"])),
            python_version=str(payload["python_version"]),
            cpu_wheels=tuple(str(item) for item in payload["cpu_wheels"]),
            install_command=tuple(str(item) for item in payload["install_command"]),
            pythonpath=str(payload["pythonpath"]),
            platform=str(payload.get("platform", "")),
            wheel_sources={
                str(key): str(value) for key, value in payload.get("wheel_sources", {}).items()
            },
            package_versions={
                str(key): str(value) for key, value in payload.get("package_versions", {}).items()
            },
            verification_status=str(payload.get("verification_status", "not_verified")),
            artifact_id=str(payload.get("artifact_id", "diff-qnode-framework-overlay-profile-v1")),
        )


@dataclass(frozen=True)
class FrameworkOverlayVerification:
    """Verification result for a framework overlay manifest."""

    ready: bool
    status: str
    pythonpath: str
    message: str
    missing_packages: tuple[str, ...]
    package_versions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready verification metadata."""

        return {
            "ready": self.ready,
            "status": self.status,
            "pythonpath": self.pythonpath,
            "message": self.message,
            "missing_packages": list(self.missing_packages),
            "package_versions": self.package_versions,
        }


def default_framework_overlay_path() -> Path:
    """Return the documented local overlay target path."""

    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / DEFAULT_OVERLAY_BASENAME


def build_framework_overlay_manifest(
    *,
    overlay_path: Path | None = None,
    python_version: str | None = None,
    package_versions: dict[str, str] | None = None,
    verification_status: str = "not_verified",
) -> FrameworkOverlayManifest:
    """Build a CPU-wheel overlay manifest without performing installation."""

    target = overlay_path or default_framework_overlay_path()
    version = python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
    command = (
        "python",
        "-m",
        "pip",
        "install",
        "--target",
        str(target),
        "--upgrade",
        *CPU_FRAMEWORK_WHEELS,
    )
    return FrameworkOverlayManifest(
        overlay_path=target,
        python_version=version,
        cpu_wheels=CPU_FRAMEWORK_WHEELS,
        install_command=command,
        pythonpath=str(target),
        platform=platform.platform(),
        wheel_sources={
            "jax[cpu]": "PyPI CPU extra",
            "torch": PYTORCH_CPU_INDEX_URL,
            "tensorflow-cpu": "PyPI CPU wheel",
            "pennylane": "PyPI",
        },
        package_versions=package_versions or {},
        verification_status=verification_status,
    )


def install_framework_overlay(overlay_path: Path) -> FrameworkOverlayManifest:
    """Install the CPU-only optional-framework overlay and return its manifest."""

    overlay_path.mkdir(parents=True, exist_ok=True)
    base_command = (
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(overlay_path),
        "--upgrade",
        "jax[cpu]",
        "tensorflow-cpu",
        "pennylane",
    )
    torch_command = (
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(overlay_path),
        "--upgrade",
        "--index-url",
        PYTORCH_CPU_INDEX_URL,
        "torch",
    )
    subprocess.run(base_command, check=True)
    subprocess.run(torch_command, check=True)
    verification = verify_framework_overlay_path(overlay_path)
    return build_framework_overlay_manifest(
        overlay_path=overlay_path,
        package_versions=verification.package_versions,
        verification_status=verification.status,
    )


def verify_framework_overlay_manifest(manifest_path: Path) -> FrameworkOverlayVerification:
    """Verify that the overlay directory and required package roots exist."""

    manifest = FrameworkOverlayManifest.from_json(manifest_path)
    return verify_framework_overlay_path(manifest.overlay_path, pythonpath=manifest.pythonpath)


def verify_framework_overlay_path(
    overlay_path: Path, *, pythonpath: str | None = None
) -> FrameworkOverlayVerification:
    """Verify that an overlay directory contains required package roots."""

    resolved_pythonpath = pythonpath or str(overlay_path)
    if not overlay_path.exists():
        return FrameworkOverlayVerification(
            ready=False,
            status="missing_overlay_path",
            pythonpath=resolved_pythonpath,
            message=f"Set PYTHONPATH={resolved_pythonpath} after creating the overlay.",
            missing_packages=CPU_FRAMEWORK_PACKAGE_ROOTS,
            package_versions={},
        )
    missing = tuple(
        package for package in CPU_FRAMEWORK_PACKAGE_ROOTS if not (overlay_path / package).exists()
    )
    if missing:
        return FrameworkOverlayVerification(
            ready=False,
            status="missing_packages",
            pythonpath=resolved_pythonpath,
            message=f"Set PYTHONPATH={resolved_pythonpath}; missing packages: {', '.join(missing)}.",
            missing_packages=missing,
            package_versions=_discover_overlay_versions(overlay_path),
        )
    versions = _discover_overlay_versions(overlay_path)
    return FrameworkOverlayVerification(
        ready=True,
        status="ready",
        pythonpath=resolved_pythonpath,
        message=f"PYTHONPATH={resolved_pythonpath}",
        missing_packages=(),
        package_versions=versions,
    )


def _discover_overlay_versions(overlay_path: Path) -> dict[str, str]:
    """Return installed package versions discoverable from overlay dist-info."""

    versions: dict[str, str] = {}
    for package in CPU_FRAMEWORK_PACKAGE_ROOTS:
        for dist_info in overlay_path.glob(f"{package.replace('_', '-')}*.dist-info"):
            metadata = dist_info / "METADATA"
            if metadata.exists():
                versions[package] = _metadata_version(metadata)
                break
        if package not in versions:
            for dist_info in overlay_path.glob(f"{package}*.dist-info"):
                metadata = dist_info / "METADATA"
                if metadata.exists():
                    versions[package] = _metadata_version(metadata)
                    break
    return versions


def _metadata_version(metadata_path: Path) -> str:
    for line in metadata_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Version:"):
            return line.split(":", maxsplit=1)[1].strip()
    return "unknown"


def framework_overlay_pythonpath(manifest_path: Path) -> str:
    """Return the exact PYTHONPATH from an existing overlay manifest."""

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    return FrameworkOverlayManifest.from_json(manifest_path).pythonpath


def main(argv: list[str] | None = None) -> int:
    """Emit or verify the CPU-framework overlay manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-path", type=Path, default=default_framework_overlay_path())
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = args.manifest_path or args.overlay_path / "framework_overlay_manifest.json"
    if args.verify:
        result = verify_framework_overlay_manifest(manifest_path)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.ready else 2

    manifest = (
        install_framework_overlay(args.overlay_path)
        if args.install
        else build_framework_overlay_manifest(overlay_path=args.overlay_path)
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(" ".join(manifest.install_command))
    print(f"PYTHONPATH={manifest.pythonpath}")
    print(manifest_path)
    return 0


__all__ = [
    "CPU_FRAMEWORK_WHEELS",
    "DEFAULT_OVERLAY_BASENAME",
    "FrameworkOverlayManifest",
    "FrameworkOverlayVerification",
    "PYTORCH_CPU_INDEX_URL",
    "build_framework_overlay_manifest",
    "default_framework_overlay_path",
    "framework_overlay_pythonpath",
    "install_framework_overlay",
    "main",
    "verify_framework_overlay_path",
    "verify_framework_overlay_manifest",
]
