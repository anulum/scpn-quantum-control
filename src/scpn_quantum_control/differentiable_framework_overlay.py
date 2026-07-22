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
import subprocess  # nosec B404
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
FRAMEWORK_OVERLAY_ARTIFACT_ID = "diff-qnode-framework-overlay-profile-v1"
FRAMEWORK_OVERLAY_CLAIM_BOUNDARY = (
    "CPU-only optional framework parity overlay; no CUDA, provider, QPU, "
    "or performance promotion claim."
)

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
    artifact_id: str = FRAMEWORK_OVERLAY_ARTIFACT_ID

    def __post_init__(self) -> None:
        """Validate the CPU-only manifest identity without coercion."""
        if not isinstance(self.overlay_path, Path):
            raise ValueError("overlay_path must be a Path")
        _validated_overlay_install_path(self.overlay_path)
        for field_name in (
            "python_version",
            "pythonpath",
            "platform",
            "verification_status",
            "artifact_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in ("cpu_wheels", "install_command"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, tuple)
                or not value
                or any(not isinstance(item, str) or not item.strip() for item in value)
            ):
                raise ValueError(f"{field_name} must contain non-empty strings")
        if self.cpu_wheels != CPU_FRAMEWORK_WHEELS:
            raise ValueError("cpu_wheels must match the canonical CPU framework profile")
        for field_name in ("wheel_sources", "package_versions"):
            value = getattr(self, field_name)
            if not isinstance(value, dict) or any(
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(item, str)
                or not item.strip()
                for key, item in value.items()
            ):
                raise ValueError(f"{field_name} must map non-empty strings to non-empty strings")
        if set(self.wheel_sources) != set(CPU_FRAMEWORK_WHEELS):
            raise ValueError("wheel_sources must cover the canonical CPU framework profile")
        if self.pythonpath != str(self.overlay_path):
            raise ValueError("pythonpath must match overlay_path exactly")
        if self.artifact_id != FRAMEWORK_OVERLAY_ARTIFACT_ID:
            raise ValueError("artifact_id must match the canonical framework overlay identity")

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
            "claim_boundary": FRAMEWORK_OVERLAY_CLAIM_BOUNDARY,
        }

    @classmethod
    def from_json(cls, path: Path) -> FrameworkOverlayManifest:
        """Load a manifest from disk."""
        payload: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("framework overlay manifest must be a JSON object")
        if payload.get("schema") != FRAMEWORK_OVERLAY_SCHEMA:
            raise ValueError("framework overlay manifest schema is unknown")
        if payload.get("claim_boundary") != FRAMEWORK_OVERLAY_CLAIM_BOUNDARY:
            raise ValueError("framework overlay manifest claim boundary is not canonical")
        return cls(
            overlay_path=Path(_manifest_string(payload, "overlay_path")),
            python_version=_manifest_string(payload, "python_version"),
            cpu_wheels=_manifest_strings(payload, "cpu_wheels"),
            install_command=_manifest_strings(payload, "install_command"),
            pythonpath=_manifest_string(payload, "pythonpath"),
            platform=_manifest_string(payload, "platform"),
            wheel_sources=_manifest_string_map(payload, "wheel_sources"),
            package_versions=_manifest_string_map(payload, "package_versions", allow_empty=True),
            verification_status=_manifest_string(payload, "verification_status"),
            artifact_id=_manifest_string(payload, "artifact_id"),
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
    target = _validated_overlay_install_path(overlay_path or default_framework_overlay_path())
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


def _validated_overlay_install_path(overlay_path: Path) -> Path:
    raw = str(overlay_path)
    if "\x00" in raw:
        raise ValueError("overlay_path cannot contain a null byte")
    target = overlay_path.expanduser()
    if not target.is_absolute():
        raise ValueError("overlay_path must be an absolute directory path")
    if target == Path(target.anchor):
        raise ValueError("overlay_path cannot be the filesystem root")
    if target.exists() and not target.is_dir():
        raise ValueError("overlay_path must be a directory when it already exists")
    return target


def install_framework_overlay(overlay_path: Path) -> FrameworkOverlayManifest:
    """Install the CPU-only optional-framework overlay and return its manifest."""
    target = _validated_overlay_install_path(overlay_path)
    target.mkdir(parents=True, exist_ok=True)
    base_command = (
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(target),
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
        str(target),
        "--upgrade",
        "--index-url",
        PYTORCH_CPU_INDEX_URL,
        "torch",
    )
    subprocess.run(base_command, check=True)  # nosec B603
    subprocess.run(torch_command, check=True)  # nosec B603
    verification = verify_framework_overlay_path(target)
    return build_framework_overlay_manifest(
        overlay_path=target,
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


def _manifest_string(payload: dict[object, object], field_name: str) -> str:
    """Read one required exact non-empty manifest string."""
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"framework overlay {field_name} must be a non-empty string")
    return value


def _manifest_strings(payload: dict[object, object], field_name: str) -> tuple[str, ...]:
    """Read one required JSON array of non-empty strings."""
    value = payload.get(field_name)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"framework overlay {field_name} must contain non-empty strings")
    return tuple(value)


def _manifest_string_map(
    payload: dict[object, object],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> dict[str, str]:
    """Read one required JSON object mapping non-empty strings."""
    value = payload.get(field_name)
    if (
        not isinstance(value, dict)
        or (not allow_empty and not value)
        or any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(item, str)
            or not item.strip()
            for key, item in value.items()
        )
    ):
        raise ValueError(
            f"framework overlay {field_name} must map non-empty strings to non-empty strings"
        )
    return value


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
    "FRAMEWORK_OVERLAY_ARTIFACT_ID",
    "FRAMEWORK_OVERLAY_CLAIM_BOUNDARY",
    "FRAMEWORK_OVERLAY_SCHEMA",
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
