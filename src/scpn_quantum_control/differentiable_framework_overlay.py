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

DEFAULT_OVERLAY_BASENAME = "scpn-qc-framework-site-py312"


@dataclass(frozen=True)
class FrameworkOverlayManifest:
    """CPU-only optional-framework overlay installation contract."""

    overlay_path: Path
    python_version: str
    cpu_wheels: tuple[str, ...]
    install_command: tuple[str, ...]
    pythonpath: str
    artifact_id: str = "diff-qnode-framework-overlay-profile-v1"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready manifest payload."""

        return {
            "schema": "scpn_qc_framework_overlay_manifest_v1",
            "artifact_id": self.artifact_id,
            "overlay_path": str(self.overlay_path),
            "python_version": self.python_version,
            "cpu_wheels": list(self.cpu_wheels),
            "install_command": list(self.install_command),
            "pythonpath": self.pythonpath,
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


def default_framework_overlay_path() -> Path:
    """Return the documented local overlay target path."""

    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / DEFAULT_OVERLAY_BASENAME


def build_framework_overlay_manifest(
    *,
    overlay_path: Path | None = None,
    python_version: str | None = None,
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
    )


def verify_framework_overlay_manifest(manifest_path: Path) -> FrameworkOverlayVerification:
    """Verify that the overlay directory and required package roots exist."""

    manifest = FrameworkOverlayManifest.from_json(manifest_path)
    if not manifest.overlay_path.exists():
        return FrameworkOverlayVerification(
            ready=False,
            status="missing_overlay_path",
            pythonpath=manifest.pythonpath,
            message=f"Set PYTHONPATH={manifest.pythonpath} after creating the overlay.",
            missing_packages=("jax", "torch", "tensorflow", "pennylane"),
        )
    missing = tuple(
        package
        for package in ("jax", "torch", "tensorflow", "pennylane")
        if not (manifest.overlay_path / package).exists()
    )
    if missing:
        return FrameworkOverlayVerification(
            ready=False,
            status="missing_packages",
            pythonpath=manifest.pythonpath,
            message=f"Set PYTHONPATH={manifest.pythonpath}; missing packages: {', '.join(missing)}.",
            missing_packages=missing,
        )
    return FrameworkOverlayVerification(
        ready=True,
        status="ready",
        pythonpath=manifest.pythonpath,
        message=f"PYTHONPATH={manifest.pythonpath}",
        missing_packages=(),
    )


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
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = args.manifest_path or args.overlay_path / "framework_overlay_manifest.json"
    if args.verify:
        result = verify_framework_overlay_manifest(manifest_path)
        print(result.message)
        return 0 if result.ready else 2

    manifest = build_framework_overlay_manifest(overlay_path=args.overlay_path)
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
    "build_framework_overlay_manifest",
    "default_framework_overlay_path",
    "framework_overlay_pythonpath",
    "main",
    "verify_framework_overlay_manifest",
]
