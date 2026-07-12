# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — license readiness audit helper
"""Audit licence, commercial-route, and core-split readiness consistency."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python < 3.11 runtime fallback
    import tomli as tomllib


EXPECTED_PROJECT_LICENSE = "AGPL-3.0-or-later"
EXPECTED_LICENSE_CLASSIFIER = (
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)"
)
PERMISSIVE_LICENSE_MARKERS = (
    "Apache-2.0",
    "MIT",
    "BSD",
    "MPL",
    "ISC",
    "License :: OSI Approved :: Apache Software License",
    "License :: OSI Approved :: MIT License",
    "License :: OSI Approved :: BSD License",
)
REQUIRED_TEXT_FILES: dict[str, tuple[str, ...]] = {
    "LICENSE": (
        "AGPL-3.0-or-later",
        "Commercial license available",
    ),
    "README.md": (
        "AGPL-3.0-or-later",
        "commercial licence",
        "not a separate permissive package today",
        "all in-repository code remains under the AGPL/commercial terms",
    ),
    "docs/core_package_boundary.md": (
        "No file is dual-licensed or permissively relicensed by this document.",
        "not relicensed",
        "AGPL-3.0-or-later",
    ),
    "docs/licensing_faq.md": (
        "AGPL-3.0-or-later",
        "commercial licence",
        "not available as a permissive package today",
    ),
}
HEADER_COMMENT_PREFIXES = {
    ".c": "//",
    ".cc": "//",
    ".cpp": "//",
    ".go": "//",
    ".h": "//",
    ".hpp": "//",
    ".jl": "#",
    ".js": "//",
    ".jsx": "//",
    ".py": "#",
    ".pyi": "#",
    ".rs": "//",
    ".sv": "//",
    ".toml": "#",
    ".ts": "//",
    ".tsx": "//",
    ".v": "//",
    ".yaml": "#",
    ".yml": "#",
}
CANONICAL_HEADER_LINES = (
    "SPDX-License-Identifier: AGPL-3.0-or-later",
    "Commercial license available",
    "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
    "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
    "ORCID: 0009-0009-3560-0851",
    "Contact: www.anulum.li | protoscience@anulum.li",
)
EXCLUDED_HEADER_PATHS = {"studio-web/pnpm-lock.yaml"}
IGNORED_HEADER_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "BACKUP",
    "ARCHIVE",
    "build",
    "dist",
    "node_modules",
    ".venv",
    ".venv-linux",
    "site",
    "target",
}


@dataclass(frozen=True)
class LicenseReadinessCheck:
    """One licence-readiness check."""

    name: str
    valid: bool
    details: dict[str, Any]
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready check metadata."""
        return {
            "name": self.name,
            "valid": self.valid,
            "details": self.details,
            "blockers": list(self.blockers),
        }


def check_project_metadata(project_root: Path) -> LicenseReadinessCheck:
    """Check pyproject licence metadata and classifiers."""
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.exists():
        return LicenseReadinessCheck(
            name="project_metadata",
            valid=False,
            details={"path": "pyproject.toml"},
            blockers=("pyproject.toml missing",),
        )
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    if not isinstance(project, dict):
        return LicenseReadinessCheck(
            name="project_metadata",
            valid=False,
            details={"path": "pyproject.toml"},
            blockers=("pyproject [project] table missing or invalid",),
        )
    license_value = project.get("license")
    classifiers = project.get("classifiers", [])
    blockers: list[str] = []
    if license_value != EXPECTED_PROJECT_LICENSE:
        blockers.append(
            "pyproject project.license must remain AGPL-3.0-or-later until an approved "
            f"split changes it; found {license_value!r}"
        )
    if not isinstance(classifiers, list) or EXPECTED_LICENSE_CLASSIFIER not in classifiers:
        blockers.append("pyproject classifiers must include the AGPLv3+ classifier")
    permissive = tuple(
        marker
        for marker in PERMISSIVE_LICENSE_MARKERS
        if any(marker in str(classifier) for classifier in classifiers)
        or marker == str(license_value)
    )
    if permissive:
        blockers.append(
            "pyproject contains permissive classifier or licence marker before split approval: "
            + ", ".join(permissive)
        )
    return LicenseReadinessCheck(
        name="project_metadata",
        valid=not blockers,
        details={
            "license": license_value,
            "has_agpl_classifier": isinstance(classifiers, list)
            and EXPECTED_LICENSE_CLASSIFIER in classifiers,
            "permissive_markers": list(permissive),
        },
        blockers=tuple(blockers),
    )


def check_required_text(project_root: Path) -> LicenseReadinessCheck:
    """Check public licence and boundary documents for required statements."""
    blockers: list[str] = []
    details: dict[str, Any] = {}
    for rel_path, required_markers in REQUIRED_TEXT_FILES.items():
        path = project_root / rel_path
        file_details = {"exists": path.exists(), "missing_markers": []}
        if not path.exists():
            blockers.append(f"{rel_path}: required licence-readiness document missing")
            details[rel_path] = file_details
            continue
        text = _normalise_text(path.read_text(encoding="utf-8"))
        missing = tuple(
            marker for marker in required_markers if _normalise_text(marker) not in text
        )
        file_details["missing_markers"] = list(missing)
        if missing:
            blockers.append(f"{rel_path}: missing required wording: {', '.join(missing)}")
        details[rel_path] = file_details
    return LicenseReadinessCheck(
        name="required_text_boundaries",
        valid=not blockers,
        details=details,
        blockers=tuple(blockers),
    )


def check_spdx_headers(project_root: Path) -> LicenseReadinessCheck:
    """Check tracked code and configuration files for the canonical header."""
    blockers: list[str] = []
    scanned: list[str] = []
    for path in _iter_header_scan_files(project_root):
        rel_path = path.relative_to(project_root).as_posix()
        scanned.append(rel_path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        prefix = HEADER_COMMENT_PREFIXES[path.suffix]
        blockers.extend(_header_blockers(rel_path, lines, prefix))
    return LicenseReadinessCheck(
        name="source_spdx_headers",
        valid=not blockers,
        details={"scanned_count": len(scanned), "scanned": scanned[:50]},
        blockers=tuple(blockers),
    )


def audit_license_readiness(project_root: Path) -> dict[str, Any]:
    """Run the licence and commercial-readiness gate."""
    root = project_root.resolve()
    checks = (
        check_project_metadata(root),
        check_required_text(root),
        check_spdx_headers(root),
    )
    blockers = tuple(blocker for check in checks for blocker in check.blockers)
    return {
        "ready": not blockers,
        "project_root": root.as_posix(),
        "checks": [check.to_dict() for check in checks],
        "blockers": list(blockers),
    }


def format_license_readiness(payload: dict[str, Any]) -> str:
    """Return a deterministic text summary for CLI output."""
    lines = [
        "License readiness audit:",
        f"ready: {payload['ready']}",
        f"project_root: {payload['project_root']}",
    ]
    for check in payload["checks"]:
        lines.append(f"- {check['name']}: {check['valid']}")
        for blocker in check["blockers"]:
            lines.append(f"  blocker: {blocker}")
    return "\n".join(lines)


def _iter_header_scan_files(project_root: Path) -> Iterable[Path]:
    tracked = _git_tracked_files(project_root)
    candidates = tracked if tracked is not None else _walk_files(project_root)
    for path in candidates:
        if not path.is_file() or path.suffix not in HEADER_COMMENT_PREFIXES:
            continue
        relative = path.relative_to(project_root)
        if relative.as_posix() in EXCLUDED_HEADER_PATHS:
            continue
        if set(relative.parts) & IGNORED_HEADER_PARTS:
            continue
        yield path


def _git_tracked_files(project_root: Path) -> tuple[Path, ...] | None:
    """Return tracked paths, or ``None`` when the root is not a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "ls-files", "-z"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    relative_paths = sorted(path for path in result.stdout.split("\0") if path)
    return tuple(project_root / relative for relative in relative_paths)


def _walk_files(project_root: Path) -> tuple[Path, ...]:
    """Return filesystem candidates while pruning generated dependency trees."""
    paths: list[Path] = []
    for current_root, directory_names, file_names in os.walk(project_root):
        directory_names[:] = sorted(
            name for name in directory_names if name not in IGNORED_HEADER_PARTS
        )
        root = Path(current_root)
        paths.extend(root / name for name in sorted(file_names))
    return tuple(paths)


def _header_blockers(relative: str, lines: list[str], prefix: str) -> tuple[str, ...]:
    """Return deterministic canonical-header blockers for one file."""
    start = 1 if lines and lines[0].startswith("#!") else 0
    expected = tuple(f"{prefix} {line}" for line in CANONICAL_HEADER_LINES)
    blockers: list[str] = []
    for index, expected_line in enumerate(expected):
        line_number = start + index + 1
        if len(lines) < line_number:
            blockers.append(f"{relative}: missing canonical header line {index + 1}")
            continue
        observed = lines[line_number - 1]
        if observed != expected_line:
            blockers.append(f"{relative}: non-canonical header line {index + 1}: {observed!r}")
    description_index = start + len(expected)
    if len(lines) <= description_index:
        blockers.append(f"{relative}: missing Project — Description header line")
        return tuple(blockers)
    description_line = lines[description_index]
    marker = f"{prefix} "
    if not description_line.startswith(marker):
        blockers.append(f"{relative}: malformed Project — Description header line")
        return tuple(blockers)
    project, separator, description = description_line[len(marker) :].partition(" — ")
    if separator != " — " or not project.strip() or not description.strip():
        blockers.append(f"{relative}: malformed Project — Description header line")
    return tuple(blockers)


def _normalise_text(value: str) -> str:
    return " ".join(value.casefold().split())


def main(argv: list[str] | None = None) -> int:
    """Run the licence-readiness audit CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args(argv)

    payload = audit_license_readiness(args.root)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_license_readiness(payload))
    return 0 if payload["ready"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
