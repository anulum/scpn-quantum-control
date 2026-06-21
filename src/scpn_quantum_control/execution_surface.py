# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — execution-surface scanner
"""Static execution-surface scanner for notebooks and publication scripts."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class ExecutionSurfaceFinding:
    """Machine-readable static execution-surface finding."""

    path: str
    rule: str
    line: int
    evidence: str

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-compatible finding record."""
        return {
            "path": self.path,
            "rule": self.rule,
            "line": self.line,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ExecutionSurfaceManifestEntry:
    """Declared execution policy for one notebook or script."""

    path: str
    classification: str
    allowed_rules: frozenset[str]
    ci_blocking: bool = True


@dataclass(frozen=True)
class ExecutionSurfaceViolation:
    """Manifest violation that should block trusted execution."""

    path: str
    reason: str
    rule: str | None = None
    line: int | None = None
    evidence: str | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        """Return a JSON-compatible violation record."""
        return {
            "path": self.path,
            "reason": self.reason,
            "rule": self.rule,
            "line": self.line,
            "evidence": self.evidence,
        }


EXECUTION_SURFACE_CLASSIFICATIONS: frozenset[str] = frozenset(
    {
        "trusted_static",
        "trusted_offline_executable",
        "external_publication",
        "hardware_gated",
        "untrusted_user",
    }
)

DEFAULT_MANIFEST_PATH = Path("docs/execution_surface_manifest.toml")
HIGH_RISK_RULES: frozenset[str] = frozenset(
    {"credential_read", "hardware_submission", "network_access", "external_publication"}
)

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("shell_magic", re.compile(r"(^|\n)\s*![^\n]+|(^|\n)\s*%pip\b|(^|\n)\s*%conda\b")),
    (
        "subprocess_execution",
        re.compile(r"\bsubprocess\.(run|call|check_call|check_output|Popen)\b"),
    ),
    ("network_access", re.compile(r"\b(requests\.|urllib\.request|httpx\.|aiohttp\.)")),
    ("credential_read", re.compile(r"\b(os\.environ|getenv)\b.*(TOKEN|CRN|API|KEY|SECRET)")),
    (
        "hardware_submission",
        re.compile(
            r"\b(SamplerV2|EstimatorV2|QiskitRuntimeService)\b|\b(sampler|estimator)\.run\(",
            re.IGNORECASE,
        ),
    ),
    ("external_publication", re.compile(r"\bkaggle\s+kernels\s+push\b")),
)


def load_execution_surface_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[ExecutionSurfaceManifestEntry, ...]:
    """Load the repository execution-surface manifest."""
    data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    entries: list[ExecutionSurfaceManifestEntry] = []
    for raw in data.get("surface", []):
        classification = str(raw["classification"])
        if classification not in EXECUTION_SURFACE_CLASSIFICATIONS:
            raise ValueError(f"unknown execution-surface classification: {classification}")
        entries.append(
            ExecutionSurfaceManifestEntry(
                path=str(raw["path"]),
                classification=classification,
                allowed_rules=frozenset(str(rule) for rule in raw.get("allowed_rules", [])),
                ci_blocking=bool(raw.get("ci_blocking", True)),
            )
        )
    return tuple(entries)


def evaluate_execution_surface_manifest(
    repo_root: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[ExecutionSurfaceViolation, ...]:
    """Return manifest violations for all CI-blocking surface entries."""
    root = repo_root.resolve()
    entries = load_execution_surface_manifest(manifest_path)
    violations: list[ExecutionSurfaceViolation] = []
    for entry in entries:
        if not entry.ci_blocking:
            continue
        surface_path = _resolve_repo_path(root, entry.path)
        if surface_path is None:
            violations.append(
                ExecutionSurfaceViolation(
                    path=entry.path,
                    reason="path escapes repository",
                )
            )
            continue
        if not surface_path.exists():
            violations.append(
                ExecutionSurfaceViolation(path=entry.path, reason="manifest path is missing")
            )
            continue
        for finding in scan_execution_surface_path(surface_path):
            if finding.rule not in entry.allowed_rules:
                violations.append(
                    ExecutionSurfaceViolation(
                        path=entry.path,
                        reason="unapproved execution-surface finding",
                        rule=finding.rule,
                        line=finding.line,
                        evidence=finding.evidence,
                    )
                )
    return tuple(violations)


def iter_execution_surface_paths(repo_root: Path) -> tuple[Path, ...]:
    """Return repository notebooks and publication scripts that need review."""
    root = repo_root.resolve()
    patterns = (
        "notebooks/*.ipynb",
        "notebooks/*.py",
        "notebooks/colab/*.ipynb",
        "notebooks/kaggle_push/*.py",
        "notebooks/kaggle_push/*.sh",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.glob(pattern))
    return tuple(sorted(path for path in paths if path.is_file()))


def find_unmanifested_high_risk_surfaces(
    repo_root: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[ExecutionSurfaceFinding, ...]:
    """Return high-risk findings from paths absent from the manifest."""
    manifest_paths = {entry.path for entry in load_execution_surface_manifest(manifest_path)}
    findings: list[ExecutionSurfaceFinding] = []
    root = repo_root.resolve()
    for path in iter_execution_surface_paths(root):
        relative = path.relative_to(root).as_posix()
        if relative in manifest_paths:
            continue
        findings.extend(
            finding
            for finding in scan_execution_surface_path(path)
            if finding.rule in HIGH_RISK_RULES
        )
    return tuple(findings)


def scan_execution_surface_path(path: Path) -> tuple[ExecutionSurfaceFinding, ...]:
    """Scan one notebook/script path without executing it."""
    repo_path = Path(path)
    text = _read_surface_text(repo_path)
    findings: list[ExecutionSurfaceFinding] = []
    for rule, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            evidence = _line_at(text, line)
            findings.append(
                ExecutionSurfaceFinding(
                    path=str(repo_path),
                    rule=rule,
                    line=line,
                    evidence=evidence[:240],
                )
            )
    return tuple(findings)


def _resolve_repo_path(repo_root: Path, relative_path: str) -> Path | None:
    path = Path(relative_path)
    if path.is_absolute():
        return None
    resolved = (repo_root / path).resolve()
    if not resolved.is_relative_to(repo_root):
        return None
    return resolved


def _read_surface_text(path: Path) -> str:
    if path.suffix == ".ipynb":
        with path.open(encoding="utf-8") as fh:
            notebook = json.load(fh)
        cells = notebook.get("cells", [])
        source_blocks: list[str] = []
        for cell in cells:
            source = cell.get("source", "")
            if isinstance(source, list):
                source_blocks.append("".join(str(part) for part in source))
            else:
                source_blocks.append(str(source))
        return "\n".join(source_blocks)
    return path.read_text(encoding="utf-8", errors="replace")


def _line_at(text: str, line: int) -> str:
    lines = text.splitlines()
    if not 1 <= line <= len(lines):
        return ""
    return lines[line - 1].strip()
