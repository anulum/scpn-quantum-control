# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — coverage-gap audit helper
"""Inventory source-file coverage gaps from a coverage.py XML report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as ET


@dataclass(frozen=True)
class CoverageFileAudit:
    """Coverage status for one Python source file."""

    path: str
    line_rate: float | None
    covered_lines: int | None
    valid_lines: int | None
    missing_lines: int | None
    status: str

    @property
    def line_percent(self) -> float | None:
        """Return line coverage as a percentage."""
        return None if self.line_rate is None else self.line_rate * 100.0


def _source_files(source_root: Path) -> tuple[Path, ...]:
    """Return deterministic package source files included in coverage scope."""
    return tuple(
        sorted(
            path
            for path in source_root.rglob("*.py")
            if path.is_file()
            and "__pycache__" not in path.parts
            and not any(part.startswith(".") for part in path.parts)
        )
    )


def _normalise_path(path: str, project_root: Path) -> str:
    """Normalise coverage XML filenames to project-relative POSIX paths."""
    raw = Path(path)
    if raw.is_absolute():
        with suppress(ValueError):
            raw = raw.resolve().relative_to(project_root.resolve())
    return raw.as_posix()


def _class_entries(coverage_xml: Path, project_root: Path) -> dict[str, Any]:
    """Return coverage XML class entries by normalised filename."""
    if not coverage_xml.exists():
        return {}
    root = ET.parse(coverage_xml).getroot()
    if root is None:
        return {}
    entries: dict[str, Any] = {}
    for item in root.findall(".//class"):
        filename = item.attrib.get("filename", "")
        if filename:
            entries[_normalise_path(filename, project_root)] = item
    return entries


def _line_counts(item: Any) -> tuple[int, int]:
    """Return covered and valid executable line counts for one class entry."""
    valid = 0
    covered = 0
    for line in item.findall(".//line"):
        valid += 1
        if int(line.attrib.get("hits", "0")) > 0:
            covered += 1
    return covered, valid


def audit_coverage_gaps(
    *,
    project_root: Path,
    source_root: Path,
    coverage_xml: Path,
    min_file_percent: float,
) -> tuple[CoverageFileAudit, ...]:
    """Audit package source files against a coverage.py XML report."""
    project_root = project_root.resolve()
    source_root = source_root.resolve()
    entries = _class_entries(coverage_xml, project_root)
    audits: list[CoverageFileAudit] = []
    for source_path in _source_files(source_root):
        rel_path = source_path.relative_to(project_root).as_posix()
        item = entries.get(rel_path)
        if item is None:
            audits.append(
                CoverageFileAudit(
                    path=rel_path,
                    line_rate=None,
                    covered_lines=None,
                    valid_lines=None,
                    missing_lines=None,
                    status="missing_from_report",
                )
            )
            continue
        line_rate = float(item.attrib.get("line-rate", "0"))
        covered, valid = _line_counts(item)
        status = "below_threshold" if line_rate * 100.0 < min_file_percent else "ok"
        audits.append(
            CoverageFileAudit(
                path=rel_path,
                line_rate=line_rate,
                covered_lines=covered,
                valid_lines=valid,
                missing_lines=valid - covered,
                status=status,
            )
        )
    return tuple(audits)


def _audit_to_dict(audit: CoverageFileAudit) -> dict[str, object]:
    """Convert one audit row to JSON-compatible data."""
    return {
        "path": audit.path,
        "line_percent": audit.line_percent,
        "covered_lines": audit.covered_lines,
        "valid_lines": audit.valid_lines,
        "missing_lines": audit.missing_lines,
        "status": audit.status,
    }


def audits_to_json(audits: Sequence[CoverageFileAudit]) -> str:
    """Serialise audits as deterministic JSON."""
    return json.dumps([_audit_to_dict(item) for item in audits], indent=2, sort_keys=True)


def format_audits(audits: Iterable[CoverageFileAudit]) -> str:
    """Render a compact human-readable coverage-gap summary."""
    rows = tuple(audits)
    missing = [item for item in rows if item.status == "missing_from_report"]
    low = [item for item in rows if item.status == "below_threshold"]
    ok = [item for item in rows if item.status == "ok"]
    measured = [item for item in rows if item.valid_lines is not None]
    covered_lines = sum(item.covered_lines or 0 for item in measured)
    valid_lines = sum(item.valid_lines or 0 for item in measured)
    aggregate = 100.0 * covered_lines / valid_lines if valid_lines else None
    lines = [
        "Coverage gap audit summary:",
        f"- source_files: {len(rows)}",
        f"- files_ok: {len(ok)}",
        f"- files_below_threshold: {len(low)}",
        f"- files_missing_from_report: {len(missing)}",
    ]
    if aggregate is not None:
        lines.append(f"- aggregate_line_percent_in_report: {aggregate:.2f}")
    elif rows:
        lines.append(
            "- coverage_report_warning: no source files from the selected "
            "package root matched the coverage XML; regenerate coverage.xml "
            "before treating this as a real coverage gap list"
        )
    for item in low[:20]:
        percent = item.line_percent if item.line_percent is not None else 0.0
        lines.append(f"- below_threshold: {item.path} ({percent:.2f}%)")
    for item in missing[:20]:
        lines.append(f"- missing_from_report: {item.path}")
    if len(low) > 20:
        lines.append(f"- additional_below_threshold_files: {len(low) - 20}")
    if len(missing) > 20:
        lines.append(f"- additional_missing_files: {len(missing) - 20}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=default_root / "src" / "scpn_quantum_control",
        help="Package source root to inventory.",
    )
    parser.add_argument(
        "--coverage-xml",
        type=Path,
        default=default_root / "coverage.xml",
        help="coverage.py XML report produced by pytest-cov.",
    )
    parser.add_argument(
        "--min-file-percent",
        type=float,
        default=95.0,
        help="Per-file line-coverage threshold for release triage.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--fail-on-gap",
        action="store_true",
        help="Return non-zero when any source file is missing or below threshold.",
    )
    args = parser.parse_args(argv)

    audits = audit_coverage_gaps(
        project_root=args.project_root,
        source_root=args.source_root,
        coverage_xml=args.coverage_xml,
        min_file_percent=args.min_file_percent,
    )
    print(audits_to_json(audits) if args.json else format_audits(audits))
    has_gap = any(item.status != "ok" for item in audits)
    return 1 if args.fail_on_gap and has_gap else 0


if __name__ == "__main__":
    raise SystemExit(main())
