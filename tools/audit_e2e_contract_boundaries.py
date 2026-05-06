# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — E2E contract boundary audit helper
"""Inventory end-to-end and contract-test coverage boundaries."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoundarySpec:
    """Expected E2E or contract boundary category."""

    key: str
    label: str
    indicators: tuple[str, ...]


@dataclass(frozen=True)
class BoundaryAudit:
    """Audit result for one E2E or contract boundary."""

    key: str
    label: str
    indicators: tuple[str, ...]
    matching_files: tuple[str, ...]

    @property
    def covered(self) -> bool:
        """Return True when at least one test file matches the boundary."""
        return bool(self.matching_files)


BOUNDARY_SPECS = (
    BoundarySpec(
        key="hardware_qpu",
        label="hardware/QPU",
        indicators=("hardware", "runner", "backend", "qpu"),
    ),
    BoundarySpec(key="bridge", label="bridge", indicators=("bridge",)),
    BoundarySpec(
        key="sc_neurocore",
        label="SC-NeuroCore",
        indicators=("arcane_neuron", "neurocore", "snn_adapter", "snn"),
    ),
    BoundarySpec(
        key="phase_orchestrator",
        label="Phase Orchestrator",
        indicators=("orchestrator",),
    ),
    BoundarySpec(key="notebook", label="notebook workflows", indicators=("notebook", "nb")),
    BoundarySpec(key="example", label="example workflows", indicators=("example", "examples")),
)


def _normalise_path(path: Path, root: Path) -> str:
    """Return a stable POSIX path relative to the test root when possible."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _matches_boundary(path: Path, spec: BoundarySpec) -> bool:
    """Return True when a test path name matches one boundary spec."""
    haystack = path.as_posix().lower()
    return any(indicator.lower() in haystack for indicator in spec.indicators)


def audit_boundaries(tests_root: Path) -> tuple[BoundaryAudit, ...]:
    """Audit expected E2E and contract-test boundary coverage by test path."""
    test_files = sorted(path for path in tests_root.glob("test_*.py") if path.is_file())
    audits = []
    for spec in BOUNDARY_SPECS:
        matches = tuple(
            _normalise_path(path, tests_root)
            for path in test_files
            if _matches_boundary(path, spec)
        )
        audits.append(
            BoundaryAudit(
                key=spec.key,
                label=spec.label,
                indicators=spec.indicators,
                matching_files=matches,
            )
        )
    return tuple(audits)


def _audit_to_dict(audit: BoundaryAudit) -> dict[str, object]:
    """Convert one audit result to JSON-compatible data."""
    return {
        "key": audit.key,
        "label": audit.label,
        "covered": audit.covered,
        "indicators": list(audit.indicators),
        "matchingFiles": list(audit.matching_files),
    }


def audits_to_json(audits: Sequence[BoundaryAudit]) -> str:
    """Serialise E2E boundary audits as deterministic JSON."""
    return json.dumps([_audit_to_dict(item) for item in audits], indent=2, sort_keys=True)


def format_audits(audits: Iterable[BoundaryAudit]) -> str:
    """Render a compact human-readable E2E boundary summary."""
    items = tuple(audits)
    covered = sum(1 for item in items if item.covered)
    lines = [
        "E2E contract boundary audit summary:",
        f"- boundaries: {len(items)}",
        f"- covered: {covered}",
        f"- missing: {len(items) - covered}",
    ]
    for item in items:
        status = "covered" if item.covered else "missing"
        files = ", ".join(item.matching_files) if item.matching_files else "none"
        lines.append(f"- {item.key}: {status} ({files})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests",
        help="Directory containing pytest modules.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Return non-zero when any boundary category has no matching test file.",
    )
    args = parser.parse_args(argv)

    audits = audit_boundaries(args.tests_root)
    print(audits_to_json(audits) if args.json else format_audits(audits))
    has_missing = any(not item.covered for item in audits)
    return 1 if args.fail_on_missing and has_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
