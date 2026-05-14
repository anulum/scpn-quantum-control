#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 validation coverage reconciliation
"""Reconcile promoted Paper 0 validation slices against the canonical ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"
RUNTIME_SURFACE_OVERRIDES = {
    "validation_strategy": ("validation_strategy.py", "test_paper0_validation_strategy.py"),
}


@dataclass(frozen=True, slots=True)
class PromotedSlice:
    """One promoted Paper 0 validation slice discovered from a builder."""

    name: str
    builder_path: str
    runtime_module_path: str
    runner_path: str
    builder_test_path: str
    runtime_test_path: str
    runner_test_path: str
    source_start: str
    source_end: str
    source_record_count: int
    has_runtime_module: bool
    has_runner: bool
    has_builder_tests: bool
    has_runtime_tests: bool
    has_runner_tests: bool


@dataclass(frozen=True, slots=True)
class CoverageReconciliation:
    """Reconciliation output for promoted Paper 0 validation coverage."""

    slices: tuple[PromotedSlice, ...]
    gaps: tuple[tuple[str, str], ...]
    overlaps: tuple[tuple[str, str], ...]
    missing_surfaces: tuple[str, ...]
    summary: dict[str, Any]


def discover_promoted_slices(repo_root: Path = REPO_ROOT) -> tuple[PromotedSlice, ...]:
    """Discover promoted Paper 0 slices from builder SOURCE_LEDGER_IDS constants."""
    slices: list[PromotedSlice] = []
    for builder_path in sorted((repo_root / "scripts").glob("build_paper0_*_specs.py")):
        module = _import_builder(builder_path)
        source_ids = tuple(str(item) for item in getattr(module, "SOURCE_LEDGER_IDS", ()))
        if not source_ids:
            continue
        raw_name = builder_path.name.removeprefix("build_paper0_").removesuffix("_specs.py")
        public_name = raw_name.removesuffix("_validation")
        runtime_filename, runtime_test_filename = RUNTIME_SURFACE_OVERRIDES.get(
            public_name,
            (
                f"{raw_name if raw_name.endswith('_validation') else f'{raw_name}_validation'}.py",
                f"test_paper0_{public_name}_validation.py",
            ),
        )
        runtime_module = repo_root / "src" / "scpn_quantum_control" / "paper0" / runtime_filename
        runner = repo_root / "scripts" / f"run_paper0_{public_name}_fixture.py"
        builder_test = repo_root / "tests" / f"test_build_paper0_{raw_name}_specs.py"
        runtime_test = repo_root / "tests" / runtime_test_filename
        runner_test = repo_root / "tests" / f"test_run_paper0_{public_name}_fixture.py"
        slices.append(
            PromotedSlice(
                name=public_name,
                builder_path=_relative(builder_path, repo_root),
                runtime_module_path=_relative(runtime_module, repo_root),
                runner_path=_relative(runner, repo_root),
                builder_test_path=_relative(builder_test, repo_root),
                runtime_test_path=_relative(runtime_test, repo_root),
                runner_test_path=_relative(runner_test, repo_root),
                source_start=source_ids[0],
                source_end=source_ids[-1],
                source_record_count=len(source_ids),
                has_runtime_module=runtime_module.exists(),
                has_runner=runner.exists(),
                has_builder_tests=builder_test.exists(),
                has_runtime_tests=runtime_test.exists(),
                has_runner_tests=runner_test.exists(),
            )
        )
    return tuple(sorted(slices, key=lambda item: _ledger_number(item.source_start)))


def reconcile_promoted_coverage(repo_root: Path = REPO_ROOT) -> CoverageReconciliation:
    """Reconcile promoted slices against the canonical Paper 0 review ledger."""
    ledger_ids = _ledger_ids(repo_root / DEFAULT_LEDGER_PATH.relative_to(REPO_ROOT))
    slices = discover_promoted_slices(repo_root)
    gaps, overlaps = _coverage_gaps_and_overlaps(slices)
    missing_surfaces = tuple(
        surface
        for item in slices
        for surface, present in (
            (f"{item.name}:runtime:{item.runtime_module_path}", item.has_runtime_module),
            (f"{item.name}:runner:{item.runner_path}", item.has_runner),
            (f"{item.name}:builder_test:{item.builder_test_path}", item.has_builder_tests),
            (f"{item.name}:runtime_test:{item.runtime_test_path}", item.has_runtime_tests),
            (f"{item.name}:runner_test:{item.runner_test_path}", item.has_runner_tests),
        )
        if not present
    )
    promoted_start = slices[0].source_start if slices else None
    promoted_end = slices[-1].source_end if slices else None
    promoted_record_count = sum(item.source_record_count for item in slices)
    unpromoted_prefix_count = (
        _ledger_number(promoted_start) - _ledger_number(ledger_ids[0])
        if promoted_start is not None and ledger_ids
        else 0
    )
    summary = {
        "title": "Paper 0 Validation Coverage Reconciliation",
        "ledger_record_count": len(ledger_ids),
        "ledger_start": ledger_ids[0] if ledger_ids else None,
        "ledger_end": ledger_ids[-1] if ledger_ids else None,
        "promoted_slice_count": len(slices),
        "promoted_start": promoted_start,
        "promoted_end": promoted_end,
        "promoted_record_count": promoted_record_count,
        "promoted_coverage_match": not gaps and not overlaps and promoted_record_count == 918,
        "promoted_surface_integrity": not overlaps and not missing_surfaces,
        "gap_count": len(gaps),
        "gaps": [list(item) for item in gaps],
        "overlap_count": len(overlaps),
        "missing_surface_count": len(missing_surfaces),
        "unpromoted_prefix_count": unpromoted_prefix_count,
        "unpromoted_prefix_span": [ledger_ids[0], f"P0R{_ledger_number(promoted_start) - 1:05d}"]
        if promoted_start is not None and unpromoted_prefix_count > 0
        else [],
        "claim_boundary": "reconciliation audit only; not scientific validation evidence",
    }
    return CoverageReconciliation(
        slices=slices,
        gaps=gaps,
        overlaps=overlaps,
        missing_surfaces=missing_surfaces,
        summary=summary,
    )


def build_report(result: CoverageReconciliation) -> str:
    """Render a compact Markdown reconciliation report."""
    lines = [
        "# Paper 0 Validation Coverage Reconciliation",
        "",
        f"- Ledger records: {result.summary['ledger_record_count']}",
        f"- Promoted span: {result.summary['promoted_start']} - {result.summary['promoted_end']}",
        f"- Promoted records: {result.summary['promoted_record_count']}",
        f"- Promoted slices: {result.summary['promoted_slice_count']}",
        f"- Coverage match: {result.summary['promoted_coverage_match']}",
        f"- Gaps: {result.summary['gap_count']}",
        f"- Overlaps: {result.summary['overlap_count']}",
        f"- Missing surfaces: {result.summary['missing_surface_count']}",
        f"- Unpromoted prefix: {result.summary['unpromoted_prefix_span']}",
        f"- Claim boundary: {result.summary['claim_boundary']}",
        "",
        "## Promoted Slices",
    ]
    for item in result.slices:
        lines.append(
            f"- `{item.name}`: {item.source_start} - {item.source_end} "
            f"({item.source_record_count} records)"
        )
    if result.missing_surfaces:
        lines.extend(["", "## Missing Surfaces"])
        lines.extend(f"- `{item}`" for item in result.missing_surfaces)
    return "\n".join(lines) + "\n"


def write_outputs(
    result: CoverageReconciliation,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-15",
) -> dict[str, Path]:
    """Write JSON and Markdown reconciliation artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_validation_coverage_reconciliation_{date_tag}.json"
    report_path = output_dir / f"paper0_validation_coverage_reconciliation_{date_tag}.md"
    payload = {
        "slices": [asdict(item) for item in result.slices],
        "gaps": result.gaps,
        "overlaps": result.overlaps,
        "missing_surfaces": result.missing_surfaces,
        "summary": result.summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(result), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> int:
    """Run Paper 0 validation coverage reconciliation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-15")
    args = parser.parse_args()

    result = reconcile_promoted_coverage(REPO_ROOT)
    write_outputs(result, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0 if result.summary["promoted_surface_integrity"] else 1


def _coverage_gaps_and_overlaps(
    slices: tuple[PromotedSlice, ...],
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    gaps: list[tuple[str, str]] = []
    overlaps: list[tuple[str, str]] = []
    previous_end: int | None = None
    for item in slices:
        start = _ledger_number(item.source_start)
        end = _ledger_number(item.source_end)
        if previous_end is not None:
            if start > previous_end + 1:
                gaps.append((f"P0R{previous_end + 1:05d}", f"P0R{start - 1:05d}"))
            if start <= previous_end:
                overlaps.append((item.source_start, f"P0R{previous_end:05d}"))
        previous_end = end
    return tuple(gaps), tuple(overlaps)


def _ledger_ids(path: Path) -> tuple[str, ...]:
    records: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(str(json.loads(stripped)["ledger_id"]))
            except (KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid canonical ledger at {path}:{line_number}") from exc
    return tuple(records)


def _import_builder(path: Path) -> Any:
    module_name = f"_paper0_reconcile_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import builder {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ledger_number(ledger_id: str | None) -> int:
    if ledger_id is None:
        raise ValueError("ledger_id is required")
    return int(ledger_id.removeprefix("P0R"))


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
