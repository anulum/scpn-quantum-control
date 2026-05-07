#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S2 slice progress reporter
"""Aggregate completed S2 no-QPU scaling slices into a claim-boundary report."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 5, 7).isoformat()
DATA_DIR = REPO_ROOT / "data" / "s2_advantage_scaling"
DOCS_DIR = REPO_ROOT / "docs"
DEFAULT_SIZES = (8, 10, 12)


def _display(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _slice_paths(size: int, *, data_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    stem = f"s2_full_campaign_slice_n{size}_{TODAY}"
    return data_dir / f"{stem}.json", data_dir / f"{stem.replace('slice_', 'slice_rows_')}.csv"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_payload(row: Mapping[str, str]) -> dict[str, Any]:
    raw = row.get("metric_payload", "{}")
    parsed = ast.literal_eval(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"metric payload is not a mapping: {raw}")
    return parsed


def _row_float(row: Mapping[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0)


def summarise_slice(size: int, *, data_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Summarise one completed S2 slice from JSON and CSV artefacts."""
    json_path, rows_path = _slice_paths(size, data_dir=data_dir)
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    rows = _load_rows(rows_path)
    if not rows:
        raise ValueError(f"empty S2 slice rows: {rows_path}")
    ok_rows = [row for row in rows if row["status"] == "ok"]
    slowest = max(rows, key=lambda row: _row_float(row, "wall_time_ms"))
    max_memory = max(rows, key=lambda row: _row_float(row, "memory_bytes"))
    metrics = [_metric_payload(row) for row in rows]
    return {
        "n_qubits": size,
        "slice_json": _display(json_path),
        "slice_rows": _display(rows_path),
        "slice_decision": summary["slice_decision"],
        "hardware_submission": bool(summary["hardware_submission"]),
        "advantage_claim": bool(summary["advantage_claim"]),
        "full_campaign_complete": bool(summary["full_campaign_complete"]),
        "executed_rows": int(summary["executed_rows"]),
        "ok_rows": len(ok_rows),
        "skipped_rows": int(summary["skipped_rows"]),
        "total_wall_time_ms": round(sum(_row_float(row, "wall_time_ms") for row in rows), 6),
        "max_memory_bytes": int(_row_float(max_memory, "memory_bytes")),
        "slowest_baseline": slowest["baseline"],
        "slowest_wall_time_ms": round(_row_float(slowest, "wall_time_ms"), 6),
        "max_hilbert_dim": max(int(payload.get("hilbert_dim", 0)) for payload in metrics),
        "max_peak_tracemalloc_bytes": max(
            int(payload.get("peak_tracemalloc_bytes", 0)) for payload in metrics
        ),
    }


def aggregate_slices(sizes: Iterable[int], *, data_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Aggregate completed slices and return a no-QPU progress decision."""
    slice_summaries = [summarise_slice(size, data_dir=data_dir) for size in sizes]
    all_ok = all(item["executed_rows"] == item["ok_rows"] for item in slice_summaries)
    any_hardware = any(item["hardware_submission"] for item in slice_summaries)
    any_advantage = any(item["advantage_claim"] for item in slice_summaries)
    return {
        "schema": "scpn_s2_slice_progress_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "date": TODAY,
        "sizes": [item["n_qubits"] for item in slice_summaries],
        "slice_count": len(slice_summaries),
        "total_executed_rows": sum(item["executed_rows"] for item in slice_summaries),
        "total_ok_rows": sum(item["ok_rows"] for item in slice_summaries),
        "total_skipped_rows": sum(item["skipped_rows"] for item in slice_summaries),
        "total_wall_time_ms": round(
            sum(item["total_wall_time_ms"] for item in slice_summaries), 6
        ),
        "max_memory_bytes": max(item["max_memory_bytes"] for item in slice_summaries),
        "max_hilbert_dim": max(item["max_hilbert_dim"] for item in slice_summaries),
        "hardware_submission": any_hardware,
        "advantage_claim": any_advantage,
        "full_campaign_complete": False,
        "all_rows_ok": all_ok,
        "slice_summaries": slice_summaries,
        "progress_decision": (
            "ready_for_next_bounded_no_qpu_slice"
            if all_ok and not any_hardware and not any_advantage
            else "blocked_until_slice_anomalies_are_resolved"
        ),
        "claim_boundary": {
            "allowed": [
                "aggregate no-QPU S2 progress across completed bounded slices",
                "resource-growth warning from generated timing and memory rows",
            ],
            "blocked": [
                "hardware scaling evidence",
                "full N=4..20 campaign completion",
                "broad quantum advantage",
                "publication crossover figure",
            ],
        },
    }


def _manifest(report: Mapping[str, Any], *, json_path: Path) -> str:
    rows = [
        "| n | rows ok/executed | total wall ms | max memory bytes | slowest baseline |",
        "|---:|---:|---:|---:|---|",
    ]
    for item in report["slice_summaries"]:
        rows.append(
            "| {n} | {ok}/{executed} | {wall:.3f} | {memory} | {slowest} |".format(
                n=item["n_qubits"],
                ok=item["ok_rows"],
                executed=item["executed_rows"],
                wall=item["total_wall_time_ms"],
                memory=item["max_memory_bytes"],
                slowest=item["slowest_baseline"],
            )
        )
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- scpn-quantum-control -- S2 slice progress report -->",
            "",
            "# S2 Slice Progress Report",
            "",
            f"Date: {TODAY}",
            "",
            "## Decision",
            "",
            f"- Progress decision: `{report['progress_decision']}`",
            f"- Slice count: `{report['slice_count']}`",
            f"- Sizes: `{report['sizes']}`",
            f"- Total rows: `{report['total_ok_rows']}/{report['total_executed_rows']}` ok",
            "- Hardware submission: `False`",
            "- Advantage claim: `False`",
            "- Full campaign complete: `False`",
            "",
            "## Aggregate artefact",
            "",
            f"- JSON report: `{_display(json_path)}`",
            "",
            "## Slice summary",
            "",
            *rows,
            "",
            "## Boundary",
            "",
            "This report aggregates completed bounded no-QPU S2 slices only. It does",
            "not establish hardware scaling, full campaign completion, or quantum advantage.",
            "The next expansion should remain deliberate because the",
            "`n=12` slice already makes the dense and tensor-network rows expensive.",
            "",
        ]
    )


def write_report(
    report: Mapping[str, Any],
    *,
    out_dir: Path = DATA_DIR,
    docs_dir: Path = DOCS_DIR,
) -> tuple[Path, Path]:
    """Write the aggregate JSON report and markdown manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"s2_slice_progress_report_{TODAY}.json"
    md_path = docs_dir / f"s2_slice_progress_report_{TODAY}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_manifest(report, json_path=json_path), encoding="utf-8")
    return json_path, md_path


def _parse_sizes(raw: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not sizes:
        raise ValueError("at least one size is required")
    return sizes


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default=",".join(str(size) for size in DEFAULT_SIZES))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    report = aggregate_slices(_parse_sizes(args.sizes), data_dir=args.data_dir)
    json_path, md_path = write_report(report, out_dir=args.data_dir, docs_dir=args.docs_dir)
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"progress_decision={report['progress_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
