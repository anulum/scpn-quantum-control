#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S2 n=14 resource gate reporter
"""Record the resource gate before promoting an S2 n=14 no-QPU slice."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 5, 7).isoformat()
DATA_DIR = REPO_ROOT / "data" / "s2_advantage_scaling"
DOCS_DIR = REPO_ROOT / "docs"
DEFAULT_PLAN_ROWS = DATA_DIR / f"s2_full_campaign_rows_{TODAY}.csv"
DEFAULT_PROGRESS = DATA_DIR / f"s2_slice_progress_report_{TODAY}.json"
TARGET_N = 14


def _display(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _load_plan_rows(path: Path, *, n_qubits: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if int(row["n_qubits"]) == n_qubits]


def build_report(
    *,
    plan_rows_path: Path = DEFAULT_PLAN_ROWS,
    progress_path: Path = DEFAULT_PROGRESS,
    n_qubits: int = TARGET_N,
) -> dict[str, Any]:
    """Build a deterministic resource gate report for the next S2 slice."""
    rows = _load_plan_rows(plan_rows_path, n_qubits=n_qubits)
    if not rows:
        raise ValueError(f"no S2 plan rows found for n={n_qubits}")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    required_rows = [row for row in rows if row["required"].lower() == "true"]
    dense_bytes = max(int(row["estimated_dense_matrix_bytes"]) for row in required_rows)
    statevector_bytes = max(int(row["estimated_statevector_bytes"]) for row in required_rows)
    prior_max_memory = int(progress["max_memory_bytes"])
    memory_ratio = dense_bytes / prior_max_memory if prior_max_memory else float("inf")
    required_statuses = sorted({row["status"] for row in required_rows})
    baselines = [row["baseline"] for row in required_rows]
    return {
        "schema": "scpn_s2_n14_resource_gate_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "date": TODAY,
        "n_qubits": n_qubits,
        "plan_rows": _display(plan_rows_path),
        "progress_report": _display(progress_path),
        "required_baselines": baselines,
        "required_statuses": required_statuses,
        "estimated_dense_matrix_bytes": dense_bytes,
        "estimated_statevector_bytes": statevector_bytes,
        "prior_max_memory_bytes": prior_max_memory,
        "dense_to_prior_memory_ratio": round(memory_ratio, 6),
        "prior_completed_sizes": progress["sizes"],
        "prior_total_ok_rows": progress["total_ok_rows"],
        "prior_total_executed_rows": progress["total_executed_rows"],
        "hardware_submission": False,
        "qpu_minutes_spent": 0.0,
        "advantage_claim": False,
        "full_campaign_complete": False,
        "interactive_n14_promotion": False,
        "resource_gate_decision": "blocked_for_scheduled_or_offloaded_no_qpu_run",
        "recommended_next": [
            "schedule n=14 as a deliberate long local run with runtime budget",
            "or offload n=14 classical rows to ML350 or Vertex before promotion",
            "or promote an explicitly capped scout with skipped dense/TN rows as a scout, not a completed slice",
        ],
        "claim_boundary": {
            "allowed": [
                "resource-gate decision for the next S2 no-QPU slice",
                "comparison of n=14 estimated memory against committed n=8..12 progress",
            ],
            "blocked": [
                "n=14 slice completion",
                "hardware scaling evidence",
                "full S2 campaign completion",
                "broad quantum advantage",
            ],
        },
    }


def _manifest(report: Mapping[str, Any], *, json_path: Path) -> str:
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- scpn-quantum-control -- S2 n=14 resource gate -->",
            "",
            "# S2 n=14 Resource Gate",
            "",
            f"Date: {TODAY}",
            "",
            "## Decision",
            "",
            f"- Resource gate decision: `{report['resource_gate_decision']}`",
            "- Hardware submission: `False`",
            "- Advantage claim: `False`",
            "- Full campaign complete: `False`",
            "- Interactive n=14 promotion: `False`",
            "",
            "## Resource comparison",
            "",
            f"- n=14 estimated dense matrix bytes: `{report['estimated_dense_matrix_bytes']}`",
            f"- n=14 estimated statevector bytes: `{report['estimated_statevector_bytes']}`",
            f"- Prior n=8..12 max recorded memory bytes: `{report['prior_max_memory_bytes']}`",
            f"- Dense/prior-memory ratio: `{report['dense_to_prior_memory_ratio']}`",
            "",
            "## Artefact",
            "",
            f"- JSON report: `{_display(json_path)}`",
            "",
            "## Boundary",
            "",
            "This is a resource-gate report, not an n=14 execution result. It does",
            "not establish hardware scaling, full S2 completion, or quantum advantage.",
            "",
            "## Recommended next step",
            "",
            "Run n=14 only as a scheduled/offloaded no-QPU job, or explicitly label",
            "a capped run with skipped dense/TN rows as a scout rather than a completed",
            "full slice.",
            "",
        ]
    )


def write_report(
    report: Mapping[str, Any],
    *,
    out_dir: Path = DATA_DIR,
    docs_dir: Path = DOCS_DIR,
) -> tuple[Path, Path]:
    """Write JSON and Markdown artefacts for the resource gate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"s2_n14_resource_gate_{TODAY}.json"
    md_path = docs_dir / f"s2_n14_resource_gate_{TODAY}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_manifest(report, json_path=json_path), encoding="utf-8")
    return json_path, md_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-rows", type=Path, default=DEFAULT_PLAN_ROWS)
    parser.add_argument("--progress-report", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--n-qubits", type=int, default=TARGET_N)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    report = build_report(
        plan_rows_path=args.plan_rows,
        progress_path=args.progress_report,
        n_qubits=args.n_qubits,
    )
    json_path, md_path = write_report(report, out_dir=args.out_dir, docs_dir=args.docs_dir)
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"resource_gate_decision={report['resource_gate_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
