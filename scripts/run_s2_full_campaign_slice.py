#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S2 full-campaign slice runner
"""Execute a bounded no-QPU slice from the S2 full scaling campaign plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from scpn_quantum_control.benchmarks.advantage_protocol import (
    default_s2_scaling_protocol,
    validate_scaling_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench_s2_scaling_lite import build_rows as build_s2_rows  # noqa: E402

TODAY = date(2026, 5, 7).isoformat()
OUT_DIR = REPO_ROOT / "data" / "s2_advantage_scaling"
DOCS_DIR = REPO_ROOT / "docs"
DEFAULT_PLAN = OUT_DIR / f"s2_full_campaign_plan_{TODAY}.json"
DEFAULT_SIZES = (8,)
EXECUTED_STATUSES = {"ready_full_campaign", "measured_lite"}
NO_QPU_BASELINES = {
    "classical_ode",
    "dense_eigh",
    "sparse_eigsh",
    "mps_tensor_network",
    "aer_statevector",
}


def _parse_sizes(raw: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not sizes:
        raise ValueError("at least one size is required")
    return sizes


def load_plan_rows(path: Path) -> list[dict[str, object]]:
    """Load planning rows from the S2 full-campaign plan JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows_path = OUT_DIR / f"s2_full_campaign_rows_{TODAY}.csv"
    if rows_path.exists():
        with rows_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    protocol = payload.get("protocol", {})
    raise FileNotFoundError(
        f"missing planning rows CSV for protocol {protocol.get('protocol_id')}"
    )


def select_executable_plan_rows(
    plan_rows: Sequence[Mapping[str, object]],
    sizes: Sequence[int],
) -> list[dict[str, object]]:
    """Select no-QPU required rows that may be executed for the requested sizes."""
    selected: list[dict[str, object]] = []
    size_set = {int(size) for size in sizes}
    for row in plan_rows:
        baseline = str(row["baseline"])
        status = str(row["status"])
        if (
            int(str(row["n_qubits"])) in size_set
            and baseline in NO_QPU_BASELINES
            and status in EXECUTED_STATUSES
            and str(row["required"]).lower() == "true"
        ):
            selected.append(dict(row))
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _display(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def execute_slice(
    sizes: Sequence[int],
    *,
    max_dense_qubits: int,
    max_sparse_qubits: int,
    max_tn_qubits: int,
    max_statevector_qubits: int,
    plan_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute selected no-QPU rows and return rows plus summary."""
    plan_rows = load_plan_rows(plan_path)
    selected = select_executable_plan_rows(plan_rows, sizes)
    rows = build_s2_rows(
        sizes,
        max_dense_qubits=max_dense_qubits,
        max_sparse_qubits=max_sparse_qubits,
        max_tn_qubits=max_tn_qubits,
        max_statevector_qubits=max_statevector_qubits,
    )
    validation = validate_scaling_rows(default_s2_scaling_protocol(), rows)
    if not validation.valid:
        raise RuntimeError(f"S2 slice rows failed validation: {validation.to_dict()}")
    executed_labels = {(int(row["n_qubits"]), str(row["baseline"])) for row in rows}
    selected_labels = {(int(str(row["n_qubits"])), str(row["baseline"])) for row in selected}
    unplanned_rows = sorted(executed_labels - selected_labels)
    summary = {
        "schema": "scpn_s2_full_campaign_slice_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "date": TODAY,
        "hardware_submission": False,
        "qpu_minutes_spent": 0.0,
        "advantage_claim": False,
        "full_campaign_complete": False,
        "plan_path": _display(plan_path),
        "sizes": list(sizes),
        "size_gates": {
            "max_dense_qubits": max_dense_qubits,
            "max_sparse_qubits": max_sparse_qubits,
            "max_tn_qubits": max_tn_qubits,
            "max_statevector_qubits": max_statevector_qubits,
        },
        "selected_plan_rows": len(selected),
        "executed_rows": len(rows),
        "ok_rows": sum(1 for row in rows if row["status"] == "ok"),
        "skipped_rows": sum(1 for row in rows if row["status"] == "skipped"),
        "unplanned_rows": [list(item) for item in unplanned_rows],
        "validation": validation.to_dict(),
        "slice_decision": "completed_no_qpu_campaign_slice",
        "claim_boundary": {
            "allowed": [
                "bounded no-QPU S2 slice timing/memory rows",
                "row-schema and campaign-executor validation",
            ],
            "blocked": [
                "full N=4..20 campaign completion",
                "hardware scaling evidence",
                "broad quantum advantage",
                "publication crossover figure",
            ],
        },
    }
    return rows, summary


def _manifest(summary: Mapping[str, Any], *, json_path: Path, csv_path: Path) -> str:
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- scpn-quantum-control -- S2 full-campaign slice -->",
            "",
            "# S2 Full Scaling Campaign Slice",
            "",
            f"Date: {TODAY}",
            "",
            "## Decision",
            "",
            f"- Slice decision: `{summary['slice_decision']}`",
            "- Hardware submission: `False`",
            "- Advantage claim: `False`",
            "- Full campaign complete: `False`",
            f"- Executed rows: `{summary['executed_rows']}`",
            "",
            "## Artefacts",
            "",
            f"- JSON summary: `{_display(json_path)}`",
            f"- Executed rows: `{_display(csv_path)}`",
            "",
            "## Reproduction",
            "",
            "```bash",
            "./.venv-linux/bin/python scripts/run_s2_full_campaign_slice.py",
            "```",
            "",
            "## Boundary",
            "",
            "This is a bounded no-QPU execution slice. It is not the full S2",
            "campaign, not hardware evidence, and not a quantum-advantage claim.",
            "",
        ]
    )


def write_outputs(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    out_dir: Path,
    docs_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write JSON summary, CSV rows, and markdown manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(str(size) for size in summary["sizes"])
    json_path = out_dir / f"s2_full_campaign_slice_n{suffix}_{TODAY}.json"
    csv_path = out_dir / f"s2_full_campaign_slice_rows_n{suffix}_{TODAY}.csv"
    md_path = docs_dir / f"s2_full_campaign_slice_n{suffix}_{TODAY}.md"
    _write_csv(csv_path, rows)
    payload = dict(summary)
    payload["rows_sha256"] = _sha256(csv_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        _manifest(payload, json_path=json_path, csv_path=csv_path), encoding="utf-8"
    )
    return json_path, csv_path, md_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default=",".join(str(size) for size in DEFAULT_SIZES))
    parser.add_argument("--max-dense-qubits", type=int, default=8)
    parser.add_argument("--max-sparse-qubits", type=int, default=8)
    parser.add_argument("--max-tn-qubits", type=int, default=8)
    parser.add_argument("--max-statevector-qubits", type=int, default=8)
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    sizes = _parse_sizes(args.sizes)
    rows, summary = execute_slice(
        sizes,
        max_dense_qubits=args.max_dense_qubits,
        max_sparse_qubits=args.max_sparse_qubits,
        max_tn_qubits=args.max_tn_qubits,
        max_statevector_qubits=args.max_statevector_qubits,
        plan_path=args.plan_path,
    )
    json_path, csv_path, md_path = write_outputs(
        rows,
        summary,
        out_dir=args.out_dir,
        docs_dir=args.docs_dir,
    )
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"slice_decision={summary['slice_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
