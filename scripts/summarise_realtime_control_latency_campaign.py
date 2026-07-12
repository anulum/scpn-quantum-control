#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — summarise realtime control latency campaign script
# scpn-quantum-control -- realtime control latency consolidation
"""Consolidate local Rust/Python loop benchmarks with IBM runtime campaigns."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "s1_feedback_loop"
DOCS_DIR = REPO_ROOT / "docs" / "campaigns"
DATE = "2026-05-22"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-summary",
        type=Path,
        default=DATA_DIR / f"realtime_control_e2e_summary_{DATE}.json",
    )
    parser.add_argument(
        "--ibm-pattern",
        default=str(DATA_DIR / "ibm_runtime_latency_campaign_ibm_kingston_*.json"),
    )
    parser.add_argument(
        "--rust-ibm-run",
        type=Path,
        default=DATA_DIR / f"ibm_runtime_rust_latency_run_{DATE}.json",
    )
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--doc-path",
        type=Path,
        default=DOCS_DIR / f"realtime_control_latency_consolidated_{DATE}.md",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "p95": float(np.quantile(np.asarray(values, dtype=np.float64), 0.95)),
        "count": float(len(values)),
    }


def _collect_local_rows(local_summary: dict[str, Any]) -> dict[str, Any]:
    rows = local_summary["rows"]
    py = [row for row in rows if row["benchmark"] == "realtime_control_e2e"]
    rust = [row for row in rows if row["benchmark"] == "realtime_control_e2e_rust_full_loop"]
    return {
        "python_rows": py,
        "rust_rows": rust,
        "python_tick_ms": [float(row["mean_tick_latency_ms"]) for row in py],
        "rust_tick_ms": [float(row["mean_tick_latency_ms"]) for row in rust],
    }


def _collect_ibm_rows(paths: list[Path]) -> dict[str, Any]:
    s1_feedback: list[float] = []
    s1_control: list[float] = []
    capacity: list[float] = []
    job_ids: list[str] = []
    used_paths: list[str] = []
    for path in paths:
        payload = _load_json(path)
        if payload.get("status") != "submitted_and_completed":
            continue
        used_paths.append(str(path))
        for row in payload.get("s1_matrix", []):
            runtime = row.get("runtime", {})
            feedback = runtime.get("feedback_dynamic")
            control = runtime.get("control_open_loop")
            if isinstance(feedback, dict):
                s1_feedback.append(float(feedback["submit_to_result_seconds"]))
                if feedback.get("job_id"):
                    job_ids.append(str(feedback["job_id"]))
            if isinstance(control, dict):
                s1_control.append(float(control["submit_to_result_seconds"]))
                if control.get("job_id"):
                    job_ids.append(str(control["job_id"]))
        for row in payload.get("capacity_sweep", []):
            for measurement in row.get("measurements", []):
                capacity.append(float(measurement["submit_to_result_seconds"]))
                if measurement.get("job_id"):
                    job_ids.append(str(measurement["job_id"]))
    return {
        "paths": used_paths,
        "s1_feedback_seconds": s1_feedback,
        "s1_control_seconds": s1_control,
        "capacity_seconds": capacity,
        "job_ids": sorted(set(job_ids)),
    }


def _collect_rust_ibm_rows(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": None, "submit_to_done_seconds": [], "job_ids": []}
    payload = _load_json(path)
    rows = payload.get("rows", [])
    durations = [float(row["submit_to_done_seconds"]) for row in rows]
    job_ids = sorted({str(row["job_id"]) for row in rows if row.get("job_id")})
    return {"path": str(path), "submit_to_done_seconds": durations, "job_ids": job_ids}


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- scpn-quantum-control -- realtime control latency consolidation -->",
        "",
        "# Realtime Control Latency Consolidation",
        "",
        f"Timestamp: `{summary['timestamp_utc']}`",
        "",
        "## Claim Boundary",
        "",
        "- Local lane: host-only no-QPU control loop latency (Python vs Rust full loop).",
        "- IBM lane: externally visible runtime windows only (submit-to-result); no direct intra-shot hardware feedforward latency claim.",
        "",
        "## Local Host Loop (ms/tick)",
        "",
        f"- Python mean: `{summary['local']['python_tick_ms']['mean']:.6f}` ms, std: `{summary['local']['python_tick_ms']['std']:.6f}`, n={int(summary['local']['python_tick_ms']['count'])}",
        f"- Rust mean: `{summary['local']['rust_tick_ms']['mean']:.6f}` ms, std: `{summary['local']['rust_tick_ms']['std']:.6f}`, n={int(summary['local']['rust_tick_ms']['count'])}",
        "",
        "## IBM Runtime (s submit-to-result)",
        "",
        f"- Feedback dynamic mean: `{summary['ibm']['s1_feedback_seconds']['mean']:.3f}` s, std: `{summary['ibm']['s1_feedback_seconds']['std']:.3f}`, n={int(summary['ibm']['s1_feedback_seconds']['count'])}",
        f"- Control open-loop mean: `{summary['ibm']['s1_control_seconds']['mean']:.3f}` s, std: `{summary['ibm']['s1_control_seconds']['std']:.3f}`, n={int(summary['ibm']['s1_control_seconds']['count'])}",
        f"- Capacity sweep mean: `{summary['ibm']['capacity_seconds']['mean']:.3f}` s, std: `{summary['ibm']['capacity_seconds']['std']:.3f}`, n={int(summary['ibm']['capacity_seconds']['count'])}",
        "",
        "## IBM Runtime Rust Orchestrator (s submit-to-done)",
        "",
        f"- Rust submit-to-done mean: `{summary['ibm_rust']['submit_to_done_seconds']['mean']:.3f}` s, std: `{summary['ibm_rust']['submit_to_done_seconds']['std']:.3f}`, n={int(summary['ibm_rust']['submit_to_done_seconds']['count'])}",
        "",
        "## Sources",
        "",
        f"- Local summary: `{summary['source_files']['local_summary']}`",
        "- IBM campaigns:",
    ]
    for path in summary["source_files"]["ibm_campaigns"]:
        lines.append(f"  - `{path}`")
    if summary["source_files"].get("ibm_rust_run"):
        lines.append(f"- IBM Rust run: `{summary['source_files']['ibm_rust_run']}`")
    lines.extend(["", "## Job IDs"])
    for job_id in summary["ibm_job_ids"]:
        lines.append(f"- `{job_id}`")
    lines.extend(["", "## Rust Job IDs"])
    for job_id in summary["ibm_rust_job_ids"]:
        lines.append(f"- `{job_id}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Generate consolidated realtime-control latency benchmark artefacts."""
    args = _parse_args()
    local_summary = _load_json(args.local_summary)
    ibm_paths = [Path(path) for path in sorted(glob.glob(args.ibm_pattern))]
    local = _collect_local_rows(local_summary)
    ibm = _collect_ibm_rows(ibm_paths)
    rust_ibm = _collect_rust_ibm_rows(args.rust_ibm_run)
    if not ibm["s1_feedback_seconds"] or not ibm["s1_control_seconds"]:
        raise RuntimeError("no submitted IBM latency campaigns found for consolidation")
    if not rust_ibm["submit_to_done_seconds"]:
        raise RuntimeError("no Rust IBM latency run found for consolidation")
    consolidated = {
        "schema": "scpn_realtime_control_latency_consolidation_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            "local_summary": str(args.local_summary),
            "ibm_campaigns": ibm["paths"],
            "ibm_rust_run": rust_ibm["path"],
        },
        "local": {
            "python_tick_ms": _mean_std(local["python_tick_ms"]),
            "rust_tick_ms": _mean_std(local["rust_tick_ms"]),
        },
        "ibm": {
            "s1_feedback_seconds": _mean_std(ibm["s1_feedback_seconds"]),
            "s1_control_seconds": _mean_std(ibm["s1_control_seconds"]),
            "capacity_seconds": _mean_std(ibm["capacity_seconds"]),
        },
        "ibm_rust": {
            "submit_to_done_seconds": _mean_std(rust_ibm["submit_to_done_seconds"]),
        },
        "ibm_job_ids": ibm["job_ids"],
        "ibm_rust_job_ids": rust_ibm["job_ids"],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.doc_path.parent.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"realtime_control_latency_consolidated_{DATE}.json"
    out_json.write_text(json.dumps(consolidated, indent=2) + "\n", encoding="utf-8")
    args.doc_path.write_text(_render_markdown(consolidated), encoding="utf-8")
    print(f"wrote_json={out_json}")
    print(f"wrote_markdown={args.doc_path}")
    print(f"sha256_json={_sha256(out_json)}")
    print(f"sha256_markdown={_sha256(args.doc_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
