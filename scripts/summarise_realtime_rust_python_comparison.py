#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — summarise realtime rust python comparison script
"""Build side-by-side comparison for dedicated realtime Rust vs Python IBM lanes."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "realtime_control_latency"
DOC_PATH = (
    REPO_ROOT / "docs" / "campaigns" / "realtime_control_latency_rust_vs_python_2026-05-22.md"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rust-json",
        type=Path,
        default=DATA_DIR / "ibm_runtime_realtime_rust_latency_run_2026-05-22.json",
    )
    parser.add_argument(
        "--python-json",
        type=Path,
        default=DATA_DIR / "ibm_runtime_realtime_python_latency_run_2026-05-22.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DATA_DIR / "realtime_control_latency_rust_vs_python_2026-05-22.json",
    )
    parser.add_argument("--out-md", type=Path, default=DOC_PATH)
    return parser.parse_args(argv)


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if values else 0.0


def _lane_stats(rows: list[dict]) -> dict[str, dict[str, float]]:
    by_lane: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_lane[str(row["lane"])].append(float(row["submit_to_done_seconds"]))
    out: dict[str, dict[str, float]] = {}
    for lane, vals in sorted(by_lane.items()):
        out[lane] = {"mean_s": _mean(vals), "std_s": _std(vals), "n": float(len(vals))}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Generate consolidated comparison artefacts for Python and Rust lanes."""
    args = _parse_args(argv)
    rust = json.loads(args.rust_json.read_text(encoding="utf-8"))
    py = json.loads(args.python_json.read_text(encoding="utf-8"))
    rust_rows = list(rust.get("rows", []))
    py_rows = list(py.get("rows", []))
    rust_vals = [float(r["submit_to_done_seconds"]) for r in rust_rows]
    py_vals = [float(r["submit_to_done_seconds"]) for r in py_rows]

    payload = {
        "schema": "scpn_realtime_latency_python_rust_comparison_v1",
        "sources": {"rust": str(args.rust_json), "python": str(args.python_json)},
        "overall": {
            "rust_mean_s": _mean(rust_vals),
            "python_mean_s": _mean(py_vals),
            "rust_std_s": _std(rust_vals),
            "python_std_s": _std(py_vals),
            "delta_python_minus_rust_s": _mean(py_vals) - _mean(rust_vals),
            "rust_n": float(len(rust_vals)),
            "python_n": float(len(py_vals)),
        },
        "per_lane": {"rust": _lane_stats(rust_rows), "python": _lane_stats(py_rows)},
        "job_ids": {
            "rust": [str(r["job_id"]) for r in rust_rows if r.get("job_id")],
            "python": [str(r["job_id"]) for r in py_rows if r.get("job_id")],
        },
    }

    lines = [
        "# Realtime Control Latency Rust vs Python",
        "",
        f"- Rust mean: `{payload['overall']['rust_mean_s']:.6f}` s (n={int(payload['overall']['rust_n'])})",
        f"- Python mean: `{payload['overall']['python_mean_s']:.6f}` s (n={int(payload['overall']['python_n'])})",
        f"- Delta (Python - Rust): `{payload['overall']['delta_python_minus_rust_s']:.6f}` s",
        "",
        "## Per-lane Means",
        "",
        "| Lane | Rust mean (s) | Python mean (s) |",
        "|---|---:|---:|",
    ]
    lanes = sorted(
        set(payload["per_lane"]["rust"].keys()) | set(payload["per_lane"]["python"].keys())
    )
    for lane in lanes:
        r = payload["per_lane"]["rust"].get(lane, {}).get("mean_s")
        p = payload["per_lane"]["python"].get(lane, {}).get("mean_s")
        lines.append(
            f"| `{lane}` | {f'{r:.6f}' if isinstance(r, float) else 'n/a'} | {f'{p:.6f}' if isinstance(p, float) else 'n/a'} |"
        )
    lines.append("")

    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote_json={args.out_json}")
    print(f"wrote_markdown={args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
