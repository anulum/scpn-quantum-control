#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- realtime control E2E benchmark
"""Regenerate reproducible end-to-end realtime control benchmark artefacts.

This harness benchmarks the software control loop path end-to-end:
FeedbackRunner -> RealtimeControllerScheduler -> RealtimeSyncFeedbackController.
It is intentionally no-submit and does not include provider queue, network,
or QPU runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from scpn_quantum_control.control.realtime_feedback import (
    RealtimeFeedbackConfig,
    RealtimeSyncFeedbackController,
)
from scpn_quantum_control.hardware.feedback_loop import (
    FeedbackLoopConfig,
    FeedbackLoopLatencySLA,
    FeedbackRunner,
    ProportionalMetricObserver,
    RealtimeControllerScheduler,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-05-22"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "s1_feedback_loop"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "campaigns" / f"realtime_control_e2e_benchmark_{DATE}.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run no-QPU end-to-end benchmark for the S1 realtime control loop "
            "and emit reproducible artefacts."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--steps", type=int, default=6)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coupling_matrix(n: int) -> np.ndarray:
    grid = np.arange(n, dtype=np.float64)
    distances = np.abs(grid[:, None] - grid[None, :])
    matrix = 0.32 * np.exp(-0.31 * distances)
    np.fill_diagonal(matrix, 0.0)
    return matrix.astype(np.float64)


def _omega(n: int) -> np.ndarray:
    return np.linspace(0.1, 0.85, n, dtype=np.float64)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires non-empty values")
    if len(ordered) == 1:
        return float(ordered[0])
    q = min(max(q, 0.0), 1.0)
    position = (len(ordered) - 1) * q
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    weight = position - left
    return float(ordered[left] * (1.0 - weight) + ordered[right] * weight)


def _rust_feedback_policy_available() -> bool:
    try:
        import scpn_quantum_engine as engine
    except ModuleNotFoundError:
        return False
    return callable(getattr(engine, "feedback_policy_batch", None))


def _rust_full_loop_available() -> bool:
    try:
        import scpn_quantum_engine as engine
    except ModuleNotFoundError:
        return False
    return callable(getattr(engine, "run_realtime_feedback_loop", None))


def _benchmark_case(n: int, *, repeats: int, steps: int, shots: int) -> dict[str, Any]:
    loop_latencies_ms: list[float] = []
    tick_latencies_ms: list[float] = []
    tick_latencies_p95_ms: list[float] = []
    tick_latencies_p99_ms: list[float] = []
    final_r_live: list[float] = []
    final_scale: list[float] = []
    completed_steps: list[int] = []
    breaches = 0

    for repeat in range(repeats):
        controller = RealtimeSyncFeedbackController(
            _coupling_matrix(n),
            _omega(n),
            config=RealtimeFeedbackConfig(
                target_r=0.72,
                deadband=0.03,
                base_dt=0.025,
                trotter_steps=1,
                measurement_shots=shots,
                base_gain=0.6,
                max_gain=2.0,
            ),
        )
        scheduler = RealtimeControllerScheduler(controller, base_seed=20260522 + 1000 * repeat)
        observer = ProportionalMetricObserver(
            initial_value=1.0,
            metric_name="r_live",
            target=0.72,
            gain=0.4,
            min_value=0.5,
            max_value=2.0,
            tolerance=0.003,
            label=f"s1e2e_n{n}",
        )
        runner = FeedbackRunner(
            scheduler,
            observer,
            FeedbackLoopConfig(
                max_steps=steps,
                max_total_latency_s=30.0,
                max_step_latency_s=5.0,
                max_qpu_seconds=0.0,
                latency_sla=FeedbackLoopLatencySLA(
                    max_latency_s=0.02,
                    p95_latency_s=0.02,
                    p99_latency_s=0.02,
                ),
            ),
        )
        started_ns = time.perf_counter_ns()
        try:
            history = runner.run()
        except RuntimeError:
            breaches += 1
            continue
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        loop_latencies_ms.append(elapsed_ms)
        latencies_ms = [record.latency_s * 1000.0 for record in history]
        tick_latencies_ms.extend(latencies_ms)
        tick_latencies_p95_ms.append(_quantile(latencies_ms, 0.95))
        tick_latencies_p99_ms.append(_quantile(latencies_ms, 0.99))
        completed_steps.append(len(history))
        final_r_live.append(float(history[-1].result.metrics["r_live"]))
        final_scale.append(float(history[-1].result.metrics["next_coupling_scale"]))

    successful = len(loop_latencies_ms)
    if successful == 0:
        raise RuntimeError(f"all repeats breached SLA for n={n}")
    return {
        "benchmark": "realtime_control_e2e",
        "n_qubits": n,
        "measurement_shots": shots,
        "repeats_requested": repeats,
        "repeats_successful": successful,
        "sla_breaches": breaches,
        "mean_loop_latency_ms": float(statistics.mean(loop_latencies_ms)),
        "median_loop_latency_ms": float(statistics.median(loop_latencies_ms)),
        "p95_loop_latency_ms": _quantile(loop_latencies_ms, 0.95),
        "p99_loop_latency_ms": _quantile(loop_latencies_ms, 0.99),
        "max_loop_latency_ms": float(max(loop_latencies_ms)),
        "mean_tick_latency_ms": float(statistics.mean(tick_latencies_ms)),
        "p95_tick_latency_ms": _quantile(tick_latencies_ms, 0.95),
        "p99_tick_latency_ms": _quantile(tick_latencies_ms, 0.99),
        "max_tick_latency_ms": float(max(tick_latencies_ms)),
        "mean_per_run_p95_tick_latency_ms": float(statistics.mean(tick_latencies_p95_ms)),
        "mean_per_run_p99_tick_latency_ms": float(statistics.mean(tick_latencies_p99_ms)),
        "mean_completed_steps": float(statistics.mean(completed_steps)),
        "mean_final_r_live": float(statistics.mean(final_r_live)),
        "mean_final_coupling_scale": float(statistics.mean(final_scale)),
        "rust_feedback_policy_available": _rust_feedback_policy_available(),
        "rust_full_loop_available": _rust_full_loop_available(),
        "qpu_seconds": 0.0,
    }


def _benchmark_case_rust_full_loop(
    n: int, *, repeats: int, steps: int, shots: int
) -> dict[str, Any] | None:
    try:
        import scpn_quantum_engine as engine
    except ModuleNotFoundError:
        return None
    if not callable(getattr(engine, "run_realtime_feedback_loop", None)):
        return None
    loop_latencies_ms: list[float] = []
    tick_latencies_ms: list[float] = []
    tick_latencies_p95_ms: list[float] = []
    tick_latencies_p99_ms: list[float] = []
    final_r_live: list[float] = []
    final_scale: list[float] = []
    completed_steps: list[int] = []

    theta0 = np.linspace(0.0, 0.2, n, dtype=np.float64)
    omega = _omega(n)
    k = _coupling_matrix(n)
    for _ in range(repeats):
        started_ns = time.perf_counter_ns()
        _, r_values, _, next_scales, _, _, tick_ms = engine.run_realtime_feedback_loop(
            theta0,
            omega,
            k,
            0.72,
            0.03,
            0.6,
            2.0,
            0.025,
            steps,
        )
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        ticks = np.asarray(tick_ms, dtype=np.float64).tolist()
        loop_latencies_ms.append(elapsed_ms)
        tick_latencies_ms.extend(ticks)
        tick_latencies_p95_ms.append(_quantile(ticks, 0.95))
        tick_latencies_p99_ms.append(_quantile(ticks, 0.99))
        completed_steps.append(len(ticks))
        final_r_live.append(float(np.asarray(r_values, dtype=np.float64)[-1]))
        final_scale.append(float(np.asarray(next_scales, dtype=np.float64)[-1]))
    return {
        "benchmark": "realtime_control_e2e_rust_full_loop",
        "n_qubits": n,
        "measurement_shots": shots,
        "repeats_requested": repeats,
        "repeats_successful": repeats,
        "sla_breaches": 0,
        "mean_loop_latency_ms": float(statistics.mean(loop_latencies_ms)),
        "median_loop_latency_ms": float(statistics.median(loop_latencies_ms)),
        "p95_loop_latency_ms": _quantile(loop_latencies_ms, 0.95),
        "p99_loop_latency_ms": _quantile(loop_latencies_ms, 0.99),
        "max_loop_latency_ms": float(max(loop_latencies_ms)),
        "mean_tick_latency_ms": float(statistics.mean(tick_latencies_ms)),
        "p95_tick_latency_ms": _quantile(tick_latencies_ms, 0.95),
        "p99_tick_latency_ms": _quantile(tick_latencies_ms, 0.99),
        "max_tick_latency_ms": float(max(tick_latencies_ms)),
        "mean_per_run_p95_tick_latency_ms": float(statistics.mean(tick_latencies_p95_ms)),
        "mean_per_run_p99_tick_latency_ms": float(statistics.mean(tick_latencies_p99_ms)),
        "mean_completed_steps": float(statistics.mean(completed_steps)),
        "mean_final_r_live": float(statistics.mean(final_r_live)),
        "mean_final_coupling_scale": float(statistics.mean(final_scale)),
        "rust_feedback_policy_available": True,
        "rust_full_loop_available": True,
        "qpu_seconds": 0.0,
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    rows = summary["rows"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- scpn-quantum-control -- realtime control E2E benchmark -->",
        "",
        "# Realtime Control E2E Benchmark",
        "",
        f"Date: `{summary['date']}`",
        "",
        f"Command: `{summary['command']}`",
        "",
        "## Claim Boundary",
        "",
        summary["claim_boundary"],
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(summary["environment"], indent=2, sort_keys=True),
        "```",
        "",
        "## Rows",
        "",
        "| n | repeats_successful | p95 tick ms | p99 tick ms | max tick ms | rust policy path |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['n_qubits']} | {row['repeats_successful']} | "
            f"{row['p95_tick_latency_ms']:.6f} | {row['p99_tick_latency_ms']:.6f} | "
            f"{row['max_tick_latency_ms']:.6f} | {str(row['rust_feedback_policy_available']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "This artefact is deterministic at the configuration level (fixed seeds and fixed rows),",
            "but wall-time values depend on host load, CPU governor, and thermal state. Re-run the",
            "command on an isolated benchmark host before publication-grade speed claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_benchmark(*, repeats: int, steps: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = [
        _benchmark_case(2, repeats=repeats, steps=steps, shots=32),
        _benchmark_case(3, repeats=repeats, steps=steps, shots=64),
        _benchmark_case(4, repeats=max(5, repeats - 2), steps=steps, shots=64),
    ]
    for n, shots in ((2, 32), (3, 64), (4, 64)):
        rust_row = _benchmark_case_rust_full_loop(n, repeats=repeats, steps=steps, shots=shots)
        if rust_row is not None:
            rows.append(rust_row)
    return {
        "schema": "scpn_realtime_control_e2e_benchmark_v1",
        "date": DATE,
        "command": (
            "PYTHONDONTWRITEBYTECODE=1 python scripts/benchmark_realtime_control_e2e.py "
            f"--repeats {repeats} --steps {steps}"
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "rust_feedback_policy_available": _rust_feedback_policy_available(),
        },
        "claim_boundary": (
            "No-QPU software control-loop benchmark only. Results exclude provider queue, "
            "network transport, runtime session setup, dynamic-circuit hardware latency, "
            "readout latency, and QPU execution latency."
        ),
        "rows": rows,
    }


def main() -> int:
    args = _parse_args()
    if args.repeats < 3:
        raise ValueError("repeats must be >= 3")
    if args.steps < 2:
        raise ValueError("steps must be >= 2")
    summary = run_benchmark(repeats=args.repeats, steps=args.steps)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.doc_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"realtime_control_e2e_summary_{DATE}.json"
    csv_path = args.out_dir / f"realtime_control_e2e_rows_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in summary["rows"] for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["rows"])
    args.doc_path.write_text(_render_markdown(summary), encoding="utf-8")
    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")
    print(f"wrote_markdown={args.doc_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_csv={_sha256(csv_path)}")
    print(f"sha256_markdown={_sha256(args.doc_path)}")
    print("hardware_submission=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
