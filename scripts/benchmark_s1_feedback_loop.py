#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — benchmark s1 feedback loop script
# scpn-quantum-control -- S1 feedback-loop latency benchmark
"""Regenerate no-QPU S1 feedback-loop latency artefacts."""

from __future__ import annotations

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
    FeedbackRunner,
    ProportionalMetricObserver,
    RealtimeControllerScheduler,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s1_feedback_loop"
DATE = "2026-05-06"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coupling_matrix(n: int) -> np.ndarray:
    indices = np.arange(n, dtype=np.float64)
    distances = np.abs(indices[:, None] - indices[None, :])
    matrix = 0.35 * np.exp(-0.25 * distances)
    np.fill_diagonal(matrix, 0.0)
    return matrix.astype(np.float64)


def _omega(n: int) -> np.ndarray:
    return np.linspace(0.1, 0.9, n, dtype=np.float64)


def _benchmark_case(n: int, *, repeats: int, steps: int, shots: int) -> dict[str, Any]:
    latencies_ms: list[float] = []
    final_r_live: list[float] = []
    final_scale: list[float] = []
    histories: list[int] = []
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
        scheduler = RealtimeControllerScheduler(controller, base_seed=20260506 + 1000 * repeat)
        observer = ProportionalMetricObserver(
            initial_value=1.0,
            metric_name="r_live",
            target=0.72,
            gain=0.4,
            min_value=0.5,
            max_value=2.0,
            tolerance=0.005,
            label=f"s1_n{n}",
        )
        runner = FeedbackRunner(
            scheduler,
            observer,
            FeedbackLoopConfig(
                max_steps=steps,
                max_total_latency_s=30.0,
                max_step_latency_s=30.0,
                max_qpu_seconds=0.0,
            ),
        )
        started = time.perf_counter_ns()
        history = runner.run()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        latencies_ms.append(elapsed_ms)
        histories.append(len(history))
        final_r_live.append(float(history[-1].result.metrics["r_live"]))
        final_scale.append(float(history[-1].result.metrics["next_coupling_scale"]))
    ordered = sorted(latencies_ms)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]
    return {
        "benchmark": "s1_realtime_controller_scheduler",
        "n_qubits": n,
        "repeats": repeats,
        "max_steps": steps,
        "measurement_shots": shots,
        "mean_latency_ms": float(statistics.mean(latencies_ms)),
        "median_latency_ms": float(statistics.median(latencies_ms)),
        "p95_latency_ms": float(p95),
        "min_latency_ms": float(min(latencies_ms)),
        "max_latency_ms": float(max(latencies_ms)),
        "mean_completed_steps": float(statistics.mean(histories)),
        "mean_final_r_live": float(statistics.mean(final_r_live)),
        "mean_final_coupling_scale": float(statistics.mean(final_scale)),
        "qpu_seconds": 0.0,
    }


def main() -> int:
    """Run the no-QPU S1 feedback-loop scheduler benchmark CLI."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        _benchmark_case(2, repeats=9, steps=4, shots=32),
        _benchmark_case(3, repeats=7, steps=4, shots=64),
        _benchmark_case(4, repeats=5, steps=4, shots=64),
    ]
    summary = {
        "date": DATE,
        "command": "PYTHONDONTWRITEBYTECODE=1 python scripts/benchmark_s1_feedback_loop.py",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "scope": (
            "No-QPU simulator latency for FeedbackRunner + RealtimeControllerScheduler "
            "wrapping RealtimeSyncFeedbackController. These timings do not include IBM "
            "Runtime session creation, queue time, provider round-trip latency, or QPU "
            "execution time."
        ),
        "timing_caveat": (
            "Opportunistic local timing on a shared workstation. Use the command field "
            "to regenerate on an isolated host before making publication-grade latency "
            "claims."
        ),
        "rows": rows,
    }
    json_path = OUT_DIR / f"s1_feedback_loop_latency_summary_{DATE}.json"
    csv_path = OUT_DIR / f"s1_feedback_loop_latency_summary_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_csv={_sha256(csv_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
