# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Sub-microsecond tracker throughput benchmark
"""Throughput benchmark for the sub-microsecond realtime tracker (QUA-C.6).

Measures three paths over identical synthetic cycle-timestamp windows and
reports per-record wall-time (median of K repeats):

* ``rust_summary`` — :func:`scpn_quantum_engine.sub_us_tracker_summary`, the
  batch ingest + jitter-percentile kernel.
* ``numpy_summary`` — the pure-NumPy fallback in
  ``scpn_quantum_control.control.realtime_runtime``.
* ``python_record`` — per-call cost of ``SubMicrosecondTracker.record`` (the
  live single-cycle convenience path).

Host load, CPU governor, reserved cores, and runtime versions are recorded so
the artefact can be classified. Runs on a shared workstation are
``functional_non_isolated`` evidence only; an ``isolated_affinity`` figure
requires a reserved-core run on the self-hosted benchmark runner.

Usage
-----

.. code-block:: shell

    python scripts/bench_sub_us_tracker.py
    python scripts/bench_sub_us_tracker.py --sizes 1024,16384,65536 --repeats 9
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from scpn_quantum_control.control.realtime_runtime import (
    CycleSample,
    SubMicrosecondTracker,
    _sub_us_summary_numpy,
)

_RATE_HZ = 100_000
_PERIOD_NS = 1_000_000_000 / _RATE_HZ
_RESULT_PATH = Path(__file__).resolve().parents[1] / "results" / "sub_us_tracker_benchmark.json"


def _engine():
    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "sub_us_tracker_summary"):
            return engine
    except ImportError:
        pass
    return None


def _synthetic_window(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    intervals = rng.integers(int(_PERIOD_NS * 0.6), int(_PERIOD_NS * 1.4), size=n)
    start = np.cumsum(intervals).astype(np.int64)
    end = start + rng.integers(1_000, 9_000, size=n).astype(np.int64)
    deadline = start + 5_000
    return start, end, deadline


def _median_per_record_ns(fn, n_records: int, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - t0) / n_records * 1e9)
    return statistics.median(timings)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _cpu_governor() -> str:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return path.read_text().strip()
    except OSError:
        return "unknown"


def _engine_version(engine) -> str:
    if engine is None:
        return "unavailable"
    return getattr(engine, "__version__", "unknown")


def run(sizes: list[int], repeats: int) -> dict:
    engine = _engine()
    load_before = os.getloadavg()
    rows = []
    for n in sizes:
        start, end, deadline = _synthetic_window(n)
        samples = [
            CycleSample(
                cycle_id=i,
                start_ns=int(start[i]),
                end_ns=int(end[i]),
                deadline_ns=int(deadline[i]),
            )
            for i in range(min(n, 50_000))  # cap the Python per-record path for runtime
        ]

        numpy_ns = _median_per_record_ns(
            lambda s=start, e=end, d=deadline: _sub_us_summary_numpy(s, e, d, _PERIOD_NS),
            n,
            repeats,
        )
        rust_ns = None
        if engine is not None:
            cs = np.ascontiguousarray(start)
            ce = np.ascontiguousarray(end)
            cd = np.ascontiguousarray(deadline)
            rust_ns = _median_per_record_ns(
                lambda a=cs, b=ce, c=cd: engine.sub_us_tracker_summary(a, b, c, _PERIOD_NS),
                n,
                repeats,
            )

        def _python_record_pass(samples=samples):
            tracker = SubMicrosecondTracker(target_rate_hz=_RATE_HZ)
            for sample in samples:
                tracker.record(sample)

        python_record_ns = _median_per_record_ns(_python_record_pass, len(samples), repeats)

        rows.append(
            {
                "n_records": n,
                "rust_summary_ns_per_record": rust_ns,
                "numpy_summary_ns_per_record": numpy_ns,
                "python_record_ns_per_record": python_record_ns,
                "python_record_sample_count": len(samples),
            }
        )

    load_after = os.getloadavg()
    return {
        "benchmark": "sub_us_tracker",
        "evidence_class": "functional_non_isolated",
        "evidence_note": (
            "Shared-workstation run with no reserved cores; a concurrent agent "
            "workload may be active. Use only as functional/regression evidence. "
            "An isolated_affinity figure requires a reserved-core run on the "
            "self-hosted isolated-benchmark runner."
        ),
        "command": "python scripts/bench_sub_us_tracker.py",
        "target_rate_hz": _RATE_HZ,
        "target_period_ns": _PERIOD_NS,
        "repeats": repeats,
        "rows": rows,
        "host": {
            "cpu_model": _cpu_model(),
            "cpu_count_logical": os.cpu_count(),
            "cpu_governor": _cpu_governor(),
            "sched_affinity": sorted(os.sched_getaffinity(0)),
            "reserved_cpus": None,
            "isolation_method": "none",
            "load_average_before": load_before,
            "load_average_after": load_after,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "engine_version": _engine_version(engine),
            "engine_available": engine is not None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1024,16384,65536,262144")
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    sizes = [int(item) for item in args.sizes.split(",") if item.strip()]

    result = run(sizes, args.repeats)
    _RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    print(f"evidence_class: {result['evidence_class']}")
    for row in result["rows"]:
        print(
            f"n={row['n_records']:>8}  "
            f"rust={row['rust_summary_ns_per_record']}  "
            f"numpy={row['numpy_summary_ns_per_record']:.1f}  "
            f"python_record={row['python_record_ns_per_record']:.1f} ns/record"
        )
    print(f"written: {_RESULT_PATH}")


if __name__ == "__main__":
    main()
