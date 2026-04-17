# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multi-tier benchmark for order_parameter
"""Wall-time micro-benchmark for the order_parameter dispatch chain.

Measures all installed tiers (Rust, Julia, Python floor) on the same
input sizes and reports median-of-K-repeats wall-time. The output is
written as JSON next to the script so it can be embedded in
``docs/pipeline_performance.md`` without hand-transcription.

This script is the authoritative source for the per-chain ordering
recorded in ``src/scpn_quantum_control/accel/dispatcher.py``. Every
re-run timestamps a new row; the ordering comment in the dispatcher
must be updated if the measured ordering changes.

Usage
-----

.. code-block:: shell

    python scripts/bench_order_parameter_tiers.py
    python scripts/bench_order_parameter_tiers.py --sizes 4,16,64,256
    python scripts/bench_order_parameter_tiers.py --repeats 11

Notes
-----

* Julia pays a one-off JIT boot cost on first call (~20 s). The
  benchmark explicitly warms Julia before the measurement loop so
  subsequent calls reflect steady-state performance, not boot
  overhead.
* Wall-time is measured with ``time.perf_counter`` around a hot
  loop of ``inner_repeats`` calls per measurement; the reported
  figure is the per-call median-of-repeats.
* The Rust tier delegates to ``scpn_quantum_engine.order_parameter``.
  If that wheel is missing, the Rust row is marked ``unavailable``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark order_parameter across tiers",
    )
    p.add_argument(
        "--sizes",
        default="4,16,64,256,1024,4096,16384",
        help="Comma-separated N values to benchmark.",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=7,
        help="Number of outer measurement repeats (median reported).",
    )
    p.add_argument(
        "--inner-repeats",
        type=int,
        default=100,
        help="Number of calls per outer repeat. Per-call time = outer / inner.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/order_parameter_tiers.json"),
        help="Where to write the JSON result.",
    )
    return p.parse_args(argv)


def _measure(fn, theta: np.ndarray, inner_repeats: int) -> float:
    """Median-excluding hot-loop wall-time, in seconds per call."""
    t0 = time.perf_counter()
    for _ in range(inner_repeats):
        _ = fn(theta)
    t1 = time.perf_counter()
    return (t1 - t0) / inner_repeats


def _warm_julia() -> bool:
    """Pay the one-off JIT cost so measurements reflect steady state."""
    try:
        from scpn_quantum_control.accel.julia import order_parameter as julia_op
    except Exception:
        return False
    try:
        julia_op(np.zeros(8))
        return True
    except Exception:
        return False


def _cpu_info() -> dict:
    try:
        with open("/proc/cpuinfo") as f:
            first = f.read().split("\n\n", 1)[0]
        info: dict = {}
        for line in first.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
        return {
            "model": info.get("model name", "unknown"),
            "cores": info.get("cpu cores", "unknown"),
        }
    except Exception:
        return {"model": "unavailable", "cores": "unavailable"}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    from scpn_quantum_control.accel import dispatcher as d

    tiers: dict[str, object] = {
        "rust": d._rust_order_parameter if d._rust_available() else None,
        "julia": d._julia_order_parameter if d._julia_available() else None,
        "python": d._python_order_parameter,
    }

    if tiers["julia"] is not None:
        print("[bench] warming Julia JIT …", file=sys.stderr, flush=True)
        _warm_julia()
        print("[bench] Julia warm", file=sys.stderr, flush=True)

    rng = np.random.default_rng(20260417)

    rows: list[dict] = []
    print(
        "{:>7s}  {:>12s}  {:>12s}  {:>12s}".format(
            "N",
            "rust",
            "julia",
            "python",
        ),
    )
    print("-" * 49)
    for n in sizes:
        theta = rng.uniform(-np.pi, np.pi, size=n)
        entry: dict = {"n": n}
        cell_strings: dict[str, str] = {}
        for name, fn in tiers.items():
            if fn is None:
                entry[name] = None
                cell_strings[name] = "unavailable"
                continue
            samples = [_measure(fn, theta, args.inner_repeats) for _ in range(args.repeats)]
            med = median(samples)
            entry[name] = {
                "median_s_per_call": med,
                "samples_s_per_call": samples,
            }
            cell_strings[name] = f"{med * 1e6:8.2f} µs"
        rows.append(entry)
        print(
            f"{n:>7d}  {cell_strings.get('rust', ''):>12s}  "
            f"{cell_strings.get('julia', ''):>12s}  "
            f"{cell_strings.get('python', ''):>12s}",
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runner": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": _cpu_info(),
        },
        "parameters": {
            "repeats": args.repeats,
            "inner_repeats": args.inner_repeats,
            "sizes": sizes,
        },
        "rows": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] JSON written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
