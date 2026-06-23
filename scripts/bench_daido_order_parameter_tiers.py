# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multi-tier benchmark for daido_order_parameter and its gradient
"""Wall-time micro-benchmark for the ``daido_order_parameter`` dispatch chains (m = 2).

Measures every installed tier (Rust, Julia, Python floor) of the m-th Daido order
parameter :math:`r_m = |\\langle e^{i m \\theta} \\rangle|` and its gradient at the
fixed harmonic m = 2 on the same input sizes, reporting
the median-of-repeats per-call wall-time. The output is written as JSON so it can be
embedded in ``docs/pipeline_performance.md`` without hand-transcription, mirroring
``bench_order_parameter_tiers.py``.

This script is the authoritative source for the Daido chain ordering recorded in
``src/scpn_quantum_control/accel/dispatcher.py``. A tier that is not installed is recorded
as ``unavailable`` rather than fabricated.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from numpy.typing import NDArray

_Tier = Callable[[NDArray[np.float64]], Any]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the benchmark command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark daido_order_parameter (m=2) across tiers"
    )
    parser.add_argument(
        "--sizes",
        default="4,16,64,256,1024,4096,16384",
        help="Comma-separated N values to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
        help="Number of outer measurement repeats (median reported).",
    )
    parser.add_argument(
        "--inner-repeats",
        type=int,
        default=100,
        help="Number of calls per outer repeat. Per-call time = outer / inner.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/daido_order_parameter_tiers.json"),
        help="Where to write the JSON result.",
    )
    return parser.parse_args(argv)


def _measure(fn: _Tier, theta: NDArray[np.float64], inner_repeats: int) -> float:
    """Return the hot-loop wall-time in seconds per call."""
    start = time.perf_counter()
    for _ in range(inner_repeats):
        _ = fn(theta)
    stop = time.perf_counter()
    return (stop - start) / inner_repeats


def _warm_julia() -> bool:
    """Pay the one-off Julia JIT cost so measurements reflect steady state."""
    try:
        from scpn_quantum_control.accel.julia import (
            daido_order_parameter,
            daido_order_parameter_gradient,
        )
    except Exception:
        return False
    try:
        daido_order_parameter(np.zeros(8, dtype=np.float64), 2)
        daido_order_parameter_gradient(np.zeros(8, dtype=np.float64), 2)
        return True
    except Exception:
        return False


def _cpu_info() -> dict[str, str]:
    """Return the runner CPU model and core count from ``/proc/cpuinfo``."""
    try:
        with open("/proc/cpuinfo") as handle:
            first = handle.read().split("\n\n", 1)[0]
        info: dict[str, str] = {}
        for line in first.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
        return {
            "model": info.get("model name", "unknown"),
            "cores": info.get("cpu cores", "unknown"),
        }
    except OSError:
        return {"model": "unavailable", "cores": "unavailable"}


def _bench_function(
    label: str,
    tiers: dict[str, _Tier | None],
    sizes: list[int],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Benchmark one dispatched function across tiers and sizes."""
    rows: list[dict[str, Any]] = []
    print(f"\n[{label}]")
    print("{:>7s}  {:>12s}  {:>12s}  {:>12s}".format("N", "rust", "julia", "python"))
    print("-" * 49)
    for size in sizes:
        theta = rng.uniform(-np.pi, np.pi, size=size).astype(np.float64)
        entry: dict[str, Any] = {"n": size}
        cells: dict[str, str] = {}
        for name, fn in tiers.items():
            if fn is None:
                entry[name] = None
                cells[name] = "unavailable"
                continue
            samples = [_measure(fn, theta, args.inner_repeats) for _ in range(args.repeats)]
            entry[name] = {
                "median_s_per_call": median(samples),
                "samples_s_per_call": samples,
            }
            cells[name] = f"{median(samples) * 1e6:8.2f} µs"
        rows.append(entry)
        print(
            f"{size:>7d}  {cells.get('rust', ''):>12s}  "
            f"{cells.get('julia', ''):>12s}  {cells.get('python', ''):>12s}",
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Run the Daido tier benchmark CLI for the value and the gradient."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    sizes = [int(token) for token in args.sizes.split(",") if token.strip()]

    from scpn_quantum_control.accel import dispatcher as d

    rust_ok = d._rust_available()
    julia_ok = d._julia_available()
    value_tiers: dict[str, _Tier | None] = {
        "rust": partial(d._rust_daido_order_parameter, m=2) if rust_ok else None,
        "julia": partial(d._julia_daido_order_parameter, m=2) if julia_ok else None,
        "python": partial(d._python_daido_order_parameter, m=2),
    }
    gradient_tiers: dict[str, _Tier | None] = {
        "rust": partial(d._rust_daido_order_parameter_gradient, m=2) if rust_ok else None,
        "julia": partial(d._julia_daido_order_parameter_gradient, m=2) if julia_ok else None,
        "python": partial(d._python_daido_order_parameter_gradient, m=2),
    }

    if julia_ok:
        print("[bench] warming Julia JIT …", file=sys.stderr, flush=True)
        _warm_julia()
        print("[bench] Julia warm", file=sys.stderr, flush=True)

    rng = np.random.default_rng(20260623)
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
        "daido_order_parameter": _bench_function(
            "daido_order_parameter (m=2)", value_tiers, sizes, args, rng
        ),
        "daido_order_parameter_gradient": _bench_function(
            "daido_order_parameter_gradient (m=2)", gradient_tiers, sizes, args, rng
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] JSON written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
