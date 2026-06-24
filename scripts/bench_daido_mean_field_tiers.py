# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multi-tier benchmark for the Daido m-th-harmonic mean-field force and Jacobian
"""Wall-time micro-benchmark for the mean-field force and stability Jacobian (K = 1).

Measures every installed tier (Rust, Julia, Python floor) of the Daido m-th-harmonic
mean-field force :math:`F_j = K (S_m \\cos m\\theta_j - C_m \\sin m\\theta_j)` (m = 2) and its stability Jacobian
:math:`J_{jk}` at the fixed coupling K = 1, reporting the median-of-repeats per-call
wall-time. Because the Jacobian is an N × N matrix the sizes stop at N = 2048. The output
is written as JSON so it can be embedded in ``docs/pipeline_performance.md`` without
hand-transcription, mirroring ``bench_daido_order_parameter_tiers.py``.

This script is the authoritative source for the Daido mean-field chain ordering recorded in
``src/scpn_quantum_control/accel/daido_mean_field.py``. A tier that is not installed is
recorded as ``unavailable`` rather than fabricated.
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
        description="Benchmark Daido mean-field force/Jacobian (m=2) across tiers"
    )
    parser.add_argument(
        "--sizes",
        default="4,16,64,256,1024,2048",
        help="Comma-separated N values to benchmark (Jacobian is O(N^2)).",
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
        default=Path("docs/benchmarks/daido_mean_field_tiers.json"),
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
            daido_mean_field_force,
            daido_mean_field_jacobian,
        )
    except Exception:
        return False
    try:
        daido_mean_field_force(np.zeros(8, dtype=np.float64), 1.0, 2)
        daido_mean_field_jacobian(np.zeros(8, dtype=np.float64), 1.0, 2)
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
    """Run the Daido mean-field tier benchmark CLI for the force and the Jacobian."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    sizes = [int(token) for token in args.sizes.split(",") if token.strip()]

    from scpn_quantum_control.accel import daido_mean_field as dmf
    from scpn_quantum_control.accel import dispatcher as disp

    rust_ok = disp._rust_available()
    julia_ok = disp._julia_available()
    force_tiers: dict[str, _Tier | None] = {
        "rust": partial(dmf._rust_daido_mean_field_force, coupling=1.0, m=2) if rust_ok else None,
        "julia": partial(dmf._julia_daido_mean_field_force, coupling=1.0, m=2)
        if julia_ok
        else None,
        "python": partial(dmf._python_daido_mean_field_force, coupling=1.0, m=2),
    }
    jacobian_tiers: dict[str, _Tier | None] = {
        "rust": partial(dmf._rust_daido_mean_field_jacobian, coupling=1.0, m=2)
        if rust_ok
        else None,
        "julia": partial(dmf._julia_daido_mean_field_jacobian, coupling=1.0, m=2)
        if julia_ok
        else None,
        "python": partial(dmf._python_daido_mean_field_jacobian, coupling=1.0, m=2),
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
            "coupling": 1.0,
        },
        "daido_mean_field_force": _bench_function(
            "daido_mean_field_force (K=1, m=2)", force_tiers, sizes, args, rng
        ),
        "daido_mean_field_jacobian": _bench_function(
            "daido_mean_field_jacobian (K=1, m=2)", jacobian_tiers, sizes, args, rng
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] JSON written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
