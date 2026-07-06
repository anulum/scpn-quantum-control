# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Inertial (second-order) networked-Kuramoto tier benchmark
"""Multi-tier benchmark for the inertial (second-order) networked-Kuramoto forward trajectory.

Times ``networked_inertial_trajectory`` across its Rust → Julia → Python floor tier chain on a
dense random coupling matrix over a fixed-step RK4 horizon, reporting the median-of-repeats
per-call wall-time. The three tiers are tolerance-parity (they share the RK4 arithmetic and differ
only in the coupling-force summation order), so the wall-times are like-for-like. A tier that is
not installed is recorded as ``null``. If the measured ordering changes, update the chain comments
in ``oscillatools/src/oscillatools/accel/networked_inertial.py``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from numpy.typing import NDArray

_Thunk = Callable[[], object]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        default="8,16,32,64,128",
        help="Comma-separated oscillator counts N.",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Fixed RK4 time step.")
    parser.add_argument("--n-steps", type=int, default=600, help="Number of RK4 steps.")
    parser.add_argument("--mass", type=float, default=1.0, help="Inertia m.")
    parser.add_argument("--damping", type=float, default=0.5, help="Damping γ.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
        help="Number of outer measurement repeats (median reported).",
    )
    parser.add_argument(
        "--inner-repeats",
        type=int,
        default=20,
        help="Number of calls per outer repeat. Per-call time = outer / inner.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/networked_inertial_tiers.json"),
        help="Where to write the JSON result.",
    )
    return parser.parse_args(argv)


def _measure(thunk: _Thunk, inner_repeats: int) -> float:
    """Return the hot-loop wall-time in seconds per call."""
    start = time.perf_counter()
    for _ in range(inner_repeats):
        _ = thunk()
    stop = time.perf_counter()
    return (stop - start) / inner_repeats


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


def _problem(
    size: int, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    theta0 = rng.uniform(-np.pi, np.pi, size=size).astype(np.float64)
    velocities = rng.normal(0.0, 0.3, size=size).astype(np.float64)
    omega = rng.uniform(-1.0, 1.0, size=size).astype(np.float64)
    coupling = rng.uniform(0.0, 0.4, size=(size, size)).astype(np.float64)
    coupling = 0.5 * (coupling + coupling.T)
    np.fill_diagonal(coupling, 0.0)
    return theta0, velocities, omega, coupling


def _bench(
    label: str,
    thunk_for_tier: dict[str, Callable[[int, np.random.Generator], _Thunk | None]],
    sizes: list[int],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Benchmark one route across tiers and sizes."""
    rows: list[dict[str, Any]] = []
    print(f"\n[{label}]")
    print("{:>7s}  {:>12s}  {:>12s}  {:>12s}".format("N", "rust", "julia", "python"))
    print("-" * 49)
    for size in sizes:
        entry: dict[str, Any] = {"n": size}
        cells: dict[str, str] = {}
        for name, builder in thunk_for_tier.items():
            thunk = builder(size, rng)
            if thunk is None:
                entry[name] = None
                cells[name] = "unavailable"
                continue
            samples = [_measure(thunk, args.inner_repeats) for _ in range(args.repeats)]
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


def _none(size: int, rng: np.random.Generator) -> _Thunk | None:
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the inertial forward tier benchmark across Rust → Julia → Python."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    sizes = [int(token) for token in args.sizes.split(",") if token.strip()]
    mass, damping, dt, n_steps = args.mass, args.damping, args.dt, args.n_steps

    from oscillatools.accel import dispatcher as disp
    from oscillatools.accel import networked_inertial as ni

    engine = disp.optional_rust_engine()
    rust_ok = engine is not None and hasattr(engine, "kuramoto_inertial_trajectory")
    julia_ok = False
    try:
        from oscillatools.accel.julia import kuramoto_inertial_trajectory as _ji

        _theta = np.zeros(8, dtype=np.float64)
        _k = np.zeros((8, 8), dtype=np.float64)
        _ji(_theta, _theta, _theta, _k, 1.0, 0.5, 0.01, 10)
        julia_ok = True
        print("[bench] Julia warm", file=sys.stderr, flush=True)
    except Exception:
        julia_ok = False

    def forward_builder(
        impl: Callable[..., object],
    ) -> Callable[[int, np.random.Generator], _Thunk]:
        def build(size: int, rng: np.random.Generator) -> _Thunk:
            theta0, velocities, omega, coupling = _problem(size, rng)
            return lambda: impl(theta0, velocities, omega, coupling, mass, damping, dt, n_steps)

        return build

    forward_tiers: dict[str, Callable[[int, np.random.Generator], _Thunk | None]] = {
        "rust": forward_builder(ni._rust_networked_inertial_trajectory) if rust_ok else _none,
        "julia": forward_builder(ni._julia_networked_inertial_trajectory) if julia_ok else _none,
        "python": forward_builder(ni._python_networked_inertial_trajectory),
    }

    rng = np.random.default_rng(20260706)
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
            "dt": dt,
            "n_steps": n_steps,
            "mass": mass,
            "damping": damping,
            "sizes": sizes,
        },
        "networked_inertial_trajectory": _bench(
            f"networked_inertial_trajectory (dt={dt}, n_steps={n_steps})",
            forward_tiers,
            sizes,
            args,
            rng,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
