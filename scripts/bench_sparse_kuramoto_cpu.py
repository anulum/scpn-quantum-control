#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — sparse Kuramoto CPU scaling benchmark
"""Benchmark the sparse CPU Kuramoto path on ring networks.

The benchmark intentionally uses a nearest-neighbour ring so the edge count is
``2N`` and the run can reach ``N=10^6`` without allocating an ``N x N`` dense
coupling matrix. Dense-vs-sparse numerical parity is covered by the focused
unit tests; this script records scaling evidence and reproducibility metadata.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oscillatools.accel.sparse_kuramoto import (
    ring_sparse_coupling,
    sparse_kuramoto_euler_trajectory,
    sparse_kuramoto_rk4_trajectory,
    sparse_networked_kuramoto_force,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "benchmarks" / "sparse_kuramoto_cpu.json"
DEFAULT_SIZES = (1_000, 10_000, 100_000, 1_000_000)


@dataclass(frozen=True)
class TimingSummary:
    """A median/min/max timing summary in milliseconds."""

    median_ms: float
    min_ms: float
    max_ms: float
    samples: int

    def to_json_dict(self) -> dict[str, float | int]:
        """Return a JSON-stable timing record."""

        return {
            "median_ms": self.median_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "samples": self.samples,
        }


def _distribution_name(package: str) -> str:
    """Return an installed package version or ``unavailable``."""

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _time_call(call: Callable[[], object], samples: int) -> TimingSummary:
    """Time ``call`` for ``samples`` iterations and return milliseconds."""

    durations: list[float] = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        call()
        elapsed_ns = time.perf_counter_ns() - start
        durations.append(elapsed_ns / 1_000_000.0)
    ordered = sorted(durations)
    return TimingSummary(
        median_ms=ordered[len(ordered) // 2],
        min_ms=ordered[0],
        max_ms=ordered[-1],
        samples=samples,
    )


def _validate_sizes(sizes: Sequence[int]) -> tuple[int, ...]:
    """Return validated positive benchmark sizes."""

    if not sizes:
        raise ValueError("at least one size is required")
    validated = tuple(int(size) for size in sizes)
    if any(size < 1 for size in validated):
        raise ValueError("sizes must be positive integers")
    return validated


def _build_row(size: int, *, samples: int, n_steps: int, dt: float) -> dict[str, Any]:
    """Build one sparse-ring scaling row."""

    coupling = ring_sparse_coupling(size, coupling_strength=0.05)
    theta = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False, dtype=np.float64)
    omega = np.zeros(size, dtype=np.float64)

    sparse_networked_kuramoto_force(theta, coupling)
    sparse_kuramoto_euler_trajectory(theta, omega, coupling, dt, n_steps)
    sparse_kuramoto_rk4_trajectory(theta, omega, coupling, dt, n_steps)

    force_timing = _time_call(lambda: sparse_networked_kuramoto_force(theta, coupling), samples)
    euler_timing = _time_call(
        lambda: sparse_kuramoto_euler_trajectory(theta, omega, coupling, dt, n_steps),
        samples,
    )
    rk4_timing = _time_call(
        lambda: sparse_kuramoto_rk4_trajectory(theta, omega, coupling, dt, n_steps),
        samples,
    )

    euler_trajectory = sparse_kuramoto_euler_trajectory(theta, omega, coupling, dt, n_steps)
    rk4_trajectory = sparse_kuramoto_rk4_trajectory(theta, omega, coupling, dt, n_steps)
    return {
        "n_oscillators": size,
        "stored_edges": coupling.nnz,
        "density": coupling.density,
        "n_steps": n_steps,
        "dt": dt,
        "force": force_timing.to_json_dict(),
        "euler_trajectory": euler_timing.to_json_dict(),
        "rk4_trajectory": rk4_timing.to_json_dict(),
        "euler_terminal_order_parameter": _order_parameter(euler_trajectory[-1]),
        "rk4_terminal_order_parameter": _order_parameter(rk4_trajectory[-1]),
        "trajectory_shape": [n_steps + 1, size],
    }


def _order_parameter(theta: NDArray[np.float64]) -> float:
    """Return the Kuramoto order-parameter magnitude for one phase vector."""

    return float(abs(np.mean(np.exp(1j * theta))))


def build_artifact(
    sizes: Sequence[int],
    *,
    samples: int,
    n_steps: int,
    dt: float,
) -> dict[str, Any]:
    """Build the sparse Kuramoto CPU scaling artifact."""

    if samples < 1:
        raise ValueError("samples must be positive")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if not np.isfinite(dt):
        raise ValueError("dt must be finite")

    rows = [
        _build_row(size, samples=samples, n_steps=n_steps, dt=dt)
        for size in _validate_sizes(sizes)
    ]
    return {
        "schema": "scpn-quantum-control.sparse-kuramoto-cpu.v1",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "claim_boundary": (
            "Sparse classical CPU scaling evidence for ring-network Kuramoto force and fixed-step "
            "integrators; not quantum hardware evidence and not a broad performance claim."
        ),
        "command": [Path(sys.argv[0]).as_posix(), *sys.argv[1:]],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": _distribution_name("scipy"),
        "oscillatools": _distribution_name("oscillatools"),
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write the JSON artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    artifact = build_artifact(args.sizes, samples=args.samples, n_steps=args.n_steps, dt=args.dt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[bench] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
