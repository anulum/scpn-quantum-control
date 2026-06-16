# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — FRC pulsed-shot cost benchmark
"""Benchmark for the FRC pulsed-shot QAOA cost (QUA-C.3).

Two measurements:

* ``mrti_growth`` — Rust kernel vs the pure-NumPy fallback for the MRTI growth
  integral across field-profile lengths (the polyglot comparison; NumPy wins on
  tiny arrays where its vectorisation amortises, Rust wins as the profile grows).
* ``schedulers`` — cost and wall-time of the QAOA solver, the classical SLSQP
  baseline, and the exact brute-force optimum.

Shared-workstation runs are ``functional_non_isolated`` evidence only.
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

from scpn_quantum_control.control.frc_pulsed_qaoa import (
    classical_sqp_schedule,
    optimal_schedule,
    solve_frc_pulsed_qaoa,
)
from scpn_quantum_control.control.qaoa_pulsed_cost import (
    FRCQAOAObjective,
    _mrti_growth_numpy,
)

_RESULT_PATH = Path(__file__).resolve().parents[1] / "results" / "frc_pulsed_qaoa_benchmark.json"


def _engine():
    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "frc_mrti_growth"):
            return engine
    except ImportError:
        pass
    return None


def _median_ns(fn, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - t0) * 1e9)
    return statistics.median(timings)


def _mrti_rows(lengths: list[int], repeats: int) -> list[dict]:
    engine = _engine()
    rng = np.random.default_rng(2026)
    rows = []
    for n in lengths:
        field = np.cumsum(rng.uniform(0.0, 0.5, size=n)).astype(np.float64)
        py = _median_ns(
            lambda f=field: _mrti_growth_numpy(f, 1e-6, 125.0, 0.9, 200.0, 2.0), repeats
        )
        ru = None
        if engine is not None:
            arr = np.ascontiguousarray(field)
            ru = _median_ns(
                lambda a=arr: engine.frc_mrti_growth(a, 1e-6, 125.0, 0.9, 200.0, 2.0, 700.0),
                repeats,
            )
        rows.append(
            {
                "profile_length": n,
                "python_ns": py,
                "rust_ns": ru,
                "speedup": (py / ru) if ru else None,
            }
        )
    return rows


def _scheduler_rows() -> list[dict]:
    target = np.linspace(0.5, 4.0, 8)
    obj = FRCQAOAObjective(
        target_s_parameter=2.5,
        bank_energy_budget_J=5e5,
        mrti_amplitude_max_m=1e-2,
        tilt_margin_required=0.3,
    )
    rows = []
    for name, fn in (
        ("bruteforce_optimal", lambda: optimal_schedule(target, 1e6, obj)),
        ("classical_sqp", lambda: classical_sqp_schedule(target, 1e6, obj, seed=0)),
        (
            "qaoa_p4",
            lambda: solve_frc_pulsed_qaoa(target, 1e6, obj, p_layers=4, restarts=8, seed=0),
        ),
    ):
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        rows.append(
            {
                "method": name,
                "cost": result.cost,
                "evaluations": result.evaluations,
                "wall_time_s": dt,
            }
        )
    optimum = rows[0]["cost"]
    for row in rows:
        row["ratio_to_optimum"] = row["cost"] / optimum if optimum > 0 else 1.0
    return rows


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def run(lengths: list[int], repeats: int) -> dict:
    """Run the local FRC pulsed-shot QAOA benchmark."""
    load_before = os.getloadavg()
    mrti = _mrti_rows(lengths, repeats)
    schedulers = _scheduler_rows()
    load_after = os.getloadavg()
    return {
        "benchmark": "frc_pulsed_qaoa",
        "evidence_class": "functional_non_isolated",
        "evidence_note": (
            "Shared-workstation run with no reserved cores. Functional/regression "
            "evidence only; an isolated_affinity figure requires a reserved-core run."
        ),
        "command": "python scripts/bench_frc_pulsed_qaoa.py",
        "repeats": repeats,
        "mrti_growth": mrti,
        "schedulers": schedulers,
        "host": {
            "cpu_model": _cpu_model(),
            "cpu_count_logical": os.cpu_count(),
            "load_average_before": load_before,
            "load_average_after": load_after,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "engine_available": _engine() is not None,
        },
    }


def main() -> None:
    """Run the FRC pulsed-shot QAOA benchmark CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="8,64,256,1024,4096")
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    lengths = [int(item) for item in args.lengths.split(",") if item.strip()]

    result = run(lengths, args.repeats)
    _RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    print(f"evidence_class: {result['evidence_class']}")
    print("MRTI growth (Rust vs NumPy):")
    for row in result["mrti_growth"]:
        speed = f"{row['speedup']:.2f}x" if row["speedup"] else "n/a"
        print(
            f"  len={row['profile_length']:>5}  python={row['python_ns'] / 1000:.2f}us  "
            f"rust={(row['rust_ns'] or 0) / 1000:.2f}us  speedup={speed}"
        )
    print("Schedulers:")
    for row in result["schedulers"]:
        print(
            f"  {row['method']:18s} cost={row['cost']:.5f}  ratio={row['ratio_to_optimum']:.4f}  "
            f"{row['wall_time_s']:.2f}s"
        )
    print(f"written: {_RESULT_PATH}")


if __name__ == "__main__":
    main()
