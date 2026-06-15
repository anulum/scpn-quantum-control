# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — NV magnetometry benchmark
"""Benchmark for the NV-centre 20 T magnetometry model (QUA-C.5).

* ``odmr_spectrum`` — Rust kernel vs the pure-NumPy fallback across
  frequency-grid sizes (the polyglot comparison).
* ``calibration`` — field-recovery accuracy across the 0.07-20 T range.

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

from scpn_quantum_control.sensing.nv_magnetometry_20T import (
    NVCenter,
    _lorentzian_dip,
    calibrate_field_from_odmr,
    odmr_resonances_hz,
    simulate_odmr_measurement,
)

_RESULT_PATH = Path(__file__).resolve().parents[1] / "results" / "nv_magnetometry_benchmark.json"


def _engine():
    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "nv_odmr_spectrum"):
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


def _odmr_rows(sizes: list[int], repeats: int) -> list[dict]:
    engine = _engine()
    centers = np.array([2.5e9, 3.3e9], dtype=np.float64)
    rows = []
    for n in sizes:
        freqs = np.linspace(2.0e9, 4.0e9, n)
        py = _median_ns(lambda f=freqs: _lorentzian_dip(f, centers, 1.0e6, 0.03), repeats)
        ru = None
        if engine is not None:
            arr = np.ascontiguousarray(freqs)
            ru = _median_ns(
                lambda a=arr: engine.nv_odmr_spectrum(a, centers, 1.0e6, 0.03), repeats
            )
        rows.append(
            {"grid_size": n, "python_ns": py, "rust_ns": ru, "speedup": (py / ru) if ru else None}
        )
    return rows


def _calibration_rows() -> list[dict]:
    nv = NVCenter()
    rows = []
    for b_true in (0.0734, 0.5, 5.0, 20.0):
        f_upper = odmr_resonances_hz(nv, b_true)[1]
        freqs = np.linspace(f_upper - 5.0e7, f_upper + 5.0e7, 4000)
        measured = simulate_odmr_measurement(
            nv=nv, freqs=freqs, field_tesla=b_true, noise_std=0.004, seed=5
        )
        cal = calibrate_field_from_odmr(freqs, measured, nv, true_field_tesla=b_true)
        rows.append(
            {
                "true_field_tesla": b_true,
                "recovered_field_tesla": cal.field_tesla,
                "abs_error_microtesla": cal.abs_error_tesla * 1e6,
            }
        )
    return rows


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def run(sizes: list[int], repeats: int) -> dict:
    load_before = os.getloadavg()
    odmr = _odmr_rows(sizes, repeats)
    calibration = _calibration_rows()
    load_after = os.getloadavg()
    return {
        "benchmark": "nv_magnetometry",
        "evidence_class": "functional_non_isolated",
        "evidence_note": (
            "Shared-workstation run with no reserved cores. Functional/regression "
            "evidence only; an isolated_affinity figure requires a reserved-core run."
        ),
        "command": "python scripts/bench_nv_magnetometry.py",
        "repeats": repeats,
        "odmr_spectrum": odmr,
        "calibration": calibration,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1024,8192,65536,262144")
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    sizes = [int(item) for item in args.sizes.split(",") if item.strip()]

    result = run(sizes, args.repeats)
    _RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    print(f"evidence_class: {result['evidence_class']}")
    print("ODMR spectrum (Rust vs NumPy):")
    for row in result["odmr_spectrum"]:
        speed = f"{row['speedup']:.2f}x" if row["speedup"] else "n/a"
        print(
            f"  grid={row['grid_size']:>7}  python={row['python_ns'] / 1000:.2f}us  "
            f"rust={(row['rust_ns'] or 0) / 1000:.2f}us  speedup={speed}"
        )
    print("Calibration (field recovery):")
    for row in result["calibration"]:
        print(f"  B={row['true_field_tesla']:>7} T  err={row['abs_error_microtesla']:.2f} uT")
    print(f"written: {_RESULT_PATH}")


if __name__ == "__main__":
    main()
