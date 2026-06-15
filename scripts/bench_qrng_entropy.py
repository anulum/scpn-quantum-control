# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QRNG entropy benchmark
"""Benchmark for the QRNG streaming harness (QUA-C.1).

Two measurements, all on identical inputs:

* ``berlekamp_massey`` — Rust kernel vs the pure-Python fallback (the
  linear-complexity hot path), median per-block wall-time and the measured
  speed-up. This is the polyglot comparison required for the accelerated
  surface.
* ``qrng_source`` — generated bit rate per quantum measurement source kind
  (``xy_measurement``, ``bell_pair``, ``phase_estimation``) with and without Von
  Neumann debiasing.

Runs on a shared workstation are ``functional_non_isolated`` evidence only.

Usage
-----

.. code-block:: shell

    python scripts/bench_qrng_entropy.py
    python scripts/bench_qrng_entropy.py --bm-sizes 500,2000 --repeats 7
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

from scpn_quantum_control.entropy import AerQuantumEntropySource
from scpn_quantum_control.entropy.nist_sp800_22 import _berlekamp_massey_python

_RESULT_PATH = Path(__file__).resolve().parents[1] / "results" / "qrng_entropy_benchmark.json"


def _engine():
    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "nist_berlekamp_massey"):
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


def _bm_rows(sizes: list[int], repeats: int) -> list[dict]:
    engine = _engine()
    rng = np.random.default_rng(2026)
    rows = []
    for size in sizes:
        block = rng.integers(0, 2, size=size).astype(np.int8)
        python_ns = _median_ns(lambda b=block: _berlekamp_massey_python(b), repeats)
        rust_ns = None
        if engine is not None:
            arr = np.ascontiguousarray(block)
            rust_ns = _median_ns(lambda a=arr: engine.nist_berlekamp_massey(a), repeats)
        rows.append(
            {
                "block_size": size,
                "python_ns": python_ns,
                "rust_ns": rust_ns,
                "speedup": (python_ns / rust_ns) if rust_ns else None,
            }
        )
    return rows


def _qrng_rows(n_bits: int) -> list[dict]:
    rows = []
    for kind, register in (("xy_measurement", 128), ("bell_pair", 128), ("phase_estimation", 12)):
        src = AerQuantumEntropySource(kind, register_qubits=register, seed=7)
        t0 = time.perf_counter()
        bits = src.sample_bits(n_bits)
        dt = time.perf_counter() - t0
        rows.append(
            {
                "kind": kind,
                "register_qubits": register,
                "n_bits": int(bits.size),
                "raw_kbit_per_s": float(bits.size / dt / 1_000.0) if dt > 0 else 0.0,
                "mean_bit": float(bits.mean()),
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


def _cpu_governor() -> str:
    try:
        return Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        return "unknown"


def run(bm_sizes: list[int], repeats: int, qrng_bits: int) -> dict:
    load_before = os.getloadavg()
    bm = _bm_rows(bm_sizes, repeats)
    qrng = _qrng_rows(qrng_bits)
    load_after = os.getloadavg()
    return {
        "benchmark": "qrng_entropy",
        "evidence_class": "functional_non_isolated",
        "evidence_note": (
            "Shared-workstation run with no reserved cores. Use only as "
            "functional/regression evidence; an isolated_affinity figure requires "
            "a reserved-core run on the self-hosted isolated-benchmark runner."
        ),
        "command": "python scripts/bench_qrng_entropy.py",
        "repeats": repeats,
        "berlekamp_massey": bm,
        "qrng_source": qrng,
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
            "engine_available": _engine() is not None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm-sizes", default="500,1000,2000,5000")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--qrng-bits", type=int, default=200_000)
    args = parser.parse_args()
    bm_sizes = [int(item) for item in args.bm_sizes.split(",") if item.strip()]

    result = run(bm_sizes, args.repeats, args.qrng_bits)
    _RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    print(f"evidence_class: {result['evidence_class']}")
    print("Berlekamp-Massey (linear-complexity hot path):")
    for row in result["berlekamp_massey"]:
        speed = f"{row['speedup']:.1f}x" if row["speedup"] else "n/a"
        print(
            f"  size={row['block_size']:>5}  python={row['python_ns'] / 1000:.1f}us  "
            f"rust={(row['rust_ns'] or 0) / 1000:.1f}us  speedup={speed}"
        )
    print("QRNG raw throughput:")
    for row in result["qrng_source"]:
        print(
            f"  {row['kind']:18s} {row['raw_kbit_per_s']:.0f} kbit/s  mean={row['mean_bit']:.4f}"
        )
    print(f"written: {_RESULT_PATH}")


if __name__ == "__main__":
    main()
