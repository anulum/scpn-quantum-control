# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum/classical co-simulation benchmark
"""Benchmark for the quantum/classical co-simulation (Task 1, diff lane).

* ``classical_substep`` — the per-step classical Kuramoto update, Rust kernel
  vs the pure-Python/NumPy reference (the polyglot comparison; this substep runs
  once per co-simulation step over the whole classical bath).
* ``cosimulate`` — end-to-end co-simulation wall time for an N=128 network with
  an 8-node quantum core.

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

from scpn_quantum_control.cosimulation import cosimulate, partition_knm
from scpn_quantum_control.cosimulation import quantum_classical as qc

_RESULT_PATH = Path(__file__).resolve().parents[1] / "results" / "cosimulation_benchmark.json"


def _engine():
    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "cosim_classical_substep"):
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


def _network(n_total: int, n_core: int, seed: int = 2026):
    rng = np.random.default_rng(seed)
    K = np.zeros((n_total, n_total))
    for i in range(n_core):
        for j in range(i + 1, n_core):
            K[i, j] = K[j, i] = 0.9
    for i in range(n_total):
        j = (i + 1) % n_total
        K[i, j] = K[j, i] = max(K[i, j], 0.05)
    omega = rng.standard_normal(n_total) * 0.3
    return K, omega


def _substep_rows(repeats: int, n_classical: int) -> dict:
    engine = _engine()
    rng = np.random.default_rng(7)
    theta = rng.uniform(-np.pi, np.pi, size=n_classical)
    omega = rng.standard_normal(n_classical)
    # Realistic classical bath: a sparse nearest-neighbour ring (each node has
    # two neighbours), matching the partitioned classical block of a real
    # network. The Rust kernel skips the zeros; the NumPy reference does not.
    K = np.zeros((n_classical, n_classical))
    for i in range(n_classical):
        j = (i + 1) % n_classical
        K[i, j] = K[j, i] = 0.05
    K = np.ascontiguousarray(K)
    da = rng.standard_normal(n_classical)
    db = rng.standard_normal(n_classical)
    python_ns = _median_ns(
        lambda: qc._classical_substep_python(theta, omega, K, da, db, 0.01), repeats
    )
    rust_ns = (
        _median_ns(
            lambda: engine.cosim_classical_substep(theta, omega, K, da, db, 0.01),
            repeats,
        )
        if engine
        else None
    )
    return {
        "n_classical": n_classical,
        "coupling_pattern": "sparse_nearest_neighbour_ring",
        "python_ns": python_ns,
        "rust_ns": rust_ns,
        "speedup": (python_ns / rust_ns) if rust_ns else None,
    }


def _cosim_rows(repeats: int, n_total: int, n_core: int, n_steps: int) -> dict:
    K, omega = _network(n_total, n_core)
    part = partition_knm(K, omega, max_quantum_nodes=n_core)
    median_ms = (
        _median_ns(
            lambda: cosimulate(K, omega, dt=0.02, n_steps=n_steps, partition=part, seed=1),
            repeats,
        )
        / 1e6
    )
    return {
        "n_total": n_total,
        "n_quantum": n_core,
        "n_steps": n_steps,
        "cross_fraction": part.conservation.cross_fraction,
        "median_ms": median_ms,
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def run(repeats: int) -> dict:
    """Run the local quantum/classical cosimulation benchmark."""
    load_before = os.getloadavg()
    substep = _substep_rows(repeats, 128)
    cosim = _cosim_rows(max(3, repeats // 4), 128, 8, 100)
    load_after = os.getloadavg()
    return {
        "benchmark": "quantum_classical_cosimulation",
        "evidence_class": "functional_non_isolated",
        "evidence_note": (
            "Shared-workstation run with no reserved cores. Functional/regression "
            "evidence only; an isolated_affinity figure requires a reserved-core run."
        ),
        "command": "python scripts/bench_cosimulation.py",
        "repeats": repeats,
        "classical_substep": substep,
        "cosimulate": cosim,
        "host": {
            "cpu_model": _cpu_model(),
            "cpu_count_logical": os.cpu_count(),
            "load_average_before": load_before,
            "load_average_after": load_after,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "engine_available": _engine() is not None,
        },
    }


def main() -> None:
    """Run the cosimulation benchmark CLI and write the JSON artefact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()

    result = run(args.repeats)
    _RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    print(f"evidence_class: {result['evidence_class']}")
    s = result["classical_substep"]
    speed = f"{s['speedup']:.1f}x" if s["speedup"] else "n/a"
    print(
        f"classical_substep (N={s['n_classical']}): python={s['python_ns'] / 1000:.2f}us  "
        f"rust={(s['rust_ns'] or 0) / 1000:.2f}us  speedup={speed}"
    )
    c = result["cosimulate"]
    print(
        f"cosimulate (N={c['n_total']}, core={c['n_quantum']}, {c['n_steps']} steps): "
        f"{c['median_ms']:.2f} ms  cross_fraction={c['cross_fraction']:.4f}"
    )
    print(f"written: {_RESULT_PATH}")


if __name__ == "__main__":
    main()
