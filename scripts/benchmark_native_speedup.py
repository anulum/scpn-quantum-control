#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native dense-Hamiltonian speedup benchmark
"""Reproducible Rust-vs-Qiskit benchmark for dense XY-Hamiltonian construction.

This harness follows the unified GOTM benchmark standard
(`agentic-shared/BENCHMARK_STANDARD.md`): warm-up then repeats, P50/P95/P99
percentiles, one row per (operation, backend), explicit unavailable backends,
and full provenance in the artefact. It re-measures the dense XY-Hamiltonian
build that the obsolete "5401x faster than Qiskit" headline referred to; the
backends are the Rust PyO3 kernel and the Qiskit ``SparsePauliOp`` dense path,
fed identical ``K``/``omega`` so the ratio is apples-to-apples and parity-checked.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.accel.rust_import import optional_rust_engine
from scpn_quantum_control.bridge import build_knm_paper27
from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_hamiltonian

SCHEMA = "scpn_quantum_control.native_speedup.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "native_speedup"
SYSTEM_SIZES = (4, 8, 10, 12)


def _stats(samples_us: list[float]) -> dict[str, float]:
    """Return P50/P95/P99/mean/min/max/throughput from microsecond samples."""
    ordered = sorted(samples_us)
    count = len(ordered)

    def _pct(fraction: float) -> float:
        return ordered[min(count - 1, int(fraction * (count - 1)))]

    mean_us = float(np.mean(ordered))
    return {
        "p50_us": float(_pct(0.50)),
        "p95_us": float(_pct(0.95)),
        "p99_us": float(_pct(0.99)),
        "mean_us": mean_us,
        "min_us": float(ordered[0]),
        "max_us": float(ordered[-1]),
        "throughput_khz": float(1_000.0 / mean_us) if mean_us > 0 else 0.0,
    }


def _measure(fn: Callable[[], object], *, warmup: int, repeats: int) -> dict[str, float]:
    """Warm up (discarded), then collect ``repeats`` perf-counter samples."""
    for _ in range(warmup):
        fn()
    samples_us: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples_us.append((time.perf_counter_ns() - start) / 1_000.0)
    return _stats(samples_us)


def _problem(system_size: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a symmetric coupling matrix and frequency vector for ``L`` oscillators."""
    coupling = build_knm_paper27(L=system_size)
    coupling = (coupling + coupling.T) / 2.0
    frequencies = np.linspace(0.1, 0.4, system_size, dtype=np.float64)
    return coupling, frequencies


def _rust_dense(
    engine: Any, coupling: NDArray[np.float64], frequencies: NDArray[np.float64], size: int
) -> NDArray[np.float64]:
    """Build the dense XY Hamiltonian operator through the Rust PyO3 kernel.

    The XY Hamiltonian is real-symmetric, so the kernel returns ``float64``; this
    is the natural operator dtype. The production ``knm_to_dense_matrix`` wrapper
    additionally casts to ``complex128`` for a complex-typed downstream API — that
    cast is a downstream cost, not part of the construction kernel, and is
    excluded from this comparison.
    """
    flat = np.asarray(
        engine.build_xy_hamiltonian_dense(
            coupling.ravel().astype(np.float64), frequencies.astype(np.float64), size
        )
    )
    return flat.reshape(2**size, 2**size)


def _qiskit_dense(
    coupling: NDArray[np.float64], frequencies: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Build the dense XY Hamiltonian through the Qiskit ``SparsePauliOp`` path."""
    return np.asarray(knm_to_hamiltonian(coupling, frequencies).to_matrix(), dtype=np.complex128)


def _repeats_for(size: int) -> tuple[int, int]:
    """Return ``(warmup, repeats)`` scaled to the dense build cost at ``L``."""
    if size <= 6:
        return 5, 200
    if size <= 8:
        return 3, 80
    if size <= 10:
        return 2, 25
    return 2, 8


def benchmark_dense_hamiltonian(sizes: tuple[int, ...]) -> list[dict[str, Any]]:
    """Benchmark the dense XY-Hamiltonian build across Rust and Qiskit backends."""
    engine = optional_rust_engine()
    rust_available = engine is not None and hasattr(engine, "build_xy_hamiltonian_dense")
    rows: list[dict[str, Any]] = []

    for size in sizes:
        warmup, repeats = _repeats_for(size)
        coupling, frequencies = _problem(size)
        parameters = {"warmup": warmup, "repeats": repeats}

        qiskit_stats = _measure(
            partial(_qiskit_dense, coupling, frequencies),
            warmup=warmup,
            repeats=repeats,
        )
        rows.append(
            {
                "name": "dense_xy_hamiltonian",
                "system_size": size,
                "hilbert_dim": 2**size,
                "backend": "qiskit_sparsepauliop",
                "stats": qiskit_stats,
                "parameters": parameters,
                "status": "measured",
            }
        )

        if rust_available:
            rust_stats = _measure(
                partial(_rust_dense, engine, coupling, frequencies, size),
                warmup=warmup,
                repeats=repeats,
            )
            parity = bool(
                np.allclose(
                    _rust_dense(engine, coupling, frequencies, size).astype(np.complex128),
                    _qiskit_dense(coupling, frequencies),
                    atol=1e-9,
                )
            )
            rows.append(
                {
                    "name": "dense_xy_hamiltonian",
                    "system_size": size,
                    "hilbert_dim": 2**size,
                    "backend": "rust_pyo3",
                    "stats": rust_stats,
                    "parameters": parameters,
                    "status": "measured",
                    "parity_vs_qiskit": parity,
                    "speedup_p50_vs_qiskit": (
                        qiskit_stats["p50_us"] / rust_stats["p50_us"]
                        if rust_stats["p50_us"] > 0
                        else None
                    ),
                }
            )
        else:
            rows.append(
                {
                    "name": "dense_xy_hamiltonian",
                    "system_size": size,
                    "hilbert_dim": 2**size,
                    "backend": "rust_pyo3",
                    "stats": None,
                    "status": "unavailable: scpn_quantum_engine.build_xy_hamiltonian_dense absent",
                }
            )
    return rows


def _git_commit() -> str:
    """Return the short HEAD commit, or ``unknown`` outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _cpu_model() -> str:
    """Return the CPU model string from ``/proc/cpuinfo`` when available."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _engine_label() -> str:
    """Return an identifier for the installed Rust engine, or its absence."""
    engine = optional_rust_engine()
    if engine is None:
        return "absent"
    return str(getattr(engine, "__version__", "installed"))


def build_artifact(sizes: tuple[int, ...]) -> dict[str, Any]:
    """Assemble the full benchmark artefact with provenance and results."""
    rows = benchmark_dense_hamiltonian(sizes)
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
        },
        "commit": _git_commit(),
        "engine": _engine_label(),
        "note": (
            "Re-measures the dense XY-Hamiltonian construction kernel for the "
            "retired '5401x faster than Qiskit' headline. rust_pyo3 builds the "
            "real float64 operator; qiskit_sparsepauliop builds via SparsePauliOp."
            " Both are parity-checked. The production knm_to_dense_matrix wrapper "
            "adds a float64->complex128 cast that dominates at large L; it is a "
            "downstream cost, excluded here. Local workstation run; for the "
            "canonical figure regenerate on the fixed CI runner."
        ),
        "results": rows,
    }


def main() -> None:
    """Run the benchmark and write the artefact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="artefact path (default: data/native_speedup/native_speedup_<date>.local.json)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(SYSTEM_SIZES),
        help="oscillator counts (L) to benchmark",
    )
    args = parser.parse_args()

    artifact = build_artifact(tuple(args.sizes))

    if args.out is not None:
        out_path = args.out
    else:
        stamp = artifact["generated_at"][:10]
        out_path = DEFAULT_OUT_DIR / f"native_speedup_{stamp}.local.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    print(f"schema={artifact['schema']} cpu={artifact['platform']['cpu_model']}")
    for row in artifact["results"]:
        if row["backend"] == "rust_pyo3" and row.get("stats"):
            speedup = row.get("speedup_p50_vs_qiskit")
            speedup_text = f"{speedup:.2f}x" if speedup is not None else "n/a"
            print(
                f"  L={row['system_size']:>2} rust p50={row['stats']['p50_us']:.1f}us "
                f"speedup_vs_qiskit={speedup_text} parity={row.get('parity_vs_qiskit')}"
            )
    print(f"artefact -> {out_path}")


if __name__ == "__main__":
    main()
