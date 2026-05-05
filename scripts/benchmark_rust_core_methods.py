#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Rust/core methods benchmark harness
"""Generate Rust/core performance tables for the methods paper."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27, knm_to_dense_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "rust_vqe_methods"
DATE = "2026-05-05"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _time_call(fn: Callable[[], object], repeats: int) -> dict[str, float]:
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns() - start) / 1_000_000.0)
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]
    return {
        "mean_ms": float(statistics.mean(values)),
        "median_ms": float(statistics.median(values)),
        "p95_ms": float(p95),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "repeats": repeats,
    }


def _python_knm(n: int, k_base: float = 0.45, k_alpha: float = 0.3) -> np.ndarray:
    idx = np.arange(n)
    k = k_base * np.exp(-k_alpha * np.abs(idx[:, None] - idx[None, :]))
    anchors = {(0, 1): 0.302, (1, 2): 0.201, (2, 3): 0.252, (3, 4): 0.154}
    for (i, j), value in anchors.items():
        if i < n and j < n:
            k[i, j] = k[j, i] = value
    if n > 15:
        k[0, 15] = k[15, 0] = max(k[0, 15], 0.05)
    if n > 6:
        k[4, 6] = k[6, 4] = max(k[4, 6], 0.15)
    return k


def _maybe_rust_knm(n: int) -> np.ndarray | None:
    try:
        import scpn_quantum_engine as engine
    except ImportError:
        return None
    if not hasattr(engine, "build_knm"):
        return None
    return np.asarray(engine.build_knm(n, 0.45, 0.3), dtype=np.float64).reshape(n, n)


def _rust_knm(n: int) -> np.ndarray:
    try:
        import scpn_quantum_engine as engine
    except ImportError as exc:
        raise RuntimeError("scpn_quantum_engine is unavailable") from exc
    if not hasattr(engine, "build_knm"):
        raise RuntimeError("scpn_quantum_engine.build_knm is unavailable")
    return np.asarray(engine.build_knm(n, 0.45, 0.3), dtype=np.float64).reshape(n, n)


def benchmark_knm() -> list[dict[str, object]]:
    rows = []
    for n in [4, 8, 16, 32, 64]:
        repeats = 1000 if n <= 32 else 300
        py_stats = _time_call(lambda n=n: _python_knm(n), repeats)
        rust_available = _maybe_rust_knm(n) is not None
        rust_stats = _time_call(lambda n=n: _rust_knm(n), repeats) if rust_available else None
        parity = bool(
            np.allclose(
                _python_knm(n),
                _rust_knm(n) if rust_available else build_knm_paper27(n),
                atol=1e-12,
            )
        )
        rows.append(
            {
                "benchmark": "knm_construction",
                "n": n,
                "python_mean_ms": py_stats["mean_ms"],
                "python_median_ms": py_stats["median_ms"],
                "rust_path_mean_ms": rust_stats["mean_ms"] if rust_stats else None,
                "rust_path_median_ms": rust_stats["median_ms"] if rust_stats else None,
                "speedup_vs_python_median": py_stats["median_ms"] / rust_stats["median_ms"]
                if rust_stats and rust_stats["median_ms"] > 0
                else None,
                "repeats": repeats,
                "rust_engine_build_knm_available": rust_available,
                "parity_with_python_reference": parity,
            }
        )
    return rows


def benchmark_dense_hamiltonian() -> list[dict[str, object]]:
    rows = []
    rng = np.random.default_rng(20260505)
    for n in [3, 4, 6, 8]:
        repeats = 50 if n <= 6 else 10
        k = build_knm_paper27(n)
        omega = rng.normal(0.0, 0.5, n)
        stats = _time_call(lambda k=k, omega=omega: knm_to_dense_matrix(k, omega), repeats)
        h = knm_to_dense_matrix(k, omega)
        hermitian_error = float(np.max(np.abs(h - h.conj().T)))
        rows.append(
            {
                "benchmark": "dense_hamiltonian_construction",
                "n": n,
                "hilbert_dim": 2**n,
                **stats,
                "hermitian_max_abs_error": hermitian_error,
            }
        )
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = benchmark_knm() + benchmark_dense_hamiltonian()
    summary = {
        "date": DATE,
        "command": "PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/benchmark_rust_core_methods.py",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "timing_caveat": (
            "Opportunistic local timing on a shared workstation. CPU load from other "
            "processes was not pinned or isolated; publication-grade numbers should be "
            "rerun on an isolated benchmark host with governor/load metadata."
        ),
        "rows": rows,
    }
    json_path = OUT_DIR / f"rust_core_benchmark_summary_{DATE}.json"
    csv_path = OUT_DIR / f"rust_core_benchmark_summary_{DATE}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_csv={_sha256(csv_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
