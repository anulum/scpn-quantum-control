#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- multi-language K_nm benchmark harness
"""Benchmark K_nm construction across available language runtimes."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "rust_vqe_methods"
SRC_DIR = OUT_DIR / "multilang_src"
DATE = "2026-05-05"
NS = [4, 8, 16, 32, 64]
REPEATS = 300


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _python_knm(n: int) -> np.ndarray:
    idx = np.arange(n)
    k = 0.45 * np.exp(-0.3 * np.abs(idx[:, None] - idx[None, :]))
    anchors = {(0, 1): 0.302, (1, 2): 0.201, (2, 3): 0.252, (3, 4): 0.154}
    for (i, j), value in anchors.items():
        if i < n and j < n:
            k[i, j] = k[j, i] = value
    if n > 15:
        k[0, 15] = k[15, 0] = max(k[0, 15], 0.05)
    if n > 6:
        k[4, 6] = k[6, 4] = max(k[4, 6], 0.15)
    return k


def _median_python(n: int) -> float:
    values = []
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        _python_knm(n)
        values.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return float(statistics.median(values))


def _median_rust(n: int) -> float | None:
    try:
        import scpn_quantum_engine as engine
    except ImportError:
        return None
    if not hasattr(engine, "build_knm"):
        return None
    values = []
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        np.asarray(engine.build_knm(n, 0.45, 0.3), dtype=np.float64)
        values.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return float(statistics.median(values))


def _write_julia() -> Path:
    path = SRC_DIR / "bench_knm.jl"
    path.write_text(
        textwrap.dedent(
            """
            using JSON
            using Statistics
            function build_knm(n)
                k = Array{Float64}(undef, n, n)
                for i in 1:n, j in 1:n
                    k[i,j] = 0.45 * exp(-0.3 * abs((i-1) - (j-1)))
                end
                anchors = Dict((1,2)=>0.302, (2,3)=>0.201, (3,4)=>0.252, (4,5)=>0.154)
                for ((i,j), value) in anchors
                    if i <= n && j <= n
                        k[i,j] = value
                        k[j,i] = value
                    end
                end
                if n > 15
                    k[1,16] = max(k[1,16], 0.05)
                    k[16,1] = k[1,16]
                end
                if n > 6
                    k[5,7] = max(k[5,7], 0.15)
                    k[7,5] = k[5,7]
                end
                return k
            end
            ns = [4, 8, 16, 32, 64]
            repeats = 300
            rows = []
            for n in ns
                vals = Float64[]
                for _ in 1:repeats
                    t0 = time_ns()
                    build_knm(n)
                    push!(vals, (time_ns() - t0) / 1.0e6)
                end
                push!(rows, Dict("language"=>"julia", "n"=>n, "median_ms"=>median(vals), "status"=>"ok"))
            end
            println(JSON.json(rows))
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_go() -> Path:
    path = SRC_DIR / "bench_knm.go"
    path.write_text(
        textwrap.dedent(
            """
            package main
            import (
                "encoding/json"
                "fmt"
                "math"
                "sort"
                "time"
            )
            type Row struct { Language string `json:"language"`; N int `json:"n"`; MedianMs float64 `json:"median_ms"`; Status string `json:"status"` }
            func buildKnm(n int) [][]float64 {
                k := make([][]float64, n)
                for i := 0; i < n; i++ { k[i] = make([]float64, n) }
                for i := 0; i < n; i++ { for j := 0; j < n; j++ { k[i][j] = 0.45 * math.Exp(-0.3 * math.Abs(float64(i-j))) } }
                anchors := map[[2]int]float64{{0,1}:0.302, {1,2}:0.201, {2,3}:0.252, {3,4}:0.154}
                for ij, v := range anchors { if ij[0] < n && ij[1] < n { k[ij[0]][ij[1]] = v; k[ij[1]][ij[0]] = v } }
                if n > 15 { if k[0][15] < 0.05 { k[0][15] = 0.05 }; k[15][0] = k[0][15] }
                if n > 6 { if k[4][6] < 0.15 { k[4][6] = 0.15 }; k[6][4] = k[4][6] }
                return k
            }
            func median(vals []float64) float64 { sort.Float64s(vals); return vals[len(vals)/2] }
            func main() {
                ns := []int{4,8,16,32,64}; repeats := 300; rows := []Row{}
                for _, n := range ns {
                    vals := make([]float64, repeats)
                    for r := 0; r < repeats; r++ { t0 := time.Now(); _ = buildKnm(n); vals[r] = float64(time.Since(t0).Nanoseconds()) / 1.0e6 }
                    rows = append(rows, Row{"go", n, median(vals), "ok"})
                }
                data, _ := json.Marshal(rows); fmt.Println(string(data))
            }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _run_json(cmd: list[str], timeout_s: int = 60) -> tuple[list[dict[str, Any]] | None, str]:
    try:
        proc = subprocess.run(
            cmd, cwd=SRC_DIR, text=True, capture_output=True, timeout=timeout_s, check=False
        )
    except Exception as exc:  # noqa: BLE001
        return None, repr(exc)
    if proc.returncode != 0:
        return None, proc.stderr.strip() or proc.stdout.strip()
    try:
        return json.loads(proc.stdout), ""
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc}: {proc.stdout[:200]}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for n in NS:
        rows.append(
            {"language": "python_numpy", "n": n, "median_ms": _median_python(n), "status": "ok"}
        )
        rust_value = _median_rust(n)
        rows.append(
            {
                "language": "rust_pyo3",
                "n": n,
                "median_ms": rust_value,
                "status": "ok" if rust_value is not None else "unavailable",
            }
        )

    julia = shutil.which("julia")
    if julia:
        result, error = _run_json([julia, "--startup-file=no", str(_write_julia())], timeout_s=90)
        rows.extend(
            result
            if result is not None
            else [
                {
                    "language": "julia",
                    "n": None,
                    "median_ms": None,
                    "status": "failed",
                    "error": error,
                }
            ]
        )
    else:
        rows.append({"language": "julia", "n": None, "median_ms": None, "status": "unavailable"})

    go = shutil.which("go")
    if go:
        result, error = _run_json([go, "run", str(_write_go())], timeout_s=90)
        rows.extend(
            result
            if result is not None
            else [
                {
                    "language": "go",
                    "n": None,
                    "median_ms": None,
                    "status": "failed",
                    "error": error,
                }
            ]
        )
    else:
        rows.append({"language": "go", "n": None, "median_ms": None, "status": "unavailable"})

    mojo = shutil.which("mojo")
    rows.append(
        {
            "language": "mojo",
            "n": None,
            "median_ms": None,
            "status": "detected_not_benchmarked" if mojo else "unavailable",
            "note": "Mojo runtime detected, but no stable local K_nm kernel is implemented in this harness yet."
            if mojo
            else "",
        }
    )

    summary = {
        "date": DATE,
        "command": "PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/benchmark_multilang_knm_methods.py",
        "timing_caveat": (
            "Opportunistic local timing on a shared workstation. CPU load from other "
            "processes was not pinned or isolated; publication-grade numbers should be "
            "rerun on an isolated benchmark host with governor/load metadata."
        ),
        "rows": rows,
    }
    json_path = OUT_DIR / f"multilang_knm_benchmark_summary_{DATE}.json"
    csv_path = OUT_DIR / f"multilang_knm_benchmark_summary_{DATE}.csv"
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
