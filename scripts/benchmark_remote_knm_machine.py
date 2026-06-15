# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Remote K_nm Benchmark Harness
"""Run self-contained K_nm construction benchmarks on one machine.

The harness intentionally avoids importing :mod:`scpn_quantum_control` so the
same file can run on secondary machines before the full project environment is
installed.  It benchmarks equivalent dense K_nm construction kernels in Python,
Rust, and Go when the corresponding compilers are available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any


def build_knm_python(n: int, k0: float, alpha: float) -> list[list[float]]:
    """Construct the exponentially decaying dense coupling matrix."""

    matrix: list[list[float]] = []
    for row_index in range(n):
        row: list[float] = []
        for column_index in range(n):
            if row_index == column_index:
                row.append(0.0)
            else:
                row.append(k0 * math.exp(-alpha * abs(row_index - column_index)))
        matrix.append(row)
    return matrix


def time_python(n: int, iterations: int, k0: float, alpha: float) -> dict[str, Any]:
    """Time the built-in Python implementation."""

    samples: list[float] = []
    checksum = 0.0
    for _ in range(iterations):
        start = time.perf_counter_ns()
        matrix = build_knm_python(n, k0, alpha)
        elapsed = time.perf_counter_ns() - start
        checksum = sum(sum(row) for row in matrix)
        samples.append(elapsed / 1_000_000.0)
    return summarise_samples("python", n, samples, checksum)


def summarise_samples(
    language: str,
    n: int,
    samples_ms: list[float],
    checksum: float,
) -> dict[str, Any]:
    """Summarise raw timing samples."""

    return {
        "language": language,
        "n": n,
        "status": "ok",
        "iterations": len(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "checksum": checksum,
    }


def write_rust_source(path: Path) -> None:
    """Write the Rust benchmark implementation."""

    path.write_text(
        textwrap.dedent(
            """
            use std::env;
            use std::time::Instant;

            fn build_knm(n: usize, k0: f64, alpha: f64) -> Vec<f64> {
                let mut matrix = vec![0.0_f64; n * n];
                for i in 0..n {
                    for j in 0..n {
                        if i != j {
                            let distance = i.abs_diff(j) as f64;
                            matrix[i * n + j] = k0 * (-alpha * distance).exp();
                        }
                    }
                }
                matrix
            }

            fn main() {
                let args: Vec<String> = env::args().collect();
                let n: usize = args[1].parse().unwrap();
                let iterations: usize = args[2].parse().unwrap();
                let k0: f64 = args[3].parse().unwrap();
                let alpha: f64 = args[4].parse().unwrap();
                let mut samples = Vec::with_capacity(iterations);
                let mut checksum = 0.0_f64;
                for _ in 0..iterations {
                    let start = Instant::now();
                    let matrix = build_knm(n, k0, alpha);
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    checksum = matrix.iter().sum();
                    samples.push(elapsed);
                }
                samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = samples[samples.len() / 2];
                let mean = samples.iter().sum::<f64>() / samples.len() as f64;
                println!(
                    "{{\\"language\\":\\"rust\\",\\"n\\":{},\\"status\\":\\"ok\\",\\"iterations\\":{},\\"median_ms\\":{},\\"mean_ms\\":{},\\"min_ms\\":{},\\"max_ms\\":{},\\"checksum\\":{}}}",
                    n,
                    iterations,
                    median,
                    mean,
                    samples[0],
                    samples[samples.len() - 1],
                    checksum
                );
            }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def write_go_source(path: Path) -> None:
    """Write the Go benchmark implementation."""

    path.write_text(
        textwrap.dedent(
            """
            package main

            import (
                "fmt"
                "math"
                "os"
                "sort"
                "strconv"
                "time"
            )

            func buildKnm(n int, k0 float64, alpha float64) []float64 {
                matrix := make([]float64, n*n)
                for i := 0; i < n; i++ {
                    for j := 0; j < n; j++ {
                        if i != j {
                            distance := math.Abs(float64(i - j))
                            matrix[i*n+j] = k0 * math.Exp(-alpha*distance)
                        }
                    }
                }
                return matrix
            }

            func main() {
                n, _ := strconv.Atoi(os.Args[1])
                iterations, _ := strconv.Atoi(os.Args[2])
                k0, _ := strconv.ParseFloat(os.Args[3], 64)
                alpha, _ := strconv.ParseFloat(os.Args[4], 64)
                samples := make([]float64, iterations)
                checksum := 0.0
                for iteration := 0; iteration < iterations; iteration++ {
                    start := time.Now()
                    matrix := buildKnm(n, k0, alpha)
                    samples[iteration] = float64(time.Since(start).Nanoseconds()) / 1000000.0
                    checksum = 0.0
                    for _, value := range matrix {
                        checksum += value
                    }
                }
                sort.Float64s(samples)
                sum := 0.0
                for _, value := range samples {
                    sum += value
                }
                fmt.Printf(
                    "{\\"language\\":\\"go\\",\\"n\\":%d,\\"status\\":\\"ok\\",\\"iterations\\":%d,\\"median_ms\\":%.12f,\\"mean_ms\\":%.12f,\\"min_ms\\":%.12f,\\"max_ms\\":%.12f,\\"checksum\\":%.12f}\\n",
                    n,
                    iterations,
                    samples[len(samples)/2],
                    sum/float64(len(samples)),
                    samples[0],
                    samples[len(samples)-1],
                    checksum,
                )
            }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def run_json_command(command: list[str], cwd: Path) -> dict[str, Any]:
    """Run a benchmark command and parse its JSON output."""

    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip())


def command_failure(language: str, n: int, reason: str) -> dict[str, Any]:
    """Return a structured unavailable row."""

    return {
        "language": language,
        "n": n,
        "status": "unavailable",
        "reason": reason,
    }


def benchmark_compiled_language(
    language: str,
    compiler: str,
    source_name: str,
    binary_name: str,
    writer: Any,
    ns: argparse.Namespace,
    workspace: Path,
) -> list[dict[str, Any]]:
    """Compile and run a generated Rust or Go kernel."""

    if shutil.which(compiler) is None:
        return [command_failure(language, n, f"{compiler} not found") for n in ns.sizes]

    source_path = workspace / source_name
    binary_path = workspace / binary_name
    writer(source_path)
    if language == "rust":
        compile_command = [compiler, "-C", "opt-level=3", str(source_path), "-o", str(binary_path)]
    else:
        compile_command = [compiler, "build", "-o", str(binary_path), str(source_path)]

    try:
        subprocess.run(compile_command, cwd=workspace, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        return [
            command_failure(language, n, exc.stderr.strip() or exc.stdout.strip())
            for n in ns.sizes
        ]

    rows: list[dict[str, Any]] = []
    for n in ns.sizes:
        try:
            rows.append(
                run_json_command(
                    [
                        str(binary_path),
                        str(n),
                        str(ns.iterations),
                        str(ns.k0),
                        str(ns.alpha),
                    ],
                    workspace,
                )
            )
        except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            rows.append(command_failure(language, n, str(exc)))
    return rows


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 digest for an artefact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def machine_metadata() -> dict[str, Any]:
    """Collect machine metadata without requiring privileged access."""

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "timing_caveat": (
            "Opportunistic shared-machine timings; CPU affinity, turbo state, "
            "thermal state, and competing workloads were not controlled."
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/rust_vqe_methods"))
    parser.add_argument("--label", default=None)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--k0", type=float, default=0.45)
    parser.add_argument("--alpha", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    """Run benchmarks and write JSON/CSV artefacts."""

    ns = parse_args()
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    label = ns.label or socket.gethostname().replace(".", "_")
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    rows: list[dict[str, Any]] = []
    rows.extend(time_python(n, ns.iterations, ns.k0, ns.alpha) for n in ns.sizes)

    with tempfile.TemporaryDirectory(prefix="scpn_knm_bench_") as temp_directory:
        workspace = Path(temp_directory)
        rows.extend(
            benchmark_compiled_language(
                "rust",
                "rustc",
                "bench_knm.rs",
                "bench_knm_rust",
                write_rust_source,
                ns,
                workspace,
            )
        )
        rows.extend(
            benchmark_compiled_language(
                "go",
                "go",
                "bench_knm.go",
                "bench_knm_go",
                write_go_source,
                ns,
                workspace,
            )
        )

    summary = {
        "schema": "scpn_remote_knm_machine_benchmark_v1",
        "generated_at": generated_at,
        "label": label,
        "parameters": {
            "iterations": ns.iterations,
            "sizes": ns.sizes,
            "k0": ns.k0,
            "alpha": ns.alpha,
        },
        "machine": machine_metadata(),
        "rows": rows,
    }

    json_path = ns.output_dir / f"remote_knm_benchmark_{label}_{generated_at[:10]}.json"
    csv_path = ns.output_dir / f"remote_knm_benchmark_{label}_{generated_at[:10]}.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "label",
            "hostname",
            "language",
            "n",
            "status",
            "iterations",
            "median_ms",
            "mean_ms",
            "min_ms",
            "max_ms",
            "checksum",
            "reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": label,
                    "hostname": summary["machine"]["hostname"],
                    **{
                        field: row.get(field)
                        for field in fieldnames
                        if field not in {"label", "hostname"}
                    },
                }
            )

    summary["artefacts"] = {
        "json": str(json_path),
        "json_sha256": sha256_file(json_path),
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["artefacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
