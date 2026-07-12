#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — run compiler isolated benchmark evidence script
# scpn-quantum-control -- compiler isolated benchmark evidence writer
"""Write compiler isolated benchmark evidence for reserved-host promotion gates."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from scpn_quantum_control.benchmarks.compiler_isolated_benchmark_evidence import (
    build_compiler_isolated_benchmark_evidence,
    write_compiler_isolated_benchmark_evidence,
)
from scpn_quantum_control.benchmarks.differentiable_evidence import (
    BenchmarkIsolationMetadata,
    capture_accelerator_metadata,
    capture_host_load,
    infer_heavy_jobs_running,
    read_cpu_frequency_mhz,
    read_cpu_governor,
)
from scpn_quantum_control.compiler import run_native_whole_program_ad_execution_evidence


def main() -> int:
    """Capture native compiler AD evidence and wrap it with isolation metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/differentiable_phase_qnode"))
    parser.add_argument("--stamp", default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--gradient-parity-tolerance", type=float, default=1e-6)
    parser.add_argument("--cpu-affinity", default=os.environ.get("SCPN_BENCH_CPU_AFFINITY"))
    parser.add_argument(
        "--isolation-method",
        default=os.environ.get("SCPN_BENCH_ISOLATION_METHOD"),
    )
    parser.add_argument("--heavy-jobs-running", action="store_true")
    args = parser.parse_args()

    load_before = capture_host_load()
    governor = read_cpu_governor()
    frequency_mhz = read_cpu_frequency_mhz()
    heavy_jobs_running = args.heavy_jobs_running or infer_heavy_jobs_running(load_before)
    native_evidence = run_native_whole_program_ad_execution_evidence(
        artifact_id=f"native-whole-program-ad-execution-{args.stamp}",
        gradient_parity_tolerance=args.gradient_parity_tolerance,
    )
    load_after = capture_host_load()
    metadata = BenchmarkIsolationMetadata.from_ci_environment(
        os.environ,
        command=tuple(sys.argv),
        cpu_affinity=args.cpu_affinity,
        isolation_method=args.isolation_method,
        load_before=load_before,
        load_after=load_after,
        governor=governor,
        frequency_mhz=frequency_mhz,
        heavy_jobs_running=heavy_jobs_running,
        accelerator_metadata=capture_accelerator_metadata(os.environ),
    )
    evidence = build_compiler_isolated_benchmark_evidence(
        native_execution_evidence=native_evidence,
        benchmark_metadata=metadata,
        stamp=args.stamp,
    )
    files = write_compiler_isolated_benchmark_evidence(args.output_dir, evidence)
    print(files.json_path)
    print(files.markdown_path)
    print(f"classification={evidence.classification}")
    print(
        "ready_for_compiler_promotion_attachment="
        f"{evidence.ready_for_compiler_promotion_attachment}"
    )
    print("No provider, hardware, QPU, GPU, or production-performance claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
