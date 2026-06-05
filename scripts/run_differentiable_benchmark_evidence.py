#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable benchmark evidence writer.
"""Write CI-safe differentiable Phase-QNode benchmark evidence artefacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from scpn_quantum_control.benchmarks.differentiable_evidence import (
    BenchmarkIsolationMetadata,
    write_differentiable_benchmark_evidence_bundle,
)
from scpn_quantum_control.benchmarks.differentiable_external_comparison import (
    run_differentiable_external_comparison_suite,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/differentiable_phase_qnode"))
    parser.add_argument("--cpu-affinity", default=os.environ.get("SCPN_BENCH_CPU_AFFINITY"))
    parser.add_argument(
        "--isolation-method", default=os.environ.get("SCPN_BENCH_ISOLATION_METHOD")
    )
    parser.add_argument("--heavy-jobs-running", action="store_true")
    args = parser.parse_args()

    command = ("python", "scripts/run_differentiable_benchmark_evidence.py")
    metadata = BenchmarkIsolationMetadata.from_ci_environment(
        os.environ,
        command=command,
        cpu_affinity=args.cpu_affinity,
        isolation_method=args.isolation_method,
        load_before=None,
        load_after=None,
        governor=None,
        frequency_mhz=None,
        heavy_jobs_running=args.heavy_jobs_running,
    )
    timing_rows = tuple(row.to_dict() for row in run_differentiable_external_comparison_suite())
    bundle = write_differentiable_benchmark_evidence_bundle(
        args.output_dir,
        metadata=metadata,
        timing_rows=timing_rows,
        artifact_id="diff-qnode-ci-evidence-schema-v1",
    )
    print(bundle.raw_json_path)
    print(bundle.csv_path)
    print(bundle.markdown_path)
    print(f"classification={bundle.classification}")
    print("No provider or QPU execution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
