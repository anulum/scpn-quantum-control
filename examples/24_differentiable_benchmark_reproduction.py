# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Reproduce differentiable benchmark evidence classification locally."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scpn_quantum_control.benchmarks.differentiable_evidence import (
    BenchmarkIsolationMetadata,
    write_differentiable_benchmark_evidence_bundle,
)
from scpn_quantum_control.benchmarks.differentiable_external_comparison import (
    run_differentiable_external_comparison_suite,
    write_differentiable_external_comparison,
)


def main() -> None:
    timing_rows = tuple(row.to_dict() for row in run_differentiable_external_comparison_suite())
    failure_classes = sorted(
        {
            row["failure_class"]
            for row in timing_rows
            if row["status"] == "hard_gap" and row["failure_class"] is not None
        }
    )
    metadata = BenchmarkIsolationMetadata.from_ci_environment(
        {},
        command=("python", "examples/24_differentiable_benchmark_reproduction.py"),
        cpu_affinity=None,
        isolation_method=None,
        load_before=None,
        load_after=None,
        governor=None,
        frequency_mhz=None,
        heavy_jobs_running=False,
    )
    with tempfile.TemporaryDirectory(prefix="scpn-qc-diff-bench-") as directory:
        external_artifact = write_differentiable_external_comparison(
            Path(directory) / "external_comparison.json",
            artifact_id="diff-qnode-local-external-comparison-example",
        )
        bundle = write_differentiable_benchmark_evidence_bundle(
            Path(directory),
            metadata=metadata,
            timing_rows=timing_rows,
            artifact_id="diff-qnode-local-reproduction-example",
        )
        payload = json.loads(bundle.raw_json_path.read_text(encoding="utf-8"))

        print("differentiable benchmark reproduction")
        print(f"  external comparison json: {external_artifact.path}")
        print(f"  json: {bundle.raw_json_path}")
        print(f"  csv: {bundle.csv_path}")
        print(f"  markdown: {bundle.markdown_path}")
        print(f"  classification: {payload['metadata']['classification']}")
        print(f"  production eligible: {payload['metadata']['production_eligible']}")
        print(f"  timing rows: {len(payload['timing_rows'])}")
        print(f"  failure classes: {', '.join(failure_classes)}")
        print(f"  gap reason: {payload['metadata']['gap_reason']}")
        print("  provider or qpu execution: false")


if __name__ == "__main__":
    main()
