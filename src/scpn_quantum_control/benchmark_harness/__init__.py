# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- public benchmark harness facade
"""Public open-data and classical-validation benchmark harness.

This facade exposes community-facing benchmark entry points without requiring
users to know the internal DLA-parity package layout. The first S5 benchmark is
the published Phase 1 DLA-parity raw-count dataset plus the noiseless classical
parity-conservation reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from scpn_quantum_control.benchmark_harness.registry import (
    BenchmarkFamily,
    benchmark_registry_payload,
    list_benchmark_families,
)
from scpn_quantum_control.dla_parity import (
    ClassicalLeakageReference,
    DlaParityDataset,
    FullHarnessResult,
    ReproductionResult,
    ReproductionTolerance,
    available_baselines,
    compute_classical_leakage_reference,
    load_dla_parity_dataset,
    run_full_harness,
)


def load_phase1_dataset(
    *,
    data_dir: Path | str | None = None,
    verify_integrity: bool = False,
) -> DlaParityDataset:
    """Load the published Phase 1 DLA-parity raw-count dataset."""
    return load_dla_parity_dataset(data_dir=data_dir, verify_integrity=verify_integrity)


def reproduce_phase1_statistics(
    *,
    data_dir: Path | str | None = None,
    verify_integrity: bool = False,
    published_summary: Path | str | None = None,
    tolerance: ReproductionTolerance | None = None,
) -> ReproductionResult:
    """Recompute and verify the published Phase 1 DLA-parity statistics."""
    result = run_full_harness(
        data_dir=data_dir,
        verify_integrity=verify_integrity,
        published_summary=published_summary,
        tolerance=tolerance,
        baselines_backend="numpy",
    )
    return result.reproduction


def run_phase1_benchmark(
    *,
    data_dir: Path | str | None = None,
    verify_integrity: bool = False,
    published_summary: Path | str | None = None,
    baselines_backend: Literal["auto", "numpy", "qutip"] = "auto",
    tolerance: ReproductionTolerance | None = None,
) -> FullHarnessResult:
    """Run Phase 1 raw-data reproduction and classical-baseline validation."""
    return run_full_harness(
        data_dir=data_dir,
        verify_integrity=verify_integrity,
        published_summary=published_summary,
        baselines_backend=baselines_backend,
        tolerance=tolerance,
    )


__all__ = [
    "ClassicalLeakageReference",
    "DlaParityDataset",
    "FullHarnessResult",
    "ReproductionResult",
    "ReproductionTolerance",
    "BenchmarkFamily",
    "available_baselines",
    "benchmark_registry_payload",
    "compute_classical_leakage_reference",
    "list_benchmark_families",
    "load_phase1_dataset",
    "reproduce_phase1_statistics",
    "run_phase1_benchmark",
]
