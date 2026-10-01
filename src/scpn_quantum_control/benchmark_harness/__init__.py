# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — benchmark harness package exports
# scpn-quantum-control -- public benchmark harness facade
"""Public open-data and classical-validation benchmark harness.

This facade exposes community-facing benchmark entry points without requiring
users to know the internal DLA-parity package layout. The first S5 benchmark is
the published Phase 1 DLA-parity raw-count dataset plus the noiseless classical
parity-conservation reference.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scpn_quantum_control.benchmark_harness.registry import (
        BenchmarkFamily,
        benchmark_registry_payload,
        list_benchmark_families,
    )
    from scpn_quantum_control.benchmark_harness.synchronisation import (
        RESULT_SCHEMA,
        SynchronisationBenchmarkInstance,
        list_synchronisation_benchmarks,
        synchronisation_benchmark_registry_payload,
    )
    from scpn_quantum_control.benchmark_harness.synchronisation_compare import (
        ObservableComparison,
        compare_default_artifacts,
        compare_files,
        compare_payloads,
    )
    from scpn_quantum_control.benchmark_harness.synchronisation_runner import (
        BenchmarkResultRow,
        ObservableRow,
        run_kuramoto_chain_n8_decay_omega,
        run_kuramoto_ring_n4_linear_omega,
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

from pathlib import Path
from typing import Literal


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


_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "BenchmarkFamily": ("scpn_quantum_control.benchmark_harness.registry", "BenchmarkFamily"),
    "benchmark_registry_payload": (
        "scpn_quantum_control.benchmark_harness.registry",
        "benchmark_registry_payload",
    ),
    "list_benchmark_families": (
        "scpn_quantum_control.benchmark_harness.registry",
        "list_benchmark_families",
    ),
    "RESULT_SCHEMA": ("scpn_quantum_control.benchmark_harness.synchronisation", "RESULT_SCHEMA"),
    "SynchronisationBenchmarkInstance": (
        "scpn_quantum_control.benchmark_harness.synchronisation",
        "SynchronisationBenchmarkInstance",
    ),
    "list_synchronisation_benchmarks": (
        "scpn_quantum_control.benchmark_harness.synchronisation",
        "list_synchronisation_benchmarks",
    ),
    "synchronisation_benchmark_registry_payload": (
        "scpn_quantum_control.benchmark_harness.synchronisation",
        "synchronisation_benchmark_registry_payload",
    ),
    "ObservableComparison": (
        "scpn_quantum_control.benchmark_harness.synchronisation_compare",
        "ObservableComparison",
    ),
    "compare_default_artifacts": (
        "scpn_quantum_control.benchmark_harness.synchronisation_compare",
        "compare_default_artifacts",
    ),
    "compare_files": (
        "scpn_quantum_control.benchmark_harness.synchronisation_compare",
        "compare_files",
    ),
    "compare_payloads": (
        "scpn_quantum_control.benchmark_harness.synchronisation_compare",
        "compare_payloads",
    ),
    "BenchmarkResultRow": (
        "scpn_quantum_control.benchmark_harness.synchronisation_runner",
        "BenchmarkResultRow",
    ),
    "ObservableRow": (
        "scpn_quantum_control.benchmark_harness.synchronisation_runner",
        "ObservableRow",
    ),
    "run_kuramoto_chain_n8_decay_omega": (
        "scpn_quantum_control.benchmark_harness.synchronisation_runner",
        "run_kuramoto_chain_n8_decay_omega",
    ),
    "run_kuramoto_ring_n4_linear_omega": (
        "scpn_quantum_control.benchmark_harness.synchronisation_runner",
        "run_kuramoto_ring_n4_linear_omega",
    ),
    "ClassicalLeakageReference": ("scpn_quantum_control.dla_parity", "ClassicalLeakageReference"),
    "DlaParityDataset": ("scpn_quantum_control.dla_parity", "DlaParityDataset"),
    "FullHarnessResult": ("scpn_quantum_control.dla_parity", "FullHarnessResult"),
    "ReproductionResult": ("scpn_quantum_control.dla_parity", "ReproductionResult"),
    "ReproductionTolerance": ("scpn_quantum_control.dla_parity", "ReproductionTolerance"),
    "available_baselines": ("scpn_quantum_control.dla_parity", "available_baselines"),
    "compute_classical_leakage_reference": (
        "scpn_quantum_control.dla_parity",
        "compute_classical_leakage_reference",
    ),
    "load_dla_parity_dataset": ("scpn_quantum_control.dla_parity", "load_dla_parity_dataset"),
    "run_full_harness": ("scpn_quantum_control.dla_parity", "run_full_harness"),
}

_INLINE_EXPORTS = {
    "load_phase1_dataset": load_phase1_dataset,
    "reproduce_phase1_statistics": reproduce_phase1_statistics,
    "run_phase1_benchmark": run_phase1_benchmark,
}
del globals()["load_phase1_dataset"]
del globals()["reproduce_phase1_statistics"]
del globals()["run_phase1_benchmark"]
_INLINE_DEPENDENCIES = (
    "BenchmarkFamily",
    "benchmark_registry_payload",
    "list_benchmark_families",
    "RESULT_SCHEMA",
    "SynchronisationBenchmarkInstance",
    "list_synchronisation_benchmarks",
    "synchronisation_benchmark_registry_payload",
    "ObservableComparison",
    "compare_default_artifacts",
    "compare_files",
    "compare_payloads",
    "BenchmarkResultRow",
    "ObservableRow",
    "run_kuramoto_chain_n8_decay_omega",
    "run_kuramoto_ring_n4_linear_omega",
    "ClassicalLeakageReference",
    "DlaParityDataset",
    "FullHarnessResult",
    "ReproductionResult",
    "ReproductionTolerance",
    "available_baselines",
    "compute_classical_leakage_reference",
    "load_dla_parity_dataset",
    "run_full_harness",
)


def __getattr__(name: str) -> Any:
    """Resolve and cache a public export from its original owning module.

    Parameters
    ----------
    name
        Public export requested through this package.

    Returns
    -------
    Any
        Original object, including module-valued exports.

    Raises
    ------
    AttributeError
        If the name is undeclared or the original module lacks its attribute.
    ImportError
        If the owning module cannot be imported.

    """
    if name in _INLINE_EXPORTS:
        for dependency in _INLINE_DEPENDENCIES:
            __getattr__(dependency)
        globals().update(_INLINE_EXPORTS)
        return _INLINE_EXPORTS[name]
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    origin = import_module(target[0])
    value = origin if target[1] is None else getattr(origin, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List cached and deferred names for inspection tools.

    Returns
    -------
    list[str]
        Sorted package namespace and declared lazy export names.

    """
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS) | set(_INLINE_EXPORTS))


__all__ = [
    "ClassicalLeakageReference",
    "DlaParityDataset",
    "FullHarnessResult",
    "ReproductionResult",
    "ReproductionTolerance",
    "BenchmarkFamily",
    "BenchmarkResultRow",
    "RESULT_SCHEMA",
    "ObservableComparison",
    "ObservableRow",
    "SynchronisationBenchmarkInstance",
    "available_baselines",
    "benchmark_registry_payload",
    "compare_default_artifacts",
    "compare_files",
    "compare_payloads",
    "compute_classical_leakage_reference",
    "list_benchmark_families",
    "list_synchronisation_benchmarks",
    "load_phase1_dataset",
    "reproduce_phase1_statistics",
    "run_kuramoto_chain_n8_decay_omega",
    "run_kuramoto_ring_n4_linear_omega",
    "run_phase1_benchmark",
    "synchronisation_benchmark_registry_payload",
]
