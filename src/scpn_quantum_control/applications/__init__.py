# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Physical Applications
"""Physical system benchmarks and application modules."""

from .app_plugins import (
    ApplicationPluginBenchmark,
    ApplicationPluginRegistry,
    compile_application_problem,
    discover_application_plugins,
    get_application_plugin,
    get_application_plugin_registry,
    load_application_dataset,
    run_application_benchmark_suite,
)
from .cross_domain import CrossDomainResult, run_cross_domain_validation
from .dataset_catalog import (
    ApplicationBenchmarkDescriptor,
    artifact_to_kuramoto_problem,
    get_application_benchmark_descriptor,
    list_application_benchmark_descriptors,
    load_application_benchmark_artifact,
)
from .eeg_benchmark import EEGBenchmarkResult, eeg_benchmark
from .fmo_benchmark import FMOBenchmarkResult, fmo_benchmark, fmo_coupling_matrix
from .iter_benchmark import ITERBenchmarkResult, iter_benchmark
from .josephson_array import JosephsonBenchmarkResult, josephson_benchmark
from .power_grid import PowerGridBenchmarkResult, power_grid_benchmark
from .qrc_baseline import (
    ClassicalESNReadoutResult,
    QRCBaselineComparison,
    classical_esn_feature_matrix,
    classical_esn_ridge_regression,
    compare_quantum_reservoir_to_esn,
)
from .quantum_evs import QuantumEVSResult, quantum_evs_enhance
from .quantum_kernel import QuantumKernelResult, compute_kernel_matrix
from .quantum_reservoir import ReservoirResult, reservoir_features

__all__ = [
    "ApplicationBenchmarkDescriptor",
    "ApplicationPluginBenchmark",
    "ApplicationPluginRegistry",
    "artifact_to_kuramoto_problem",
    "ClassicalESNReadoutResult",
    "classical_esn_feature_matrix",
    "classical_esn_ridge_regression",
    "compile_application_problem",
    "CrossDomainResult",
    "compare_quantum_reservoir_to_esn",
    "discover_application_plugins",
    "EEGBenchmarkResult",
    "eeg_benchmark",
    "FMOBenchmarkResult",
    "fmo_benchmark",
    "fmo_coupling_matrix",
    "ITERBenchmarkResult",
    "iter_benchmark",
    "JosephsonBenchmarkResult",
    "josephson_benchmark",
    "PowerGridBenchmarkResult",
    "power_grid_benchmark",
    "get_application_benchmark_descriptor",
    "get_application_plugin",
    "get_application_plugin_registry",
    "list_application_benchmark_descriptors",
    "load_application_benchmark_artifact",
    "load_application_dataset",
    "QuantumEVSResult",
    "quantum_evs_enhance",
    "QuantumKernelResult",
    "compute_kernel_matrix",
    "QRCBaselineComparison",
    "ReservoirResult",
    "reservoir_features",
    "run_application_benchmark_suite",
    "run_cross_domain_validation",
]
