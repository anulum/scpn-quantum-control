# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Physical Applications
"""Physical system benchmarks and application modules."""

from .cross_domain import CrossDomainResult, run_cross_domain_validation
from .eeg_benchmark import EEGBenchmarkResult, eeg_benchmark
from .fmo_benchmark import FMOBenchmarkResult, fmo_benchmark, fmo_coupling_matrix
from .iter_benchmark import ITERBenchmarkResult, iter_benchmark
from .josephson_array import JosephsonBenchmarkResult, josephson_benchmark
from .power_grid import PowerGridBenchmarkResult, power_grid_benchmark
from .quantum_evs import QuantumEVSResult, quantum_evs_enhance
from .quantum_kernel import QuantumKernelResult, compute_kernel_matrix
from .quantum_reservoir import ReservoirResult, reservoir_features

__all__ = [
    "CrossDomainResult",
    "run_cross_domain_validation",
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
    "QuantumEVSResult",
    "quantum_evs_enhance",
    "QuantumKernelResult",
    "compute_kernel_matrix",
    "ReservoirResult",
    "reservoir_features",
]
