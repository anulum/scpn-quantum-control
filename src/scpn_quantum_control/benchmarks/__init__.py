# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Advantage Benchmarks
"""Quantum advantage and scaling benchmarks."""

from .classical_baselines import (
    ClassicalBaselineRun,
    available_baselines,
    mps_tebd_baseline,
    qutip_lindblad_baseline,
    run_documented_classical_baselines,
    scipy_ode_baseline,
)
from .differentiable_external_comparison import (
    REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS,
    ExternalComparisonArtifact,
    ExternalComparisonRow,
    IdenticalCircuitGradientComparisonArtifact,
    IdenticalCircuitGradientComparisonRow,
    external_comparison_failure_mode_rows,
    run_differentiable_external_comparison_suite,
    run_identical_circuit_gradient_comparison_suite,
    write_differentiable_external_comparison,
    write_identical_circuit_gradient_comparison,
)
from .differentiable_hardening_gate import (
    DifferentiableBenchmarkClassificationCase,
    DifferentiableHardeningGateCheck,
    DifferentiableHardeningSliceGateResult,
    run_differentiable_hardening_slice_gate,
)
from .differentiable_programming import (
    DifferentiableProgrammingBenchmarkResult,
    DifferentiableProgrammingExternalReferenceResult,
    QuantumGradientBenchmarkResult,
    run_differentiable_programming_benchmark_suite,
    run_differentiable_programming_external_reference_suite,
    run_quantum_gradient_benchmark_suite,
)
from .gpu_baseline import GPUBaselineResult, gpu_baseline_comparison
from .isolated_host_readiness import (
    HostReadiness,
    assess_host_readiness,
    capture_host_readiness,
)
from .mps_baseline import MPSBaselineResult, mps_baseline_comparison
from .quantum_advantage import (
    AdvantageResult,
    classical_benchmark,
    estimate_crossover,
    quantum_benchmark,
    run_scaling_benchmark,
)

__all__ = [
    "ClassicalBaselineRun",
    "available_baselines",
    "mps_tebd_baseline",
    "qutip_lindblad_baseline",
    "run_documented_classical_baselines",
    "scipy_ode_baseline",
    "DifferentiableProgrammingBenchmarkResult",
    "DifferentiableProgrammingExternalReferenceResult",
    "QuantumGradientBenchmarkResult",
    "ExternalComparisonArtifact",
    "ExternalComparisonRow",
    "IdenticalCircuitGradientComparisonArtifact",
    "IdenticalCircuitGradientComparisonRow",
    "REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS",
    "external_comparison_failure_mode_rows",
    "run_differentiable_external_comparison_suite",
    "run_identical_circuit_gradient_comparison_suite",
    "write_differentiable_external_comparison",
    "write_identical_circuit_gradient_comparison",
    "DifferentiableBenchmarkClassificationCase",
    "DifferentiableHardeningGateCheck",
    "DifferentiableHardeningSliceGateResult",
    "run_differentiable_hardening_slice_gate",
    "run_differentiable_programming_benchmark_suite",
    "run_differentiable_programming_external_reference_suite",
    "run_quantum_gradient_benchmark_suite",
    "HostReadiness",
    "assess_host_readiness",
    "capture_host_readiness",
    "GPUBaselineResult",
    "gpu_baseline_comparison",
    "MPSBaselineResult",
    "mps_baseline_comparison",
    "AdvantageResult",
    "classical_benchmark",
    "estimate_crossover",
    "quantum_benchmark",
    "run_scaling_benchmark",
]
