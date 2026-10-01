# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Advantage Benchmarks
"""Quantum advantage and scaling benchmarks."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .classical_baselines import (
        ClassicalBaselineRun,
        available_baselines,
        mps_tebd_baseline,
        qutip_lindblad_baseline,
        run_documented_classical_baselines,
        scipy_ode_baseline,
    )
    from .compiler_isolated_benchmark_evidence import (
        COMPILER_ISOLATED_BENCHMARK_CLAIM_BOUNDARY,
        COMPILER_ISOLATED_BENCHMARK_EVIDENCE_PREFIX,
        COMPILER_ISOLATED_BENCHMARK_EVIDENCE_SCHEMA,
        COMPILER_ISOLATED_BENCHMARK_SOURCE_IDS,
        COMPILER_ISOLATED_BENCHMARK_SOURCE_PATHS,
        CompilerBenchmarkClassification,
        CompilerIsolatedBenchmarkEvidence,
        CompilerIsolatedBenchmarkEvidenceFiles,
        build_compiler_isolated_benchmark_evidence,
        render_compiler_isolated_benchmark_evidence_markdown,
        write_compiler_isolated_benchmark_evidence,
    )
    from .coupling_recovery_evidence import (
        COUPLING_RECOVERY_EVIDENCE_SCHEMA,
        CouplingRecoveryEvidenceArtifact,
        coupling_recovery_evidence_payload,
        render_coupling_recovery_evidence_markdown,
        write_coupling_recovery_evidence_artifact,
    )
    from .differentiable_catalyst_comparison import (
        CATALYST_UNSUPPORTED_PROVIDER_ROUTES,
        CatalystCompilerWorkflowComparison,
        catalyst_compiler_workflow_comparison,
    )
    from .differentiable_evidence import (
        AcceleratorEvidenceMetadata,
        BenchmarkIsolationMetadata,
    )
    from .differentiable_external_comparison import (
        PERMANENT_EXTERNAL_COMPARISON_BOUNDARIES,
        REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS,
        ComparisonClosureStatus,
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
    from .differentiable_isolated_benchmark_plan import (
        DifferentiableIsolatedBenchmarkPlan,
        DifferentiableIsolatedBenchmarkPlanRow,
        DifferentiableIsolatedBenchmarkPlanValidation,
        render_differentiable_isolated_benchmark_plan_markdown,
        run_differentiable_isolated_benchmark_plan,
        validate_differentiable_isolated_benchmark_plan,
    )
    from .differentiable_optimizer_convergence import (
        GROUND_STATE_OPTIMIZER_CONVERGENCE_SCHEMA,
        GroundStateOptimizerConvergenceArtifact,
        ground_state_optimizer_convergence_payload,
        render_ground_state_optimizer_convergence_markdown,
        write_ground_state_optimizer_convergence_artifact,
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
    from .kuramoto_competitive_benchmark import (
        CompetitorRow,
        KuramotoCompetitiveComparison,
        KuramotoProblem,
        build_default_problem,
        default_julia_runner,
        run_kuramoto_competitive_comparison,
    )
    from .mps_baseline import MPSBaselineResult, mps_baseline_comparison
    from .open_system_objective_evidence import (
        OPEN_SYSTEM_OBJECTIVE_EVIDENCE_SCHEMA,
        OpenSystemObjectiveEvidenceArtifact,
        open_system_objective_evidence_payload,
        render_open_system_objective_evidence_markdown,
        write_open_system_objective_evidence_artifact,
    )
    from .quantum_advantage import (
        AdvantageResult,
        classical_benchmark,
        estimate_crossover,
        quantum_benchmark,
        run_scaling_benchmark,
    )
    from .reproducible_comparison import (
        ComparisonMethodRow,
        ReproducibleKuramotoComparison,
        run_reproducible_kuramoto_comparison,
    )
    from .sync_witness_evidence import (
        SYNC_WITNESS_EVIDENCE_SCHEMA,
        SyncWitnessEvidenceArtifact,
        render_sync_witness_evidence_markdown,
        sync_witness_evidence_payload,
        write_sync_witness_evidence_artifact,
    )
    from .tn_mps_baseline_design import (
        TN_MPS_BASELINE_DESIGN_CLAIM_BOUNDARY,
        TN_MPS_BASELINE_DESIGN_SCHEMA,
        TNBaselineAdapter,
        TNBaselineDesign,
        TNBaselineSizePlan,
        build_tn_mps_baseline_design,
        render_tn_mps_baseline_design_markdown,
    )
    from .tn_mps_crossover_admission import (
        TN_MPS_CROSSOVER_ADMISSION_SCHEMA,
        TN_MPS_CROSSOVER_CLAIM_BOUNDARY,
        TN_MPS_CROSSOVER_PROTOCOL_ID,
        TN_MPS_CROSSOVER_REQUIRED_FIELDS,
        TNMPSCrossoverAdmissionReport,
        TNMPSCrossoverGate,
        TNMPSCrossoverRowSchema,
        TNMPSCrossoverRowValidation,
        build_tn_mps_crossover_admission,
        render_tn_mps_crossover_admission_markdown,
        validate_tn_mps_crossover_rows,
    )

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ClassicalBaselineRun": (
        "scpn_quantum_control.benchmarks.classical_baselines",
        "ClassicalBaselineRun",
    ),
    "available_baselines": (
        "scpn_quantum_control.benchmarks.classical_baselines",
        "available_baselines",
    ),
    "mps_tebd_baseline": (
        "scpn_quantum_control.benchmarks.classical_baselines",
        "mps_tebd_baseline",
    ),
    "qutip_lindblad_baseline": (
        "scpn_quantum_control.benchmarks.classical_baselines",
        "qutip_lindblad_baseline",
    ),
    "run_documented_classical_baselines": (
        "scpn_quantum_control.benchmarks.classical_baselines",
        "run_documented_classical_baselines",
    ),
    "scipy_ode_baseline": (
        "scpn_quantum_control.benchmarks.classical_baselines",
        "scipy_ode_baseline",
    ),
    "COMPILER_ISOLATED_BENCHMARK_CLAIM_BOUNDARY": (
        "scpn_quantum_control.benchmarks.compiler_isolated_benchmark_evidence",
        "COMPILER_ISOLATED_BENCHMARK_CLAIM_BOUNDARY",
    ),
    "COMPILER_ISOLATED_BENCHMARK_EVIDENCE_PREFIX": (
        "scpn_quantum_control.benchmarks.compiler_isolated_benchmark_evidence",
        "COMPILER_ISOLATED_BENCHMARK_EVIDENCE_PREFIX",
    ),
    "COMPILER_ISOLATED_BENCHMARK_EVIDENCE_SCHEMA": (
        "scpn_quantum_control.benchmarks.compiler_isolated_benchmark_evidence",
        "COMPILER_ISOLATED_BENCHMARK_EVIDENCE_SCHEMA",
    ),
    "COMPILER_ISOLATED_BENCHMARK_SOURCE_IDS": (
        "scpn_quantum_control.benchmarks.compiler_isolated_benchmark_evidence",
        "COMPILER_ISOLATED_BENCHMARK_SOURCE_IDS",
    ),
    "COMPILER_ISOLATED_BENCHMARK_SOURCE_PATHS": (
        "scpn_quantum_control.benchmarks.compiler_isolated_benchmark_evidence",
        "COMPILER_ISOLATED_BENCHMARK_SOURCE_PATHS",
    ),
    "CompilerBenchmarkClassification": (
        "scpn_quantum_control.benchmarks.compiler_isolated_benchmark_evidence",
        "CompilerBenchmarkClassification",
    ),
    "CompilerIsolatedBenchmarkEvidence": (
        "scpn_quantum_control.benchmarks.compiler_isolated_benchmark_evidence",
        "CompilerIsolatedBenchmarkEvidence",
    ),
    "CompilerIsolatedBenchmarkEvidenceFiles": (
        "scpn_quantum_control.benchmarks.compiler_isolated_benchmark_evidence",
        "CompilerIsolatedBenchmarkEvidenceFiles",
    ),
    "build_compiler_isolated_benchmark_evidence": (
        "scpn_quantum_control.benchmarks.compiler_isolated_benchmark_evidence",
        "build_compiler_isolated_benchmark_evidence",
    ),
    "render_compiler_isolated_benchmark_evidence_markdown": (
        "scpn_quantum_control.benchmarks.compiler_isolated_benchmark_evidence",
        "render_compiler_isolated_benchmark_evidence_markdown",
    ),
    "write_compiler_isolated_benchmark_evidence": (
        "scpn_quantum_control.benchmarks.compiler_isolated_benchmark_evidence",
        "write_compiler_isolated_benchmark_evidence",
    ),
    "COUPLING_RECOVERY_EVIDENCE_SCHEMA": (
        "scpn_quantum_control.benchmarks.coupling_recovery_evidence",
        "COUPLING_RECOVERY_EVIDENCE_SCHEMA",
    ),
    "CouplingRecoveryEvidenceArtifact": (
        "scpn_quantum_control.benchmarks.coupling_recovery_evidence",
        "CouplingRecoveryEvidenceArtifact",
    ),
    "coupling_recovery_evidence_payload": (
        "scpn_quantum_control.benchmarks.coupling_recovery_evidence",
        "coupling_recovery_evidence_payload",
    ),
    "render_coupling_recovery_evidence_markdown": (
        "scpn_quantum_control.benchmarks.coupling_recovery_evidence",
        "render_coupling_recovery_evidence_markdown",
    ),
    "write_coupling_recovery_evidence_artifact": (
        "scpn_quantum_control.benchmarks.coupling_recovery_evidence",
        "write_coupling_recovery_evidence_artifact",
    ),
    "CATALYST_UNSUPPORTED_PROVIDER_ROUTES": (
        "scpn_quantum_control.benchmarks.differentiable_catalyst_comparison",
        "CATALYST_UNSUPPORTED_PROVIDER_ROUTES",
    ),
    "CatalystCompilerWorkflowComparison": (
        "scpn_quantum_control.benchmarks.differentiable_catalyst_comparison",
        "CatalystCompilerWorkflowComparison",
    ),
    "catalyst_compiler_workflow_comparison": (
        "scpn_quantum_control.benchmarks.differentiable_catalyst_comparison",
        "catalyst_compiler_workflow_comparison",
    ),
    "AcceleratorEvidenceMetadata": (
        "scpn_quantum_control.benchmarks.differentiable_evidence",
        "AcceleratorEvidenceMetadata",
    ),
    "BenchmarkIsolationMetadata": (
        "scpn_quantum_control.benchmarks.differentiable_evidence",
        "BenchmarkIsolationMetadata",
    ),
    "PERMANENT_EXTERNAL_COMPARISON_BOUNDARIES": (
        "scpn_quantum_control.benchmarks.differentiable_external_comparison",
        "PERMANENT_EXTERNAL_COMPARISON_BOUNDARIES",
    ),
    "REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS": (
        "scpn_quantum_control.benchmarks.differentiable_external_comparison",
        "REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS",
    ),
    "ComparisonClosureStatus": (
        "scpn_quantum_control.benchmarks.differentiable_external_comparison",
        "ComparisonClosureStatus",
    ),
    "ExternalComparisonArtifact": (
        "scpn_quantum_control.benchmarks.differentiable_external_comparison",
        "ExternalComparisonArtifact",
    ),
    "ExternalComparisonRow": (
        "scpn_quantum_control.benchmarks.differentiable_external_comparison",
        "ExternalComparisonRow",
    ),
    "IdenticalCircuitGradientComparisonArtifact": (
        "scpn_quantum_control.benchmarks.differentiable_external_comparison",
        "IdenticalCircuitGradientComparisonArtifact",
    ),
    "IdenticalCircuitGradientComparisonRow": (
        "scpn_quantum_control.benchmarks.differentiable_external_comparison",
        "IdenticalCircuitGradientComparisonRow",
    ),
    "external_comparison_failure_mode_rows": (
        "scpn_quantum_control.benchmarks.differentiable_external_comparison",
        "external_comparison_failure_mode_rows",
    ),
    "run_differentiable_external_comparison_suite": (
        "scpn_quantum_control.benchmarks.differentiable_external_comparison",
        "run_differentiable_external_comparison_suite",
    ),
    "run_identical_circuit_gradient_comparison_suite": (
        "scpn_quantum_control.benchmarks.differentiable_external_comparison",
        "run_identical_circuit_gradient_comparison_suite",
    ),
    "write_differentiable_external_comparison": (
        "scpn_quantum_control.benchmarks.differentiable_external_comparison",
        "write_differentiable_external_comparison",
    ),
    "write_identical_circuit_gradient_comparison": (
        "scpn_quantum_control.benchmarks.differentiable_external_comparison",
        "write_identical_circuit_gradient_comparison",
    ),
    "DifferentiableBenchmarkClassificationCase": (
        "scpn_quantum_control.benchmarks.differentiable_hardening_gate",
        "DifferentiableBenchmarkClassificationCase",
    ),
    "DifferentiableHardeningGateCheck": (
        "scpn_quantum_control.benchmarks.differentiable_hardening_gate",
        "DifferentiableHardeningGateCheck",
    ),
    "DifferentiableHardeningSliceGateResult": (
        "scpn_quantum_control.benchmarks.differentiable_hardening_gate",
        "DifferentiableHardeningSliceGateResult",
    ),
    "run_differentiable_hardening_slice_gate": (
        "scpn_quantum_control.benchmarks.differentiable_hardening_gate",
        "run_differentiable_hardening_slice_gate",
    ),
    "DifferentiableIsolatedBenchmarkPlan": (
        "scpn_quantum_control.benchmarks.differentiable_isolated_benchmark_plan",
        "DifferentiableIsolatedBenchmarkPlan",
    ),
    "DifferentiableIsolatedBenchmarkPlanRow": (
        "scpn_quantum_control.benchmarks.differentiable_isolated_benchmark_plan",
        "DifferentiableIsolatedBenchmarkPlanRow",
    ),
    "DifferentiableIsolatedBenchmarkPlanValidation": (
        "scpn_quantum_control.benchmarks.differentiable_isolated_benchmark_plan",
        "DifferentiableIsolatedBenchmarkPlanValidation",
    ),
    "render_differentiable_isolated_benchmark_plan_markdown": (
        "scpn_quantum_control.benchmarks.differentiable_isolated_benchmark_plan",
        "render_differentiable_isolated_benchmark_plan_markdown",
    ),
    "run_differentiable_isolated_benchmark_plan": (
        "scpn_quantum_control.benchmarks.differentiable_isolated_benchmark_plan",
        "run_differentiable_isolated_benchmark_plan",
    ),
    "validate_differentiable_isolated_benchmark_plan": (
        "scpn_quantum_control.benchmarks.differentiable_isolated_benchmark_plan",
        "validate_differentiable_isolated_benchmark_plan",
    ),
    "GROUND_STATE_OPTIMIZER_CONVERGENCE_SCHEMA": (
        "scpn_quantum_control.benchmarks.differentiable_optimizer_convergence",
        "GROUND_STATE_OPTIMIZER_CONVERGENCE_SCHEMA",
    ),
    "GroundStateOptimizerConvergenceArtifact": (
        "scpn_quantum_control.benchmarks.differentiable_optimizer_convergence",
        "GroundStateOptimizerConvergenceArtifact",
    ),
    "ground_state_optimizer_convergence_payload": (
        "scpn_quantum_control.benchmarks.differentiable_optimizer_convergence",
        "ground_state_optimizer_convergence_payload",
    ),
    "render_ground_state_optimizer_convergence_markdown": (
        "scpn_quantum_control.benchmarks.differentiable_optimizer_convergence",
        "render_ground_state_optimizer_convergence_markdown",
    ),
    "write_ground_state_optimizer_convergence_artifact": (
        "scpn_quantum_control.benchmarks.differentiable_optimizer_convergence",
        "write_ground_state_optimizer_convergence_artifact",
    ),
    "DifferentiableProgrammingBenchmarkResult": (
        "scpn_quantum_control.benchmarks.differentiable_programming",
        "DifferentiableProgrammingBenchmarkResult",
    ),
    "DifferentiableProgrammingExternalReferenceResult": (
        "scpn_quantum_control.benchmarks.differentiable_programming",
        "DifferentiableProgrammingExternalReferenceResult",
    ),
    "QuantumGradientBenchmarkResult": (
        "scpn_quantum_control.benchmarks.differentiable_programming",
        "QuantumGradientBenchmarkResult",
    ),
    "run_differentiable_programming_benchmark_suite": (
        "scpn_quantum_control.benchmarks.differentiable_programming",
        "run_differentiable_programming_benchmark_suite",
    ),
    "run_differentiable_programming_external_reference_suite": (
        "scpn_quantum_control.benchmarks.differentiable_programming",
        "run_differentiable_programming_external_reference_suite",
    ),
    "run_quantum_gradient_benchmark_suite": (
        "scpn_quantum_control.benchmarks.differentiable_programming",
        "run_quantum_gradient_benchmark_suite",
    ),
    "GPUBaselineResult": ("scpn_quantum_control.benchmarks.gpu_baseline", "GPUBaselineResult"),
    "gpu_baseline_comparison": (
        "scpn_quantum_control.benchmarks.gpu_baseline",
        "gpu_baseline_comparison",
    ),
    "HostReadiness": ("scpn_quantum_control.benchmarks.isolated_host_readiness", "HostReadiness"),
    "assess_host_readiness": (
        "scpn_quantum_control.benchmarks.isolated_host_readiness",
        "assess_host_readiness",
    ),
    "capture_host_readiness": (
        "scpn_quantum_control.benchmarks.isolated_host_readiness",
        "capture_host_readiness",
    ),
    "CompetitorRow": (
        "scpn_quantum_control.benchmarks.kuramoto_competitive_benchmark",
        "CompetitorRow",
    ),
    "KuramotoCompetitiveComparison": (
        "scpn_quantum_control.benchmarks.kuramoto_competitive_benchmark",
        "KuramotoCompetitiveComparison",
    ),
    "KuramotoProblem": (
        "scpn_quantum_control.benchmarks.kuramoto_competitive_benchmark",
        "KuramotoProblem",
    ),
    "build_default_problem": (
        "scpn_quantum_control.benchmarks.kuramoto_competitive_benchmark",
        "build_default_problem",
    ),
    "default_julia_runner": (
        "scpn_quantum_control.benchmarks.kuramoto_competitive_benchmark",
        "default_julia_runner",
    ),
    "run_kuramoto_competitive_comparison": (
        "scpn_quantum_control.benchmarks.kuramoto_competitive_benchmark",
        "run_kuramoto_competitive_comparison",
    ),
    "MPSBaselineResult": ("scpn_quantum_control.benchmarks.mps_baseline", "MPSBaselineResult"),
    "mps_baseline_comparison": (
        "scpn_quantum_control.benchmarks.mps_baseline",
        "mps_baseline_comparison",
    ),
    "OPEN_SYSTEM_OBJECTIVE_EVIDENCE_SCHEMA": (
        "scpn_quantum_control.benchmarks.open_system_objective_evidence",
        "OPEN_SYSTEM_OBJECTIVE_EVIDENCE_SCHEMA",
    ),
    "OpenSystemObjectiveEvidenceArtifact": (
        "scpn_quantum_control.benchmarks.open_system_objective_evidence",
        "OpenSystemObjectiveEvidenceArtifact",
    ),
    "open_system_objective_evidence_payload": (
        "scpn_quantum_control.benchmarks.open_system_objective_evidence",
        "open_system_objective_evidence_payload",
    ),
    "render_open_system_objective_evidence_markdown": (
        "scpn_quantum_control.benchmarks.open_system_objective_evidence",
        "render_open_system_objective_evidence_markdown",
    ),
    "write_open_system_objective_evidence_artifact": (
        "scpn_quantum_control.benchmarks.open_system_objective_evidence",
        "write_open_system_objective_evidence_artifact",
    ),
    "AdvantageResult": ("scpn_quantum_control.benchmarks.quantum_advantage", "AdvantageResult"),
    "classical_benchmark": (
        "scpn_quantum_control.benchmarks.quantum_advantage",
        "classical_benchmark",
    ),
    "estimate_crossover": (
        "scpn_quantum_control.benchmarks.quantum_advantage",
        "estimate_crossover",
    ),
    "quantum_benchmark": (
        "scpn_quantum_control.benchmarks.quantum_advantage",
        "quantum_benchmark",
    ),
    "run_scaling_benchmark": (
        "scpn_quantum_control.benchmarks.quantum_advantage",
        "run_scaling_benchmark",
    ),
    "ComparisonMethodRow": (
        "scpn_quantum_control.benchmarks.reproducible_comparison",
        "ComparisonMethodRow",
    ),
    "ReproducibleKuramotoComparison": (
        "scpn_quantum_control.benchmarks.reproducible_comparison",
        "ReproducibleKuramotoComparison",
    ),
    "run_reproducible_kuramoto_comparison": (
        "scpn_quantum_control.benchmarks.reproducible_comparison",
        "run_reproducible_kuramoto_comparison",
    ),
    "SYNC_WITNESS_EVIDENCE_SCHEMA": (
        "scpn_quantum_control.benchmarks.sync_witness_evidence",
        "SYNC_WITNESS_EVIDENCE_SCHEMA",
    ),
    "SyncWitnessEvidenceArtifact": (
        "scpn_quantum_control.benchmarks.sync_witness_evidence",
        "SyncWitnessEvidenceArtifact",
    ),
    "render_sync_witness_evidence_markdown": (
        "scpn_quantum_control.benchmarks.sync_witness_evidence",
        "render_sync_witness_evidence_markdown",
    ),
    "sync_witness_evidence_payload": (
        "scpn_quantum_control.benchmarks.sync_witness_evidence",
        "sync_witness_evidence_payload",
    ),
    "write_sync_witness_evidence_artifact": (
        "scpn_quantum_control.benchmarks.sync_witness_evidence",
        "write_sync_witness_evidence_artifact",
    ),
    "TN_MPS_BASELINE_DESIGN_CLAIM_BOUNDARY": (
        "scpn_quantum_control.benchmarks.tn_mps_baseline_design",
        "TN_MPS_BASELINE_DESIGN_CLAIM_BOUNDARY",
    ),
    "TN_MPS_BASELINE_DESIGN_SCHEMA": (
        "scpn_quantum_control.benchmarks.tn_mps_baseline_design",
        "TN_MPS_BASELINE_DESIGN_SCHEMA",
    ),
    "TNBaselineAdapter": (
        "scpn_quantum_control.benchmarks.tn_mps_baseline_design",
        "TNBaselineAdapter",
    ),
    "TNBaselineDesign": (
        "scpn_quantum_control.benchmarks.tn_mps_baseline_design",
        "TNBaselineDesign",
    ),
    "TNBaselineSizePlan": (
        "scpn_quantum_control.benchmarks.tn_mps_baseline_design",
        "TNBaselineSizePlan",
    ),
    "build_tn_mps_baseline_design": (
        "scpn_quantum_control.benchmarks.tn_mps_baseline_design",
        "build_tn_mps_baseline_design",
    ),
    "render_tn_mps_baseline_design_markdown": (
        "scpn_quantum_control.benchmarks.tn_mps_baseline_design",
        "render_tn_mps_baseline_design_markdown",
    ),
    "TN_MPS_CROSSOVER_ADMISSION_SCHEMA": (
        "scpn_quantum_control.benchmarks.tn_mps_crossover_admission",
        "TN_MPS_CROSSOVER_ADMISSION_SCHEMA",
    ),
    "TN_MPS_CROSSOVER_CLAIM_BOUNDARY": (
        "scpn_quantum_control.benchmarks.tn_mps_crossover_admission",
        "TN_MPS_CROSSOVER_CLAIM_BOUNDARY",
    ),
    "TN_MPS_CROSSOVER_PROTOCOL_ID": (
        "scpn_quantum_control.benchmarks.tn_mps_crossover_admission",
        "TN_MPS_CROSSOVER_PROTOCOL_ID",
    ),
    "TN_MPS_CROSSOVER_REQUIRED_FIELDS": (
        "scpn_quantum_control.benchmarks.tn_mps_crossover_admission",
        "TN_MPS_CROSSOVER_REQUIRED_FIELDS",
    ),
    "TNMPSCrossoverAdmissionReport": (
        "scpn_quantum_control.benchmarks.tn_mps_crossover_admission",
        "TNMPSCrossoverAdmissionReport",
    ),
    "TNMPSCrossoverGate": (
        "scpn_quantum_control.benchmarks.tn_mps_crossover_admission",
        "TNMPSCrossoverGate",
    ),
    "TNMPSCrossoverRowSchema": (
        "scpn_quantum_control.benchmarks.tn_mps_crossover_admission",
        "TNMPSCrossoverRowSchema",
    ),
    "TNMPSCrossoverRowValidation": (
        "scpn_quantum_control.benchmarks.tn_mps_crossover_admission",
        "TNMPSCrossoverRowValidation",
    ),
    "build_tn_mps_crossover_admission": (
        "scpn_quantum_control.benchmarks.tn_mps_crossover_admission",
        "build_tn_mps_crossover_admission",
    ),
    "render_tn_mps_crossover_admission_markdown": (
        "scpn_quantum_control.benchmarks.tn_mps_crossover_admission",
        "render_tn_mps_crossover_admission_markdown",
    ),
    "validate_tn_mps_crossover_rows": (
        "scpn_quantum_control.benchmarks.tn_mps_crossover_admission",
        "validate_tn_mps_crossover_rows",
    ),
}


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
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS))


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
    "CATALYST_UNSUPPORTED_PROVIDER_ROUTES",
    "CatalystCompilerWorkflowComparison",
    "catalyst_compiler_workflow_comparison",
    "COMPILER_ISOLATED_BENCHMARK_CLAIM_BOUNDARY",
    "COMPILER_ISOLATED_BENCHMARK_EVIDENCE_PREFIX",
    "COMPILER_ISOLATED_BENCHMARK_EVIDENCE_SCHEMA",
    "COMPILER_ISOLATED_BENCHMARK_SOURCE_IDS",
    "COMPILER_ISOLATED_BENCHMARK_SOURCE_PATHS",
    "CompilerBenchmarkClassification",
    "CompilerIsolatedBenchmarkEvidence",
    "CompilerIsolatedBenchmarkEvidenceFiles",
    "ComparisonClosureStatus",
    "build_compiler_isolated_benchmark_evidence",
    "render_compiler_isolated_benchmark_evidence_markdown",
    "write_compiler_isolated_benchmark_evidence",
    "COUPLING_RECOVERY_EVIDENCE_SCHEMA",
    "CouplingRecoveryEvidenceArtifact",
    "coupling_recovery_evidence_payload",
    "render_coupling_recovery_evidence_markdown",
    "write_coupling_recovery_evidence_artifact",
    "SYNC_WITNESS_EVIDENCE_SCHEMA",
    "SyncWitnessEvidenceArtifact",
    "render_sync_witness_evidence_markdown",
    "sync_witness_evidence_payload",
    "write_sync_witness_evidence_artifact",
    "AcceleratorEvidenceMetadata",
    "BenchmarkIsolationMetadata",
    "ExternalComparisonArtifact",
    "ExternalComparisonRow",
    "IdenticalCircuitGradientComparisonArtifact",
    "IdenticalCircuitGradientComparisonRow",
    "ComparisonClosureStatus",
    "PERMANENT_EXTERNAL_COMPARISON_BOUNDARIES",
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
    "DifferentiableIsolatedBenchmarkPlan",
    "DifferentiableIsolatedBenchmarkPlanRow",
    "DifferentiableIsolatedBenchmarkPlanValidation",
    "render_differentiable_isolated_benchmark_plan_markdown",
    "run_differentiable_isolated_benchmark_plan",
    "validate_differentiable_isolated_benchmark_plan",
    "GROUND_STATE_OPTIMIZER_CONVERGENCE_SCHEMA",
    "GroundStateOptimizerConvergenceArtifact",
    "ground_state_optimizer_convergence_payload",
    "render_ground_state_optimizer_convergence_markdown",
    "write_ground_state_optimizer_convergence_artifact",
    "OPEN_SYSTEM_OBJECTIVE_EVIDENCE_SCHEMA",
    "OpenSystemObjectiveEvidenceArtifact",
    "open_system_objective_evidence_payload",
    "render_open_system_objective_evidence_markdown",
    "write_open_system_objective_evidence_artifact",
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
    "TN_MPS_BASELINE_DESIGN_CLAIM_BOUNDARY",
    "TN_MPS_BASELINE_DESIGN_SCHEMA",
    "TNBaselineAdapter",
    "TNBaselineDesign",
    "TNBaselineSizePlan",
    "build_tn_mps_baseline_design",
    "render_tn_mps_baseline_design_markdown",
    "TN_MPS_CROSSOVER_CLAIM_BOUNDARY",
    "TN_MPS_CROSSOVER_PROTOCOL_ID",
    "TN_MPS_CROSSOVER_REQUIRED_FIELDS",
    "TN_MPS_CROSSOVER_ADMISSION_SCHEMA",
    "TNMPSCrossoverGate",
    "TNMPSCrossoverRowSchema",
    "TNMPSCrossoverRowValidation",
    "TNMPSCrossoverAdmissionReport",
    "build_tn_mps_crossover_admission",
    "render_tn_mps_crossover_admission_markdown",
    "validate_tn_mps_crossover_rows",
    "AdvantageResult",
    "classical_benchmark",
    "estimate_crossover",
    "quantum_benchmark",
    "run_scaling_benchmark",
    "ComparisonMethodRow",
    "ReproducibleKuramotoComparison",
    "run_reproducible_kuramoto_comparison",
    "CompetitorRow",
    "KuramotoCompetitiveComparison",
    "KuramotoProblem",
    "build_default_problem",
    "default_julia_runner",
    "run_kuramoto_competitive_comparison",
]
