# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable module hardening audit.
"""Module coverage and diagnostic audit for differentiable-programming surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DIFFERENTIABLE_MODULE_PATTERNS = (
    "src/scpn_quantum_control/differentiable*.py",
    "src/scpn_quantum_control/diff.py",
    "src/scpn_quantum_control/program_ad*.py",
    "src/scpn_quantum_control/phase/differentiable*.py",
    "src/scpn_quantum_control/phase/*gradient*.py",
    "src/scpn_quantum_control/phase/*provider*.py",
    "src/scpn_quantum_control/phase/*qnode*.py",
    "src/scpn_quantum_control/phase/*bridge*.py",
    "src/scpn_quantum_control/phase/*compiler*.py",
    "src/scpn_quantum_control/whole_program*.py",
    "src/scpn_quantum_control/benchmarks/differentiable*.py",
)


@dataclass(frozen=True)
class DifferentiableModuleHardeningRecord:
    """Hardening evidence for one differentiable-programming module."""

    module_path: str
    test_paths: tuple[str, ...]
    diagnostic_surfaces: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate that every registry row contains concrete evidence paths."""
        if not self.module_path:
            raise ValueError("module_path must be non-empty")
        if not self.test_paths or any(not path for path in self.test_paths):
            raise ValueError("test_paths must contain non-empty entries")
        if not self.diagnostic_surfaces or any(
            not surface for surface in self.diagnostic_surfaces
        ):
            raise ValueError("diagnostic_surfaces must contain non-empty entries")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready module-hardening row."""
        return {
            "module_path": self.module_path,
            "test_paths": list(self.test_paths),
            "diagnostic_surfaces": list(self.diagnostic_surfaces),
        }


@dataclass(frozen=True)
class DifferentiableModuleHardeningAuditResult:
    """Audit result for the differentiable module hardening registry."""

    passed: bool
    records: tuple[DifferentiableModuleHardeningRecord, ...]
    discovered_module_paths: tuple[str, ...]
    missing_registry_paths: tuple[str, ...]
    stale_registry_paths: tuple[str, ...]
    missing_test_paths: tuple[str, ...]
    errors: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready audit evidence."""
        return {
            "passed": self.passed,
            "records": [record.to_dict() for record in self.records],
            "discovered_module_paths": list(self.discovered_module_paths),
            "missing_registry_paths": list(self.missing_registry_paths),
            "stale_registry_paths": list(self.stale_registry_paths),
            "missing_test_paths": list(self.missing_test_paths),
            "errors": list(self.errors),
            "claim_boundary": self.claim_boundary,
        }


def differentiable_module_hardening_registry() -> tuple[DifferentiableModuleHardeningRecord, ...]:
    """Return the registered differentiable module hardening evidence map."""
    return (
        _record(
            "src/scpn_quantum_control/benchmarks/differentiable_evidence.py",
            ("tests/test_differentiable_benchmark_evidence.py",),
            ("runner isolation classification", "accelerator fallback hard gaps"),
        ),
        _record(
            "src/scpn_quantum_control/benchmarks/differentiable_external_comparison.py",
            ("tests/test_differentiable_external_comparisons.py",),
            ("dependency-missing rows", "correctness-mismatch rows"),
        ),
        _record(
            "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py",
            ("tests/test_differentiable_hardening_gate.py",),
            ("bucket-test rejection", "benchmark-classification smoke cases"),
        ),
        _record(
            "src/scpn_quantum_control/benchmarks/differentiable_programming.py",
            ("tests/test_differentiable_programming_benchmarks.py",),
            ("claim-boundary benchmark rows", "external-reference hard gaps"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable.py",
            ("tests/test_differentiable_package_exports.py",),
            ("facade export compatibility", "diagnostic-only finite differences"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_batch_helpers.py",
            ("tests/test_differentiable_batch_helpers.py",),
            ("parameter-shift sample tensors", "batched tangent/cotangent validation"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_vmap.py",
            ("tests/test_differentiable_vmap.py",),
            ("eager vectorization transform", "nested output stacking boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_elementwise_primitives.py",
            (
                "tests/test_program_ad_elementwise_registry.py",
                "tests/test_program_ad_binary_elementwise_registry.py",
                "tests/test_program_ad_runtime_registry_dispatch.py",
            ),
            ("elementwise direct rules", "elementwise registry dispatch boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_array_indexing.py",
            ("tests/test_program_ad_array_indexing_registry.py",),
            ("static gather direct rules", "scatter-add VJP boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_shape_transforms.py",
            ("tests/test_program_ad_shape_transforms.py",),
            ("static shape direct rules", "shape-transform VJP boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_signal_primitives.py",
            ("tests/test_program_ad_signal_primitives.py",),
            ("static signal direct rules", "signal VJP window boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_cumulative_primitives.py",
            ("tests/test_program_ad_cumulative_primitives.py",),
            ("static cumulative direct rules", "axis-aware cumulative VJP boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_trapezoid_primitives.py",
            ("tests/test_program_ad_trapezoid.py",),
            ("static trapezoid direct rules", "static-grid trapezoid VJP boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_reduction_primitives.py",
            (
                "tests/test_program_ad_reduction_norms.py",
                "tests/test_program_ad_runtime_registry_dispatch.py",
            ),
            ("static reduction direct rules", "reduction registry dispatch boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_stack_block_assembly.py",
            ("tests/test_program_ad_stack_block_assembly.py",),
            ("static stack/block direct rules", "nested assembly VJP boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_broadcast_assembly.py",
            ("tests/test_program_ad_broadcast_assembly.py",),
            ("static broadcast direct rules", "broadcast VJP reduction boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_assembly_primitives.py",
            (
                "tests/test_program_ad_broadcast_assembly.py",
                "tests/test_program_ad_split_assembly.py",
                "tests/test_program_ad_stack_block_assembly.py",
                "tests/test_program_ad_static_array_assembly.py",
                "tests/test_program_ad_triangular_diagonal_assembly.py",
                "tests/test_program_ad_runtime_registry_dispatch.py",
            ),
            ("assembly registry dispatch boundaries", "static assembly contract rules"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_stencil_primitives.py",
            ("tests/test_program_ad_finite_difference_stencils.py",),
            ("static stencil direct rules", "gradient VJP boundary coefficients"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_interpolation_primitives.py",
            ("tests/test_program_ad_interpolation.py",),
            ("static interpolation direct rules", "piecewise VJP boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_linalg_primitives.py",
            (
                "tests/test_program_ad_linalg_core.py",
                "tests/test_program_ad_linalg_direct_rules.py",
                "tests/test_program_ad_linalg_matrix_ops.py",
                "tests/test_program_ad_linalg_registry.py",
                "tests/test_program_ad_linalg_spectral.py",
            ),
            ("static linalg direct rules", "linalg registry dispatch boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_selection_primitives.py",
            (
                "tests/test_program_ad_selection_direct_rules.py",
                "tests/test_program_ad_selection_registry.py",
                "tests/test_program_ad_selection_folds.py",
            ),
            ("selection direct rules", "selection registry dispatch boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_product_primitives.py",
            (
                "tests/test_program_ad_product_contractions.py",
                "tests/test_program_ad_runtime_registry_dispatch.py",
            ),
            ("product direct rules", "product registry dispatch boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_adjoint.py",
            ("tests/test_program_adjoint_replay.py",),
            (
                "reverse-adjoint result records",
                "input-token replay helpers",
                "reverse-adjoint result accessors",
            ),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_adjoint_generation.py",
            (
                "tests/test_program_ad_adjoint_generation.py",
                "tests/test_program_adjoint_replay.py",
            ),
            (
                "reverse-adjoint contribution rules",
                "adjoint generation dispatcher",
                "reverse-replay driver boundaries",
            ),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_alias_analysis.py",
            ("tests/test_program_ad_alias_effects.py",),
            ("alias/effect summaries", "static alias-lattice readiness"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_effect_ir.py",
            ("tests/test_program_ad_effect_ir.py",),
            ("effect-IR record validation", "effect-IR parser contracts"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_registry.py",
            ("tests/test_program_ad_registry.py",),
            ("primitive registry contracts", "dispatch coverage reports"),
        ),
        _record(
            "src/scpn_quantum_control/program_ad_rust_bridge.py",
            ("tests/test_program_ad_rust_bridge.py",),
            ("Rust bridge result parsing", "extension fail-closed boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/whole_program_frontend.py",
            ("tests/test_whole_program_frontend.py",),
            ("static bytecode/source frontend", "unsupported Python-semantics hard gaps"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_api.py",
            ("tests/test_differentiable_api.py",),
            ("unified support reports", "unsupported-route diagnostics"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_benchmark_report.py",
            ("tests/test_differentiable_benchmark_report.py",),
            ("local conformance bundle", "non-isolated benchmark claim boundary"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_canonical_api.py",
            ("tests/test_differentiable_package_exports.py",),
            ("canonical transform exports", "unsupported method rejection"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_claim_ledger.py",
            ("tests/test_differentiable_claim_ledger.py",),
            ("public-language guard", "support-surface alignment errors"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_external_validation.py",
            ("tests/test_differentiable_external_validation.py",),
            ("lockfile checksum drift", "artefact-bundle checksum drift"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_framework_overlay.py",
            (
                "tests/test_differentiable_framework_overlay.py",
                "tests/test_differentiable_framework_ci_workflow.py",
            ),
            ("optional-dependency verification", "CPU-only overlay boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_jax_adapter.py",
            ("tests/test_differentiable_jax.py",),
            ("optional-JAX dependency boundary", "JAX value-and-gradient validation"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_consistency.py",
            ("tests/test_differentiable_consistency.py",),
            (
                "parameter-shift consistency diagnostics",
                "custom derivative consistency diagnostics",
            ),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_custom_derivatives.py",
            ("tests/test_differentiable_custom_derivatives.py",),
            (
                "exact custom JVP wrappers",
                "exact custom VJP wrappers",
                "custom Jacobian materialisation",
            ),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_exact_modes.py",
            ("tests/test_differentiable_exact_modes.py",),
            ("forward-mode dual gradients", "reverse-mode tape gradients"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_fisher.py",
            ("tests/test_differentiable_fisher.py",),
            (
                "matrix-free empirical-Fisher products",
                "least-squares covariance estimation",
            ),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_finite_difference.py",
            ("tests/test_differentiable_finite_difference.py",),
            (
                "finite-difference diagnostics",
                "complex-step diagnostics",
                "Jacobian/JVP/VJP/Hessian/HVP wrappers",
            ),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_gradient_descent.py",
            ("tests/test_differentiable_gradient_descent_optimizer.py",),
            (
                "gradient-descent optimizer step",
                "gradient-descent minimize loop controls",
            ),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_implicit_sensitivity.py",
            ("tests/test_differentiable_implicit_sensitivity.py",),
            ("implicit stationary sensitivity", "fixed-point sensitivity solves"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_levenberg_marquardt.py",
            ("tests/test_differentiable_levenberg_marquardt.py",),
            (
                "Gauss-Newton residual solves",
                "Levenberg-Marquardt trial and damping policy",
            ),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_module_hardening_audit.py",
            ("tests/test_differentiable_module_hardening_audit.py",),
            ("module inventory discovery", "module-specific test enforcement"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_transform_algebra.py",
            ("tests/test_differentiable_transform_algebra.py",),
            (
                "transform-algebra metamorphic identities",
                "fail-closed unsupported transform boundaries",
            ),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_natural_gradient.py",
            (
                "tests/test_differentiable_natural_gradient.py",
                "tests/test_differentiable_natural_gradient_line_search.py",
            ),
            (
                "natural-gradient metric solves",
                "Armijo line-search and weighted-gradient aggregation",
                "natural-gradient optimizer loop controls",
            ),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_parameter_contracts.py",
            ("tests/test_differentiable_parameter_contracts.py",),
            ("parameter metadata validation", "multi-frequency shift conditioning"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_parameter_shift.py",
            ("tests/test_differentiable_parameter_shift.py",),
            (
                "parameter-shift transform wrappers",
                "finite-shot uncertainty propagation",
            ),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_result_contracts.py",
            ("tests/test_differentiable_result_contracts.py",),
            ("derivative result validation", "facade export compatibility"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_residual_weights.py",
            ("tests/test_differentiable_residual_weights.py",),
            ("Huber residual weights", "Soft-L1 residual weights"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_registered_custom.py",
            ("tests/test_differentiable_registered_custom.py",),
            ("registry-backed custom JVP/VJP wrappers", "facade export compatibility"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_scalar_kernels.py",
            ("tests/test_differentiable_scalar_kernels.py",),
            ("forward scalar elementary kernels", "reverse scalar pullback kernels"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_sparse_derivatives.py",
            ("tests/test_differentiable_sparse_derivatives.py",),
            ("sparse derivative conversion", "empirical-Fisher metric conversion"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_stochastic_policy.py",
            ("tests/test_stochastic_gradient_failure_policy.py",),
            ("confidence interval validation", "fail-closed uncertainty policy"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_stochastic_estimators.py",
            ("tests/test_differentiable_stochastic_estimators.py",),
            ("SPSA gradient estimation", "score-function and shot-budget estimators"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_transform_helpers.py",
            ("tests/test_differentiable_transform_helpers.py",),
            ("scalar transform helper contracts", "bounds and gradient clipping"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_rust_python_inventory.py",
            ("tests/test_differentiable_rust_python_inventory.py",),
            ("rustification classification rows", "claim-ledger readiness guards"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_architecture_map.py",
            ("tests/test_differentiable_architecture_map.py",),
            ("architecture routing rows", "rustification readiness guards"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_dependency_environment_map.py",
            ("tests/test_differentiable_dependency_environment_map.py",),
            ("dependency profile rows", "environment readiness guards"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_sota_scorecard.py",
            ("tests/test_differentiable_sota_scorecard.py",),
            ("category promotion blockers", "claim-ledger promotion guards"),
        ),
        _record(
            "src/scpn_quantum_control/differentiable_competitive_baselines.py",
            ("tests/test_differentiable_competitive_baselines.py",),
            ("competitive baseline freshness", "public promotion freshness gate"),
        ),
        _record(
            "src/scpn_quantum_control/whole_program_ad_result.py",
            ("tests/test_whole_program_ad_contracts.py",),
            ("whole-program result validation", "facade export compatibility"),
        ),
        _record(
            "src/scpn_quantum_control/whole_program_ad_api.py",
            ("tests/test_program_adjoint_replay.py",),
            ("whole-program API exports", "adjoint replay provenance"),
        ),
        _record(
            "src/scpn_quantum_control/whole_program_trace_runtime.py",
            ("tests/test_whole_program_trace_runtime.py",),
            ("runtime SSA/effect metadata", "source-line trace capture"),
        ),
        _record(
            "src/scpn_quantum_control/whole_program_trace_metadata.py",
            ("tests/test_whole_program_trace_metadata.py",),
            (
                "trace operand shape normalisation",
                "trace axis and reduction validation",
            ),
        ),
        _record(
            "src/scpn_quantum_control/whole_program_trace_predicates.py",
            ("tests/test_whole_program_trace_predicates.py",),
            (
                "primal control-flow predicate recording",
                "predicate-array shape boundaries",
            ),
        ),
        _record(
            "src/scpn_quantum_control/whole_program_trace_values.py",
            ("tests/test_whole_program_ad_contracts.py",),
            (
                "operator-intercepted trace value classes",
                "NumPy __array_function__ trace dispatch",
                "fail-closed nondifferentiability boundaries",
            ),
        ),
        _record(
            "src/scpn_quantum_control/phase/differentiable_audit.py",
            ("tests/test_phase_differentiable_audit.py",),
            ("audit-suite failure rows", "bounded workflow diagnostics"),
        ),
        _record(
            "src/scpn_quantum_control/phase/differentiable_readiness.py",
            ("tests/test_phase_differentiable_readiness.py",),
            ("readiness ledger failures", "hardware-blocked promotion gates"),
        ),
        _record(
            "src/scpn_quantum_control/phase/gradient_backend.py",
            ("tests/test_phase_gradient_backend.py",),
            ("backend support planning", "unsafe hardware route blocking"),
        ),
        _record(
            "src/scpn_quantum_control/phase/gradient_descent.py",
            ("tests/test_phase_gradient_descent.py",),
            ("descent certificate failures", "finite-gradient validation"),
        ),
        _record(
            "src/scpn_quantum_control/phase/gradient_support_matrix.py",
            ("tests/test_phase_gradient_support_matrix.py",),
            ("gate and observable support reports", "unsupported transform alternatives"),
        ),
        _record(
            "src/scpn_quantum_control/phase/gradient_tape.py",
            ("tests/test_phase_gradient_tape.py",),
            ("tape support records", "provider-boundary records"),
        ),
        _record(
            "src/scpn_quantum_control/diff.py",
            ("tests/test_diff_namespace.py",),
            ("canonical namespace imports", "fail-closed circuit diagnostics"),
        ),
        _record(
            "src/scpn_quantum_control/phase/hardware_gradient_campaign.py",
            ("tests/test_phase_hardware_gradient_campaign.py",),
            ("live-ticket gates", "raw-count replay requirements"),
        ),
        _record(
            "src/scpn_quantum_control/phase/hardware_gradient_policy.py",
            ("tests/test_phase_hardware_gradient_policy.py",),
            ("provider allowlist gates", "shot-budget gates"),
        ),
        _record(
            "src/scpn_quantum_control/phase/hardware_gradient_publication.py",
            ("tests/test_phase_hardware_gradient_publication.py",),
            ("no-submit publication rows", "unpromoted benchmark placeholders"),
        ),
        _record(
            "src/scpn_quantum_control/phase/jax_bridge.py",
            ("tests/test_phase_jax_bridge.py",),
            ("missing-JAX dependency records", "host-callback boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/phase/natural_gradient.py",
            ("tests/test_phase_natural_gradient.py",),
            ("singular-metric damping", "accepted-descent certificates"),
        ),
        _record(
            "src/scpn_quantum_control/phase/pennylane_bridge.py",
            ("tests/test_phase_pennylane_bridge.py", "tests/test_phase_pennylane_import.py"),
            ("missing-PennyLane dependency records", "round-trip failure diagnostics"),
        ),
        _record(
            "src/scpn_quantum_control/phase/pennylane_provider_plugin.py",
            ("tests/test_phase_pennylane_provider_plugin.py",),
            ("provider-plugin artifact validation", "provider-exceedance blockers"),
        ),
        _record(
            "src/scpn_quantum_control/phase/provider_gradient.py",
            ("tests/test_phase_provider_gradient.py",),
            ("malformed callback records", "finite-shot uncertainty records"),
        ),
        _record(
            "src/scpn_quantum_control/phase/provider_gradient_audit.py",
            ("tests/test_phase_provider_gradient_audit.py",),
            ("provider readiness failures", "hardware-blocked rows"),
        ),
        _record(
            "src/scpn_quantum_control/phase/provider_hardware_gradient_audit.py",
            ("tests/test_phase_provider_hardware_gradient_audit.py",),
            ("no-submit provider audits", "missing evidence gates"),
        ),
        _record(
            "src/scpn_quantum_control/phase/provider_hardware_safety_audit.py",
            ("tests/test_phase_provider_hardware_safety_audit.py",),
            ("aggregate provider safety gate", "hardware-promotion blockers"),
        ),
        _record(
            "src/scpn_quantum_control/phase/qiskit_bridge.py",
            ("tests/test_phase_qiskit_bridge.py",),
            ("no-submit Qiskit maturity rows", "provider execution blockers"),
        ),
        _record(
            "src/scpn_quantum_control/phase/qnn_framework_bridge_matrix.py",
            ("tests/test_phase_qnn_framework_bridge_matrix.py",),
            ("bridge support assertions", "arbitrary simulator blockers"),
        ),
        _record(
            "src/scpn_quantum_control/phase/qnode_affinity_benchmark.py",
            ("tests/test_phase_qnode_affinity_benchmark.py",),
            ("isolation metadata boundaries", "non-isolated benchmark labels"),
        ),
        _record(
            "src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py",
            ("tests/test_differentiable_isolated_benchmark_plan.py",),
            ("isolated benchmark batch rows", "promotion readiness guards"),
        ),
        _record(
            "src/scpn_quantum_control/phase/qnode_circuit.py",
            ("tests/test_phase_qnode_circuit.py",),
            ("gate and observable support errors", "density-route boundaries"),
        ),
        _record(
            "src/scpn_quantum_control/phase/qnode_framework_parity.py",
            ("tests/test_phase_qnode_framework_parity.py",),
            ("framework parity gap rows", "missing dependency records"),
        ),
        _record(
            "src/scpn_quantum_control/phase/qnode_provider_transforms.py",
            ("tests/test_phase_qnode_provider_transforms.py",),
            ("provider callback support decisions", "hardware route blocking"),
        ),
        _record(
            "src/scpn_quantum_control/phase/qnode_tape.py",
            ("tests/test_phase_qnode_tape.py",),
            ("tape replay support records", "dynamic-route blockers"),
        ),
        _record(
            "src/scpn_quantum_control/phase/qnode_transforms.py",
            ("tests/test_phase_qnode_transforms.py",),
            ("transform nesting support reports", "finite-shot curvature blockers"),
        ),
        _record(
            "src/scpn_quantum_control/phase/qnode_vector_transforms.py",
            ("tests/test_phase_qnode_vector_transforms.py",),
            ("JVP/VJP/Hessian shape checks", "complex-parameter blockers"),
        ),
        _record(
            "src/scpn_quantum_control/phase/tensorflow_bridge.py",
            ("tests/test_phase_framework_bridges.py",),
            ("missing-TensorFlow dependency records", "host-boundary diagnostics"),
        ),
        _record(
            "src/scpn_quantum_control/phase/torch_bridge.py",
            ("tests/test_phase_framework_bridges.py",),
            ("missing-PyTorch dependency records", "compile and func blockers"),
        ),
        _record(
            "src/scpn_quantum_control/phase/xy_compiler.py",
            ("tests/test_xy_compiler.py", "tests/test_xy_compiler_contracts.py"),
            ("compiler input validation", "unsupported mapping diagnostics"),
        ),
    )


def run_differentiable_module_hardening_audit(
    *,
    repo_root: Path = REPO_ROOT,
    registry: Sequence[DifferentiableModuleHardeningRecord] | None = None,
) -> DifferentiableModuleHardeningAuditResult:
    """Audit differentiable modules against registered tests and diagnostics."""
    records = tuple(registry or differentiable_module_hardening_registry())
    discovered = _discover_differentiable_modules(repo_root)
    registered_paths = tuple(record.module_path for record in records)
    registered_set = set(registered_paths)
    discovered_set = set(discovered)
    missing_registry = tuple(sorted(discovered_set - registered_set))
    stale_registry = tuple(sorted(registered_set - discovered_set))
    missing_tests = tuple(
        sorted(
            {
                test_path
                for record in records
                for test_path in record.test_paths
                if not (repo_root / test_path).is_file()
            }
        )
    )
    duplicate_paths = _duplicates(registered_paths)
    errors: list[str] = []
    errors.extend(f"missing registry entry: {path}" for path in missing_registry)
    errors.extend(f"stale registry entry: {path}" for path in stale_registry)
    errors.extend(f"missing test path: {path}" for path in missing_tests)
    errors.extend(f"duplicate registry entry: {path}" for path in duplicate_paths)
    for record in records:
        if not record.module_path.startswith("src/scpn_quantum_control/"):
            errors.append(f"{record.module_path}: module path is outside package")
        for test_path in record.test_paths:
            if not test_path.startswith("tests/test_"):
                errors.append(f"{record.module_path}: non-module-specific test path {test_path}")
    return DifferentiableModuleHardeningAuditResult(
        passed=not errors,
        records=records,
        discovered_module_paths=discovered,
        missing_registry_paths=missing_registry,
        stale_registry_paths=stale_registry,
        missing_test_paths=missing_tests,
        errors=tuple(errors),
        claim_boundary=(
            "This audit verifies differentiable module inventory, module-specific "
            "test surfaces, and declared fail-closed diagnostic surfaces. It does "
            "not prove full formal correctness, provider execution, hardware "
            "execution, or isolated benchmark promotion."
        ),
    )


def _record(
    module_path: str,
    test_paths: Sequence[str],
    diagnostic_surfaces: Sequence[str],
) -> DifferentiableModuleHardeningRecord:
    return DifferentiableModuleHardeningRecord(
        module_path=module_path,
        test_paths=tuple(test_paths),
        diagnostic_surfaces=tuple(diagnostic_surfaces),
    )


def _discover_differentiable_modules(repo_root: Path) -> tuple[str, ...]:
    paths: set[str] = set()
    for pattern in DIFFERENTIABLE_MODULE_PATTERNS:
        for path in repo_root.glob(pattern):
            if path.is_file() and path.name != "__init__.py":
                paths.add(path.relative_to(repo_root).as_posix())
    return tuple(sorted(paths))


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


__all__ = [
    "DIFFERENTIABLE_MODULE_PATTERNS",
    "DifferentiableModuleHardeningAuditResult",
    "DifferentiableModuleHardeningRecord",
    "differentiable_module_hardening_registry",
    "run_differentiable_module_hardening_audit",
]
