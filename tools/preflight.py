# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Local CI preflight — mirrors every CI gate so failures are caught before push.

Gates (in order):
  1. repository lint, format, documentation, generated-surface, policy, and security audits
  2. focused strict-typing and NumPy-docstring ratchets for promoted owner cohorts
  3. Rust formatting, version/export consistency, and repository typing gates
  4. exact MLIR-leaf, Phase-QNode-affinity, Phase-QNode-vector, Phase-QNode
     JAX, Studio Program-AD, and trace-value statement/branch coverage (default
     coverage mode only)
  5. repository pytest with the selected coverage mode
  6. Bandit security scan

Each exact owner gate uses an explicit responsibility-scoped test cohort, an
isolated coverage data file, and a 100% statement/branch threshold for only its
named production modules. ``--no-tests`` and ``--no-coverage`` skip those
executable coverage gates while retaining their static typing/docstring
ratchets.

Usage:
  python tools/preflight.py                # all gates (default)
  python tools/preflight.py --no-tests     # skip pytest entirely (quick lint pass)
  python tools/preflight.py --no-coverage  # run tests without coverage threshold
"""

from __future__ import annotations

import subprocess  # nosec B404
import sys
import time
from collections.abc import Iterable
from importlib import import_module
from os import X_OK, access, devnull, environ, pathsep
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools import phase_jax_qnode_quality_gates as _phase_jax_qnode_quality_gates
    from tools import program_ad_quality_gates as _program_ad_quality_gates
else:
    _repo_root = str(Path(__file__).resolve().parents[1])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    _phase_jax_qnode_quality_gates = import_module("tools.phase_jax_qnode_quality_gates")
    _program_ad_quality_gates = import_module("tools.program_ad_quality_gates")

ROOT = Path(__file__).resolve().parent.parent
_PY = sys.executable
_CARGO = which("cargo") or "cargo"
_PNPM = which("pnpm") or "pnpm"
_RUNTIME_SOURCE_ROOTS = (ROOT / "src", ROOT / "oscillatools" / "src")
_HELP_FLAGS = frozenset({"-h", "--help"})

STUDIO_PROGRAM_AD_QUALITY_RATCHET = _program_ad_quality_gates.STUDIO_PROGRAM_AD_QUALITY_RATCHET
STUDIO_PROGRAM_AD_COVERAGE_COHORT = _program_ad_quality_gates.STUDIO_PROGRAM_AD_COVERAGE_COHORT
STUDIO_PROGRAM_AD_BROWSER_TESTS = _program_ad_quality_gates.STUDIO_PROGRAM_AD_BROWSER_TESTS
STUDIO_PROGRAM_AD_COVERAGE_DATA_FILE = (
    _program_ad_quality_gates.STUDIO_PROGRAM_AD_COVERAGE_DATA_FILE
)
PHASE_JAX_QNODE_QUALITY_RATCHET = _phase_jax_qnode_quality_gates.PHASE_JAX_QNODE_QUALITY_RATCHET
PHASE_JAX_QNODE_COVERAGE_COHORT = _phase_jax_qnode_quality_gates.PHASE_JAX_QNODE_COVERAGE_COHORT
PHASE_JAX_QNODE_COVERAGE_DATA_FILE = (
    _phase_jax_qnode_quality_gates.PHASE_JAX_QNODE_COVERAGE_DATA_FILE
)

DIFFERENTIABLE_DOCSTRING_RATCHET = [
    "src/scpn_quantum_control/differentiable_architecture_map.py",
    "src/scpn_quantum_control/differentiable_claim_ledger.py",
    "src/scpn_quantum_control/differentiable_claim_rendering.py",
    "src/scpn_quantum_control/differentiable_competitive_baselines.py",
    "src/scpn_quantum_control/differentiable_dependency_environment_evidence.py",
    "src/scpn_quantum_control/differentiable_dependency_environment_map.py",
    "src/scpn_quantum_control/differentiable_baseline_scorecard.py",
    "src/scpn_quantum_control/differentiable_external_validation.py",
    "src/scpn_quantum_control/differentiable_finite_difference.py",
    "src/scpn_quantum_control/differentiable_module_hardening_audit.py",
    "src/scpn_quantum_control/differentiable_transform_algebra.py",
    "src/scpn_quantum_control/program_ad_alias_contracts.py",
    "src/scpn_quantum_control/program_ad_registry.py",
    "src/scpn_quantum_control/program_ad_shape_transforms.py",
    "src/scpn_quantum_control/studio/evidence_bundle.py",
    "src/scpn_quantum_control/phase/tensorflow_maintenance.py",
    "src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py",
    "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py",
    "tests/test_differentiable_external_validation.py",
    "tests/test_differentiable_finite_difference.py",
    "tests/test_differentiable_competitive_baselines.py",
    "tests/test_differentiable_module_hardening_audit.py",
    "tests/test_differentiable_transform_algebra.py",
    "tests/test_program_ad_alias_contracts.py",
    "tests/test_program_ad_registry.py",
    "tests/test_program_ad_shape_transforms.py",
    "tests/test_phase_tensorflow_maintenance.py",
    "tests/test_differentiable_hardening_gate.py",
    "tools/differentiable_support_matrix_page.py",
    "tests/test_differentiable_support_matrix_page.py",
    "tools/differentiable_reviewer_evidence_catalog.py",
    "tools/differentiable_reviewer_evidence_page.py",
    "tests/test_differentiable_reviewer_evidence_page.py",
]

REALTIME_RUNTIME_QUALITY_RATCHET = [
    "src/scpn_quantum_control/control/realtime_runtime.py",
    "tests/test_realtime_runtime.py",
    "tests/test_realtime_runtime_branches.py",
]

PHASE_QNODE_AFFINITY_QUALITY_RATCHET = [
    "src/scpn_quantum_control/phase/qnode_affinity_benchmark.py",
    "tools/lean_phase_import.py",
    "tools/run_phase_qnode_affinity_benchmark.py",
    "tests/test_phase_qnode_affinity_benchmark.py",
    "tests/test_lean_phase_import.py",
]

PHASE_QNODE_AFFINITY_COVERAGE_COHORT = [
    "tests/test_phase_qnode_affinity_benchmark.py",
    "tests/test_lean_phase_import.py",
]

PHASE_QNODE_VECTOR_QUALITY_RATCHET = [
    "src/scpn_quantum_control/phase/qnode_vector_transforms.py",
    "tests/test_phase_qnode_vector_transforms.py",
    "tests/test_phase_qnode_rust_parity.py",
]

MLIR_LEAF_QUALITY_RATCHET = [
    "src/scpn_quantum_control/compiler/mlir_enzyme_audit.py",
    "src/scpn_quantum_control/compiler/mlir_phase_qnode_runtime.py",
    "src/scpn_quantum_control/compiler/mlir_transform_plan_assembly.py",
    "src/scpn_quantum_control/compiler/mlir_workload_compilation.py",
    "tests/_mlir_native_compilation_test_helpers.py",
    "tests/test_mlir_enzyme_audit.py",
    "tests/test_mlir_toolchain_probe_hardening.py",
    "tests/test_mlir_phase_qnode_runtime.py",
    "tests/test_phase_qnode_compiler_lowering.py",
    "tests/test_mlir_transform_plan.py",
    "tests/test_mlir_transform_plan_assembly.py",
    "tests/test_mlir_workload_compilation.py",
    "tests/test_mlir_executable_batching_integration.py",
    "tests/test_mlir_native_compilation_integration.py",
    "tests/test_mlir_scalar_native_compilation_integration.py",
    "tests/test_mlir_vector_native_compilation_integration.py",
    "tests/test_mlir_matrix_native_compilation_integration.py",
    "tests/test_mlir_matrix_2x2_native_compilation_integration.py",
    "tests/test_mlir_symmetric_native_compilation_integration.py",
]

MLIR_LEAF_COVERAGE_COHORT = [
    "tests/test_mlir_enzyme_audit.py",
    "tests/test_mlir_toolchain_probe_hardening.py",
    "tests/test_mlir_phase_qnode_runtime.py",
    "tests/test_phase_qnode_compiler_lowering.py",
    "tests/test_mlir_transform_plan.py",
    "tests/test_mlir_transform_plan_assembly.py",
    "tests/test_mlir_workload_compilation.py",
    "tests/test_mlir_executable_batching_integration.py",
    "tests/test_mlir_native_compilation_integration.py",
    "tests/test_mlir_scalar_native_compilation_integration.py",
    "tests/test_mlir_vector_native_compilation_integration.py",
    "tests/test_mlir_matrix_native_compilation_integration.py",
    "tests/test_mlir_matrix_2x2_native_compilation_integration.py",
    "tests/test_mlir_symmetric_native_compilation_integration.py",
]

PHASE_QNODE_VECTOR_COVERAGE_COHORT = [
    "tests/test_phase_qnode_vector_transforms.py",
]

WHOLE_PROGRAM_TRACE_VALUE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/whole_program_trace_values.py",
    "tests/test_whole_program_trace_values.py",
    "tests/test_whole_program_trace_value_operators.py",
    "tests/test_whole_program_trace_value_selection.py",
    "tests/test_whole_program_trace_value_signal.py",
    "tests/test_whole_program_trace_value_linalg.py",
    "tests/test_whole_program_trace_value_shapes.py",
]

WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_COHORT = [
    "tests/test_program_ad_adjoint_generation.py",
    "tests/test_program_ad_adjoint_generation_docstrings.py",
    "tests/test_program_ad_alias_contracts.py",
    "tests/test_program_ad_alias_effects.py",
    "tests/test_program_ad_array_indexing_registry.py",
    "tests/test_program_ad_binary_elementwise_registry.py",
    "tests/test_program_ad_broadcast_assembly.py",
    "tests/test_program_ad_cumulative_primitives.py",
    "tests/test_program_ad_cumulative_primitives_docstrings.py",
    "tests/test_program_ad_effect_ir.py",
    "tests/test_program_ad_elementwise_registry.py",
    "tests/test_program_ad_fail_closed_boundaries.py",
    "tests/test_program_ad_finite_difference_gradient_check.py",
    "tests/test_program_ad_finite_difference_stencils.py",
    "tests/test_program_ad_interpolation.py",
    "tests/test_program_ad_interpolation_primitives_docstrings.py",
    "tests/test_program_ad_like_constructors.py",
    "tests/test_program_ad_linalg_core.py",
    "tests/test_program_ad_linalg_direct_rules.py",
    "tests/test_program_ad_linalg_matrix_ops.py",
    "tests/test_program_ad_linalg_registry.py",
    "tests/test_program_ad_linalg_spectral.py",
    "tests/test_program_ad_product_contractions.py",
    "tests/test_program_ad_reduction_norms.py",
    "tests/test_program_ad_reduction_primitives_docstrings.py",
    "tests/test_program_ad_registry.py",
    "tests/test_program_ad_runtime_registry_dispatch.py",
    "tests/test_program_ad_selection_direct_rules.py",
    "tests/test_program_ad_selection_folds.py",
    "tests/test_program_ad_selection_order_statistics.py",
    "tests/test_program_ad_selection_primitives_docstrings.py",
    "tests/test_program_ad_selection_registry.py",
    "tests/test_program_ad_shape_transforms.py",
    "tests/test_program_ad_signal_primitives.py",
    "tests/test_program_ad_split_assembly.py",
    "tests/test_program_ad_stack_block_assembly.py",
    "tests/test_program_ad_static_array_assembly.py",
    "tests/test_program_ad_stencil_primitives_docstrings.py",
    "tests/test_program_ad_structural_finite_difference_gradient_check.py",
    "tests/test_program_ad_trapezoid.py",
    "tests/test_program_ad_triangular_diagonal_assembly.py",
    "tests/test_program_ad_unary_ufuncs.py",
    "tests/test_program_adjoint_replay.py",
    "tests/test_whole_program_ad_contracts.py",
    "tests/test_whole_program_ad_finite_difference_gradient_check.py",
    "tests/test_whole_program_ad_numpy_structural.py",
    "tests/test_whole_program_ad_runtime.py",
    "tests/test_whole_program_frontend.py",
    "tests/test_whole_program_frontend_contracts.py",
    "tests/test_whole_program_trace_metadata.py",
    "tests/test_whole_program_trace_predicates.py",
    "tests/test_whole_program_trace_runtime.py",
    "tests/test_whole_program_trace_value_linalg.py",
    "tests/test_whole_program_trace_value_operators.py",
    "tests/test_whole_program_trace_value_selection.py",
    "tests/test_whole_program_trace_value_shapes.py",
    "tests/test_whole_program_trace_value_signal.py",
    "tests/test_whole_program_trace_values.py",
]

MLIR_LEAF_COVERAGE_DATA_FILE = ".coverage.mlir-leaf-quality"
MLIR_LEAF_COVERAGE_SOURCE = "src/scpn_quantum_control/compiler"
MLIR_LEAF_COVERAGE_INCLUDE = (
    "*/mlir_enzyme_audit.py,*/mlir_phase_qnode_runtime.py,"
    "*/mlir_transform_plan_assembly.py,*/mlir_workload_compilation.py"
)
PHASE_QNODE_AFFINITY_COVERAGE_DATA_FILE = ".coverage.phase-qnode-affinity"
PHASE_QNODE_VECTOR_COVERAGE_DATA_FILE = ".coverage.phase-qnode-vector"
WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_DATA_FILE = ".coverage.whole-program-trace-values"

_PYTEST_BASE = [
    _PY,
    "-m",
    "pytest",
    "tests/",
    "-x",
    "--tb=short",
    "-q",
    "--ignore=tests/test_hardware_runner.py",
    "--ignore=tests/test_dynamical_lie_algebra.py",  # DLA: 27 min/test, skip for pre-push
]

_PYTEST_COV = _PYTEST_BASE + [
    "--cov=src/scpn_quantum_control",
    "--cov-branch",
    "--cov-fail-under=70",  # temporary combined local smoke guard; CI separately gates lines
]

STATIC_GATES: list[tuple[str, list[str]]] = [
    ("ruff check", [_PY, "-m", "ruff", "check", "src/", "tests/"]),
    ("ruff format", [_PY, "-m", "ruff", "format", "--check", "src/", "tests/"]),
    (
        "documentation-surface",
        [
            _PY,
            "tools/audit_documentation_surface.py",
            "--allowlist",
            "tools/documentation_surface_allowlist.json",
            "--fail-on-findings",
        ],
    ),
    (
        "differentiable-promotion-language",
        [_PY, "tools/check_differentiable_promotion_language.py"],
    ),
    (
        "differentiable-competitive-baselines",
        [_PY, "tools/check_differentiable_competitive_baselines.py"],
    ),
    (
        "differentiable-transform-algebra",
        [_PY, "tools/check_differentiable_transform_algebra.py"],
    ),
    (
        "differentiable-support-matrix-page",
        [_PY, "tools/differentiable_support_matrix_page.py", "--check"],
    ),
    (
        "mypy-strict-differentiable-support-matrix-page",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "--explicit-package-bases",
            "tools/differentiable_support_matrix_page.py",
            "tests/test_differentiable_support_matrix_page.py",
        ],
    ),
    (
        "differentiable-reviewer-evidence-page",
        [_PY, "tools/differentiable_reviewer_evidence_page.py", "--check"],
    ),
    (
        "mypy-strict-differentiable-reviewer-evidence-page",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "--explicit-package-bases",
            "tools/differentiable_reviewer_evidence_catalog.py",
            "tools/differentiable_reviewer_evidence_page.py",
            "tests/test_differentiable_reviewer_evidence_page.py",
        ],
    ),
    (
        "ruff D differentiable module-hardening ratchet",
        [
            _PY,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "D,D413",
            "--config",
            'lint.pydocstyle.convention = "numpy"',
            *DIFFERENTIABLE_DOCSTRING_RATCHET,
        ],
    ),
    (
        "mypy-strict-realtime-runtime",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "--explicit-package-bases",
            *REALTIME_RUNTIME_QUALITY_RATCHET,
        ],
    ),
    (
        "ruff D realtime-runtime quality ratchet",
        [
            _PY,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "D,D413",
            "--config",
            'lint.pydocstyle.convention = "numpy"',
            *REALTIME_RUNTIME_QUALITY_RATCHET,
        ],
    ),
    (
        "mypy-strict-phase-qnode-affinity",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "--explicit-package-bases",
            *PHASE_QNODE_AFFINITY_QUALITY_RATCHET,
        ],
    ),
    (
        "ruff D phase-qnode-affinity quality ratchet",
        [
            _PY,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "D,D413",
            "--config",
            'lint.pydocstyle.convention = "numpy"',
            *PHASE_QNODE_AFFINITY_QUALITY_RATCHET,
        ],
    ),
    *_program_ad_quality_gates.build_static_quality_gates(_PY),
    (
        "mypy-strict-mlir-leaf-quality",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            *MLIR_LEAF_QUALITY_RATCHET,
        ],
    ),
    (
        "ruff D MLIR-leaf quality ratchet",
        [
            _PY,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "D,D413",
            "--config",
            'lint.pydocstyle.convention = "numpy"',
            *MLIR_LEAF_QUALITY_RATCHET,
        ],
    ),
    (
        "mypy-strict-phase-qnode-vector",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "--explicit-package-bases",
            *PHASE_QNODE_VECTOR_QUALITY_RATCHET,
        ],
    ),
    (
        "ruff D phase-qnode-vector quality ratchet",
        [
            _PY,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "D,D413",
            "--config",
            'lint.pydocstyle.convention = "numpy"',
            *PHASE_QNODE_VECTOR_QUALITY_RATCHET,
        ],
    ),
    *_phase_jax_qnode_quality_gates.build_static_quality_gates(_PY),
    (
        "mypy-strict-whole-program-trace-values",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "--explicit-package-bases",
            *WHOLE_PROGRAM_TRACE_VALUE_QUALITY_RATCHET,
        ],
    ),
    (
        "ruff D whole-program trace-value quality ratchet",
        [
            _PY,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "D,D413",
            "--config",
            'lint.pydocstyle.convention = "numpy"',
            *WHOLE_PROGRAM_TRACE_VALUE_QUALITY_RATCHET,
        ],
    ),
    ("test-quality", [_PY, "tools/audit_test_quality.py"]),
    ("module-size-policy", [_PY, "tools/audit_module_size_policy.py"]),
    (
        "mypy-strict-module-size-policy",
        [_PY, "-m", "mypy", "--strict", "tools/audit_module_size_policy.py"],
    ),
    ("licence-readiness", [_PY, "tools/audit_license_readiness.py"]),
    (
        "mypy-strict-licence-readiness",
        [_PY, "-m", "mypy", "--strict", "tools/audit_license_readiness.py"],
    ),
    ("test-typing-policy", [_PY, "tools/audit_test_typing_policy.py"]),
    (
        "mypy-strict-test-typing-policy",
        [_PY, "-m", "mypy", "--strict", "tools/audit_test_typing_policy.py"],
    ),
    (
        "coverage-policy",
        [_PY, "tools/audit_coverage_policy.py", "--validate-policy"],
    ),
    (
        "mypy-strict-coverage-policy",
        [_PY, "-m", "mypy", "--strict", "tools/audit_coverage_policy.py"],
    ),
    ("coverage-debt", [_PY, "tools/audit_coverage_debt.py"]),
    (
        "mypy-strict-coverage-debt",
        [_PY, "-m", "mypy", "--strict", "tools/audit_coverage_debt.py"],
    ),
    (
        "differentiable-external-validation",
        [_PY, "tools/check_differentiable_external_validation.py"],
    ),
    (
        "mypy-strict-differentiable-external-validation",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "tools/check_differentiable_external_validation.py",
        ],
    ),
    (
        "rustfmt",
        [
            _CARGO,
            "fmt",
            "--manifest-path",
            "scpn_quantum_engine/Cargo.toml",
            "--all",
            "--",
            "--check",
        ],
    ),
    ("version-sync", [_PY, "scripts/check_version_consistency.py"]),
    ("rust-pyi", [_PY, "tools/check_rust_pyi_exports.py"]),
    ("mypy", [_PY, "-m", "mypy"]),
    (
        "mypy-strict-differentiable",
        [
            _PY,
            "-m",
            "mypy",
            "--strict",
            "src/scpn_quantum_control/differentiable.py",
            "src/scpn_quantum_control/differentiable_claim_ledger.py",
            "src/scpn_quantum_control/differentiable_architecture_map.py",
            "src/scpn_quantum_control/differentiable_competitive_baselines.py",
            "src/scpn_quantum_control/diff.py",
            "src/scpn/diff.py",
            "src/scpn/__init__.py",
            "src/scpn_quantum_control/differentiable_dependency_environment_evidence.py",
            "src/scpn_quantum_control/differentiable_dependency_environment_map.py",
            "src/scpn_quantum_control/differentiable_baseline_scorecard.py",
            "src/scpn_quantum_control/differentiable_api.py",
            "src/scpn_quantum_control/benchmarks/differentiable_programming.py",
            "src/scpn_quantum_control/differentiable_external_validation.py",
            "src/scpn_quantum_control/differentiable_framework_overlay.py",
            "src/scpn_quantum_control/differentiable_module_hardening_audit.py",
            "src/scpn_quantum_control/differentiable_transform_algebra.py",
            "src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py",
            "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py",
            "src/scpn_quantum_control/benchmarks/differentiable_evidence.py",
            "src/scpn_quantum_control/phase/differentiable_readiness.py",
            "src/scpn_quantum_control/phase/differentiable_audit.py",
            "src/scpn_quantum_control/phase/gradient_support_matrix.py",
            "src/scpn_quantum_control/phase/provider_gradient.py",
            "src/scpn_quantum_control/phase/hardware_gradient_policy.py",
            "src/scpn_quantum_control/phase/provider_gradient_audit.py",
            "src/scpn_quantum_control/phase/hardware_gradient_publication.py",
            "src/scpn_quantum_control/phase/provider_hardware_gradient_audit.py",
            "src/scpn_quantum_control/phase/hardware_gradient_campaign.py",
            "src/scpn_quantum_control/phase/gradient_backend.py",
            "src/scpn_quantum_control/phase/gradient_tape.py",
            "src/scpn_quantum_control/phase/natural_gradient.py",
            "src/scpn_quantum_control/phase/gradient_descent.py",
            "src/scpn_quantum_control/phase/qnode_affinity_benchmark.py",
            "src/scpn_quantum_control/phase/qnode_tape.py",
            "src/scpn_quantum_control/phase/qnode_provider_transforms.py",
            "src/scpn_quantum_control/phase/qnode_transforms.py",
            "src/scpn_quantum_control/phase/qnode_vector_transforms.py",
            "src/scpn_quantum_control/phase/qnode_framework_parity.py",
            "src/scpn_quantum_control/phase/qnode_circuit_builders.py",
            "src/scpn_quantum_control/phase/qnode_circuit.py",
            "src/scpn_quantum_control/phase/qnode_circuit_contracts.py",
            "src/scpn_quantum_control/phase/qnode_circuit_differentiation.py",
            "src/scpn_quantum_control/phase/qnode_circuit_execution.py",
            "src/scpn_quantum_control/phase/qnode_circuit_support.py",
            "src/scpn_quantum_control/phase/pennylane_bridge.py",
            "src/scpn_quantum_control/phase/pennylane_provider_plugin.py",
            "src/scpn_quantum_control/phase/jax_bridge.py",
            "src/scpn_quantum_control/phase/jax_bridge_contracts.py",
            "src/scpn_quantum_control/phase/jax_compatibility.py",
            "src/scpn_quantum_control/phase/jax_gradients.py",
            "src/scpn_quantum_control/phase/jax_maturity.py",
            "src/scpn_quantum_control/phase/jax_qnode_transforms.py",
            "src/scpn_quantum_control/phase/torch_bridge.py",
            "src/scpn_quantum_control/phase/torch_bridge_contracts.py",
            "src/scpn_quantum_control/phase/torch_compatibility.py",
            "src/scpn_quantum_control/phase/torch_gradients.py",
            "src/scpn_quantum_control/phase/torch_maturity.py",
            "src/scpn_quantum_control/phase/torch_qnode_transforms.py",
            "src/scpn_quantum_control/phase/tensorflow_bridge.py",
            "src/scpn_quantum_control/phase/tensorflow_bridge_contracts.py",
            "src/scpn_quantum_control/phase/tensorflow_compatibility.py",
            "src/scpn_quantum_control/phase/tensorflow_gradients.py",
            "src/scpn_quantum_control/phase/tensorflow_maintenance.py",
            "src/scpn_quantum_control/phase/qiskit_bridge.py",
            "src/scpn_quantum_control/phase/qiskit_bridge_contracts.py",
            "src/scpn_quantum_control/phase/qiskit_gradients.py",
            "src/scpn_quantum_control/phase/qiskit_runtime.py",
            "src/scpn_quantum_control/phase/qnn_framework_bridge_matrix.py",
            "src/scpn_quantum_control/phase/transform_nesting.py",
            "src/scpn_quantum_control/benchmarks/differentiable_external_comparison.py",
            "src/scpn_quantum_control/phase/xy_compiler.py",
            "src/scpn_quantum_control/phase/pennylane_import.py",
            "src/scpn_quantum_control/phase/qnn_optimizer_benchmark.py",
            "src/scpn_quantum_control/phase/qnn_training.py",
            "src/scpn_quantum_control/phase/qnn_conformance.py",
            "src/scpn_quantum_control/phase/qnn_finite_shot.py",
            "src/scpn_quantum_control/phase/qnn_convergence.py",
            "src/scpn_quantum_control/phase/qnn_loss_landscape.py",
            "src/scpn_quantum_control/phase/qgnn.py",
            "src/scpn_quantum_control/phase/qnn_framework_agreement.py",
            "src/scpn_quantum_control/phase/model_training_evidence.py",
            "src/scpn_quantum_control/phase/domain_benchmark_datasets.py",
            "src/scpn_quantum_control/phase/objectives.py",
            "src/scpn_quantum_control/phase/objective_planner.py",
            "src/scpn_quantum_control/phase/objective_audit.py",
            "src/scpn_quantum_control/phase/optimizer_audit.py",
            "src/scpn_quantum_control/phase/param_shift.py",
            "src/scpn_quantum_control/phase/general_unitary.py",
            "src/scpn_quantum_control/phase/phase_vqe.py",
            "src/scpn_quantum_control/phase/structured_ansatz.py",
            "src/scpn_quantum_control/phase/xy_kuramoto.py",
            "src/scpn_quantum_control/phase/kuramoto_variants.py",
            "src/scpn_quantum_control/phase/adapt_vqe.py",
            "src/scpn_quantum_control/phase/trotter_error.py",
            "src/scpn_quantum_control/phase/ansatz_methodology.py",
            "src/scpn_quantum_control/phase/results.py",
            "src/scpn_quantum_control/phase/provider_hardware_safety_audit.py",
            "src/scpn_quantum_control/phase/backend_selector.py",
            "src/scpn_quantum_control/phase/ansatz_bench.py",
            "src/scpn_quantum_control/phase/trotter_upde.py",
            "src/scpn_quantum_control/phase/adiabatic_preparation.py",
            "src/scpn_quantum_control/phase/ancilla_lindblad.py",
            "src/scpn_quantum_control/phase/avqds.py",
            "src/scpn_quantum_control/phase/varqite.py",
            "src/scpn_quantum_control/phase/variational_metric.py",
            "src/scpn_quantum_control/phase/coupling_learning.py",
            "src/scpn_quantum_control/phase/contraction_optimiser.py",
            "src/scpn_quantum_control/phase/cross_domain_transfer.py",
            "src/scpn_quantum_control/phase/floquet_kuramoto.py",
        ],
    ),
]

MLIR_LEAF_COVERAGE_GATES: list[tuple[str, list[str]]] = [
    (
        "MLIR leaf focused coverage",
        [
            _PY,
            "-m",
            "coverage",
            "run",
            f"--rcfile={devnull}",
            f"--data-file={MLIR_LEAF_COVERAGE_DATA_FILE}",
            "--branch",
            f"--source={MLIR_LEAF_COVERAGE_SOURCE}",
            "-m",
            "pytest",
            "-q",
            *MLIR_LEAF_COVERAGE_COHORT,
        ],
    ),
    (
        "MLIR leaf exact coverage threshold",
        [
            _PY,
            "-m",
            "coverage",
            "report",
            f"--rcfile={devnull}",
            f"--data-file={MLIR_LEAF_COVERAGE_DATA_FILE}",
            "--precision=2",
            "--fail-under=100",
            f"--include={MLIR_LEAF_COVERAGE_INCLUDE}",
        ],
    ),
]

PHASE_QNODE_AFFINITY_COVERAGE_GATES: list[tuple[str, list[str]]] = [
    (
        "phase-qnode affinity focused coverage",
        [
            _PY,
            "-m",
            "coverage",
            "run",
            f"--rcfile={devnull}",
            f"--data-file={PHASE_QNODE_AFFINITY_COVERAGE_DATA_FILE}",
            "--branch",
            "-m",
            "pytest",
            "-q",
            *PHASE_QNODE_AFFINITY_COVERAGE_COHORT,
        ],
    ),
    (
        "phase-qnode affinity exact coverage threshold",
        [
            _PY,
            "-m",
            "coverage",
            "report",
            f"--rcfile={devnull}",
            f"--data-file={PHASE_QNODE_AFFINITY_COVERAGE_DATA_FILE}",
            "--precision=2",
            "--fail-under=100",
            "--include=*/qnode_affinity_benchmark.py",
        ],
    ),
]

STUDIO_PROGRAM_AD_COVERAGE_GATES = _program_ad_quality_gates.build_python_coverage_gates(_PY)
STUDIO_PROGRAM_AD_RUNTIME_GATES = _program_ad_quality_gates.build_runtime_gates(
    _CARGO,
    _PNPM,
)
STUDIO_PROGRAM_AD_BROWSER_TEST_GATE = _program_ad_quality_gates.build_browser_test_gate(_PNPM)
STUDIO_PROGRAM_AD_BROWSER_COVERAGE_GATE = _program_ad_quality_gates.build_browser_coverage_gate(
    _PNPM
)

PHASE_QNODE_VECTOR_COVERAGE_GATES: list[tuple[str, list[str]]] = [
    (
        "phase-qnode vector focused coverage",
        [
            _PY,
            "-m",
            "coverage",
            "run",
            f"--rcfile={devnull}",
            f"--data-file={PHASE_QNODE_VECTOR_COVERAGE_DATA_FILE}",
            "--branch",
            "-m",
            "pytest",
            "-q",
            *PHASE_QNODE_VECTOR_COVERAGE_COHORT,
        ],
    ),
    (
        "phase-qnode vector exact coverage threshold",
        [
            _PY,
            "-m",
            "coverage",
            "report",
            f"--rcfile={devnull}",
            f"--data-file={PHASE_QNODE_VECTOR_COVERAGE_DATA_FILE}",
            "--precision=2",
            "--fail-under=100",
            "--include=*/qnode_vector_transforms.py",
        ],
    ),
]

PHASE_JAX_QNODE_COVERAGE_GATES = _phase_jax_qnode_quality_gates.build_coverage_gates(_PY)

WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_GATES: list[tuple[str, list[str]]] = [
    (
        "whole-program trace-value focused coverage",
        [
            _PY,
            "-m",
            "coverage",
            "run",
            f"--rcfile={devnull}",
            f"--data-file={WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_DATA_FILE}",
            "--branch",
            "-m",
            "pytest",
            "-q",
            *WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_COHORT,
        ],
    ),
    (
        "whole-program trace-value exact coverage threshold",
        [
            _PY,
            "-m",
            "coverage",
            "report",
            f"--rcfile={devnull}",
            f"--data-file={WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_DATA_FILE}",
            "--precision=2",
            "--fail-under=100",
            "--include=*/whole_program_trace_values.py",
        ],
    ),
    (
        "program AD alias-contract exact coverage threshold",
        [
            _PY,
            "-m",
            "coverage",
            "report",
            f"--rcfile={devnull}",
            f"--data-file={WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_DATA_FILE}",
            "--precision=2",
            "--fail-under=100",
            "--include=*/program_ad_alias_contracts.py",
        ],
    ),
    (
        "program AD shape-transform exact coverage threshold",
        [
            _PY,
            "-m",
            "coverage",
            "report",
            f"--rcfile={devnull}",
            f"--data-file={WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_DATA_FILE}",
            "--precision=2",
            "--fail-under=100",
            "--include=*/program_ad_shape_transforms.py",
        ],
    ),
]

BANDIT_GATE: tuple[str, list[str]] = (
    "bandit",
    [_PY, "-m", "bandit", "-r", "src/", "-ll", "-q"],
)


def _admit_gate_command(cmd: list[str]) -> list[str]:
    """Return a shell-free command with a verified executable path."""
    if not cmd:
        raise ValueError("gate command is empty")
    executable = Path(cmd[0])
    if not executable.is_absolute():
        raise ValueError(f"gate executable is not absolute: {cmd[0]}")
    try:
        exists = executable.exists()
    except (OSError, ValueError) as exc:
        raise ValueError(f"gate executable is not resolvable: {cmd[0]}") from exc
    if not exists:
        raise ValueError(f"gate executable is not resolvable: {cmd[0]}")
    if not executable.is_file():
        raise ValueError(f"gate executable is not a file: {executable}")
    if not access(executable, X_OK):
        raise ValueError(f"gate executable is not executable: {executable}")
    return [str(executable), *cmd[1:]]


def _deduplicated_path_entries(entries: Iterable[str]) -> list[str]:
    """Return path entries in first-seen order without empty duplicates."""
    seen: set[str] = set()
    deduplicated: list[str] = []
    for entry in entries:
        if not entry or entry in seen:
            continue
        seen.add(entry)
        deduplicated.append(entry)
    return deduplicated


def _gate_environment() -> dict[str, str]:
    """Return the subprocess environment for preflight gates.

    Tool scripts execute from ``tools/`` but import the repository packages.
    Prepending the source roots keeps local runtime checks aligned with the
    package layout and the explicit mypy path used for the install-free
    ``oscillatools`` sibling source tree.
    """
    env = dict(environ)
    source_roots = [str(path) for path in _RUNTIME_SOURCE_ROOTS if path.is_dir()]
    existing_pythonpath = env.get("PYTHONPATH", "")
    entries = _deduplicated_path_entries([*source_roots, *existing_pythonpath.split(pathsep)])
    if entries:
        env["PYTHONPATH"] = pathsep.join(entries)
    return env


def run_gate(name: str, cmd: list[str]) -> bool:
    """Run a named preflight command and print a compact result summary."""
    t0 = time.monotonic()
    try:
        admitted_cmd = _admit_gate_command(cmd)
    except ValueError as exc:
        elapsed = time.monotonic() - t0
        print(f"  FAIL  {name} ({elapsed:.1f}s)")
        print(f"        {exc}")
        return False
    result = subprocess.run(  # nosec B603
        admitted_cmd,
        cwd=ROOT,
        capture_output=True,
        env=_gate_environment(),
        text=True,
        shell=False,
    )
    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        print(f"  PASS  {name} ({elapsed:.1f}s)")
        return True
    print(f"  FAIL  {name} ({elapsed:.1f}s)")
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines()[-10:]:
            print(f"        {line}")
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines()[-10:]:
            print(f"        {line}")
    return False


def _wants_help(args: Iterable[str]) -> bool:
    """Return whether the supplied CLI arguments request usage text."""
    return any(arg in _HELP_FLAGS for arg in args)


def main() -> int:
    """Run the configured preflight gate suite."""
    args = sys.argv[1:]
    if _wants_help(args):
        print((__doc__ or "").strip())
        return 0

    skip_tests = "--no-tests" in args
    no_coverage = "--no-coverage" in args

    gates: list[tuple[str, list[str]]] = list(STATIC_GATES)

    if not skip_tests:
        gates.extend(STUDIO_PROGRAM_AD_RUNTIME_GATES)
        if no_coverage:
            gates.append(STUDIO_PROGRAM_AD_BROWSER_TEST_GATE)
            gates.append(("pytest", _PYTEST_BASE))
        else:
            gates.extend(MLIR_LEAF_COVERAGE_GATES)
            gates.extend(PHASE_QNODE_AFFINITY_COVERAGE_GATES)
            gates.extend(STUDIO_PROGRAM_AD_COVERAGE_GATES)
            gates.extend(PHASE_QNODE_VECTOR_COVERAGE_GATES)
            gates.extend(PHASE_JAX_QNODE_COVERAGE_GATES)
            gates.extend(WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_GATES)
            gates.append(STUDIO_PROGRAM_AD_BROWSER_COVERAGE_GATE)
            gates.append(("pytest + coverage", _PYTEST_COV))

    gates.append(BANDIT_GATE)

    print(f"preflight: {len(gates)} gates")
    print()

    t_start = time.monotonic()
    failed: list[str] = []

    for name, cmd in gates:
        if not run_gate(name, cmd):
            failed.append(name)
            break

    elapsed = time.monotonic() - t_start
    print()
    if failed:
        print(f"BLOCKED: {', '.join(failed)} ({elapsed:.1f}s)")
        return 1
    print(f"ALL CLEAR: ready to push ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
