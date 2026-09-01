# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole-program trace-value quality gates
"""Build strict static and exact shared-coverage gates for trace values."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
PROGRAM_AD_LINALG_SOURCE = "src/scpn_quantum_control/program_ad_linalg_primitives.py"
PROGRAM_AD_LINALG_SPECTRAL_TEST = "tests/test_program_ad_linalg_spectral.py"
PROGRAM_AD_LINALG_EXACT_CONTRACTS_TEST = "tests/test_program_ad_linalg_exact_contracts.py"
WHOLE_PROGRAM_TRACE_VALUE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/whole_program_trace_values.py",
    "src/scpn_quantum_control/whole_program_trace_predicates.py",
    PROGRAM_AD_LINALG_SOURCE,
    "tests/test_whole_program_trace_predicates.py",
    "tests/test_whole_program_trace_values.py",
    "tests/test_whole_program_trace_value_operators.py",
    "tests/test_whole_program_trace_value_selection.py",
    "tests/test_whole_program_trace_value_signal.py",
    "tests/test_whole_program_trace_value_linalg.py",
    "tests/test_whole_program_trace_value_shapes.py",
    PROGRAM_AD_LINALG_SPECTRAL_TEST,
    PROGRAM_AD_LINALG_EXACT_CONTRACTS_TEST,
    "tools/whole_program_trace_value_quality_gates.py",
    "tests/test_whole_program_trace_value_quality_gate.py",
]
WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_COHORT = [
    "tests/test_program_ad_adjoint_generation.py",
    "tests/test_program_ad_adjoint_generation_docstrings.py",
    "tests/test_program_ad_alias_contracts.py",
    "tests/test_program_ad_alias_effects.py",
    "tests/test_program_ad_array_indexing_registry.py",
    "tests/test_program_ad_array_indexing_quality.py",
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
    PROGRAM_AD_LINALG_SPECTRAL_TEST,
    PROGRAM_AD_LINALG_EXACT_CONTRACTS_TEST,
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
WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_DATA_FILE = ".coverage.whole-program-trace-values"
WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_INCLUDE = (
    "*/whole_program_trace_values.py,*/whole_program_trace_predicates.py,"
    "*/program_ad_linalg_primitives.py"
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete preview documentation gates."""
    return [
        (
            "mypy-strict-whole-program-trace-values",
            [
                python,
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
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--preview",
                "--select",
                "D,D413,D417,D420",
                "--config",
                "lint.explicit-preview-rules = true",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *WHOLE_PROGRAM_TRACE_VALUE_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build the shared execution and exact owner-only coverage reports."""
    common = [
        python,
        "-m",
        "coverage",
        "report",
        f"--rcfile={devnull}",
        f"--data-file={WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_DATA_FILE}",
        "--precision=2",
        "--fail-under=100",
    ]
    return [
        (
            "whole-program trace-value focused coverage",
            [
                python,
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
            [*common, f"--include={WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_INCLUDE}"],
        ),
        (
            "program AD alias-contract exact coverage threshold",
            [*common, "--include=*/program_ad_alias_contracts.py"],
        ),
        (
            "program AD shape-transform exact coverage threshold",
            [*common, "--include=*/program_ad_shape_transforms.py"],
        ),
    ]


__all__ = [
    "PROGRAM_AD_LINALG_SOURCE",
    "PROGRAM_AD_LINALG_SPECTRAL_TEST",
    "WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_COHORT",
    "WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_DATA_FILE",
    "WHOLE_PROGRAM_TRACE_VALUE_COVERAGE_INCLUDE",
    "WHOLE_PROGRAM_TRACE_VALUE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
