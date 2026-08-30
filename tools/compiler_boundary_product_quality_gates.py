# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — compiler-boundary quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
COMPILER_BOUNDARY_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/compiler_boundary_product.py",
    "src/scpn_quantum_control/compiler/mlir_llvm_jit_claim_gate.py",
    "tests/test_compiler_boundary_product.py",
    "tests/test_llvm_jit_claim_gate.py",
    "tools/compiler_boundary_product_quality_gates.py",
    "tests/test_compiler_boundary_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
COMPILER_BOUNDARY_PRODUCT_COVERAGE_COHORT = [
    "tests/test_compiler_boundary_product.py",
    "tests/test_llvm_jit_claim_gate.py",
]
"""Tests that own exact compiler-boundary product coverage."""
COMPILER_BOUNDARY_PRODUCT_COVERAGE_DATA_FILE = ".coverage.compiler-boundary-product-quality"
"""Isolated coverage database for the compiler-boundary product owner."""
COMPILER_BOUNDARY_PRODUCT_COVERAGE_INCLUDE = (
    "*/compiler_boundary_product.py,*/compiler/mlir_llvm_jit_claim_gate.py"
)
"""Production compiler-boundary sources enforced at exact branch coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-compiler-boundary-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *COMPILER_BOUNDARY_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D compiler-boundary-product quality ratchet",
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
                'lint.pydocstyle.convention = "numpy"',
                *COMPILER_BOUNDARY_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "compiler-boundary-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={COMPILER_BOUNDARY_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *COMPILER_BOUNDARY_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "compiler-boundary-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={COMPILER_BOUNDARY_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={COMPILER_BOUNDARY_PRODUCT_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "COMPILER_BOUNDARY_PRODUCT_COVERAGE_COHORT",
    "COMPILER_BOUNDARY_PRODUCT_COVERAGE_DATA_FILE",
    "COMPILER_BOUNDARY_PRODUCT_COVERAGE_INCLUDE",
    "COMPILER_BOUNDARY_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
