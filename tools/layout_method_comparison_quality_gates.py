# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — layout method comparison quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
LAYOUT_METHOD_COMPARISON_QUALITY_RATCHET = [
    "src/scpn_quantum_control/benchmarks/layout_method_comparison.py",
    "tests/test_layout_method_comparison.py",
    "scripts/run_layout_method_comparison.py",
    "tests/test_run_layout_method_comparison.py",
    "tools/layout_method_comparison_quality_gates.py",
    "tests/test_layout_method_comparison_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
LAYOUT_METHOD_COMPARISON_COVERAGE_COHORT = [
    "tests/test_layout_method_comparison.py",
    "tests/test_run_layout_method_comparison.py",
]
"""Tests that own exact layout-method-comparison coverage."""
LAYOUT_METHOD_COMPARISON_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-layout-method-comparison-quality.coverage"  # nosec B108
)
"""Isolated coverage database for layout-method-comparison diagnostics."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-layout-method-comparison-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *LAYOUT_METHOD_COMPARISON_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D layout-method-comparison quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--select",
                "D,D413",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *LAYOUT_METHOD_COMPARISON_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    data = LAYOUT_METHOD_COMPARISON_COVERAGE_DATA_FILE
    return [
        (
            "layout-method-comparison focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *LAYOUT_METHOD_COMPARISON_COVERAGE_COHORT,
            ],
        ),
        (
            "layout-method-comparison exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/benchmarks/layout_method_comparison.py",
            ],
        ),
    ]


__all__ = [
    "LAYOUT_METHOD_COMPARISON_COVERAGE_COHORT",
    "LAYOUT_METHOD_COMPARISON_COVERAGE_DATA_FILE",
    "LAYOUT_METHOD_COMPARISON_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
