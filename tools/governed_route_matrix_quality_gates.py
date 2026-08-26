# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — governed route-matrix quality gates
"""Build strict documentation, typing, and coverage gates for BL-19."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
GOVERNED_ROUTE_MATRIX_QUALITY_RATCHET = [
    "src/scpn_quantum_control/governed_route_matrix.py",
    "tests/test_governed_route_matrix.py",
    "tools/governed_route_matrix_quality_gates.py",
    "tests/test_governed_route_matrix_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
GOVERNED_ROUTE_MATRIX_COVERAGE_COHORT = ["tests/test_governed_route_matrix.py"]
"""Tests that own exact governed route-matrix coverage."""
GOVERNED_ROUTE_MATRIX_COVERAGE_DATA_FILE = ".coverage.governed-route-matrix-quality"
"""Isolated coverage database for the governed route matrix."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-governed-route-matrix-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *GOVERNED_ROUTE_MATRIX_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D governed-route-matrix quality ratchet",
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
                *GOVERNED_ROUTE_MATRIX_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    data = GOVERNED_ROUTE_MATRIX_COVERAGE_DATA_FILE
    return [
        (
            "governed-route-matrix focused coverage",
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
                *GOVERNED_ROUTE_MATRIX_COVERAGE_COHORT,
            ],
        ),
        (
            "governed-route-matrix exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/governed_route_matrix.py",
            ],
        ),
    ]


__all__ = [
    "GOVERNED_ROUTE_MATRIX_COVERAGE_COHORT",
    "GOVERNED_ROUTE_MATRIX_COVERAGE_DATA_FILE",
    "GOVERNED_ROUTE_MATRIX_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
