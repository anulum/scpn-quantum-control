# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — geometric-control product quality-gate specification
"""Build strict typing, documentation, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

GEOMETRIC_CONTROL_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/geometric_control_product.py",
    "tests/test_geometric_control_product.py",
    "tools/geometric_control_product_quality_gates.py",
    "tests/test_geometric_control_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

GEOMETRIC_CONTROL_PRODUCT_COVERAGE_COHORT = ["tests/test_geometric_control_product.py"]
"""Tests that own exact geometric-control product coverage."""

GEOMETRIC_CONTROL_PRODUCT_COVERAGE_DATA_FILE = ".coverage.geometric-control-product-quality"
"""Isolated coverage database for the geometric-control product owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-geometric-control-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *GEOMETRIC_CONTROL_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D geometric-control-product quality ratchet",
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
                *GEOMETRIC_CONTROL_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "geometric-control-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={GEOMETRIC_CONTROL_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *GEOMETRIC_CONTROL_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "geometric-control-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={GEOMETRIC_CONTROL_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/geometric_control_product.py",
            ],
        ),
    ]


__all__ = [
    "GEOMETRIC_CONTROL_PRODUCT_COVERAGE_COHORT",
    "GEOMETRIC_CONTROL_PRODUCT_COVERAGE_DATA_FILE",
    "GEOMETRIC_CONTROL_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
