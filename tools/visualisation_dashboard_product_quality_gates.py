# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — visualisation-dashboard quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
VISUALISATION_DASHBOARD_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/visualisation_dashboard_product.py",
    "tests/test_visualisation_dashboard_product.py",
    "tools/visualisation_dashboard_product_quality_gates.py",
    "tests/test_visualisation_dashboard_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
VISUALISATION_DASHBOARD_PRODUCT_COVERAGE_COHORT = ["tests/test_visualisation_dashboard_product.py"]
"""Tests that own exact visualisation-dashboard product coverage."""
VISUALISATION_DASHBOARD_PRODUCT_COVERAGE_DATA_FILE = ".coverage.visualisation-dashboard-quality"
"""Isolated coverage database for the visualisation-dashboard product owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-visualisation-dashboard-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *VISUALISATION_DASHBOARD_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D visualisation-dashboard-product quality ratchet",
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
                *VISUALISATION_DASHBOARD_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "visualisation-dashboard-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={VISUALISATION_DASHBOARD_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *VISUALISATION_DASHBOARD_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "visualisation-dashboard-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={VISUALISATION_DASHBOARD_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/visualisation_dashboard_product.py",
            ],
        ),
    ]


__all__ = [
    "VISUALISATION_DASHBOARD_PRODUCT_COVERAGE_COHORT",
    "VISUALISATION_DASHBOARD_PRODUCT_COVERAGE_DATA_FILE",
    "VISUALISATION_DASHBOARD_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
