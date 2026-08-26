# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — notebook-programme quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
NOTEBOOK_PROGRAMME_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/notebook_programme_product.py",
    "tests/test_notebook_programme_product.py",
    "tools/notebook_programme_product_quality_gates.py",
    "tests/test_notebook_programme_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
NOTEBOOK_PROGRAMME_PRODUCT_COVERAGE_COHORT = ["tests/test_notebook_programme_product.py"]
"""Tests that own exact notebook-programme product coverage."""
NOTEBOOK_PROGRAMME_PRODUCT_COVERAGE_DATA_FILE = ".coverage.notebook-programme-quality"
"""Isolated coverage database for the notebook-programme product owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-notebook-programme-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *NOTEBOOK_PROGRAMME_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D notebook-programme-product quality ratchet",
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
                *NOTEBOOK_PROGRAMME_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "notebook-programme-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={NOTEBOOK_PROGRAMME_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *NOTEBOOK_PROGRAMME_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "notebook-programme-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={NOTEBOOK_PROGRAMME_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/notebook_programme_product.py",
            ],
        ),
    ]


__all__ = [
    "NOTEBOOK_PROGRAMME_PRODUCT_COVERAGE_COHORT",
    "NOTEBOOK_PROGRAMME_PRODUCT_COVERAGE_DATA_FILE",
    "NOTEBOOK_PROGRAMME_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
