# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — custom-derivatives product quality-gate specification
"""Build strict documentation, typing, and coverage gates for BL-19."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

CUSTOM_DERIVATIVES_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/custom_derivatives_product.py",
    "tests/test_custom_derivatives_product.py",
    "tools/custom_derivatives_product_quality_gates.py",
    "tests/test_custom_derivatives_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

CUSTOM_DERIVATIVES_PRODUCT_COVERAGE_COHORT = ["tests/test_custom_derivatives_product.py"]
"""Tests that own exact custom-derivatives product coverage."""

CUSTOM_DERIVATIVES_PRODUCT_COVERAGE_DATA_FILE = ".coverage.custom-derivatives-product-quality"
"""Isolated coverage database for the custom-derivatives product owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by preflight.

    Returns
    -------
    list[Gate]
        Ordered static gates for the product owner cohort.

    """
    return [
        (
            "mypy-strict-custom-derivatives-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *CUSTOM_DERIVATIVES_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D custom-derivatives-product quality ratchet",
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
                *CUSTOM_DERIVATIVES_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by preflight.

    Returns
    -------
    list[Gate]
        Focused owner execution followed by the exact report.

    """
    return [
        (
            "custom-derivatives-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={CUSTOM_DERIVATIVES_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *CUSTOM_DERIVATIVES_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "custom-derivatives-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={CUSTOM_DERIVATIVES_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/custom_derivatives_product.py",
            ],
        ),
    ]


__all__ = [
    "CUSTOM_DERIVATIVES_PRODUCT_COVERAGE_COHORT",
    "CUSTOM_DERIVATIVES_PRODUCT_COVERAGE_DATA_FILE",
    "CUSTOM_DERIVATIVES_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
