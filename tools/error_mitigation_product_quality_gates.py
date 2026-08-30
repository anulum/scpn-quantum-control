# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — error-mitigation product quality-gate specification
"""Build strict typing, documentation, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

ERROR_MITIGATION_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/error_mitigation_product.py",
    "src/scpn_quantum_control/mitigation/compound_mitigation.py",
    "src/scpn_quantum_control/mitigation/cpdr.py",
    "tests/test_error_mitigation_product.py",
    "tests/test_compound_mitigation.py",
    "tests/test_cpdr.py",
    "tools/error_mitigation_product_quality_gates.py",
    "tests/test_error_mitigation_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

ERROR_MITIGATION_PRODUCT_COVERAGE_COHORT = [
    "tests/test_error_mitigation_product.py",
    "tests/test_compound_mitigation.py",
    "tests/test_cpdr.py",
]
"""Tests that own exact error-mitigation product coverage."""

ERROR_MITIGATION_PRODUCT_COVERAGE_DATA_FILE = ".coverage.error-mitigation-product-quality"
"""Isolated coverage database for the error-mitigation product owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-error-mitigation-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *ERROR_MITIGATION_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D error-mitigation-product quality ratchet",
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
                *ERROR_MITIGATION_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "error-mitigation-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={ERROR_MITIGATION_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *ERROR_MITIGATION_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "error-mitigation-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={ERROR_MITIGATION_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/error_mitigation_product.py,*/mitigation/compound_mitigation.py,*/mitigation/cpdr.py",
            ],
        ),
    ]


__all__ = [
    "ERROR_MITIGATION_PRODUCT_COVERAGE_COHORT",
    "ERROR_MITIGATION_PRODUCT_COVERAGE_DATA_FILE",
    "ERROR_MITIGATION_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
