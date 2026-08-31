# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Wirtinger-implicit quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
WIRTINGER_IMPLICIT_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/wirtinger_implicit_product.py",
    "src/scpn_quantum_control/wirtinger_calculus.py",
    "tests/test_wirtinger_implicit_product.py",
    "tools/wirtinger_implicit_product_quality_gates.py",
    "tests/test_wirtinger_implicit_product_quality_gate.py",
]
"""Ordered strict-typing cohort."""
WIRTINGER_IMPLICIT_PRODUCT_DOCSTRING_RATCHET = [
    "src/scpn_quantum_control/wirtinger_implicit_product.py",
    "src/scpn_quantum_control/wirtinger_calculus.py",
    "tests/test_wirtinger_implicit_product.py",
    "tests/test_wirtinger_calculus.py",
    "tools/wirtinger_implicit_product_quality_gates.py",
    "tests/test_wirtinger_implicit_product_quality_gate.py",
]
"""Ordered complete NumPy-docstring cohort."""
WIRTINGER_IMPLICIT_PRODUCT_COVERAGE_COHORT = [
    "tests/test_wirtinger_implicit_product.py",
    "tests/test_wirtinger_calculus.py",
]
"""Tests that own exact Wirtinger-implicit product coverage."""
WIRTINGER_IMPLICIT_PRODUCT_COVERAGE_DATA_FILE = ".coverage.wirtinger-implicit-quality"
"""Isolated coverage database for the Wirtinger-implicit product owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-wirtinger-implicit-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *WIRTINGER_IMPLICIT_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D wirtinger-implicit-product quality ratchet",
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
                *WIRTINGER_IMPLICIT_PRODUCT_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "wirtinger-implicit-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={WIRTINGER_IMPLICIT_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *WIRTINGER_IMPLICIT_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "wirtinger-implicit-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={WIRTINGER_IMPLICIT_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/wirtinger_implicit_product.py,*/wirtinger_calculus.py",
            ],
        ),
    ]


__all__ = [
    "WIRTINGER_IMPLICIT_PRODUCT_COVERAGE_COHORT",
    "WIRTINGER_IMPLICIT_PRODUCT_COVERAGE_DATA_FILE",
    "WIRTINGER_IMPLICIT_PRODUCT_DOCSTRING_RATCHET",
    "WIRTINGER_IMPLICIT_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
