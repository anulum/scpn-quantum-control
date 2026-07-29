# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — KYMA mechanism product quality-gate specification
"""Build strict documentation, typing, and coverage gates for BL-19."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

KYMA_MECHANISM_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/kyma_mechanism_benchmark_product.py",
    "tests/test_kyma_mechanism_benchmark_product.py",
    "tools/kyma_mechanism_benchmark_product_quality_gates.py",
    "tests/test_kyma_mechanism_benchmark_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

KYMA_MECHANISM_PRODUCT_COVERAGE_COHORT = ["tests/test_kyma_mechanism_benchmark_product.py"]
"""Tests that own exact KYMA mechanism product coverage."""

KYMA_MECHANISM_PRODUCT_COVERAGE_DATA_FILE = ".coverage.kyma-mechanism-product-quality"
"""Isolated coverage database for the KYMA mechanism product owner."""


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
            "mypy-strict-kyma-mechanism-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *KYMA_MECHANISM_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D kyma-mechanism-product quality ratchet",
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
                *KYMA_MECHANISM_PRODUCT_QUALITY_RATCHET,
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
            "kyma-mechanism-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={KYMA_MECHANISM_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *KYMA_MECHANISM_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "kyma-mechanism-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={KYMA_MECHANISM_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/kyma_mechanism_benchmark_product.py",
            ],
        ),
    ]


__all__ = [
    "KYMA_MECHANISM_PRODUCT_COVERAGE_COHORT",
    "KYMA_MECHANISM_PRODUCT_COVERAGE_DATA_FILE",
    "KYMA_MECHANISM_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
