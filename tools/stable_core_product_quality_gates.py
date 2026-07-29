# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — stable-core product quality-gate specification
"""Build strict documentation, typing, and coverage gates for BL-19."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

STABLE_CORE_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/stable_core_product.py",
    "tests/test_stable_core_product.py",
    "tools/stable_core_product_quality_gates.py",
    "tests/test_stable_core_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

STABLE_CORE_PRODUCT_COVERAGE_COHORT = [
    "tests/test_stable_core_product.py",
]
"""Tests that own exact stable-core product statement and branch coverage."""

STABLE_CORE_PRODUCT_COVERAGE_DATA_FILE = ".coverage.stable-core-product-quality"
"""Isolated coverage database for the stable-core product source owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Ordered static gates for the source, owner test, and gate contract.

    """
    return [
        (
            "mypy-strict-stable-core-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *STABLE_CORE_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D stable-core-product quality ratchet",
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
                *STABLE_CORE_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build isolated exact statement and branch coverage gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Focused execution followed by the exact source-only report.

    """
    return [
        (
            "stable-core-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={STABLE_CORE_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *STABLE_CORE_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "stable-core-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={STABLE_CORE_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/stable_core_product.py",
            ],
        ),
    ]


__all__ = [
    "STABLE_CORE_PRODUCT_COVERAGE_COHORT",
    "STABLE_CORE_PRODUCT_COVERAGE_DATA_FILE",
    "STABLE_CORE_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
