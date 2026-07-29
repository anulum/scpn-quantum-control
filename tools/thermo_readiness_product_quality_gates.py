# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — thermo-readiness product quality-gate specification
"""Build documentation, typing, and exact coverage gates for BL-19."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

THERMO_READINESS_PRODUCT_TYPING_RATCHET = [
    "src/scpn_quantum_control/thermo_readiness_product.py",
    "tests/test_thermo_readiness_product.py",
    "tools/thermo_readiness_product_quality_gates.py",
    "tests/test_thermo_readiness_product_quality_gate.py",
]
"""Ordered strict-typing cohort."""

THERMO_READINESS_PRODUCT_DOCSTRING_RATCHET = [
    "src/scpn_quantum_control/thermo_readiness_product.py",
    "tools/thermo_readiness_product_quality_gates.py",
    "tests/test_thermo_readiness_product_quality_gate.py",
]
"""Public source and gate-contract NumPy-docstring cohort."""

THERMO_READINESS_PRODUCT_COVERAGE_COHORT = ["tests/test_thermo_readiness_product.py"]
"""Tests that own exact thermo-readiness statement and branch coverage."""

THERMO_READINESS_PRODUCT_COVERAGE_DATA_FILE = ".coverage.thermo-readiness-product-quality"
"""Isolated coverage database for the thermo-readiness source owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and public-source NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by preflight.

    Returns
    -------
    list[Gate]
        Ordered static gates for the owned product cohort.

    """
    return [
        (
            "mypy-strict-thermo-readiness-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *THERMO_READINESS_PRODUCT_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D thermo-readiness-product quality ratchet",
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
                *THERMO_READINESS_PRODUCT_DOCSTRING_RATCHET,
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
            "thermo-readiness-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={THERMO_READINESS_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *THERMO_READINESS_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "thermo-readiness-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={THERMO_READINESS_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/thermo_readiness_product.py",
            ],
        ),
    ]


__all__ = [
    "THERMO_READINESS_PRODUCT_COVERAGE_COHORT",
    "THERMO_READINESS_PRODUCT_COVERAGE_DATA_FILE",
    "THERMO_READINESS_PRODUCT_DOCSTRING_RATCHET",
    "THERMO_READINESS_PRODUCT_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
