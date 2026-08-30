# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — stochastic-estimators quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
STOCHASTIC_ESTIMATORS_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/stochastic_estimators_product.py",
    "src/scpn_quantum_control/differentiable_stochastic_estimators.py",
    "tests/test_stochastic_estimators_product.py",
    "tests/test_differentiable_stochastic_estimators.py",
    "tools/stochastic_estimators_product_quality_gates.py",
    "tests/test_stochastic_estimators_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
STOCHASTIC_ESTIMATORS_PRODUCT_COVERAGE_COHORT = [
    "tests/test_stochastic_estimators_product.py",
    "tests/test_differentiable_stochastic_estimators.py",
]
"""Tests that own exact stochastic-estimators product coverage."""
STOCHASTIC_ESTIMATORS_PRODUCT_COVERAGE_DATA_FILE = ".coverage.stochastic-estimators-quality"
"""Isolated coverage database for the stochastic-estimators product owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-stochastic-estimators-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *STOCHASTIC_ESTIMATORS_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D stochastic-estimators-product quality ratchet",
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
                *STOCHASTIC_ESTIMATORS_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "stochastic-estimators-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={STOCHASTIC_ESTIMATORS_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *STOCHASTIC_ESTIMATORS_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "stochastic-estimators-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={STOCHASTIC_ESTIMATORS_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/stochastic_estimators_product.py,*/differentiable_stochastic_estimators.py",
            ],
        ),
    ]


__all__ = [
    "STOCHASTIC_ESTIMATORS_PRODUCT_COVERAGE_COHORT",
    "STOCHASTIC_ESTIMATORS_PRODUCT_COVERAGE_DATA_FILE",
    "STOCHASTIC_ESTIMATORS_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
