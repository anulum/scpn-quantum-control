# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — neural-operator baseline quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

NEURAL_OPERATOR_BASELINE_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/neural_operator_baseline_product.py",
    "src/scpn_quantum_control/forecasting/neural_operator_advantage.py",
    "tests/test_neural_operator_baseline_product.py",
    "tests/test_neural_operator_advantage.py",
    "tools/neural_operator_baseline_product_quality_gates.py",
    "tests/test_neural_operator_baseline_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

NEURAL_OPERATOR_BASELINE_PRODUCT_COVERAGE_COHORT = [
    "tests/test_neural_operator_baseline_product.py",
    "tests/test_neural_operator_advantage.py",
]
"""Tests that own exact neural-operator baseline product coverage."""

NEURAL_OPERATOR_BASELINE_PRODUCT_COVERAGE_DATA_FILE = (
    ".coverage.neural-operator-baseline-product-quality"
)
"""Isolated coverage database for the neural-operator baseline product."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-neural-operator-baseline-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *NEURAL_OPERATOR_BASELINE_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D neural-operator-baseline-product quality ratchet",
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
                *NEURAL_OPERATOR_BASELINE_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "neural-operator-baseline-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={NEURAL_OPERATOR_BASELINE_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *NEURAL_OPERATOR_BASELINE_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "neural-operator-baseline-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={NEURAL_OPERATOR_BASELINE_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/neural_operator_baseline_product.py,*/forecasting/neural_operator_advantage.py",
            ],
        ),
    ]


__all__ = [
    "NEURAL_OPERATOR_BASELINE_PRODUCT_COVERAGE_COHORT",
    "NEURAL_OPERATOR_BASELINE_PRODUCT_COVERAGE_DATA_FILE",
    "NEURAL_OPERATOR_BASELINE_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
