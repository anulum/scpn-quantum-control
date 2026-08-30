# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — migration-guides quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
MIGRATION_GUIDES_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/migration_guides_product.py",
    "src/scpn_quantum_control/phase/pennylane_import.py",
    "tests/test_migration_guides_product.py",
    "tests/test_phase_pennylane_import.py",
    "tools/migration_guides_product_quality_gates.py",
    "tests/test_migration_guides_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
MIGRATION_GUIDES_PRODUCT_COVERAGE_COHORT = [
    "tests/test_migration_guides_product.py",
    "tests/test_phase_pennylane_import.py",
]
"""Tests that own exact migration-guides product coverage."""
MIGRATION_GUIDES_PRODUCT_COVERAGE_DATA_FILE = ".coverage.migration-guides-quality"
"""Isolated coverage database for the migration-guides product owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-migration-guides-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *MIGRATION_GUIDES_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D migration-guides-product quality ratchet",
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
                *MIGRATION_GUIDES_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "migration-guides-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={MIGRATION_GUIDES_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *MIGRATION_GUIDES_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "migration-guides-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={MIGRATION_GUIDES_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/migration_guides_product.py,*/phase/pennylane_import.py",
            ],
        ),
    ]


__all__ = [
    "MIGRATION_GUIDES_PRODUCT_COVERAGE_COHORT",
    "MIGRATION_GUIDES_PRODUCT_COVERAGE_DATA_FILE",
    "MIGRATION_GUIDES_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
