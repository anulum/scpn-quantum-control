# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Advanced-witnesses quality-gate specification
"""Build strict documentation, typing, and exact advanced-witnesses gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
ADVANCED_WITNESSES_QUALITY_RATCHET = [
    "src/scpn_quantum_control/advanced_witnesses_product.py",
    "tests/test_advanced_witnesses_product.py",
    "tools/advanced_witnesses_quality_gates.py",
    "tests/test_advanced_witnesses_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
ADVANCED_WITNESSES_TEST_COHORT = ["tests/test_advanced_witnesses_product.py"]
"""Tests that own exact advanced-witnesses source coverage."""
ADVANCED_WITNESSES_COVERAGE_DATA_FILE = ".coverage.advanced-witnesses-quality"
"""Isolated coverage database for the advanced-witnesses owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-advanced-witnesses-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *ADVANCED_WITNESSES_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D advanced-witnesses quality ratchet",
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
                *ADVANCED_WITNESSES_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "advanced-witnesses focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={ADVANCED_WITNESSES_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *ADVANCED_WITNESSES_TEST_COHORT,
            ],
        ),
        (
            "advanced-witnesses exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={ADVANCED_WITNESSES_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/advanced_witnesses_product.py",
            ],
        ),
    ]


__all__ = [
    "ADVANCED_WITNESSES_COVERAGE_DATA_FILE",
    "ADVANCED_WITNESSES_QUALITY_RATCHET",
    "ADVANCED_WITNESSES_TEST_COHORT",
    "build_coverage_gates",
    "build_static_quality_gates",
]
