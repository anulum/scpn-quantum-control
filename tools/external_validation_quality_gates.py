# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable external-validation quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
EXTERNAL_VALIDATION_QUALITY_RATCHET = [
    "src/scpn_quantum_control/differentiable_external_validation.py",
    "tests/test_differentiable_external_validation.py",
    "tools/check_differentiable_external_validation.py",
    "tools/external_validation_quality_gates.py",
    "tests/test_external_validation_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
EXTERNAL_VALIDATION_COVERAGE_COHORT = ["tests/test_differentiable_external_validation.py"]
"""Tests that own exact differentiable external-validation coverage."""
EXTERNAL_VALIDATION_COVERAGE_DATA_FILE = "/tmp/scpn-qc-external-validation.coverage"  # nosec B108
"""Isolated coverage database for differentiable external validation."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-external-validation-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *EXTERNAL_VALIDATION_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D external-validation quality ratchet",
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
                *EXTERNAL_VALIDATION_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "external-validation focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={EXTERNAL_VALIDATION_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *EXTERNAL_VALIDATION_COVERAGE_COHORT,
            ],
        ),
        (
            "external-validation exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={EXTERNAL_VALIDATION_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/differentiable_external_validation.py",
            ],
        ),
    ]


__all__ = [
    "EXTERNAL_VALIDATION_COVERAGE_COHORT",
    "EXTERNAL_VALIDATION_COVERAGE_DATA_FILE",
    "EXTERNAL_VALIDATION_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
