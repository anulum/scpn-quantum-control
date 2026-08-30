# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — logging-setup quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

LOGGING_SETUP_QUALITY_RATCHET = [
    "src/scpn_quantum_control/logging_setup.py",
    "tests/test_logging_setup.py",
    "tools/logging_setup_quality_gates.py",
    "tests/test_logging_setup_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

LOGGING_SETUP_COVERAGE_COHORT = [
    "tests/test_logging_setup.py",
]
"""Tests that own exact logging-setup coverage."""

LOGGING_SETUP_COVERAGE_DATA_FILE = "/tmp/scpn-qc-logging-setup-quality.coverage"  # nosec B108
"""Isolated coverage database for the logging-setup owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-logging-setup-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *LOGGING_SETUP_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D logging-setup quality ratchet",
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
                *LOGGING_SETUP_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "logging-setup focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={LOGGING_SETUP_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *LOGGING_SETUP_COVERAGE_COHORT,
            ],
        ),
        (
            "logging-setup exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={LOGGING_SETUP_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/logging_setup.py",
            ],
        ),
    ]


__all__ = [
    "LOGGING_SETUP_COVERAGE_COHORT",
    "LOGGING_SETUP_COVERAGE_DATA_FILE",
    "LOGGING_SETUP_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
