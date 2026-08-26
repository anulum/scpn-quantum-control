# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware-safe execution quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
HARDWARE_SAFE_EXECUTION_QUALITY_RATCHET = [
    "src/scpn_quantum_control/hardware_safe_execution.py",
    "tests/test_hardware_safe_execution.py",
    "tools/hardware_safe_execution_quality_gates.py",
    "tests/test_hardware_safe_execution_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
HARDWARE_SAFE_EXECUTION_COVERAGE_COHORT = ["tests/test_hardware_safe_execution.py"]
"""Tests that own exact hardware-safe execution coverage."""
HARDWARE_SAFE_EXECUTION_COVERAGE_DATA_FILE = ".coverage.hardware-safe-execution-quality"
"""Isolated coverage database for the hardware-safe execution owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-hardware-safe-execution-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *HARDWARE_SAFE_EXECUTION_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D hardware-safe-execution quality ratchet",
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
                *HARDWARE_SAFE_EXECUTION_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "hardware-safe-execution focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={HARDWARE_SAFE_EXECUTION_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *HARDWARE_SAFE_EXECUTION_COVERAGE_COHORT,
            ],
        ),
        (
            "hardware-safe-execution exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={HARDWARE_SAFE_EXECUTION_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/hardware_safe_execution.py",
            ],
        ),
    ]


__all__ = [
    "HARDWARE_SAFE_EXECUTION_COVERAGE_COHORT",
    "HARDWARE_SAFE_EXECUTION_COVERAGE_DATA_FILE",
    "HARDWARE_SAFE_EXECUTION_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
