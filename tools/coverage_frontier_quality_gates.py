# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Coverage-frontier quality-gate specification
"""Build strict documentation, typing, and exact coverage-frontier gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
COVERAGE_FRONTIER_QUALITY_RATCHET = [
    "src/scpn_quantum_control/studio/coverage_frontier.py",
    "tests/test_coverage_frontier.py",
    "tools/coverage_frontier_quality_gates.py",
    "tests/test_studio_claim_frontier_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
COVERAGE_FRONTIER_TEST_COHORT = ["tests/test_coverage_frontier.py"]
"""Tests that own exact coverage-frontier source coverage."""
COVERAGE_FRONTIER_COVERAGE_DATA_FILE = ".coverage.coverage-frontier-quality"
"""Isolated coverage database for the coverage-frontier owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-coverage-frontier-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *COVERAGE_FRONTIER_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D coverage-frontier quality ratchet",
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
                *COVERAGE_FRONTIER_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "coverage-frontier focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={COVERAGE_FRONTIER_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *COVERAGE_FRONTIER_TEST_COHORT,
            ],
        ),
        (
            "coverage-frontier exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={COVERAGE_FRONTIER_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/studio/coverage_frontier.py",
            ],
        ),
    ]


__all__ = [
    "COVERAGE_FRONTIER_COVERAGE_DATA_FILE",
    "COVERAGE_FRONTIER_QUALITY_RATCHET",
    "COVERAGE_FRONTIER_TEST_COHORT",
    "build_coverage_gates",
    "build_static_quality_gates",
]
