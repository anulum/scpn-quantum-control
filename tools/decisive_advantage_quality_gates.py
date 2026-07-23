# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — decisive-advantage quality-gate specification
"""Build exact quality gates for the decisive-advantage protocol owner."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

DECISIVE_ADVANTAGE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/benchmarks/decisive_advantage_protocol.py",
    "tests/test_decisive_advantage_protocol.py",
    "tools/decisive_advantage_quality_gates.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

DECISIVE_ADVANTAGE_COVERAGE_COHORT = [
    "tests/test_decisive_advantage_protocol.py",
]
"""Tests that own exact decisive-protocol statement and branch coverage."""

DECISIVE_ADVANTAGE_COVERAGE_DATA_FILE = ".coverage.decisive-advantage-quality"
"""Isolated coverage database for the decisive-protocol owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Ordered static quality gates for the source and owner test.

    """
    return [
        (
            "mypy-strict-decisive-advantage-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DECISIVE_ADVANTAGE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D decisive-advantage quality ratchet",
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
                *DECISIVE_ADVANTAGE_QUALITY_RATCHET,
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
        Focused execution followed by the exact owner-only report.

    """
    return [
        (
            "decisive-advantage focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DECISIVE_ADVANTAGE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DECISIVE_ADVANTAGE_COVERAGE_COHORT,
            ],
        ),
        (
            "decisive-advantage exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DECISIVE_ADVANTAGE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/decisive_advantage_protocol.py",
            ],
        ),
    ]


__all__ = [
    "DECISIVE_ADVANTAGE_COVERAGE_COHORT",
    "DECISIVE_ADVANTAGE_COVERAGE_DATA_FILE",
    "DECISIVE_ADVANTAGE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
