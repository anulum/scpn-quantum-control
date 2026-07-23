# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Program AD array-indexing quality-gate specification
"""Build exact quality gates for the Program AD array-indexing owner."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

PROGRAM_AD_ARRAY_INDEXING_QUALITY_RATCHET = [
    "src/scpn_quantum_control/program_ad_array_indexing.py",
    "tests/test_program_ad_array_indexing_quality.py",
    "tools/program_ad_array_indexing_quality_gates.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

PROGRAM_AD_ARRAY_INDEXING_COVERAGE_COHORT = [
    "tests/test_program_ad_array_indexing_registry.py",
    "tests/test_program_ad_structural_finite_difference_gradient_check.py",
    "tests/test_program_ad_fail_closed_boundaries.py",
    "tests/test_program_ad_array_indexing_quality.py",
]
"""Tests that own exact array-indexing statement and branch coverage."""

PROGRAM_AD_ARRAY_INDEXING_COVERAGE_DATA_FILE = ".coverage.program-ad-array-indexing"
"""Isolated coverage database for the array-indexing owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Ordered static quality gates for the source, owner test, and gate spec.

    """
    return [
        (
            "mypy-strict-program-ad-array-indexing-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PROGRAM_AD_ARRAY_INDEXING_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D program-ad-array-indexing quality ratchet",
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
                *PROGRAM_AD_ARRAY_INDEXING_QUALITY_RATCHET,
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
            "program-ad-array-indexing focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PROGRAM_AD_ARRAY_INDEXING_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PROGRAM_AD_ARRAY_INDEXING_COVERAGE_COHORT,
            ],
        ),
        (
            "program-ad-array-indexing exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PROGRAM_AD_ARRAY_INDEXING_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/program_ad_array_indexing.py",
            ],
        ),
    ]


__all__ = [
    "PROGRAM_AD_ARRAY_INDEXING_COVERAGE_COHORT",
    "PROGRAM_AD_ARRAY_INDEXING_COVERAGE_DATA_FILE",
    "PROGRAM_AD_ARRAY_INDEXING_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
