# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Program AD fuzz assurance quality-gate specification
"""Build strict documentation, typing, and coverage gates for BL-19."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

PROGRAM_AD_FUZZ_ASSURANCE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/program_ad_fuzz_assurance.py",
    "tests/test_program_ad_fuzz_assurance.py",
    "tools/program_ad_fuzz_assurance_quality_gates.py",
    "tests/test_program_ad_fuzz_assurance_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

PROGRAM_AD_FUZZ_ASSURANCE_COVERAGE_COHORT = ["tests/test_program_ad_fuzz_assurance.py"]
"""Tests that own exact Program AD fuzz assurance coverage."""

PROGRAM_AD_FUZZ_ASSURANCE_COVERAGE_DATA_FILE = ".coverage.program-ad-fuzz-assurance-quality"
"""Isolated coverage database for the Program AD fuzz assurance owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by preflight.

    Returns
    -------
    list[Gate]
        Ordered static gates for the fuzz-assurance owner cohort.

    """
    return [
        (
            "mypy-strict-program-ad-fuzz-assurance-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PROGRAM_AD_FUZZ_ASSURANCE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D program-ad-fuzz-assurance quality ratchet",
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
                *PROGRAM_AD_FUZZ_ASSURANCE_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by preflight.

    Returns
    -------
    list[Gate]
        Focused owner execution followed by the exact report.

    """
    return [
        (
            "program-ad-fuzz-assurance focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PROGRAM_AD_FUZZ_ASSURANCE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PROGRAM_AD_FUZZ_ASSURANCE_COVERAGE_COHORT,
            ],
        ),
        (
            "program-ad-fuzz-assurance exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PROGRAM_AD_FUZZ_ASSURANCE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/program_ad_fuzz_assurance.py",
            ],
        ),
    ]


__all__ = [
    "PROGRAM_AD_FUZZ_ASSURANCE_COVERAGE_COHORT",
    "PROGRAM_AD_FUZZ_ASSURANCE_COVERAGE_DATA_FILE",
    "PROGRAM_AD_FUZZ_ASSURANCE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
