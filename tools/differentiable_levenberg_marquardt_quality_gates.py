# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Levenberg-Marquardt quality-gate specification
"""Build exact quality gates for the differentiable LM owner cohort."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

LEVENBERG_MARQUARDT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/differentiable_levenberg_marquardt.py",
    "tests/test_differentiable_levenberg_marquardt.py",
    "tests/test_differentiable_custom_derivatives.py",
    "tools/differentiable_levenberg_marquardt_quality_gates.py",
    "tools/differentiable_quality_gates.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

LEVENBERG_MARQUARDT_COVERAGE_COHORT = [
    "tests/test_differentiable_levenberg_marquardt.py",
    "tests/test_differentiable_custom_derivatives.py",
]
"""Tests that own exact LM statement and branch coverage."""

LEVENBERG_MARQUARDT_COVERAGE_DATA_FILE = ".coverage.levenberg-marquardt-quality"
"""Isolated coverage database for the LM source owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Ordered static quality gates for the LM owner cohort.

    """
    return [
        (
            "mypy-strict-differentiable-levenberg-marquardt-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *LEVENBERG_MARQUARDT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D differentiable-levenberg-marquardt quality ratchet",
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
                *LEVENBERG_MARQUARDT_QUALITY_RATCHET,
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
            "differentiable-levenberg-marquardt focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={LEVENBERG_MARQUARDT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *LEVENBERG_MARQUARDT_COVERAGE_COHORT,
            ],
        ),
        (
            "differentiable-levenberg-marquardt exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={LEVENBERG_MARQUARDT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/differentiable_levenberg_marquardt.py",
            ],
        ),
    ]


__all__ = [
    "LEVENBERG_MARQUARDT_COVERAGE_COHORT",
    "LEVENBERG_MARQUARDT_COVERAGE_DATA_FILE",
    "LEVENBERG_MARQUARDT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
