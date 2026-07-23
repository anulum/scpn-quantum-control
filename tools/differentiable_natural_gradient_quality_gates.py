# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — natural-gradient quality-gate specification
"""Build exact quality gates for the differentiable natural-gradient cohort."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

NATURAL_GRADIENT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/differentiable_natural_gradient.py",
    "tests/test_differentiable_natural_gradient.py",
    "tests/test_differentiable_natural_gradient_line_search.py",
    "tools/differentiable_natural_gradient_quality_gates.py",
    "tools/differentiable_quality_gates.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

NATURAL_GRADIENT_COVERAGE_COHORT = [
    "tests/test_differentiable_natural_gradient.py",
    "tests/test_differentiable_natural_gradient_line_search.py",
]
"""Tests that own exact natural-gradient statement and branch coverage."""

NATURAL_GRADIENT_COVERAGE_DATA_FILE = ".coverage.natural-gradient-quality"
"""Isolated coverage database for the natural-gradient source owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Ordered static quality gates for the natural-gradient cohort.

    """
    return [
        (
            "mypy-strict-differentiable-natural-gradient-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *NATURAL_GRADIENT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D differentiable-natural-gradient quality ratchet",
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
                *NATURAL_GRADIENT_QUALITY_RATCHET,
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
            "differentiable-natural-gradient focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={NATURAL_GRADIENT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *NATURAL_GRADIENT_COVERAGE_COHORT,
            ],
        ),
        (
            "differentiable-natural-gradient exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={NATURAL_GRADIENT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/differentiable_natural_gradient.py",
            ],
        ),
    ]


__all__ = [
    "NATURAL_GRADIENT_COVERAGE_COHORT",
    "NATURAL_GRADIENT_COVERAGE_DATA_FILE",
    "NATURAL_GRADIENT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
