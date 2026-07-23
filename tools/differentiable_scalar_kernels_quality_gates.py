# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable scalar-kernel quality-gate specification
"""Build exact quality gates for the differentiable scalar-kernel owner."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

DIFFERENTIABLE_SCALAR_KERNELS_QUALITY_RATCHET = [
    "src/scpn_quantum_control/differentiable_scalar_kernels.py",
    "tests/test_differentiable_scalar_kernels.py",
    "tests/test_differentiable_transform_helpers.py",
    "tools/differentiable_scalar_kernels_quality_gates.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

DIFFERENTIABLE_SCALAR_KERNELS_COVERAGE_COHORT = [
    "tests/test_differentiable_scalar_kernels.py",
    "tests/test_differentiable_transform_helpers.py",
]
"""Tests that own exact scalar-kernel statement and branch coverage."""

DIFFERENTIABLE_SCALAR_KERNELS_COVERAGE_DATA_FILE = ".coverage.scalar-kernels-quality"
"""Isolated coverage database for the scalar-kernel owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Ordered static quality gates for the source, owner tests, and gate spec.

    """
    return [
        (
            "mypy-strict-differentiable-scalar-kernels-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DIFFERENTIABLE_SCALAR_KERNELS_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D differentiable-scalar-kernels quality ratchet",
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
                *DIFFERENTIABLE_SCALAR_KERNELS_QUALITY_RATCHET,
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
            "differentiable-scalar-kernels focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_SCALAR_KERNELS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DIFFERENTIABLE_SCALAR_KERNELS_COVERAGE_COHORT,
            ],
        ),
        (
            "differentiable-scalar-kernels exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_SCALAR_KERNELS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/differentiable_scalar_kernels.py",
            ],
        ),
    ]


__all__ = [
    "DIFFERENTIABLE_SCALAR_KERNELS_COVERAGE_COHORT",
    "DIFFERENTIABLE_SCALAR_KERNELS_COVERAGE_DATA_FILE",
    "DIFFERENTIABLE_SCALAR_KERNELS_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
