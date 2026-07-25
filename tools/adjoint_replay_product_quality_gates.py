# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — adjoint replay product quality-gate specification
"""Build exact quality gates for the adjoint replay product."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

ADJOINT_REPLAY_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/adjoint_replay_product.py",
    "tests/test_adjoint_replay_product.py",
    "tools/adjoint_replay_product_quality_gates.py",
    "tools/differentiable_quality_gates.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

ADJOINT_REPLAY_PRODUCT_COVERAGE_COHORT = [
    "tests/test_adjoint_replay_product.py",
]
"""Tests that own exact adjoint replay product coverage."""

ADJOINT_REPLAY_PRODUCT_COVERAGE_DATA_FILE = ".coverage.adjoint-replay-product-quality"
"""Isolated coverage database for the adjoint replay product owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Ordered static gates for the adjoint replay product cohort.

    """
    return [
        (
            "mypy-strict-adjoint-replay-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *ADJOINT_REPLAY_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D adjoint-replay-product quality ratchet",
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
                *ADJOINT_REPLAY_PRODUCT_QUALITY_RATCHET,
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
            "adjoint-replay-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={ADJOINT_REPLAY_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *ADJOINT_REPLAY_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "adjoint-replay-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={ADJOINT_REPLAY_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/adjoint_replay_product.py",
            ],
        ),
    ]


__all__ = [
    "ADJOINT_REPLAY_PRODUCT_COVERAGE_COHORT",
    "ADJOINT_REPLAY_PRODUCT_COVERAGE_DATA_FILE",
    "ADJOINT_REPLAY_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
